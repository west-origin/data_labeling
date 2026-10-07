"""데이터셋 스냅샷 저장소. 한 번 커밋한 스냅샷은 바뀌지 않으며 URI로 다시 읽을 수 있다.

- LakeFSSnapshotStore: lakeFS 저장소에 파일을 올리고 커밋한다. URI는 lakefs://저장소/커밋ID/경로.
- LocalSnapshotStore: 디렉터리에 내용 해시로 둔다 (테스트·오프라인 개발용).
"""

# 외부 JSON 응답을 다루므로 이 모듈에서만 알 수 없는 타입 경고를 끈다.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import Protocol

import httpx


class SnapshotStore(Protocol):
    def commit(
        self, prefix: str, files: dict[str, Path], message: str, metadata: dict[str, str]
    ) -> str:
        """files: 스냅샷 안 상대 경로 → 로컬 파일. 스냅샷 URI를 돌려준다."""
        ...

    def read(self, snapshot_uri: str, path: str, dest: Path) -> None: ...


class SnapshotError(RuntimeError):
    pass


class LakeFSSnapshotStore:
    def __init__(
        self, endpoint: str, access_key: str, secret_key: str, *, repository: str, branch: str,
        storage_namespace: str,
    ) -> None:  # fmt: skip
        self.http = httpx.Client(
            base_url=f"{endpoint}/api/v1", auth=(access_key, secret_key), timeout=120
        )
        self.repo, self.branch = repository, branch
        resp = self.http.post(
            "/repositories",
            json={
                "name": repository,
                "storage_namespace": storage_namespace,
                "default_branch": branch,
            },
        )
        if resp.status_code not in (201, 409):
            raise SnapshotError(f"lakeFS 저장소를 만들 수 없습니다: {resp.status_code} {resp.text}")

    @classmethod
    def from_env(
        cls, *, repository: str, branch: str, storage_namespace: str
    ) -> LakeFSSnapshotStore:
        return cls(
            f"http://localhost:{os.environ.get('DLP_LAKEFS_PORT', '8000')}",
            os.environ.get("DLP_LAKEFS_ACCESS_KEY", "AKIAJDLPDEVACCESSKEY"),
            os.environ.get("DLP_LAKEFS_SECRET_KEY", "dlpdevlakefssecretkey0000000000000000000"),
            repository=repository, branch=branch, storage_namespace=storage_namespace,
        )  # fmt: skip

    def commit(
        self, prefix: str, files: dict[str, Path], message: str, metadata: dict[str, str]
    ) -> str:
        base = f"/repositories/{self.repo}/branches/{self.branch}"
        for rel, local in files.items():
            with local.open("rb") as f:
                resp = self.http.post(
                    f"{base}/objects", params={"path": f"{prefix}/{rel}"}, files={"content": f}
                )
            if resp.status_code != 201:
                raise SnapshotError(f"lakeFS 업로드 실패 {rel}: {resp.status_code} {resp.text}")
        resp = self.http.post(f"{base}/commits", json={"message": message, "metadata": metadata})
        if resp.status_code != 201:
            raise SnapshotError(f"lakeFS 커밋 실패: {resp.status_code} {resp.text}")
        return f"lakefs://{self.repo}/{resp.json()['id']}/{prefix}"

    def read(self, snapshot_uri: str, path: str, dest: Path) -> None:
        repo, commit, prefix = snapshot_uri.removeprefix("lakefs://").split("/", 2)
        resp = self.http.get(
            f"/repositories/{repo}/refs/{commit}/objects", params={"path": f"{prefix}/{path}"}
        )
        if resp.status_code != 200:
            raise SnapshotError(f"lakeFS 읽기 실패 {path}: {resp.status_code}")
        dest.write_bytes(resp.content)


class LocalSnapshotStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def commit(
        self, prefix: str, files: dict[str, Path], message: str, metadata: dict[str, str]
    ) -> str:
        h = hashlib.sha256(prefix.encode())
        for rel in sorted(files):
            h.update(rel.encode())
            h.update(files[rel].read_bytes())
        digest = h.hexdigest()[:16]
        for rel, local in files.items():
            dest = self.root / digest / prefix / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(local, dest)
        return f"local-snapshot://{digest}/{prefix}"

    def read(self, snapshot_uri: str, path: str, dest: Path) -> None:
        digest, prefix = snapshot_uri.removeprefix("local-snapshot://").split("/", 1)
        shutil.copyfile(self.root / digest / prefix / path, dest)
