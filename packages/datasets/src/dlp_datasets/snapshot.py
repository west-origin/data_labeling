"""데이터셋 스냅샷 저장소. 한 번 커밋한 스냅샷은 바뀌지 않으며 URI로 다시 읽을 수 있다.

- LakeFSSnapshotStore: lakeFS 저장소에 파일을 올리고 커밋한다. URI는 lakefs://저장소/커밋ID/경로.
- LocalSnapshotStore: 디렉터리에 내용 해시로 둔다 (테스트·오프라인 개발용).

왜 lakeFS인가 (ADR 0007): 데이터셋 버전은 학습·평가·내보내기의 재현성 기준이다. 브랜치 이름이 아니라
커밋 ID를 URI에 넣으므로, 그 뒤 같은 브랜치에 다른 버전을 커밋해도 이 URI의 내용은 바뀌지 않는다.
스냅샷 안에는 labels.jsonl(라벨 레코드), sessions.jsonl(세션 메타데이터), manifest.json이 있다
(`dlp_datasets.build`). 영상은 넣지 않는다 (블러본은 라벨링 버킷, 원본은 원본 버킷에 그대로 있다).
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
    """스냅샷 저장소 공통 인터페이스 (빌드는 commit, 내보내기·학습은 read)."""

    def commit(
        self, prefix: str, files: dict[str, Path], message: str, metadata: dict[str, str]
    ) -> str:
        """files: 스냅샷 안 상대 경로 → 로컬 파일. 스냅샷 URI를 돌려준다."""
        ...

    def read(self, snapshot_uri: str, path: str, dest: Path) -> None:
        """스냅샷 URI 안의 `path` 파일을 로컬 `dest`에 쓴다."""
        ...


class SnapshotError(RuntimeError):
    """lakeFS 요청 실패 (저장소 생성·업로드·커밋·읽기)."""


class LakeFSSnapshotStore:
    """lakeFS REST API(/api/v1)로 스냅샷을 커밋·읽는다."""

    def __init__(
        self, endpoint: str, access_key: str, secret_key: str, *, repository: str, branch: str,
        storage_namespace: str,
    ) -> None:  # fmt: skip
        """연결하고 저장소가 없으면 만든다 (201 생성, 409 이미 있음은 정상).

        Args:
            endpoint: lakeFS 주소 (예: http://localhost:8000).
            access_key, secret_key: lakeFS 자격 증명 (HTTP 기본 인증).
            repository, branch, storage_namespace: dataset.yaml `lakefs` 절.

        Raises:
            SnapshotError: 저장소를 만들 수 없을 때.
        """
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
        """환경 변수 DLP_LAKEFS_PORT·DLP_LAKEFS_ACCESS_KEY·DLP_LAKEFS_SECRET_KEY로 연결한다.

        값이 없으면 개발 compose(services/)의 기본값을 쓴다 (localhost, 개발용 키).
        """
        return cls(
            f"http://localhost:{os.environ.get('DLP_LAKEFS_PORT', '8000')}",
            os.environ.get("DLP_LAKEFS_ACCESS_KEY", "AKIAJDLPDEVACCESSKEY"),
            os.environ.get("DLP_LAKEFS_SECRET_KEY", "dlpdevlakefssecretkey0000000000000000000"),
            repository=repository, branch=branch, storage_namespace=storage_namespace,
        )  # fmt: skip

    def commit(
        self, prefix: str, files: dict[str, Path], message: str, metadata: dict[str, str]
    ) -> str:
        """브랜치에 `prefix/<상대 경로>`로 파일을 올리고 커밋한다.

        Args:
            prefix: 스냅샷 안 접두어 (예: "datasets/<버전 ID>").
            files: 상대 경로 → 로컬 파일.
            message, metadata: lakeFS 커밋 메시지·메타데이터.

        Returns:
            "lakefs://<저장소>/<커밋 ID>/<prefix>" (커밋 ID로 고정되어 불변).

        Raises:
            SnapshotError: 업로드(201 아님)나 커밋 실패. 일부만 올라간 경우 브랜치에 커밋되지 않은
                변경이 남는다 (다음 커밋에 섞일 수 있음).
        """
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
        """커밋 ID 참조(refs/<커밋>)로 파일을 읽어 `dest`에 쓴다.

        Raises:
            SnapshotError: 200이 아닐 때 (없는 파일 포함).
        """
        repo, commit, prefix = snapshot_uri.removeprefix("lakefs://").split("/", 2)
        resp = self.http.get(
            f"/repositories/{repo}/refs/{commit}/objects", params={"path": f"{prefix}/{path}"}
        )
        if resp.status_code != 200:
            raise SnapshotError(f"lakeFS 읽기 실패 {path}: {resp.status_code}")
        dest.write_bytes(resp.content)


class LocalSnapshotStore:
    """로컬 디렉터리 스냅샷 (테스트·오프라인 개발용).

    `<root>/<내용 해시 16자>/<prefix>/<상대 경로>`에 복사한다. 내용 해시는 prefix·상대 경로·파일
    내용으로 계산하므로 같은 내용이면 같은 URI(멱등), 내용이 바뀌면 다른 URI다.
    """

    def __init__(self, root: Path) -> None:
        """root: 스냅샷을 둘 디렉터리 (없으면 커밋 때 만든다)."""
        self.root = root

    def commit(
        self, prefix: str, files: dict[str, Path], message: str, metadata: dict[str, str]
    ) -> str:
        """파일을 내용 해시 디렉터리에 복사한다. message·metadata는 쓰지 않는다.

        Returns:
            "local-snapshot://<내용 해시>/<prefix>".
        """
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
        """스냅샷 파일을 `dest`로 복사한다.

        Raises:
            FileNotFoundError: 그 파일이 스냅샷에 없을 때 (내보내기가 sessions.jsonl 없음을
                이것으로 안다).
        """
        digest, prefix = snapshot_uri.removeprefix("local-snapshot://").split("/", 1)
        shutil.copyfile(self.root / digest / prefix / path, dest)
