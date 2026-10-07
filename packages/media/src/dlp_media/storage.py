"""객체 저장소. 원본(raw)과 라벨링(labeling) 버킷을 별도 인스턴스로 다룬다.

원본은 불변이다. 같은 키에 같은 내용(sha256)을 다시 올리면 건너뛰고, 다른 내용이면 오류를 낸다.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import boto3
from botocore.exceptions import ClientError

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client


class ImmutableObjectError(RuntimeError):
    pass


@dataclass(frozen=True)
class StoredObject:
    key: str
    size: int
    sha256: str | None


class ObjectStore(Protocol):
    bucket: str

    def uri(self, key: str) -> str: ...
    def head(self, key: str) -> StoredObject | None: ...
    def put_file(self, key: str, path: Path, sha256: str) -> None: ...
    def get_file(self, key: str, dest: Path) -> None: ...


def blurred_key(session_id: str, stream_id: str) -> str:
    """라벨링 버킷의 블러본 위치 (프라이버시 렌더가 쓰고, 검수·행동·큐레이션·내보내기가 읽는다)."""
    return f"sessions/{session_id}/blurred/{stream_id}.mp4"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def put_immutable(store: ObjectStore, key: str, path: Path) -> bool:
    """올렸으면 True, 같은 내용이 이미 있어 건너뛰었으면 False."""
    digest = sha256_file(path)
    existing = store.head(key)
    if existing is not None:
        if existing.sha256 == digest:
            return False
        raise ImmutableObjectError(f"{store.uri(key)}에 다른 내용이 이미 있습니다")
    store.put_file(key, path, digest)
    return True


class LocalStore:
    """파일 시스템 저장소 (테스트·오프라인 개발용). sha256은 옆 파일에 둔다."""

    def __init__(self, root: Path, bucket: str) -> None:
        self.root = root / bucket
        self.bucket = bucket

    def uri(self, key: str) -> str:
        return f"local://{self.bucket}/{key}"

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError(f"버킷 밖을 가리키는 키: {key}")
        return path

    def head(self, key: str) -> StoredObject | None:
        path = self._path(key)
        if not path.is_file():
            return None
        sidecar = path.with_name(path.name + ".sha256")
        sha = sidecar.read_text().strip() if sidecar.is_file() else None
        return StoredObject(key, path.stat().st_size, sha)

    def put_file(self, key: str, path: Path, sha256: str) -> None:
        dest = self._path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        dest.with_name(dest.name + ".sha256").write_text(sha256)

    def get_file(self, key: str, dest: Path) -> None:
        shutil.copyfile(self._path(key), dest)


class S3Store:
    """S3 호환 저장소 (SeaweedFS). sha256은 객체 메타데이터에 둔다."""

    def __init__(self, bucket: str, *, endpoint_url: str, access_key: str, secret_key: str) -> None:
        self.bucket = bucket
        self.client: S3Client = boto3.client(  # pyright: ignore[reportUnknownMemberType]
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="us-east-1",
        )

    @classmethod
    def from_env(cls, bucket: str) -> S3Store:
        return cls(
            bucket,
            endpoint_url=os.environ.get("DLP_S3_ENDPOINT", "http://localhost:8333"),
            access_key=os.environ.get("DLP_S3_ACCESS_KEY", "dlp-dev-access"),
            secret_key=os.environ.get("DLP_S3_SECRET_KEY", "dlp-dev-secret"),
        )

    @classmethod
    def labeler_from_env(cls, bucket: str) -> S3Store:
        """일반 라벨러 자격 증명 (라벨링 버킷 읽기 전용). 검수 화면용 서명 URL을 만들 때 쓴다."""
        return cls(
            bucket,
            endpoint_url=os.environ.get("DLP_S3_PUBLIC_ENDPOINT")
            or os.environ.get("DLP_S3_ENDPOINT", "http://localhost:8333"),
            access_key=os.environ.get("DLP_S3_LABELER_ACCESS_KEY", "dlp-dev-labeler"),
            secret_key=os.environ.get("DLP_S3_LABELER_SECRET_KEY", "dlp-dev-labeler-secret"),
        )

    def uri(self, key: str) -> str:
        return f"s3://{self.bucket}/{key}"

    def presign(self, key: str, expires_s: int = 7 * 24 * 3600) -> str:
        """이 저장소의 자격 증명으로 서명한 읽기 URL."""
        return self.client.generate_presigned_url(
            "get_object", Params={"Bucket": self.bucket, "Key": key}, ExpiresIn=expires_s
        )

    def head(self, key: str) -> StoredObject | None:
        try:
            resp = self.client.head_object(Bucket=self.bucket, Key=key)
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                return None
            raise
        return StoredObject(key, resp["ContentLength"], resp.get("Metadata", {}).get("sha256"))

    def put_file(self, key: str, path: Path, sha256: str) -> None:
        self.client.upload_file(
            str(path), self.bucket, key, ExtraArgs={"Metadata": {"sha256": sha256}}
        )

    def get_file(self, key: str, dest: Path) -> None:
        self.client.download_file(self.bucket, key, str(dest))


def store_from_spec(spec: str, bucket: str) -> ObjectStore:
    """'s3' 또는 'local:<디렉터리>'."""
    if spec == "s3":
        return S3Store.from_env(bucket)
    if spec.startswith("local:"):
        return LocalStore(Path(spec.removeprefix("local:")), bucket)
    raise ValueError(f"알 수 없는 저장소 지정: {spec}")
