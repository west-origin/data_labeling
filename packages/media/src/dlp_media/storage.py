"""객체 저장소. 원본(raw)과 라벨링(labeling) 버킷을 별도 인스턴스로 다룬다.

원본은 불변이다. 같은 키에 같은 내용(sha256)을 다시 올리면 건너뛰고, 다른 내용이면 오류를 낸다.

WP3·WP16, ADR 0003·0020. 모든 단계가 이 프로토콜(`ObjectStore`)로 파일을 주고받는다.

버킷 (이름은 config/defaults.yaml `buckets`):
- 원본 버킷(`dlp-raw`): 블러 전 원본 영상·센서 파일(`sessions/<세션>/raw/`), 파생물(PTS 인덱스,
  프록시, 정규화 Parquet, 검수 우선 구간, 프라이버시 탐지 표시: `sessions/<세션>/derived/`).
  얼굴 등이 그대로 보이므로
  원본 접근 권한자만 본다. 이 버킷의 저장소는 반드시 `dlp_cli.raw_access.raw_store`로 만든
  `dlp_media.audit.AuditedStore`로 감싸 쓴다 (모든 읽기·쓰기·서명 URL이 감사 기록에 남는다).
  정적 검사 테스트가 다른 경로로 원본 저장소를 만드는 것을 막는다.
- 라벨링 버킷(`dlp-labeling`): 블러본(`blurred_key`)과 렌더 기록 등 일반 라벨러가 봐도 되는 것.
  라벨러용 URL은 라벨러 자격 증명(`S3Store.labeler_from_env`, 읽기 전용)으로 서명한다.

구현:
- `LocalStore`: 로컬 디렉터리 (테스트·오프라인 개발, `local://<버킷>/<키>`).
- `S3Store`: S3 호환 (개발은 SeaweedFS, `s3://<버킷>/<키>`).
- `store_from_spec`: CLI `--store` 값('s3' | 'local:<디렉터리>')으로 저장소를 만든다.
- `put_immutable`: 불변 업로드 (원본 수집용). `sha256_file`: 파일 해시.
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
    """불변 키에 다른 내용을 올리려 했다 (원본이 바뀌었거나 다른 파일을 같은 스트림으로 지정)."""


@dataclass(frozen=True)
class StoredObject:
    """`head` 결과: 객체 메타데이터 (내용은 읽지 않는다)."""

    key: str
    # 바이트 수
    size: int
    # 올릴 때 기록한 sha256 (16진수). 기록이 없으면 None (외부에서 올린 객체 등).
    sha256: str | None


class ObjectStore(Protocol):
    """객체 저장소 프로토콜. 키는 버킷 안 상대 경로 ("sessions/<세션>/…")."""

    # 버킷 이름 (예: dlp-raw, dlp-labeling). 원본/라벨링 구분과 감사 기록에 쓴다.
    bucket: str

    def uri(self, key: str) -> str:
        """키의 URI (Stream.uri 등에 저장하는 값). 존재 여부는 보지 않는다."""
        ...

    def head(self, key: str) -> StoredObject | None:
        """객체 메타데이터. 없으면 None. 내용을 읽지 않으므로 감사 기록 대상이 아니다."""
        ...

    def put_file(self, key: str, path: Path, sha256: str) -> None:
        """로컬 파일을 키에 (덮어)쓴다. sha256은 호출자가 계산해 함께 기록한다."""
        ...

    def get_file(self, key: str, dest: Path) -> None:
        """키의 객체를 로컬 dest에 받는다. 없으면 예외 (구현마다 종류가 다르다)."""
        ...


def blurred_key(session_id: str, stream_id: str) -> str:
    """라벨링 버킷의 블러본 위치 (프라이버시 렌더가 쓰고, 검수·행동·큐레이션·내보내기가 읽는다)."""
    return f"sessions/{session_id}/blurred/{stream_id}.mp4"


def sha256_file(path: Path) -> str:
    """파일의 sha256 (16진수 64자). 1 MiB씩 읽어 큰 영상도 메모리를 적게 쓴다."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def put_immutable(store: ObjectStore, key: str, path: Path) -> bool:
    """올렸으면 True, 같은 내용이 이미 있어 건너뛰었으면 False.

    원본 수집(`ingest.ingest_session`)이 쓴다. 기존 객체의 기록된 sha256과 비교하므로, 해시 기록이
    없는 기존 객체(None)는 "다른 내용"으로 본다.

    Raises:
        ImmutableObjectError: 같은 키에 다른 내용이 이미 있을 때.
    """
    digest = sha256_file(path)
    existing = store.head(key)
    if existing is not None:
        if existing.sha256 == digest:
            return False
        raise ImmutableObjectError(f"{store.uri(key)}에 다른 내용이 이미 있습니다")
    store.put_file(key, path, digest)
    return True


class LocalStore:
    """파일 시스템 저장소 (테스트·오프라인 개발용). sha256은 옆 파일에 둔다.

    객체는 `<root>/<버킷>/<키>`, 해시는 `<…>.sha256` 파일이다. 버킷 밖을 가리키는 키("../")는
    거부한다.
    """

    def __init__(self, root: Path, bucket: str) -> None:
        """root: 버킷 디렉터리들의 상위 디렉터리, bucket: 버킷 이름 (하위 디렉터리)."""
        self.root = root / bucket
        self.bucket = bucket

    def uri(self, key: str) -> str:
        """`local://<버킷>/<키>`."""
        return f"local://{self.bucket}/{key}"

    def _path(self, key: str) -> Path:
        """키 → 실제 경로. 경로 탈출을 막는다.

        Raises:
            ValueError: 키가 버킷 디렉터리 밖을 가리킬 때.
        """
        path = (self.root / key).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError(f"버킷 밖을 가리키는 키: {key}")
        return path

    def head(self, key: str) -> StoredObject | None:
        """파일이 있으면 크기와 옆 파일의 해시 (옆 파일이 없으면 sha256=None)."""
        path = self._path(key)
        if not path.is_file():
            return None
        sidecar = path.with_name(path.name + ".sha256")
        sha = sidecar.read_text().strip() if sidecar.is_file() else None
        return StoredObject(key, path.stat().st_size, sha)

    def put_file(self, key: str, path: Path, sha256: str) -> None:
        """파일을 복사하고 해시 옆 파일을 쓴다 (상위 디렉터리를 만든다, 덮어쓴다)."""
        dest = self._path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        dest.with_name(dest.name + ".sha256").write_text(sha256)

    def get_file(self, key: str, dest: Path) -> None:
        """파일을 dest로 복사한다. 없으면 FileNotFoundError."""
        shutil.copyfile(self._path(key), dest)


class S3Store:
    """S3 호환 저장소 (SeaweedFS). sha256은 객체 메타데이터에 둔다."""

    def __init__(self, bucket: str, *, endpoint_url: str, access_key: str, secret_key: str) -> None:
        """자격 증명으로 boto3 S3 클라이언트를 만든다 (네트워크 호출은 아직 없다).

        region은 SeaweedFS가 보지 않으므로 고정값이다.
        """
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
        """서비스 자격 증명으로 만든다.

        환경 변수 DLP_S3_ENDPOINT·DLP_S3_ACCESS_KEY·DLP_S3_SECRET_KEY (없으면 개발 기본값).

        원본 버킷에 쓸 때는 직접 부르지 말고 `dlp_cli.raw_access.raw_store`를 쓴다 (감사).
        """
        return cls(
            bucket,
            endpoint_url=os.environ.get("DLP_S3_ENDPOINT", "http://localhost:8333"),
            access_key=os.environ.get("DLP_S3_ACCESS_KEY", "dlp-dev-access"),
            secret_key=os.environ.get("DLP_S3_SECRET_KEY", "dlp-dev-secret"),
        )

    @classmethod
    def labeler_from_env(cls, bucket: str) -> S3Store:
        """일반 라벨러 자격 증명 (라벨링 버킷 읽기 전용). 검수 화면용 서명 URL을 만들 때 쓴다.

        엔드포인트는 브라우저가 닿는 주소(DLP_S3_PUBLIC_ENDPOINT)를 먼저 쓴다. 이 자격 증명은
        원본 버킷을 읽을 수 없으므로, 실수로 원본 키를 서명해도 그 URL은 열리지 않는다.
        """
        return cls(
            bucket,
            endpoint_url=os.environ.get("DLP_S3_PUBLIC_ENDPOINT")
            or os.environ.get("DLP_S3_ENDPOINT", "http://localhost:8333"),
            access_key=os.environ.get("DLP_S3_LABELER_ACCESS_KEY", "dlp-dev-labeler"),
            secret_key=os.environ.get("DLP_S3_LABELER_SECRET_KEY", "dlp-dev-labeler-secret"),
        )

    def uri(self, key: str) -> str:
        """`s3://<버킷>/<키>`."""
        return f"s3://{self.bucket}/{key}"

    def presign(self, key: str, expires_s: int = 7 * 24 * 3600) -> str:
        """이 저장소의 자격 증명으로 서명한 읽기 URL.

        expires_s: 유효 시간(초, 기본 7일). 원본 버킷이면 `AuditedStore.presign`을 거쳐야 한다.
        """
        return self.client.generate_presigned_url(
            "get_object", Params={"Bucket": self.bucket, "Key": key}, ExpiresIn=expires_s
        )

    def head(self, key: str) -> StoredObject | None:
        """HEAD 요청. 없으면(404·NoSuchKey·NotFound) None, 그 밖의 오류는 다시 던진다."""
        try:
            resp = self.client.head_object(Bucket=self.bucket, Key=key)
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                return None
            raise
        return StoredObject(key, resp["ContentLength"], resp.get("Metadata", {}).get("sha256"))

    def put_file(self, key: str, path: Path, sha256: str) -> None:
        """업로드한다 (sha256은 사용자 메타데이터 `x-amz-meta-sha256`)."""
        self.client.upload_file(
            str(path), self.bucket, key, ExtraArgs={"Metadata": {"sha256": sha256}}
        )

    def get_file(self, key: str, dest: Path) -> None:
        """내려받는다. 없으면 botocore ClientError."""
        self.client.download_file(self.bucket, key, str(dest))


def store_from_spec(spec: str, bucket: str) -> ObjectStore:
    """'s3' 또는 'local:<디렉터리>'.

    CLI `--store` 값으로 저장소를 만든다. 원본 버킷은 이 함수를 직접 쓰지 말고
    `dlp_cli.raw_access.raw_store`(감사 저장소)로 만든다.

    Raises:
        ValueError: 알 수 없는 지정일 때.
    """
    if spec == "s3":
        return S3Store.from_env(bucket)
    if spec.startswith("local:"):
        return LocalStore(Path(spec.removeprefix("local:")), bucket)
    raise ValueError(f"알 수 없는 저장소 지정: {spec}")
