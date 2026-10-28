"""객체 저장소. 원본(raw)과 라벨링(labeling) 버킷을 별도 인스턴스로 다룬다.

원본은 불변이다. 같은 키에 같은 내용(sha256)을 다시 올리면 건너뛰고, 다른 내용이면 오류를 낸다.
`ObjectStore` 프로토콜에는 삭제가 없다. 삭제는 `DeletableStore`만 하며, 라벨링 버킷의 일반 라벨러용
워터마크 사본을 회수 실행기(`dlp_review.revoke`, ADR 0039)가 지울 때만 쓴다. 원본 감사 저장소
(`dlp_media.audit.AuditedStore`)는 삭제가 없어 원본은 계속 지울 수 없다.

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
- `CreateOnlyStore`: 키가 없을 때만 원자적으로 쓰는 저장소 (`LocalStore`: 임시 파일 + 배타 링크,
  `S3Store`: 조건부 쓰기 `If-None-Match: *`, 원본 감사 저장소는 감싼 저장소에 넘긴다).
  동시에 같은 키로 다른 내용이 와도 하나만 올라간다 (F-PMS-40).
- `DeletableStore`: 삭제할 수 있는 저장소 (`LocalStore`, `S3Store`). 회수 실행기만 쓴다.
- `ListableStore`, `ListedObject`: 접두사로 객체를 나열할 수 있는 저장소와 그 결과
  (`LocalStore`, `S3Store`). DB 기록 없이 남은 라벨링 버킷 사본(고아)을 찾는 `dlp_review.orphans`만
  쓴다 (F-REV-15). 원본 감사 저장소는 나열이 없다.
- `STORE_ERRORS`: 저장소 호출 실패로 볼 예외 묶음.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import boto3
from botocore.exceptions import BotoCoreError, ClientError

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client
    from mypy_boto3_s3.type_defs import CompletedPartTypeDef


# 저장소 호출 실패로 볼 예외 (S3 응답·연결 오류, 로컬 파일 오류). 실패를 모아 보고할 때 쓴다.
STORE_ERRORS: tuple[type[Exception], ...] = (ClientError, BotoCoreError, OSError)


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


@runtime_checkable
class CreateOnlyStore(ObjectStore, Protocol):
    """키가 없을 때만 원자적으로 쓰는 저장소 (`put_immutable`이 먼저 쓴다, F-PMS-40)."""

    def put_file_new(self, key: str, path: Path, sha256: str) -> bool:
        """키가 없으면 올리고 True, 이미 있으면 아무것도 쓰지 않고 False.

        확인과 쓰기 사이에 다른 쓰기가 끼어들 수 없다 (동시에 여러 개가 와도 True는 하나).
        """
        ...


@runtime_checkable
class DeletableStore(ObjectStore, Protocol):
    """객체를 지울 수 있는 저장소. 라벨링 버킷의 워터마크 사본 회수(ADR 0039)에만 쓴다.

    `ObjectStore`를 받는 함수는 삭제할 수 없다. 원본 감사 저장소는 이 프로토콜을 만족하지 않는다.
    """

    def delete(self, key: str) -> None:
        """키의 객체를 지운다. 없어도 성공이다 (멱등). 그 밖의 오류는 예외."""
        ...


@dataclass(frozen=True)
class ListedObject:
    """`list_objects` 결과 하나: 키, 바이트 수, 마지막 수정 시각(UTC, 시간대 있음)."""

    key: str
    size: int
    modified_at: datetime


@runtime_checkable
class ListableStore(ObjectStore, Protocol):
    """접두사로 객체를 나열할 수 있는 저장소. 라벨링 버킷의 고아 사본 찾기(F-REV-15)에만 쓴다.

    원본 감사 저장소는 이 프로토콜을 만족하지 않는다 (원본 목록은 감사 경로 밖으로 내지 않는다).
    """

    def list_objects(self, prefix: str) -> list[ListedObject]:
        """키가 prefix로 시작하는 객체 (키 순). 없으면 빈 목록. 그 밖의 오류는 예외."""
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


# 원자적 생성에서 진 쪽이 승자의 해시 기록을 기다리는 횟수·간격 (LocalStore는 객체를 먼저 만들고
# 해시 옆 파일을 곧바로 쓴다. 그 사이에 head하면 해시가 아직 없다)
_HASH_WAIT_TRIES = 20
_HASH_WAIT_S = 0.05


def put_immutable(store: ObjectStore, key: str, path: Path) -> bool:
    """올렸으면 True, 같은 내용이 이미 있어 건너뛰었으면 False.

    원본 수집(`ingest.ingest_session`)이 쓴다. 기존 객체의 기록된 sha256과 비교하므로, 해시 기록이
    없는 기존 객체(None)는 "다른 내용"으로 본다.

    동시 쓰기 (F-PMS-40): `CreateOnlyStore`면 키가 없을 때만 원자적으로 쓴다 (SeaweedFS 4.48은
    `If-None-Match: *`를 지키는 것을 실측했다: 동시 8개 중 하나만 성공, 나머지 412). 그렇지 않은
    저장소는 head → 쓰기 사이를 막을 수 없으므로 쓴 뒤 다시 head해 기록된 해시가 내 것인지 본다.
    덮인 쪽은 ImmutableObjectError를 낸다 (먼저 쓴 내용은 이미 덮였을 수 있다 → 그 저장소는 원본에
    쓰지 않는다). 어느 경우든 쓴 뒤 해시를 다시 대조한다.

    Raises:
        ImmutableObjectError: 같은 키에 다른 내용이 이미 있을 때
            (동시에 다른 내용이 먼저 올라간 때 포함).
    """
    digest = sha256_file(path)
    existing = store.head(key)
    if existing is None:
        if isinstance(store, CreateOnlyStore):
            created = store.put_file_new(key, path, digest)
        else:
            store.put_file(key, path, digest)
            created = True
        existing = _head_with_hash(store, key)
        if created and existing is not None and existing.sha256 == digest:
            return True
    if existing is not None and existing.sha256 == digest:
        return False
    raise ImmutableObjectError(f"{store.uri(key)}에 다른 내용이 이미 있습니다")


def _head_with_hash(store: ObjectStore, key: str) -> StoredObject | None:
    """head하되, 객체는 있고 해시 기록이 아직 없으면 잠깐 기다려 다시 본다 (위 상수)."""
    head = store.head(key)
    for _ in range(_HASH_WAIT_TRIES):
        if head is None or head.sha256 is not None:
            break
        time.sleep(_HASH_WAIT_S)
        head = store.head(key)
    return head


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

    def put_file_new(self, key: str, path: Path, sha256: str) -> bool:
        """없을 때만 쓴다: 같은 디렉터리의 임시 파일에 복사한 뒤 `os.link`로 배타 생성한다.

        `os.link`는 대상이 있으면 FileExistsError라 확인과 생성이 한 번의 시스템 호출이다. 해시 옆
        파일은 이긴 쪽만 객체를 만든 뒤 쓴다 (그 사이 head는 sha256=None을 볼 수 있다,
        `put_immutable`이 잠깐 기다린다).
        """
        dest = self._path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.name}.", suffix=".tmp")
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            shutil.copyfile(path, tmp)
            try:
                os.link(tmp, dest)
            except FileExistsError:
                return False
        finally:
            tmp.unlink(missing_ok=True)
        side_tmp = tmp.with_name(tmp.name + ".sha256")
        side_tmp.write_text(sha256)
        side_tmp.replace(dest.with_name(dest.name + ".sha256"))
        return True

    def get_file(self, key: str, dest: Path) -> None:
        """파일을 dest로 복사한다. 없으면 FileNotFoundError."""
        shutil.copyfile(self._path(key), dest)

    def delete(self, key: str) -> None:
        """파일과 해시 옆 파일을 지운다. 없어도 성공이다 (경로 탈출 키는 ValueError)."""
        path = self._path(key)
        path.unlink(missing_ok=True)
        path.with_name(path.name + ".sha256").unlink(missing_ok=True)

    def list_objects(self, prefix: str) -> list[ListedObject]:
        """키가 prefix로 시작하는 파일 (키 순). 해시 옆 파일(`<객체>.sha256`)은 객체가 아니라 뺀다.

        수정 시각은 파일 mtime(UTC)이다.
        """
        if not self.root.is_dir():
            return []
        out: list[ListedObject] = []
        for path in self.root.rglob("*"):
            key = path.relative_to(self.root).as_posix()
            if path.name.startswith(".") and ".tmp" in path.name:
                continue  # put_file_new의 쓰는 중 임시 파일
            sidecar = path.suffix == ".sha256" and path.with_name(path.stem).is_file()
            if not path.is_file() or sidecar or not key.startswith(prefix):
                continue
            stat = path.stat()
            out.append(
                ListedObject(key, stat.st_size, datetime.fromtimestamp(stat.st_mtime, tz=UTC))
            )
        return sorted(out, key=lambda o: o.key)


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
    def labeler_from_env(cls, bucket: str, *, public: bool = True) -> S3Store:
        """일반 라벨러 자격 증명 (라벨링 버킷 읽기 전용). 검수 화면용 서명 URL을 만들 때 쓴다.

        엔드포인트는 브라우저가 닿는 주소(DLP_S3_PUBLIC_ENDPOINT)를 먼저 쓴다. public이 False면
        서버가 닿는 주소(DLP_S3_ENDPOINT)만 쓴다 (권한 범위 점검 `dlp services check --security`).
        이 자격 증명은 원본 버킷을 읽을 수 없으므로, 실수로 원본 키를 서명해도 그 URL은 열리지
        않는다 (그 범위는 `dlp_media.storage_checks.check_labeler_scope`로 점검한다).
        """
        public_url = os.environ.get("DLP_S3_PUBLIC_ENDPOINT") if public else None
        return cls(
            bucket,
            endpoint_url=public_url or os.environ.get("DLP_S3_ENDPOINT", "http://localhost:8333"),
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

    def put_file_new(
        self, key: str, path: Path, sha256: str, *, part_size: int | None = None
    ) -> bool:
        """조건부 쓰기(`If-None-Match: *`): 키가 없을 때만 만든다. 이미 있으면(412) False.

        boto3 전송 관리자(`upload_file`)는 이 조건을 넘기지 못해 직접 요청한다. 조각 크기 이하는
        PutObject 한 번, 그보다 크면 분할 업로드를 하고 완료 요청에 조건을 건다 (조건이 실패하면
        분할 업로드를 버린다). part_size: 조각 크기 바이트 (테스트용, 기본 `_PART_SIZE`).
        """
        part = part_size or _PART_SIZE
        meta = {"sha256": sha256}
        try:
            if path.stat().st_size <= part:
                with path.open("rb") as f:
                    self.client.put_object(
                        Bucket=self.bucket, Key=key, Body=f, Metadata=meta, IfNoneMatch="*"
                    )
            else:
                self._multipart_new(key, path, meta, part)
        except ClientError as exc:
            if _precondition_failed(exc):
                return False
            raise
        return True

    def _multipart_new(self, key: str, path: Path, meta: dict[str, str], part: int) -> None:
        """분할 업로드 후 `If-None-Match: *`로 완료한다. 실패하면 업로드를 버리고 다시 던진다."""
        upload_id = self.client.create_multipart_upload(Bucket=self.bucket, Key=key, Metadata=meta)[
            "UploadId"
        ]
        try:
            parts: list[CompletedPartTypeDef] = []
            with path.open("rb") as f:
                for number, chunk in enumerate(iter(lambda: f.read(part), b""), start=1):
                    resp = self.client.upload_part(
                        Bucket=self.bucket,
                        Key=key,
                        UploadId=upload_id,
                        PartNumber=number,
                        Body=chunk,
                    )
                    parts.append({"ETag": resp["ETag"], "PartNumber": number})
            self.client.complete_multipart_upload(
                Bucket=self.bucket,
                Key=key,
                UploadId=upload_id,
                MultipartUpload={"Parts": parts},
                IfNoneMatch="*",
            )
        except BaseException:
            self.client.abort_multipart_upload(Bucket=self.bucket, Key=key, UploadId=upload_id)
            raise

    def get_file(self, key: str, dest: Path) -> None:
        """내려받는다. 없으면 botocore ClientError."""
        self.client.download_file(self.bucket, key, str(dest))

    def delete(self, key: str) -> None:
        """DELETE 요청. S3는 없는 키도 성공으로 답한다 (멱등). 권한·연결 오류는 ClientError."""
        self.client.delete_object(Bucket=self.bucket, Key=key)

    def list_objects(self, prefix: str) -> list[ListedObject]:
        """ListObjectsV2를 끝까지 넘겨 읽는다 (키 순). 권한·연결 오류는 ClientError.

        수정 시각은 S3 LastModified(시간대 있음)다.
        """
        out: list[ListedObject] = []
        pages = self.client.get_paginator("list_objects_v2").paginate(
            Bucket=self.bucket, Prefix=prefix
        )
        for page in pages:
            for obj in page.get("Contents", []):
                key, size, at = obj.get("Key"), obj.get("Size"), obj.get("LastModified")
                if key is not None and size is not None and at is not None:
                    out.append(ListedObject(key, size, at))
        return sorted(out, key=lambda o: o.key)


# S3 조건부 쓰기 조각 크기: 이 이하는 PutObject 한 번, 넘으면 이 크기 조각으로 분할 업로드
# (S3 조각 수 상한 10000 → 최대 약 1.25 TB, 조각 하나를 메모리에 읽는다)
_PART_SIZE = 128 << 20


def _precondition_failed(exc: ClientError) -> bool:
    """조건부 쓰기가 "이미 있음"으로 거절됐는지 (412 PreconditionFailed, 동시 경합의 409 포함)."""
    code = exc.response.get("Error", {}).get("Code")
    status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return code in ("PreconditionFailed", "ConditionalRequestConflict") or status == 412


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
