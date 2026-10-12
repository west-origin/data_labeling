"""라벨링 버킷 노출 경로의 운영 점검 (F-REV-19, ADR 0039 한계).

DB 기준 회수(ADR 0039)는 라벨링 버킷 사본과 도구 작업을 지운다. 그래도 남는 길 중 코드로 확인할 수
있는 것을 점검한다 (`dlp services check --security`):

- 라벨링 버킷 버전 관리: 켜져 있으면(Enabled) 지운 사본의 이전 버전이 남아 같은 키로 다시 받을 수
  있다. 일시 중지(Suspended)여도 켜져 있던 동안의 비현재 버전이 남아 있으면 실패다.
- 라벨러 자격 증명 범위: 라벨링 버킷 읽기만 돼야 한다. 쓰기·삭제나 다른 버킷(원본·데이터셋·MLflow)
  읽기가 되면, 서명 URL을 만드는 이 키가 새는 순간 원본까지 열린다. 쓰기 거부는 임의 키에 빈
  객체를 써 보는 것으로 확인한다. 거부되지 않으면(설정 오류) 서비스 자격 증명으로 바로 지운다.
  삭제 거부는 없는 키로 확인한다. 다른 버킷은 목록과 (객체가 있으면) 객체 하나의 HEAD를 본다.

코드로 막지 못하는 것(이미 내려받은 사본, CVAT·Label Studio 내보내기, 사람에게 나눠 준 자격 증명의
사용)은 운영 절차다 (ADR 0039 "사고 대응").

공개 이름:
- `StorageCheck`: 점검 하나의 결과.
- `check_versioning_off`, `check_labeler_scope`: 점검 함수 (S3 호환 저장소).
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from botocore.exceptions import BotoCoreError, ClientError

from dlp_media.storage import S3Store

# 접근 거부로 보는 S3 오류 코드 (HEAD는 본문이 없어 코드가 상태 숫자다)
_DENIED_CODES = {"AccessDenied", "Forbidden", "403", "AllAccessDisabled"}
# 쓰기·삭제 거부를 확인할 때 쓰는 키 접두사 (라벨링 버킷, `sessions/` 밖)
PROBE_PREFIX = "ops-check/labeler-probe-"


@dataclass(frozen=True)
class StorageCheck:
    """점검 하나: 이름, 통과 여부, 설명 (실패면 고칠 방법)."""

    name: str
    ok: bool
    detail: str


def _denied(exc: ClientError) -> bool:
    """접근 거부(403)인가."""
    code = str(exc.response.get("Error", {}).get("Code", ""))
    status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return code in _DENIED_CODES or status == 403


def check_versioning_off(service: S3Store) -> StorageCheck:
    """라벨링 버킷 버전 관리가 꺼져 있고 비현재 버전이 남지 않았는지 (서비스 자격 증명).

    Enabled면 실패. Suspended면 `sessions/` 아래 비현재 버전·삭제 표시가 하나라도 있으면 실패.
    상태를 읽을 수 없으면(미지원·권한) 실패로 보고 이유를 남긴다.
    """
    name = f"versioning-off:{service.bucket}"
    try:
        status = service.client.get_bucket_versioning(Bucket=service.bucket).get("Status")
        if status == "Enabled":
            return StorageCheck(
                name, False, "버전 관리가 켜져 있어 회수로 지운 사본의 이전 버전이 남습니다 "
                "(끄고 비현재 버전을 정리하세요)",
            )  # fmt: skip
        if status != "Suspended":
            return StorageCheck(name, True, "버전 관리 꺼짐 (켠 적 없음)")
        old, marks, more = _leftover_versions(service)
    except (ClientError, BotoCoreError) as exc:
        return StorageCheck(name, False, f"버전 관리 상태를 확인할 수 없습니다 ({exc})")
    if old or marks:
        tail = " 이상" if more else ""
        return StorageCheck(
            name, False, f"버전 관리는 일시 중지지만 이전 버전 {old}개·삭제 표시 "
            f"{marks}개{tail}가 남아 있습니다 (비현재 버전을 지우세요)",
        )  # fmt: skip
    return StorageCheck(name, True, "버전 관리 일시 중지, 남은 이전 버전 없음")


def _leftover_versions(service: S3Store) -> tuple[int, int, bool]:
    """`sessions/` 아래 비현재 버전·삭제 표시 수와 '세지 않은 쪽이 남았는가'.

    S3는 한 번에 최대 1000개만 돌려주므로 IsTruncated면 다음 표시(KeyMarker·VersionIdMarker)로
    이어 읽는다. 키 순서상 뒤쪽에 남은 이전 버전을 놓치지 않게 끝 쪽까지 보되, 하나라도 찾은 쪽에서
    멈춘다 (점검은 있고 없음만 가리면 된다).
    """
    pages = service.client.get_paginator("list_object_versions").paginate(
        Bucket=service.bucket, Prefix="sessions/"
    )
    for page in pages:
        old = sum(1 for v in page.get("Versions", []) if not v.get("IsLatest", True))
        marks = len(page.get("DeleteMarkers", []))
        if old or marks:
            return old, marks, bool(page.get("IsTruncated"))
    return 0, 0, False


def _expect_denied(name: str, call: str, attempt: Callable[[], object]) -> StorageCheck:
    """attempt(인자 없는 호출)가 접근 거부로 끝나면 통과. 성공하면 실패, 다른 오류도 실패."""
    try:
        attempt()
    except ClientError as exc:
        if _denied(exc):
            return StorageCheck(name, True, f"{call} 거부됨")
        return StorageCheck(name, False, f"{call}: 거부가 아닌 오류 ({exc})")
    except BotoCoreError as exc:
        return StorageCheck(name, False, f"{call}: 확인할 수 없음 ({exc})")
    return StorageCheck(name, False, f"{call}이 허용됩니다 (라벨러 키 권한을 줄이세요)")


def _labeling_read(labeler: S3Store, service: S3Store) -> StorageCheck:
    """라벨러 키로 라벨링 버킷 객체를 읽을 수 있는가 (서명 URL이 열리려면 필요하다).

    서비스 키로 객체 하나를 찾아 라벨러 키로 HEAD한다. 라벨러 키의 목록(List) 권한은 요구하지 않는다
    (서명 URL에는 필요 없다). 버킷이 비었으면 확인하지 않고 통과로 둔다.
    """
    name = f"labeler-read:{labeler.bucket}"
    try:
        found = service.client.list_objects_v2(Bucket=service.bucket, MaxKeys=1).get("Contents", [])
        key = found[0].get("Key") if found else None
        if key is None:
            return StorageCheck(name, True, "버킷이 비어 읽기 확인 생략")
        labeler.client.head_object(Bucket=labeler.bucket, Key=key)
    except (ClientError, BotoCoreError) as exc:
        return StorageCheck(name, False, f"라벨러 키로 라벨링 버킷을 읽지 못합니다 ({exc})")
    return StorageCheck(name, True, "라벨링 버킷 객체 읽기 허용")


def _labeling_write(labeler: S3Store, service: S3Store) -> list[StorageCheck]:
    """라벨러 키로 라벨링 버킷에 쓰기·삭제가 거부되는가. 써지면 서비스 키로 바로 지운다."""
    probe = f"{PROBE_PREFIX}{uuid.uuid4().hex}"
    write = _expect_denied(
        f"labeler-no-write:{labeler.bucket}",
        "쓰기(PutObject)",
        lambda: labeler.client.put_object(Bucket=labeler.bucket, Key=probe, Body=b""),
    )
    if not write.ok:
        service.delete(probe)  # 설정 오류로 써진 빈 객체를 남기지 않는다
    missing = f"{PROBE_PREFIX}{uuid.uuid4().hex}"
    delete = _expect_denied(
        f"labeler-no-delete:{labeler.bucket}",
        "삭제(DeleteObject)",
        lambda: labeler.client.delete_object(Bucket=labeler.bucket, Key=missing),
    )
    return [write, delete]


def _other_bucket(labeler: S3Store, service: S3Store, bucket: str) -> list[StorageCheck]:
    """라벨러 키로 다른 버킷의 목록과 (있으면) 객체 하나의 HEAD가 거부되는가."""
    out = [
        _expect_denied(
            f"labeler-no-list:{bucket}",
            "목록(ListObjects)",
            lambda: labeler.client.list_objects_v2(Bucket=bucket, MaxKeys=1),
        )
    ]
    try:
        found = service.client.list_objects_v2(Bucket=bucket, MaxKeys=1).get("Contents", [])
    except (ClientError, BotoCoreError) as exc:
        out.append(StorageCheck(f"labeler-no-read:{bucket}", False, f"확인할 수 없음 ({exc})"))
        return out
    key = found[0].get("Key") if found else None
    if key is not None:
        out.append(
            _expect_denied(
                f"labeler-no-read:{bucket}",
                "객체 읽기(HeadObject)",
                lambda: labeler.client.head_object(Bucket=bucket, Key=key),
            )
        )
    return out


def check_labeler_scope(
    labeler: S3Store, service: S3Store, others: Sequence[str]
) -> list[StorageCheck]:
    """라벨러 자격 증명이 라벨링 버킷 읽기만 하는지 점검한다.

    Args:
        labeler: 라벨러 자격 증명의 라벨링 버킷 저장소 (서버가 닿는 엔드포인트).
        service: 서비스 자격 증명의 라벨링 버킷 저장소 (각 버킷의 객체 하나를 찾고, 설정 오류로
            써진 점검 객체를 지운다).
        others: 라벨러가 읽으면 안 되는 버킷 (원본·데이터셋·MLflow).

    Returns:
        점검 결과 (라벨링 객체 읽기, 쓰기·삭제 거부, 다른 버킷마다 목록·읽기 거부).

    부작용: 라벨링 버킷에 빈 점검 객체 쓰기 시도(거부돼야 한다, 써지면 바로 지운다), 없는 키 삭제
    시도, 다른 버킷 목록·HEAD 시도. 원본 객체 내용은 읽지 않는다.
    """
    checks = [_labeling_read(labeler, service), *_labeling_write(labeler, service)]
    for bucket in others:
        checks += _other_bucket(labeler, service, bucket)
    return checks
