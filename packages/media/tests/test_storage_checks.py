"""라벨링 버킷 노출 경로 운영 점검 (F-REV-19, `dlp_media.storage_checks`).

네트워크 없이 botocore `Stubber`로 S3 응답을 정해 둔다.

정답 근거: 버전 관리가 켜졌거나 일시 중지 뒤 비현재 버전이 남으면 실패, 켠 적 없으면 통과. 라벨러
키는 라벨링 버킷 읽기만 허용돼야 하고, 쓰기·삭제·다른 버킷 목록·객체 읽기가 거부(403)되지 않으면
실패다. 설정 오류로 점검 객체가 써지면 서비스 키로 지운다.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import pytest
from botocore.stub import ANY, Stubber

from dlp_media.storage import S3Store
from dlp_media.storage_checks import check_labeler_scope, check_versioning_off

LABELING = "dlp-labeling"


def _store(bucket: str = LABELING) -> S3Store:
    """네트워크를 쓰지 않는 S3 저장소 (응답은 Stubber가 준다)."""
    return S3Store(bucket, endpoint_url="http://s3.invalid", access_key="a", secret_key="b")


@pytest.fixture
def stores() -> Iterator[tuple[S3Store, Stubber, S3Store, Stubber]]:
    """(라벨러, 라벨러 stub, 서비스, 서비스 stub). 테스트 끝에 모든 응답을 썼는지 본다."""
    labeler, service = _store(), _store()
    with Stubber(labeler.client) as ls, Stubber(service.client) as ss:
        yield labeler, ls, service, ss
        ls.assert_no_pending_responses()
        ss.assert_no_pending_responses()


def _denied(stub: Stubber, method: str, params: dict[str, Any]) -> None:
    """접근 거부(403) 응답을 하나 정해 둔다."""
    stub.add_client_error(
        method, service_error_code="AccessDenied", http_status_code=403, expected_params=params
    )


def _listing(keys: list[str]) -> dict[str, Any]:
    """ListObjectsV2 응답."""
    when = datetime(2026, 10, 11, tzinfo=UTC)
    return {"Contents": [{"Key": k, "Size": 1, "LastModified": when} for k in keys]}


def test_versioning_rules() -> None:
    """켠 적 없음 → 통과, 켜짐 → 실패, 일시 중지 + 이전 버전 남음 → 실패, 일시 중지 + 없음 → 통과.

    상태를 읽을 수 없으면(미지원) 실패다.
    """
    service = _store()
    params = {"Bucket": LABELING}
    versions = {"Bucket": LABELING, "Prefix": "sessions/"}
    old = {"Versions": [{"Key": "sessions/s/review/a/b.mp4", "VersionId": "1", "IsLatest": False}]}
    with Stubber(service.client) as stub:
        stub.add_response("get_bucket_versioning", {}, params)
        stub.add_response("get_bucket_versioning", {"Status": "Enabled"}, params)
        stub.add_response("get_bucket_versioning", {"Status": "Suspended"}, params)
        stub.add_response("list_object_versions", old, versions)
        stub.add_response("get_bucket_versioning", {"Status": "Suspended"}, params)
        stub.add_response("list_object_versions", {}, versions)
        stub.add_client_error("get_bucket_versioning", "NotImplemented", http_status_code=501)
        got = [check_versioning_off(service) for _ in range(5)]
    assert [c.ok for c in got] == [True, False, False, True, False]
    assert "켜져" in got[1].detail and "이전 버전 1개" in got[2].detail
    assert "확인할 수 없습니다" in got[4].detail


def test_labeler_scope_passes_when_only_labeling_read_is_allowed(
    stores: tuple[S3Store, Stubber, S3Store, Stubber],
) -> None:
    """라벨링 읽기만 되고 쓰기·삭제·원본 목록·원본 객체 읽기가 모두 403이면 모두 통과다."""
    labeler, ls, service, ss = stores
    video = "sessions/s/review/a/v.mp4"
    ss.add_response("list_objects_v2", _listing([video]), {"Bucket": LABELING, "MaxKeys": 1})
    ls.add_response("head_object", {}, {"Bucket": LABELING, "Key": video})
    _denied(ls, "put_object", {"Bucket": LABELING, "Key": ANY, "Body": b""})
    _denied(ls, "delete_object", {"Bucket": LABELING, "Key": ANY})
    _denied(ls, "list_objects_v2", {"Bucket": "dlp-raw", "MaxKeys": 1})
    raw = {"Bucket": "dlp-raw", "MaxKeys": 1}
    ss.add_response("list_objects_v2", _listing(["sessions/s/raw/v.mp4"]), raw)
    ls.add_client_error("head_object", "403", http_status_code=403)
    got = check_labeler_scope(labeler, service, ["dlp-raw"])
    assert [(c.name, c.ok) for c in got] == [
        ("labeler-read:dlp-labeling", True),
        ("labeler-no-write:dlp-labeling", True),
        ("labeler-no-delete:dlp-labeling", True),
        ("labeler-no-list:dlp-raw", True),
        ("labeler-no-read:dlp-raw", True),
    ]


def test_labeler_scope_fails_on_write_and_other_bucket_and_cleans_probe(
    stores: tuple[S3Store, Stubber, S3Store, Stubber],
) -> None:
    """쓰기가 허용되면 실패하고 써진 점검 객체를 서비스 키로 지운다. 다른 버킷 목록이 허용돼도
    실패다 (빈 버킷이면 객체 읽기는 보지 않는다). 403이 아닌 오류는 거부로 보지 않는다."""
    labeler, ls, service, ss = stores
    ss.add_response("list_objects_v2", _listing([]), {"Bucket": LABELING, "MaxKeys": 1})
    ls.add_response("put_object", {}, {"Bucket": LABELING, "Key": ANY, "Body": b""})
    ss.add_response("delete_object", {}, {"Bucket": LABELING, "Key": ANY})
    ls.add_client_error("delete_object", "InternalError", http_status_code=500)
    ls.add_response("list_objects_v2", _listing([]), {"Bucket": "dlp-datasets", "MaxKeys": 1})
    ss.add_response("list_objects_v2", _listing([]), {"Bucket": "dlp-datasets", "MaxKeys": 1})
    got = {c.name: c for c in check_labeler_scope(labeler, service, ["dlp-datasets"])}
    assert got["labeler-read:dlp-labeling"].ok  # 빈 버킷이면 읽기 확인 생략
    assert not got["labeler-no-write:dlp-labeling"].ok
    assert "허용됩니다" in got["labeler-no-write:dlp-labeling"].detail
    assert not got["labeler-no-delete:dlp-labeling"].ok
    assert "거부가 아닌 오류" in got["labeler-no-delete:dlp-labeling"].detail
    assert not got["labeler-no-list:dlp-datasets"].ok
    assert "labeler-no-read:dlp-datasets" not in got
