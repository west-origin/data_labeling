"""열린 Label Studio 작업의 서명 URL 갱신 (F-REV-08).

서비스 없이 시간대를 보존하는 SQLite(`dlp_schema.testing.sqlite_db`)와 LocalStore로 돈다. Label
Studio는 실제 `LabelStudioClient`를 `httpx.MockTransport`(가짜 응답)에 붙여 쓴다. 라벨러 자격 증명
서명은 만료 초를 URL에 그대로 적는 가짜 서명기다. 실제 S3·Label Studio에서의 반영은 `test_flow.py`의
서비스 테스트가 본다.

정답 근거: 갱신 대상 = 이 세션의 Label Studio 작업 라벨 작업 중 열린(open) 작업이고, 렌더 해시가
지금 운영 블러와 같고(낡지 않음), 도구의 주인(session_id·assignee)이 작업 행과 같고, 라벨링 버킷에
매체가 있는 것. 새 URL은 라벨러 서명기로 `review.yaml media.signed_url_ttl_s` 초 동안 유효하게
서명하고 원본 버킷을 담지 않는다. 수집·회수한 작업과 CVAT 작업은 건드리지 않는다.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
import sqlalchemy as sa
from pydantic import ValidationError

from dlp_media.storage import LocalStore, S3Store
from dlp_privacy.hold import hold_session
from dlp_privacy.policy import PrivacyPolicy
from dlp_privacy.policy import load_policy as load_privacy_policy
from dlp_privacy.runner import operational_blur, render_hash
from dlp_review.clients import LabelStudioClient
from dlp_review.ops.policy import MediaPolicy, load_policy
from dlp_review.refresh import refresh_session_urls, sessions_with_open_ls_tasks
from dlp_review.tasks import ReviewSetup, label_studio_task_data, timeseries_key
from dlp_schema import load_ontology, repo_root
from dlp_schema.db.repository import (
    insert_review_task,
    insert_session,
    register_ontology,
    set_lifecycle,
)
from dlp_schema.db.tables import review_tasks
from dlp_schema.review import ReviewStage, ReviewTask, ReviewTaskStatus, ReviewTool
from dlp_schema.session import LifecycleState, PrivacyState
from dlp_schema.testing import FIXED_TIME, make_session, sqlite_db

SID = "s001"
TTL = 3600  # 시험 정책의 서명 URL 유효 시간 (기본 7일과 다르게)


class FakeSigner:
    """라벨러 자격 증명 서명기 대역. 서명한 (키, 만료 초)를 남기고 만료를 URL에 적는다."""

    def __init__(self, bucket: str = "dlp-labeling") -> None:
        self.bucket = bucket
        self.calls: list[tuple[str, int]] = []

    def presign(self, key: str, expires_s: int = 7 * 24 * 3600) -> str:
        """`https://s3.test/<버킷>/<키>?X-Amz-Expires=<초>`."""
        self.calls.append((key, expires_s))
        return f"https://s3.test/{self.bucket}/{key}?X-Amz-Expires={expires_s}"


def test_review_policy_holds_signed_url_ttl() -> None:
    """서명 URL 유효 시간은 정책 값이다 (기본 7일, S3 SigV4 상한 7일을 넘을 수 없다)."""
    media = load_policy(repo_root()).media
    assert media.signed_url_ttl_s == 7 * 24 * 3600
    with pytest.raises(ValidationError):
        MediaPolicy.model_validate({**media.model_dump(), "signed_url_ttl_s": 7 * 24 * 3600 + 1})


def _setup(tmp_path: Path, ls: LabelStudioClient | None, signer: FakeSigner) -> ReviewSetup:
    """로컬 저장소, 가짜 Label Studio·서명기, 유효 시간 TTL인 매체 정책."""
    media = load_policy(repo_root()).media.model_copy(update={"signed_url_ttl_s": TTL})
    return ReviewSetup(
        raw=LocalStore(tmp_path, "dlp-raw"), labeling=LocalStore(tmp_path, "dlp-labeling"),
        labeling_reader=cast(S3Store, signer), ontology=None, label_studio=ls, media=media,
        privacy=load_privacy_policy(repo_root()),
    )  # fmt: skip


def test_task_data_is_signed_with_policy_ttl_by_labeler_signer(tmp_path: Path) -> None:
    """작업 data의 영상·시계열 URL은 라벨러 서명기로 정책 유효 시간만큼 서명한다.

    회귀(F-REV-08): 예전에는 `presign` 기본값(7일)이 코드에 박혀 정책으로 바꿀 수 없었다.
    """
    signer = FakeSigner()
    setup = _setup(tmp_path, None, signer)
    data = label_studio_task_data(setup, SID, "labeler01", "v.mp4", timeseries_key(SID))
    assert signer.calls == [("v.mp4", TTL), (f"sessions/{SID}/review/timeseries.csv", TTL)]
    assert data == {
        "video": f"https://s3.test/dlp-labeling/v.mp4?X-Amz-Expires={TTL}",
        "timeseries": f"https://s3.test/dlp-labeling/sessions/{SID}/review/timeseries.csv"
        f"?X-Amz-Expires={TTL}",
        "session_id": SID,
        "assignee": "labeler01",
    }
    # 원본 버킷을 서명하는 설정이면(잘못 연결) 작업 data를 만들지 않는다
    with pytest.raises(Exception, match="원본"):
        label_studio_task_data(
            _setup(tmp_path, None, FakeSigner("dlp-raw")), SID, "labeler01", "v.mp4", "c.csv"
        )


# ── 갱신 ──


@pytest.fixture
def engine() -> Iterator[sa.Engine]:
    """표와 온톨로지 v1을 만든 메모리 SQLite."""
    eng = sqlite_db()
    with eng.begin() as conn:
        register_ontology(conn, load_ontology(repo_root() / "config/ontology/v1"))
    yield eng
    eng.dispose()


class FakeLabelStudio:
    """Label Studio 작업 data를 들고 GET·PATCH에 답하는 가짜 전송. PATCH 본문을 남긴다."""

    def __init__(self, tasks: dict[int, dict[str, Any]]) -> None:
        self.tasks = tasks
        self.patches: list[tuple[int, dict[str, Any]]] = []

    def __call__(self, req: httpx.Request) -> httpx.Response:
        parts = req.url.path.strip("/").split("/")
        if parts[:2] != ["api", "tasks"] or len(parts) != 3:
            return httpx.Response(404)
        tid = int(parts[2])
        if tid not in self.tasks:
            return httpx.Response(404, json={"detail": "not found"})
        if req.method == "PATCH":
            body = json.loads(req.content)
            self.patches.append((tid, body))
            self.tasks[tid]["data"] = body["data"]
        return httpx.Response(200, json={"id": tid, **self.tasks[tid]})

    def client(self) -> LabelStudioClient:
        """이 전송을 쓰는 Label Studio 클라이언트."""
        ls = LabelStudioClient("http://ls", "token")
        ls.http = httpx.Client(base_url="http://ls", transport=httpx.MockTransport(self))
        return ls


def _current_hash(policy: PrivacyPolicy) -> str:
    """블러 라벨이 없는 세션 바디캠의 지금 렌더 해시 (낡지 않은 작업의 해시)."""
    return render_hash(operational_blur([], "bodycam"), policy)


def _ls_task(key: str, uri: str, rhash: str, **kw: Any) -> ReviewTask:
    """담당자 labeler01의 Label Studio 작업 라벨 작업 (외부 ID는 키의 숫자)."""
    fields: dict[str, Any] = {
        "task_key": key, "tool": ReviewTool.LABEL_STUDIO, "external_id": key.split(":")[1],
        "session_id": SID, "stream_id": "bodycam", "stage": ReviewStage.LABELING,
        "assignee": "labeler01", "media_uri": uri, "label_kinds": ("action",),
        "created_at": FIXED_TIME, "render_hash": rhash,
    }  # fmt: skip
    return ReviewTask.model_validate({**fields, **kw})


def test_refresh_resigns_only_open_current_label_studio_tasks(
    engine: sa.Engine, tmp_path: Path
) -> None:
    """열린·낡지 않은 Label Studio 작업만 다시 서명해 작업 data를 바꾼다.

    작업 1: 갱신. 2(수집)·3(회수)·cvat:7: 건드리지 않음. 4(렌더 해시가 다름): 건너뜀(회수 안내).
    5(도구 주인이 다른 세션)·6(도구에 없음)·8(매체 없음)·9(원본 버킷 매체): 오류.
    오류·건너뜀이 있어도 다른 작업은 갱신한다.
    """
    signer = FakeSigner()
    privacy = load_privacy_policy(repo_root())
    labeling = LocalStore(tmp_path, "dlp-labeling")
    current = _current_hash(privacy)
    video = f"sessions/{SID}/review/labeler01/bodycam-{current[:16]}.mp4"
    media = tmp_path / "m.mp4"
    media.write_bytes(b"video")
    for key in (video, timeseries_key(SID)):
        labeling.put_file(key, media, "0" * 64)
    uri = labeling.uri(video)
    missing = labeling.uri(f"sessions/{SID}/review/labeler01/gone.mp4")
    # 낡은 매체는 키도 다르다 (내용 주소: 렌더 해시 앞부분)
    old_uri = labeling.uri(f"sessions/{SID}/review/labeler01/bodycam-{'f' * 16}.mp4")
    rows = [
        _ls_task("label_studio:1", uri, current),
        _ls_task("label_studio:2", uri, current, status=ReviewTaskStatus.COLLECTED,
                 collected_at=FIXED_TIME),
        _ls_task("label_studio:3", uri, current, status=ReviewTaskStatus.REVOKED,
                 revoke_requested_at=FIXED_TIME, revoke_reason="sweep"),
        _ls_task("label_studio:4", old_uri, "f" * 64),
        _ls_task("label_studio:5", uri, current),
        _ls_task("label_studio:6", uri, current),
        _ls_task("cvat:7", uri, current, tool=ReviewTool.CVAT),
        _ls_task("label_studio:8", missing, current),
        _ls_task("label_studio:9", f"local://dlp-raw/sessions/{SID}/raw/bodycam.mp4", current),
    ]  # fmt: skip
    with engine.begin() as conn:
        insert_session(conn, make_session(SID, privacy_state=PrivacyState.APPROVED))
        for row in rows:
            insert_review_task(conn, row)
    old = {"video": "https://s3.test/old?X-Amz-Expires=1", "session_id": SID,
           "assignee": "labeler01"}  # fmt: skip
    fake = FakeLabelStudio({
        n: {"data": dict(old)} for n in (1, 2, 3, 4, 8, 9)
    } | {5: {"data": {**old, "session_id": "other"}}})  # fmt: skip
    setup = _setup(tmp_path, fake.client(), signer)
    with engine.connect() as conn:
        assert sessions_with_open_ls_tasks(conn) == [SID]
    with engine.begin() as conn:
        report = refresh_session_urls(conn, SID, setup)
    assert report.refreshed == ["label_studio:1"]
    [skipped] = report.skipped
    assert skipped.startswith("label_studio:4:") and "dlp review revoke" in skipped
    failed = sorted(x.split(":")[1] for x in report.errors)
    assert failed == ["5", "6", "8", "9"]
    assert [tid for tid, _ in fake.patches] == [1]
    [(_, body)] = fake.patches
    assert body == {"data": label_studio_task_data(setup, SID, "labeler01", video,
                                                   timeseries_key(SID))}  # fmt: skip
    assert f"X-Amz-Expires={TTL}" in body["data"]["video"]
    assert "dlp-raw" not in json.dumps(body)
    assert fake.tasks[2]["data"] == old and fake.tasks[4]["data"] == old  # 수집·낡은 작업 그대로


def test_refresh_without_label_studio_client_is_an_error(engine: sa.Engine, tmp_path: Path) -> None:
    """Label Studio 클라이언트가 없으면 열린 작업마다 오류로 남기고 아무것도 바꾸지 않는다."""
    privacy = load_privacy_policy(repo_root())
    with engine.begin() as conn:
        insert_session(conn, make_session(SID, privacy_state=PrivacyState.APPROVED))
        insert_review_task(
            conn, _ls_task("label_studio:1", "local://dlp-labeling/x.mp4", _current_hash(privacy))
        )
    with engine.begin() as conn:
        report = refresh_session_urls(conn, SID, _setup(tmp_path, None, FakeSigner()))
    assert report.refreshed == [] and len(report.errors) == 1


@pytest.mark.parametrize(
    ("case", "hint"),
    [
        ("withdrawn", "--all --reason withdrawn"),
        ("unapproved", "승인"),
        ("held", "보류"),
        ("legacy", "--include-legacy"),
    ],
)
def test_refresh_skips_withdrawn_unapproved_held_or_hashless_tasks(
    engine: sa.Engine, tmp_path: Path, case: str, hint: str
) -> None:
    """사용 중지·승인 아님·승인 보류 세션의 작업과 렌더 해시 없는 구 작업은 다시 서명하지 않는다.

    정답 근거: 다시 서명하지 않으면 URL은 유효 시간이 지나 끊긴다 (노출이 줄어드는 쪽). 회수 요청
    없이 사용 중지된 세션(F-REV-17 이전)을 cron이 7일씩 다시 서명하면 동의를 철회한 영상이 계속
    열린다. 승인이 풀린 세션과 해시 없는 구 작업은 블러가 맞는지 모른다. 건너뛴 작업에는 회수
    명령을 안내한다.
    """
    signer = FakeSigner()
    privacy = load_privacy_policy(repo_root())
    labeling = LocalStore(tmp_path, "dlp-labeling")
    current = _current_hash(privacy)
    video = f"sessions/{SID}/review/labeler01/bodycam-{current[:16]}.mp4"
    media = tmp_path / "m.mp4"
    media.write_bytes(b"video")
    for key in (video, timeseries_key(SID)):
        labeling.put_file(key, media, "0" * 64)
    state = PrivacyState.AUTO_BLURRED if case == "unapproved" else PrivacyState.APPROVED
    row = _ls_task("label_studio:1", labeling.uri(video), current)
    if case == "legacy":
        row = row.model_copy(update={"render_hash": None})
    with engine.begin() as conn:
        insert_session(conn, make_session(SID, privacy_state=state))
        insert_review_task(conn, row)
        if case == "withdrawn":
            set_lifecycle(conn, SID, LifecycleState.WITHDRAWN, at=FIXED_TIME, actor="dpo01")
        if case == "held":
            # 보류는 회수도 요청하므로, 요청이 남지 않은 작업을 흉내 내려 요청만 되돌린다
            # (보류 이유가 승인 상태보다 먼저 안내된다)
            hold_session(
                conn, SID, privacy, FIXED_TIME, reason="얼굴 누락 의심", actor="dpo01",
                labeling=None,
            )  # fmt: skip
            conn.execute(
                sa.update(review_tasks).values(
                    revoke_requested_at=None, revoke_reason=None, status="open"
                )
            )
    old = {"video": "https://s3.test/old?X-Amz-Expires=1", "session_id": SID,
           "assignee": "labeler01"}  # fmt: skip
    fake = FakeLabelStudio({1: {"data": dict(old)}})
    with engine.begin() as conn:
        report = refresh_session_urls(conn, SID, _setup(tmp_path, fake.client(), signer))
    assert report.refreshed == [] and report.errors == []
    [skipped] = report.skipped
    assert skipped.startswith("label_studio:1:") and hint in skipped
    assert fake.patches == [] and signer.calls == []
