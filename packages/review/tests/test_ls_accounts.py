"""Label Studio 결과는 작업 담당자의 Label Studio 계정이 낸 주석만 받는다 (F-REV-06, A-10).

서비스 없이 시간대를 보존하는 SQLite(`dlp_schema.testing.sqlite_db`)와 LocalStore로 돈다. Label
Studio는 실제 `LabelStudioClient`를 `httpx.MockTransport`(가짜 응답)에 붙여 쓴다. 응답 모양은
Label Studio 1.23.2가 실제로 돌려주는 것과 같다 (`completed_by`·`updated_by`는 사용자 숫자 ID,
`/api/users/`는 배열). 실제 도구 응답은 `test_flow.py`의 서비스 테스트가 본다.

정답 근거: 담당자 labeler01 = Label Studio 사용자 11 (`label_studio.users`의 이메일로 찾는다),
다른 검수자 = 12, 서비스 계정 = 1. 주석은 만든 사람(`completed_by`)과 마지막으로 고친 사람
(`updated_by`)이 모두 11이고 건너뛰지 않은 것만 담당자의 것이다. 서비스 계정이 `completed_by`를
11로 적어 올린 주석은 `updated_by`가 1이라 담당자의 것이 아니다 (실제 1.23.2 동작).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
import sqlalchemy as sa
from pydantic import ValidationError

from dlp_media.storage import LocalStore
from dlp_review.clients import IgnoredAnnotation, LabelStudioClient
from dlp_review.collect import collect_task
from dlp_review.labelstudio import to_ls_results
from dlp_review.ops.policy import LabelStudioPolicy, load_policy
from dlp_review.tasks import ReviewSetup, TaskError, create_labeling_tasks, label_studio_user_id
from dlp_schema import load_ontology, repo_root
from dlp_schema.db.repository import (
    get_labels,
    get_review_task,
    insert_labels,
    insert_review_task,
    insert_session,
    list_review_work,
    register_ontology,
)
from dlp_schema.labels import LabelRecord
from dlp_schema.review import ReviewStage, ReviewTask, ReviewTaskStatus, ReviewTool
from dlp_schema.session import PrivacyState
from dlp_schema.testing import FIXED_TIME, make_label, make_session, sqlite_db

SERVICE, ASSIGNEE, OTHER = 1, 11, 12
USERS = [
    {"id": SERVICE, "email": "admin@dlp.local"},
    {"id": ASSIGNEE, "email": "Labeler01@Example.com"},  # 대소문자가 달라도 같은 이메일
    {"id": OTHER, "email": "labeler02@example.com"},
]
ACCOUNTS = LabelStudioPolicy(
    users={"labeler01": "labeler01@example.com", "labeler02": "labeler02@example.com"}
)


def _ann(aid: int, by: Any, editor: Any, result: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
    """Label Studio 주석 응답 하나 (by: completed_by, editor: updated_by)."""
    return {"id": aid, "completed_by": by, "updated_by": editor, "result": result, **kw}


def _client(annotations: list[dict[str, Any]]) -> LabelStudioClient:
    """가짜 응답을 쓰는 Label Studio 클라이언트 (작업 3의 주석, 사용자 목록, 서비스 계정 1)."""

    def handler(req: httpx.Request) -> httpx.Response:
        path = req.url.path
        if path == "/api/users/":
            return httpx.Response(200, json=USERS)
        if path == "/api/current-user/whoami":
            return httpx.Response(200, json={"id": SERVICE})
        if path == "/api/tasks/3/annotations":
            return httpx.Response(200, json=annotations)
        return httpx.Response(404)

    client = LabelStudioClient("http://ls", "token")
    client.http = httpx.Client(base_url="http://ls", transport=httpx.MockTransport(handler))
    return client


def test_client_keeps_only_annotations_made_and_last_edited_by_the_account() -> None:
    """최신 결과·작업 시간은 그 사용자가 만들고 마지막으로 고친, 건너뛰지 않은 주석에서만 온다.

    회귀(F-REV-06): 예전 `latest_results`는 작성자를 보지 않고 가장 최근 주석을 골랐다.
    회귀(A-10): 예전 `lead_seconds`는 건너뛴 주석·다른 사용자·서비스 계정의 시간도 더했다.
    """
    anns = [
        _ann(1, ASSIGNEE, ASSIGNEE, [{"id": "own-old"}], lead_time=5.0, updated_at="2026-10-01"),
        _ann(2, OTHER, OTHER, [{"id": "other"}], lead_time=7.0, updated_at="2026-10-03"),
        _ann(3, ASSIGNEE, SERVICE, [{"id": "forged"}], lead_time=1.0, updated_at="2026-10-04"),
        _ann(4, {"id": ASSIGNEE}, ASSIGNEE, [], lead_time=2.0, was_cancelled=True,
             updated_at="2026-10-05"),
        _ann(5, {"id": ASSIGNEE}, ASSIGNEE, [{"id": "own-new"}], lead_time=3.0,
             updated_at="2026-10-02"),
        _ann(6, ASSIGNEE, OTHER, [{"id": "edited"}], lead_time=4.0, updated_at="2026-10-06"),
    ]  # fmt: skip
    ls = _client(anns)
    assert ls.latest_results(3, user=ASSIGNEE) == [{"id": "own-new"}]
    assert ls.lead_seconds(3, user=ASSIGNEE) == 8.0  # 5 + 3 (건너뜀·남의 수정 제외)
    assert ls.latest_results(3, user=99) is None and ls.lead_seconds(3, user=99) == 0.0
    # 받지 않은 주석: 건너뛰지 않았는데 그 사용자 것이 아닌 주석 (기록·오류 메시지용)
    assert ls.ignored_annotations(3, user=ASSIGNEE) == [
        IgnoredAnnotation(2, OTHER, OTHER),
        IgnoredAnnotation(3, ASSIGNEE, SERVICE),
        IgnoredAnnotation(6, ASSIGNEE, OTHER),
    ]


def test_find_user_id_by_email_ignores_case() -> None:
    """Label Studio 사용자 이메일 → 숫자 ID (대소문자 무시). 없으면 None."""
    ls = _client([])
    assert ls.find_user_id("labeler01@example.com") == ASSIGNEE
    assert ls.find_user_id("nobody@example.com") is None


def test_label_studio_user_id_requires_existing_non_service_account() -> None:
    """연결이 없으면 None, 연결한 계정이 없거나 서비스 계정이면 `TaskError`."""
    ls = _client([])
    assert label_studio_user_id(ls, ACCOUNTS.users, "labeler01") == ASSIGNEE
    assert label_studio_user_id(ls, ACCOUNTS.users, "labeler03") is None
    with pytest.raises(TaskError, match="없습니다"):
        label_studio_user_id(ls, {"labeler03": "ghost@example.com"}, "labeler03")
    with pytest.raises(TaskError, match="서비스 계정"):
        label_studio_user_id(ls, {"labeler03": "admin@dlp.local"}, "labeler03")


def test_policy_maps_one_account_per_reviewer() -> None:
    """한 Label Studio 계정을 두 검수자에게 줄 수 없다 (이메일 대소문자 무시). 기본값은 빈 연결."""
    with pytest.raises(ValidationError, match="두 검수자"):
        LabelStudioPolicy(users={"a": "x@example.com", "b": "X@example.com"})
    with pytest.raises(ValidationError, match="이메일"):
        LabelStudioPolicy(users={"a": "not-an-email"})
    assert load_policy(repo_root()).label_studio.users == {}


# ── 수집 ──

SID = "s001"


@pytest.fixture
def engine() -> Iterator[sa.Engine]:
    """표와 온톨로지 v1을 만든 메모리 SQLite."""
    eng = sqlite_db()
    with eng.begin() as conn:
        register_ontology(conn, load_ontology(repo_root() / "config/ontology/v1"))
    yield eng
    eng.dispose()


def _origin() -> LabelRecord:
    """작업에 보낸 시간 라벨 하나 (담당자가 그대로 두면 승인, 지우면 삭제)."""
    return make_label({"kind": "gap", "hand": "right", "gap_type": "unknown"}, label_id="gap-1")


def _labeling_task(engine: sa.Engine, origin: LabelRecord) -> None:
    """승인된 세션 s001과 담당자 labeler01의 Label Studio 작업(label_studio:3)."""
    task = ReviewTask(
        task_key="label_studio:3", tool=ReviewTool.LABEL_STUDIO, external_id="3", session_id=SID,
        stream_id="bodycam", stage=ReviewStage.LABELING, assignee="labeler01",
        media_uri=f"s3://dlp-labeling/sessions/{SID}/review/labeler01/bodycam.mp4",
        label_kinds=("gap",), created_at=FIXED_TIME, sent_label_ids=(origin.label_id,),
    )  # fmt: skip
    with engine.begin() as conn:
        insert_session(conn, make_session(SID, privacy_state=PrivacyState.APPROVED))
        insert_labels(conn, [origin])
        insert_review_task(conn, task)


def _setup(tmp_path: Path, ls: LabelStudioClient, accounts: LabelStudioPolicy) -> ReviewSetup:
    """로컬 저장소와 가짜 Label Studio, 시험 계정 연결로 만든 설정."""
    return ReviewSetup(
        raw=LocalStore(tmp_path, "dlp-raw"), labeling=LocalStore(tmp_path, "dlp-labeling"),
        labeling_reader=MagicMock(), ontology=None, label_studio=ls,
        label_studio_config=accounts,
    )  # fmt: skip


def test_collect_adopts_only_the_assignee_annotation(engine: sa.Engine, tmp_path: Path) -> None:
    """다른 사용자가 더 늦게 낸 주석(라벨 삭제)과 서비스 계정이 담당자 이름으로 올린 주석은 받지
    않고, 담당자 주석(그대로 승인)만 이력에 쓴다. 검수 시간도 담당자 주석의 lead_time만이다.

    회귀(F-REV-06): 예전에는 가장 최근 주석(다른 사용자의 삭제)을 담당자의 검수로 기록했다.
    """
    origin = _origin()
    _labeling_task(engine, origin)
    kept = to_ls_results([origin])
    anns = [
        _ann(1, ASSIGNEE, ASSIGNEE, kept, lead_time=30.0, updated_at="2026-10-01T00:00:00Z"),
        _ann(2, OTHER, OTHER, [], lead_time=90.0, updated_at="2026-10-02T00:00:00Z"),
        _ann(3, ASSIGNEE, SERVICE, [], lead_time=5.0, updated_at="2026-10-03T00:00:00Z"),
    ]
    setup = _setup(tmp_path, _client(anns), ACCOUNTS)
    with engine.begin() as conn:
        outcome = collect_task(conn, "label_studio:3", setup, "labeler01", FIXED_TIME)
    assert outcome is not None
    assert outcome.approved == [origin.label_id] and outcome.retracted == 0
    with engine.connect() as conn:
        assert [x.label_id for x in get_labels(conn, SID)] == [origin.label_id]
        assert get_review_task(conn, "label_studio:3").status is ReviewTaskStatus.COLLECTED
        [work] = list_review_work(conn)
    assert (work.reviewer, work.seconds) == ("labeler01", 30.0)


@pytest.mark.parametrize(
    ("accounts", "anns", "match"),
    [
        # 연결 없음: 누가 냈는지 확인할 수 없다
        (LabelStudioPolicy(), [_ann(1, ASSIGNEE, ASSIGNEE, [])], "Label Studio 계정 연결"),
        # 다른 사용자의 주석만 있다: 담당자 주석이 없다 (받지 않은 주석 수를 알린다)
        (ACCOUNTS, [_ann(2, OTHER, OTHER, [])], "다른 사용자 주석 1개"),
        # 연결한 계정이 서비스 계정이다
        (LabelStudioPolicy(users={"labeler01": "admin@dlp.local"}), [], "서비스 계정"),
    ],
)
def test_collect_refuses_without_assignee_annotation(
    engine: sa.Engine,
    tmp_path: Path,
    accounts: LabelStudioPolicy,
    anns: list[dict[str, Any]],
    match: str,
) -> None:
    """담당자 계정의 주석을 확인할 수 없으면 아무것도 쓰지 않고 `TaskError` (작업은 열린 채)."""
    origin = _origin()
    _labeling_task(engine, origin)
    setup = _setup(tmp_path, _client(anns), accounts)
    with engine.begin() as conn, pytest.raises(TaskError, match=match):
        collect_task(conn, "label_studio:3", setup, "labeler01", FIXED_TIME)
    with engine.connect() as conn:
        assert get_review_task(conn, "label_studio:3").status is ReviewTaskStatus.OPEN
        assert list_review_work(conn) == []


def test_label_studio_task_needs_assignee_account_before_any_upload(
    engine: sa.Engine, tmp_path: Path
) -> None:
    """작업 라벨 시간 검수(Label Studio)는 담당자의 Label Studio 계정 연결이 없으면 매체를 올리기
    전에 거부한다. 연결 없이 만든 작업은 누구의 제출인지 확인할 수 없어 수집할 수 없다."""
    with engine.begin() as conn:
        insert_session(conn, make_session(SID, privacy_state=PrivacyState.APPROVED))
    setup = _setup(tmp_path, _client([]), LabelStudioPolicy())
    with engine.begin() as conn, pytest.raises(TaskError, match="Label Studio 계정 연결"):
        create_labeling_tasks(
            conn, SID, setup, "labeler01", FIXED_TIME, tools={ReviewTool.LABEL_STUDIO}
        )
    assert not (tmp_path / "dlp-labeling").exists()  # 아무것도 올리지 않았다
