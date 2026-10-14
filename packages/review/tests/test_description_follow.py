"""설명 라벨이 행동 검수 결과를 따른다 (F-PAR-15, ADR 0043 3절 "행동 승인할 때 같이").

서비스 없이 시간대를 보존하는 SQLite(`dlp_schema.testing.sqlite_db`)와 가짜 Label Studio
(`httpx.MockTransport`)로 수집(`collect_task`)부터 돈다.

정답 근거 (손으로 만든 장면, 세션 s001, 모두 행동 단계가 낸 것 같은 모델 라벨)
- 행동 a1(a001 grasp 0~1 s), a2(a002 lift 2~3 s), a3(a003 push 왼손 0~1 s), a4(a004 grasp 왼손
  1~2 s), 상위 구간 s1(sk1 skill wipe, 자식 a001·a002).
- 설명: d1(a001 ko)과 그 영어 번역 d1en, d2(a002 ko), d3(a003 ko)과 d3en(검수자 lead가 미리 승인한
  번역), d4(a004 ko)와 d4en, dsk(sk1 ko), dx(이력에 없는 구간 nope을 가리킴).
- 검수자 행동 (Label Studio): a1 그대로, a2 시작만 2.0 → 2.05 s(시각만 고침), a3 동사 push → pull,
  a4 지움, s1 그대로.
- 정답: d1·d1en·dsk human_approved(같은 검수자, 수집 시각), d2 human_approved(시각만 고친 행동),
  d3 지움(동사가 바뀜), d3en 그대로 승인 상태(이미 검수됨 — 지우지 않는다), d4·d4en 지움(행동
  삭제), dx 미검수 그대로(주인을 모른다).
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
import sqlalchemy as sa

from dlp_media.storage import LocalStore
from dlp_review.clients import LabelStudioClient
from dlp_review.collect import collect_task
from dlp_review.labelstudio import to_ls_results
from dlp_review.ops.policy import LabelStudioPolicy
from dlp_review.tasks import ReviewSetup
from dlp_review.verify import verify_session
from dlp_schema import load_ontology, repo_root
from dlp_schema.db.description_repository import apply_description_follow
from dlp_schema.db.repository import (
    get_labels,
    insert_assignment,
    insert_labels,
    insert_review_task,
    insert_session,
    record_review,
    register_ontology,
    set_lifecycle,
)
from dlp_schema.description_follow import FOLLOW_VERSION, inherited_owner
from dlp_schema.episode import current_labels
from dlp_schema.labels import LabelRecord, Provenance, Source, VerificationState
from dlp_schema.review import (
    ReviewAssignment,
    ReviewMode,
    ReviewStage,
    ReviewTask,
    ReviewTool,
)
from dlp_schema.session import LifecycleState, PrivacyState
from dlp_schema.testing import FIXED_TIME, action_payload, make_label, make_session, sqlite_db

SID, TASK, ASSIGNEE = "s001", "label_studio:3", "labeler01"
LS_USER = 11
ACCOUNTS = LabelStudioPolicy(users={ASSIGNEE: "labeler01@example.com"})
# 수집 시각 (설명 생성 시각 FIXED_TIME보다 늦다: 물려받은 검수 시각 = 수집 시각)
LATER = FIXED_TIME + timedelta(hours=1)
MODEL: dict[str, Any] = {
    "provenance": Provenance(source=Source.MODEL, model_version="actions-1"),
    "confidence": 0.9,
}
# 번역 단계가 낸 것 같은 영어 번역의 출처
TRANSLATED: dict[str, Any] = {
    "provenance": Provenance(source=Source.MODEL, model_version="translate-x+stub-glossary-1+g0"),
    "confidence": 0.9,
}


def _owners() -> dict[str, LabelRecord]:
    """장면의 행동·상위 구간 (모듈 docstring)."""
    left = {"hand": "left"}
    return {
        "a1": make_label(action_payload(), label_id="s001-a1", **MODEL),
        "a2": make_label(action_payload(action_id="a002", verb="lift", t_approach_ms=2000,
                                        t_contact_start_ms=2100, t_contact_end_ms=2900,
                                        t_end_ms=3000),
                         label_id="s001-a2", t_start_ms=2000, t_end_ms=3000, **MODEL),
        "a3": make_label(action_payload(action_id="a003", verb="push", **left),
                         label_id="s001-a3", **MODEL),
        "a4": make_label(action_payload(action_id="a004", t_approach_ms=1000,
                                        t_contact_start_ms=1100, t_contact_end_ms=1900,
                                        t_end_ms=2000, **left),
                         label_id="s001-a4", t_start_ms=1000, t_end_ms=2000, **MODEL),
        "s1": make_label({"kind": "segment", "segment_id": "sk1", "level": "skill",
                          "ref_id": "wipe", "child_ids": ["a001", "a002"]},
                         label_id="s001-s1", t_start_ms=0, t_end_ms=3000, **MODEL),
    }  # fmt: skip


def _desc(
    label_id: str, segment: str, text: str, *, source: str | None = None, **extra: Any
) -> LabelRecord:
    """설명 라벨 (source가 있으면 영어 번역)."""
    payload: dict[str, Any] = {"kind": "description", "segment_id": segment, "text": text}
    if source is not None:
        payload |= {"language": "en", "source_label_id": source}
    data = {**(TRANSLATED if source else MODEL), **extra}
    return make_label(payload, label_id=label_id, **data)


def _descriptions() -> list[LabelRecord]:
    """장면의 설명 (모듈 docstring)."""
    return [
        _desc("d1", "a001", "걸레를 잡는다"),
        _desc("d1en", "a001", "Grasps the cloth", source="d1"),
        _desc("d2", "a002", "걸레를 들어 올린다"),
        _desc("d3", "a003", "걸레를 민다"),
        _desc("d3en", "a003", "Pushes the cloth", source="d3"),
        _desc("d4", "a004", "걸레를 잡는다"),
        _desc("d4en", "a004", "Grasps the cloth", source="d4"),
        _desc("dsk", "sk1", "바닥을 닦는다"),
        _desc("dx", "nope", "알 수 없는 구간"),
    ]


@pytest.fixture
def engine() -> Iterator[sa.Engine]:
    """표와 온톨로지 v1을 만든 메모리 SQLite."""
    eng = sqlite_db()
    with eng.begin() as conn:
        register_ontology(conn, load_ontology(repo_root() / "config/ontology/v1"))
    yield eng
    eng.dispose()


def _client(results: list[dict[str, Any]]) -> LabelStudioClient:
    """담당자(사용자 11)가 낸 주석 하나를 돌려주는 가짜 Label Studio (작업 3)."""
    users = [
        {"id": 1, "email": "admin@dlp.local"},
        {"id": LS_USER, "email": "labeler01@example.com"},
    ]
    ann = {"id": 1, "completed_by": LS_USER, "updated_by": LS_USER, "result": results,
           "lead_time": 10.0, "updated_at": "2026-10-07T09:10:00Z"}  # fmt: skip

    def handler(req: httpx.Request) -> httpx.Response:
        path = req.url.path
        if path == "/api/users/":
            return httpx.Response(200, json=users)
        if path == "/api/current-user/whoami":
            return httpx.Response(200, json={"id": 1})
        if path == "/api/tasks/3/annotations":
            return httpx.Response(200, json=[ann])
        return httpx.Response(404)

    client = LabelStudioClient("http://ls", "token")
    client.http = httpx.Client(base_url="http://ls", transport=httpx.MockTransport(handler))
    return client


def _setup(tmp_path: Path, results: list[dict[str, Any]]) -> ReviewSetup:
    """로컬 저장소 + 가짜 Label Studio + 담당자 계정 연결 (AI 끔)."""
    return ReviewSetup(
        raw=LocalStore(tmp_path, "dlp-raw"), labeling=LocalStore(tmp_path, "dlp-labeling"),
        labeling_reader=MagicMock(), ontology=None, label_studio=_client(results),
        label_studio_config=ACCOUNTS,
    )  # fmt: skip


def _task(sent: list[LabelRecord], assignment_id: str | None = None) -> ReviewTask:
    """행동·상위 구간 Label Studio 작업 (설명은 보내지 않는다: LS_KINDS에 없다)."""
    return ReviewTask(
        task_key=TASK, tool=ReviewTool.LABEL_STUDIO, external_id="3", session_id=SID,
        stream_id="bodycam", stage=ReviewStage.LABELING, assignee=ASSIGNEE,
        media_uri=f"s3://dlp-labeling/sessions/{SID}/review/{ASSIGNEE}/bodycam.mp4",
        label_kinds=("action", "segment"), created_at=FIXED_TIME, mode=ReviewMode.STANDARD,
        sent_label_ids=tuple(x.label_id for x in sent), assignment_id=assignment_id,
    )  # fmt: skip


def _scene(engine: sa.Engine) -> dict[str, LabelRecord]:
    """세션·행동·설명·작업을 넣고, d3en은 검수자 lead가 미리 승인해 둔다."""
    owners = _owners()
    with engine.begin() as conn:
        insert_session(conn, make_session(SID, privacy_state=PrivacyState.APPROVED))
        insert_labels(conn, [*owners.values(), *_descriptions()])
        record_review(conn, "d3en", VerificationState.HUMAN_APPROVED, "lead", FIXED_TIME)
        insert_review_task(conn, _task(list(owners.values())))
    return owners


def _reviewed(owners: dict[str, LabelRecord]) -> list[dict[str, Any]]:
    """검수자 결과 (모듈 docstring): a2 시작 2.05 s, a3 동사 pull, a4 지움, 나머지 그대로."""
    results = [r for r in to_ls_results(list(owners.values())) if r["id"] != "s001-a4"]
    for r in results:
        if r["id"] == "s001-a2":
            r["value"] = {**r["value"], "start": 2050}
        if r["id"] == "s001-a3":
            r["value"] = {**r["value"], "timeserieslabels": ["action.left:pull"]}
    return results


def _state(engine: sa.Engine) -> tuple[dict[str, LabelRecord], list[LabelRecord]]:
    """(현재 운영 라벨 ID → 레코드, 이력 전체)."""
    with engine.connect() as conn:
        history = get_labels(conn, SID)
    return {x.label_id: x for x in current_labels(history)}, history


def test_descriptions_follow_approve_time_fix_verb_change_and_delete(
    engine: sa.Engine, tmp_path: Path
) -> None:
    """수집 한 번으로 설명이 행동 결과를 따른다 (모듈 docstring 정답 표)."""
    owners = _scene(engine)
    with engine.begin() as conn:
        outcome = collect_task(conn, TASK, _setup(tmp_path, _reviewed(owners)), ASSIGNEE, LATER)
    assert outcome is not None
    # 행동 결과 자체는 이 규칙과 무관하게 같다 (승인 a1·s1, 수정 a2·a3, 삭제 a4)
    assert sorted(outcome.approved) == ["s001-a1", "s001-s1"]
    assert (outcome.corrected, outcome.retracted) == (2, 1)
    assert sorted(outcome.descriptions_approved) == ["d1", "d1en", "d2", "dsk"]
    assert sorted(outcome.descriptions_retracted) == ["d3", "d4", "d4en"]

    current, history = _state(engine)
    for lid in ("d1", "d1en", "d2", "dsk"):
        v = current[lid].verification
        assert (v.state, v.reviewer_id, v.reviewed_at) == (
            VerificationState.HUMAN_APPROVED, ASSIGNEE, LATER,
        ), lid  # fmt: skip
    # 동사가 바뀐 행동·지운 행동의 검수 전 설명은 지운다 (모델 출처 삭제 레코드, 규칙 버전)
    for lid in ("d3", "d4", "d4en"):
        assert lid not in current
        gone = next(x for x in history if x.parent_label_id == lid)
        assert gone.retracted and gone.provenance.model_version == FOLLOW_VERSION
        assert gone.created_at == LATER
    # 이미 검수된 번역은 건드리지 않는다
    assert current["d3en"].verification.reviewer_id == "lead"
    # 주인을 모르는 설명은 그대로 미검수
    assert current["dx"].verification.state is VerificationState.UNREVIEWED
    # 감사: 물려준 주인을 이력에서 다시 계산한다 (시각만 고친 a2는 새 사람 레코드가 물려준다)
    first = inherited_owner(current["d1en"], history)
    assert first is not None and first.label_id == "s001-a1"
    giver = inherited_owner(current["d2"], history)
    assert giver is not None and giver.parent_label_id == "s001-a2"
    assert inherited_owner(current["d3en"], history) is None  # 직접 검수 (lead)

    # 멱등: 다시 적용해도 바뀌는 것이 없다
    with engine.begin() as conn:
        again = apply_description_follow(conn, SID, LATER + timedelta(hours=1))
    assert (again.approve, again.retract) == ([], [])


def test_unreviewed_action_keeps_its_descriptions_blocking_verify(engine: sa.Engine) -> None:
    """행동이 검수 전이면 설명도 기다리고, verify는 계속 막는다. 행동을 승인하면 풀린다.

    정답: 행동 a1만 있는 세션. 검수 전에는 verify 이유에 미검수 모델 라벨 3개(a1·d1·d1en),
    a1을 승인한 뒤(이 규칙 전 검수를 흉내 내 record_review로 직접) verify가 설명을 메워
    human_verified로 옮긴다.
    """
    a1 = _owners()["a1"]
    with engine.begin() as conn:
        insert_session(conn, make_session(SID, privacy_state=PrivacyState.APPROVED))
        set_lifecycle(conn, SID, LifecycleState.PRIVACY_APPROVED)
        set_lifecycle(conn, SID, LifecycleState.PRELABELED)
        insert_labels(conn, [a1, *_descriptions()[:2]])
    with engine.begin() as conn:
        blocked = verify_session(conn, SID, LATER, "lead")
    assert not blocked.verified
    assert any("검수하지 않은 모델 라벨 3개" in r for r in blocked.reasons), blocked.reasons

    with engine.begin() as conn:
        record_review(conn, "s001-a1", VerificationState.HUMAN_APPROVED, ASSIGNEE, LATER)
        done = verify_session(conn, SID, LATER + timedelta(hours=1), "lead")
    assert done.verified, done.reasons
    assert done.descriptions_approved == 2
    current, _ = _state(engine)
    assert current["d1"].verification.reviewer_id == ASSIGNEE


def test_sample_verified_action_passes_sample_state_to_descriptions(
    engine: sa.Engine, tmp_path: Path
) -> None:
    """표본 합격으로 표본 검증된 행동의 설명도 sample_verified (검수자 `sampling:<배정>`).

    정답: 표본 a1(그대로 승인) + 보류 a3. 합격하면 a3과 그 설명 d3·d3en(미리 승인한 d3en은 그대로)
    중 d3이 sample_verified, 표본 a1의 설명 d1·d1en은 human_approved.
    """
    owners = _owners()
    a1, a3 = owners["a1"], owners["a3"]
    assignment = ReviewAssignment(
        assignment_id="asg1", session_id=SID, label_kinds=("action",),
        mode=ReviewMode.STANDARD, priority=1.0, assignee=ASSIGNEE, task_key=TASK,
        sample_label_ids=(a1.label_id,), withheld_label_ids=(a3.label_id,),
        created_at=FIXED_TIME,
    )  # fmt: skip
    with engine.begin() as conn:
        insert_session(conn, make_session(SID, privacy_state=PrivacyState.APPROVED))
        insert_labels(conn, [a1, a3, *_descriptions()[:2], *_descriptions()[3:5]])
        record_review(conn, "d3en", VerificationState.HUMAN_APPROVED, "lead", FIXED_TIME)
        insert_assignment(conn, assignment)
        insert_review_task(conn, _task([a1], "asg1"))
    with engine.begin() as conn:
        outcome = collect_task(conn, TASK, _setup(tmp_path, to_ls_results([a1])), ASSIGNEE, LATER)
    assert outcome is not None and outcome.approved == ["s001-a1"]
    current, _ = _state(engine)
    assert current["s001-a3"].verification.state is VerificationState.SAMPLE_VERIFIED
    d3 = current["d3"].verification
    assert (d3.state, d3.reviewer_id, d3.reviewed_at) == (
        VerificationState.SAMPLE_VERIFIED, "sampling:asg1", LATER,
    )  # fmt: skip
    assert current["d1en"].verification.state is VerificationState.HUMAN_APPROVED
    assert current["d3en"].verification.reviewer_id == "lead"
