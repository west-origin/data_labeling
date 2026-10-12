"""AI 1차 검수 제안의 사람 적용과 수집 결과 기록 (`ai_collect`, WP17.7, ADR 0040 6.6).

서비스 없이 시간대를 보존하는 SQLite(`dlp_schema.testing.sqlite_db`)와 가짜 Label Studio
(`httpx.MockTransport`, 응답 모양은 1.23.2와 같다)로 돈다. 실제 도구 종단은
`test_ai_display_flow.py`(services)가 본다.

정답 근거 (손으로 만든 장면, 세션 s001)
- 오른손: a1 grasp 0~1 s(a001) → g1 대기 1~2 s → a2 lift 2~3 s(a002). 상위 구간 s1(skill wipe)이
  a001·a002를 자식으로 묶는다. 이벤트 e1 slip 3~4 s. 왼손: a3 push 0~1 s(a003), a4 grasp 1~2 s.
- 판정(모두 노출 기록 shown, a4의 ok는 hidden):
  a1 틀림 + 동사 press 제안(set), g1 의심 + push 행동으로 종류 변경(새 action_id `h-right-1000`),
  a2 틀림 + 대기 사이 구간으로 종류 변경(적용하면 s1의 자식 a002가 사라져 세션 검증 실패),
  e1 틀림 + 삭제 제안, a3 틀림 + 동사 pull 제안, a4 맞음.
- 적용 표(모듈 docstring)의 행마다 결과 레코드와 `AiOutcomeItem`이 정해진다.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
import sqlalchemy as sa

from dlp_media.storage import LocalStore
from dlp_review.ai_display import AiDisplayConfig, flag_region_id
from dlp_review.clients import LabelStudioClient
from dlp_review.collect import collect_task
from dlp_review.labelstudio import AI_DECISION, AI_FLAGS, FROM_NAME, TO_NAME, to_ls_results
from dlp_review.ops.policy import AiReviewDisplayPolicy, LabelStudioPolicy
from dlp_review.reconcile import ReviewedItem, same_as_sent
from dlp_review.tasks import ReviewSetup
from dlp_schema import load_ontology, repo_root
from dlp_schema.aireview import (
    AiApplyResult,
    AiArm,
    AiDecision,
    AiExposure,
    AiHumanAction,
    AiOutcome,
    AiOutcomeItem,
    AiSuggestion,
    AiVerdict,
)
from dlp_schema.db.aireview_repository import (
    get_ai_outcome,
    insert_ai_exposure,
    insert_ai_run,
    insert_ai_units,
)
from dlp_schema.db.repository import (
    get_labels,
    get_review_task,
    insert_labels,
    insert_review_task,
    insert_session,
    register_ontology,
)
from dlp_schema.episode import current_labels
from dlp_schema.labels import (
    ActionPayload,
    GapPayload,
    Hand,
    LabelRecord,
    Provenance,
    Source,
    VerificationState,
)
from dlp_schema.review import ReviewMode, ReviewStage, ReviewTask, ReviewTaskStatus, ReviewTool
from dlp_schema.session import PrivacyState
from dlp_schema.testing import (
    AI_TARGET_RULE,
    AI_VERSION,
    FIXED_TIME,
    action_payload,
    make_ai_run,
    make_ai_unit,
    make_ai_verdict,
    make_label,
    make_session,
    sqlite_db,
)
from dlp_schema.validation import OntologyViolationError

SID, TASK, ASSIGNEE = "s001", "label_studio:3", "labeler01"
LS_USER = 11
ACCOUNTS = LabelStudioPolicy(users={ASSIGNEE: "labeler01@example.com"})
RUN = "air-run-" + "1" * 24
UNIT = "air-" + "2" * 24
MODEL: dict[str, Any] = {
    "provenance": Provenance(source=Source.MODEL, model_version="actions-1"),
    "confidence": 0.9,
}


def _labels(stream: str | None = None) -> dict[str, LabelRecord]:
    """장면의 검수 전 모델 라벨 (모듈 docstring). stream: 행동·사이 구간의 스트림 (실제 행동 단계는
    기준 스트림을 쓴다)."""
    right = {"hand": "right"}
    left = {"hand": "left"}
    m: dict[str, Any] = {**MODEL, "stream_id": stream}
    return {
        "a1": make_label(action_payload(), label_id="s001-a1", **m),
        "g1": make_label({"kind": "gap", "gap_type": "idle", **right}, label_id="s001-g1",
                         t_start_ms=1000, t_end_ms=2000, **m),
        "a2": make_label(action_payload(action_id="a002", verb="lift", t_approach_ms=2000,
                                        t_contact_start_ms=2100, t_contact_end_ms=2900,
                                        t_end_ms=3000),
                         label_id="s001-a2", t_start_ms=2000, t_end_ms=3000, **m),
        "e1": make_label({"kind": "event", "event_type": "slip", "severity": 2},
                         label_id="s001-e1", t_start_ms=3000, t_end_ms=4000, **MODEL),
        "s1": make_label({"kind": "segment", "segment_id": "sk1", "level": "skill",
                          "ref_id": "wipe", "child_ids": ["a001", "a002"]},
                         label_id="s001-s1", t_start_ms=0, t_end_ms=3000, **MODEL),
        "a3": make_label(action_payload(action_id="a003", verb="push", **left),
                         label_id="s001-a3", **m),
        "a4": make_label(action_payload(action_id="a004", t_approach_ms=1000,
                                        t_contact_start_ms=1100, t_contact_end_ms=1900,
                                        t_end_ms=2000, **left),
                         label_id="s001-a4", t_start_ms=1000, t_end_ms=2000, **m),
    }  # fmt: skip


IDLE = GapPayload(hand=Hand.RIGHT, gap_type="idle")
PUSH = ActionPayload(
    action_id="h-right-1000", hand=Hand.RIGHT, verb="push", t_approach_ms=1000, t_end_ms=2000
)  # fmt: skip


def _verdicts(labels: dict[str, LabelRecord]) -> list[AiVerdict]:
    """장면의 판정 (모듈 docstring). 대상 순서 = 이 목록 순서."""
    a1, a3 = labels["a1"].payload, labels["a3"].payload
    return [
        make_ai_verdict(labels["a1"].label_id, "wrong", evidence_t_ms=500, suggestion=AiSuggestion(
            op="set", fields=("verb",), payload=a1.model_copy(update={"verb": "press"}))),
        make_ai_verdict(labels["g1"].label_id, "suspect", kind="gap",
                        problems=("should_be_action",), suggestion=AiSuggestion(
                            op="change_kind", fields=("kind", "verb"), payload=PUSH)),
        make_ai_verdict(labels["a2"].label_id, "wrong", problems=("should_be_gap",),
                        suggestion=AiSuggestion(op="change_kind", fields=("gap_type", "kind"),
                                                payload=IDLE)),
        make_ai_verdict(labels["e1"].label_id, "wrong", kind="event", problems=("false_event",),
                        suggestion=AiSuggestion(op="delete", fields=("delete",))),
        make_ai_verdict(labels["a3"].label_id, "wrong", suggestion=AiSuggestion(
            op="set", fields=("verb",), payload=a3.model_copy(update={"verb": "pull"}))),
        make_ai_verdict(labels["a4"].label_id, "ok"),
    ]  # fmt: skip


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


def _ai(**policy: Any) -> AiDisplayConfig:
    """AI 켬 (advisory, 허용 버전 `AI_VERSION`)."""
    data: dict[str, Any] = {"mode": "advisory"} | policy
    policy_ = AiReviewDisplayPolicy.model_validate(data)
    return AiDisplayConfig(policy_, (AI_VERSION,), AI_TARGET_RULE)


def _setup(
    tmp_path: Path, results: list[dict[str, Any]], ai: AiDisplayConfig | None
) -> ReviewSetup:
    """로컬 저장소 + 가짜 Label Studio + 담당자 계정 연결."""
    return ReviewSetup(
        raw=LocalStore(tmp_path, "dlp-raw"), labeling=LocalStore(tmp_path, "dlp-labeling"),
        labeling_reader=MagicMock(), ontology=None, label_studio=_client(results),
        label_studio_config=ACCOUNTS, ai=ai,
    )  # fmt: skip


def _scene(
    engine: sa.Engine,
    *,
    arm: AiArm = AiArm.SHOWN,
    mode: ReviewMode = ReviewMode.STANDARD,
    verdicts: list[AiVerdict] | None = None,
    exposure: bool = True,
    stream: str | None = None,
) -> dict[str, LabelRecord]:
    """세션·라벨·작업(label_studio:3)·판정 단위·노출 기록을 넣는다. 라벨(이름 → 레코드)을 돌려준다.

    노출: arm이 shown이면 맞음(a4)만 hidden, 그 밖이면 전부 hidden. stream: `_labels`와 같다.
    """
    labels = _labels(stream)
    sent = list(labels.values())
    judged = verdicts if verdicts is not None else _verdicts(labels)
    task = ReviewTask(
        task_key=TASK, tool=ReviewTool.LABEL_STUDIO, external_id="3", session_id=SID,
        stream_id="bodycam", stage=ReviewStage.LABELING, assignee=ASSIGNEE,
        media_uri=f"s3://dlp-labeling/sessions/{SID}/review/{ASSIGNEE}/bodycam.mp4",
        label_kinds=("action", "gap", "event", "segment"), created_at=FIXED_TIME, mode=mode,
        sent_label_ids=tuple(x.label_id for x in sent),
    )  # fmt: skip
    flagged = {v.label_id for v in judged if v.verdict.value != "ok"}
    shown = {lid: UNIT for lid in flagged} if arm is AiArm.SHOWN else {}
    hidden = {v.label_id: UNIT for v in judged if v.label_id not in shown}
    with engine.begin() as conn:
        insert_session(conn, make_session(SID, privacy_state=PrivacyState.APPROVED))
        insert_labels(conn, sent)
        insert_review_task(conn, task)
        insert_ai_run(conn, make_ai_run(SID, RUN))
        insert_ai_units(conn, [make_ai_unit(UNIT, judged, run_id=RUN, session_id=SID)])
        if exposure:
            insert_ai_exposure(conn, AiExposure(
                task_key=TASK, session_id=SID, arm=arm, arm_key=TASK, versions=(AI_VERSION,),
                coverage=1.0, shown=shown, hidden=hidden, created_at=FIXED_TIME,
            ))  # fmt: skip
    return labels


def _decide(region_id: str, alias: str) -> dict[str, Any]:
    """검수자가 구간에 고른 AI 결정 결과 하나 (Label Studio 결과 형식)."""
    return {"id": region_id, "from_name": AI_DECISION, "to_name": TO_NAME, "type": "choices",
            "value": {"choices": [alias]}}  # fmt: skip


def _submitted(labels: dict[str, LabelRecord], *extra: dict[str, Any]) -> list[dict[str, Any]]:
    """검수자가 예측(프리라벨 결과)을 그대로 내고 extra 결과를 더한 주석."""
    return [*to_ls_results(list(labels.values())), *extra]


def _relabel(results: list[dict[str, Any]], region_id: str, name: str) -> None:
    """검수자가 Label Studio에서 구간의 라벨 이름을 바꿨다 (제자리에서 고친다)."""
    for r in results:
        if r["id"] == region_id and r.get("type") == "timeserieslabels":
            r["value"] = {**r["value"], "timeserieslabels": [name]}


def _drawn(start: int, end: int, name: str) -> dict[str, Any]:
    """검수자가 라벨 컨트롤로 새로 그린 구간 (원 라벨 없음)."""
    value = {"start": start, "end": end, "instant": False, "timeserieslabels": [name]}
    return {"id": f"new-{start}", "from_name": FROM_NAME, "to_name": TO_NAME,
            "type": "timeserieslabels", "value": value}  # fmt: skip


def _collect(engine: sa.Engine, setup: ReviewSetup) -> tuple[Any, AiOutcome | None]:
    """수집하고 (수집 결과, AI 결과 기록)을 돌려준다."""
    with engine.begin() as conn:
        outcome = collect_task(conn, TASK, setup, ASSIGNEE, FIXED_TIME)
    with engine.connect() as conn:
        return outcome, get_ai_outcome(conn, TASK)


def _items(record: AiOutcome | None) -> dict[str, AiOutcomeItem]:
    """결과 기록의 항목 (라벨 ID → 항목)."""
    assert record is not None
    return {x.label_id: x for x in record.items}


def test_apply_table_rows(engine: sa.Engine, tmp_path: Path) -> None:
    """적용 표의 행: set 적용·종류 변경 적용·삭제 적용·세션 검증 실패 되돌림·적용+직접 수정·결정
    없음. AI 구간에 붙은 결정은 원 라벨로, 검수자가 AI 라벨로 그린 구간은 경고만."""
    labels = _scene(engine)
    a1, g1, a2, e1, a3, a4, s1 = (labels[k] for k in ("a1", "g1", "a2", "e1", "a3", "a4", "s1"))
    results = _submitted(labels)
    # a3: 적용을 고르고 구간도 0~800 ms로 줄였다 (검수자 결과가 우선)
    for r in results:
        if r["id"] == a3.label_id:
            r["value"] = {**r["value"], "end": 800}
    drawn = {
        "id": "mine", "from_name": AI_FLAGS, "to_name": TO_NAME, "type": "timeserieslabels",
        "value": {"start": 0, "end": 10, "instant": False, "timeserieslabels": ["AI:틀림"]},
    }  # fmt: skip
    results += [
        _decide(a1.label_id, "apply"), _decide(g1.label_id, "apply"),
        _decide(a2.label_id, "제안 적용"),  # 화면 값도 받는다
        _decide(flag_region_id(UNIT, e1.label_id), "apply"),  # AI 구간을 고른 채 누른 결정
        _decide(a3.label_id, "apply"), drawn,
    ]  # fmt: skip
    outcome, record = _collect(engine, _setup(tmp_path, results, _ai()))
    items = _items(record)

    # a1: set 적용 → 원 라벨의 사람 수정 레코드 (제안 페이로드 그대로)
    [fix] = [r for r in outcome.new_records if r.parent_label_id == a1.label_id]
    assert isinstance(fix.payload, ActionPayload) and fix.payload.verb == "press"
    assert fix.provenance.source is Source.HUMAN and fix.verification.reviewer_id == ASSIGNEE
    assert fix.verification.state is VerificationState.HUMAN_CORRECTED and not fix.retracted
    assert items[a1.label_id].model_dump(exclude={"unit_id", "verdict", "arm"}) == {
        "label_id": a1.label_id, "human": AiHumanAction.CORRECTED, "decision": AiDecision.APPLY,
        "apply_result": AiApplyResult.APPLIED, "match": "exact",
        "result_label_ids": (fix.label_id,),
    }  # fmt: skip
    # g1: 종류 변경 → 원 라벨 삭제 + 원 라벨 없는 새 행동 (h-<손>-<시작>)
    gone = next(r for r in outcome.new_records if r.parent_label_id == g1.label_id)
    new = next(r for r in outcome.new_records if r.parent_label_id is None)
    assert gone.retracted and new.payload == PUSH and new.provenance.source is Source.HUMAN
    g = items[g1.label_id]
    assert (g.human, g.apply_result, g.match) == (
        AiHumanAction.REPLACED, AiApplyResult.APPLIED, "exact",
    )  # fmt: skip
    assert set(g.result_label_ids) == {gone.label_id, new.label_id}
    # a2: 적용하면 s1의 자식 a002가 사라진다 → AI 적용만 되돌리고 수집은 계속 (승인)
    assert a2.label_id in outcome.approved and s1.label_id in outcome.approved
    assert (items[a2.label_id].apply_result, items[a2.label_id].human) == (
        AiApplyResult.INVALID_NOW, AiHumanAction.APPROVED,
    )  # fmt: skip
    # e1: AI 구간 id에 붙은 결정이 원 라벨로 → 삭제 적용
    [drop] = [r for r in outcome.new_records if r.parent_label_id == e1.label_id]
    assert drop.retracted and drop.provenance.source is Source.HUMAN
    e = items[e1.label_id]
    assert (e.human, e.apply_result, e.match) == (
        AiHumanAction.DELETED, AiApplyResult.APPLIED, "exact",
    )  # fmt: skip
    # a3: 적용했지만 구간도 고쳤다 → 검수자 결과(동사 그대로, 끝 800 ms)
    [edit] = [r for r in outcome.new_records if r.parent_label_id == a3.label_id]
    assert isinstance(edit.payload, ActionPayload) and edit.payload.verb == "push"
    assert edit.t_end_ms == 800
    t = items[a3.label_id]
    assert (t.human, t.apply_result, t.match) == (
        AiHumanAction.CORRECTED, AiApplyResult.CONFLICT_EDITED, "none",
    )  # fmt: skip
    # a4: 아무것도 하지 않음 (맞음 판정은 hidden이라 결과 기록에만)
    keys = {"human", "decision", "apply_result", "match"}
    assert items[a4.label_id].model_dump(include=keys) == {
        "human": AiHumanAction.APPROVED, "decision": AiDecision.NONE, "apply_result": None,
        "match": "na",
    }  # fmt: skip
    assert items[s1.label_id].unit_id is None and items[s1.label_id].verdict is None
    assert {x.arm for x in items.values()} == {AiArm.SHOWN}
    assert record is not None and outcome.ai == record
    assert any("세션 검증" in w for w in record.warnings)
    assert any("그린 구간 1개" in w for w in record.warnings)
    # 저장된 이력도 같다: 적용 레코드는 사람 수정 레코드와 같은 형식, AI 표시는 라벨 계약에 없다
    with engine.connect() as conn:
        current = {x.label_id: x for x in current_labels(get_labels(conn, SID))}
        assert get_review_task(conn, TASK).status is ReviewTaskStatus.COLLECTED
    assert fix.label_id in current and new.label_id in current and a2.label_id in current
    assert e1.label_id not in current and g1.label_id not in current


def test_dismiss_and_no_decision_leave_results_as_without_ai(
    engine: sa.Engine, tmp_path: Path
) -> None:
    """기각과 결정 없음은 도구 결과 그대로다 (모두 승인). AI가 꺼진 수집과 결과 레코드가 같다."""
    labels = _scene(engine)
    a1 = labels["a1"]
    results = _submitted(labels, _decide(a1.label_id, "dismiss"))
    outcome, record = _collect(engine, _setup(tmp_path, results, _ai()))
    assert not outcome.new_records and sorted(outcome.approved) == sorted(
        x.label_id for x in labels.values()
    )
    items = _items(record)
    assert (items[a1.label_id].decision, items[a1.label_id].apply_result) == (
        AiDecision.DISMISS, None,
    )  # fmt: skip
    # AI 판정이 wrong인데 승인: 불일치로 집계될 원자료 (match none)
    assert (items[a1.label_id].human, items[a1.label_id].match) == (AiHumanAction.APPROVED, "none")


def test_ai_off_reads_no_ai_tables_and_ignores_decisions(engine: sa.Engine, tmp_path: Path) -> None:
    """AI 끔(`ReviewSetup.ai` None)이면 결정을 반영하지 않고 결과 기록도 쓰지 않는다."""
    labels = _scene(engine)
    results = _submitted(labels, _decide(labels["a1"].label_id, "apply"))
    outcome, record = _collect(engine, _setup(tmp_path, results, None))
    assert record is None and outcome.ai is None and not outcome.new_records


def test_task_without_exposure_has_no_ai_outcome(engine: sa.Engine, tmp_path: Path) -> None:
    """노출 기록이 없는 작업(AI 단계 전에 만든 작업)은 AI가 켜져 있어도 예전과 같다."""
    labels = _scene(engine, exposure=False)
    results = _submitted(labels, _decide(labels["a1"].label_id, "apply"))
    outcome, record = _collect(engine, _setup(tmp_path, results, _ai()))
    assert record is None and not outcome.new_records


@pytest.mark.parametrize("arm", [AiArm.HOLDOUT, AiArm.NO_VERDICTS, AiArm.MODE_EXCLUDED])
def test_decisions_on_tasks_that_showed_nothing_are_not_shown(
    arm: AiArm, engine: sa.Engine, tmp_path: Path
) -> None:
    """집단이 shown이 아니면 결정을 무시한다 (not_shown, 대조군 지표의 원자료)."""
    labels = _scene(engine, arm=arm)
    a1 = labels["a1"]
    results = _submitted(labels, _decide(a1.label_id, "apply"))
    outcome, record = _collect(engine, _setup(tmp_path, results, _ai()))
    assert not outcome.new_records
    item = _items(record)[a1.label_id]
    assert (item.arm, item.apply_result, item.human) == (
        arm, AiApplyResult.NOT_SHOWN, AiHumanAction.APPROVED,
    )  # fmt: skip


def test_apply_without_suggestion_and_hidden_verdict(engine: sa.Engine, tmp_path: Path) -> None:
    """보인 판정에 제안이 없으면 no_suggestion, 화면에 따로 보이지 않은 판정(맞음)의 결정은
    not_shown. 둘 다 결과는 도구 결과 그대로다."""
    labels = _labels()
    verdicts = [
        make_ai_verdict(labels["a1"].label_id, "suspect"),
        make_ai_verdict(labels["a4"].label_id, "ok"),
    ]
    _scene(engine, verdicts=verdicts)
    results = _submitted(
        labels, _decide(labels["a1"].label_id, "apply"), _decide(labels["a4"].label_id, "apply")
    )
    outcome, record = _collect(engine, _setup(tmp_path, results, _ai()))
    items = _items(record)
    assert not outcome.new_records
    assert items[labels["a1"].label_id].apply_result is AiApplyResult.NO_SUGGESTION
    assert items[labels["a4"].label_id].apply_result is AiApplyResult.NOT_SHOWN
    assert items[labels["a1"].label_id].match == "na"


def test_suggestion_is_rechecked_from_db_now(engine: sa.Engine, tmp_path: Path) -> None:
    """제안 내용은 DB의 판정 단위에서 다시 읽어 지금 온톨로지로 검사한다 (I6). 원시 동작이 아닌
    동사(wipe는 skill) 제안은 적용하지 않는다 (invalid_now, 수집은 승인으로 계속)."""
    labels = _labels()
    bad = labels["a1"].payload.model_copy(update={"verb": "wipe"})
    verdicts = [
        make_ai_verdict(labels["a1"].label_id, "wrong", suggestion=AiSuggestion(
            op="set", fields=("verb",), payload=bad)),
    ]  # fmt: skip
    _scene(engine, verdicts=verdicts)
    results = _submitted(labels, _decide(labels["a1"].label_id, "apply"))
    outcome, record = _collect(engine, _setup(tmp_path, results, _ai()))
    assert not outcome.new_records and labels["a1"].label_id in outcome.approved
    assert _items(record)[labels["a1"].label_id].apply_result is AiApplyResult.INVALID_NOW
    assert record is not None and any("지금 검사" in w for w in record.warnings)


@pytest.mark.parametrize(
    ("name", "retracted"),
    [("a1", False), ("g1", False), ("g1", True), ("e1", False)],
    ids=["set-corrected", "change_kind-corrected", "change_kind-retracted", "delete-corrected"],
)
def test_stale_label_apply_is_invalid_now(
    name: str, retracted: bool, engine: sa.Engine, tmp_path: Path
) -> None:
    """작업을 보낸 뒤 다른 단계가 고쳤거나 지운 라벨은 stale이고 적용 결정도 반영하지 않는다
    (set·change_kind·delete 모두). 회귀: 종류 변경 제안(g1 사이 구간 → push 행동)을 적용하면 parent
    없는 새 행동 레코드가 drop_retracted를 지나, 다른 단계가 지운 g1이 행동으로 되살아나거나 새 버전
    사이 구간(g1-v2)과 같은 손·같은 구간에 행동이 겹쳐 운영 라벨이 됐다. 결과 기록은 invalid_now라
    기록과 저장이 어긋났다."""
    labels = _scene(engine)
    x = labels[name]
    newer = x.model_copy(update={"label_id": f"{x.label_id}-v2", "parent_label_id": x.label_id,
                                 "retracted": retracted})  # fmt: skip
    with engine.begin() as conn:
        insert_labels(conn, [newer])
    results = _submitted(labels, _decide(x.label_id, "apply"))
    outcome, record = _collect(engine, _setup(tmp_path, results, _ai()))
    assert outcome.stale == [x.label_id] and not outcome.new_records
    item = _items(record)[x.label_id]
    assert (item.human, item.apply_result, item.match, item.result_label_ids) == (
        AiHumanAction.STALE, AiApplyResult.INVALID_NOW, "na", (),
    )  # fmt: skip
    assert record is not None and any("다른 단계가 바꾼" in w for w in record.warnings)
    with engine.connect() as conn:
        current = {r.label_id for r in current_labels(get_labels(conn, SID))}
    assert current == ({r.label_id for r in labels.values()} - {x.label_id}) | (
        set() if retracted else {newer.label_id}
    )


@pytest.mark.parametrize("segment", [True, False], ids=["with-segment", "no-segment"])
def test_change_kind_apply_with_same_drawn_region_is_one_record(
    segment: bool, engine: sa.Engine, tmp_path: Path
) -> None:
    """종류 변경 제안(g1 → push)을 적용하면서 검수자가 같은 구간에 같은 행동을 직접 그렸다: 원
    항목만 빼고 새 항목은 하나만 남는다 (replaced, applied, exact). 회귀: 두 항목의 내용이 같아 같은
    label_id 레코드 둘이 생겨, 상위 구간이 없으면 insert_labels가 IntegrityError(원시 DB 오류)로
    수집 전체를 멈췄다 (상위 구간이 있으면 action_id 중복으로 적용이 invalid_now가 됐다)."""
    labels = _scene(engine)
    g1, s1 = labels["g1"], labels["s1"]
    results = _submitted(labels, _decide(g1.label_id, "apply"),
                         _drawn(1000, 2000, "action.right:push"))  # fmt: skip
    if not segment:  # 검수자가 상위 구간 s1을 지웠다
        results = [r for r in results if r["id"] != s1.label_id]
    outcome, record = _collect(engine, _setup(tmp_path, results, _ai()))
    fresh = [r for r in outcome.new_records if r.parent_label_id is None]
    assert [r.payload for r in fresh] == [PUSH]
    gone = next(r for r in outcome.new_records if r.parent_label_id == g1.label_id)
    assert gone.retracted
    g = _items(record)[g1.label_id]
    assert (g.human, g.apply_result, g.match) == (
        AiHumanAction.REPLACED, AiApplyResult.APPLIED, "exact",
    )  # fmt: skip
    assert set(g.result_label_ids) == {gone.label_id, fresh[0].label_id}


@pytest.mark.parametrize("stream", [None, "bodycam"])
def test_kind_changed_in_tool_like_suggestion_is_applied(
    stream: str | None, engine: sa.Engine, tmp_path: Path
) -> None:
    """검수자가 Label Studio에서 g1의 라벨만 제안대로 바꿔 붙이고(gap → push 행동) 적용도 골랐다:
    검수자 결과가 제안과 같으므로 applied (replaced, exact). 다른 동사로 바꿨으면 conflict_edited
    (같은 종류라 partial, 다음 테스트). 바꿔 붙인 구간은 원 라벨의 스트림을 잇는다 (원 라벨이 기준
    스트림인 실제 행동 단계 출력도 같다). 회귀: 종류를 바꾸면 원 라벨 ID가 사라져 늘
    conflict_edited였다.
    """
    labels = _scene(engine, stream=stream)
    g1 = labels["g1"]
    results = _submitted(labels, _decide(g1.label_id, "apply"))
    _relabel(results, g1.label_id, "action.right:push")
    outcome, record = _collect(engine, _setup(tmp_path, results, _ai()))
    [new] = [r for r in outcome.new_records if r.parent_label_id is None]
    assert new.payload == PUSH and new.stream_id == stream
    g = _items(record)[g1.label_id]
    assert (g.human, g.apply_result, g.match) == (
        AiHumanAction.REPLACED, AiApplyResult.APPLIED, "exact",
    )  # fmt: skip


def test_kind_changed_in_tool_unlike_suggestion_is_conflict(
    engine: sa.Engine, tmp_path: Path
) -> None:
    """적용을 골랐지만 검수자가 다른 동사(pull) 행동으로 바꿔 붙였다 → conflict_edited, partial."""
    labels = _scene(engine)
    g1 = labels["g1"]
    results = _submitted(labels, _decide(g1.label_id, "apply"))
    _relabel(results, g1.label_id, "action.right:pull")
    _, record = _collect(engine, _setup(tmp_path, results, _ai()))
    g = _items(record)[g1.label_id]
    assert (g.human, g.apply_result, g.match) == (
        AiHumanAction.REPLACED, AiApplyResult.CONFLICT_EDITED, "partial",
    )  # fmt: skip


def test_direct_correction_without_decision_measures_match(
    engine: sa.Engine, tmp_path: Path
) -> None:
    """결정 없이 직접 고친 행 (적용 표 "직접 고침"): 제안이 바꾸자고 한 필드 기준으로 match를 잰다.

    a1 제안은 동사 press 하나 → 검수자가 press로 고침 → exact. a3 제안은 동사 pull + 대상 bucket_01
    → 검수자가 동사만 pull로 고침(대상 그대로) → partial (2개 중 1개). a2 제안(사이 구간으로 종류
    변경)인데 동사만 바꿈 → none. 셋 다 decision=none, apply_result 없음.
    """
    labels = _labels()
    a1, a2, a3 = labels["a1"], labels["a2"], labels["a3"]
    pull = a3.payload.model_copy(update={"verb": "pull", "target_id": "bucket_01"})
    verdicts = [
        make_ai_verdict(a1.label_id, "wrong", suggestion=AiSuggestion(
            op="set", fields=("verb",), payload=a1.payload.model_copy(update={"verb": "press"}))),
        make_ai_verdict(a2.label_id, "wrong", problems=("should_be_gap",), suggestion=AiSuggestion(
            op="change_kind", fields=("gap_type", "kind"), payload=IDLE)),
        make_ai_verdict(a3.label_id, "wrong", suggestion=AiSuggestion(
            op="set", fields=("target", "verb"), payload=pull)),
    ]  # fmt: skip
    _scene(engine, verdicts=verdicts)
    results = _submitted(labels)
    _relabel(results, a1.label_id, "action.right:press")
    _relabel(results, a2.label_id, "action.right:push")
    _relabel(results, a3.label_id, "action.left:pull")
    _, record = _collect(engine, _setup(tmp_path, results, _ai()))
    items = _items(record)
    got = {k: (items[x.label_id].human, items[x.label_id].match) for k, x in
           (("a1", a1), ("a2", a2), ("a3", a3))}  # fmt: skip
    assert got == {
        "a1": (AiHumanAction.CORRECTED, "exact"), "a2": (AiHumanAction.CORRECTED, "none"),
        "a3": (AiHumanAction.CORRECTED, "partial"),
    }  # fmt: skip
    assert all(
        (items[x.label_id].decision, items[x.label_id].apply_result) == (AiDecision.NONE, None)
        for x in (a1, a2, a3)
    )


def test_double_task_applies_into_measurement_records(engine: sa.Engine, tmp_path: Path) -> None:
    """이중 작업(arm=shown)도 적용을 같게 반영하고 결과는 측정 레코드다 (human=na)."""
    labels = _scene(engine, mode=ReviewMode.DOUBLE)
    a1 = labels["a1"]
    results = _submitted(labels, _decide(a1.label_id, "apply"))
    outcome, record = _collect(engine, _setup(tmp_path, results, _ai()))
    assert all(r.measurement == "double" for r in outcome.new_records)
    pressed = [r for r in outcome.new_records if isinstance(r.payload, ActionPayload)
               and r.payload.action_id == "a001"]  # fmt: skip
    assert [r.payload.verb for r in pressed if isinstance(r.payload, ActionPayload)] == ["press"]
    item = _items(record)[a1.label_id]
    assert (item.human, item.apply_result, item.match) == (
        AiHumanAction.NA, AiApplyResult.APPLIED, "na",
    )  # fmt: skip


def test_reviewer_own_invalid_result_still_fails_collection(
    engine: sa.Engine, tmp_path: Path
) -> None:
    """검수자 자신의 결과가 세션 검증에 걸리면(상위 구간의 자식 행동을 지움) 지금과 같이 수집이
    실패한다. AI 적용 되돌림은 검수자 오류를 덮지 않는다 (작업은 열린 채)."""
    labels = _scene(engine)
    a2 = labels["a2"]
    results = [r for r in _submitted(labels) if r["id"] != a2.label_id]
    results.append(_decide(labels["a1"].label_id, "apply"))
    setup = _setup(tmp_path, results, _ai())
    with engine.begin() as conn, pytest.raises(OntologyViolationError):
        collect_task(conn, TASK, setup, ASSIGNEE, FIXED_TIME)
    with engine.connect() as conn:
        assert get_review_task(conn, TASK).status is ReviewTaskStatus.OPEN
        assert get_ai_outcome(conn, TASK) is None


def test_same_as_sent_is_the_reconcile_criterion() -> None:
    """`same_as_sent`: 페이로드·시작·끝·스트림이 모두 같아야 같다 (검수 상태·출처는 보지 않는다)."""
    a1 = _labels()["a1"]
    item = ReviewedItem(a1.label_id, a1.stream_id, a1.t_start_ms, a1.t_end_ms, a1.payload)
    assert same_as_sent(a1, item)
    assert not same_as_sent(a1, replace(item, t_end_ms=900))
    human = a1.model_copy(update={"provenance": Provenance(source=Source.HUMAN)})
    assert same_as_sent(human, item)
