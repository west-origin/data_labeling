"""사람이 의미를 고친 행동의 설명 재생성 (`dlp_actions.describe`, ADR 0043 3절).

서비스 없이 시간대 보존 SQLite와 `OracleVlm`(정답 라벨로 답하는 stub)으로 돈다.

정답 근거: 직접 만든 이력. 모델 행동 a001(오른손 grasp rag_01)과 그 모델 설명. 검수자 r1이 T1에
동사를 lift로 고친다 → 설명 따르기가 모델 설명을 지운다 → 재생성은 고친 동사로 묶인 요청을 보내
정답 설명("걸레를 들어 올린다")을 쓰고, 고친 행동의 검수(human_approved, r1)를 물려받는다. 검수
시각은 생성 시각이다 (검수 뒤 기계가 만든 글). 번역은 평소대로 만들고 같은 검수를 물려받는다.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa

from dlp_actions.clients import OracleVlm
from dlp_actions.describe import (
    DescribeTarget,
    describe_session,
    describe_version,
    plan_describe,
    validate_description,
)
from dlp_actions.policy import ActionsPolicy
from dlp_actions.policy import load_policy as load_actions_policy
from dlp_actions.translate import run_translate
from dlp_actions.translate_policy import load_policy as load_translate_policy
from dlp_actions.translators import GlossaryStubTranslator
from dlp_actions.vlm import SegmentRequest
from dlp_schema.db.description_repository import apply_description_follow
from dlp_schema.db.repository import (
    get_description_inheritance,
    get_labels,
    insert_labels,
    insert_session,
    register_ontology,
)
from dlp_schema.description_follow import FOLLOW_VERSION, REGENERATED_PREFIX, inherited_owner
from dlp_schema.episode import current_labels
from dlp_schema.labels import (
    DescriptionPayload,
    LabelRecord,
    Provenance,
    Source,
    Verification,
    VerificationState,
)
from dlp_schema.ontology import Ontology, load_ontology
from dlp_schema.review import InheritanceMechanism
from dlp_schema.testing import FIXED_TIME, action_payload, make_label, make_session, sqlite_db

ROOT = Path(__file__).resolve().parents[3]
SID = "s001"
T1 = FIXED_TIME + timedelta(hours=1)
T2 = FIXED_TIME + timedelta(hours=2)
T3 = FIXED_TIME + timedelta(hours=3)
ACTIONS = Provenance(source=Source.MODEL, model_version="actions-abc+oracle-vlm-1+i00000000")
NEW_TEXT = "걸레를 들어 올린다"


@pytest.fixture(scope="module")
def ontology() -> Ontology:
    """온톨로지 v1."""
    return load_ontology(ROOT / "config/ontology/v1")


@pytest.fixture(scope="module")
def policy() -> ActionsPolicy:
    """저장소의 actions.yaml."""
    return load_actions_policy(ROOT)


def _action(label_id: str, **payload: Any) -> LabelRecord:
    """행동 단계가 낸 모델 행동 (기본 a001 오른손 grasp rag_01)."""
    return make_label(
        action_payload(**payload), label_id=label_id, session_id=SID, provenance=ACTIONS,
        confidence=0.9,
    )  # fmt: skip


def _model_desc(label_id: str = "d", segment: str = "a001") -> LabelRecord:
    """행동 단계가 낸 모델 설명."""
    return make_label(
        {"kind": "description", "segment_id": segment, "text": "걸레를 잡는다"},
        label_id=label_id, session_id=SID, provenance=ACTIONS, confidence=0.9,
    )  # fmt: skip


def _human(
    parent: LabelRecord, label_id: str, *, retracted: bool = False, **payload: Any
) -> LabelRecord:
    """검수자 r1이 T1에 고친(또는 지운) 레코드."""
    return parent.model_copy(
        update={
            "label_id": label_id, "parent_label_id": parent.label_id, "retracted": retracted,
            "provenance": Provenance(source=Source.HUMAN), "confidence": None, "created_at": T1,
            "verification": Verification(state=VerificationState.HUMAN_CORRECTED,
                                         reviewer_id="r1", reviewed_at=T1),
            "payload": parent.payload.model_copy(update=payload),
        }
    )  # fmt: skip


def _approved(x: LabelRecord) -> LabelRecord:
    """검수자 r1이 T1에 승인한 레코드 (같은 ID)."""
    return x.model_copy(
        update={"verification": Verification(state=VerificationState.HUMAN_APPROVED,
                                             reviewer_id="r1", reviewed_at=T1)}
    )  # fmt: skip


@pytest.fixture
def db(ontology: Ontology) -> Iterator[sa.Engine]:
    """세션 s001(바디캠 기준 스트림)만 있는 메모리 DB."""
    engine = sqlite_db()
    with engine.begin() as conn:
        register_ontology(conn, ontology)
        insert_session(conn, make_session(SID))
    yield engine
    engine.dispose()


def _oracle(*truth: LabelRecord) -> OracleVlm:
    """고친 행동과 정답 설명으로 답하는 stub."""
    text = make_label(
        {"kind": "description", "segment_id": "a001", "text": NEW_TEXT}, label_id="truth-d"
    )
    return OracleVlm([*truth, text])


def _describe(engine: sa.Engine, ontology: Ontology, policy: ActionsPolicy, vlm: Any) -> Any:
    with engine.begin() as conn:
        return describe_session(conn, SID, vlm, ontology, policy, T2)


def _descriptions(engine: sa.Engine) -> list[LabelRecord]:
    with engine.connect() as conn:
        current = current_labels(get_labels(conn, SID))
    return [x for x in current if isinstance(x.payload, DescriptionPayload)]


def test_meaning_correction_regenerates_and_inherits_approval(
    db: sa.Engine, ontology: Ontology, policy: ActionsPolicy
) -> None:
    """동사를 고친 행동: 모델 설명은 지워지고, 고친 동사로 묶은 요청으로 새 설명을 만들어 행동
    검수를 물려받는다 (검수 시각 = 생성 시각 T2). 번역도 같은 검수를 물려받는다. 다시 돌리면 아무
    것도 묻지 않는다 (멱등)."""
    act = _action("act")
    fix = _human(act, "fix", verb="lift")
    with db.begin() as conn:
        insert_labels(conn, [act, _model_desc(), fix])
        follow = apply_description_follow(conn, SID, T1, InheritanceMechanism.COLLECT)
        assert [d.label_id for d in follow.retract] == ["d"]
    vlm = _oracle(fix)
    s = _describe(db, ontology, policy, vlm)
    assert s.described == ["a001"] and s.fallbacks == [] and s.inherited == 1
    assert s.version == describe_version(policy, vlm.version)
    (d,) = _descriptions(db)
    p = d.payload
    assert isinstance(p, DescriptionPayload)
    assert (p.segment_id, p.text, p.source_label_id) == ("a001", NEW_TEXT, None)
    assert d.provenance.source is Source.MODEL
    assert (d.provenance.model_version or "").startswith(REGENERATED_PREFIX)
    assert d.created_at == T2  # 행동 검수(T1) 뒤에 기계가 만든 글
    v = d.verification
    assert (v.state, v.reviewer_id, v.reviewed_at) == (VerificationState.HUMAN_APPROVED, "r1", T2)
    with db.connect() as conn:
        recs = get_description_inheritance(conn, SID)
        owner = inherited_owner(d, get_labels(conn, SID), recs)
    assert [r.mechanism for r in recs] == [InheritanceMechanism.DESCRIBE]
    assert owner is not None and owner.label_id == "fix"
    calls = vlm.calls
    again = _describe(db, ontology, policy, vlm)
    assert (again.described, vlm.calls) == ([], calls)
    # 번역은 평소대로 만들고 같은 검수를 물려받는다
    stub = GlossaryStubTranslator({NEW_TEXT: "Lifts the cloth"})
    with db.begin() as conn:
        t = run_translate(conn, SID, stub, ontology, load_translate_policy(ROOT), T3)
    assert t.translated == 1 and t.pending == []
    en = next(x for x in _descriptions(db) if x.label_id != d.label_id)
    assert en.verification.state is VerificationState.HUMAN_APPROVED
    assert en.verification.reviewer_id == "r1"


def test_request_is_constrained_to_the_corrected_values(
    db: sa.Engine, ontology: Ontology, policy: ActionsPolicy
) -> None:
    """요청 스키마는 고친 동사·대상만 허용하고, 다른 동작을 답하면 위반으로 다시 묻는다. 끝내
    위반이면
    만들지 않고(fallback) 다음 실행이 다시 묻는다."""
    act = _action("act")
    fix = _human(act, "fix", verb="lift")
    with db.begin() as conn:
        insert_labels(conn, [act, _model_desc(), fix])
    seen: list[dict[str, Any]] = []

    class Spy(OracleVlm):
        def complete(self, request: SegmentRequest, prompt: str, schema: dict[str, Any]) -> str:
            seen.append(schema)
            assert "lift" in prompt
            return super().complete(request, prompt, schema)

    retries = policy.vlm.max_retries
    bad = Spy([act], faults=[])  # 고치기 전 동작(grasp)으로 답한다 → 매번 위반
    s = _describe(db, ontology, policy, bad)
    assert s.described == [] and s.fallbacks == ["a001"] and bad.calls == retries + 1
    assert seen[0]["properties"]["verb"] == {"enum": ["lift"]}
    assert seen[0]["properties"]["target_id"] == {"enum": ["rag_01"]}
    assert [x for x in _descriptions(db) if x.label_id != "d"] == []
    target = DescribeTarget("a001", (fix,), "x")
    ok = json.dumps(
        {"label": "action", "verb": "lift", "target_id": "rag_01", "description": "들기"}
    )
    assert validate_description(ok, target)[0] is not None
    s2 = _describe(db, ontology, policy, _oracle(fix))
    assert s2.described == ["a001"]


def test_waits_for_review_and_skips_untouched_or_time_only(
    db: sa.Engine, ontology: Ontology, policy: ActionsPolicy
) -> None:
    """양손 행동 중 한 손만 고치고 다른 손이 검수 전이면 기다린다. 사람이 손대지 않은 행동, 시각만
    고친 행동(설명이 남아 물려받는다)은 대상이 아니다."""
    right = _action("r")
    left = _action("l", hand="left")
    other = _action("b", action_id="b001", t_approach_ms=0)
    moved = _action("m", action_id="m001")
    history = [
        right, left, _human(left, "l2", verb="lift"), other, _model_desc("db", "b001"),
        moved, _model_desc("dm", "m001"), _human(moved, "m2", t_end_ms=900),
    ]  # fmt: skip
    plan = plan_describe(history, "describe-x+oracle-vlm-1")
    assert (plan.targets, plan.waiting, plan.deleted) == ([], 1, 0)


def test_partial_deletion_regenerates_for_the_remaining_hand(
    db: sa.Engine, ontology: Ontology, policy: ActionsPolicy
) -> None:
    """양손 행동에서 왼손을 지우고 오른손을 승인: 양손 설명은 지워지고(F-PAR-16) 남은 오른손
    행동으로
    설명을 다시 만들어 승인을 물려받는다."""
    right = _approved(_action("r"))
    left = _action("l", hand="left")
    with db.begin() as conn:
        insert_labels(conn, [right, left, _model_desc(), _human(left, "l-gone", retracted=True)])
    s = _describe(db, ontology, policy, _oracle(right))
    assert s.described == ["a001"] and s.retracted == 1  # 모델 설명은 설명 따르기가 지웠다
    (d,) = _descriptions(db)
    assert d.verification.state is VerificationState.HUMAN_APPROVED
    assert isinstance(d.payload, DescriptionPayload) and d.payload.text == NEW_TEXT


def test_inherited_description_is_replaced_when_owner_changes_in_a_later_round(
    db: sa.Engine, ontology: Ontology, policy: ActionsPolicy
) -> None:
    """ADR 0043 3절 규칙 6: 회차 1에 승인된 행동의 설명(ko·en)이 검수를 물려받은 뒤, 회차 2에
    검수자 r2가 동사를 lift로 고치면 물려받은 설명은 보호를 잃어 지워지고, 재생성·번역이 새로
    만들어 r2의 검수를 물려받는다. 직접 검수한 설명은 아니므로 저장 검증이 막지 않는다.

    정답: 회차 1 뒤 현재 설명 = 모델 원문 d(r1, T1)와 번역(r1, T2). 회차 2 뒤 = 재생성 원문
    (NEW_TEXT, r2, 검수 시각 = 생성 시각 T3)과 그 번역(r2, T4). 예전 d·번역은 설명 따르기 삭제.
    """
    act = _approved(_action("act"))
    texts = {"걸레를 잡는다": "Grasps the cloth", NEW_TEXT: "Lifts the cloth"}
    stub = GlossaryStubTranslator(texts)
    tpolicy = load_translate_policy(ROOT)
    with db.begin() as conn:
        insert_labels(conn, [act, _model_desc()])
        first = apply_description_follow(conn, SID, T1, InheritanceMechanism.COLLECT)
        assert [r.label_id for r in first.approve] == ["d"]
        assert run_translate(conn, SID, stub, ontology, tpolicy, T2).inherited == 1
    first = {x.label_id for x in _descriptions(db)}
    assert len(first) == 2

    fix = _human(act, "fix", verb="lift").model_copy(
        update={"created_at": T2, "verification": Verification(
            state=VerificationState.HUMAN_CORRECTED, reviewer_id="r2", reviewed_at=T2)}
    )  # fmt: skip
    with db.begin() as conn:
        insert_labels(conn, [fix])
        s = describe_session(conn, SID, _oracle(fix), ontology, policy, T3)
    assert s.described == ["a001"] and s.retracted == 2 and s.inherited == 1
    with db.begin() as conn:
        t = run_translate(conn, SID, stub, ontology, tpolicy, T3 + timedelta(hours=1))
    assert t.translated == 1 and t.inherited == 1
    now = _descriptions(db)
    assert first.isdisjoint({x.label_id for x in now}) and len(now) == 2
    for x in now:
        v = x.verification
        assert (v.state, v.reviewer_id) == (VerificationState.HUMAN_APPROVED, "r2")
    ko = next(x for x in now if isinstance(x.payload, DescriptionPayload)
              and x.payload.source_label_id is None)  # fmt: skip
    assert isinstance(ko.payload, DescriptionPayload) and ko.payload.text == NEW_TEXT
    with db.connect() as conn:
        history = get_labels(conn, SID)
    gone = [x for x in history if x.retracted and x.parent_label_id in first]
    assert {x.provenance.model_version for x in gone} == {FOLLOW_VERSION}
