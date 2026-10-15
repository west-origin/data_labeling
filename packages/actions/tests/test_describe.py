"""사람이 의미를 고친 행동의 설명 재생성 (`dlp_actions.describe`, ADR 0043 3절).

서비스 없이 시간대 보존 SQLite와 `OracleVlm`(정답 라벨로 답하는 stub)으로 돈다.

정답 근거: 직접 만든 이력. 모델 행동 a001(오른손 grasp rag_01)과 그 모델 설명. 검수자 r1이 T1에
동사를 lift로 고친다 → 설명 따르기가 모델 설명을 지운다 → 재생성은 고친 동사로 묶인 요청을 보내
정답 설명("걸레를 들어 올린다")을 쓰고, 고친 행동의 검수(human_approved, r1)를 물려받는다. 검수
시각은 생성 시각이다 (검수 뒤 기계가 만든 글). 번역은 평소대로 만들고 같은 검수를 물려받는다.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa

from dlp_actions.clients import OracleVlm
from dlp_actions.describe import (
    DescribeTarget,
    describe_schema,
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
from dlp_media.storage import LocalStore, blurred_key, sha256_file
from dlp_privacy.policy import load_policy as load_privacy_policy
from dlp_privacy.runner import RenderMeta, render_hash, write_render_meta
from dlp_schema.db.description_repository import apply_description_follow
from dlp_schema.db.repository import (
    get_description_inheritance,
    get_labels,
    insert_labels,
    insert_session,
    insert_withdrawal,
    record_privacy_hold,
    register_ontology,
    set_privacy_state,
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
from dlp_schema.lineage import Withdrawal
from dlp_schema.ontology import Ontology, load_ontology
from dlp_schema.review import InheritanceMechanism
from dlp_schema.session import PrivacyState
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
    (hand,) = seen[0]["properties"]["hands"]["prefixItems"]
    assert hand["properties"]["verb"] == {"enum": ["lift"]}
    assert hand["properties"]["target_id"] == {"enum": ["rag_01"]}
    assert [x for x in _descriptions(db) if x.label_id != "d"] == []
    target = DescribeTarget("a001", (fix,), "x")
    meaning: dict[str, Any] = {
        "hand": "right", "verb": "lift", "target_id": "rag_01", "target_body_part": None,
        "tool_id": None, "direction": None, "pre_state": {}, "post_state": {},
    }  # fmt: skip
    ok = json.dumps({"label": "action", "hands": [meaning], "description": "들기"})
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
    plan = plan_describe(history, "describe-x+oracle-vlm-1", ())
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


def test_direction_correction_rejects_an_answer_for_the_old_direction(
    db: sa.Engine, ontology: Ontology, policy: ActionsPolicy
) -> None:
    """F-PAR-21 (Codex 2차 #1): 검수자가 방향만 고쳤다 (clockwise → counterclockwise). 방향은 의미
    필드라 모델 설명은 지워지고 재생성 대상이다. 요청은 고친 방향을 싣고, 예전 방향으로 답하는
    stub(고치기 전 레코드를 정답으로 아는 stub)은 매번 위반이라 만들지 않는다(fallback).

    정답: 요청 프롬프트·스키마에 counterclockwise가 있고 clockwise는 없다. 예전 행동으로 답하면
    재시도(정책 max_retries) 뒤 fallback, 고친 행동으로 답하면 만든다.
    """
    act = _action("act", direction="clockwise")
    fix = _human(act, "fix", direction="counterclockwise")
    with db.begin() as conn:
        insert_labels(conn, [act, _model_desc(), fix])
    seen: list[tuple[str, str]] = []

    class Spy(OracleVlm):
        def complete(self, request: SegmentRequest, prompt: str, schema: dict[str, Any]) -> str:
            seen.append((prompt, json.dumps(schema)))
            return super().complete(request, prompt, schema)

    stale = Spy([act])
    s = _describe(db, ontology, policy, stale)
    assert s.described == [] and s.fallbacks == ["a001"]
    assert stale.calls == policy.vlm.max_retries + 1
    prompt, schema = seen[0]
    assert "counterclockwise" in prompt and "counterclockwise" in schema
    assert "clockwise" not in prompt.replace("counterclockwise", "")
    assert _describe(db, ontology, policy, _oracle(fix)).described == ["a001"]


def test_bimanual_answer_is_checked_per_hand_combination() -> None:
    """F-PAR-21: 양손 행동(오른손 grasp rag_01 도구 없음, 왼손 lift bottle_01 도구 sponge_01)에서
    손마다 값을 섞은 응답(오른손 grasp bottle_01 sponge_01)은 값 집합 안이어도 위반이다. 손마다
    확정한 조합 그대로면 받는다. 대상 부위·전후 상태도 손 조합에 든다.

    정답: 확정 조합 = {(right, grasp, rag_01, None), (left, lift, bottle_01, sponge_01)}.
    """
    right = _human(_action("r"), "r2", verb="grasp", target_id="rag_01", tool_id=None)
    left = _human(_action("l", hand="left"), "l2", verb="lift", target_id="bottle_01",
                  tool_id="sponge_01", post_state={"wetness": "wet"})  # fmt: skip
    target = DescribeTarget("a001", (right, left), "x")
    base: dict[str, Any] = {"target_body_part": None, "direction": None, "pre_state": {}}
    good_hands: list[dict[str, Any]] = [
        {**base, "hand": "right", "verb": "grasp", "target_id": "rag_01", "tool_id": None,
         "post_state": {}},
        {**base, "hand": "left", "verb": "lift", "target_id": "bottle_01", "tool_id": "sponge_01",
         "post_state": {"wetness": "wet"}},
    ]  # fmt: skip
    good: dict[str, Any] = {"label": "action", "hands": good_hands, "description": "잡고 든다"}
    assert validate_description(json.dumps(good), target)[0] is not None
    crossed = json.loads(json.dumps(good))
    crossed["hands"][0] |= {"target_id": "bottle_01", "tool_id": "sponge_01"}
    answer, errors = validate_description(json.dumps(crossed), target)
    assert answer is None and errors
    stale_state = json.loads(json.dumps(good))
    stale_state["hands"][1]["post_state"] = {}
    assert validate_description(json.dumps(stale_state), target)[0] is None
    one_hand: dict[str, Any] = {**good, "hands": good_hands[:1]}
    assert validate_description(json.dumps(one_hand), target)[0] is None
    schema = describe_schema(target)["properties"]["hands"]
    assert (schema["minItems"], schema["maxItems"]) == (2, 2)
    assert [h["properties"]["verb"] for h in schema["prefixItems"]] == [
        {"enum": ["grasp"]}, {"enum": ["lift"]},
    ]  # fmt: skip


class _Interrupting(OracleVlm):
    """첫 VLM 호출 때 다른 실행이 세션 상태를 바꾼 것처럼 hook을 부른다."""

    hook: Callable[[], None] | None = None

    def complete(self, request: SegmentRequest, prompt: str, schema: dict[str, Any]) -> str:
        if self.calls == 0 and self.hook is not None:
            self.hook()
        return super().complete(request, prompt, schema)


def _approved_render(db: sa.Engine, tmp_path: Path) -> LocalStore:
    """세션을 승인 상태로 두고 지금 렌더(블러 0개)의 블러본·렌더 기록을 라벨링 저장소에 둔다."""
    labeling = LocalStore(tmp_path / "store", "dlp-labeling")
    video = tmp_path / "v.bin"
    video.write_bytes(b"synthetic")
    sha = sha256_file(video)
    labeling.put_file(blurred_key(SID, "bodycam"), video, sha)
    current = render_hash([], load_privacy_policy(ROOT))
    write_render_meta(labeling, SID, "bodycam", RenderMeta(current, sha), tmp_path)
    with db.begin() as conn:
        set_privacy_state(conn, SID, PrivacyState.APPROVED)
    return labeling


@pytest.mark.parametrize("change", ["hold", "withdraw", "approval"])
def test_final_write_rechecks_hold_withdrawal_and_render_under_the_lock(
    db: sa.Engine, ontology: Ontology, policy: ActionsPolicy, change: str, tmp_path: Path
) -> None:
    """F-PAR-23: VLM에 묻는 동안(잠금 전) 세션이 승인 보류·사용 중지되거나 승인이 풀리면(블러
    변경 등, 받은 블러본이 더는 지금 렌더가 아님) 쓰기 직전 확인이 막아 아무것도 쓰지 않고 이유를
    알린다.

    정답: blocked에 이유가 있고, 새 설명·설명 따르기 삭제·물려받기 기록이 하나도 없다 (이력 그대로).
    바뀌지 않으면 같은 준비로 설명을 만든다 (대조).
    """
    act = _action("act")
    fix = _human(act, "fix", verb="lift")
    with db.begin() as conn:
        insert_labels(conn, [act, _model_desc(), fix])
    labeling = _approved_render(db, tmp_path)
    with db.connect() as conn:
        before = get_labels(conn, SID)
    vlm = _Interrupting([fix])
    with db.begin() as conn:
        hooks: dict[str, Callable[[], Any]] = {
            "hold": lambda: record_privacy_hold(
                conn, SID, "hold", reason="report", actor="ops01", at=T1
            ),
            "withdraw": lambda: insert_withdrawal(
                conn, Withdrawal(session_id=SID, reason="consent", withdrawn_at=T1)
            ),
            "approval": lambda: set_privacy_state(conn, SID, PrivacyState.AUTO_BLURRED),
        }
        vlm.hook = hooks[change]
        s = describe_session(conn, SID, vlm, ontology, policy, T2, labeling)
    assert s.described == [] and s.blocked, s
    with db.connect() as conn:
        assert get_labels(conn, SID) == before
        assert get_description_inheritance(conn, SID) == []


def test_final_write_check_passes_when_nothing_changed(
    db: sa.Engine, ontology: Ontology, policy: ActionsPolicy, tmp_path: Path
) -> None:
    """대조: 같은 준비(승인·지금 렌더)에서 아무것도 바뀌지 않으면 쓴다."""
    act = _action("act")
    fix = _human(act, "fix", verb="lift")
    with db.begin() as conn:
        insert_labels(conn, [act, _model_desc(), fix])
    labeling = _approved_render(db, tmp_path)
    with db.begin() as conn:
        s = describe_session(conn, SID, _oracle(fix), ontology, policy, T2, labeling)
    assert s.described == ["a001"] and s.blocked is None
