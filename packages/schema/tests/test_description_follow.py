"""설명 라벨이 행동 검수 결과를 따르는 규칙 (`dlp_schema.description_follow`, F-PAR-15).

순수 계산만 본다 (DB 적용과 수집 연결은 `packages/review/tests/test_description_follow.py`).
정답 근거: 손으로 만든 이력. 행동 a001(오른손 grasp)과 그 설명 d(한국어)·den(영어 번역).
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from dlp_schema.description_follow import (
    InheritedReview,
    meaning,
    plan_description_follow,
)
from dlp_schema.labels import (
    Hand,
    LabelRecord,
    Provenance,
    Source,
    Verification,
    VerificationState,
)
from dlp_schema.testing import FIXED_TIME, action_payload, make_label

T1 = FIXED_TIME + timedelta(hours=1)
T2 = FIXED_TIME + timedelta(hours=2)
MODEL: dict[str, Any] = {
    "provenance": Provenance(source=Source.MODEL, model_version="actions-1"),
    "confidence": 0.9,
}


def _checked(state: VerificationState, who: str, at: Any = T1) -> Verification:
    return Verification(state=state, reviewer_id=who, reviewed_at=at)


def _action(label_id: str = "act", **kw: Any) -> LabelRecord:
    """모델 행동 (기본 a001 오른손 grasp 0~1 s). kw는 페이로드 덮어쓰기와 `record`(레코드 필드)."""
    record: dict[str, Any] = kw.pop("record", {})
    return make_label(action_payload(**kw), label_id=label_id, **{**MODEL, **record})


def _desc(label_id: str = "d", segment: str = "a001", **kw: Any) -> LabelRecord:
    """모델 설명 (source가 있으면 영어 번역)."""
    payload: dict[str, Any] = {"kind": "description", "segment_id": segment, "text": "설명"}
    if (source := kw.pop("source", None)) is not None:
        payload |= {"language": "en", "source_label_id": source, "text": "desc"}
    return make_label(payload, label_id=label_id, **{**MODEL, **kw})


def _fix(parent: LabelRecord, label_id: str, **payload: Any) -> LabelRecord:
    """검수자 r1이 T1에 고친 레코드 (새 사람 레코드 + parent)."""
    return parent.model_copy(
        update={
            "label_id": label_id,
            "parent_label_id": parent.label_id,
            "provenance": Provenance(source=Source.HUMAN),
            "confidence": None,
            "verification": _checked(VerificationState.HUMAN_CORRECTED, "r1"),
            "payload": parent.payload.model_copy(update=payload),
        }
    )


def _ids(records: list[LabelRecord]) -> list[str]:
    return [x.label_id for x in records]


def test_approved_action_passes_review_to_original_and_translation() -> None:
    """승인된 행동 → 원문·번역 human_approved, 검수자 같음, 시각 = max(행동 검수, 설명 생성)."""
    act = _action(record={"verification": _checked(VerificationState.HUMAN_APPROVED, "r1")})
    late_en = _desc("den", source="d", created_at=T2)  # 승인 뒤에 만든 번역
    plan = plan_description_follow([act, _desc(), late_en])
    assert plan.retract == []
    assert plan.approve == [
        InheritedReview("d", VerificationState.HUMAN_APPROVED, "r1", T1, "act"),
        InheritedReview("den", VerificationState.HUMAN_APPROVED, "r1", T2, "act"),
    ]


def test_unreviewed_owner_waits_and_unknown_owner_is_left_alone() -> None:
    """행동이 검수 전이면 기다리고, 주인 ID가 이력에 없으면 건드리지 않는다."""
    plan = plan_description_follow([_action(), _desc(), _desc("dx", "nope")])
    assert (plan.approve, plan.retract) == ([], [])


def test_bimanual_action_needs_every_hand_reviewed_and_weakest_state_wins() -> None:
    """같은 action_id의 두 손 레코드: 하나라도 검수 전이면 기다리고, 하나가 표본 검증이면
    설명도 sample_verified (그 표본 검증의 검수자)."""
    right = _action("r", record={"verification": _checked(VerificationState.HUMAN_APPROVED, "r1")})
    left = _action("l", hand="left")
    assert plan_description_follow([right, left, _desc()]).approve == []
    sampled = left.model_copy(
        update={"verification": _checked(VerificationState.SAMPLE_VERIFIED, "sampling:a", T2)}
    )
    plan = plan_description_follow([right, sampled, _desc()])
    assert plan.approve == [
        InheritedReview("d", VerificationState.SAMPLE_VERIFIED, "sampling:a", T2, "l")
    ]


def test_time_only_correction_approves_but_meaning_change_retracts() -> None:
    """경계만 고친 행동 → 설명 승인(human_approved). 동사·대상·손이 바뀌면 검수 전 설명을 지운다.
    이미 검수된 설명·사람 설명은 어느 쪽도 아니다."""
    act = _action()
    moved = _fix(act, "fix", t_approach_ms=0, t_contact_start_ms=300)
    plan = plan_description_follow([act, moved, _desc()])
    assert plan.approve == [InheritedReview("d", VerificationState.HUMAN_APPROVED, "r1", T1, "fix")]
    for change in ({"verb": "lift"}, {"target_id": "bucket_01"}, {"hand": Hand.LEFT}):
        reviewed = _desc("dok", verification=_checked(VerificationState.HUMAN_APPROVED, "lead"))
        human = _desc("dh", provenance=Provenance(source=Source.HUMAN), confidence=None)
        history = [act, _fix(act, "fix", **change), _desc(), _desc("den", source="d"), reviewed,
                   human]  # fmt: skip
        plan = plan_description_follow(history)
        assert (plan.approve, _ids(plan.retract)) == ([], ["d", "den"]), change


def test_meaning_ignores_only_time_fields() -> None:
    """의미 비교는 행동 시각 필드만 뺀다 (상위 구간은 페이로드 전체)."""
    act = _action()
    assert meaning(act) == meaning(_fix(act, "x", t_end_ms=1000, contact_held=True))
    assert meaning(act) != meaning(_fix(act, "x", pre_state={"wetness": "dry"}))


def test_deleted_owner_retracts_and_stale_translation_is_skipped() -> None:
    """행동을 지우면(현재 주인 없음) 원문·번역을 지운다. 원문이 현재 라벨이 아닌 번역은 승인하지
    않는다 (번역 단계가 낡은 번역으로 지운다)."""
    act = _action()
    gone = act.model_copy(
        update={"label_id": "gone", "parent_label_id": "act", "retracted": True,
                "provenance": Provenance(source=Source.HUMAN), "confidence": None,
                "verification": _checked(VerificationState.HUMAN_CORRECTED, "r1")}
    )  # fmt: skip
    plan = plan_description_follow([act, gone, _desc(), _desc("den", source="d")])
    assert (plan.approve, _ids(plan.retract)) == ([], ["d", "den"])

    approved = _action(record={"verification": _checked(VerificationState.HUMAN_APPROVED, "r1")})
    orphan_en = _desc("den", source="old")  # 원문 old는 이력에 없다
    plan = plan_description_follow([approved, orphan_en])
    assert (plan.approve, plan.retract) == ([], [])


def test_segment_owner_and_seeded_copies() -> None:
    """상위 구간도 주인이다 (ref_id가 바뀌면 지운다). 오류 삽입 사본은 주인으로 보지 않는다."""
    seg = make_label(
        {"kind": "segment", "segment_id": "sk1", "level": "skill", "ref_id": "wipe",
         "child_ids": []}, label_id="seg", t_end_ms=3000, **MODEL,
    )  # fmt: skip
    plan = plan_description_follow([seg, _fix(seg, "seg2", ref_id="sweep"), _desc("ds", "sk1")])
    assert _ids(plan.retract) == ["ds"]
    seeded = _action("seed", record={"seeded_error": True})
    plan = plan_description_follow([seeded, _desc()])
    assert (plan.approve, plan.retract) == ([], [])
