"""설명 라벨이 행동 검수 결과를 따르는 규칙 (`dlp_schema.description_follow`, F-PAR-15).

순수 계산만 본다 (DB 적용과 수집 연결은 `packages/review/tests/test_description_follow.py`).
정답 근거: 손으로 만든 이력. 행동 a001(오른손 grasp)과 그 설명 d(한국어)·den(영어 번역).
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from dlp_schema.description_follow import (
    FOLLOW_VERSION,
    InheritedReview,
    active_inheritance,
    inheritance_only,
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
from dlp_schema.review import DescriptionInheritance, InheritanceMechanism
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
    plan = plan_description_follow([act, _desc(), late_en], ())
    assert plan.retract == []
    assert plan.approve == [
        InheritedReview("d", VerificationState.HUMAN_APPROVED, "r1", T1, "act", ("act",)),
        InheritedReview("den", VerificationState.HUMAN_APPROVED, "r1", T2, "act", ("act",)),
    ]


def test_unreviewed_owner_waits_and_unknown_owner_is_left_alone() -> None:
    """행동이 검수 전이면 기다리고, 주인 ID가 이력에 없으면 건드리지 않는다."""
    plan = plan_description_follow([_action(), _desc(), _desc("dx", "nope")], ())
    assert (plan.approve, plan.retract) == ([], [])


def test_bimanual_action_needs_every_hand_reviewed_and_weakest_state_wins() -> None:
    """같은 action_id의 두 손 레코드: 하나라도 검수 전이면 기다리고, 하나가 표본 검증이면
    설명도 sample_verified (그 표본 검증의 검수자)."""
    right = _action("r", record={"verification": _checked(VerificationState.HUMAN_APPROVED, "r1")})
    left = _action("l", hand="left")
    assert plan_description_follow([right, left, _desc()], ()).approve == []
    sampled = left.model_copy(
        update={"verification": _checked(VerificationState.SAMPLE_VERIFIED, "sampling:a", T2)}
    )
    plan = plan_description_follow([right, sampled, _desc()], ())
    assert plan.approve == [
        InheritedReview("d", VerificationState.SAMPLE_VERIFIED, "sampling:a", T2, "l", ("l", "r"))
    ]


def test_time_only_correction_approves_but_meaning_change_retracts() -> None:
    """경계만 고친 행동 → 설명 승인(human_approved). 동사·대상·손이 바뀌면 검수 전 설명을 지운다.
    이미 검수된 설명·사람 설명은 어느 쪽도 아니다."""
    act = _action()
    moved = _fix(act, "fix", t_approach_ms=0, t_contact_start_ms=300)
    plan = plan_description_follow([act, moved, _desc()], ())
    assert plan.approve == [
        InheritedReview("d", VerificationState.HUMAN_APPROVED, "r1", T1, "fix", ("fix",))
    ]
    for change in ({"verb": "lift"}, {"target_id": "bucket_01"}, {"hand": Hand.LEFT}):
        reviewed = _desc("dok", verification=_checked(VerificationState.HUMAN_APPROVED, "lead"))
        human = _desc("dh", provenance=Provenance(source=Source.HUMAN), confidence=None)
        history = [act, _fix(act, "fix", **change), _desc(), _desc("den", source="d"), reviewed,
                   human]  # fmt: skip
        plan = plan_description_follow(history, ())
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
    plan = plan_description_follow([act, gone, _desc(), _desc("den", source="d")], ())
    assert (plan.approve, _ids(plan.retract)) == ([], ["d", "den"])

    approved = _action(record={"verification": _checked(VerificationState.HUMAN_APPROVED, "r1")})
    orphan_en = _desc("den", source="old")  # 원문 old는 이력에 없다
    plan = plan_description_follow([approved, orphan_en], ())
    assert (plan.approve, plan.retract) == ([], [])


def test_segment_owner_and_seeded_copies() -> None:
    """상위 구간도 주인이다 (ref_id가 바뀌면 지운다). 오류 삽입 사본은 주인으로 보지 않는다."""
    seg = make_label(
        {"kind": "segment", "segment_id": "sk1", "level": "skill", "ref_id": "wipe",
         "child_ids": []}, label_id="seg", t_end_ms=3000, **MODEL,
    )  # fmt: skip
    plan = plan_description_follow([seg, _fix(seg, "seg2", ref_id="sweep"), _desc("ds", "sk1")], ())
    assert _ids(plan.retract) == ["ds"]
    seeded = _action("seed", record={"seeded_error": True})
    plan = plan_description_follow([seeded, _desc()], ())
    assert (plan.approve, plan.retract) == ([], [])


def _deleted(x: LabelRecord, label_id: str, at: Any = T1) -> LabelRecord:
    """검수자 r1이 at에 지운 레코드 (삭제 레코드 + parent)."""
    return x.model_copy(
        update={"label_id": label_id, "parent_label_id": x.label_id, "retracted": True,
                "provenance": Provenance(source=Source.HUMAN), "confidence": None,
                "verification": _checked(VerificationState.HUMAN_CORRECTED, "r1", at),
                "created_at": at}
    )  # fmt: skip


def test_bimanual_partial_deletion_retracts_description() -> None:
    """F-PAR-16: 양손 행동(같은 action_id 두 레코드)에서 한 손을 지우고 다른 손을 승인하면 설명은
    의미가 바뀐 것이다 (양손 동작을 말하던 설명이 한 손 동작을 말하게 된다) → 원문·번역을 지운다.

    정답: 설명을 만들 때 현재였던 주인 가지(오른손·왼손 사슬의 처음 레코드) 중 왼손 가지에 현재
    후손이 없다 → 부분 삭제 → 검수 전 설명(ko·en) 삭제, 승인 없음.
    """
    right = _action("r")
    left = _action("l", hand="left")
    approved = right.model_copy(
        update={"verification": _checked(VerificationState.HUMAN_APPROVED, "r1")}
    )
    history = [approved, left, _deleted(left, "l-gone"), _desc(), _desc("den", source="d")]
    plan = plan_description_follow(history, ())
    assert (plan.approve, _ids(plan.retract)) == ([], ["d", "den"])


def test_baseline_is_the_stage_run_that_made_the_description() -> None:
    """단계 재실행처럼 이전 주인을 지우고 새 주인과 새 설명을 함께 만든 경우는 부분 삭제가 아니다.

    정답: 새 설명(actions-2)의 비교 기준은 같은 실행이 만든 새 행동 하나다. 그 행동이 승인되면
    설명도 승인한다. 이전 행동(actions-1, 지움)은 기준이 아니다.
    """
    old = _action("old")
    old_gone = _deleted(old, "old-gone").model_copy(
        update={"provenance": Provenance(source=Source.MODEL, model_version="actions-2"),
                "verification": Verification()}
    )  # fmt: skip
    v2 = Provenance(source=Source.MODEL, model_version="actions-2")
    new = _action("new", record={"created_at": T1, "provenance": v2,
                                 "verification": _checked(VerificationState.HUMAN_APPROVED, "r1",
                                                          T2)})  # fmt: skip
    d2 = _desc("d2", created_at=T1, provenance=v2)
    plan = plan_description_follow([old, old_gone, new, d2], ())
    assert plan.retract == []
    assert plan.approve == [
        InheritedReview("d2", VerificationState.HUMAN_APPROVED, "r1", T2, "new", ("new",))
    ]


# -------------------------------- 규칙 6: 물려받은 검수만 있는 설명
#
# 2026-10-15 사용자 결정 ("추천안대로"): 물려받은 승인은 사람이 설명문을 읽은 것이 아니다. 물려받은
# 뒤 다음 검수 회차에서 주인이 다시 바뀌면 그 설명(ko·en)도 지운다. "물려받음"은 명시 기록
# (`DescriptionInheritance`)으로 정한다 (2026-10-15 사용자 결정 "추천대로", F-PAR-20).
# 정답 근거: 회차 1에 검수자 r1이 T1에 행동을 승인했고 설명 d·den이 그 검수(r1, T1)를 물려받았다
# (기록 있음). 회차 2에 검수자 r2가 T2에 행동을 고치거나 지운다.

T3 = FIXED_TIME + timedelta(hours=3)


def _inherited(x: LabelRecord, state: VerificationState = VerificationState.HUMAN_APPROVED,
               who: str = "r1", at: Any = T1) -> LabelRecord:  # fmt: skip
    """검수 상태(apply_description_follow가 record_review로 쓰는 값)를 가진 같은 레코드."""
    return x.model_copy(update={"verification": _checked(state, who, at)})


def _rec(
    d: LabelRecord,
    giver: str = "act",
    event_id: int = 1,
    mechanism: InheritanceMechanism = InheritanceMechanism.COLLECT,
) -> DescriptionInheritance:
    """d의 지금 검수 상태를 준 물려받기 기록 (apply_description_follow가 같은 트랜잭션에 쓴다)."""
    v = d.verification
    assert v.reviewer_id is not None and v.reviewed_at is not None
    return DescriptionInheritance(
        event_id=event_id, session_id=d.session_id, description_label_id=d.label_id,
        owner_label_ids=(giver,), giver_label_id=giver, state=v.state, reviewer_id=v.reviewer_id,
        reviewed_at=v.reviewed_at, mechanism=mechanism, rule_version=FOLLOW_VERSION,
        recorded_at=v.reviewed_at,
    )  # fmt: skip


def _round2(parent: LabelRecord, label_id: str, **payload: Any) -> LabelRecord:
    """회차 2에 검수자 r2가 T2에 고친 레코드."""
    fixed = _fix(parent, label_id, **payload)
    return fixed.model_copy(
        update={"verification": _checked(VerificationState.HUMAN_CORRECTED, "r2", T2),
                "created_at": T2}
    )  # fmt: skip


def _round1() -> tuple[LabelRecord, list[LabelRecord], list[DescriptionInheritance]]:
    """회차 1 뒤: 승인된 행동 act, 물려받은 설명 d·den과 그 물려받기 기록."""
    act = _action(record={"verification": _checked(VerificationState.HUMAN_APPROVED, "r1")})
    descs = [_inherited(_desc()), _inherited(_desc("den", source="d"))]
    return act, descs, [_rec(d, event_id=i + 1) for i, d in enumerate(descs)]


def test_inherited_description_is_retracted_when_owner_changes_again() -> None:
    """물려받은 뒤 행동의 동사를 다시 고치면 원문·번역을 지운다 (승인을 다시 주지 않는다).
    그대로 두거나 시각만 고치면 아무것도 하지 않는다 (멱등)."""
    act, descs, recs = _round1()
    assert all(inheritance_only(d, recs) for d in descs)
    unchanged = plan_description_follow([act, *descs], recs)
    assert (unchanged.approve, unchanged.retract) == ([], [])
    timed = plan_description_follow([act, _round2(act, "t2", t_end_ms=1000), *descs], recs)
    assert (timed.approve, timed.retract) == ([], [])
    changed = plan_description_follow([act, _round2(act, "v2", verb="lift"), *descs], recs)
    assert (changed.approve, _ids(changed.retract)) == ([], ["d", "den"])


def test_inherited_description_follows_owner_deletion_and_bimanual_partial_delete() -> None:
    """물려받은 뒤 행동을 지우거나, 양손 행동에서 한 손만 지우면 원문·번역을 지운다."""
    act, descs, recs = _round1()
    gone = plan_description_follow([act, _deleted(act, "gone", T2), *descs], recs)
    assert _ids(gone.retract) == ["d", "den"]
    ok = _checked(VerificationState.HUMAN_APPROVED, "r1")
    right = _action("r", record={"verification": ok})
    left = _action("l", hand="left", record={"verification": ok})
    both = [right, left, *descs]
    assert plan_description_follow(both, recs).retract == []
    partial = plan_description_follow([*both, _deleted(left, "l-gone", T2)], recs)
    assert (partial.approve, _ids(partial.retract)) == ([], ["d", "den"])


def test_sample_verified_inherited_description_follows_owner_change() -> None:
    """표본 합격으로 물려받은 설명(sample_verified, 검수자 `sampling:<배정>`)도 같다."""
    sampled = _checked(VerificationState.SAMPLE_VERIFIED, "sampling:a")
    act = _action(record={"verification": sampled})
    d = _inherited(_desc(), VerificationState.SAMPLE_VERIFIED, "sampling:a")
    recs = [_rec(d)]
    assert inheritance_only(d, recs)
    plan = plan_description_follow([act, _round2(act, "v2", target_id="bucket_01"), d], recs)
    assert _ids(plan.retract) == ["d"]


def test_direct_review_and_human_text_stay_protected() -> None:
    """물려받기 기록이 없으면 보호한다: 직접 검수(lead), 사람이 쓴·고친 설명, 사람 수정 상태."""
    act, _, _ = _round1()
    direct = _inherited(_desc("dl"), who="lead")
    human = _inherited(_desc("dh", provenance=Provenance(source=Source.HUMAN), confidence=None))
    corrected = _inherited(_desc("dc"), VerificationState.HUMAN_CORRECTED)
    for x in (direct, human, corrected):
        assert not inheritance_only(x, []), x.label_id
        plan = plan_description_follow([act, _round2(act, "v2", verb="lift"), x], ())
        assert plan.retract == [], x.label_id
    # 사람 설명은 기록이 있어도(잘못 쓴 기록) 물려받은 것이 아니다
    assert not inheritance_only(human, [_rec(human)])


def test_direct_review_with_same_reviewer_and_time_is_not_inheritance() -> None:
    """F-PAR-20 (Codex 2차 #2): 검수자가 같은 배치에서 행동과 모델 설명을 직접 승인하면(검수자·
    시각이 행동과 같다) 물려받은 것이 아니다. 다음 회차에 행동의 의미를 고쳐도 설명은 보호된다.

    정답: 기록이 없다 → inheritance_only 거짓, 계획은 지우지 않는다. 같은 상태를 물려받은 경우
    (기록 있음)는 지운다 (대조).
    """
    act, _, _ = _round1()
    d = _inherited(_desc())  # r1, T1 = 행동 검수와 같다 (직접 검수)
    history = [act, _round2(act, "v2", verb="lift"), d]
    assert not inheritance_only(d, [])
    assert plan_description_follow(history, ()).retract == []
    assert _ids(plan_description_follow(history, [_rec(d)]).retract) == ["d"]


def test_rereviewing_the_giving_owner_keeps_inheritance() -> None:
    """F-PAR-19 남은 한계를 없앰: 물려준 행동 레코드를 회차 3에 같은 레코드로 다시 승인해(시각 T3)
    시각이 기록과 달라져도 물려받음은 그대로다. 그 뒤 의미를 고치면 설명을 지운다.

    정답: 기록(r1, T1, 준 주인 act)이 설명의 지금 상태와 같다 → 물려받음. 주인 시각은 보지 않는다.
    """
    act, descs, recs = _round1()
    rereviewed = _inherited(act, at=T3)
    assert all(inheritance_only(d, recs) for d in descs)
    plan = plan_description_follow(
        [rereviewed, _round2(rereviewed, "v2", verb="lift"), *descs], recs
    )
    assert _ids(plan.retract) == ["d", "den"]


def test_later_direct_review_ends_inheritance() -> None:
    """물려받은 뒤 검수자가 설명을 직접 다시 검수하면(상태·검수자·시각이 기록과 다름) 더는 물려받은
    것이 아니다 → 보호. 기록이 여럿이면 마지막 기록만 본다."""
    act, descs, recs = _round1()
    d = _inherited(descs[0], who="lead", at=T2)
    assert not inheritance_only(d, recs)
    plan = plan_description_follow([act, _round2(act, "v2", verb="lift"), d], recs)
    assert plan.retract == []
    again = _rec(d, event_id=9, mechanism=InheritanceMechanism.VERIFY)
    assert active_inheritance(d, [*recs, again]) == again


def test_translation_of_human_description_compares_owners_at_writing() -> None:
    """F-PAR-22 (Codex 2차 #3): 모델 grasp → 사람이 lift로 고침 → 사람이 한국어 설명을 씀 → 그
    번역은 쓴 시각의 주인(lift)과 비교한다. 처음 모델 행동(grasp)과 비교해 지우지 않는다.

    정답: 사람 설명은 T1(고친 뒤)에 썼다. 기준 = 그때 현재 주인 fix(lift) = 지금 주인 → 의미 같음 →
    번역은 남고, 행동(사람 수정)이 확인됐으므로 번역이 물려받는다. 사람 설명 뒤에 다시 고치면
    지운다.
    """
    act = _action()
    fix = _fix(act, "fix", verb="lift").model_copy(update={"created_at": T1})
    checked = _checked(VerificationState.HUMAN_CORRECTED, "r1")
    human = _desc("dh", provenance=Provenance(source=Source.HUMAN), confidence=None,
                  created_at=T1, verification=checked)  # fmt: skip
    en = _desc("den", source="dh", created_at=T2)
    plan = plan_description_follow([act, fix, human, en], ())
    assert plan.retract == []
    assert _ids([x for x in [en] if x.label_id in {r.label_id for r in plan.approve}]) == ["den"]
    again = _round2(fix, "v3", verb="wipe").model_copy(update={"created_at": T3})
    assert _ids(plan_description_follow([act, fix, human, en, again], ()).retract) == ["den"]
