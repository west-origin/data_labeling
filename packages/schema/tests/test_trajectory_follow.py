"""3D 손 궤적이 그것을 계산한 2D 손 트랙의 검수를 따른다 (F-COE-19, ADR 0048).

서비스 없이 도는 순수 계산(`plan_trajectory_follow`)과 시간대 보존 SQLite의 적용
(`apply_trajectory_follow`) 시험이다.

정답 근거 (손으로 만든 장면, 세션 s001, 스트림 bodycam)
- g1: 오른손 hand21 2D 트랙 (모델). c1: g1에서 계산한 카메라 좌표 손목 궤적 (derived_from=g1).
  pose: 카메라 자세. w1: (c1, pose)에서 계산한 월드 좌표 손목 궤적.
- g1을 승인(rev01, T_REVIEW)하면 c1·w1이 human_approved·rev01·max(T_REVIEW, 3D 생성 시각)을
  물려받는다. 표본 검증이면 sample_verified.
- g1을 사람이 고치면(g2, parent g1) c1은 낡은 3D다 (출처가 현재가 아님): 물려받지 않는다. g2에서
  다시 만든 c2는 물려받는다. 이미 물려받은 c1은 끊김(lapsed)이고 지우지 않는다.
- 직접 검수(기록 없음)·사람 출처 3D는 건드리지 않고 끊김으로도 보지 않는다.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta
from typing import Any

import pytest
from pydantic import ValidationError

from dlp_schema.db.inheritance_repository import get_trajectory_inheritance
from dlp_schema.db.repository import (
    get_labels,
    insert_labels,
    insert_session,
    record_review,
    register_ontology,
)
from dlp_schema.db.trajectory_repository import apply_trajectory_follow
from dlp_schema.labels import LabelRecord, Provenance, Source, Verification, VerificationState
from dlp_schema.ontology import Ontology
from dlp_schema.review import TrajectoryInheritance, TrajectoryInheritanceMechanism
from dlp_schema.testing import (
    FIXED_TIME,
    hand_track_payload,
    make_label,
    make_session,
    sqlite_db,
    trajectory_payload,
)
from dlp_schema.trajectory_follow import (
    TRAJECTORY_FOLLOW_VERSION,
    lapsed_trajectories,
    plan_trajectory_follow,
    review_basis,
)

# 2D 검수 시각 (3D 생성 시각 FIXED_TIME보다 늦다: 물려받은 검수 시각 = 2D 검수 시각)
T_REVIEW = FIXED_TIME + timedelta(hours=1)
MODEL: dict[str, Any] = {
    "provenance": Provenance(source=Source.MODEL, model_version="m-1"),
    "confidence": 0.8,
    "stream_id": "bodycam",
}


def _track(label_id: str = "g1", **extra: Any) -> LabelRecord:
    """오른손 2D 트랙 (모델 출처 기본)."""
    return make_label(hand_track_payload(), label_id=label_id, **{**MODEL, **extra})


def _cam(label_id: str, source: str | None, **extra: Any) -> LabelRecord:
    """2D 트랙 source에서 계산한 카메라 좌표 손목 궤적 (source None이면 derived_from 없음)."""
    derived = None if source is None else (source,)
    return make_label(
        trajectory_payload(derived_from=derived, offset_m=float(len(label_id))),
        label_id=label_id,
        **{**MODEL, **extra},
    )


def _pose(label_id: str = "pose1", **extra: Any) -> LabelRecord:
    """카메라 자세 궤적."""
    return make_label(trajectory_payload("camera"), label_id=label_id, **{**MODEL, **extra})


def _world(label_id: str, cam: str, pose: str = "pose1") -> LabelRecord:
    """(cam, pose)에서 계산한 월드 좌표 손목 궤적."""
    payload = trajectory_payload(frame="world", derived_from=(cam, pose))
    return make_label(payload, label_id=label_id, **MODEL)


def _reviewed(x: LabelRecord, state: VerificationState, who: str = "rev01") -> LabelRecord:
    """검수 상태를 바꾼 사본 (`record_review` 흉내)."""
    v = Verification(state=state, reviewer_id=who, reviewed_at=T_REVIEW)
    return x.model_copy(update={"verification": v})


def _corrected(g: LabelRecord, label_id: str = "g2") -> LabelRecord:
    """검수자가 고친 2D 트랙 (사람 출처, parent g, human_corrected)."""
    return make_label(
        hand_track_payload(times=(0, 500, 1_000)),
        label_id=label_id,
        stream_id="bodycam",
        parent_label_id=g.label_id,
        verification=Verification(
            state=VerificationState.HUMAN_CORRECTED, reviewer_id="rev01", reviewed_at=T_REVIEW
        ),
    )


def _apply(
    history: list[LabelRecord], inheritance: list[TrajectoryInheritance]
) -> tuple[list[LabelRecord], list[TrajectoryInheritance]]:
    """계획을 이력·기록에 적용한 결과 (DB 적용 흉내, 기록 event_id는 순번)."""
    plan = plan_trajectory_follow(history, inheritance)
    by_id = {r.label_id: r for r in plan.approve}
    out: list[LabelRecord] = []
    for x in history:
        r = by_id.get(x.label_id)
        if r is not None:
            v = Verification(state=r.state, reviewer_id=r.reviewer_id, reviewed_at=r.reviewed_at)
            x = x.model_copy(update={"verification": v})
        out.append(x)
    records = list(inheritance)
    for r in plan.approve:
        rec = r.record("s001", TrajectoryInheritanceMechanism.COLLECT, T_REVIEW)
        records.append(rec.model_copy(update={"event_id": len(records) + 1}))
    return out, records


def _approved(history: Sequence[LabelRecord], inheritance: list[TrajectoryInheritance]) -> set[str]:
    return {r.label_id for r in plan_trajectory_follow(history, inheritance).approve}


def test_camera_and_world_3d_inherit_the_approved_hand_track() -> None:
    """승인한 2D → 카메라·월드 좌표 3D가 상태·검수자·시각을 물려받는다 (자세는 물려받지 않는다)."""
    g1 = _reviewed(_track(), VerificationState.HUMAN_APPROVED)
    history = [g1, _cam("c1", "g1"), _pose(), _world("w1", "c1")]
    plan = plan_trajectory_follow(history, [])
    got = {r.label_id: r for r in plan.approve}
    assert set(got) == {"c1", "w1"} and plan.lapsed == []
    assert got["c1"].state is VerificationState.HUMAN_APPROVED
    assert (got["c1"].reviewer_id, got["c1"].reviewed_at) == ("rev01", T_REVIEW)
    assert (got["c1"].source_label_id, got["c1"].via_label_ids) == ("g1", ())
    assert (got["w1"].source_label_id, got["w1"].via_label_ids) == ("g1", ("c1", "pose1"))
    rec = got["w1"].record("s001", TrajectoryInheritanceMechanism.PRELABEL, T_REVIEW)
    assert rec.rule_version == TRAJECTORY_FOLLOW_VERSION and rec.trajectory_label_id == "w1"


def test_sample_verified_hand_track_gives_sample_verified_and_time_never_precedes_3d() -> None:
    """표본 검증된 2D → sample_verified. 3D가 2D 검수 뒤에 생겼으면 검수 시각 = 3D 생성 시각."""
    g1 = _reviewed(_track(), VerificationState.SAMPLE_VERIFIED)
    late = FIXED_TIME + timedelta(hours=2)
    c1 = _cam("c1", "g1", created_at=late)
    (r,) = plan_trajectory_follow([g1, c1], []).approve
    assert r.state is VerificationState.SAMPLE_VERIFIED and r.reviewed_at == late


def test_nothing_inherits_without_a_reviewed_current_hand_track() -> None:
    """2D 검수 전, derived_from 없음(예전 3D), 사람 3D, 박스·자세는 물려받지 않는다."""
    box = make_label(
        {"kind": "box_track", "entity_id": "rag_01", "class_id": "rag",
         "keyframes": [{"t_ms": 0, "x": 1, "y": 1, "w": 5, "h": 5}]},
        label_id="b1", **MODEL,
    )  # fmt: skip
    box3d = make_label(
        trajectory_payload("rag_01", None, derived_from=("b1",)), label_id="cb", **MODEL
    )
    human3d = make_label(
        trajectory_payload(derived_from=("g9",)), label_id="h3", stream_id="bodycam"
    )
    approved = _reviewed(_track("g9"), VerificationState.HUMAN_APPROVED)
    history = [_track(), _cam("c1", "g1"), approved, _cam("c0", None), human3d]
    history += [_reviewed(box, VerificationState.HUMAN_APPROVED), box3d, _pose()]
    assert _approved(history, []) == set()


def test_corrected_track_leaves_stale_3d_and_regenerated_3d_inherits() -> None:
    """2D를 고치면 예전 2D로 만든 3D는 물려받지 않고, 고친 2D로 다시 만든 3D가 물려받는다."""
    g1 = _track()
    g2 = _corrected(g1)
    stale = _cam("c1", "g1")
    assert _approved([g1, g2, stale], []) == set()
    fresh = _cam("c2", "g2")
    (r,) = plan_trajectory_follow([g1, g2, stale, fresh], []).approve
    assert (r.label_id, r.source_label_id, r.state) == (
        "c2",
        "g2",
        VerificationState.HUMAN_APPROVED,
    )


def test_inherited_3d_lapses_when_its_source_changes_but_is_never_removed() -> None:
    """물려받은 3D는 출처 2D가 고쳐지면 끊김이다 (지우지 않는다). 직접 검수 3D는 끊기지 않는다."""
    g1 = _reviewed(_track(), VerificationState.HUMAN_APPROVED)
    history, records = _apply([g1, _cam("c1", "g1"), _pose(), _world("w1", "c1")], [])
    direct = _reviewed(_cam("cd", "g1"), VerificationState.HUMAN_APPROVED)  # 기록 없는 검수
    history.append(direct)
    assert plan_trajectory_follow(history, records).lapsed == []
    history.append(_corrected(g1))
    plan = plan_trajectory_follow(history, records)
    assert {x.label_id for x in plan.lapsed} == {"c1", "w1"} and plan.approve == []
    assert lapsed_trajectories(history, records) == {"c1", "w1"}
    # 기록 뒤 직접 다시 검수해 상태가 바뀐 3D는 물려받은 것이 아니다 (보호)
    redone = [
        _reviewed(x, VerificationState.HUMAN_CORRECTED, "lead") if x.label_id == "c1" else x
        for x in history
    ]
    assert lapsed_trajectories(redone, records) == {"w1"}


def test_world_3d_lapses_when_the_camera_pose_changes() -> None:
    """월드 좌표 3D는 자세가 바뀌면(새 자세 레코드가 이전 것을 대신) 끊기고 물려받지 않는다."""
    g1 = _reviewed(_track(), VerificationState.HUMAN_APPROVED)
    pose = _pose()
    history, records = _apply([g1, _cam("c1", "g1"), pose, _world("w1", "c1")], [])
    rerun: dict[str, Any] = {
        **MODEL, "provenance": Provenance(source=Source.MODEL, model_version="m-2")
    }  # fmt: skip
    gone = make_label(
        pose.payload, label_id="pose1-x", parent_label_id="pose1", retracted=True, **rerun
    )
    history += [gone, _pose("pose2"), _world("w2", "c1", "pose2"), _world("w3", "c1")]
    plan = plan_trajectory_follow(history, records)
    assert [x.label_id for x in plan.lapsed] == ["w1"]
    assert {r.label_id for r in plan.approve} == {"w2"}  # w3은 지워진 자세에서 나왔다


def test_follow_is_idempotent() -> None:
    """적용한 뒤 다시 계산하면 물려받을 것도 끊긴 것도 없다."""
    g1 = _reviewed(_track(), VerificationState.SAMPLE_VERIFIED)
    history, records = _apply([g1, _cam("c1", "g1"), _pose(), _world("w1", "c1")], [])
    plan = plan_trajectory_follow(history, records)
    assert plan.approve == [] and plan.lapsed == [] and len(records) == 2
    names = {x.label_id: review_basis(x, records) for x in history}
    assert names == {
        "g1": "direct", "c1": "inherited_from_2d", "pose1": "unreviewed",
        "w1": "inherited_from_2d",
    }  # fmt: skip


def test_trajectory_inheritance_contract_rejects_bad_records() -> None:
    """물려받는 상태는 승인·표본 검증뿐이고, 자기 자신에게서 물려받을 수 없다."""
    base: dict[str, Any] = {
        "event_id": 1, "session_id": "s001", "trajectory_label_id": "c1",
        "source_label_id": "g1", "state": "human_approved", "reviewer_id": "rev01",
        "reviewed_at": T_REVIEW, "mechanism": "verify", "rule_version": TRAJECTORY_FOLLOW_VERSION,
        "recorded_at": T_REVIEW,
    }  # fmt: skip
    TrajectoryInheritance.model_validate(base)
    with pytest.raises(ValidationError, match="human_approved"):
        TrajectoryInheritance.model_validate({**base, "state": "human_corrected"})
    with pytest.raises(ValidationError, match="자기 자신"):
        TrajectoryInheritance.model_validate({**base, "via_label_ids": ["c1"]})


def test_apply_writes_review_and_record_in_one_step_and_is_idempotent(ontology: Ontology) -> None:
    """SQLite: 적용하면 3D 검수 상태와 물려받기 기록이 함께 생기고, 다시 적용해도 그대로다."""
    engine = sqlite_db()
    with engine.begin() as conn:
        register_ontology(conn, ontology)
        insert_session(conn, make_session())
        insert_labels(conn, [_track(), _cam("c1", "g1"), _pose(), _world("w1", "c1")])
        first = apply_trajectory_follow(
            conn, "s001", T_REVIEW, TrajectoryInheritanceMechanism.VERIFY
        )
        assert first.approve == []  # 2D 검수 전
        record_review(conn, "g1", VerificationState.HUMAN_APPROVED, "rev01", T_REVIEW)
        second = apply_trajectory_follow(
            conn, "s001", T_REVIEW, TrajectoryInheritanceMechanism.VERIFY
        )
        third = apply_trajectory_follow(
            conn, "s001", T_REVIEW, TrajectoryInheritanceMechanism.VERIFY
        )
        labels = {x.label_id: x for x in get_labels(conn, "s001")}
        records = get_trajectory_inheritance(conn, "s001")
    assert {r.label_id for r in second.approve} == {"c1", "w1"} and third.approve == []
    assert labels["c1"].verification.state is VerificationState.HUMAN_APPROVED
    assert labels["pose1"].verification.state is VerificationState.UNREVIEWED
    assert [(r.event_id, r.trajectory_label_id, r.via_label_ids) for r in records] == [
        (1, "c1", ()),
        (2, "w1", ("c1", "pose1")),
    ]
    assert all(r.mechanism is TrajectoryInheritanceMechanism.VERIFY for r in records)
