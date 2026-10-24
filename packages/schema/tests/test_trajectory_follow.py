"""3D 궤적이 그것을 계산한 2D 트랙의 검수를 따른다 (F-COE-19·21·22, ADR 0048·0050).

서비스 없이 도는 순수 계산(`plan_trajectory_follow`)과 시간대 보존 SQLite의 적용
(`apply_trajectory_follow`) 시험이다.

정답 근거 (손으로 만든 장면, 세션 s001, 스트림 bodycam)
- g1: 오른손 hand21 2D 트랙 (모델). c1: g1에서 계산한 카메라 좌표 손목 궤적 (derived_from=g1).
  pose: 카메라 자세. w1: (c1, pose)에서 계산한 월드 좌표 손목 궤적.
- g1을 승인(rev01, T_REVIEW)하면 c1·w1이 human_approved·rev01·max(T_REVIEW, 3D 생성 시각)을
  물려받는다. 표본 검증이면 sample_verified.
- g1을 사람이 고치면(g2, parent g1) c1은 낡은 3D다 (출처가 현재가 아님): 물려받지 않는다. g2에서
  다시 만든 c2는 물려받는다. 이미 물려받은 c1은 끊김이고 궤적 따르기가 지운다 (ADR 0050).
- 직접 검수(기록 없음)·기록 뒤 다시 검수한 3D·사람 출처 3D는 건드리지 않고 끊김으로도 보지 않는다.
- 박스 중심 3D(part 없음, derived_from=박스 트랙)는 박스 트랙 검수를 같은 방식으로 물려받는다.
- 대체됨 (ADR 0051): 물려받기만 한 c1과 같은 출처 g1에서 다른 판(m-2, 캘리브레이션 등록 등)으로
  새로 계산한 검수 전 c1b가 같은 대상(스트림, 개체, 부위, 좌표계)에 있으면 c1을 지우고 c1b가 같은
  검수를 물려받는다. 직접 검수·다른 출처·같은 판은 대체하지 않는다.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta
from typing import Any

import pytest
import sqlalchemy as sa
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
from dlp_schema.episode import current_labels, retractions
from dlp_schema.label_validation import ProtectedLabelRetractionError
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
    assert set(got) == {"c1", "w1"} and plan.retract == []
    assert got["c1"].state is VerificationState.HUMAN_APPROVED
    assert (got["c1"].reviewer_id, got["c1"].reviewed_at) == ("rev01", T_REVIEW)
    assert (got["c1"].source_label_id, got["c1"].via_label_ids) == ("g1", ())
    assert (got["w1"].source_label_id, got["w1"].via_label_ids) == ("g1", ("c1", "pose1"))
    rec = got["w1"].record("s001", TrajectoryInheritanceMechanism.PRELABEL, T_REVIEW)
    assert rec.rule_version == TRAJECTORY_FOLLOW_VERSION and rec.trajectory_label_id == "w1"


def test_wearer_arm_3d_inherits_the_arm_track_like_hand_3d() -> None:
    """ADR 0058: 착용자 팔꿈치 3D(`left_arm`, elbow, derived_from = arm3 트랙)는 그 arm3 트랙 검수를
    손 3D처럼 물려받는다. 개체가 다른 2D(손 트랙)를 가리키면 출처가 아니다."""
    arm = {"kind": "keypoint_track", "entity_id": "left_arm", "skeleton": "arm3", "hand": "left",
           "keyframes": [{"t_ms": t, "points": [{"x": 1.0, "y": 2.0, "visibility": 2}] * 3}
                         for t in (0, 1_000)]}  # fmt: skip
    a1 = _reviewed(make_label(arm, label_id="a1", **MODEL), VerificationState.HUMAN_APPROVED)
    e1 = make_label(
        trajectory_payload("left_arm", "elbow", derived_from=("a1",)), label_id="e1", **MODEL
    )
    (r,) = plan_trajectory_follow([a1, e1], []).approve
    assert (r.label_id, r.source_label_id, r.state) == (
        "e1",
        "a1",
        VerificationState.HUMAN_APPROVED,
    )
    g1 = _reviewed(_track(), VerificationState.HUMAN_APPROVED)
    wrong = make_label(
        trajectory_payload("left_arm", "elbow", derived_from=("g1",)), label_id="e2", **MODEL
    )
    assert plan_trajectory_follow([g1, wrong], []).approve == []


def test_sample_verified_hand_track_gives_sample_verified_and_time_never_precedes_3d() -> None:
    """표본 검증된 2D → sample_verified. 3D가 2D 검수 뒤에 생겼으면 검수 시각 = 3D 생성 시각."""
    g1 = _reviewed(_track(), VerificationState.SAMPLE_VERIFIED)
    late = FIXED_TIME + timedelta(hours=2)
    c1 = _cam("c1", "g1", created_at=late)
    (r,) = plan_trajectory_follow([g1, c1], []).approve
    assert r.state is VerificationState.SAMPLE_VERIFIED and r.reviewed_at == late


def test_nothing_but_box_center_inherits_without_a_reviewed_current_hand_track() -> None:
    """2D 검수 전, derived_from 없음(예전 3D), 사람 3D, 자세는 물려받지 않는다. 승인된 박스
    트랙에서 계산한 박스 중심 3D(cb)만 물려받는다 (ADR 0050)."""
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
    assert _approved(history, []) == {"cb"}


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


def test_inherited_3d_is_retracted_when_its_source_changes() -> None:
    """물려받은 3D는 출처 2D가 고쳐지면 끊김이고 지운다 (ADR 0050). 직접 검수 3D는 끊기지 않는다."""
    g1 = _reviewed(_track(), VerificationState.HUMAN_APPROVED)
    history, records = _apply([g1, _cam("c1", "g1"), _pose(), _world("w1", "c1")], [])
    direct = _reviewed(_cam("cd", "g1"), VerificationState.HUMAN_APPROVED)  # 기록 없는 검수
    history.append(direct)
    assert plan_trajectory_follow(history, records).retract == []
    history.append(_corrected(g1))
    plan = plan_trajectory_follow(history, records)
    assert {x.label_id for x in plan.retract} == {"c1", "w1"} and plan.approve == []
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
    assert [x.label_id for x in plan.retract] == ["w1"]
    assert {r.label_id for r in plan.approve} == {"w2"}  # w3은 지워진 자세에서 나왔다


M2: dict[str, Any] = {**MODEL, "provenance": Provenance(source=Source.MODEL, model_version="m-2")}


def test_inherited_3d_is_superseded_by_a_new_version_from_the_same_2d() -> None:
    """물려받기만 한 c1은 같은 g1에서 다른 판으로 새로 계산한 c1b가 대신한다 (규칙 5, ADR 0051).

    계획 한 번에 c1을 지우고 c1b가 같은 2D 검수를 물려받으며, 적용 뒤 다시 계산하면 비어 있다.
    """
    g1 = _reviewed(_track(), VerificationState.HUMAN_APPROVED)
    history, records = _apply([g1, _cam("c1", "g1")], [])
    newer = _cam("c1b", "g1", created_at=LATE, **M2)
    plan = plan_trajectory_follow([*history, newer], records)
    assert [x.label_id for x in plan.retract] == ["c1"]
    assert [(r.label_id, r.source_label_id) for r in plan.approve] == [("c1b", "g1")]
    assert lapsed_trajectories([*history, newer], records) == {"c1"}
    gone = retractions(plan.retract, TRAJECTORY_FOLLOW_VERSION, LATE)
    after, records = _apply([*history, newer, *gone], records)
    again = plan_trajectory_follow(after, records)
    assert again.retract == [] and again.approve == []


def test_supersede_needs_inheritance_only_same_source_and_another_version() -> None:
    """대체하지 않는 경우: 직접 검수(기록 없음)·사람 출처, 다른 2D에서 계산한 새 궤적, 같은 판의
    사본, 이미 검수된 새 궤적, 더 먼저 만든 궤적 (규칙 4·5)."""
    g1 = _reviewed(_track(), VerificationState.HUMAN_APPROVED)
    g9 = _reviewed(_track("g9"), VerificationState.HUMAN_APPROVED)
    history, records = _apply([g1, g9, _cam("c1", "g1")], [])
    direct = _reviewed(_cam("cd", "g1", **M2), VerificationState.HUMAN_APPROVED)
    human = make_label(trajectory_payload(derived_from=("g1",)), label_id="h3", stream_id="bodycam")
    for extra in (
        [_cam("c9", "g9", created_at=LATE, **M2)],  # 다른 출처 2D
        [_cam("cs", "g1", created_at=LATE)],  # 같은 판 (m-1)
        [_cam("ce", "g1", created_at=FIXED_TIME - timedelta(hours=1), **M2)],  # 더 먼저 만듦
    ):
        assert plan_trajectory_follow([*history, *extra], records).retract == [], extra
    # 직접 검수·사람 출처 3D 옆에 새 판이 와도 그것들은 지우지 않는다 (c1만 대체된다)
    newer = _cam("c1b", "g1", created_at=LATE, **M2)
    plan = plan_trajectory_follow([*history, direct, human, newer], records)
    assert [x.label_id for x in plan.retract] == ["c1"]


def test_world_3d_inherits_a_directly_reviewed_camera_trajectory() -> None:
    """F-PAR-58: 카메라 좌표 궤적 자체를 사람이 확인했으면(직접 검수·검수자 있는 사람 출처) 2D가
    검수 전이거나 없어도 그 월드 좌표 궤적이 카메라 궤적의 검수를 물려받는다 (출처 = 카메라 궤적,
    경유 = 자세). 예전에는 출처가 없어 월드 좌표가 검수 전으로 남아 완료 판정을 영구히 막았다.
    카메라 궤적이 다시 바뀌면(지워짐) 끊긴다."""
    direct = _reviewed(_cam("cd", "g1"), VerificationState.HUMAN_APPROVED)  # 2D g1은 검수 전
    human = make_label(
        trajectory_payload(part="index_tip"), label_id="hc", stream_id="bodycam",
        verification=Verification(
            state=VerificationState.HUMAN_APPROVED, reviewer_id="lead", reviewed_at=T_REVIEW
        ),
    )  # fmt: skip
    wh = make_label(
        trajectory_payload(part="index_tip", frame="world", derived_from=("hc", "pose1")),
        label_id="wh", **MODEL,
    )  # fmt: skip
    history = [_track(), direct, human, _pose(), _world("wd", "cd"), wh]
    plan = plan_trajectory_follow(history, [])
    got = {r.label_id: r for r in plan.approve}
    assert set(got) == {"wd", "wh"}
    assert (got["wd"].source_label_id, got["wd"].via_label_ids) == ("cd", ("pose1",))
    assert (got["wh"].source_label_id, got["wh"].reviewer_id) == ("hc", "lead")
    applied, records = _apply(history, [])
    gone = make_label(direct.payload, label_id="cd-x", parent_label_id="cd", retracted=True, **M2)
    assert lapsed_trajectories([*applied, gone], records) == {"wd"}


def test_follow_is_idempotent() -> None:
    """적용한 뒤 다시 계산하면 물려받을 것도 끊긴 것도 없다."""
    g1 = _reviewed(_track(), VerificationState.SAMPLE_VERIFIED)
    history, records = _apply([g1, _cam("c1", "g1"), _pose(), _world("w1", "c1")], [])
    plan = plan_trajectory_follow(history, records)
    assert plan.approve == [] and plan.retract == [] and len(records) == 2
    names = {x.label_id: review_basis(x, records) for x in history}
    assert names == {
        "g1": "direct", "c1": "inherited_from_2d", "pose1": "automatic",
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
    # 계약 설명이 출처를 손 트랙으로만 적지 않는다 (박스 트랙 출처, ADR 0050, F-COE-34)
    described = TrajectoryInheritance.model_fields["source_label_id"].description or ""
    assert "박스 트랙" in described and "hand21" in described


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


def _box(label_id: str = "b1", **extra: Any) -> LabelRecord:
    """걸레(rag_01) 박스 트랙 (모델 출처 기본)."""
    payload = {
        "kind": "box_track", "entity_id": "rag_01", "class_id": "rag",
        "keyframes": [{"t_ms": 0, "x": 1, "y": 1, "w": 5, "h": 5}],
    }  # fmt: skip
    return make_label(payload, label_id=label_id, **{**MODEL, **extra})


def _box3d(label_id: str, source: str, frame: str = "camera", **extra: Any) -> LabelRecord:
    """박스 중심 3D (카메라 좌표: derived_from=(박스 트랙), 월드 좌표: (카메라 좌표 궤적, 자세))."""
    derived = (source,) if frame == "camera" else (source, "pose1")
    payload = trajectory_payload("rag_01", None, frame=frame, derived_from=derived)
    return make_label(payload, label_id=label_id, **{**MODEL, **extra})


def test_box_center_3d_inherits_the_box_track_and_lapses_with_it() -> None:
    """박스 중심 3D(카메라·월드 좌표)가 승인한 박스 트랙의 검수를 물려받고(출처 b1, 월드는 경유
    (cb, pose1)), 박스 트랙을 고치면 둘 다 끊겨 지울 대상이 된다 (ADR 0050).

    박스 트랙이 아닌 2D(손 트랙)를 가리키는 박스 중심 3D와 개체가 다른 3D는 출처가 없다.
    """
    b1 = _reviewed(_box(), VerificationState.HUMAN_APPROVED)
    wrong_kind = make_label(
        trajectory_payload("right_hand", None, derived_from=("g1",)), label_id="cx", **MODEL
    )
    other = make_label(
        trajectory_payload("rag_02", None, derived_from=("b1",)), label_id="cy", **MODEL
    )
    g1 = _reviewed(_track(), VerificationState.HUMAN_APPROVED)
    history = [b1, g1, _box3d("cb", "b1"), _pose(), _box3d("wb", "cb", "world"), wrong_kind, other]
    plan = plan_trajectory_follow(history, [])
    got = {r.label_id: (r.source_label_id, r.via_label_ids) for r in plan.approve}
    assert got == {"cb": ("b1", ()), "wb": ("b1", ("cb", "pose1"))}
    history, records = _apply(history, [])
    fixed = make_label(
        _box().payload.model_dump(mode="json"), label_id="b2", stream_id="bodycam",
        parent_label_id="b1",
        verification=Verification(
            state=VerificationState.HUMAN_CORRECTED, reviewer_id="rev01", reviewed_at=T_REVIEW
        ),
    )  # fmt: skip
    assert lapsed_trajectories([*history, fixed], records) == {"cb", "wb"}


def test_storage_allows_retracting_a_superseded_inherited_3d_only_after_the_new_one(
    ontology: Ontology,
) -> None:
    """저장 검증: 같은 g1의 새 판 c1b를 저장하기 전에는 물려받은 c1을 지울 수 없고, 저장한 뒤에는
    궤적 따르기가 c1(과 그 월드 좌표 w1)을 지우며 c1b가 물려받는다. 직접 검수 cd·다시 검수한 cr은
    그대로다 (ADR 0051)."""
    engine = _db_with_inherited(ontology)
    with engine.begin() as conn:
        labels = {x.label_id: x for x in get_labels(conn, "s001")}
    with engine.begin() as conn, pytest.raises(ProtectedLabelRetractionError):
        insert_labels(conn, retractions([labels["c1"]], TRAJECTORY_FOLLOW_VERSION, LATE))
    later = LATE + timedelta(hours=1)
    with engine.begin() as conn:
        insert_labels(conn, [_cam("c1b", "g1", created_at=later, **M2)])
        plan = apply_trajectory_follow(conn, "s001", later, TrajectoryInheritanceMechanism.PRELABEL)
        again = apply_trajectory_follow(
            conn, "s001", later, TrajectoryInheritanceMechanism.PRELABEL
        )
        live = {x.label_id: x for x in current_labels(get_labels(conn, "s001"))}
    # c1을 경유한 물려받기만 한 월드 좌표 w1도 같은 적용에서 지운다 (다음 월드 단계가 다시 만든다)
    assert [x.label_id for x in plan.retract] == ["c1", "w1"]
    assert [r.label_id for r in plan.approve] == ["c1b"]
    assert again.retract == [] and again.approve == []
    assert {"c1b", "cd", "cr"} <= set(live) and not {"c1", "w1"} & set(live)
    assert live["c1b"].verification.state is VerificationState.HUMAN_APPROVED


def _db_with_inherited(ontology: Ontology) -> sa.Engine:
    """SQLite: g1 승인 → c1·w1이 물려받고, cd(직접 검수)·cr(물려받은 뒤 다시 검수)가 있는 세션."""
    engine = sqlite_db()
    with engine.begin() as conn:
        register_ontology(conn, ontology)
        insert_session(conn, make_session())
        labels = [_track(), _cam("c1", "g1"), _cam("cr", "g1"), _pose(), _world("w1", "c1")]
        insert_labels(conn, labels)
        record_review(conn, "g1", VerificationState.HUMAN_APPROVED, "rev01", T_REVIEW)
        apply_trajectory_follow(conn, "s001", T_REVIEW, TrajectoryInheritanceMechanism.COLLECT)
        # cr: 물려받은 뒤 검수자가 직접 다시 검수했다 (기록 뒤 상태가 바뀜 → 직접 검수로 보호)
        record_review(conn, "cr", VerificationState.HUMAN_CORRECTED, "lead", LATE)
        # cd: 기록 없는 직접 검수 (물려받은 적 없다)
        insert_labels(conn, [_cam("cd", "g1", created_at=LATE)])
        record_review(conn, "cd", VerificationState.HUMAN_APPROVED, "lead", LATE)
    return engine


LATE = T_REVIEW + timedelta(hours=1)


def test_apply_retracts_lapsed_inherited_3d_only_and_is_idempotent(ontology: Ontology) -> None:
    """출처 2D를 고치면 물려받기만 한 c1과 그 월드 좌표 w1을 궤적 따르기가 지운다. 직접 검수 cd와
    기록 뒤 다시 검수한 cr은 남는다. 다시 적용해도 그대로다 (F-COE-22, ADR 0050)."""
    engine = _db_with_inherited(ontology)
    with engine.begin() as conn:
        insert_labels(conn, [_corrected(_track())])
        first = apply_trajectory_follow(conn, "s001", LATE, TrajectoryInheritanceMechanism.VERIFY)
        again = apply_trajectory_follow(conn, "s001", LATE, TrajectoryInheritanceMechanism.VERIFY)
        history = get_labels(conn, "s001")
    assert {x.label_id for x in first.retract} == {"c1", "w1"}
    assert again.retract == [] and again.approve == []
    live = {x.label_id for x in current_labels(history)}
    assert {"cd", "cr", "pose1", "g2"} <= live and not {"c1", "w1"} & live
    gone = [x for x in history if x.retracted]
    assert sorted(x.parent_label_id or "" for x in gone) == ["c1", "w1"]
    assert all(x.provenance.model_version == TRAJECTORY_FOLLOW_VERSION for x in gone)
    assert all(x.created_at == LATE for x in gone)


def test_storage_exemption_is_narrow(ontology: Ontology) -> None:
    """저장 검증: 궤적 따르기 출처라도 직접 검수·다시 검수한 3D, 출처가 아직 성립하는 물려받은 3D는
    지울 수 없다. 다른 모델 버전은 끊긴 물려받은 3D도 지울 수 없다 (ADR 0050)."""
    engine = _db_with_inherited(ontology)
    with engine.begin() as conn:
        labels = {x.label_id: x for x in get_labels(conn, "s001")}
    # 출처가 그대로인 물려받은 c1
    with engine.begin() as conn, pytest.raises(ProtectedLabelRetractionError):
        insert_labels(conn, retractions([labels["c1"]], TRAJECTORY_FOLLOW_VERSION, LATE))
    with engine.begin() as conn:
        insert_labels(conn, [_corrected(_track())])
    for target, version in (("cd", TRAJECTORY_FOLLOW_VERSION), ("cr", TRAJECTORY_FOLLOW_VERSION),
                            ("c1", "depth3d-x"), ("g1", TRAJECTORY_FOLLOW_VERSION)):  # fmt: skip
        with engine.begin() as conn, pytest.raises(ProtectedLabelRetractionError):
            insert_labels(conn, retractions([labels[target]], version, LATE))
    with engine.begin() as conn:  # 끊긴 물려받은 c1·w1은 궤적 따르기가 지울 수 있다
        insert_labels(
            conn, retractions([labels["c1"], labels["w1"]], TRAJECTORY_FOLLOW_VERSION, LATE)
        )
