"""3D 궤적이 수집·완료 판정에서 2D 트랙 검수를 따른다 (F-COE-19·21·22, ADR 0048·0050).

서비스 없이 시간대를 보존하는 SQLite와 가짜 Label Studio로 돈다 (설명 따르기 시험과 같은 틀).

정답 근거 (세션 s001, 바디캠)
- g1: 오른손 hand21 2D 트랙(모델), c1: g1에서 계산한 카메라 좌표 손목 3D(derived_from=g1).
- 수집: 2D 검수 결과(CVAT 수집 흉내로 g1 승인)가 이미 있으면 Label Studio 작업 수집 끝에 c1이
  human_approved·같은 검수자·max(2D 검수 시각, 3D 생성 시각)를 물려받고 기록 경로는 collect다.
- 완료 판정: g1이 표본 검증이면 c1도 sample_verified가 되어 완료를 막지 않는다. 박스 중심 3D는
  박스 트랙 검수를 물려받고, 카메라 자세는 자동 라벨이라 (review.yaml verify.automatic 기본)
  완료를 막지 않는다 (F-COE-21, ADR 0050).
- 그 뒤 검수자가 g1을 고치면(g2) 물려받기만 한 c1(과 그 월드 좌표 w1)은 궤적 따르기가 지운다
  (F-COE-22, ADR 0050). 직접 검수 3D는 남는다.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa

from dlp_review.collect import collect_task
from dlp_review.labelstudio import to_ls_results
from dlp_review.verify import verify_session
from dlp_schema import load_ontology, repo_root
from dlp_schema.db.repository import (
    get_labels,
    get_trajectory_inheritance,
    insert_labels,
    insert_review_task,
    insert_session,
    record_review,
    register_ontology,
    set_lifecycle,
)
from dlp_schema.episode import current_labels
from dlp_schema.labels import LabelRecord, Provenance, Source, Verification, VerificationState
from dlp_schema.review import TrajectoryInheritanceMechanism
from dlp_schema.session import LifecycleState, PrivacyState
from dlp_schema.testing import (
    action_payload,
    hand_track_payload,
    make_label,
    make_session,
    sqlite_db,
    trajectory_payload,
)
from dlp_schema.trajectory_follow import TRAJECTORY_FOLLOW_VERSION

from .test_description_follow import (
    ASSIGNEE,
    LATER,
    SID,
    TASK,
    _setup,  # pyright: ignore[reportPrivateUsage]
    _task,  # pyright: ignore[reportPrivateUsage]
)

MODEL: dict[str, Any] = {
    "provenance": Provenance(source=Source.MODEL, model_version="m-1"),
    "confidence": 0.8,
}
SPATIAL: dict[str, Any] = {**MODEL, "stream_id": "bodycam"}


@pytest.fixture
def engine() -> Iterator[sa.Engine]:
    """표와 온톨로지 v1을 만든 메모리 SQLite."""
    eng = sqlite_db()
    with eng.begin() as conn:
        register_ontology(conn, load_ontology(repo_root() / "config/ontology/v1"))
    yield eng
    eng.dispose()


def _hand_and_3d() -> list[LabelRecord]:
    """[g1 2D 손 트랙, c1 카메라 좌표 손목 3D]."""
    return [
        make_label(hand_track_payload(), label_id="g1", **SPATIAL),
        make_label(trajectory_payload(derived_from=("g1",)), label_id="c1", **SPATIAL),
    ]


def _current(engine: sa.Engine) -> dict[str, LabelRecord]:
    with engine.connect() as conn:
        return {x.label_id: x for x in current_labels(get_labels(conn, SID))}


def test_collect_passes_the_2d_review_to_its_3d(engine: sa.Engine, tmp_path: Path) -> None:
    """수집 끝에 3D가 이미 승인된 2D의 검수를 물려받는다 (경로 collect, 멱등)."""
    action = make_label(action_payload(), label_id="s001-a1", **MODEL)
    with engine.begin() as conn:
        insert_session(conn, make_session(SID, privacy_state=PrivacyState.APPROVED))
        insert_labels(conn, [action, *_hand_and_3d()])
        # 2D 손 트랙은 CVAT 작업 수집이 승인했다 (그 수집 흉내)
        record_review(conn, "g1", VerificationState.HUMAN_APPROVED, ASSIGNEE, LATER)
        insert_review_task(conn, _task([action]))
    with engine.begin() as conn:
        setup = _setup(tmp_path, to_ls_results([action]))
        outcome = collect_task(conn, TASK, setup, ASSIGNEE, LATER + timedelta(minutes=5))
    assert outcome is not None and outcome.trajectories_approved == ["c1"]
    v = _current(engine)["c1"].verification
    assert (v.state, v.reviewer_id, v.reviewed_at) == (
        VerificationState.HUMAN_APPROVED, ASSIGNEE, LATER,
    )  # fmt: skip
    with engine.connect() as conn:
        (rec,) = get_trajectory_inheritance(conn, SID)
    assert rec.mechanism is TrajectoryInheritanceMechanism.COLLECT
    assert (rec.source_label_id, rec.trajectory_label_id) == ("g1", "c1")


def _depth_and_pose() -> list[LabelRecord]:
    """깊이·자세 단계를 거친 세션: 손(g1→c1→w1), 박스(b1→cb→wb), 카메라 자세 pose1."""
    box = {
        "kind": "box_track", "entity_id": "rag_01", "class_id": "rag",
        "keyframes": [{"t_ms": 0, "x": 1, "y": 1, "w": 5, "h": 5}],
    }  # fmt: skip
    box3d = trajectory_payload("rag_01", None, derived_from=("b1",))
    world = trajectory_payload(frame="world", derived_from=("c1", "pose1"))
    return [
        *_hand_and_3d(),
        make_label(box, label_id="b1", **SPATIAL),
        make_label(box3d, label_id="cb", **SPATIAL),
        make_label(trajectory_payload("camera"), label_id="pose1", **SPATIAL),
        make_label(world, label_id="w1", **SPATIAL),
        make_label(
            trajectory_payload("rag_01", None, frame="world", derived_from=("cb", "pose1")),
            label_id="wb", **SPATIAL,
        ),
    ]  # fmt: skip


def _prelabeled(conn: sa.Connection, labels: list[LabelRecord]) -> None:
    insert_session(conn, make_session(SID, privacy_state=PrivacyState.APPROVED))
    set_lifecycle(conn, SID, LifecycleState.PRIVACY_APPROVED)
    set_lifecycle(conn, SID, LifecycleState.PRELABELED)
    insert_labels(conn, labels)


def test_session_with_depth_and_pose_is_verified_after_2d_approvals(engine: sa.Engine) -> None:
    """F-COE-21 (ADR 0050): 손 트랙(표본 검증)·박스 트랙(승인)만 검수하면 손·박스 3D(카메라·월드
    좌표)가 물려받고, 카메라 자세는 자동 라벨(review.yaml verify.automatic)이라 막지 않아 완료된다.
    정책에서 camera_pose를 빼면 자세가 다시 완료를 막는다. 자세 자체의 검수 상태는 그대로다."""
    with engine.begin() as conn:
        _prelabeled(conn, _depth_and_pose())
        record_review(conn, "g1", VerificationState.SAMPLE_VERIFIED, "sampling:as1", LATER)
        record_review(conn, "b1", VerificationState.HUMAN_APPROVED, "rev01", LATER)
    with engine.begin() as conn:
        strict = verify_session(conn, SID, LATER, "lead", automatic=())
    assert strict.trajectories_approved == 4 and not strict.verified
    assert any("pose1" in r for r in strict.reasons), strict.reasons
    now = _current(engine)
    assert now["c1"].verification.state is VerificationState.SAMPLE_VERIFIED
    assert now["wb"].verification.state is VerificationState.HUMAN_APPROVED
    with engine.begin() as conn:
        done = verify_session(conn, SID, LATER + timedelta(hours=1), "lead")
    assert done.verified, done.reasons
    assert done.automatic_unreviewed == 1 and done.trajectories_approved == 0
    assert _current(engine)["pose1"].verification.state is VerificationState.UNREVIEWED


def test_verify_retracts_lapsed_inherited_3d_and_keeps_direct_reviews(engine: sa.Engine) -> None:
    """F-COE-22 (ADR 0050): 물려받은 뒤 출처 2D를 고치면 완료 판정이 물려받기만 한 c1과 그 월드
    좌표 w1을 지운다 (궤적 따르기 삭제 레코드). 박스 쪽(cb·wb)은 출처가 그대로라 남고, 직접 검수한
    3D(cd)도 남는다. 다시 판정해도 더 지우지 않는다."""
    blocker = make_label(hand_track_payload(), label_id="g9", **SPATIAL)
    direct = make_label(trajectory_payload(derived_from=("g1",)), label_id="cd", **SPATIAL)
    with engine.begin() as conn:
        _prelabeled(conn, [*_depth_and_pose(), blocker, direct])
        record_review(conn, "g1", VerificationState.HUMAN_APPROVED, "rev01", LATER)
        record_review(conn, "b1", VerificationState.HUMAN_APPROVED, "rev01", LATER)
        record_review(conn, "cd", VerificationState.HUMAN_APPROVED, "lead", LATER)
    with engine.begin() as conn:
        blocked = verify_session(conn, SID, LATER, "lead")
    assert not blocked.verified and blocked.trajectories_approved == 4
    fixed = make_label(
        hand_track_payload(times=(0, 500, 1_000)), label_id="g2", stream_id="bodycam",
        parent_label_id="g1",
        verification=Verification(
            state=VerificationState.HUMAN_CORRECTED, reviewer_id="lead", reviewed_at=LATER
        ),
    )  # fmt: skip
    with engine.begin() as conn:
        insert_labels(conn, [fixed])
        record_review(conn, "g9", VerificationState.HUMAN_APPROVED, "lead", LATER)
        done = verify_session(conn, SID, LATER + timedelta(hours=1), "lead")
    assert done.verified, done.reasons
    assert (done.trajectories_approved, done.trajectories_retracted) == (0, 2)
    live = _current(engine)
    assert {"cb", "wb", "cd", "pose1"} <= set(live) and not {"c1", "w1"} & set(live)
    with engine.connect() as conn:
        gone = [x for x in get_labels(conn, SID) if x.retracted]
    assert {x.parent_label_id for x in gone} == {"c1", "w1"}
    assert {x.provenance.model_version for x in gone} == {TRAJECTORY_FOLLOW_VERSION}
