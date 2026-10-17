"""3D 손 궤적이 수집·완료 판정에서 2D 손 트랙 검수를 물려받는다 (F-COE-19, ADR 0048).

서비스 없이 시간대를 보존하는 SQLite와 가짜 Label Studio로 돈다 (설명 따르기 시험과 같은 틀).

정답 근거 (세션 s001, 바디캠)
- g1: 오른손 hand21 2D 트랙(모델), c1: g1에서 계산한 카메라 좌표 손목 3D(derived_from=g1).
- 수집: 2D 검수 결과(CVAT 수집 흉내로 g1 승인)가 이미 있으면 Label Studio 작업 수집 끝에 c1이
  human_approved·같은 검수자·max(2D 검수 시각, 3D 생성 시각)를 물려받고 기록 경로는 collect다.
- 완료 판정: g1이 표본 검증이면 c1도 sample_verified가 되어 완료를 막지 않는다. 카메라 자세처럼
  출처 2D가 없는 3D는 계속 막는다 (F-COE-21).
- 그 뒤 검수자가 g1을 고치면(g2) c1은 끊김으로 알려지고 지워지지 않는다 (검수 상태 그대로).
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


def test_verify_passes_sample_verification_and_reports_lapse_without_removing(
    engine: sa.Engine,
) -> None:
    """표본 검증된 2D → 3D sample_verified, 완료. 자세(출처 2D 없음)는 계속 막는다 (F-COE-21).
    그 뒤 2D를 고치면 물려받은 3D는 끊김으로 알리고 그대로 남는다 (지우지 않는다)."""
    with engine.begin() as conn:
        insert_session(conn, make_session(SID, privacy_state=PrivacyState.APPROVED))
        set_lifecycle(conn, SID, LifecycleState.PRIVACY_APPROVED)
        set_lifecycle(conn, SID, LifecycleState.PRELABELED)
        insert_labels(conn, _hand_and_3d())
        record_review(conn, "g1", VerificationState.SAMPLE_VERIFIED, "sampling:as1", LATER)
        pose = make_label(trajectory_payload("camera"), label_id="pose1", **SPATIAL)
        insert_labels(conn, [pose])
    with engine.begin() as conn:
        blocked = verify_session(conn, SID, LATER, "lead")
    assert blocked.trajectories_approved == 1 and not blocked.verified
    assert any("pose1" in r for r in blocked.reasons), blocked.reasons
    assert _current(engine)["c1"].verification.state is VerificationState.SAMPLE_VERIFIED

    fixed = make_label(
        hand_track_payload(times=(0, 500, 1_000)), label_id="g2", stream_id="bodycam",
        parent_label_id="g1",
        verification=Verification(
            state=VerificationState.HUMAN_CORRECTED, reviewer_id="lead", reviewed_at=LATER
        ),
    )  # fmt: skip
    with engine.begin() as conn:
        insert_labels(conn, [fixed])
        record_review(conn, "pose1", VerificationState.HUMAN_APPROVED, "lead", LATER)
        done = verify_session(conn, SID, LATER + timedelta(hours=1), "lead")
    assert done.verified, done.reasons
    assert (done.trajectories_approved, done.trajectories_lapsed) == (0, 1)
    c1 = _current(engine)["c1"]  # 끊겼어도 지우지 않는다 (검수 상태 그대로)
    assert c1.verification.state is VerificationState.SAMPLE_VERIFIED
