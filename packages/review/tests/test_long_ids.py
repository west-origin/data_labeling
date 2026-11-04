"""긴 세션 ID에서도 파생 배정·라벨 ID가 128자 안에 든다 (F-REV-36).

정답 근거: `Identifier`와 DB 열은 128자다. 오류 삽입 배정 ID는 원천·대상 두 단위 ID를, 사본 ID는
그 배정 ID를 담아 길어지므로 길이를 넘으면 앞부분 + 해시로 줄인다. 같은 입력은 같은 ID(멱등)이고,
길이 안의 ID는 예전 형식 그대로다 (기존 DB 행과 맞아야 한다).
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from dlp_review.ops.assign import PlannedUnit, assignment_id, plan, plan_qa
from dlp_review.ops.policy import ReviewOpsPolicy, load_policy
from dlp_review.ops.priority import Unit
from dlp_review.ops.seeding import (
    detected,
    seed_labels,
    seed_new_prefix,
    seed_prefix,
    seeded_owned,
)
from dlp_schema.labels import (
    BlurTrackPayload,
    BoxKeyframe,
    LabelRecord,
    Provenance,
    Source,
    Verification,
    VerificationState,
)
from dlp_schema.ontology import Ontology, load_ontology
from dlp_schema.review import AssignmentStatus, ReviewAssignment, ReviewMode
from dlp_schema.testing import FIXED_TIME, make_label

ROOT = Path(__file__).resolve().parents[3]
LONG_SESSION = "pilot-2026-10-22-cleaning-w03-site02-0001-camera-a"


@pytest.fixture(scope="module")
def policy() -> ReviewOpsPolicy:
    """저장소 정책에서 오류 삽입 비율만 1로 올린 것 (모든 단위가 오류 삽입 과제를 받는다)."""
    pol = load_policy(ROOT)
    ratios = pol.ratios.model_copy(update={"seeded_error_task_ratio": 1.0})
    return pol.model_copy(update={"ratios": ratios})


@pytest.fixture(scope="module")
def ontology() -> Ontology:
    """온톨로지 v1."""
    return load_ontology(ROOT / "config/ontology/v1")


def _unit(session_id: str) -> Unit:
    return Unit(session_id, "bodycam", "spatial", ("box_track",))


def _seeded(policy: ReviewOpsPolicy, targets: list[str], source: str) -> list[ReviewAssignment]:
    planned = [PlannedUnit(_unit(t), (), 1.0, generation="g1234abcd") for t in targets]
    out = plan(planned, ["r1", "r2"], policy, seed=0, now=FIXED_TIME, seed_pool=[_unit(source)])
    return [a for a in out if a.mode is ReviewMode.SEEDED_ERROR]


def test_seeded_assignment_ids_fit_and_stay_distinct_f_rev_36(policy: ReviewOpsPolicy) -> None:
    """F-REV-36: 긴 세션 둘의 오류 삽입 배정 ID가 128자 안이고 계약 검증을 통과한다.

    정답: 같은 원천을 고른 대상 단위 둘은 다른 ID, 다시 계획하면 같은 ID.
    """
    targets = [f"{LONG_SESSION}-1", f"{LONG_SESSION}-2"]
    first = _seeded(policy, targets, f"golden-{LONG_SESSION}")
    assert len(first) == 2
    ids = [a.assignment_id for a in first]
    assert all(len(i) <= 128 for i in ids), ids
    assert len(set(ids)) == 2
    assert [a.assignment_id for a in _seeded(policy, targets, f"golden-{LONG_SESSION}")] == ids
    for a in first:
        ReviewAssignment.model_validate(a.model_dump())


def test_short_ids_keep_the_documented_format_f_rev_36() -> None:
    """F-REV-36: 길이 안의 ID는 `<단위>:<방식>[:<tag>]`, `seed-<배정>-` 그대로다."""
    unit = _unit("s1")
    assert assignment_id(unit, ReviewMode.SEEDED_ERROR, tag="s2.bodycam.spatial.g1") == (
        "s1:bodycam:spatial:seeded_error:s2.bodycam.spatial.g1"
    )
    assert seed_prefix("s1:x") == seed_new_prefix("s1:x") == "seed-s1:x-"


def test_seed_copies_of_long_assignment_fit_and_are_owned_f_rev_36(
    policy: ReviewOpsPolicy, ontology: Ontology
) -> None:
    """F-REV-36: 긴 배정의 사본 ID가 128자 안이고 계약 검증을 통과하며 그 배정의 것으로 가려진다."""
    [a] = _seeded(policy, [LONG_SESSION], f"golden-{LONG_SESSION}")
    truth = make_label(
        {"kind": "box_track", "entity_id": "cup_1", "class_id": "cup",
         "keyframes": [{"t_ms": 0, "x": 1, "y": 1, "w": 5, "h": 5}]},
        label_id="g-box-1", session_id="gold", stream_id="bodycam",
        provenance=Provenance(source=Source.HUMAN),
    )  # fmt: skip
    task = seed_labels(
        [truth], assignment_id=a.assignment_id, ontology=ontology, policy=policy.seeding,
        seed=0, now=FIXED_TIME,
    )  # fmt: skip
    for x in task.labels:
        assert len(x.label_id) <= 128
        LabelRecord.model_validate(x.model_dump())
        assert seeded_owned(x.label_id, a.assignment_id)
        assert not seeded_owned(x.label_id, f"{a.assignment_id}x")


def test_blur_drawn_in_long_assignment_counts_as_detection_f_rev_36(
    policy: ReviewOpsPolicy, ontology: Ontology
) -> None:
    """F-REV-36: 사본은 들어가지만 새로 그린 레코드는 넘는 길이(110자)의 배정에서도 블러 삭제
    발견을 그 배정의 새 레코드(`seed_new_prefix`)로 인정한다. 다른 배정 접두사로 그린 것은 인정하지
    않는다."""
    aid = "a" * 110
    frames = tuple(BoxKeyframe(t_ms=t, x=1, y=1, w=5, h=5) for t in (0, 500))
    truth = make_label(
        BlurTrackPayload(target="face", keyframes=frames),
        label_id="gold-blur0", session_id="gold", t_end_ms=500, stream_id="bodycam",
    )  # fmt: skip
    seeding = policy.seeding.model_copy(update={"errors_per_task": 1, "types": ("blur_deletion",)})
    task = seed_labels(
        [truth], assignment_id=aid, ontology=ontology, policy=seeding, seed=0, now=FIXED_TIME
    )
    [err] = task.injected
    assert all(x.label_id.startswith(f"seed-{aid}-") for x in task.labels)
    human = Verification(
        state=VerificationState.HUMAN_CORRECTED, reviewer_id="r1", reviewed_at=FIXED_TIME
    )

    def drawn(label_id: str) -> LabelRecord:
        return truth.model_copy(
            update={"label_id": label_id, "provenance": Provenance(source=Source.HUMAN),
                    "seeded_error": True, "verification": human}
        )  # fmt: skip

    new_id = f"{seed_new_prefix(aid)}new-{'0' * 16}"
    assert len(new_id) <= 128
    other = drawn(f"{seed_new_prefix('b' * 110)}new-{'1' * 16}")
    assert not detected(err, [truth, *task.labels, other], "r1", 200, 0.5)
    assert detected(err, [truth, *task.labels, drawn(new_id)], "r1", 200, 0.5)


def test_qa_id_of_long_standard_assignment_fits_f_rev_36(policy: ReviewOpsPolicy) -> None:
    """F-REV-36: 126자 표준 배정의 QA 배정 ID(model_copy, 검증 없음)도 128자 안이다."""
    done = ReviewAssignment(
        assignment_id="s" * 126, session_id="s1", stream_id="bodycam", label_kinds=("box_track",),
        mode=ReviewMode.STANDARD, priority=1.0, assignee="r1",
        status=AssignmentStatus.DONE, created_at=FIXED_TIME,
    )  # fmt: skip
    qa_policy = policy.model_copy(
        update={"ratios": policy.ratios.model_copy(update={"qa_sample_ratio": 1.0})}
    )
    [qa] = plan_qa([done], qa_policy, seed=0, now=FIXED_TIME + timedelta(days=1))
    assert len(qa.assignment_id) <= 128
    assert qa.assignment_id.endswith(":qa")
    ReviewAssignment.model_validate(qa.model_dump())
