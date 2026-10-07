"""검수 운영 실행: 세션 배정 계획, 배정별 검수 작업 만들기, 배정 마무리(표본 판정), 품질 리포트.

- `dlp review plan <세션> --reviewer …`: 배정(review_assignments)을 만든다 (같은 배정 ID는 건너뜀).
- `dlp review assign <배정>`: 검수 도구 작업을 만든다.
- 결과 수집(`dlp review collect`)이 끝나면 배정을 마무리한다. 표본 검수 배정이면 묶음 합격 여부에
  따라 나머지를 표본 검증으로 두거나 재검수 배정을 만든다.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

import sqlalchemy as sa

from dlp_review.ops.assign import Loads, PlannedUnit, plan
from dlp_review.ops.measure import Agreement, DetectionRate, agreement, as_items, prelabel_bias
from dlp_review.ops.policy import ReviewOpsPolicy
from dlp_review.ops.priority import Unit, flag_unit, unit_priority, units_for
from dlp_review.ops.sampling import draw_sample, judge, lots
from dlp_review.ops.seeding import detected, seed_labels
from dlp_review.ops.selection import assignment_selector
from dlp_review.tasks import ReviewSetup, create_labeling_tasks, create_privacy_tasks
from dlp_schema.db.repository import (
    get_labels,
    get_session,
    insert_assignment,
    insert_labels,
    list_assignments,
    list_session_ids,
    record_review,
    update_assignment,
)
from dlp_schema.episode import current_labels
from dlp_schema.labels import (
    BoxTrackPayload,
    LabelRecord,
    MaskTrackPayload,
    Source,
    VerificationState,
)
from dlp_schema.ontology import Ontology
from dlp_schema.review import (
    AssignmentStatus,
    FlaggedSpan,
    ReviewAssignment,
    ReviewMode,
    ReviewReason,
    ReviewTask,
    ReviewTool,
)
from dlp_schema.session import StreamKind

VERIFIED = {VerificationState.HUMAN_APPROVED, VerificationState.HUMAN_CORRECTED}


def is_verified(x: LabelRecord) -> bool:
    return x.provenance.source is Source.HUMAN or x.verification.state in VERIFIED


def known_classes(conn: sa.Connection, exclude_session: str) -> set[str]:
    """다른 세션에서 사람이 확인한 객체 클래스."""
    out: set[str] = set()
    for sid in list_session_ids(conn):
        if sid == exclude_session:
            continue
        for x in current_labels(get_labels(conn, sid, kinds=["box_track", "mask_track"])):
            if is_verified(x) and isinstance(x.payload, BoxTrackPayload | MaskTrackPayload):
                out.add(x.payload.class_id)
    return out


def seed_pool(
    conn: sa.Connection,
    session_ids: Sequence[str],
    policy: ReviewOpsPolicy,
    groups: set[str] | None = None,
) -> list[Unit]:
    """정답을 아는 단위: 운영 라벨이 있고 모두 사람이 만들었거나 승인·수정한 단위.

    groups로 단위 묶음(spatial, temporal, privacy)을 좁힌다.
    """
    pool: list[Unit] = []
    for sid in session_ids:
        session = get_session(conn, sid)
        current = current_labels(get_labels(conn, sid))
        for unit in [*units_for(session, policy), *units_for(session, policy, privacy=True)]:
            labels = unit.select(current)
            if groups is not None and unit.group not in groups:
                continue
            if labels and all(is_verified(x) for x in labels):
                pool.append(unit)
    return pool


def _open_loads(conn: sa.Connection, reviewers: Sequence[str]) -> Loads:
    counts = Counter(
        a.assignee for a in list_assignments(conn, status=AssignmentStatus.OPEN) if a.assignee
    )
    return Loads(reviewers, counts)


def plan_session(
    conn: sa.Connection,
    session_id: str,
    reviewers: Sequence[str],
    policy: ReviewOpsPolicy,
    ontology: Ontology,
    *,
    seed: int,
    now: datetime,
    seed_sessions: Sequence[str] = (),
    seed_groups: set[str] | None = None,
    privacy: bool = False,
) -> list[ReviewAssignment]:
    session = get_session(conn, session_id)
    current = current_labels(get_labels(conn, session_id))
    known = known_classes(conn, session_id)
    glove = any(s.kind in (StreamKind.GLOVE_LEFT, StreamKind.GLOVE_RIGHT) for s in session.streams)
    planned: list[PlannedUnit] = []
    for unit in units_for(session, policy, privacy=privacy):
        labels = unit.select(current)
        if not labels:
            continue
        flagged = flag_unit(labels, policy, known_classes=known, glove_session=glove)
        sample: tuple[str, ...] = ()
        withheld: tuple[str, ...] = ()
        if not privacy:  # 블러는 종료 조건 전까지 전수 검수한다
            by_id = {x.label_id: x for x in labels}
            for lot in lots(labels, policy.sampling):
                picked = draw_sample(lot, policy.sampling, seed)
                sample += picked
                withheld += tuple(i for i in lot.label_ids if i not in picked)
                flagged += [
                    FlaggedSpan(
                        reason=ReviewReason.SAMPLE, t_start_ms=by_id[i].t_start_ms,
                        t_end_ms=by_id[i].t_end_ms, label_ids=(i,),
                    )
                    for i in picked
                ]  # fmt: skip
        planned.append(
            PlannedUnit(unit, tuple(flagged), unit_priority(flagged, policy), sample, withheld)
        )
    pool = seed_pool(conn, seed_sessions, policy, seed_groups) if seed_sessions else []
    existing = {a.assignment_id for a in list_assignments(conn)}
    created: list[ReviewAssignment] = []
    for a in plan(
        planned,
        reviewers,
        policy,
        seed=seed,
        now=now,
        seed_pool=pool,
        loads=_open_loads(conn, reviewers),
    ):
        if a.assignment_id in existing:
            continue
        if a.mode is ReviewMode.SEEDED_ERROR:
            a = _prepare_seeded(conn, a, policy, ontology, seed, now)
        insert_assignment(conn, a)
        created.append(a)
    return created


def _prepare_seeded(
    conn: sa.Connection, a: ReviewAssignment, policy: ReviewOpsPolicy, ontology: Ontology,
    seed: int, now: datetime,
) -> ReviewAssignment:  # fmt: skip
    group = (
        "privacy" if a.label_kinds == ("blur_track",) else "spatial" if a.stream_id else "temporal"
    )
    unit = Unit(a.session_id, a.stream_id, group, a.label_kinds)
    truth = unit.select(current_labels(get_labels(conn, a.session_id)))
    task = seed_labels(
        truth,
        assignment_id=a.assignment_id,
        ontology=ontology,
        policy=policy.seeding,
        seed=seed,
        now=now,
    )
    insert_labels(conn, task.labels)
    return a.model_copy(update={"injected": tuple(task.injected)})


def create_assignment_tasks(
    conn: sa.Connection, a: ReviewAssignment, setup: ReviewSetup, now: datetime
) -> list[ReviewTask]:
    select = assignment_selector(conn, a)
    assignee = a.assignee or "unassigned"
    if a.label_kinds == ("blur_track",):
        tasks = create_privacy_tasks(
            conn, a.session_id, setup, now, select=select, mode=a.mode,
            assignment_id=a.assignment_id, assignee=a.assignee,
            streams={a.stream_id} if a.stream_id else None,
        )  # fmt: skip
    else:
        tool = ReviewTool.LABEL_STUDIO if a.stream_id is None else ReviewTool.CVAT
        tasks = create_labeling_tasks(
            conn, a.session_id, setup, assignee, now, select=select, mode=a.mode,
            assignment_id=a.assignment_id, tools={tool},
            streams={a.stream_id} if a.stream_id else None,
        )  # fmt: skip
    if tasks:
        update_assignment(conn, a.assignment_id, task_key=tasks[0].task_key)
    return tasks


@dataclass
class FinishResult:
    sample_accepted: bool | None = None
    sample_verified: int = 0
    resample_assignment: str | None = None


def finish_assignment(conn: sa.Connection, a: ReviewAssignment, now: datetime) -> FinishResult:
    """수집이 끝난 배정을 마무리한다. 표본이 있으면 묶음 합격 판정을 적용한다."""
    from dlp_review.ops.policy import load_policy
    from dlp_schema import repo_root

    update_assignment(conn, a.assignment_id, status=AssignmentStatus.DONE, completed_at=now)
    result = FinishResult()
    if not a.sample_label_ids:
        return result
    policy = load_policy(repo_root())
    labels = get_labels(conn, a.session_id)
    verdict = judge(
        (*a.sample_label_ids, *a.withheld_label_ids), a.sample_label_ids, labels, policy.sampling
    )
    result.sample_accepted = verdict.accepted
    if verdict.accepted:
        for label_id in verdict.to_verify:
            record_review(
                conn,
                label_id,
                VerificationState.SAMPLE_VERIFIED,
                f"sampling:{a.assignment_id}",
                now,
            )
        result.sample_verified = len(verdict.to_verify)
    elif verdict.accepted is False and a.withheld_label_ids:
        follow = a.model_copy(
            update={
                "assignment_id": f"{a.assignment_id}:resample",
                "flagged": tuple(s for s in a.flagged if s.reason is not ReviewReason.SAMPLE),
                "sample_label_ids": (), "withheld_label_ids": (),
                "only_label_ids": a.withheld_label_ids, "task_key": None,
                "status": AssignmentStatus.OPEN, "created_at": now, "completed_at": None,
                "priority": a.priority + 1.0,
            }
        )  # fmt: skip
        insert_assignment(conn, follow)
        result.resample_assignment = follow.assignment_id
    return result


@dataclass
class QualityReport:
    detection: list[DetectionRate] = field(default_factory=list[DetectionRate])
    double: dict[str, Agreement] = field(default_factory=dict[str, Agreement])  # 배정 → 일치도
    blind_bias: dict[str, float] = field(default_factory=dict[str, float])  # 배정 → 프리라벨 편향


def quality_report(
    conn: sa.Connection, policy: ReviewOpsPolicy, tolerance_ms: int = 200
) -> QualityReport:
    report = QualityReport()
    done = list_assignments(conn, status=AssignmentStatus.DONE)
    by_id = {a.assignment_id: a for a in done}
    found: Counter[str] = Counter()
    total: Counter[str] = Counter()
    for a in done:
        labels = get_labels(conn, a.session_id)
        if a.mode is ReviewMode.SEEDED_ERROR:
            who = a.assignee or "unassigned"
            for err in a.injected:
                total[who] += 1
                found[who] += detected(err, labels, a.assignee, policy.seeding.detect_tolerance_ms)
            continue
        if a.mode not in (ReviewMode.BLIND, ReviewMode.DOUBLE) or a.pair_id not in by_id:
            continue
        kinds = set(a.label_kinds)
        unit = [
            x
            for x in labels
            if x.kind in kinds and (a.stream_id is None or x.stream_id == a.stream_id)
        ]
        operational = current_labels(unit)
        measured = [
            x for x in unit
            if x.measurement == a.mode.value and x.verification.reviewer_id in (None, a.assignee)
        ]  # fmt: skip
        if a.mode is ReviewMode.DOUBLE:
            report.double[a.assignment_id] = agreement(
                as_items(operational), as_items(measured), tolerance_ms
            )
        else:
            model = [
                x
                for x in unit
                if x.provenance.source is Source.MODEL
                and x.measurement is None
                and not x.seeded_error
            ]
            report.blind_bias[a.assignment_id] = prelabel_bias(
                as_items(model), as_items(operational), as_items(measured), tolerance_ms
            )
    report.detection = [DetectionRate(r, total[r], found[r]) for r in sorted(total)]
    return report
