"""수집 후 배정 마무리·표본 판정·검수 품질 보고 (WP12, ADR 0014).

작업 계획·외부 도구 생성은 runner에 둔다. 기존 runner 진입점은 재수출로 유지한다.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

import sqlalchemy as sa

from dlp_review.ops.measure import Agreement, DetectionRate, agreement, as_items, prelabel_bias
from dlp_review.ops.policy import ReviewOpsPolicy
from dlp_review.ops.sampling import judge
from dlp_review.ops.seeding import detected
from dlp_schema.db.repository import (
    get_assignment,
    get_labels,
    insert_assignment,
    list_assignments,
    list_review_tasks,
    record_review,
    update_assignment,
)
from dlp_schema.episode import current_labels
from dlp_schema.labels import LabelRecord, Source, VerificationState
from dlp_schema.review import (
    AssignmentStatus,
    ReviewAssignment,
    ReviewMode,
    ReviewReason,
    ReviewTask,
    ReviewTaskStatus,
)


@dataclass
class FinishResult:
    """배정 마무리 결과."""

    # 표본 판정 결과 (표본이 없거나 아직 마무리 전이면 None)
    sample_accepted: bool | None = None
    # 표본 합격으로 sample_verified 기록한 라벨 수
    sample_verified: int = 0
    # 표본 불합격으로 만든 재검수 배정 ID
    resample_assignment: str | None = None


def finish_assignment(
    conn: sa.Connection,
    a: ReviewAssignment,
    now: datetime,
    policy: ReviewOpsPolicy | None = None,
) -> FinishResult:
    """배정의 검수 작업이 모두 수집되면 마무리한다 (한 번만).

    표본이 있으면 묶음 합격 판정을 적용한다.

    policy가 없으면 저장소의 정책을 읽는다 (CLI·웹훅 서버는 넘겨 준다).

    부작용: 배정 상태 done 갱신, 합격이면 나머지 라벨에 `sample_verified` 기록(검수자 ID
    `sampling:<배정>`), 불합격이면 재검수 배정(`<배정>:resample`, 보류 라벨만, 우선순위 +1) 삽입.
    배정의 작업이 하나도 없으면 아무것도 하지 않는다 (done이 아니다).
    """
    tasks = [t for t in list_review_tasks(conn, a.session_id) if t.assignment_id == a.assignment_id]
    if not tasks:
        # 회귀: any([])가 거짓이라 작업이 없는(아직 `dlp review assign` 전인) 배정이 사람 검수 없이
        # done이 되고, 표본 배정이면 판정까지 돌았다. 작업을 만들고 수집해야만 마무리한다.
        return FinishResult()
    if any(t.status is not ReviewTaskStatus.COLLECTED for t in tasks):
        return FinishResult()  # 아직 남은 작업이 있다
    # 인자 a는 오래된 사본일 수 있어 DB에서 상태를 다시 읽는다
    if get_assignment(conn, a.assignment_id).status is AssignmentStatus.DONE:
        return FinishResult()  # 이미 마무리했다
    update_assignment(conn, a.assignment_id, status=AssignmentStatus.DONE, completed_at=now)
    result = FinishResult()
    if not a.sample_label_ids:
        return result
    if policy is None:
        from dlp_review.ops.policy import load_policy
        from dlp_schema import repo_root

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
        # 불합격: 보류했던 라벨만 다시 보는 재검수 배정 (표본 사유 구간은 뺀다)
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
    """검수 품질 리포트 (`dlp review quality`)."""

    # 검수자별 오류 삽입 발견율 (검수자 이름순)
    detection: list[DetectionRate] = field(default_factory=list[DetectionRate])
    double: dict[str, Agreement] = field(default_factory=dict[str, Agreement])  # 배정 → 일치도
    blind_bias: dict[str, float] = field(default_factory=dict[str, float])  # 배정 → 프리라벨 편향


def shown_prelabels(
    unit: Sequence[LabelRecord], pair_tasks: Sequence[ReviewTask]
) -> list[LabelRecord]:
    """표준(짝) 배정의 검수자에게 실제로 보낸 모델 프리라벨.

    회귀: 프리라벨 편향을 이력 전체의 모델 레코드(지워진 레코드·이전 모델 버전 포함)로 재서,
    검수자가 본 적 없는 프리라벨까지 기준에 들어가 편향이 틀렸다.

    작업마다:
    - `sent_label_ids`가 있으면 그 ID의 레코드 (작업에 실제로 보낸 라벨).
    - 없으면(예전 작업) 작업 created_at 시점의 운영 현재 라벨 (그 시각까지 만든 레코드로
      `current_labels`: 그 뒤의 삭제·수정 레코드는 아직 없던 것으로 본다).
    둘 다 모델 출처이고 측정·오류 삽입 레코드가 아닌 것만 남긴다.

    Args:
        unit: 측정 단위(종류·스트림)의 전체 이력.
        pair_tasks: 짝 표준 배정의 검수 작업 (없으면 보낸 프리라벨도 없다).

    Returns:
        모델 프리라벨 (중복 없이, label_id 순).
    """
    by_id = {x.label_id: x for x in unit}
    shown: dict[str, LabelRecord] = {}
    for t in pair_tasks:
        if t.sent_label_ids is not None:
            # 단위 밖(다른 종류·스트림) ID는 by_id에 없어 빠진다
            picked = [by_id[i] for i in t.sent_label_ids if i in by_id]
        else:
            picked = current_labels([x for x in unit if x.created_at <= t.created_at])
        for x in picked:
            if x.provenance.source is Source.MODEL and x.measurement is None and not x.seeded_error:
                shown[x.label_id] = x
    return [shown[i] for i in sorted(shown)]


def quality_report(conn: sa.Connection, policy: ReviewOpsPolicy) -> QualityReport:
    """끝난(done) 배정 전체로 검수 품질을 잰다. 읽기만 한다.

    - 오류 삽입 배정: 넣은 오류마다 `seeding.detected`로 발견 여부를 담당자별로 센다.
    - 이중 배정: 짝 표준 배정이 끝났으면, 단위의 현재 운영 라벨과 이 배정의 `double` 측정 레코드의
      일치도(`measure.agreement`).
    - 블라인드 배정: 짝 표준 배정이 끝났으면 프리라벨 편향(`measure.prelabel_bias`).
      표준 결과 = 현재 운영 라벨, 블라인드 결과 = `blind` 측정 레코드, 모델 프리라벨 = 짝 표준
      배정 작업에 실제로 보낸 모델 라벨(`shown_prelabels`).
    정책 값: `measurement.tolerance_ms`·`match_iou`, `seeding.detect_tolerance_ms`·`blur_overlap`.
    """
    mp = policy.measurement
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
                found[who] += detected(
                    err, labels, a.assignee, policy.seeding.detect_tolerance_ms,
                    policy.seeding.blur_overlap,
                )  # fmt: skip
            continue
        # 측정 배정이고 짝(표준 배정)이 끝났을 때만 비교한다
        if a.mode not in (ReviewMode.BLIND, ReviewMode.DOUBLE) or a.pair_id not in by_id:
            continue
        kinds = set(a.label_kinds)
        unit = [
            x
            for x in labels
            if x.kind in kinds and (a.stream_id is None or x.stream_id == a.stream_id)
        ]
        operational = current_labels(unit)
        # 이 배정 방식의 측정 레코드 중 담당자가 남긴 것 (검수자 ID가 없는 레코드도 받는다)
        measured = [
            x for x in unit
            if x.measurement == a.mode.value and x.verification.reviewer_id in (None, a.assignee)
        ]  # fmt: skip
        if a.mode is ReviewMode.DOUBLE:
            report.double[a.assignment_id] = agreement(
                as_items(operational), as_items(measured), mp.tolerance_ms, mp.match_iou
            )
        else:
            # 모델 프리라벨: 짝 표준 배정의 검수자가 실제로 본 것 (이전 버전·지워진 레코드 제외)
            pair_tasks = [
                t for t in list_review_tasks(conn, a.session_id) if t.assignment_id == a.pair_id
            ]
            model = shown_prelabels(unit, pair_tasks)
            report.blind_bias[a.assignment_id] = prelabel_bias(
                as_items(model),
                as_items(operational),
                as_items(measured),
                mp.tolerance_ms,
                mp.match_iou,
            )
    report.detection = [DetectionRate(r, total[r], found[r]) for r in sorted(total)]
    return report
