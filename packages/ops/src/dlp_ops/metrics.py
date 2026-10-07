"""주간 운영 지표 (`dlp ops weekly`).

주는 ISO 주(월요일 0시 UTC부터 7일)다. 지표마다 그 주에 일어난 일만 센다.

- 검수 시간: 영상 1시간당 검수 분 (작업 라벨 검수·QA, 블러 검수는 따로). review_work 기록에서.
- 수정률: 그 주에 검수한 라벨 중 고침·지움·추가 비율 (개별 검수만, 블러 제외).
  골든셋 세션(정답을 사람이 처음부터 만들어 모두 "추가"로 보인다)과 사용 중지 세션은 뺀다
  (dlp_active와 같은 기준).
- 자동 승인율: 그 주에 검수한 모델 라벨 중 수정 없이 승인된 비율 (같은 세션 기준).
- 프리라벨 편향: 그 주에 끝난 블라인드 배정의 편향 평균 (ADR 0014).
- 오류 삽입 발견율: 그 주에 끝난 오류 삽입 배정에서 발견한 오류 비율.
- 잔여 블러 누락: 그 주 감사의 영상 1시간당 잔여 누락 수.
- 검증 에피소드: 검증 완료 시각이 그 주인 세션.
  검증 완료 시각 = 생애주기 기록(session_lifecycle_events)에서 human_verified로 처음 옮긴 시각
  (`dlp review verify`가 남긴다, ADR 0028). 한 번 정해지면 바뀌지 않아 지난 주의 수가 그대로다.
  그 기록이 없는 세션(0011 이전에 검증 단계를 지난 세션)만 라벨 이력에서 추정한다
  (verified_at: 운영 라벨 항목마다 처음 검수한 시각 중 가장 늦은 것, ADR 0027).
  블러(프라이버시 검수)는 따로 센다.
  생산원가 = 검수 시간 * 인건비 / 그 수.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any

import sqlalchemy as sa

from dlp_ops.policy import OpsPolicy
from dlp_privacy.audit import AuditResult, residual_miss_rate
from dlp_review.ops.policy import ReviewOpsPolicy
from dlp_review.ops.runner import quality_report
from dlp_review.ops.seeding import detected
from dlp_schema.db.repository import (
    get_labels,
    get_session,
    list_assignments,
    list_golden_sets,
    list_lifecycle_events,
    list_privacy_audits,
    list_review_work,
    list_session_ids,
    withdrawn_session_ids,
)
from dlp_schema.episode import current_labels, non_operational_ids
from dlp_schema.history import review_changes
from dlp_schema.labels import LabelRecord, Source, VerificationState
from dlp_schema.review import AssignmentStatus, ReviewMode
from dlp_schema.session import LifecycleState

REVIEWED = (VerificationState.HUMAN_APPROVED, VerificationState.HUMAN_CORRECTED)
# 검증 완료로 보는 모델 라벨 상태 (표본 검증 묶음의 나머지도 검증된 것으로 본다)
VERIFIED_LABEL = (*REVIEWED, VerificationState.SAMPLE_VERIFIED)
VERIFIED = (LifecycleState.HUMAN_VERIFIED, LifecycleState.SPLIT_ASSIGNED, LifecycleState.EXPORTED)


def week_range(week: str) -> tuple[datetime, datetime]:
    """'2026-W41' → (월요일 0시 UTC, 다음 월요일 0시 UTC)."""
    year, w = week.split("-W")
    start = datetime.combine(date.fromisocalendar(int(year), int(w), 1), datetime.min.time(), UTC)
    return start, start + timedelta(days=7)


def week_of(t: datetime) -> str:
    y, w, _ = t.astimezone(UTC).isocalendar()
    return f"{y}-W{w:02d}"


def verified_time(
    conn: sa.Connection, session_id: str, state: LifecycleState, history: list[LabelRecord]
) -> datetime | None:
    """세션의 검증 완료 시각.

    생애주기 기록에 human_verified로 옮긴 전이(이전 상태가 있는 것)가 있으면 그 첫 시각이다.
    0011 이관의 보충 기록(이전 상태 없음)뿐인 옛 세션은 지금 상태가 검증 이후일 때만 라벨 이력에서
    추정한다 (verified_at).
    """
    for event in list_lifecycle_events(conn, session_id):
        if event.to_state is LifecycleState.HUMAN_VERIFIED and event.from_state is not None:
            return event.at
    return verified_at(history) if state in VERIFIED else None


def verified_at(history: list[LabelRecord]) -> datetime | None:
    """세션의 검증 완료 시각 (블러 제외, 운영 라벨만).

    단위는 "항목"(수정 이력 사슬)이다. 항목이 처음 사람 손을 거친 시각 = 사슬(자기와 조상)에서
    가장 이른 검수 시각 (모델 레코드는 VERIFIED_LABEL 상태의 reviewed_at, 사람 레코드는 작성 시각).
    - 완료 시각 T = 현재 운영 라벨과 사람이 지운 항목마다의 "처음 검수 시각" 중 가장 늦은 것.
      단계별로 늦게 생긴 라벨(예: 2주 뒤의 행동 구간)은 자기 검수 시각까지 T를 늦춘다.
    - 이미 검수한 항목을 나중에 고치거나 지우는 QA는 사슬의 처음 검수 시각을 바꾸지 않아 T도
      그대로다.
    - T 이전(같은 시각 포함)에 있던 현재 모델 라벨 중 미검수가 있으면 완료가 아니다 (None).
      T 뒤에 생긴 미검수 모델 라벨(검증 뒤의 새 모델 버전)은 보지 않는다.
    한계: 수명 주기 전이 시각이 기록되지 않아(ADR 0027) 검증 뒤에 생긴 라벨을 나중에 검수하면 그
    검수 시각으로 T가 늦춰지고, 수명 주기를 검수보다 늦게 바꾸면 지난 주의 수가 늘 수 있다.
    """
    labels = [x for x in history if x.kind != "blur_track"]
    excluded = non_operational_ids(labels)
    ops = [x for x in labels if x.label_id not in excluded]
    by_id = {x.label_id: x for x in ops}

    def own_review(x: LabelRecord) -> datetime | None:
        if x.provenance.source is Source.HUMAN:
            return x.created_at
        v = x.verification
        return v.reviewed_at if v.state in VERIFIED_LABEL else None

    def first_review(x: LabelRecord) -> datetime | None:
        times: list[datetime] = []
        seen: set[str] = set()
        cur: LabelRecord | None = x
        while cur is not None and cur.label_id not in seen:
            seen.add(cur.label_id)
            t = own_review(cur)
            if t is not None:
                times.append(t)
            cur = by_id.get(cur.parent_label_id) if cur.parent_label_id else None
        return min(times) if times else None

    current = current_labels(ops, operational=False)
    pending = [x for x in current if x.provenance.source is Source.MODEL and own_review(x) is None]
    waiting = {x.label_id for x in pending}
    items = [x for x in current if x.label_id not in waiting]
    # 사람이 지운 항목 (모델 버전 교체로 지운 삭제 레코드는 검수가 아니다)
    items += [x for x in ops if x.retracted and x.provenance.source is Source.HUMAN]
    reviewed = [t for t in (first_review(x) for x in items) if t is not None]
    if not reviewed:
        return None
    done = max(reviewed)
    if any(x.created_at <= done for x in pending):
        return None
    return done


def _ratio(a: float, b: float) -> float:
    return a / b if b else math.nan


@dataclass
class WeeklyMetrics:
    week: str
    review_minutes_per_video_hour: float = math.nan
    privacy_review_minutes_per_video_hour: float = math.nan
    correction_rate: float = math.nan
    auto_approval_rate: float = math.nan
    prelabel_bias: float = math.nan
    seeded_detection_rate: float = math.nan
    residual_blur_miss_per_hour: float = math.nan
    verified_episodes: int = 0
    review_hours: float = 0.0
    cost_per_episode: float = math.nan
    counts: dict[str, int] = field(default_factory=dict[str, int])

    def as_dict(self) -> dict[str, Any]:
        return {
            k: (None if isinstance(v, float) and math.isnan(v) else v)
            for k, v in asdict(self).items()
        }


def weekly_metrics(
    conn: sa.Connection, week: str, review_policy: ReviewOpsPolicy, policy: OpsPolicy
) -> WeeklyMetrics:
    start, end = week_range(week)
    m = WeeklyMetrics(week)

    def inside(t: datetime | None) -> bool:
        return t is not None and start <= t < end

    # 검수 시간
    work = list_review_work(conn, start, end)
    labeling = [w for w in work if w.stage != "privacy"]
    privacy = [w for w in work if w.stage == "privacy"]
    m.review_minutes_per_video_hour = _ratio(
        sum(w.seconds for w in labeling) / 60, sum(w.video_ms for w in labeling) / 3_600_000
    )
    m.privacy_review_minutes_per_video_hour = _ratio(
        sum(w.seconds for w in privacy) / 60, sum(w.video_ms for w in privacy) / 3_600_000
    )
    m.review_hours = sum(w.seconds for w in work) / 3600

    # 수정률·자동 승인율 (골든셋·사용 중지 세션 제외), 검증 에피소드
    accepted = corrected = deleted = added = 0
    withdrawn = withdrawn_session_ids(conn)
    golden = {sid for g in list_golden_sets(conn) for sid in g.session_ids}
    for sid in list_session_ids(conn):
        if sid in withdrawn:
            continue
        history = get_labels(conn, sid)
        session = get_session(conn, sid)
        if inside(verified_time(conn, sid, session.lifecycle_state, history)):
            m.verified_episodes += 1
        if sid in golden:
            continue
        changes = [c for c in review_changes(history, REVIEWED) if c.label.kind != "blur_track"]
        for c in changes:
            if not inside(c.at):
                continue
            if c.change == "accepted":
                accepted += 1
            elif c.change == "corrected":
                corrected += 1
            elif c.change == "deleted":
                deleted += 1
            else:
                added += 1
    m.counts = {"accepted": accepted, "corrected": corrected, "deleted": deleted, "added": added}
    m.correction_rate = _ratio(corrected + deleted + added, accepted + corrected + deleted + added)
    m.auto_approval_rate = _ratio(accepted, accepted + corrected + deleted)

    # 블라인드 편향·오류 삽입 발견율 (그 주에 끝난 배정)
    done = [
        a for a in list_assignments(conn, status=AssignmentStatus.DONE) if inside(a.completed_at)
    ]
    total = found = 0
    for a in done:
        if a.mode is ReviewMode.SEEDED_ERROR:
            labels = get_labels(conn, a.session_id)
            for err in a.injected:
                total += 1
                found += detected(
                    err, labels, a.assignee, review_policy.seeding.detect_tolerance_ms,
                    review_policy.seeding.blur_overlap,
                )  # fmt: skip
    m.seeded_detection_rate = _ratio(found, total)
    m.counts |= {"seeded_errors": total, "seeded_found": found}
    blind = {a.assignment_id for a in done if a.mode is ReviewMode.BLIND}
    if blind:
        biases = [
            v
            for k, v in quality_report(conn, review_policy).blind_bias.items()
            if k in blind and not math.isnan(v)
        ]
        m.prelabel_bias = sum(biases) / len(biases) if biases else math.nan

    # 잔여 블러 누락
    audits = list_privacy_audits(conn, start, end)
    m.counts["privacy_audits"] = len(audits)
    if audits:
        m.residual_blur_miss_per_hour = residual_miss_rate(
            [
                AuditResult(
                    a.session_id, a.stream_id, a.duration_ms, a.misses, a.auditor, a.blur_reviewer
                )
                for a in audits
            ]
        )

    if policy.cost.hourly_cost is not None and m.verified_episodes:
        m.cost_per_episode = m.review_hours * policy.cost.hourly_cost / m.verified_episodes
    return m


def alerts(history: list[WeeklyMetrics], policy: OpsPolicy) -> list[str]:
    """주간 지표 추이에서 경고 (오래된 주부터 정렬된 목록)."""
    out: list[str] = []
    if history:
        cur = history[-1]
        privacy_work = not math.isnan(cur.privacy_review_minutes_per_video_hour)
        if cur.counts.get("privacy_audits", 0) == 0 and (privacy_work or cur.verified_episodes):
            out.append(
                f"{cur.week}: 블러 잔여 누락 감사가 없다 — 감사 없는 주는 통과로 보지 않는다 "
                "(dlp privacy audit-sample로 표본을 뽑아 감사한다)"
            )
    ap = policy.alerts
    if len(history) >= 2:
        prev, cur = history[-2], history[-1]
        rise = cur.auto_approval_rate - prev.auto_approval_rate
        drop = prev.seeded_detection_rate - cur.seeded_detection_rate
        if rise >= ap.auto_approval_rise and drop >= ap.detection_drop:
            out.append(
                f"{cur.week}: 자동 승인율이 {rise:+.1%} 올랐는데 오류 삽입 발견율이 {-drop:+.1%} "
                "떨어졌다 — 모델 개선이 아니라 검수 품질 저하로 본다"
            )
    n = ap.stagnation_weeks
    if len(history) >= n:
        first, last = history[-n], history[-1]
        if (
            last.review_minutes_per_video_hour >= first.review_minutes_per_video_hour
            and last.correction_rate >= first.correction_rate
        ):
            out.append(
                f"{first.week}~{last.week}: 검수 시간과 수정률이 {n}주 동안 줄지 않았다 — "
                "모델보다 가이드라인·온톨로지를 점검한다"
            )
    return out


def markdown(history: list[WeeklyMetrics], warnings: list[str], currency: str) -> str:
    rows = [
        ("검수 분 / 영상 1시간 (작업 라벨)", "review_minutes_per_video_hour", "{:.1f}"),
        ("검수 분 / 영상 1시간 (블러)", "privacy_review_minutes_per_video_hour", "{:.1f}"),
        ("수정률", "correction_rate", "{:.1%}"),
        ("자동 승인율", "auto_approval_rate", "{:.1%}"),
        ("프리라벨 편향 (블라인드)", "prelabel_bias", "{:.3f}"),
        ("오류 삽입 발견율", "seeded_detection_rate", "{:.1%}"),
        ("잔여 블러 누락 / 영상 1시간", "residual_blur_miss_per_hour", "{:.2f}"),
        ("검증 에피소드", "verified_episodes", "{}"),
        (f"에피소드당 생산원가 ({currency})", "cost_per_episode", "{:,.0f}"),
    ]
    lines = ["# 주간 운영 지표", "", "| 지표 | " + " | ".join(m.week for m in history) + " |"]
    lines.append("| --- | " + " | ".join("---" for _ in history) + " |")
    for name, key, fmt in rows:
        cells: list[str] = []
        for m in history:
            v = getattr(m, key)
            cells.append("-" if isinstance(v, float) and math.isnan(v) else fmt.format(v))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    if warnings:
        lines += ["", "## 경고", ""] + [f"- {w}" for w in warnings]
    return "\n".join(lines) + "\n"
