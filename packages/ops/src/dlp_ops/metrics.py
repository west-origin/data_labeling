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
- 검증 에피소드: 사람 검증을 마친 세션(수명 주기) 중 검증 완료 시각이 그 주인 것.
  검증 완료 시각(verified_at) = 그 시각까지 있던 운영 현재 라벨 중 모델 라벨이 모두
  검수된(승인·수정·표본 검증) 가장 이른 검수 시각. 그 뒤의 재검수·QA 수정·새 모델 버전은
  이 시각을 바꾸지 않으므로 지난 주의 수가 나중에 바뀌지 않는다. 블러(프라이버시 검수)는
  따로 센다.
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
    list_privacy_audits,
    list_review_work,
    list_session_ids,
    withdrawn_session_ids,
)
from dlp_schema.episode import current_labels
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


def verified_at(history: list[LabelRecord]) -> datetime | None:
    """세션의 검증 완료 시각 (블러 제외).

    후보 시각 T(검수 시각과 사람 레코드 작성 시각)마다 T까지 만든 레코드만으로 운영 현재
    라벨을 구해, 모델 라벨이 모두 T 이전에 검수(VERIFIED_LABEL)됐으면 완료다.
    그런 T 중 가장 이른 것.
    T 뒤에 생긴 레코드(재검수·QA 수정·새 모델 버전)는 T의 판정에 끼지 않아 값이 안정적이다.
    """
    labels = [x for x in history if x.kind != "blur_track"]

    def reviewed_by(x: LabelRecord, t: datetime) -> bool:
        v = x.verification
        return v.state in VERIFIED_LABEL and v.reviewed_at is not None and v.reviewed_at <= t

    candidates = sorted(
        {x.verification.reviewed_at for x in labels if x.verification.reviewed_at is not None}
        | {x.created_at for x in labels if x.provenance.source is Source.HUMAN}
    )
    for t in candidates:
        current = current_labels([x for x in labels if x.created_at <= t])
        if current and all(
            x.provenance.source is not Source.MODEL or reviewed_by(x, t) for x in current
        ):
            return t
    return None


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
        if session.lifecycle_state in VERIFIED and inside(verified_at(history)):
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
