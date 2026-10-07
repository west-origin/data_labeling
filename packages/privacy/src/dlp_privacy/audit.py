"""잔여 누락 감사와 전수 검수 종료 판정.

실제 위험은 검수 후에도 남은 누락이다. 승인된 블러본에서 표본을 뽑아 원 검수자가 아닌 감사자가
독립적으로 다시 보고, 영상 1시간당 잔여 누락 수를 추정한다.

- 표본: 그 주(ISO 주)에 블러 검수를 수집하고 승인된 세션의 영상 스트림 (`dlp privacy audit-sample`).
- 감사 결과는 privacy_audits에 남는다 (`dlp ops privacy-audit`).
- 감사가 없는 주는 목표를 지킨 것으로 보지 않는다 (전수 검수 종료 판정에서 실패로 센다).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal

import sqlalchemy as sa

from dlp_schema.db.repository import get_session, list_review_tasks, list_session_ids
from dlp_schema.review import ReviewMode as TaskMode
from dlp_schema.review import ReviewStage, ReviewTaskStatus
from dlp_schema.session import PrivacyState, StreamKind

VIDEO_KINDS = {StreamKind.BODYCAM, StreamKind.THIRD_PERSON}


@dataclass(frozen=True)
class AuditCandidate:
    session_id: str
    stream_id: str
    duration_ms: int
    blur_reviewer: str


@dataclass(frozen=True)
class AuditResult:
    session_id: str
    stream_id: str
    duration_ms: int
    misses: int
    auditor: str
    blur_reviewer: str


def select_audit_sample(
    candidates: list[AuditCandidate], ratio: float, week: str
) -> list[AuditCandidate]:
    """그 주 승인된 블러본 중 ratio만큼(최소 1개)을 뽑는다.

    같은 주·같은 후보면 같은 결과가 나오도록 (주, 세션, 스트림)의 해시 순서로 고른다.
    """
    if not candidates:
        return []
    n = max(1, math.ceil(len(candidates) * ratio))

    def key(c: AuditCandidate) -> str:
        return hashlib.sha256(f"{week}|{c.session_id}|{c.stream_id}".encode()).hexdigest()

    return sorted(candidates, key=key)[:n]


def residual_miss_rate(results: list[AuditResult]) -> float:
    """영상 1시간당 잔여 누락 수. 감사자가 원 검수자와 같으면 오류."""
    for r in results:
        if r.auditor == r.blur_reviewer:
            raise ValueError(f"{r.session_id}/{r.stream_id}: 감사자가 원 검수자와 같습니다")
    hours = sum(r.duration_ms for r in results) / 3_600_000
    if hours == 0:
        raise ValueError("감사한 영상이 없습니다")
    return sum(r.misses for r in results) / hours


# 블러 검수 방식 (검수 작업 방식 dlp_schema.review.ReviewMode와 다르다)
ReviewMode = Literal["full", "sampled"]


def review_mode(
    weekly_rates: Sequence[float | None], target: float | None, weeks_below_target: int
) -> ReviewMode:
    """전수 검수 종료 판정. 최근 weeks_below_target주 연속 목표 이하면 표본 검수로 전환한다.

    weekly_rates의 None은 감사가 없던 주다. 감사가 없으면 목표를 지켰는지 모르므로 통과로 보지
    않는다 (전수 검수 유지). 가장 최근 주가 목표를 넘으면 즉시 전수 검수로 돌아간다.
    목표가 아직 없으면 전수 검수다.
    """
    if target is None or len(weekly_rates) < weeks_below_target:
        return "full"
    recent = weekly_rates[-weeks_below_target:]
    return "sampled" if all(r is not None and r <= target for r in recent) else "full"


def iso_week_bounds(week: str) -> tuple[datetime, datetime]:
    """'2026-W41' → 그 주 월요일 0시(UTC)와 다음 주 월요일 0시."""
    year, _, num = week.partition("-W")
    start = datetime.fromisocalendar(int(year), int(num), 1).replace(tzinfo=UTC)
    return start, start + timedelta(days=7)


def iso_week(at: datetime) -> str:
    year, week, _ = at.astimezone(UTC).isocalendar()
    return f"{year}-W{week:02d}"


def previous_weeks(week: str, n: int) -> list[str]:
    """week를 포함해 거슬러 n주 (오래된 주부터)."""
    start, _ = iso_week_bounds(week)
    return [iso_week(start - timedelta(weeks=i)) for i in reversed(range(n))]


def weekly_miss_rates(
    audits: Iterable[tuple[datetime, AuditResult]], weeks: Sequence[str]
) -> list[float | None]:
    """주마다 잔여 누락률. 감사가 없던 주는 None (통과로 보지 않는다)."""
    by_week: dict[str, list[AuditResult]] = {}
    for at, result in audits:
        by_week.setdefault(iso_week(at), []).append(result)
    return [residual_miss_rate(by_week[w]) if w in by_week else None for w in weeks]


def audit_candidates(conn: sa.Connection, week: str) -> list[AuditCandidate]:
    """그 주에 블러 검수 작업을 수집했고 지금 승인 상태인 세션의 영상 스트림.

    원 검수자(blur_reviewer)는 그 스트림의 마지막 운영 블러 검수 작업 담당자다.
    """
    start, end = iso_week_bounds(week)
    out: list[AuditCandidate] = []
    for sid in list_session_ids(conn):
        session = get_session(conn, sid)
        if session.privacy_state is not PrivacyState.APPROVED:
            continue
        tasks = [
            t
            for t in list_review_tasks(conn, sid, ReviewStage.PRIVACY)
            if t.status is ReviewTaskStatus.COLLECTED
            and t.mode in (TaskMode.STANDARD, TaskMode.QA)
            and t.collected_at is not None
            and start <= t.collected_at < end
        ]
        for s in session.streams:
            if s.kind not in VIDEO_KINDS:
                continue
            mine = [t for t in tasks if t.stream_id == s.stream_id]
            if not mine:
                continue
            last = max(mine, key=lambda t: t.collected_at or t.created_at)
            out.append(
                AuditCandidate(sid, s.stream_id, session.duration_ms, last.assignee or "unknown")
            )
    return out
