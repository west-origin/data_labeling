"""잔여 누락 감사와 전수 검수 종료 판정.

실제 위험은 검수 후에도 남은 누락이다. 승인된 블러본에서 표본을 뽑아 원 검수자가 아닌 감사자가
독립적으로 다시 보고, 영상 1시간당 잔여 누락 수를 추정한다.

- 표본: 그 주(ISO 주)에 블러 검수를 수집하고 승인된 세션의 영상 스트림 (`dlp privacy audit-sample`).
- 감사 결과는 privacy_audits에 남는다 (`dlp ops privacy-audit`).
- 감사가 없는 주는 목표를 지킨 것으로 보지 않는다 (전수 검수 종료 판정에서 실패로 센다).

WP5·WP16, ADR 0020·0023. 정책 값: defaults.yaml `privacy.full_review_exit`
(weeks_below_target: 연속 몇 주, audit_sample_ratio: 표본 비율),
`success_criteria.residual_blur_miss_per_hour_max`(목표, 없으면 전수 검수 유지).

공개 함수:
- `select_audit_sample`: 결정적 표본 추출.
- `residual_miss_rate` / `weekly_miss_rates`: 1시간당 잔여 누락 수 (주별).
- `review_mode`: 전수(full) / 표본(sampled) 검수 판정.
- `iso_week` / `iso_week_bounds` / `previous_weeks`: ISO 주 계산 (UTC).
- `audit_candidates`: DB에서 그 주의 감사 후보를 고른다 (읽기만).
- `stream_duration_ms`: 감사할 영상 스트림 자신의 길이 (바디캠은 세션 길이, 그 밖은 PTS 인덱스).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import sqlalchemy as sa

from dlp_media.pts import PtsIndex
from dlp_media.storage import ObjectStore
from dlp_schema.db.repository import get_session, list_review_tasks, list_session_ids
from dlp_schema.review import ReviewMode as TaskMode
from dlp_schema.review import ReviewStage, ReviewTaskStatus
from dlp_schema.session import PrivacyState, Session, Stream, StreamKind

# 감사 대상 영상 스트림 종류 (runner.VIDEO_KINDS와 같다)
VIDEO_KINDS = {StreamKind.BODYCAM, StreamKind.THIRD_PERSON}


@dataclass(frozen=True)
class AuditCandidate:
    """감사 후보 (승인된 블러본 하나)."""

    session_id: str
    stream_id: str
    # 그 영상 스트림의 길이 (ms, `stream_duration_ms`). 3인칭은 바디캠(세션)과 길이가 다를 수 있다.
    duration_ms: int
    # 원 블러 검수자 (감사자는 이 사람이 아니어야 한다)
    blur_reviewer: str


@dataclass(frozen=True)
class AuditResult:
    """감사 결과 하나 (DB privacy_audits 행과 대응)."""

    session_id: str
    stream_id: str
    # 감사한 영상 길이 (ms)
    duration_ms: int
    # 감사자가 찾은 블러 누락 수 (`dlp ops privacy-audit`로 입력)
    misses: int
    auditor: str
    blur_reviewer: str


def select_audit_sample(
    candidates: list[AuditCandidate], ratio: float, week: str
) -> list[AuditCandidate]:
    """그 주 승인된 블러본 중 ratio만큼(최소 1개)을 뽑는다.

    같은 주·같은 후보면 같은 결과가 나오도록 (주, 세션, 스트림)의 해시 순서로 고른다.

    Args:
        candidates: `audit_candidates` 결과 (순서 무관).
        ratio: 표본 비율 (defaults.yaml privacy.full_review_exit.audit_sample_ratio).
        week: ISO 주 문자열 ("2026-W41"). 주가 바뀌면 다른 표본이 나온다.

    Returns:
        ceil(후보 수 x ratio)개 (후보가 있으면 최소 1개). 후보가 없으면 빈 목록.
    """
    if not candidates:
        return []
    n = max(1, math.ceil(len(candidates) * ratio))

    def key(c: AuditCandidate) -> str:
        return hashlib.sha256(f"{week}|{c.session_id}|{c.stream_id}".encode()).hexdigest()

    return sorted(candidates, key=key)[:n]


def residual_miss_rate(results: list[AuditResult]) -> float:
    """영상 1시간당 잔여 누락 수. 감사자가 원 검수자와 같으면 오류.

    Returns:
        (누락 수 합) / (영상 길이 합, 시간).

    Raises:
        ValueError: 감사자 = 원 검수자인 결과가 있거나, 감사한 영상 길이 합이 0일 때.
    """
    for r in results:
        if r.auditor == r.blur_reviewer:
            raise ValueError(f"{r.session_id}/{r.stream_id}: 감사자가 원 검수자와 같습니다")
    hours = sum(r.duration_ms for r in results) / 3_600_000  # ms → 시간
    if hours == 0:
        raise ValueError("감사한 영상이 없습니다")
    return sum(r.misses for r in results) / hours


# 블러 검수 방식 (검수 작업 방식 dlp_schema.review.ReviewMode와 다르다)
# full: 모든 블러본을 사람이 전수 검수, sampled: 표본만 검수.
ReviewMode = Literal["full", "sampled"]


def review_mode(
    weekly_rates: Sequence[float | None], target: float | None, weeks_below_target: int
) -> ReviewMode:
    """전수 검수 종료 판정. 최근 weeks_below_target주 연속 목표 이하면 표본 검수로 전환한다.

    weekly_rates의 None은 감사가 없던 주다. 감사가 없으면 목표를 지켰는지 모르므로 통과로 보지
    않는다 (전수 검수 유지). 가장 최근 주가 목표를 넘으면 즉시 전수 검수로 돌아간다.
    목표가 아직 없으면 전수 검수다.

    Args:
        weekly_rates: 주별 잔여 누락률 (오래된 주부터, `weekly_miss_rates`).
        target: 1시간당 허용 잔여 누락 수 (success_criteria). None이면 미정.
        weeks_below_target: 연속으로 목표 이하여야 하는 주 수.
    """
    if target is None or len(weekly_rates) < weeks_below_target:
        return "full"
    recent = weekly_rates[-weeks_below_target:]
    return "sampled" if all(r is not None and r <= target for r in recent) else "full"


def iso_week_bounds(week: str) -> tuple[datetime, datetime]:
    """'2026-W41' → 그 주 월요일 0시(UTC)와 다음 주 월요일 0시.

    반환 구간은 [시작, 끝) 반열림이다.
    """
    year, _, num = week.partition("-W")
    start = datetime.fromisocalendar(int(year), int(num), 1).replace(tzinfo=UTC)
    return start, start + timedelta(days=7)


def iso_week(at: datetime) -> str:
    """시각(시간대 필수) → UTC 기준 ISO 주 문자열 ("2026-W41")."""
    year, week, _ = at.astimezone(UTC).isocalendar()
    return f"{year}-W{week:02d}"


def previous_weeks(week: str, n: int) -> list[str]:
    """week를 포함해 거슬러 n주 (오래된 주부터)."""
    start, _ = iso_week_bounds(week)
    return [iso_week(start - timedelta(weeks=i)) for i in reversed(range(n))]


def weekly_miss_rates(
    audits: Iterable[tuple[datetime, AuditResult]], weeks: Sequence[str]
) -> list[float | None]:
    """주마다 잔여 누락률. 감사가 없던 주는 None (통과로 보지 않는다).

    Args:
        audits: (감사 시각, 결과) 목록. 감사 시각의 ISO 주로 묶는다.
        weeks: 결과를 낼 주 목록 (`previous_weeks`).

    Raises:
        ValueError: `residual_miss_rate`와 같다.
    """
    by_week: dict[str, list[AuditResult]] = {}
    for at, result in audits:
        by_week.setdefault(iso_week(at), []).append(result)
    return [residual_miss_rate(by_week[w]) if w in by_week else None for w in weeks]


def stream_duration_ms(
    session: Session, stream: Stream, raw: ObjectStore | None, work: Path
) -> int:
    """감사할 영상 스트림 자신의 길이 (정수 ms). 누락률(누락 수 / 시간)의 분모다.

    회귀: 감사 후보와 `dlp ops privacy-audit`가 3인칭 스트림에도 세션(바디캠) 길이를 써서, 길이가
    다른 3인칭 영상의 시간당 잔여 누락률이 틀렸다.

    - 바디캠: 세션 길이 (수집이 바디캠 PTS 인덱스의 `duration_ms`를 반올림해 넣은 값과 같다).
      원본 버킷을 읽지 않는다.
    - 그 밖의 영상: 그 스트림의 PTS 인덱스(`Stream.pts_index_uri`)를 원본 버킷에서 읽어 같은
      규칙(`PtsIndex.duration_ms` 반올림)으로 계산한다 (CLAUDE.md: 영상 시각은 PTS 인덱스로만).

    Args:
        session: 세션.
        stream: 그 세션의 영상 스트림.
        raw: 원본 버킷 저장소 (CLI는 감사 저장소 → 읽기 기록이 남는다). 바디캠이면 None이어도 된다.
        work: PTS 인덱스를 받을 임시 디렉터리.

    Raises:
        ValueError: 바디캠이 아닌데 PTS 인덱스가 없거나 저장소가 없을 때, 또는 인덱스 URI가 그
            저장소에 있지 않을 때 (세션 길이로 대신하면 잘못된 누락률이 조용히 기록된다).
    """
    if stream.kind is StreamKind.BODYCAM:
        return session.duration_ms
    if stream.pts_index_uri is None:
        raise ValueError(f"{session.session_id}/{stream.stream_id}: PTS 인덱스가 없습니다")
    if raw is None:
        raise ValueError(
            f"{session.session_id}/{stream.stream_id}: 길이를 읽을 원본 저장소가 없습니다"
        )
    prefix = raw.uri("")
    if not stream.pts_index_uri.startswith(prefix):
        raise ValueError(f"{stream.pts_index_uri}는 저장소 {raw.bucket}에 있지 않습니다")
    dest = work / f"{session.session_id}__{stream.stream_id}.pts.parquet"
    raw.get_file(stream.pts_index_uri.removeprefix(prefix), dest)
    return round(PtsIndex.read(dest).duration_ms)


def audit_candidates(
    conn: sa.Connection, week: str, duration_ms: Callable[[Session, Stream], int]
) -> list[AuditCandidate]:
    """그 주에 블러 검수 작업을 수집했고 지금 승인 상태인 세션의 영상 스트림.

    원 검수자(blur_reviewer)는 그 스트림의 마지막 운영 블러 검수 작업 담당자다.
    duration_ms: 후보 스트림의 길이를 정하는 함수 (CLI는 `stream_duration_ms`에 원본 저장소를 묶어
    넘긴다). 후보가 된 스트림에만 부른다 (원본 읽기를 줄인다).

    운영 작업(표준·QA)만 본다. 담당자가 비어 있으면 "unknown". 모든 세션을 하나씩 읽으므로 세션 수에
    비례해 느려진다 (DB 읽기만, 쓰기 없음).
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
            # 영상 길이는 세션 길이가 아니라 그 스트림의 길이다 (3인칭은 바디캠과 다를 수 있다)
            out.append(
                AuditCandidate(
                    sid, s.stream_id, duration_ms(session, s), last.assignee or "unknown"
                )
            )
    return out
