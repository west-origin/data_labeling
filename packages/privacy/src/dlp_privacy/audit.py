"""잔여 누락 감사와 전수 검수 종료 판정.

실제 위험은 검수 후에도 남은 누락이다. 승인된 블러본에서 표본을 뽑아 원 검수자가 아닌 감사자가
독립적으로 다시 보고, 영상 1시간당 잔여 누락 수를 추정한다.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Literal


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


ReviewMode = Literal["full", "sampled"]


def review_mode(
    weekly_rates: list[float], target: float | None, weeks_below_target: int
) -> ReviewMode:
    """전수 검수 종료 판정. 최근 weeks_below_target주 연속 목표 이하면 표본 검수로 전환한다.

    가장 최근 주가 목표를 넘으면 즉시 전수 검수로 돌아간다. 목표가 아직 없으면 전수 검수다.
    """
    if target is None or len(weekly_rates) < weeks_below_target:
        return "full"
    recent = weekly_rates[-weeks_below_target:]
    return "sampled" if all(r <= target for r in recent) else "full"
