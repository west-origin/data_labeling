"""잔여 누락 감사·전수 검수 종료 판정 단위 테스트 (dlp_privacy.audit, WP5·WP16).

DB 없이 순수 함수만 시험한다. 정답은 손으로 계산한 값이다 (ISO 주 경계, 1시간당 누락 수 등).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from dlp_privacy.audit import (
    AuditCandidate,
    AuditResult,
    iso_week,
    iso_week_bounds,
    previous_weeks,
    residual_miss_rate,
    review_mode,
    select_audit_sample,
    weekly_miss_rates,
)


def test_weeks_without_audit_do_not_count_as_passing() -> None:
    """회귀: 감사가 없던 주를 목표 이하(통과)로 보면 감사 없이 표본 검수로 넘어간다.

    시나리오: 2026-W39·W41에만 감사(누락 0)가 있고 W40은 비어 있다.
    정답: W40은 None, 최근 3주 판정은 전수(full). 세 주 모두 0이면 표본(sampled).
    ISO 주 경계(월요일 0시 UTC)와 previous_weeks 순서도 함께 확인한다.
    """
    weeks = previous_weeks("2026-W41", 3)
    assert weeks == ["2026-W39", "2026-W40", "2026-W41"]
    start, end = iso_week_bounds("2026-W41")
    assert (start, end) == (datetime(2026, 10, 5, tzinfo=UTC), datetime(2026, 10, 12, tzinfo=UTC))
    assert iso_week(datetime(2026, 10, 11, 23, tzinfo=UTC)) == "2026-W41"
    hour = AuditResult("s1", "bodycam", 3_600_000, 0, "aud", "rev")
    audits = [(datetime(2026, 9, 22, tzinfo=UTC), hour), (datetime(2026, 10, 6, tzinfo=UTC), hour)]
    rates = weekly_miss_rates(audits, weeks)
    assert rates == [0.0, None, 0.0]  # W40에는 감사가 없다
    assert review_mode(rates, 0.2, 3) == "full"
    assert review_mode([0.0, 0.0, 0.0], 0.2, 3) == "sampled"


def _candidates(n: int) -> list[AuditCandidate]:
    """세션 ID만 다른 감사 후보 n개 (10분 영상, 원 검수자 rev1)."""
    return [AuditCandidate(f"s{i:03d}", "bodycam", 600_000, "rev1") for i in range(n)]


def test_audit_sample_is_deterministic_and_sized_by_ratio() -> None:
    """표본 수와 결정성.

    표본 수 = ceil(후보 x 비율)(최소 1)이고 입력 순서와 무관하며 주가 바뀌면 달라진다.
    100개 x 5% = 5개, 3개 x 5% → 최소 1개, 후보가 없으면 빈 목록.
    """
    c = _candidates(100)
    a = select_audit_sample(c, 0.05, "2026-W41")
    assert len(a) == 5
    assert a == select_audit_sample(list(reversed(c)), 0.05, "2026-W41")
    assert a != select_audit_sample(c, 0.05, "2026-W42")
    assert len(select_audit_sample(_candidates(3), 0.05, "2026-W41")) == 1
    assert select_audit_sample([], 0.05, "2026-W41") == []


def test_residual_miss_rate_per_video_hour() -> None:
    """1시간당 잔여 누락 수 = 누락 합 / 영상 시간 합.

    30분 두 개에서 누락 1+2 → 3.0/시간. 감사자가 원 검수자와 같으면 ValueError.
    """
    results = [
        AuditResult("s1", "bodycam", 1_800_000, 1, "aud", "rev1"),
        AuditResult("s2", "bodycam", 1_800_000, 2, "aud", "rev2"),
    ]
    assert residual_miss_rate(results) == pytest.approx(3.0)
    with pytest.raises(ValueError, match="원 검수자"):
        residual_miss_rate([AuditResult("s1", "bodycam", 1_000, 0, "rev1", "rev1")])


def test_full_review_exit_and_revert() -> None:
    """전수 검수 종료·복귀 규칙.

    목표 미정이면 전수, 기간이 모자라면 전수, 최근 8주 연속 목표 이하면 표본,
    가장 최근 주가 목표를 넘으면 즉시 전수로 돌아간다.
    """
    assert review_mode([0.1] * 8, None, 8) == "full"  # 목표 미정
    assert review_mode([0.1] * 7, 0.2, 8) == "full"  # 기간 부족
    assert review_mode([0.5] + [0.1] * 8, 0.2, 8) == "sampled"
    assert review_mode([0.1] * 8 + [0.3], 0.2, 8) == "full"  # 최근 주가 넘으면 즉시 복귀
