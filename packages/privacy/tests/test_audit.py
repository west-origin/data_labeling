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
    """회귀: 감사가 없던 주를 목표 이하(통과)로 보면 감사 없이 표본 검수로 넘어간다."""
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
    return [AuditCandidate(f"s{i:03d}", "bodycam", 600_000, "rev1") for i in range(n)]


def test_audit_sample_is_deterministic_and_sized_by_ratio() -> None:
    c = _candidates(100)
    a = select_audit_sample(c, 0.05, "2026-W41")
    assert len(a) == 5
    assert a == select_audit_sample(list(reversed(c)), 0.05, "2026-W41")
    assert a != select_audit_sample(c, 0.05, "2026-W42")
    assert len(select_audit_sample(_candidates(3), 0.05, "2026-W41")) == 1
    assert select_audit_sample([], 0.05, "2026-W41") == []


def test_residual_miss_rate_per_video_hour() -> None:
    results = [
        AuditResult("s1", "bodycam", 1_800_000, 1, "aud", "rev1"),
        AuditResult("s2", "bodycam", 1_800_000, 2, "aud", "rev2"),
    ]
    assert residual_miss_rate(results) == pytest.approx(3.0)
    with pytest.raises(ValueError, match="원 검수자"):
        residual_miss_rate([AuditResult("s1", "bodycam", 1_000, 0, "rev1", "rev1")])


def test_full_review_exit_and_revert() -> None:
    assert review_mode([0.1] * 8, None, 8) == "full"  # 목표 미정
    assert review_mode([0.1] * 7, 0.2, 8) == "full"  # 기간 부족
    assert review_mode([0.5] + [0.1] * 8, 0.2, 8) == "sampled"
    assert review_mode([0.1] * 8 + [0.3], 0.2, 8) == "full"  # 최근 주가 넘으면 즉시 복귀
