from __future__ import annotations

import pytest

from dlp_privacy.audit import (
    AuditCandidate,
    AuditResult,
    residual_miss_rate,
    review_mode,
    select_audit_sample,
)


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
