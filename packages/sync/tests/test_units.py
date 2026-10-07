from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import ValidationError

from dlp_media.tables import write_parquet
from dlp_schema.session import StreamKind
from dlp_sync.anchors import Anchor, FitError, fit_clock
from dlp_sync.policy import SyncPolicy
from dlp_sync.signals import glove_series
from dlp_sync.taps import DoubleTap, detect_double_taps, match_taps
from dlp_sync.xcorr import correlate, find_peak


def test_fit_estimates_drift_only_with_enough_span() -> None:
    def truth(s: float) -> float:
        return 120.0 + s * (1 + 50e-6)

    long = [Anchor(truth(s), s) for s in (0.0, 30_000.0, 60_000.0)]
    fit = fit_clock(long, min_drift_span_ms=10_000, max_drift_ppm=300)
    assert fit.drift_estimated
    assert fit.offset_ms == pytest.approx(120.0) and (fit.clock_scale - 1) * 1e6 == pytest.approx(
        50
    )
    assert fit.residual_rms_ms == pytest.approx(0, abs=1e-6)

    short = [Anchor(truth(s), s) for s in (1_000.0, 1_200.0)]
    fit = fit_clock(short, min_drift_span_ms=10_000, max_drift_ppm=300)
    assert not fit.drift_estimated and fit.clock_scale == 1.0
    assert fit.offset_ms == pytest.approx(120.06, abs=0.01)

    with pytest.raises(FitError, match="ppm"):
        fit_clock(
            [Anchor(0, 0), Anchor(20_010, 20_000)], min_drift_span_ms=10_000, max_drift_ppm=300
        )
    with pytest.raises(FitError):
        fit_clock([], min_drift_span_ms=1, max_drift_ppm=1)


def _pulses(
    times: list[float], width: float, rate: float = 1000.0, length: float = 10_000
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    t = np.arange(0, length, 1000 / rate)
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.01, t.size)
    for s in times:
        x[(t >= s) & (t < s + width)] += 1.0
    return t, x


def test_double_taps_ignore_long_contacts_and_single_spikes(policy: SyncPolicy) -> None:
    t, x = _pulses([1_000, 1_200, 4_000, 6_000, 6_210], width=30)
    _, long = _pulses([2_000], width=900)
    taps = detect_double_taps(t, x + long, policy.tap)
    assert [(d.first_ms, d.second_ms) for d in taps] == [(1_000, 1_200), (6_000, 6_210)]


def test_tap_matching_tolerates_missing_and_spurious_taps(policy: SyncPolicy) -> None:
    ref = [DoubleTap(1_000, 1_200), DoubleTap(9_000, 9_200), DoubleTap(20_000, 20_200)]
    # 대상 시계 = 기준 - 500 ms. 가운데 두드림을 놓쳤고, 잡음 하나가 두드림처럼 잡혔다.
    tgt = [DoubleTap(300, 450), DoubleTap(500, 700), DoubleTap(19_500, 19_700)]
    anchors = match_taps(ref, tgt, policy.tap)
    assert {(a.master_ms - a.stream_ms) for a in anchors} == {500}
    assert len(anchors) == 4


def test_correlation_peak_is_subsample_accurate() -> None:
    rng = np.random.default_rng(1)
    base = np.convolve(rng.standard_normal(20_000), np.ones(4) / 4, mode="same")
    ref = base[5_000:6_000]
    corr = correlate(ref, base[4_000:8_000])
    peak = find_peak(corr, exclude=4)
    assert peak.lag == pytest.approx(1_000, abs=0.05)
    assert peak.psr > 20


def test_policy_methods_are_keyed_by_known_stream_kinds(policy: SyncPolicy) -> None:
    assert set(policy.methods) <= set(StreamKind)
    data = policy.model_dump(mode="json")
    with pytest.raises(ValidationError):
        SyncPolicy.model_validate({**data, "methods": {"thrid_person": ["qr_slate"]}})
    with pytest.raises(ValidationError, match="bodycam"):
        SyncPolicy.model_validate({**data, "methods": {"bodycam": ["qr_slate"]}})


def test_glove_series_sums_only_pressure_channels(policy: SyncPolicy, tmp_path: Path) -> None:
    """시각·IMU·온도 열은 압력 합에 넣지 않는다."""
    t = np.arange(0.0, 100.0, 10.0)
    path = tmp_path / "glove.parquet"
    write_parquet(
        path,
        {"t_ms": t, "timestamp_ms": t + 1e6, "pressure_0": np.ones(t.size),
         "pressure_1": np.full(t.size, 2.0), "ax": np.full(t.size, 9.8),
         "temperature": np.full(t.size, 31.5)},
        {},
    )  # fmt: skip
    series = glove_series(path, policy.glove.pressure_prefixes)
    assert np.array_equal(series.t_ms, t)
    assert np.allclose(series.values, 3.0)
    assert np.allclose(glove_series(path).values, 3.0)  # 기본값은 sync.yaml에서 읽는다
    with pytest.raises(ValueError, match="압력 채널"):
        glove_series(path, ("force",))
