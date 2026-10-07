"""동기화 구성 요소 단위 테스트 (WP4).

시계 맞춤, 두드림 검출·짝짓기, 상관 봉우리, 정책 검증, 장갑 신호를 따로 본다.

영상 없이 손으로 만든 신호·앵커를 쓴다. 정답은 테스트 안에서 정한 값(오프셋, ppm, 펄스 시각)이다.
"""

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
from dlp_sync.signals import glove_pressure_prefixes, glove_series
from dlp_sync.taps import DoubleTap, detect_double_taps, match_taps
from dlp_sync.xcorr import correlate, find_peak


def test_fit_estimates_drift_only_with_enough_span() -> None:
    """`fit_clock`이 앵커 범위가 충분할 때만 드리프트를 추정하는지 검증한다.

    정답: master = 120 + s·(1 + 50e-6). 60초 범위 앵커 → 오프셋 120, 50 ppm, 잔차 0.
    200 ms 범위 앵커 → 배율 1, 오프셋은 그 구간 차의 중앙값(약 120.06). 500 ppm(10 ms/20 s)은 상한
    300을 넘어 `FitError`, 빈 앵커도 `FitError`.
    """

    def truth(s: float) -> float:
        """정답 관계: 오프셋 120 ms, 드리프트 +50 ppm."""
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
    """`rate` Hz 격자(길이 `length` ms)에 잡음(표준편차 0.01) + `times`에서 시작하는 폭 `width` ms
    사각 펄스.
    """
    t = np.arange(0, length, 1000 / rate)
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.01, t.size)
    for s in times:
        x[(t >= s) & (t < s + width)] += 1.0
    return t, x


def test_double_taps_ignore_long_contacts_and_single_spikes(policy: SyncPolicy) -> None:
    """긴 접촉(900 ms)과 홀로 있는 스파이크(4000 ms)는 두드림이 아니고, 간격 200·210 ms의 펄스 쌍만
    두 번 두드림으로 잡히는지 검증한다. 정답은 펄스를 넣은 시각 (1000, 1200), (6000, 6210).
    """
    t, x = _pulses([1_000, 1_200, 4_000, 6_000, 6_210], width=30)
    _, long = _pulses([2_000], width=900)
    taps = detect_double_taps(t, x + long, policy.tap)
    assert [(d.first_ms, d.second_ms) for d in taps] == [(1_000, 1_200), (6_000, 6_210)]


def test_tap_matching_tolerates_missing_and_spurious_taps(policy: SyncPolicy) -> None:
    """한쪽이 두드림을 놓치고 다른 쪽에 가짜 두드림이 있어도 다수를 설명하는 오프셋(500 ms)을
    고르는지.

    정답: 대상 시계 = 기준 - 500 ms. 맞는 짝 2개 → 앵커 4개, 모든 앵커의 차가 500.
    """
    ref = [DoubleTap(1_000, 1_200), DoubleTap(9_000, 9_200), DoubleTap(20_000, 20_200)]
    # 대상 시계 = 기준 - 500 ms. 가운데 두드림을 놓쳤고, 잡음 하나가 두드림처럼 잡혔다.
    tgt = [DoubleTap(300, 450), DoubleTap(500, 700), DoubleTap(19_500, 19_700)]
    anchors = match_taps(ref, tgt, policy.tap)
    assert {(a.master_ms - a.stream_ms) for a in anchors} == {500}
    assert len(anchors) == 4


def test_tap_matching_tolerance_grows_with_drift(policy: SyncPolicy) -> None:
    """20분 떨어진 두드림은 80 ppm 드리프트로 96 ms 어긋난다.

    드리프트 상한만큼 허용 오차를 넓힌다. 넓히지 않으면 한 쌍(앵커 2개), `max_drift_ppm`을 주면
    두 쌍(앵커 4개)이 짝지어져야 한다 (ADR 0028 결정 4).
    """
    scale = 1 + 80e-6
    ref = [DoubleTap(3_000, 3_200), DoubleTap(1_203_000, 1_203_200)]
    tgt = [DoubleTap(r.first_ms / scale - 500, r.second_ms / scale - 500) for r in ref]
    assert len(match_taps(ref, tgt, policy.tap)) == 2  # 드리프트를 감안하지 않으면 한 쌍뿐
    anchors = match_taps(ref, tgt, policy.tap, max_drift_ppm=policy.max_drift_ppm)
    assert len(anchors) == 4


def test_correlation_peak_is_subsample_accurate() -> None:
    """FFT 상관과 포물선 보간이 알려진 지연(1000샘플)을 0.05샘플 안으로 찾고 PSR이 충분히 큰지
    검증한다.
    """
    rng = np.random.default_rng(1)
    base = np.convolve(rng.standard_normal(20_000), np.ones(4) / 4, mode="same")
    ref = base[5_000:6_000]
    corr = correlate(ref, base[4_000:8_000])
    peak = find_peak(corr, exclude=4)
    assert peak.lag == pytest.approx(1_000, abs=0.05)
    assert peak.psr > 20


def test_policy_methods_are_keyed_by_known_stream_kinds(policy: SyncPolicy) -> None:
    """정책 `methods` 키가 `StreamKind`로 검증되는지: 오타 키는 거부, 기준(bodycam) 키도 거부."""
    assert set(policy.methods) <= set(StreamKind)
    data = policy.model_dump(mode="json")
    with pytest.raises(ValidationError):
        SyncPolicy.model_validate({**data, "methods": {"thrid_person": ["qr_slate"]}})
    with pytest.raises(ValidationError, match="bodycam"):
        SyncPolicy.model_validate({**data, "methods": {"bodycam": ["qr_slate"]}})


def test_glove_series_sums_only_pressure_channels(policy: SyncPolicy, tmp_path: Path) -> None:
    """시각·IMU·온도 열은 압력 합에 넣지 않는다.

    압력 1 + 2 = 3이 정답. 접두사를 생략하면 DeprecationWarning과 함께 저장소 정책을 쓰고, 압력
    채널이 없으면 채널 목록과 정책 위치가 든 ValueError를 낸다.
    """
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
    assert glove_pressure_prefixes(policy) == policy.glove.pressure_prefixes
    with pytest.warns(DeprecationWarning, match="pressure_prefixes"):
        assert np.allclose(glove_series(path).values, 3.0)  # 호환: 저장소 sync.yaml을 읽는다
    # 압력 채널이 없으면 다른 채널을 합치지 않고, 채널 목록과 정책 위치를 알려 준다
    with pytest.raises(ValueError, match=r"압력 채널.*ax, pressure_0.*glove\.pressure_prefixes"):
        glove_series(path, ("force",))
