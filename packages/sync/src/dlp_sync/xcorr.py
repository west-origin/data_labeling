"""상호상관으로 오프셋 찾기.

오디오: 두 마이크가 같은 주변 소리를 듣는다는 점을 쓴다.
1. 거친 탐색: 낮은 샘플레이트에서 기준 오디오 한 구간을 대상 전체와 FFT 상관해 오프셋을 찾는다.
2. 정밀 탐색: 겹치는 구간에 창을 여러 개 두고, 각 창을 원래 샘플레이트에서
   예측 위치 ± 탐색 폭 안에서 다시 상관한다. 최대값은 포물선 보간으로 샘플 이하까지 구한다.
   창마다 앵커가 하나 나온다. 거친 추정은 오프셋 하나라 드리프트가 있으면 거친 추정 지점에서
   멀어질수록 실제 위치가 벗어난다. 그래서 탐색 폭 = refine_ms + max_drift_ppm·|창 - 거친 추정 지점|
   이다 (20분·300 ppm이면 끝에서 약 ±200 ms).
   슬레이트 같은 다른 방법의 맞춤이 있으면(LagPrior) 거친 탐색 대신 그 맞춤으로 예측한다.
   PSR이 min_psr에 못 미친 창은 앵커로 쓰지 않는다 (넓은 탐색 폭에서 엉뚱한 봉우리를 잡은 창).

운동: 장갑 압력과 IMU 가속도처럼 서로 다른 센서의 같은 사건을 같은 격자로 리샘플해 상관한다.

상관 최대값의 신뢰도는 PSR(최대값이 나머지 상관값 분포에서 몇 표준편차 위인지)로 잰다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from dlp_sync.anchors import Anchor
from dlp_sync.policy import AudioXcorrPolicy, MotionXcorrPolicy
from dlp_sync.signals import Audio, Series


@dataclass(frozen=True)
class Peak:
    lag: float  # 샘플 단위, 포물선 보간 포함
    psr: float


def correlate(ref: NDArray[np.float64], tgt: NDArray[np.float64]) -> NDArray[np.float64]:
    """c[k] = Σ ref[n] · tgt[n + k]  (k = 0 … len(tgt) - len(ref)). FFT로 계산한다."""
    n = ref.size + tgt.size
    size = 1 << (n - 1).bit_length()
    spec = np.fft.rfft(tgt, size) * np.conj(np.fft.rfft(ref, size))
    full = np.fft.irfft(spec, size)
    return full[: tgt.size - ref.size + 1]


def find_peak(corr: NDArray[np.float64], exclude: int) -> Peak:
    i = int(np.argmax(corr))
    lag = float(i)
    if 0 < i < corr.size - 1:
        a, b, c = corr[i - 1], corr[i], corr[i + 1]
        denom = a - 2 * b + c
        if denom != 0:
            lag += 0.5 * (a - c) / denom
    mask = np.ones(corr.size, dtype=bool)
    mask[max(0, i - exclude) : i + exclude + 1] = False
    side = corr[mask]
    psr = float((corr[i] - side.mean()) / (side.std() or 1e-12)) if side.size > 10 else 0.0
    return Peak(lag, psr)


def _decimate(x: NDArray[np.float32], factor: int) -> NDArray[np.float64]:
    n = x.size // factor * factor
    return x[:n].astype(np.float64).reshape(-1, factor).mean(axis=1)


def _normalize(x: NDArray[np.float64]) -> NDArray[np.float64]:
    x = x - x.mean()
    return x / (x.std() or 1.0)


@dataclass(frozen=True)
class XcorrResult:
    anchors: list[Anchor]
    psr: float  # 창들의 PSR 중앙값


@dataclass(frozen=True)
class LagPrior:
    """정밀 탐색의 출발점: master = offset_ms + stream * clock_scale 과 그 불확실성.

    창의 기준 시각 m에서 탐색 폭(±ms) = base_ms + drift_ppm·1e-6·(m에서 가장 가까운 앵커까지 거리).
    앵커가 없으면 base_ms.
    """

    offset_ms: float
    clock_scale: float
    anchor_master_ms: tuple[float, ...]
    base_ms: float
    drift_ppm: float

    def to_stream(self, master_ms: float) -> float:
        return (master_ms - self.offset_ms) / self.clock_scale

    def to_master(self, stream_ms: float) -> float:
        return self.offset_ms + stream_ms * self.clock_scale

    def half_width_ms(self, master_ms: float) -> float:
        if not self.anchor_master_ms:
            return self.base_ms
        dist = min(abs(master_ms - a) for a in self.anchor_master_ms)
        return self.base_ms + self.drift_ppm * 1e-6 * dist


def audio_anchors(
    reference: Audio,
    target: Audio,
    policy: AudioXcorrPolicy,
    max_offset_ms: float,
    *,
    max_drift_ppm: float = 0.0,
    prior: LagPrior | None = None,
) -> XcorrResult | None:
    """기준 오디오 시각 = master라고 보고 (master, 대상 스트림 시각) 앵커를 만든다.

    prior가 없으면 거친 탐색으로 만든다. max_drift_ppm은 sync.yaml의 드리프트 상한이다.
    """
    if reference.rate != target.rate:
        raise ValueError("두 오디오의 샘플레이트가 같아야 합니다")
    rate = reference.rate
    if prior is None:
        prior = _coarse_prior(reference, target, policy, max_offset_ms, max_drift_ppm)
        if prior is None:
            return None

    # 2. 정밀 탐색: 겹치는 구간에 창을 고르게 둔다. 구간 끝은 예측 불확실성만큼 안쪽으로 줄인다
    span = _overlap(
        reference.start_ms, reference.duration_ms, target.start_ms, target.duration_ms,
        prior, policy.window_ms,
    )  # fmt: skip
    if span is None:
        return None
    anchors, psrs = _refine(
        _Uniform(reference.samples, reference.start_ms, rate),
        _Uniform(target.samples, target.start_ms, rate),
        prior,
        policy.window_ms,
        [float(m) for m in np.linspace(span[0], span[1], policy.windows)],
        policy.min_psr,
        exclude=max(2, rate // 1000),
        label="audio_xcorr",
    )
    if not psrs:
        return None
    return XcorrResult(anchors, float(np.median(psrs)))


@dataclass(frozen=True)
class _Uniform:
    """균일 샘플 신호: 첫 샘플의 시각 start_ms, 샘플레이트 rate."""

    samples: NDArray[np.float32] | NDArray[np.float64]
    start_ms: float
    rate: float

    def index(self, ms: float) -> int:
        return round((ms - self.start_ms) * self.rate / 1000)


def _overlap(
    ref_start: float,
    ref_len: float,
    tgt_start: float,
    tgt_len: float,
    prior: LagPrior,
    window_ms: float,
) -> tuple[float, float] | None:
    """창 시작 기준 시각의 범위: 두 신호가 겹치는 구간을 예측 불확실성만큼 안쪽으로 줄인 것."""
    lo = max(ref_start, prior.to_master(tgt_start))
    hi = min(ref_start + ref_len, prior.to_master(tgt_start + tgt_len))
    lo_ms = lo + prior.half_width_ms(lo)
    hi_ms = hi - window_ms - prior.half_width_ms(hi)
    return (lo_ms, hi_ms) if hi_ms > lo_ms else None


def _refine(
    reference: _Uniform,
    target: _Uniform,
    prior: LagPrior,
    window_ms: float,
    starts_ms: list[float],
    min_psr: float,
    *,
    exclude: int,
    label: str,
) -> tuple[list[Anchor], list[float]]:
    """기준 시각 starts_ms에서 시작하는 창마다 대상의 예측 위치 ± 탐색 폭에서 상관 최대값을 찾는다.

    PSR이 min_psr 미만인 창은 앵커로 쓰지 않는다. (앵커, 모든 창의 PSR)을 돌려준다.
    """
    win = int(window_ms * reference.rate / 1000)
    anchors: list[Anchor] = []
    psrs: list[float] = []
    for m in starts_ms:
        r0 = reference.index(m)
        ref = _normalize(reference.samples[r0 : r0 + win].astype(np.float64))
        pad = int(prior.half_width_ms(m) * target.rate / 1000)
        t_guess = target.index(prior.to_stream(m))
        t0 = max(0, t_guess - pad)
        tgt = target.samples[t0 : t_guess + win + pad].astype(np.float64)
        if ref.size < 16 or tgt.size <= ref.size:
            continue
        peak = find_peak(correlate(ref, _normalize(tgt)), exclude=exclude)
        psrs.append(peak.psr)
        if peak.psr < min_psr:
            continue
        master_ms = reference.start_ms + r0 / reference.rate * 1000
        stream_ms = target.start_ms + (t0 + peak.lag) / target.rate * 1000
        anchors.append(Anchor(master_ms, stream_ms, label))
    return anchors, psrs


def _coarse_prior(
    reference: Audio,
    target: Audio,
    policy: AudioXcorrPolicy,
    max_offset_ms: float,
    max_drift_ppm: float,
) -> LagPrior | None:
    """1. 거친 탐색: 기준의 가운데 구간을 대상 전체에서 찾는다.

    대상이 기준보다 늦게 시작하거나 일찍 끝날 수 있으므로 대상 앞뒤를 max_offset만큼 0으로 덧대
    부분 겹침도 찾을 수 있게 한다. 결과 오프셋은 구간 가운데에서 맞고, 거기서 멀수록 드리프트만큼
    벗어난다.
    """
    rate = reference.rate
    factor = max(1, rate // policy.analysis_rate_hz)
    low_rate = rate / factor
    seg_len = min(int(policy.coarse_segment_ms * rate / 1000), reference.samples.size)
    seg_start = (reference.samples.size - seg_len) // 2
    ref_low = _normalize(_decimate(reference.samples[seg_start : seg_start + seg_len], factor))
    pad = int(max_offset_ms * low_rate / 1000)
    tgt_low = np.concatenate(
        [np.zeros(pad), _normalize(_decimate(target.samples, factor)), np.zeros(pad)]
    )
    if ref_low.size < 16 or tgt_low.size <= ref_low.size:
        return None
    coarse = find_peak(correlate(ref_low, tgt_low), exclude=max(2, int(low_rate / 1000)))
    seg_ref_ms = reference.start_ms + seg_start / rate * 1000
    seg_tgt_ms = target.start_ms + (coarse.lag - pad) / low_rate * 1000
    offset0 = seg_ref_ms - seg_tgt_ms  # master ≈ offset0 + stream
    if abs(offset0) > max_offset_ms:
        return None
    center = seg_ref_ms + seg_len / rate * 1000 / 2
    return LagPrior(offset0, 1.0, (center,), policy.refine_ms, max_drift_ppm)


def motion_anchors(
    reference: Series,
    target: Series,
    policy: MotionXcorrPolicy,
    max_offset_ms: float,
    *,
    max_drift_ppm: float = 0.0,
) -> XcorrResult | None:
    """같은 사건(두드림 등)을 본 두 센서 신호의 앵커.

    1. 전체 상관: 오프셋이 ±max_offset_ms 안인 위치만 본다. 긴 녹화에서 앞 두드림과 뒤 두드림이
       드리프트로 어긋나면 전체 상관 봉우리가 두 개로 갈라지는데, 범위를 두지 않으면 한쪽 끝
       두드림을 다른 쪽 끝 두드림에 맞춘 엉뚱한 위치(수십 분 차이)를 잡을 수 있다.
    2. 창별 정밀 탐색: 겹치는 구간을 window_ms 창으로 나눠, 각 창을 전체 상관 추정 ±(refine_ms +
       max_drift_ppm·녹화 길이) 안에서 다시 찾는다. 사건이 없는 창은 PSR이 낮아 버려지고, 사건이 든
       창마다 앵커가 하나 나온다 (드리프트 추정용). 창의 탐색 범위가 좁아 봉우리 옆 두 번째 두드림이
       PSR을 낮추므로 창 기준은 window_min_psr로 따로 둔다.
       통과한 창이 없으면 전체 상관 앵커 하나를 쓴다.
    PSR은 전체 상관의 값이다.
    """
    rate = policy.rate_hz
    ref_t, ref_v = reference.resample(rate)
    tgt_t, tgt_v = target.resample(rate)
    if ref_t.size < 16 or tgt_t.size < 16:
        return None
    ref_n, tgt_n = _normalize(ref_v), _normalize(tgt_v)
    # 대상 앞뒤로 덧대 기준 전체가 어느 위치에도 놓일 수 있게 하고, 오프셋 범위 안의 위치만 본다.
    # c[k]: 기준 첫 샘플이 대상 시각 tgt_t[0] + (k - pad)/rate 에 놓일 때의 상관
    pad = ref_n.size + int(max_offset_ms * rate / 1000)
    padded = np.concatenate([np.zeros(pad), tgt_n, np.zeros(pad)])
    corr = correlate(ref_n, padded)
    lag0 = pad + (float(ref_t[0]) - float(tgt_t[0])) * rate / 1000  # 오프셋 0인 위치
    k_lo = max(0, int(np.floor(lag0 - max_offset_ms * rate / 1000)))
    k_hi = min(corr.size, int(np.ceil(lag0 + max_offset_ms * rate / 1000)) + 1)
    if k_hi - k_lo < 16:
        return None
    peak = find_peak(corr[k_lo:k_hi], exclude=max(2, int(rate / 20)))
    stream_ms = float(tgt_t[0]) + (k_lo + peak.lag - pad) * 1000 / rate
    master_ms = float(ref_t[0])
    offset0 = master_ms - stream_ms
    if abs(offset0) > max_offset_ms:
        return None
    coarse = [Anchor(master_ms, stream_ms, "motion_xcorr")]

    length = float(ref_t[-1] - ref_t[0])
    prior = LagPrior(offset0, 1.0, (), policy.refine_ms + max_drift_ppm * 1e-6 * length, 0.0)
    ref_u, tgt_u = _Uniform(ref_n, float(ref_t[0]), rate), _Uniform(tgt_n, float(tgt_t[0]), rate)
    span = _overlap(
        ref_u.start_ms, ref_n.size / rate * 1000, tgt_u.start_ms, tgt_n.size / rate * 1000,
        prior, policy.window_ms,
    )  # fmt: skip
    if span is None:
        return XcorrResult(coarse, peak.psr)
    # 겹치는 구간 전체를 덮도록 창을 고르게 둔다 (마지막 창이 구간 끝에 닿는다)
    n = int(np.ceil((span[1] - span[0]) / policy.window_ms)) + 1
    starts = [float(m) for m in np.linspace(span[0], span[1], n)]
    anchors, _ = _refine(
        ref_u, tgt_u, prior, policy.window_ms, starts, policy.window_min_psr,
        exclude=max(2, int(rate / 20)), label="motion_xcorr",
    )  # fmt: skip
    return XcorrResult(anchors or coarse, peak.psr)
