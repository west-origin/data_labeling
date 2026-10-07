"""상호상관으로 오프셋 찾기.

오디오: 두 마이크가 같은 주변 소리를 듣는다는 점을 쓴다.
1. 거친 탐색: 낮은 샘플레이트에서 기준 오디오 한 구간을 대상 전체와 FFT 상관해 오프셋을 찾는다.
2. 정밀 탐색: 겹치는 구간에 창을 여러 개 두고, 각 창을 원래 샘플레이트에서 거친 추정 ±refine_ms
   안에서 다시 상관한다. 최대값은 포물선 보간으로 샘플 이하까지 구한다. 창마다 앵커가 하나 나온다.

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


def audio_anchors(
    reference: Audio, target: Audio, policy: AudioXcorrPolicy, max_offset_ms: float
) -> XcorrResult | None:
    """기준 오디오 시각 = master라고 보고 (master, 대상 스트림 시각) 앵커를 만든다."""
    if reference.rate != target.rate:
        raise ValueError("두 오디오의 샘플레이트가 같아야 합니다")
    rate = reference.rate
    factor = max(1, rate // policy.analysis_rate_hz)
    low_rate = rate / factor

    # 1. 거친 탐색: 기준의 가운데 구간을 대상 전체에서 찾는다. 대상이 기준보다 늦게 시작하거나
    #    일찍 끝날 수 있으므로 대상 앞뒤를 max_offset만큼 0으로 덧대 부분 겹침도 찾을 수 있게 한다.
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

    # 2. 정밀 탐색: 겹치는 구간에 창을 고르게 둔다
    win = int(policy.window_ms * rate / 1000)
    pad = int(policy.refine_ms * rate / 1000)
    lo_ms = max(reference.start_ms, target.start_ms + offset0) + policy.refine_ms
    hi_ms = (
        min(
            reference.start_ms + reference.duration_ms,
            target.start_ms + target.duration_ms + offset0,
        )
        - policy.window_ms
        - policy.refine_ms
    )
    if hi_ms <= lo_ms:
        return None
    anchors: list[Anchor] = []
    psrs: list[float] = []
    for m in np.linspace(lo_ms, hi_ms, policy.windows):
        r0 = reference.index(float(m))
        ref = _normalize(reference.samples[r0 : r0 + win].astype(np.float64))
        t_guess = target.index(float(m) - offset0)
        t0 = max(0, t_guess - pad)
        tgt = target.samples[t0 : t_guess + win + pad].astype(np.float64)
        if tgt.size <= ref.size:
            continue
        peak = find_peak(correlate(ref, _normalize(tgt)), exclude=max(2, rate // 1000))
        master_ms = reference.start_ms + r0 / rate * 1000
        stream_ms = target.start_ms + (t0 + peak.lag) / rate * 1000
        anchors.append(Anchor(master_ms, stream_ms, "audio_xcorr"))
        psrs.append(peak.psr)
    if not anchors:
        return None
    return XcorrResult(anchors, float(np.median(psrs)))


def motion_anchors(
    reference: Series, target: Series, policy: MotionXcorrPolicy, max_offset_ms: float
) -> XcorrResult | None:
    """같은 사건을 본 두 센서 신호의 오프셋 (드리프트는 추정하지 않는다)."""
    ref_t, ref_v = reference.resample(policy.rate_hz)
    tgt_t, tgt_v = target.resample(policy.rate_hz)
    ref_n, tgt_n = _normalize(ref_v), _normalize(tgt_v)
    # 대상 앞뒤로 max_offset만큼 0을 덧대 기준 전체가 어느 위치에도 놓일 수 있게 한다
    pad = int(max_offset_ms * policy.rate_hz / 1000)
    padded = np.concatenate([np.zeros(pad), tgt_n, np.zeros(pad + ref_n.size)])
    peak = find_peak(correlate(ref_n, padded), exclude=max(2, int(policy.rate_hz / 20)))
    stream_ms = tgt_t[0] + (peak.lag - pad) * 1000 / policy.rate_hz
    master_ms = float(ref_t[0])
    if abs(master_ms - stream_ms) > max_offset_ms:
        return None
    return XcorrResult([Anchor(master_ms, stream_ms, "motion_xcorr")], peak.psr)
