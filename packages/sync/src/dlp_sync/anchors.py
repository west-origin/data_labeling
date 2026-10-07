"""앵커와 시계 맞춤.

앵커 하나는 같은 물리 사건을 두 시계로 본 한 쌍이다: (기준 시각 master_ms, 스트림 시각 stream_ms).
계약과 같은 관계 master = offset_ms + stream_ms * clock_scale을 최소제곱으로 맞춘다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Anchor:
    master_ms: float
    stream_ms: float
    label: str = ""


@dataclass(frozen=True)
class ClockFit:
    offset_ms: float
    clock_scale: float
    residual_rms_ms: float
    n_anchors: int
    span_ms: float
    drift_estimated: bool

    def to_master(self, stream_ms: float) -> float:
        return self.offset_ms + stream_ms * self.clock_scale


class FitError(ValueError):
    pass


def fit_clock(anchors: list[Anchor], *, min_drift_span_ms: float, max_drift_ppm: float) -> ClockFit:
    """앵커 간격이 충분하면 오프셋과 드리프트를, 아니면 오프셋만(드리프트 0) 맞춘다."""
    if not anchors:
        raise FitError("앵커가 없습니다")
    m = np.array([a.master_ms for a in anchors])
    s = np.array([a.stream_ms for a in anchors])
    span = float(s.max() - s.min())
    if len(anchors) >= 2 and span >= min_drift_span_ms:
        design = np.stack([np.ones_like(s), s], axis=1)
        (offset, scale), *_ = np.linalg.lstsq(design, m, rcond=None)
        if abs(scale - 1) * 1e6 > max_drift_ppm:
            raise FitError(f"드리프트 추정 {abs(scale - 1) * 1e6:.0f} ppm이 허용 범위를 넘습니다")
        drift = True
    else:
        offset, scale, drift = float(np.median(m - s)), 1.0, False
    residual = m - (offset + s * scale)
    return ClockFit(
        offset_ms=float(offset),
        clock_scale=float(scale),
        residual_rms_ms=float(np.sqrt(np.mean(residual**2))),
        n_anchors=len(anchors),
        span_ms=span,
        drift_estimated=drift,
    )
