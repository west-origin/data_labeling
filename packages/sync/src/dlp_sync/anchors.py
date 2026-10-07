"""앵커와 시계 맞춤 (WP4, ADR 0004·0028).

앵커 하나는 같은 물리 사건을 두 시계로 본 한 쌍이다: (기준 시각 master_ms, 스트림 시각 stream_ms).
계약과 같은 관계 master = offset_ms + stream_ms * clock_scale을 최소제곱으로 맞춘다.

모든 동기화 방법(슬레이트·두드림·오디오 상관·운동 상관)은 결국 앵커 목록을 만들고, 이 모듈의
`fit_clock`으로 오프셋·배율을 얻는다. 드리프트 추정 여부와 상한은 `sync.yaml`의 `min_drift_span_ms`,
`max_drift_ppm`(슬레이트는 `slate.min_drift_span_ms`)에서 온다.

- `Anchor`: (기준 ms, 스트림 ms, 출처 표시) 한 쌍
- `ClockFit`: 맞춘 결과 (오프셋, 배율, 잔차, 앵커 수, 앵커 범위, 드리프트 추정 여부)
- `fit_clock`: 앵커 → `ClockFit`
- `FitError`: 맞출 수 없을 때 (앵커 없음, 드리프트 상한 초과)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Anchor:
    """같은 사건을 기준 시계와 대상 스트림 시계로 본 시각 쌍.

    Attributes:
        master_ms: 기준(바디캠) 시계 시각 ms. 오디오 상관에서는 기준 오디오의 스트림 시각이다
            (바디캠은 기준이므로 그 자체가 마스터 시각).
        stream_ms: 대상 스트림 자기 시계 시각 ms.
        label: 출처 표시 (슬레이트 payload, `"tap1"`/`"tap2"`, `"audio_xcorr"`, `"motion_xcorr"`).
            보고서에만 쓰고 계산에는 쓰지 않는다.
    """

    master_ms: float
    stream_ms: float
    label: str = ""


@dataclass(frozen=True)
class ClockFit:
    """앵커로 맞춘 시계 관계 master = offset_ms + stream * clock_scale.

    Attributes:
        offset_ms: 스트림 시각 0의 마스터 시각 ms.
        clock_scale: 배율. 1이면 드리프트 없음 (드리프트를 추정하지 않으면 정확히 1.0).
        residual_rms_ms: 앵커 잔차(master - 예측)의 RMS ms. 신뢰도 계산에 쓴다.
        n_anchors: 사용한 앵커 수.
        span_ms: 앵커 스트림 시각의 범위(최대 - 최소) ms.
        drift_estimated: 배율까지 맞췄는지. False면 오프셋만 맞췄다.
    """

    offset_ms: float
    clock_scale: float
    residual_rms_ms: float
    n_anchors: int
    span_ms: float
    drift_estimated: bool

    def to_master(self, stream_ms: float) -> float:
        """스트림 시각 ms → 마스터 시각 ms (사람 조정값은 넣지 않는다)."""
        return self.offset_ms + stream_ms * self.clock_scale


class FitError(ValueError):
    """앵커로 시계를 맞출 수 없을 때 (앵커 없음, 드리프트가 `max_drift_ppm`을 넘음).

    `pipeline._try`가 잡아 그 방법의 시도를 신뢰도 0으로 기록하고 다음 방법으로 넘어간다.
    """


def fit_clock(anchors: list[Anchor], *, min_drift_span_ms: float, max_drift_ppm: float) -> ClockFit:
    """앵커 간격이 충분하면 오프셋과 드리프트를, 아니면 오프셋만(드리프트 0) 맞춘다.

    - 앵커가 2개 이상이고 스트림 시각 범위가 `min_drift_span_ms` 이상: 1차 최소제곱
      master = offset + scale * stream. 앵커 간격이 짧으면 앵커 오차가 기울기 오차로 크게
      번지므로(오차 / 간격) 범위 조건을 둔다.
    - 그 밖: 배율 1로 고정하고 오프셋 = 중앙값(master - stream). 평균이 아닌 중앙값이라
      튀는 앵커 하나에 덜 흔들린다.

    Args:
        anchors: 앵커 목록 (순서 무관).
        min_drift_span_ms: 드리프트를 추정할 최소 앵커 범위 ms. `math.inf`를 주면 늘 오프셋만
            맞춘다.
        max_drift_ppm: 허용 드리프트 상한 ppm (`sync.yaml max_drift_ppm`).

    Returns:
        `ClockFit`.

    Raises:
        FitError: 앵커가 없거나, 추정 드리프트 |scale - 1|·1e6이 `max_drift_ppm`을 넘을 때
            (잘못 짝지은 앵커일 가능성이 크다).
    """
    if not anchors:
        raise FitError("앵커가 없습니다")
    m = np.array([a.master_ms for a in anchors])
    s = np.array([a.stream_ms for a in anchors])
    span = float(s.max() - s.min())
    if len(anchors) >= 2 and span >= min_drift_span_ms:
        # 설계 행렬 [1, s]: 열 0의 계수가 offset, 열 1의 계수가 scale
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
