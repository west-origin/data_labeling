"""시계와 신호를 아는 촉각장갑 기록 (내보내기 촉각 채널 시험, ADR 0069).

장갑 채널마다 마스터 시각의 매끄러운 함수(양수 사인)를 정해 두고, 장갑 기기 시계의 샘플 시각에서 그
값을 찍는다. 그래서 어느 마스터 시각에서든 정답 값을 식으로 구할 수 있고(`GloveTruth.truth`),
동기화 값(오프셋·드리프트)을 쓰지 않거나 틀리게 쓰면 정답과 어긋난다. 동기화 시나리오
(`sync.py`)의 장갑 신호는 두드림 펄스와 잡음이라 프레임 시각의 정답 값을 식으로 구할 수 없어
따로 둔다. 시계 정답은 같은 `ClockTruth`(master = offset_ms + stream_ms * clock_scale)를 쓴다.

정답
- `GloveTruth.clock`: 장갑 시계 → 마스터 시각. 녹화는 장갑 시각 0(마스터 `clock.offset_ms`)에서
  시작해 `duration_ms`(장갑 시계) 동안이다. 그 밖의 마스터 시각에는 기록이 없다.
- `GloveTruth.truth(마스터 시각)`: 채널 값. 샘플 사이의 선형 보간은 이 값과 `linear_error_bound`
  안에서 같다.
- `GloveTruth.gaps_master_ms`: 샘플을 뺀 구간(마스터 시각, 열린 구간). 센서 끊김을 흉내 낸다.

파일은 `write`가 정규화 장갑 Parquet(열 `t_ms` + 채널, `dlp_media.glove`가 원본 버킷 derived/에
두는 것과 같은 열)으로 쓴다. 메모리에서만 쓰고 `generate_all`은 쓰지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from dlp_fixtures.io import write_parquet
from dlp_fixtures.sync import GLOVE_RATE, ClockTruth

# 기본 채널 이름 (동기화 시나리오의 장갑과 같은 압력 채널 5개)
PRESSURE_CHANNELS = tuple(f"pressure_{i}" for i in range(5))


@dataclass(frozen=True)
class GloveTruth:
    """장갑 하나의 기록과 정답.

    Attributes:
        clock: 장갑 시계 → 마스터 시각 정답.
        channels: 채널 이름 (열 순서).
        t_ms: 샘플 시각 (장갑 시계 ms, 순증가). `gaps_master_ms` 안의 샘플은 빠져 있다.
        gaps_master_ms: 샘플을 뺀 구간 (마스터 ms, 양끝 제외).
        level, amplitude, period_ms, phase: 채널마다의 신호 값
            (값 = level + amplitude * sin(2π * 마스터 시각 / period_ms + phase), 늘 양수).
    """

    clock: ClockTruth
    channels: tuple[str, ...]
    t_ms: NDArray[np.float64] = field(repr=False)
    gaps_master_ms: tuple[tuple[float, float], ...]
    level: NDArray[np.float64] = field(repr=False)
    amplitude: NDArray[np.float64] = field(repr=False)
    period_ms: NDArray[np.float64] = field(repr=False)
    phase: NDArray[np.float64] = field(repr=False)

    @property
    def master_ms(self) -> NDArray[np.float64]:
        """샘플의 마스터 시각 ms (정답 시계로 옮긴 것)."""
        return self.clock.to_master(self.t_ms)

    def truth(self, master_ms: NDArray[np.float64] | float) -> NDArray[np.float64]:
        """마스터 시각의 정답 채널 값.

        Returns:
            (시각 수, 채널 수) 배열. 기록 범위 밖 시각에도 식의 값을 돌려준다 (범위 판정은
            호출자가 `master_ms`의 처음·끝과 `gaps_master_ms`로 한다).
        """
        m = np.atleast_1d(np.asarray(master_ms, dtype=np.float64))[:, None]
        return self.level + self.amplitude * np.sin(2 * np.pi * m / self.period_ms + self.phase)

    def linear_error_bound(self, interval_ms: float) -> float:
        """간격 `interval_ms`(마스터 ms)인 두 샘플 사이를 선형 보간할 때 정답과의 최대 차이.

        사인의 2차 도함수 상한으로 구한다: 진폭 * (2π / 주기)² * 간격² / 8 (채널 중 최대).
        """
        curvature = self.amplitude * (2 * np.pi / self.period_ms) ** 2
        return float(np.max(curvature) * interval_ms**2 / 8)

    def columns(self) -> dict[str, NDArray[np.float64]]:
        """정규화 장갑 Parquet의 열: `t_ms`(장갑 시계)와 채널마다 그 샘플 시각의 정답 값."""
        values = self.truth(self.master_ms)
        return {"t_ms": self.t_ms, **{c: values[:, i] for i, c in enumerate(self.channels)}}

    def write(self, path: Path) -> None:
        """정규화 장갑 Parquet으로 쓴다 (상위 디렉터리를 만든다)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        write_parquet(path, self.columns(), {"clock": "glove"})


def generate_glove_truth(
    seed: int = 0,
    *,
    clock: ClockTruth,
    duration_ms: float,
    channels: tuple[str, ...] = PRESSURE_CHANNELS,
    rate_hz: float = GLOVE_RATE,
    gaps_master_ms: tuple[tuple[float, float], ...] = (),
) -> GloveTruth:
    """정답을 아는 장갑 기록 하나를 만든다.

    Args:
        seed: 난수 seed (채널 신호의 높이·진폭·주기·위상이 여기서 정해진다).
        clock: 장갑 시계 정답. 오프셋이 곧 녹화 시작의 마스터 시각이다.
        duration_ms: 녹화 길이 (장갑 시계 ms).
        channels: 채널 이름. 채널마다 신호가 다르다.
        rate_hz: 샘플레이트 Hz (장갑 시계).
        gaps_master_ms: 샘플을 뺄 구간 (마스터 ms, 양끝 제외). 구간 양쪽에 샘플이 남아야 한다.

    Returns:
        `GloveTruth`. 주기는 400~900 ms라 100 Hz 샘플 사이 선형 보간 오차가 0.003 미만이고,
        수백 ms의 오프셋을 빼먹으면 값이 진폭만큼 어긋난다.
    """
    rng = np.random.default_rng(seed)
    n = len(channels)
    t = np.arange(0, duration_ms, 1000 / rate_hz)
    master = clock.to_master(t)
    keep = np.ones(t.size, dtype=bool)
    for a, b in gaps_master_ms:
        keep &= ~((master > a) & (master < b))
    return GloveTruth(
        clock=clock,
        channels=channels,
        t_ms=t[keep],
        gaps_master_ms=gaps_master_ms,
        level=rng.uniform(1.0, 2.0, n),
        amplitude=rng.uniform(0.3, 0.9, n),
        period_ms=rng.uniform(400.0, 900.0, n),
        phase=rng.uniform(0.0, 2 * np.pi, n),
    )
