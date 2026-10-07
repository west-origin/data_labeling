"""동기화에 쓰는 신호 적재. 모든 시각은 해당 스트림 시계의 ms다."""

# PyAV 타입 스텁이 일부 반환 타입을 비워 두어 이 모듈에서만 해당 경고를 끈다.
# pyright: reportUnknownMemberType=false

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
from numpy.typing import NDArray

from dlp_media.glove import TIME_KEYS
from dlp_media.tables import read_parquet
from dlp_schema.config import repo_root
from dlp_sync.policy import load_policy

AUDIO_RATE = 16_000


@dataclass(frozen=True)
class Audio:
    """균일 샘플 오디오. 첫 샘플의 스트림 시각은 start_ms."""

    samples: NDArray[np.float32]
    rate: int
    start_ms: float = 0.0

    @property
    def duration_ms(self) -> float:
        return self.samples.size / self.rate * 1000

    def index(self, ms: float) -> int:
        return round((ms - self.start_ms) * self.rate / 1000)


@dataclass(frozen=True)
class Series:
    """불균일할 수 있는 시계열 (장갑 압력 합, IMU 가속도 크기 등)."""

    t_ms: NDArray[np.float64]
    values: NDArray[np.float64]

    def resample(self, rate_hz: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        grid = np.arange(self.t_ms[0], self.t_ms[-1], 1000 / rate_hz)
        return grid, np.interp(grid, self.t_ms, self.values)


def load_audio(path: Path, rate: int = AUDIO_RATE) -> Audio | None:
    """영상·오디오 파일의 첫 오디오 트랙을 모노 float32로. 오디오가 없으면 None."""
    with av.open(str(path)) as c:
        if not c.streams.audio:
            return None
        stream = c.streams.audio[0]
        resampler = av.AudioResampler(format="flt", layout="mono", rate=rate)
        chunks: list[NDArray[np.float32]] = []
        start_ms: float | None = None
        for frame in c.decode(stream):
            if start_ms is None and frame.time is not None:
                start_ms = float(frame.time) * 1000
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1).astype(np.float32))
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().reshape(-1).astype(np.float32))
    samples = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    return Audio(samples, rate, start_ms or 0.0)


def glove_series(path: Path, pressure_prefixes: Sequence[str] | None = None) -> Series:
    """정규화된 장갑 Parquet → 압력 채널 합.

    압력 채널은 이름이 pressure_prefixes(sync.yaml glove.pressure_prefixes) 중 하나로 시작하는
    열이다. 시각 열과 IMU·온도 같은 다른 채널은 합에 넣지 않는다. 접두사를 주지 않으면 저장소의
    sync.yaml에서 읽는다.
    """
    prefixes = tuple(pressure_prefixes) if pressure_prefixes is not None else _pressure_prefixes()
    cols, _ = read_parquet(path)
    channels = [
        np.asarray(v, dtype=np.float64)
        for k, v in sorted(cols.items())
        if k not in TIME_KEYS and k.startswith(prefixes)
    ]
    if not channels:
        raise ValueError(f"{path.name}: 압력 채널({', '.join(prefixes)}*)이 없습니다")
    return Series(np.asarray(cols["t_ms"], dtype=np.float64), np.sum(channels, axis=0))


def _pressure_prefixes() -> tuple[str, ...]:
    root = repo_root(Path(__file__).parent)
    return load_policy(root / "config" / "policies" / "sync.yaml").glove.pressure_prefixes


def imu_series(path: Path) -> Series:
    """정규화된 IMU Parquet → 가속도 크기에서 중력(중앙값)을 뺀 절댓값."""
    cols, _ = read_parquet(path)
    acc = np.stack([np.asarray(cols[c], dtype=np.float64) for c in ("ax", "ay", "az")], axis=1)
    mag = np.linalg.norm(acc, axis=1)
    return Series(np.asarray(cols["t_ms"], dtype=np.float64), np.abs(mag - np.median(mag)))
