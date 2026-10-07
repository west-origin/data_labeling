"""동기화에 쓰는 신호 적재. 모든 시각은 해당 스트림 시계의 ms다 (WP4).

- `Audio`: 균일 샘플 모노 오디오 (`load_audio`가 영상·오디오 파일에서 16 kHz로 만든다)
- `Series`: 불균일할 수 있는 시계열 (장갑 압력 합 `glove_series`, IMU 가속도 크기 `imu_series`)
- `glove_pressure_prefixes`: 정책에서 장갑 압력 채널 접두사를 꺼낸다

다른 패키지도 이 모듈을 쓴다: 접촉 프리라벨(`dlp_prelabel.runner`)과 검수 동기 재생
(`dlp_review.tasks`, `dlp_review.timeseries`)이 `glove_series`·`imu_series`·`Series`를 가져다 쓴다.
신호 정의를 바꾸면 그쪽 결과도 바뀐다.
"""

# PyAV 타입 스텁이 일부 반환 타입을 비워 두어 이 모듈에서만 해당 경고를 끈다.
# pyright: reportUnknownMemberType=false

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
from numpy.typing import NDArray

from dlp_media.glove import TIME_KEYS
from dlp_media.tables import read_parquet
from dlp_schema.config import repo_root
from dlp_sync.policy import SyncPolicy, load_policy

# 동기화용 오디오 샘플레이트(Hz). 두드림 검출과 오디오 상관의 정밀 탐색이 이 레이트에서 돈다
# (1 샘플 = 0.0625 ms). 거친 탐색은 sync.yaml audio_xcorr.analysis_rate_hz로 낮춰 쓴다
AUDIO_RATE = 16_000


@dataclass(frozen=True)
class Audio:
    """균일 샘플 오디오. 첫 샘플의 스트림 시각은 start_ms.

    Attributes:
        samples: 모노 float32 샘플.
        rate: 샘플레이트 Hz.
        start_ms: 첫 샘플의 스트림 시각 ms (컨테이너의 첫 오디오 프레임 시각, 보통 0).
    """

    samples: NDArray[np.float32]
    rate: int
    start_ms: float = 0.0

    @property
    def duration_ms(self) -> float:
        """오디오 길이 ms (샘플 수 / 레이트)."""
        return self.samples.size / self.rate * 1000

    def index(self, ms: float) -> int:
        """스트림 시각 ms → 가장 가까운 샘플 인덱스 (범위 밖일 수 있다. 호출자가 자른다)."""
        return round((ms - self.start_ms) * self.rate / 1000)


@dataclass(frozen=True)
class Series:
    """불균일할 수 있는 시계열 (장갑 압력 합, IMU 가속도 크기 등).

    Attributes:
        t_ms: 샘플 시각 ms (오름차순, 그 스트림 시계).
        values: 같은 길이의 값.
    """

    t_ms: NDArray[np.float64]
    values: NDArray[np.float64]

    def resample(self, rate_hz: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """`rate_hz` 균일 격자로 선형 보간한다.

        Returns:
            (격자 시각 ms, 보간 값). 격자는 첫 샘플 시각에서 시작해 마지막 샘플 시각 앞에서 끝난다.
        """
        grid = np.arange(self.t_ms[0], self.t_ms[-1], 1000 / rate_hz)
        return grid, np.interp(grid, self.t_ms, self.values)


def load_audio(path: Path, rate: int = AUDIO_RATE) -> Audio | None:
    """영상·오디오 파일의 첫 오디오 트랙을 모노 float32로. 오디오가 없으면 None.

    Args:
        path: 로컬 파일 (MP4·WAV 등 PyAV가 여는 형식).
        rate: 리샘플할 샘플레이트 Hz.

    Returns:
        `Audio`. `start_ms`는 첫 디코딩 프레임의 시각(없으면 0). 트랙은 있지만 프레임이 없으면
        길이 0인 `Audio`.
    """
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
        # None을 넣어 리샘플러 내부 버퍼에 남은 샘플을 비운다
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().reshape(-1).astype(np.float32))
    samples = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    return Audio(samples, rate, start_ms or 0.0)


def glove_pressure_prefixes(policy: SyncPolicy) -> tuple[str, ...]:
    """불러온 동기화 정책에서 장갑 압력 채널 접두사 (sync.yaml glove.pressure_prefixes).

    장갑 신호를 쓰는 다른 단계(접촉 프리라벨, 검수 동기 재생)는 이 값을 glove_series에 넘긴다.
    """
    return policy.glove.pressure_prefixes


def glove_series(path: Path, pressure_prefixes: Sequence[str] | None = None) -> Series:
    """정규화된 장갑 Parquet → 압력 채널 합.

    압력 채널은 이름이 pressure_prefixes(sync.yaml glove.pressure_prefixes) 중 하나로 시작하는
    열이다. 시각 열과 IMU·온도 같은 다른 채널은 합에 넣지 않는다.

    호출하는 쪽이 불러온 정책에서 접두사를 넘겨야 한다 (glove_pressure_prefixes). 주지 않으면
    호환을 위해 저장소의 sync.yaml을 읽고 DeprecationWarning을 낸다 (다른 설정 디렉터리를 쓰는
    실행에서 엉뚱한 값을 쓸 수 있다).

    압력 채널이 하나도 없으면 ValueError를 낸다. 다른 채널(IMU·온도 등)을 대신 합치지 않는다:
    두드림·접촉 신호가 아니어서 동기화·접촉 판정이 조용히 틀어진다. 메시지에 파일의 채널 목록과
    정책 위치를 적어, 장갑 기종의 채널 이름에 맞게 접두사를 더하도록 안내한다.

    Args:
        path: 정규화된 장갑 Parquet (`dlp_media` 수집 결과, 시각 열 `t_ms`).
        pressure_prefixes: 압력 채널 이름 접두사. None이면 저장소 정책(사용 중단 경로).

    Returns:
        `Series(t_ms, 압력 채널 합)`. 채널은 이름순으로 더한다.

    Raises:
        ValueError: 접두사가 비었거나 압력 채널이 없을 때.
    """
    if pressure_prefixes is None:
        warnings.warn(
            "glove_series: pressure_prefixes를 넘기지 않아 저장소 sync.yaml을 읽습니다. "
            "불러온 정책의 glove_pressure_prefixes(policy)를 넘기세요",
            DeprecationWarning,
            stacklevel=2,
        )
        prefixes = _repo_pressure_prefixes()
    else:
        prefixes = tuple(pressure_prefixes)
    if not prefixes:
        raise ValueError("glove_series: 압력 채널 접두사가 비어 있습니다")
    cols, _ = read_parquet(path)
    # 시각 열(TIME_KEYS)은 이름이 접두사와 맞아도 뺀다
    channels = [
        np.asarray(v, dtype=np.float64)
        for k, v in sorted(cols.items())
        if k not in TIME_KEYS and k.startswith(prefixes)
    ]
    if not channels:
        others = ", ".join(sorted(k for k in cols if k not in TIME_KEYS)) or "없음"
        raise ValueError(
            f"{path.name}: 압력 채널({', '.join(p + '*' for p in prefixes)})이 없습니다. "
            f"파일의 채널: {others}. 이 장갑의 압력 채널 이름에 맞게 "
            "config/policies/sync.yaml glove.pressure_prefixes를 고치세요"
        )
    return Series(np.asarray(cols["t_ms"], dtype=np.float64), np.sum(channels, axis=0))


def _repo_pressure_prefixes() -> tuple[str, ...]:
    """이 소스가 든 저장소의 `config/policies/sync.yaml`에서 접두사를 읽는다 (사용 중단 경로용)."""
    root = repo_root(Path(__file__).parent)
    return glove_pressure_prefixes(load_policy(root / "config" / "policies" / "sync.yaml"))


def imu_series(path: Path) -> Series:
    """정규화된 IMU Parquet → 가속도 크기에서 중력(중앙값)을 뺀 절댓값.

    Args:
        path: 정규화된 IMU Parquet (열 `t_ms`, `ax`, `ay`, `az`).

    Returns:
        `Series(t_ms, |‖a‖ - median(‖a‖)|)`. 자세와 무관하게 충격(두드림) 크기만 남긴다.

    Raises:
        KeyError: 필요한 열이 없을 때.
    """
    cols, _ = read_parquet(path)
    acc = np.stack([np.asarray(cols[c], dtype=np.float64) for c in ("ax", "ay", "az")], axis=1)
    mag = np.linalg.norm(acc, axis=1)
    return Series(np.asarray(cols["t_ms"], dtype=np.float64), np.abs(mag - np.median(mag)))
