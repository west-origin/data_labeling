# h5py에는 타입 스텁이 없어 이 모듈에서만 알 수 없는 타입 경고를 끈다.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

"""촉각장갑 데이터 수집. Parquet·HDF5를 읽어 정규화된 Parquet(t_ms + 채널 열)으로 바꾼다.

원 샘플레이트를 그대로 유지한다. 영상 프레임 시각으로의 리샘플은 동기화 이후 단계에서 한다.

WP3, ADR 0003. 수집(`ingest`)이 장갑 스트림(glove_left·glove_right)마다 `read_glove`로 읽어
원본 버킷 `sessions/<세션>/derived/<스트림>.parquet`에 정규화본을 둔다 (Stream.uri가 가리킨다).
시각 `t_ms`는 장갑 기기 시계의 ms다 (마스터 타임라인과의 오프셋·드리프트는 `dlp sync`가 정한다).

- `GloveData`: 시각 + 채널 배열 (검증 포함), `write`로 정규화 Parquet.
- `read_glove`: .parquet / .h5 / .hdf5 → `GloveData`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray

from dlp_media.tables import read_parquet, write_parquet

# 시각 열로 인정하는 이름 (앞의 것이 우선). 나머지 시각 열은 채널에서도 뺀다.
TIME_KEYS = ("t_ms", "timestamp_ms")


@dataclass(frozen=True)
class GloveData:
    """정규화된 장갑 데이터.

    Raises (생성 시):
        ValueError: 샘플 2개 미만, 시각이 순증가가 아님, 채널 길이가 다름, NaN·무한대 값,
            채널이 없음.
    """

    # 샘플 시각 (장갑 시계 ms, 순증가)
    t_ms: NDArray[np.float64]
    # 채널 이름 → 값 (t_ms와 같은 길이, 유한값). 단위는 기기마다 다르다 (압력 등).
    channels: dict[str, NDArray[np.float64]]
    # 원 형식 ("parquet" | "hdf5")
    source: str

    def __post_init__(self) -> None:
        """길이·순서·유한값을 검사한다."""
        if self.t_ms.size < 2:
            raise ValueError("장갑 샘플이 너무 적습니다")
        if np.any(np.diff(self.t_ms) <= 0):
            raise ValueError("장갑 시각이 증가하지 않습니다")
        for name, values in self.channels.items():
            if values.shape != self.t_ms.shape:
                raise ValueError(f"채널 {name}의 길이가 시각과 다릅니다")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"채널 {name}에 유한하지 않은 값이 있습니다")
        if not self.channels:
            raise ValueError("장갑 채널이 없습니다")

    @property
    def sample_rate_hz(self) -> float:
        """샘플레이트 (Hz) = 1000 / 샘플 간격 중앙값(ms)."""
        return float(1000 / np.median(np.diff(self.t_ms)))

    def write(self, path: Path) -> None:
        """정규화 Parquet: 열 t_ms + 채널들, 메타 source·sample_rate_hz."""
        write_parquet(
            path,
            {"t_ms": self.t_ms, **self.channels},
            {"source": self.source, "sample_rate_hz": str(self.sample_rate_hz)},
        )


def read_glove(path: Path) -> GloveData:
    """장갑 파일을 읽는다 (확장자로 형식을 정한다).

    Raises:
        ValueError: 지원하지 않는 형식, 시각 열 없음, 또는 `GloveData` 검증 실패.
    """
    if path.suffix == ".parquet":
        cols, _ = read_parquet(path)
        return _from_columns(cols, "parquet")
    if path.suffix in (".h5", ".hdf5"):
        return _from_columns(_read_h5(path), "hdf5")
    raise ValueError(f"지원하지 않는 장갑 파일 형식: {path.suffix}")


def _read_h5(path: Path) -> dict[str, NDArray[Any]]:
    """최상위 데이터셋을 읽는다. 2차원 데이터셋 (N, C)은 <이름>_<i> 채널 C개로 펼친다.

    그룹·3차원 이상 데이터셋은 무시한다.
    """
    out: dict[str, NDArray[Any]] = {}
    with h5py.File(path, "r") as f:
        for name in f:
            obj: Any = f[name]
            if not isinstance(obj, h5py.Dataset):
                continue
            arr: NDArray[Any] = np.asarray(obj[()])
            if arr.ndim == 1:
                out[str(name)] = arr
            elif arr.ndim == 2:
                for i in range(arr.shape[1]):
                    out[f"{name}_{i}"] = arr[:, i]
    return out


def _from_columns(cols: dict[str, NDArray[Any]], source: str) -> GloveData:
    """열 사전 → `GloveData`.

    시각 열은 TIME_KEYS 중 처음 찾은 것. 채널은 시각 열이 아닌 숫자형 열 전부 (문자열 열 등은
    버린다).
    """
    time_key = next((k for k in TIME_KEYS if k in cols), None)
    if time_key is None:
        raise ValueError(f"장갑 데이터에 시각 열({' 또는 '.join(TIME_KEYS)})이 없습니다")
    channels = {
        k: np.asarray(v, dtype=np.float64)
        for k, v in cols.items()
        # 남은 시각 열도 채널이 아니다 (압력 합 등에 섞이지 않게)
        if k not in TIME_KEYS and np.issubdtype(np.asarray(v).dtype, np.number)
    }
    return GloveData(np.asarray(cols[time_key], dtype=np.float64), channels, source)
