# h5py에는 타입 스텁이 없어 이 모듈에서만 알 수 없는 타입 경고를 끈다.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

"""촉각장갑 데이터 수집. Parquet·HDF5를 읽어 정규화된 Parquet(t_ms + 채널 열)으로 바꾼다.

원 샘플레이트를 그대로 유지한다. 영상 프레임 시각으로의 리샘플은 동기화 이후 단계에서 한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray

from dlp_media.tables import read_parquet, write_parquet

TIME_KEYS = ("t_ms", "timestamp_ms")


@dataclass(frozen=True)
class GloveData:
    t_ms: NDArray[np.float64]
    channels: dict[str, NDArray[np.float64]]
    source: str

    def __post_init__(self) -> None:
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
        return float(1000 / np.median(np.diff(self.t_ms)))

    def write(self, path: Path) -> None:
        write_parquet(
            path,
            {"t_ms": self.t_ms, **self.channels},
            {"source": self.source, "sample_rate_hz": str(self.sample_rate_hz)},
        )


def read_glove(path: Path) -> GloveData:
    if path.suffix == ".parquet":
        cols, _ = read_parquet(path)
        return _from_columns(cols, "parquet")
    if path.suffix in (".h5", ".hdf5"):
        return _from_columns(_read_h5(path), "hdf5")
    raise ValueError(f"지원하지 않는 장갑 파일 형식: {path.suffix}")


def _read_h5(path: Path) -> dict[str, NDArray[Any]]:
    """최상위 데이터셋을 읽는다. 2차원 데이터셋 (N, C)은 <이름>_<i> 채널 C개로 펼친다."""
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
    time_key = next((k for k in TIME_KEYS if k in cols), None)
    if time_key is None:
        raise ValueError(f"장갑 데이터에 시각 열({' 또는 '.join(TIME_KEYS)})이 없습니다")
    channels = {
        k: np.asarray(v, dtype=np.float64)
        for k, v in cols.items()
        if k != time_key and np.issubdtype(np.asarray(v).dtype, np.number)
    }
    return GloveData(np.asarray(cols[time_key], dtype=np.float64), channels, source)
