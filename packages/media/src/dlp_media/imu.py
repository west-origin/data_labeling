"""IMU 추출. 카메라 기종별 추출기를 등록해 쓴다.

- GPMF (GoPro): MP4의 `gpmd` 데이터 트랙에 든 KLV 구조를 해석한다. 패킷 하나(보통 1초)에 든
  샘플을 패킷 구간에 균등하게 배치한다. 축 순서는 ORIN 태그를 따르고, 없으면 HERO5 이후 기본값
  ZXY로 본다. 소문자 축은 부호가 반대다.
- 사이드카: 카메라 앱이 따로 내보낸 CSV·Parquet (열: t_ms, ax, ay, az, gx, gy, gz).

모든 시각은 해당 카메라 스트림 시계의 ms다. 바디캠 내장 IMU면 바디캠과 같은 시계다.
"""

# pyright: reportUnknownMemberType=false

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import av
import numpy as np
from numpy.typing import NDArray

from dlp_media.probe import MediaInfo, to_fraction
from dlp_media.tables import read_parquet, write_parquet

IMU_COLUMNS = ("ax", "ay", "az", "gx", "gy", "gz")


@dataclass(frozen=True)
class ImuData:
    t_ms: NDArray[np.float64]
    acc: NDArray[np.float64]  # (N, 3) m/s²
    gyro: NDArray[np.float64]  # (N, 3) rad/s
    source: str

    def __post_init__(self) -> None:
        n = self.t_ms.size
        if self.acc.shape != (n, 3) or self.gyro.shape != (n, 3):
            raise ValueError("IMU 배열 길이가 맞지 않습니다")
        if n and np.any(np.diff(self.t_ms) <= 0):
            raise ValueError("IMU 시각이 증가하지 않습니다")

    @property
    def sample_rate_hz(self) -> float:
        return float(1000 / np.median(np.diff(self.t_ms))) if self.t_ms.size > 1 else 0.0

    def write(self, path: Path) -> None:
        cols = {"t_ms": self.t_ms}
        for i, name in enumerate(IMU_COLUMNS[:3]):
            cols[name] = self.acc[:, i]
        for i, name in enumerate(IMU_COLUMNS[3:]):
            cols[name] = self.gyro[:, i]
        write_parquet(
            path, cols, {"source": self.source, "sample_rate_hz": str(self.sample_rate_hz)}
        )


class ImuExtractor(Protocol):
    name: str

    def can_handle(self, info: MediaInfo) -> bool: ...
    def extract(self, path: Path) -> ImuData: ...


# ---------------------------------------------------------------- GPMF


@dataclass(frozen=True)
class Klv:
    key: str
    type: str
    size: int
    repeat: int
    data: bytes
    children: tuple[Klv, ...] = ()

    def find(self, key: str) -> Klv | None:
        return next((c for c in self.children if c.key == key), None)


_FORMATS: dict[str, str] = {
    "b": "b", "B": "B", "s": "h", "S": "H", "l": "i", "L": "I",
    "j": "q", "J": "Q", "f": "f", "d": "d", "q": "i", "Q": "q",
}  # fmt: skip


def parse_klv(buf: bytes) -> list[Klv]:
    items: list[Klv] = []
    pos = 0
    while pos + 8 <= len(buf):
        key = buf[pos : pos + 4].decode("latin-1")
        typ = chr(buf[pos + 4])
        size = buf[pos + 5]
        (repeat,) = struct.unpack(">H", buf[pos + 6 : pos + 8])
        length = size * repeat
        data = buf[pos + 8 : pos + 8 + length]
        if len(data) < length:
            raise ValueError(f"GPMF {key}: 데이터가 잘렸습니다")
        children = tuple(parse_klv(data)) if typ == "\x00" else ()
        items.append(Klv(key, typ, size, repeat, data, children))
        pos += 8 + (length + 3) // 4 * 4
    return items


def klv_values(klv: Klv) -> NDArray[np.float64]:
    """숫자형 KLV를 (repeat, 원소 수) 배열로. 고정소수점(q, Q)은 실수로 바꾼다."""
    fmt = _FORMATS.get(klv.type)
    if fmt is None:
        raise ValueError(f"GPMF {klv.key}: 숫자형이 아닌 타입 {klv.type!r}")
    width = struct.calcsize(fmt)
    per_sample = klv.size // width
    count = per_sample * klv.repeat
    values = np.asarray(struct.unpack(f">{count}{fmt}", klv.data[: count * width]), dtype=float)
    if klv.type == "q":
        values /= 1 << 16
    elif klv.type == "Q":
        values /= 1 << 32
    return values.reshape(klv.repeat, per_sample)


def _strings(klv: Klv) -> str:
    return klv.data.split(b"\x00", 1)[0].decode("latin-1")


def sensor_samples(payload: bytes, fourcc: str) -> NDArray[np.float64]:
    """한 GPMF 페이로드에서 센서(ACCL, GYRO) 샘플을 SCAL·ORIN 적용해 (N, 3) XYZ로."""
    for devc in parse_klv(payload):
        for strm in (c for c in devc.children if c.key == "STRM"):
            data = strm.find(fourcc)
            if data is None:
                continue
            raw = klv_values(data)
            scal = strm.find("SCAL")
            if scal is not None:
                raw = raw / klv_values(scal).reshape(-1)
            orin_klv = strm.find("ORIN")
            orin = _strings(orin_klv) if orin_klv is not None else "ZXY"
            out = np.zeros_like(raw)
            for col, axis in enumerate(orin[: raw.shape[1]]):
                sign = -1.0 if axis.islower() else 1.0
                out[:, "XYZ".index(axis.upper())] = sign * raw[:, col]
            return out
    return np.zeros((0, 3))


def timestamps(packets: list[tuple[float, float, int]]) -> NDArray[np.float64]:
    """(패킷 시작 ms, 패킷 길이 ms, 샘플 수) 목록 → 샘플마다 시각. 패킷 안에서 균등 배치."""
    parts = [start + np.arange(n) * (dur / n) for start, dur, n in packets if n > 0]
    return np.asarray(np.concatenate(parts) if parts else np.zeros(0), dtype=np.float64)


def imu_from_gpmf_payloads(payloads: list[tuple[float, float, bytes]]) -> ImuData:
    acc_parts = [sensor_samples(p, "ACCL") for _, _, p in payloads]
    gyro_parts = [sensor_samples(p, "GYRO") for _, _, p in payloads]
    t_acc = timestamps(
        [(s, d, a.shape[0]) for (s, d, _), a in zip(payloads, acc_parts, strict=True)]
    )
    t_gyro = timestamps(
        [(s, d, g.shape[0]) for (s, d, _), g in zip(payloads, gyro_parts, strict=True)]
    )
    acc = np.concatenate(acc_parts) if acc_parts else np.zeros((0, 3))
    gyro_raw = np.concatenate(gyro_parts) if gyro_parts else np.zeros((0, 3))
    # 자이로를 가속도 시각으로 선형 보간한다 (기종에 따라 샘플레이트가 다르다)
    gyro = np.stack([np.interp(t_acc, t_gyro, gyro_raw[:, i]) for i in range(3)], axis=1)
    return ImuData(t_acc, acc, gyro, source="gpmf")


class GpmfExtractor:
    name = "gpmf"

    def can_handle(self, info: MediaInfo) -> bool:
        return any(d.codec_tag == "gpmd" for d in info.data_streams)

    def extract(self, path: Path) -> ImuData:
        payloads: list[tuple[float, float, bytes]] = []
        with av.open(str(path)) as c:
            stream = next(s for s in c.streams if s.type == "data" and s.codec_tag == "gpmd")
            tb = to_fraction(stream.time_base)
            for packet in c.demux(stream):
                if packet.pts is None:
                    continue
                payloads.append(
                    (
                        float(packet.pts * tb * 1000),
                        float((packet.duration or 0) * tb * 1000),
                        bytes(packet),
                    )
                )
        return imu_from_gpmf_payloads(payloads)


EXTRACTORS: list[ImuExtractor] = [GpmfExtractor()]


def extract_embedded_imu(path: Path, info: MediaInfo) -> ImuData | None:
    for extractor in EXTRACTORS:
        if extractor.can_handle(info):
            return extractor.extract(path)
    return None


# ---------------------------------------------------------------- 사이드카


def read_imu_table(path: Path) -> ImuData:
    """CSV 또는 Parquet (열: t_ms, ax, ay, az, gx, gy, gz)."""
    if path.suffix == ".parquet":
        cols, _ = read_parquet(path)
    elif path.suffix == ".csv":
        table = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
        names = table.dtype.names or ()
        cols = {name: np.asarray(table[name], dtype=float) for name in names}
    else:
        raise ValueError(f"지원하지 않는 IMU 파일 형식: {path.suffix}")
    missing = [c for c in ("t_ms", *IMU_COLUMNS) if c not in cols]
    if missing:
        raise ValueError(f"{path.name}: IMU 열이 없습니다: {missing}")
    return ImuData(
        np.asarray(cols["t_ms"], dtype=float),
        np.stack([np.asarray(cols[c], dtype=float) for c in IMU_COLUMNS[:3]], axis=1),
        np.stack([np.asarray(cols[c], dtype=float) for c in IMU_COLUMNS[3:]], axis=1),
        source=f"sidecar:{path.suffix.lstrip('.')}",
    )
