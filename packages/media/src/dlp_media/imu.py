"""IMU 추출. 카메라 기종별 추출기를 등록해 쓴다.

- GPMF (GoPro): MP4의 `gpmd` 데이터 트랙에 든 KLV 구조를 해석한다. 패킷 하나(보통 1초)에 든
  샘플을 패킷 구간에 균등하게 배치한다. 축 순서는 ORIN 태그를 따르고, 없으면 HERO5 이후 기본값
  ZXY로 본다. 소문자 축은 부호가 반대다.
- 사이드카: 카메라 앱이 따로 내보낸 CSV·Parquet (열: t_ms, ax, ay, az, gx, gy, gz).

모든 시각은 해당 카메라 스트림 시계의 ms다. 바디캠 내장 IMU면 바디캠과 같은 시계다.

WP3, ADR 0003. 수집(`ingest`)이 쓴다: 바디캠에 내장 IMU가 있고 매니페스트에 IMU 스트림이 없으면
`extract_embedded_imu`로 꺼내 "imu" 스트림(SHARED_CLOCK)을 만들고, 매니페스트의 IMU 사이드카는
`read_imu_table`로 읽는다(UNSYNCED, 동기화가 오프셋을 정한다). 정규화본은 원본 버킷
`sessions/<세션>/derived/<스트림>.parquet`.

단위: 가속도 m/s², 자이로 rad/s (GPMF는 SCAL로 나눈 값, 사이드카는 파일 값 그대로).

새 기종 추가: `ImuExtractor`(name, can_handle, extract)를 구현해 `EXTRACTORS`에 넣는다.
"""

# pyright: reportUnknownMemberType=false

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import av
import numpy as np
from numpy.typing import NDArray

from dlp_media.probe import MediaInfo, to_fraction
from dlp_media.tables import read_parquet, write_parquet

log = logging.getLogger(__name__)

# 정규화 Parquet·사이드카의 센서 열 이름 (가속도 3축, 자이로 3축 순서)
IMU_COLUMNS = ("ax", "ay", "az", "gx", "gy", "gz")


@dataclass(frozen=True)
class ImuData:
    """IMU 샘플 (시각 증가 순).

    Raises (생성 시):
        ValueError: 배열 길이가 맞지 않거나 시각이 순증가가 아닐 때.
    """

    # 샘플 시각 (그 카메라 시계 ms)
    t_ms: NDArray[np.float64]
    acc: NDArray[np.float64]  # (N, 3) m/s²
    gyro: NDArray[np.float64]  # (N, 3) rad/s
    # 출처: "gpmf" | "sidecar:csv" | "sidecar:parquet" 등
    source: str

    def __post_init__(self) -> None:
        """모양과 시각 순서를 검사한다 (빈 데이터는 허용)."""
        n = self.t_ms.size
        if self.acc.shape != (n, 3) or self.gyro.shape != (n, 3):
            raise ValueError("IMU 배열 길이가 맞지 않습니다")
        if n and np.any(np.diff(self.t_ms) <= 0):
            raise ValueError("IMU 시각이 증가하지 않습니다")

    @property
    def sample_rate_hz(self) -> float:
        """샘플레이트 (Hz) = 1000 / 샘플 간격 중앙값. 샘플이 1개 이하면 0."""
        return float(1000 / np.median(np.diff(self.t_ms))) if self.t_ms.size > 1 else 0.0

    def write(self, path: Path) -> None:
        """정규화 Parquet: 열 t_ms, ax..gz, 메타 source·sample_rate_hz."""
        cols = {"t_ms": self.t_ms}
        for i, name in enumerate(IMU_COLUMNS[:3]):
            cols[name] = self.acc[:, i]
        for i, name in enumerate(IMU_COLUMNS[3:]):
            cols[name] = self.gyro[:, i]
        write_parquet(
            path, cols, {"source": self.source, "sample_rate_hz": str(self.sample_rate_hz)}
        )


class ImuExtractor(Protocol):
    """카메라 기종별 내장 IMU 추출기."""

    # 추출기 이름 (기록·디버깅용)
    name: str

    def can_handle(self, info: MediaInfo) -> bool:
        """이 파일(probe 결과)을 처리할 수 있는가."""
        ...

    def extract(self, path: Path) -> ImuData | None:
        """IMU를 꺼낸다. 쓸 만한 샘플이 없으면 None (IMU 스트림을 만들지 않는다)."""
        ...


# ---------------------------------------------------------------- GPMF


@dataclass(frozen=True)
class Klv:
    """GPMF KLV 항목 하나. 중첩 항목(type 0)은 children에 풀어 둔다."""

    # 4글자 키 (DEVC, STRM, ACCL, GYRO, SCAL, ORIN …)
    key: str
    # 1글자 타입 (GPMF 형식 문자, "\\x00"이면 중첩)
    type: str
    # 원소 하나의 바이트 크기
    size: int
    # 반복 수 (샘플 수)
    repeat: int
    # 패딩을 뺀 원 데이터 (size x repeat 바이트)
    data: bytes
    children: tuple[Klv, ...] = ()

    def find(self, key: str) -> Klv | None:
        """직계 자식 중 그 키의 첫 항목 (없으면 None)."""
        return next((c for c in self.children if c.key == key), None)


# GPMF 타입 문자 → struct 형식 문자 (빅 엔디언으로 읽는다).
# q: Q15.16 고정소수점(32비트), Q: Q31.32 고정소수점(64비트) → 정수로 읽고 klv_values가 나눈다.
_FORMATS: dict[str, str] = {
    "b": "b", "B": "B", "s": "h", "S": "H", "l": "i", "L": "I",
    "j": "q", "J": "Q", "f": "f", "d": "d", "q": "i", "Q": "q",
}  # fmt: skip


def parse_klv(buf: bytes) -> list[Klv]:
    """GPMF 바이트열을 KLV 목록으로 해석한다 (중첩은 재귀).

    항목 형식: 키 4바이트 + 타입 1바이트 + 크기 1바이트 + 반복 2바이트(빅 엔디언) + 데이터
    (크기 x 반복 바이트, 4바이트 경계로 패딩). 끝의 8바이트 미만 조각은 무시한다.

    Raises:
        ValueError: 데이터가 헤더가 말한 길이보다 짧을 때 (잘린 페이로드).
    """
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
        pos += 8 + (length + 3) // 4 * 4  # 4바이트 경계로 올림
    return items


def klv_values(klv: Klv) -> NDArray[np.float64]:
    """숫자형 KLV를 (repeat, 원소 수) 배열로. 고정소수점(q, Q)은 실수로 바꾼다.

    원소 수 = size // 타입 바이트 수 (예: ACCL "s" size 6 → 3축).

    Raises:
        ValueError: 숫자형이 아닌 타입 (문자열 "c" 등).
    """
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
    """문자열 KLV의 첫 NUL 앞까지 (latin-1)."""
    return klv.data.split(b"\x00", 1)[0].decode("latin-1")


def sensor_samples(payload: bytes, fourcc: str) -> NDArray[np.float64]:
    """한 GPMF 페이로드에서 센서(ACCL, GYRO) 샘플을 SCAL·ORIN 적용해 (N, 3) XYZ로.

    DEVC → STRM 중 fourcc가 든 첫 스트림만 쓴다.
    - SCAL: 원 정수값을 나눌 배율 (축마다 하나이거나 공통 하나).
    - ORIN: 파일 열 순서의 축 이름 (예: "ZXY"면 0열이 Z). 소문자 축은 부호를 뒤집는다.
    센서가 없으면 (0, 3) 배열.
    """
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
            # 파일 열 col의 축 이름 → 출력 열 (X=0, Y=1, Z=2)
            for col, axis in enumerate(orin[: raw.shape[1]]):
                sign = -1.0 if axis.islower() else 1.0
                out[:, "XYZ".index(axis.upper())] = sign * raw[:, col]
            return out
    return np.zeros((0, 3))


def timestamps(packets: list[tuple[float, float, int]]) -> NDArray[np.float64]:
    """(패킷 시작 ms, 패킷 길이 ms, 샘플 수) 목록 → 샘플마다 시각. 패킷 안에서 균등 배치.

    샘플 i의 시각 = 시작 + i x (길이 / 샘플 수). 샘플이 없는 패킷은 건너뛴다.
    """
    parts = [start + np.arange(n) * (dur / n) for start, dur, n in packets if n > 0]
    return np.asarray(np.concatenate(parts) if parts else np.zeros(0), dtype=np.float64)


def fill_durations(
    payloads: list[tuple[float, float, bytes]],
) -> list[tuple[float, float, bytes]] | None:
    """길이(duration)가 없는(0 이하) 패킷의 길이를 채운다.

    회귀: 컨테이너에 packet.duration이 없으면 길이 0으로 들어와 패킷 안 샘플이 모두 시작 시각에
    몰렸고, `ImuData`의 시각 순증가 검증이 실패해 수집 전체가 실패했다.

    채우는 규칙:
    - 마지막이 아닌 패킷: 다음 패킷 시작 - 이 패킷 시작 (GPMF 패킷은 빈틈없이 이어진다).
    - 마지막 패킷: 길이를 아는(원래 있거나 위에서 채운) 패킷 길이의 중앙값.

    Args:
        payloads: (패킷 시작 ms, 패킷 길이 ms, GPMF 바이트) 목록, 시각 순.

    Returns:
        길이를 채운 목록. 길이를 정할 수 없으면(패킷 하나뿐인데 길이가 없거나, 다음 패킷 시작이
        같거나 앞서 있음) None.
    """
    out: list[tuple[float, float, bytes]] = []
    for i, (start, dur, data) in enumerate(payloads):
        if dur <= 0 and i + 1 < len(payloads):
            dur = payloads[i + 1][0] - start
            if dur <= 0:
                return None  # 시작 시각이 같거나 거꾸로다: 패킷 안 샘플 시각을 정할 수 없다
        out.append((start, dur, data))
    if out and out[-1][1] <= 0:
        known = [d for _, d, _ in out[:-1] if d > 0]
        if not known:
            return None  # 패킷 하나뿐이고 길이가 없다
        start, _, data = out[-1]
        out[-1] = (start, float(np.median(known)), data)
    return out


def imu_from_gpmf_payloads(payloads: list[tuple[float, float, bytes]]) -> ImuData | None:
    """가속도 샘플이 2개 미만이면 None (샘플레이트를 정할 수 없어 IMU 스트림을 만들지 않는다).

    자이로가 없는 기종·파일이면 자이로 열은 NaN이다 (0으로 채우면 정지로 오해된다).
    길이가 없는 패킷은 `fill_durations`로 채우고, 채울 수 없으면 경고를 남기고 None이다
    (IMU 하나 때문에 세션 수집 전체를 실패시키지 않는다).

    Args:
        payloads: (패킷 시작 ms, 패킷 길이 ms, GPMF 바이트) 목록, 시각 순.

    Returns:
        가속도 시각 기준 `ImuData` (source "gpmf"). 자이로는 가속도 시각으로 선형 보간한다.
    """
    filled = fill_durations(payloads)
    if filled is None:
        log.warning("GPMF 패킷 길이를 정할 수 없어 내장 IMU 스트림을 만들지 않습니다")
        return None
    payloads = filled
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
    if t_acc.size < 2:
        return None
    if t_gyro.size == 0:
        gyro = np.full((t_acc.size, 3), np.nan)
    else:
        # 자이로를 가속도 시각으로 선형 보간한다 (기종에 따라 샘플레이트가 다르다)
        # np.interp는 범위 밖을 양 끝 값으로 채운다
        gyro = np.stack([np.interp(t_acc, t_gyro, gyro_raw[:, i]) for i in range(3)], axis=1)
    return ImuData(t_acc, acc, gyro, source="gpmf")


class GpmfExtractor:
    """GoPro GPMF(`gpmd` 데이터 트랙) 추출기."""

    name = "gpmf"

    def can_handle(self, info: MediaInfo) -> bool:
        """`gpmd` 코덱 태그의 데이터 트랙이 있는가."""
        return any(d.codec_tag == "gpmd" for d in info.data_streams)

    def extract(self, path: Path) -> ImuData | None:
        """`gpmd` 트랙 패킷을 모아 `imu_from_gpmf_payloads`로 넘긴다.

        패킷 시각은 PTS x time_base x 1000 (ms), 길이는 packet.duration. 길이가 없으면 0으로 넘기고
        `imu_from_gpmf_payloads`가 다음 패킷 시작(마지막은 중앙값)으로 채운다.
        """
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


# 등록된 추출기 (앞에서부터 can_handle이 참인 첫 추출기를 쓴다). 테스트는 monkeypatch로 바꾼다.
EXTRACTORS: list[ImuExtractor] = [GpmfExtractor()]


def extract_embedded_imu(path: Path, info: MediaInfo) -> ImuData | None:
    """내장 IMU를 추출한다. 처리할 추출기가 없거나 쓸 만한 샘플이 없으면 None."""
    for extractor in EXTRACTORS:
        if extractor.can_handle(info):
            return extractor.extract(path)
    return None


# ---------------------------------------------------------------- 사이드카


def read_imu_table(path: Path) -> ImuData:
    """CSV 또는 Parquet (열: t_ms, ax, ay, az, gx, gy, gz).

    CSV는 첫 줄이 열 이름이어야 한다. 값의 단위는 변환하지 않는다 (m/s², rad/s로 내보내야 한다).

    Raises:
        ValueError: 지원하지 않는 확장자, 필요한 열이 없음, 샘플 2개 미만, `ImuData` 검증 실패.
    """
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
    if np.asarray(cols["t_ms"]).size < 2:
        raise ValueError(f"{path.name}: IMU 샘플이 2개 미만이라 샘플레이트를 정할 수 없습니다")
    return ImuData(
        np.asarray(cols["t_ms"], dtype=float),
        np.stack([np.asarray(cols[c], dtype=float) for c in IMU_COLUMNS[:3]], axis=1),
        np.stack([np.asarray(cols[c], dtype=float) for c in IMU_COLUMNS[3:]], axis=1),
        source=f"sidecar:{path.suffix.lstrip('.')}",
    )
