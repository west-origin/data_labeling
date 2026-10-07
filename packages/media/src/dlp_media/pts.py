"""PTS 인덱스: 프레임 번호 ↔ 표시 시각(ms).

바디캠은 가변 프레임레이트일 수 있어 프레임 번호로 시각을 계산하지 않는다. 패킷의 PTS를
스트림 time_base 그대로(정수) 저장하고, ms는 필요할 때 정확한 분수 연산으로 구한다.
마스터 타임라인의 ms는 바디캠 PTS 시각(초)에 1000을 곱한 값이다.
"""

# pyright: reportUnknownMemberType=false

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from numpy.typing import NDArray

from dlp_media.probe import to_fraction
from dlp_media.tables import read_parquet, write_parquet


@dataclass(frozen=True)
class PtsIndex:
    time_base: Fraction
    pts: NDArray[np.int64]  # 표시 순서로 정렬
    keyframe: NDArray[np.bool_]

    def __post_init__(self) -> None:
        if self.pts.size == 0:
            raise ValueError("프레임이 없습니다")
        if np.any(np.diff(self.pts) <= 0):
            raise ValueError("PTS가 중복되거나 정렬되지 않았습니다")

    def __len__(self) -> int:
        return int(self.pts.size)

    @property
    def ms(self) -> NDArray[np.float64]:
        # 정수 곱을 먼저 하고 한 번만 나눠, ms가 정수인 프레임은 정확히 정수로 나오게 한다
        tb = self.time_base
        return (self.pts * (tb.numerator * 1000)).astype(np.float64) / tb.denominator

    def frame_ms(self, i: int) -> Fraction:
        """프레임 i의 정확한 시각(ms)."""
        return int(self.pts[i]) * self.time_base * 1000

    def frame_at(self, ms: float) -> int:
        """시각 ms에 화면에 떠 있는 프레임 (그 시각 이하의 마지막 프레임).

        첫 프레임 이전이면 0. PTS가 정수이므로 경계를 정확한 분수로 비교한다.
        """
        bound = math.floor(Fraction(ms) / 1000 / self.time_base)
        return max(int(np.searchsorted(self.pts, bound, side="right")) - 1, 0)

    def nearest(self, ms: float) -> int:
        i = int(np.searchsorted(self.ms, ms))
        if i == 0:
            return 0
        if i >= len(self):
            return len(self) - 1
        return i if self.ms[i] - ms < ms - self.ms[i - 1] else i - 1

    @property
    def duration_ms(self) -> float:
        """마지막 프레임 시각 + 마지막 프레임 간격."""
        ms = self.ms
        last_gap = float(np.median(np.diff(ms))) if ms.size > 1 else 0.0
        return float(ms[-1] + last_gap)

    @property
    def is_vfr(self) -> bool:
        if self.pts.size < 3:
            return False
        gaps = np.diff(self.pts)
        return bool(gaps.max() - gaps.min() > max(1, int(gaps.min()) // 100))

    def write(self, path: Path) -> None:
        write_parquet(
            path,
            {"pts": self.pts, "pts_ms": self.ms, "keyframe": self.keyframe},
            {"time_base": f"{self.time_base.numerator}/{self.time_base.denominator}"},
        )

    @classmethod
    def read(cls, path: Path) -> PtsIndex:
        cols, meta = read_parquet(path)
        return cls(
            Fraction(meta["time_base"]),
            cols["pts"].astype(np.int64),
            cols["keyframe"].astype(bool),
        )


def build_pts_index(path: Path, stream_index: int | None = None) -> PtsIndex:
    """디코딩 없이 패킷만 읽어 만든다."""
    with av.open(str(path)) as c:
        stream = c.streams.video[0] if stream_index is None else c.streams[stream_index]
        pts: list[int] = []
        key: list[bool] = []
        for packet in c.demux(stream):
            if packet.pts is None:
                continue  # 스트림 끝을 알리는 빈 패킷
            pts.append(packet.pts)
            key.append(packet.is_keyframe)
        time_base = to_fraction(stream.time_base)
    order = np.argsort(pts, kind="stable")
    return PtsIndex(
        time_base, np.asarray(pts, dtype=np.int64)[order], np.asarray(key, dtype=bool)[order]
    )
