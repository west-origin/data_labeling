"""블러본에서 시각에 맞는 프레임을 고른다 (PTS 인덱스로만).

시각 규약 (ADR 0019): 공간 라벨(박스·마스크·키포인트·블러·3D 궤적)의 키프레임 시각은 그 스트림
영상의 PTS 시각(스트림 시각, ms)이고, 시간 구간 라벨(행동·손 상태·관계 등)은 마스터 시각이다.
기준 스트림(바디캠)은 두 시각이 같다.
마스터 시각에서 스트림 영상 프레임을 고를 때는 stream_ms로 바꾼다.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import av
import numpy as np
from numpy.typing import NDArray

from dlp_media.pts import PtsIndex
from dlp_schema.session import Stream


def stream_ms(stream: Stream, master_ms: float) -> float:
    return (master_ms - stream.offset_ms - stream.manual_adjustment_ms) / stream.clock_scale


def exact_frame(index: PtsIndex, stream_t_ms: float, tolerance_ms: float) -> int | None:
    """공간 라벨 키프레임(스트림 시각)에 정확히 있는 프레임 (허용 오차 밖이면 None)."""
    t = stream_t_ms
    i = index.nearest(t)
    return i if abs(float(index.ms[i]) - t) <= tolerance_ms else None


def decode_frames(video: Path, wanted: set[int]) -> Iterator[tuple[int, NDArray[np.uint8]]]:
    """표시 순서 i번째 프레임 중 wanted에 든 것을 RGB로 낸다 (순서대로 한 번 디코딩)."""
    if not wanted:
        return
    last = max(wanted)
    with av.open(str(video)) as c:
        for i, frame in enumerate(c.decode(c.streams.video[0])):
            if i in wanted:
                rgb: NDArray[np.uint8] = frame.to_ndarray(format="rgb24")  # pyright: ignore[reportAssignmentType]
                yield i, rgb
            if i >= last:
                return
