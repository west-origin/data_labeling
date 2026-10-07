"""블러본에서 시각에 맞는 프레임을 고른다 (PTS 인덱스로만)."""

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


def exact_frame(
    index: PtsIndex, stream: Stream, master_ms: float, tolerance_ms: float
) -> int | None:
    """키프레임 시각에 정확히 있는 프레임 (허용 오차 밖이면 None)."""
    t = stream_ms(stream, master_ms)
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
