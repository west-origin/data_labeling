"""블러본에서 시각에 맞는 프레임을 고른다 (PTS 인덱스로만).

시각 규약 (ADR 0019): 공간 라벨(박스·마스크·키포인트·블러·3D 궤적)의 키프레임 시각은 그 스트림
영상의 PTS 시각(스트림 시각, ms)이고, 시간 구간 라벨(행동·손 상태·관계 등)은 마스터 시각이다.
기준 스트림(바디캠)은 두 시각이 같다.
마스터 시각에서 스트림 영상 프레임을 고를 때는 stream_ms로 바꾼다.

프레임 번호는 디코딩할 때만 쓰고 어디에도 저장하지 않는다 (CLAUDE.md 규칙). 프레임 간격을 곱해
시각을 계산하지 않고, 늘 `PtsIndex`(실제 PTS 목록)로 찾는다. 가변 프레임률(VFR) 블러본에서도 맞는다.

공개 함수: `stream_ms`(마스터 → 스트림 시각), `exact_frame`(키프레임에 정확히 있는 프레임),
`decode_frames`(필요한 프레임만 RGB로 디코딩). COCO·LeRobot이 쓴다.
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
    """마스터 시각(ms) → 그 스트림의 영상 시각(ms).

    `Stream.to_master_ms`(= offset + 수동 조정 + 스트림 시각 * 클럭 배율)의 역함수다.
    """
    return (master_ms - stream.offset_ms - stream.manual_adjustment_ms) / stream.clock_scale


def exact_frame(index: PtsIndex, stream_t_ms: float, tolerance_ms: float) -> int | None:
    """공간 라벨 키프레임(스트림 시각)에 정확히 있는 프레임 (허용 오차 밖이면 None).

    Args:
        index: 블러본 PTS 인덱스.
        stream_t_ms: 키프레임 시각 (스트림 시각, ms).
        tolerance_ms: 허용 차이 (export.yaml `coco.frame_tolerance_ms`, 반올림 오차만).

    Returns:
        표시 순서 프레임 번호 (0부터) 또는 None (두 프레임 사이에 둔 키프레임).
    """
    t = stream_t_ms
    i = index.nearest(t)
    return i if abs(float(index.ms[i]) - t) <= tolerance_ms else None


def decode_frames(video: Path, wanted: set[int]) -> Iterator[tuple[int, NDArray[np.uint8]]]:
    """표시 순서 i번째 프레임 중 wanted에 든 것을 RGB로 낸다 (순서대로 한 번 디코딩).

    가장 큰 번호를 지나면 멈춘다. `wanted`가 비면 아무것도 내지 않는다.

    Yields:
        (프레임 번호, H*W*3 uint8 RGB 배열).
    """
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
