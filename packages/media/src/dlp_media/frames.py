"""PTS 인덱스로 프레임을 고르고 한 번의 순차 디코드로 꺼낸다 (WP17.2, ADR 0040).

AI 1차 검수(`dlp_aireview`)가 블러본에서 판정용 프레임을 얻을 때 쓴다. 영상 시각은 PTS
인덱스로만 계산한다 (CLAUDE.md: 프레임 번호에 간격을 곱하지 않는다). VFR 영상에서도 "그 시각에
화면에 떠 있는 프레임"(`PtsIndex.frame_at`)을 고른다.

- `pick_frames`: 원하는 시각(ms) 목록 → 프레임 번호 (중복 제거, 정렬).
- `plan_seeks`: 정렬된 목표 프레임마다 seek이 필요한지 (필요하면 그 키프레임 번호).
- `fit_size`: 긴 변 상한·정렬 단위에 맞춘 출력 크기.
- `decode_frames`: 목표 프레임을 한 번 순차로 디코드해 (번호, 정수 ms, RGB 배열)로 낸다.
  다음 목표 직전의 키프레임이 현재 디코드 위치보다 뒤일 때만 그 키프레임으로 seek한다.

같은 입력이면 같은 픽셀이 나온다 (축소는 swscale AREA 보간, 스레드 1). 영상 파일은 읽기만 한다.
"""

# PyAV 타입 스텁이 일부 반환 타입을 비워 두어 이 모듈에서만 해당 경고를 끈다.
# pyright: reportUnknownMemberType=false

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

import av
import numpy as np
from numpy.typing import NDArray

from dlp_media.pts import PtsIndex


def pick_frames(pts: PtsIndex, wanted_ms: Iterable[float]) -> list[int]:
    """원하는 시각마다 그 시각에 화면에 떠 있는 프레임 번호 (`PtsIndex.frame_at`), 중복 제거·정렬.

    Args:
        pts: 영상의 PTS 인덱스.
        wanted_ms: 원하는 시각 ms (그 스트림 PTS 기준).
    """
    return sorted({pts.frame_at(float(ms)) for ms in wanted_ms})


def frame_ms(pts: PtsIndex, index: int) -> int:
    """프레임 번호의 시각 (PTS 인덱스, 정수 ms로 반올림). 검수 작업 `frame_times`와 같은 규칙."""
    return round(float(pts.ms[index]))


def _key_before(pts: PtsIndex, index: int) -> int:
    """`index` 이하에서 마지막 키프레임 번호 (없으면 0)."""
    keys = np.flatnonzero(pts.keyframe[: index + 1])
    return int(keys[-1]) if keys.size else 0


def plan_seeks(pts: PtsIndex, targets: Sequence[int]) -> list[int | None]:
    """정렬된 목표 프레임마다 seek할 키프레임 번호 (필요 없으면 None).

    디코드 위치(다음에 나올 프레임 번호)는 0에서 시작해 목표를 꺼낼 때마다 그 다음으로 옮긴다.
    목표 직전의 키프레임이 디코드 위치보다 뒤면 거기로 건너뛰는 것이 이득이라 seek하고, 아니면
    그대로 순차 디코드한다.

    Raises:
        ValueError: 목표가 정렬·중복 없음이 아니거나 범위를 벗어날 때.
    """
    if list(targets) != sorted(set(targets)) or (
        targets and (targets[0] < 0 or targets[-1] >= len(pts))
    ):
        raise ValueError("목표 프레임은 범위 안에서 중복 없이 오름차순이어야 합니다")
    plan: list[int | None] = []
    pos = 0
    for t in targets:
        key = _key_before(pts, t)
        plan.append(key if key > pos else None)
        pos = t + 1
    return plan


def fit_size(width: int, height: int, max_side: int, align: int = 1) -> tuple[int, int]:
    """긴 변이 `max_side` 이하가 되게 비율을 유지해 줄이고(키우지 않음), 각 변을 `align`의 배수로
    내림한다 (최소 `align`). VLM 서버가 다시 크기를 바꾸며 이미지 토큰 수가 달라지는 일을 막는다.

    Args:
        width, height: 원래 크기 픽셀.
        max_side: 긴 변 상한 픽셀.
        align: 정렬 단위 픽셀 (Qwen2.5-VL 28, Qwen3-VL 32, 1이면 정렬하지 않음).
    """
    scale = min(1.0, max_side / max(width, height))
    w, h = round(width * scale), round(height * scale)
    return max(align, w // align * align), max(align, h // align * align)


def decode_frames(
    video: Path, pts: PtsIndex, frame_idx: Iterable[int], *, max_side: int, align: int = 1
) -> Iterator[tuple[int, int, NDArray[np.uint8]]]:
    """목표 프레임을 PTS 순서로 한 번 순차 디코드해 (번호, 정수 ms, RGB 배열 (h, w, 3))로 낸다.

    `plan_seeks`가 정한 곳에서만 키프레임으로 seek한다. 디코드한 프레임은 PTS로 인덱스와 대조하므로
    프레임 번호에 간격을 곱해 시각을 계산하지 않는다.

    Args:
        video: 영상 파일 (블러본). `pts`는 이 파일로 만든 인덱스여야 한다.
        pts: PTS 인덱스.
        frame_idx: 목표 프레임 번호 (중복·순서는 상관없다).
        max_side, align: 출력 크기 (`fit_size`).

    Raises:
        ValueError: 목표가 범위를 벗어나거나, 영상에서 목표 PTS의 프레임을 찾지 못할 때
            (인덱스와 영상이 맞지 않음).
    """
    targets = sorted(set(frame_idx))
    seeks = plan_seeks(pts, targets)
    if not targets:
        return
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        size: tuple[int, int] | None = None
        frames: Iterator[av.VideoFrame] = container.decode(stream)
        for target, seek in zip(targets, seeks, strict=True):
            if seek is not None:
                container.seek(int(pts.pts[seek]), stream=stream, backward=True, any_frame=False)
                frames = container.decode(stream)
            want = int(pts.pts[target])
            frame = _advance(frames, want)
            if size is None:
                size = fit_size(frame.width, frame.height, max_side, align)
            img = frame.to_ndarray(
                width=size[0], height=size[1], format="rgb24", interpolation="AREA", threads=1
            )
            yield target, frame_ms(pts, target), np.ascontiguousarray(img, dtype=np.uint8)


def _advance(frames: Iterator[av.VideoFrame], want: int) -> av.VideoFrame:
    """디코드 반복자를 PTS `want`의 프레임까지 진행한다.

    Raises:
        ValueError: 그 PTS를 지나쳤거나 영상이 끝났을 때.
    """
    for frame in frames:
        if frame.pts is None or frame.pts < want:
            continue
        if frame.pts == want:
            return frame
        raise ValueError(f"PTS {want} 프레임이 영상에 없습니다 (다음 프레임 {frame.pts})")
    raise ValueError(f"PTS {want} 프레임 전에 영상이 끝났습니다")
