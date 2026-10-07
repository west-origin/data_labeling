"""블러 렌더. 원본 PTS를 그대로 두고 오디오는 넣지 않는다.

블러는 강한 모자이크(박스 짧은 변을 blocks_per_box개 이하 블록으로 평균) 또는 단색 채움이다.
약한 가우시안 블러는 쓰지 않는다. 박스는 라벨 키프레임 시각(정수 ms)과 프레임 시각을 반올림해
맞춘다. 키프레임이 없는 프레임은 검수 화면(CVAT)이 보여 준 것과 같게 정한다.
- 직전 키프레임이 화면 밖(outside)이면 블러 없음.
- 직전·다음 키프레임이 모두 보이면 두 박스를 시각 비율로 선형 보간한다.
- 다음 키프레임이 화면 밖이거나 없으면 직전 박스를 그대로 유지한다.
검수자는 CVAT에서 키프레임 몇 개만 고치므로 (CVAT는 키프레임만 돌려준다) 보간하지 않으면
검수 화면과 블러본의 박스가 달라진다.
"""

# PyAV 타입 스텁이 일부 반환 타입을 비워 두어 이 모듈에서만 해당 경고를 끈다.
# pyright: reportUnknownMemberType=false

from __future__ import annotations

import bisect
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import av
import numpy as np

from dlp_media.probe import to_fraction
from dlp_privacy.detection import Image
from dlp_privacy.geometry import Box
from dlp_schema.labels import BlurTrackPayload, BoxKeyframe, LabelRecord


@dataclass(frozen=True)
class _Track:
    times: list[int]
    frames: list[BoxKeyframe]

    def box_at(self, t_ms: int) -> Box | None:
        i = bisect.bisect_right(self.times, t_ms) - 1
        if i < 0:
            return None
        k = self.frames[i]
        if k.outside:
            return None
        box = Box(k.x, k.y, k.w, k.h)
        if k.t_ms == t_ms or i + 1 >= len(self.frames):
            return box
        nxt = self.frames[i + 1]
        if nxt.outside:
            return box
        s = (t_ms - k.t_ms) / (nxt.t_ms - k.t_ms)
        return box.lerp(Box(nxt.x, nxt.y, nxt.w, nxt.h), s)


def blur_tracks(labels: list[LabelRecord]) -> list[_Track]:
    out: list[_Track] = []
    for label in labels:
        p = label.payload
        if isinstance(p, BlurTrackPayload):
            out.append(_Track([k.t_ms for k in p.keyframes], list(p.keyframes)))
    return out


def apply_blur(
    img: Image, box: Box, mode: Literal["mosaic", "solid"], min_block: int, blocks: int
) -> None:
    h, w = img.shape[:2]
    clipped = box.clipped(w, h)
    if clipped is None:
        return
    x1, y1 = int(np.floor(clipped.x)), int(np.floor(clipped.y))
    x2, y2 = int(np.ceil(clipped.x + clipped.w)), int(np.ceil(clipped.y + clipped.h))
    region = img[y1:y2, x1:x2]
    if mode == "solid":
        region[:] = 128
        return
    block = max(min_block, int(np.ceil(min(region.shape[0], region.shape[1]) / blocks)))
    for by in range(0, region.shape[0], block):
        for bx in range(0, region.shape[1], block):
            cell = region[by : by + block, bx : bx + block]
            cell[:] = cell.reshape(-1, 3).mean(axis=0).astype(np.uint8)


def render_blurred(
    src: Path,
    dst: Path,
    labels: list[LabelRecord],
    *,
    mode: Literal["mosaic", "solid"],
    min_block_px: int,
    blocks_per_box: int,
    encoder_rate: int,
    crf: int,
) -> int:
    """블러본을 쓰고 블러를 적용한 (프레임, 박스) 수를 돌려준다.

    encoder_rate·crf는 privacy.yaml render에서 온다.
    """
    tracks = blur_tracks(labels)
    applied = 0
    with av.open(str(src)) as inp, av.open(str(dst), "w") as out:
        vin = inp.streams.video[0]
        tb = to_fraction(vin.time_base)
        vout = out.add_stream("libx264", rate=encoder_rate)
        assert isinstance(vout, av.VideoStream)
        vout.width, vout.height = vin.codec_context.width, vin.codec_context.height
        vout.pix_fmt = "yuv420p"
        vout.time_base = tb
        vout.codec_context.time_base = tb
        vout.options = {"crf": str(crf), "preset": "veryfast", "threads": "1"}
        for frame in inp.decode(vin):
            if frame.pts is None:
                continue
            img = np.asarray(frame.to_ndarray(format="rgb24"), dtype=np.uint8)
            t_ms = round(float(frame.pts * tb * 1000))
            for track in tracks:
                box = track.box_at(t_ms)
                if box is not None:
                    apply_blur(img, box, mode, min_block_px, blocks_per_box)
                    applied += 1
            new = av.VideoFrame.from_ndarray(img, format="rgb24")
            new.pts, new.time_base = frame.pts, tb
            out.mux(vout.encode(new))
        out.mux(vout.encode(None))
    return applied
