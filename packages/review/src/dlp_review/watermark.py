"""라벨러 ID 워터마크. 화면 캡처가 유출되면 누가 본 영상인지 알 수 있게 한다.

반투명 글씨를 화면 전체에 비스듬히 반복해 넣는다. 원본 PTS를 유지하고 오디오는 넣지 않는다.
"""

# PyAV 타입 스텁이 일부 반환 타입을 비워 두어 이 모듈에서만 해당 경고를 끈다.
# pyright: reportUnknownMemberType=false

from __future__ import annotations

from pathlib import Path

import av
import cv2
import numpy as np
from numpy.typing import NDArray

from dlp_media.probe import to_fraction


def watermark_layer(text: str, width: int, height: int) -> NDArray[np.uint8]:
    """흰 글씨 마스크 (0~255). 45도 대각선 방향으로 반복한다."""
    size = int(np.hypot(width, height)) + 1
    canvas = np.zeros((size, size), dtype=np.uint8)
    scale = max(0.4, width / 900)
    step_x, step_y = int(260 * scale * 2), int(70 * scale * 2)
    for row, y in enumerate(range(0, size, step_y)):
        for x in range(-(row % 2) * step_x // 2, size, step_x):
            cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, 255, 1, cv2.LINE_AA)
    rot = cv2.getRotationMatrix2D((size / 2, size / 2), 30, 1.0)
    rotated = cv2.warpAffine(canvas, rot, (size, size))
    oy, ox = (size - height) // 2, (size - width) // 2
    return np.ascontiguousarray(rotated[oy : oy + height, ox : ox + width], dtype=np.uint8)


def burn_watermark(
    src: Path, dst: Path, text: str, *, opacity: float, crf: int, encoder_rate: int
) -> None:
    """opacity·crf·encoder_rate는 config/policies/review.yaml media에서 온다 (PTS는 원본 그대로)."""
    with av.open(str(src)) as inp, av.open(str(dst), "w") as out:
        vin = inp.streams.video[0]
        tb = to_fraction(vin.time_base)
        w, h = vin.codec_context.width, vin.codec_context.height
        mask = watermark_layer(text, w, h).astype(np.float32)[:, :, None] / 255 * opacity
        vout = out.add_stream("libx264", rate=encoder_rate)
        assert isinstance(vout, av.VideoStream)
        vout.width, vout.height, vout.pix_fmt = w, h, "yuv420p"
        vout.time_base = tb
        vout.codec_context.time_base = tb
        vout.options = {"crf": str(crf), "preset": "veryfast", "threads": "1"}
        for frame in inp.decode(vin):
            if frame.pts is None:
                continue
            img = np.asarray(frame.to_ndarray(format="rgb24"), dtype=np.float32)
            marked = (img * (1 - mask) + 255 * mask).astype(np.uint8)
            new = av.VideoFrame.from_ndarray(marked, format="rgb24")
            new.pts, new.time_base = frame.pts, tb
            out.mux(vout.encode(new))
        out.mux(vout.encode(None))
