"""검수 화면용 프록시 영상. 원본 PTS를 그대로 유지해 마스터 타임라인과 어긋나지 않게 한다.

해상도를 낮추고 일정 간격(설정)마다 키프레임을 넣어 빠른 스크러빙이 가능하게 한다.
오디오는 넣지 않는다
(오디오는 원본 저장소에만 둔다).
"""

# pyright: reportUnknownMemberType=false

from __future__ import annotations

from pathlib import Path

import av
from av.video.frame import PictureType

from dlp_media.probe import to_fraction
from dlp_schema.config import ProxyConfig


def proxy_size(width: int, height: int, max_height: int) -> tuple[int, int]:
    """프록시 해상도. 높이는 max_height 이하로 줄이고(비율 유지), 가로·세로를 짝수로 맞춘다.

    libx264 yuv420p는 홀수 크기를 인코딩하지 못한다. 원본이 홀수면 1픽셀 줄인다 (최소 2).
    """
    if height > max_height:
        width, height = round(width * max_height / height), max_height
    return max(2, width // 2 * 2), max(2, height // 2 * 2)


def make_proxy(src: Path, dst: Path, settings: ProxyConfig) -> None:
    """settings: config/defaults.yaml media.proxy (높이 상한, CRF, 키프레임 간격)."""
    max_height, crf, keyframe_ms = settings.max_height, settings.crf, settings.keyframe_ms
    with av.open(str(src)) as inp, av.open(str(dst), "w") as out:
        vin = inp.streams.video[0]
        width, height = proxy_size(vin.codec_context.width, vin.codec_context.height, max_height)
        time_base = to_fraction(vin.time_base)

        vout = out.add_stream("libx264", rate=30)
        assert isinstance(vout, av.VideoStream)
        vout.width, vout.height, vout.pix_fmt = width, height, "yuv420p"
        vout.time_base = time_base
        vout.codec_context.time_base = time_base
        vout.options = {"crf": str(crf), "preset": "veryfast", "threads": "1"}

        next_key = None
        for frame in inp.decode(vin):
            if frame.pts is None:
                continue
            small = frame.reformat(width=width, height=height, format="yuv420p")
            small.pts, small.time_base = frame.pts, time_base
            t_ms = float(frame.pts * time_base * 1000)
            if next_key is None or t_ms >= next_key:
                small.pict_type = PictureType.I
                next_key = t_ms + keyframe_ms
            else:
                small.pict_type = PictureType.NONE
            out.mux(vout.encode(small))
        out.mux(vout.encode(None))
