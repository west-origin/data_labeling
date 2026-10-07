"""컨테이너 정보 조회 (PyAV, ffprobe와 같은 libav 사용)."""

# PyAV 타입 스텁이 일부 반환 타입을 비워 두어 이 모듈에서만 해당 경고를 끈다.
# pyright: reportUnknownMemberType=false

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from pathlib import Path

import av


@dataclass(frozen=True)
class VideoInfo:
    index: int
    codec: str
    width: int
    height: int
    time_base: Fraction
    frames: int


@dataclass(frozen=True)
class AudioInfo:
    index: int
    codec: str
    sample_rate: int
    channels: int


@dataclass(frozen=True)
class DataStreamInfo:
    index: int
    codec_tag: str


@dataclass(frozen=True)
class MediaInfo:
    path: Path
    format_name: str
    duration_ms: float | None
    creation_time: datetime | None
    video: VideoInfo | None
    audio: AudioInfo | None
    data_streams: tuple[DataStreamInfo, ...]


def probe(path: Path) -> MediaInfo:
    with av.open(str(path)) as c:
        video = audio = None
        data: list[DataStreamInfo] = []
        for s in c.streams:
            if isinstance(s, av.VideoStream) and video is None:
                video = VideoInfo(
                    s.index,
                    s.codec_context.name,
                    s.codec_context.width,
                    s.codec_context.height,
                    to_fraction(s.time_base),
                    s.frames,
                )
            elif isinstance(s, av.AudioStream) and audio is None:
                audio = AudioInfo(
                    s.index, s.codec_context.name, s.codec_context.sample_rate, s.channels
                )
            elif s.type == "data":
                data.append(DataStreamInfo(s.index, _tag(s.codec_tag)))
        created = c.metadata.get("creation_time")
        return MediaInfo(
            path=path,
            format_name=c.format.name,
            duration_ms=c.duration / 1000 if c.duration is not None else None,
            creation_time=datetime.fromisoformat(created) if created else None,
            video=video,
            audio=audio,
            data_streams=tuple(data),
        )


def to_fraction(value: object) -> Fraction:
    """PyAV의 AVRational·Fraction을 Fraction으로. 없으면 오류."""
    num, den = getattr(value, "numerator", None), getattr(value, "denominator", None)
    if not isinstance(num, int) or not isinstance(den, int) or den == 0:
        raise ValueError(f"time_base를 해석할 수 없습니다: {value!r}")
    return Fraction(num, den)


def _tag(tag: str | int | None) -> str:
    if isinstance(tag, int):
        return tag.to_bytes(4, "little").decode("latin-1")
    return tag or ""
