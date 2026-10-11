"""컨테이너 정보 조회 (PyAV, ffprobe와 같은 libav 사용).

WP3. 수집(`ingest`)이 영상마다 불러 비디오 트랙 유무, 촬영 시각(creation_time), 데이터 트랙
(GoPro GPMF `gpmd` 등 IMU 추출 판단)을 본다. `to_fraction`은 PTS → ms 변환에 쓰는 time_base를
정확한 분수로 바꾸며 다른 패키지(프라이버시 탐지·렌더)도 쓴다.

- `probe`: 파일 → `MediaInfo`.
- `MediaInfo` / `VideoInfo` / `AudioInfo` / `DataStreamInfo`: 결과 데이터.
- `to_fraction`: PyAV time_base → `Fraction`.
"""

# PyAV 타입 스텁이 일부 반환 타입을 비워 두어 이 모듈에서만 해당 경고를 끈다.
# pyright: reportUnknownMemberType=false

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path

import av


@dataclass(frozen=True)
class VideoInfo:
    """첫 비디오 트랙 정보."""

    # 컨테이너 안 스트림 번호
    index: int
    # 코덱 이름 (예: "h264")
    codec: str
    # 인코딩된 픽셀 크기 (표시 회전 적용 전)
    width: int
    height: int
    # PTS 단위 (초). PTS x time_base = 초.
    time_base: Fraction
    # 컨테이너가 알려 준 프레임 수 (0이면 모름). 시각 계산에는 쓰지 않는다 (PTS 인덱스를 쓴다).
    frames: int
    # 표시 방향 (반시계 방향 각도). 인코딩된 픽셀 좌표와 플레이어 표시 좌표를 구별한다.
    rotation_degrees: int = 0


@dataclass(frozen=True)
class AudioInfo:
    """첫 오디오 트랙 정보."""

    index: int
    codec: str
    # 샘플레이트 (Hz)
    sample_rate: int
    channels: int


@dataclass(frozen=True)
class DataStreamInfo:
    """데이터 트랙 정보 (GoPro GPMF 등)."""

    index: int
    # 4글자 코덱 태그 (예: "gpmd")
    codec_tag: str


@dataclass(frozen=True)
class MediaInfo:
    """`probe` 결과."""

    path: Path
    # 컨테이너 형식 이름 (예: "mov,mp4,m4a,3gp,3g2,mj2")
    format_name: str
    # 컨테이너 길이 (ms, 실수). 모르면 None. 세션 길이는 이 값이 아니라 PTS 인덱스로 정한다.
    duration_ms: float | None
    # 컨테이너 creation_time 태그 (시간대 있음, 없으면 UTC로 본다). 태그가 없으면 None.
    creation_time: datetime | None
    video: VideoInfo | None
    audio: AudioInfo | None
    data_streams: tuple[DataStreamInfo, ...]


def probe(path: Path) -> MediaInfo:
    """미디어 파일의 컨테이너·트랙 정보를 읽는다.

    비디오·오디오는 첫 트랙만 본다. 데이터 트랙은 모두 모은다.
    표시 행렬을 확인하기 위해 첫 비디오 프레임 하나를 디코딩한다 (PyAV VideoFrame.rotation).

    Raises:
        av.error.*: 파일을 열 수 없을 때.
        ValueError: creation_time이 ISO 형식이 아니거나 time_base를 해석할 수 없을 때.
    """
    with av.open(str(path)) as c:
        video = audio = None
        data: list[DataStreamInfo] = []
        for s in c.streams:
            if isinstance(s, av.VideoStream) and video is None:
                first = next(c.decode(s), None)
                video = VideoInfo(
                    s.index,
                    s.codec_context.name,
                    s.codec_context.width,
                    s.codec_context.height,
                    to_fraction(s.time_base),
                    s.frames,
                    first.rotation if first is not None else 0,
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
            # c.duration은 AV_TIME_BASE(마이크로초) 단위다 → / 1000 = ms
            duration_ms=c.duration / 1000 if c.duration is not None else None,
            creation_time=_aware(datetime.fromisoformat(created)) if created else None,
            video=video,
            audio=audio,
            data_streams=tuple(data),
        )


def require_upright(info: MediaInfo) -> None:
    """회전 태그에 의존하는 영상을 명시적으로 거부한다. 현재 공간 라벨은 저장된 픽셀 기준이다.

    방향만 프록시에 반영하면 원본·검수 화면의 좌표가 달라진다. 입력의 픽셀 방향을 정규화한 별도
    사본이 필요하며 원본은 보존해야 한다. 이미 수집된 세션은 자동으로 고치지 않는다.
    """
    if info.video is not None and info.video.rotation_degrees % 360:
        raise ValueError(
            f"{info.path}: 표시 회전 {info.video.rotation_degrees}° 영상은 지원하지 않습니다. "
            "원본을 보존하고 픽셀 방향·PTS를 정규화한 사본을 준비하세요 "
            "(docs/developer-guide.md의 스마트폰 영상 입력 참조)."
        )


def _aware(t: datetime) -> datetime:
    """컨테이너 태그에 시간대가 없으면 UTC로 본다 (촬영 시각은 시간대가 있어야 한다)."""
    return t if t.tzinfo is not None else t.replace(tzinfo=UTC)


def to_fraction(value: object) -> Fraction:
    """PyAV의 AVRational·Fraction을 Fraction으로. 없으면 오류.

    Raises:
        ValueError: numerator·denominator가 정수가 아니거나 분모가 0일 때 (time_base 없음).
    """
    num, den = getattr(value, "numerator", None), getattr(value, "denominator", None)
    if not isinstance(num, int) or not isinstance(den, int) or den == 0:
        raise ValueError(f"time_base를 해석할 수 없습니다: {value!r}")
    return Fraction(num, den)


def _tag(tag: str | int | None) -> str:
    """코덱 태그를 4글자 문자열로 (정수 FourCC는 리틀 엔디언 바이트로 푼다, 없으면 "")."""
    if isinstance(tag, int):
        return tag.to_bytes(4, "little").decode("latin-1")
    return tag or ""
