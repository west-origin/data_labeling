"""세션 스트림의 URI에서 동기화 입력을 불러온다."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from dlp_schema.session import Session, StreamKind
from dlp_sync.pipeline import StreamMedia
from dlp_sync.signals import glove_series, imu_series, load_audio

VIDEO = {StreamKind.BODYCAM, StreamKind.THIRD_PERSON}
GLOVE = {StreamKind.GLOVE_LEFT, StreamKind.GLOVE_RIGHT}


def load_session_media(session: Session, fetch: Callable[[str], Path]) -> dict[str, StreamMedia]:
    """fetch: URI → 로컬 파일 경로 (저장소에서 내려받기 등)."""
    media: dict[str, StreamMedia] = {}
    for s in session.streams:
        if s.kind in VIDEO:
            path = fetch(s.uri)
            media[s.stream_id] = StreamMedia(video=path, audio=load_audio(path))
        elif s.kind in GLOVE:
            media[s.stream_id] = StreamMedia(series=glove_series(fetch(s.uri)))
        elif s.kind is StreamKind.IMU:
            media[s.stream_id] = StreamMedia(series=imu_series(fetch(s.uri)))
        elif s.kind is StreamKind.AUDIO:
            media[s.stream_id] = StreamMedia(audio=load_audio(fetch(s.uri)))
    return media
