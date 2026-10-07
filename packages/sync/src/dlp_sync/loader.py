"""세션 스트림의 URI에서 동기화 입력(`StreamMedia`)을 불러온다 (WP4).

`dlp sync run`의 `runner.run_sync`가 원본 저장소에서 파일을 받아 오는 `fetch`와 함께 부른다.
스트림 종류별로 무엇을 읽는지:

- 영상(바디캠·3인칭): 영상 파일 경로(QR 슬레이트용) + 첫 오디오 트랙(두드림·오디오 상관용)
- 장갑(좌·우): 정규화 Parquet의 압력 채널 합 (`sync.yaml glove.pressure_prefixes`)
- IMU: 정규화 Parquet의 가속도 크기에서 중력을 뺀 값
- 외부 오디오: 오디오 파일

그 밖의 종류는 입력을 만들지 않는다 (`synchronize`는 빈 `StreamMedia`로 다룬다).
모든 시각은 그 스트림 자기 시계의 ms다.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from dlp_schema.session import Session, StreamKind
from dlp_sync.pipeline import StreamMedia
from dlp_sync.policy import SyncPolicy
from dlp_sync.signals import glove_pressure_prefixes, glove_series, imu_series, load_audio

# 영상 파일을 갖는 스트림 종류 (슬레이트 검출 + 내장 오디오)
VIDEO = {StreamKind.BODYCAM, StreamKind.THIRD_PERSON}
# 장갑 Parquet을 갖는 스트림 종류
GLOVE = {StreamKind.GLOVE_LEFT, StreamKind.GLOVE_RIGHT}


def load_session_media(
    session: Session, fetch: Callable[[str], Path], policy: SyncPolicy
) -> dict[str, StreamMedia]:
    """세션의 모든 스트림에서 동기화 입력을 만든다.

    Args:
        session: 동기화할 세션 (스트림 목록과 각 URI).
        fetch: URI → 로컬 파일 경로 (저장소에서 내려받기 등). 같은 URI를 여러 번 부를 수 있으므로
            호출자가 캐시한다. 원본 버킷이면 호출자가 감사 저장소(`raw_store`)로 받아야
            한다 (ADR 0020).
        policy: 동기화 정책. 장갑 압력 채널 접두사를 여기서 읽는다.

    Returns:
        `stream_id` → `StreamMedia`. 입력을 만들지 않는 종류의 스트림은 키가 없다.

    Raises:
        ValueError: 장갑 Parquet에 압력 채널이 없을 때 (`glove_series`).
    """
    media: dict[str, StreamMedia] = {}
    for s in session.streams:
        if s.kind in VIDEO:
            path = fetch(s.uri)
            # 영상 경로는 슬레이트 검출이, 오디오는 두드림·상관이 쓴다 (오디오 트랙이 없으면 None)
            media[s.stream_id] = StreamMedia(video=path, audio=load_audio(path))
        elif s.kind in GLOVE:
            media[s.stream_id] = StreamMedia(
                series=glove_series(fetch(s.uri), glove_pressure_prefixes(policy))
            )
        elif s.kind is StreamKind.IMU:
            media[s.stream_id] = StreamMedia(series=imu_series(fetch(s.uri)))
        elif s.kind is StreamKind.AUDIO:
            media[s.stream_id] = StreamMedia(audio=load_audio(fetch(s.uri)))
    return media
