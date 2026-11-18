"""바디캠 블러본 경계: 장면 안 위치 추정이 영상 프레임을 얻는 유일한 길 (ADR 0067).

위치 추정(`localize`)의 입력 프레임은 라벨링 버킷의 **지금 승인된 블러본** 바디캠 영상에서만
나온다. 원본 버킷을 열지 않는다. 영상 게이트(`dlp_privacy.rendering`, ADR 0024)를 그대로 쓴다.

`open_blurred_video`의 확인 순서 (AI 1차 검수의 `dlp_aireview.media.open_blurred`와 같다):
1. 받은 저장소가 원본 버킷 이름이면 거부한다 (설정 오류).
2. 짧은 읽기 트랜잭션: 세션이 지금 프라이버시 승인 상태이고 렌더 기록의 해시가 지금 운영 블러
   라벨·렌더 정책의 해시와 같은지 본다 (`assert_render_current`). 아니면 `RenderNotCurrentError`.
3. 블러본을 작업 디렉터리로 받아 파일 해시를 렌더 기록과 대조하고(`check_fetched`), 받은 파일로
   PTS 인덱스를 만든다. 블러본은 원본의 PTS를 그대로 옮긴 영상이라 원본 버킷의 PTS 인덱스를 읽을
   필요가 없다.
4. 짧은 읽기 트랜잭션: 렌더 해시를 다시 계산해 2와 같은지 본다 (받는 사이 블러가 바뀐 경우).

프레임 시각은 PTS 인덱스로만 정한다 (`sample_frames`: 앞서 고른 프레임보다 정책 간격 이상 뒤인
프레임, `read_frames`: 그 프레임의 PTS 정수 ms). 프레임 번호에 간격을 곱하지 않는다.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import sqlalchemy as sa

from dlp_media.frames import decode_frames
from dlp_media.pts import PtsIndex, build_pts_index
from dlp_media.storage import ObjectStore, blurred_key
from dlp_privacy.errors import RenderNotCurrentError
from dlp_privacy.policy import PrivacyPolicy
from dlp_privacy.rendering import assert_render_current, check_fetched, expected_render_hash
from dlp_scene3d.backends import FrameImage

# 원 해상도로 디코드한다 (긴 변 상한 없음): 특징 위치가 캘리브레이션과 같은 화소 좌표여야 한다
_FULL_SIZE = 1 << 30


class VideoBoundaryError(RuntimeError):
    """블러본 경계 위반: 원본 버킷 이름의 저장소로 프레임을 얻으려 했다 (설정 오류)."""


@dataclass(frozen=True)
class BlurredVideo:
    """지금 승인된 블러본 영상 하나 (작업 디렉터리에 받은 파일).

    render_hash는 받을 때 확인한 렌더 해시, blurred_sha256은 렌더 기록의 파일 해시(이전 형식
    기록이면 None), pts는 받은 파일의 PTS 인덱스다.
    """

    session_id: str
    stream_id: str
    path: Path
    render_hash: str
    blurred_sha256: str | None
    pts: PtsIndex = field(repr=False)


def check_blurred_video(
    engine: sa.Engine,
    labeling: ObjectStore,
    session_id: str,
    stream_id: str,
    privacy_policy: PrivacyPolicy,
    *,
    raw_bucket: str,
) -> tuple[str, str | None]:
    """블러본을 받지 않고 게이트만 본다: (지금 렌더 해시, 렌더 기록의 블러본 파일 해시).

    같은 입력의 위치 기록이 이미 있는지 영상을 받기 전에 알려고 쓴다.

    Raises:
        VideoBoundaryError: `labeling`이 원본 버킷 저장소일 때.
        RenderNotCurrentError: 승인 전, 렌더 기록 없음·무효·불일치.
    """
    if labeling.bucket == raw_bucket:
        raise VideoBoundaryError(
            f"{session_id}/{stream_id}: 프레임은 라벨링 버킷의 블러본에서만 얻습니다 "
            "(원본 버킷 저장소를 받았습니다)"
        )
    with engine.connect() as conn:
        render = expected_render_hash(conn, session_id, stream_id, privacy_policy)
        sha = assert_render_current(conn, labeling, session_id, stream_id, privacy_policy)
    return render, sha


def open_blurred_video(
    engine: sa.Engine,
    labeling: ObjectStore,
    session_id: str,
    stream_id: str,
    privacy_policy: PrivacyPolicy,
    work: Path,
    *,
    raw_bucket: str,
) -> BlurredVideo:
    """지금 승인된 블러본 영상을 작업 디렉터리로 받는다 (모듈 설명의 1~4단계).

    Args:
        engine: DB 엔진 (짧은 읽기 트랜잭션).
        labeling: 라벨링 버킷 저장소.
        session_id, stream_id: 세션과 영상 스트림.
        privacy_policy: 프라이버시 정책 (렌더 해시 계산).
        work: 작업 디렉터리 (받은 블러본을 둔다, 호출자가 지운다).
        raw_bucket: 원본 버킷 이름 (이 이름의 저장소는 거부한다).

    Raises:
        VideoBoundaryError: `labeling`이 원본 버킷 저장소일 때.
        RenderNotCurrentError: 승인 전, 렌더 기록 없음·무효·불일치, 받은 파일 해시 불일치, 받는 사이
            렌더 변경.
    """
    render, sha = check_blurred_video(
        engine, labeling, session_id, stream_id, privacy_policy, raw_bucket=raw_bucket
    )
    work.mkdir(parents=True, exist_ok=True)
    path = work / f"{stream_id}-blurred.mp4"
    labeling.get_file(blurred_key(session_id, stream_id), path)
    check_fetched(path, sha, session_id, stream_id)
    pts = build_pts_index(path)
    with engine.connect() as conn:
        now = expected_render_hash(conn, session_id, stream_id, privacy_policy)
    if now != render:
        raise RenderNotCurrentError(
            f"{session_id}/{stream_id}: 블러본을 받는 사이 블러가 바뀌었습니다. 다시 실행하세요"
        )
    return BlurredVideo(session_id, stream_id, path, render, sha, pts)


def sample_frames(pts: PtsIndex, interval_ms: int) -> list[int]:
    """풀 프레임 번호: 첫 프레임과, 앞서 고른 프레임보다 PTS가 interval_ms 이상 뒤인 프레임.

    시각은 PTS 인덱스의 값만 본다 (가변 프레임레이트에서도 간격이 정책 값 이상이다).
    """
    times = pts.ms
    picked = [0]
    for i in range(1, len(pts)):
        if times[i] - times[picked[-1]] >= interval_ms:
            picked.append(i)
    return picked


def read_frames(video: BlurredVideo, indices: Sequence[int]) -> Iterator[FrameImage]:
    """블러본에서 고른 프레임을 시각 순으로 낸다 (한 번의 순차 디코드, 원 해상도).

    Raises:
        ValueError: 번호가 범위를 벗어나거나 영상이 PTS 인덱스와 맞지 않을 때.
    """
    for _, t_ms, rgb in decode_frames(video.path, video.pts, indices, max_side=_FULL_SIZE):
        yield FrameImage(t_ms, rgb)
