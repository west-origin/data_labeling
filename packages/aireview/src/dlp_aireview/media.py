"""블러본 경계: AI 판정이 픽셀을 얻는 유일한 길 (WP17.5, ADR 0040 2.4·I1).

이 패키지에서 저장소 파일 받기(`get_file`), 영상 디코드(`decode_frames`), 판정 프레임 객체
(`BlurredFrame`) 만들기는 이 모듈에서만 한다 (정적 검사 `tests/test_aireview_boundaries.py`).
판정 입력 픽셀은 라벨링 버킷의 **지금 승인된 블러본**에서만 나온다.

`open_blurred(engine, labeling, 세션, 스트림, 프라이버시 정책, 작업 디렉터리,
raw_bucket=…)`의 확인 순서
1. `labeling.bucket`이 원본 버킷 이름이면 `MediaBoundaryError`. 감사 원본 저장소
   (`AuditedStore`)도 `bucket`이 안쪽 저장소의 버킷이라 여기서 걸린다 (클래스를 보지 않고 버킷
   이름만 비교한다).
2. 짧은 읽기 트랜잭션: `expected_render_hash`(세션이 지금 APPROVED가 아니면
   `RenderNotCurrentError`)와 `assert_render_current`(렌더 기록 해시·블러본 파일 해시)로 렌더
   해시와 블러본 sha를 얻고 닫는다.
3. 트랜잭션 없이 블러본을 작업 디렉터리로 받고(`check_fetched`로 파일 해시 확인) PTS 인덱스를
   만든다.
4. 짧은 읽기 트랜잭션: 렌더 해시를 다시 계산해 2와 같은지 본다. 받는 사이 블러가 바뀌었으면
   `RenderNotCurrentError`.
5. `BlurredMedia`(경로, 렌더 해시, 블러본 sha, PTS 인덱스)를 돌려준다.

- `frames_for(media, idx)`: 고른 프레임을 한 번의 순차 디코드로 꺼내 `BlurredFrame`으로 낸다
  (원 해상도, 공간 표식·확대 조각이 영상 픽셀 좌표를 쓴다). 프레임 번호 순서로 하나씩 낸다
  (실행기가 다 쓴 프레임을 바로 버려 메모리를 묶어 둔다).
- `render_hashes_now(engine, 세션, 스트림들, 정책)`: 지금 DB 상태의 렌더 해시 (실행기 기록자의
  주기 확인, `open_blurred` 4단계). 승인 상태가 아니면 `RenderNotCurrentError`.

블러본과 프레임은 호출자가 준 실행 임시 디렉터리에만 둔다 (버킷·영구 캐시에 두지 않는다).
이미지는 `dlp_models.vlm`이 data URI로만 보낸다 (서명 URL을 쓰지 않는다).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import sqlalchemy as sa
from numpy.typing import NDArray

from dlp_aireview.plan import StreamMedia
from dlp_media.frames import decode_frames
from dlp_media.pts import PtsIndex, build_pts_index
from dlp_media.storage import ObjectStore, blurred_key
from dlp_privacy.errors import RenderNotCurrentError
from dlp_privacy.policy import PrivacyPolicy
from dlp_privacy.rendering import assert_render_current, check_fetched, expected_render_hash

# 원 해상도 디코드 (긴 변 상한을 두지 않는다. 줄이기는 overlay가 단위 묶음별 상한으로 한다)
_FULL_SIZE = 1 << 30


class MediaBoundaryError(RuntimeError):
    """블러본 경계 위반: 원본 버킷 이름의 저장소로 판정 프레임을 얻으려 했다 (설정 오류)."""


@dataclass(frozen=True)
class BlurredMedia:
    """지금 승인된 블러본 하나 (실행 임시 디렉터리에 받은 파일).

    Attributes:
        session_id, stream_id: 세션과 영상 스트림.
        path: 받은 블러본 (실행이 끝나면 호출자가 지운다).
        render_hash: 받을 때 확인한 렌더 해시 (단위 키·표시 프로필에 들어간다).
        blurred_sha256: 렌더 기록의 블러본 파일 해시 (이전 형식 기록이면 None, 감사용).
        pts: 블러본 PTS 인덱스 (프레임 시각은 이것으로만 계산한다).
    """

    session_id: str
    stream_id: str
    path: Path
    render_hash: str
    blurred_sha256: str | None
    pts: PtsIndex = field(repr=False)

    @property
    def stream_media(self) -> StreamMedia:
        """계획(`plan_session`)에 넘길 PTS 인덱스·렌더 해시."""
        return StreamMedia(self.pts, self.render_hash)


@dataclass(frozen=True)
class BlurredFrame:
    """승인된 블러본에서 꺼낸 프레임 하나 (이 모듈의 `frames_for`만 만든다).

    Attributes:
        stream_id: 스트림.
        index: 블러본 프레임 번호 (PTS 인덱스 위치).
        t_ms: 그 프레임의 PTS 정수 ms (`dlp_media.frames.frame_ms`).
        rgb: 원 해상도 RGB 배열 (h, w, 3).
    """

    stream_id: str
    index: int
    t_ms: int
    rgb: NDArray[np.uint8] = field(repr=False)


def render_hashes_now(
    engine: sa.Engine, session_id: str, stream_ids: Iterable[str], policy: PrivacyPolicy
) -> dict[str, str]:
    """지금 DB 상태로 본 스트림별 렌더 해시 (짧은 읽기 트랜잭션).

    Raises:
        RenderNotCurrentError: 세션이 지금 프라이버시 승인 상태가 아닐 때.
    """
    with engine.connect() as conn:
        return {s: expected_render_hash(conn, session_id, s, policy) for s in sorted(stream_ids)}


def open_blurred(
    engine: sa.Engine,
    labeling: ObjectStore,
    session_id: str,
    stream_id: str,
    privacy_policy: PrivacyPolicy,
    work: Path,
    *,
    raw_bucket: str,
) -> BlurredMedia:
    """지금 승인된 블러본을 작업 디렉터리로 받는다 (모듈 docstring의 1~5단계).

    Args:
        engine: DB 엔진 (짧은 읽기 트랜잭션 두 번).
        labeling: 라벨링 버킷 저장소 (블러본을 읽는다).
        session_id, stream_id: 세션과 영상 스트림.
        privacy_policy: 프라이버시 정책 (렌더 해시 계산).
        work: 실행 임시 디렉터리 (받은 블러본을 둔다).
        raw_bucket: 원본 버킷 이름 (이 이름의 저장소는 거부한다).

    Raises:
        MediaBoundaryError: `labeling`이 원본 버킷 저장소일 때.
        RenderNotCurrentError: 승인 전, 렌더 기록 없음·무효·불일치, 받은 파일 해시 불일치, 받는 사이
            렌더 변경.
    """
    if labeling.bucket == raw_bucket:
        raise MediaBoundaryError(
            f"{session_id}/{stream_id}: 판정 프레임은 라벨링 버킷의 블러본에서만 얻습니다 "
            "(원본 버킷 저장소를 받았습니다)"
        )
    with engine.connect() as conn:
        rhash = expected_render_hash(conn, session_id, stream_id, privacy_policy)
        sha = assert_render_current(conn, labeling, session_id, stream_id, privacy_policy)
    path = work / f"{session_id}-{stream_id}-blurred.mp4"
    labeling.get_file(blurred_key(session_id, stream_id), path)
    check_fetched(path, sha, session_id, stream_id)
    pts = build_pts_index(path)
    now = render_hashes_now(engine, session_id, [stream_id], privacy_policy)[stream_id]
    if now != rhash:
        raise RenderNotCurrentError(
            f"{session_id}/{stream_id}: 블러본을 받는 사이 블러가 바뀌었습니다. 다시 실행하세요"
        )
    return BlurredMedia(session_id, stream_id, path, rhash, sha, pts)


def frames_for(media: BlurredMedia, frame_idx: Sequence[int]) -> Iterator[BlurredFrame]:
    """블러본에서 고른 프레임을 프레임 번호 순으로 낸다 (한 번의 순차 디코드, 원 해상도).

    Args:
        media: `open_blurred` 결과.
        frame_idx: 프레임 번호 (중복·순서 무관).

    Raises:
        ValueError: 번호가 범위를 벗어나거나 영상이 PTS 인덱스와 맞지 않을 때.
    """
    for index, t_ms, rgb in decode_frames(media.path, media.pts, frame_idx, max_side=_FULL_SIZE):
        yield BlurredFrame(media.stream_id, index, t_ms, rgb)
