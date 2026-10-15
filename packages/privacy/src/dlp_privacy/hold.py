"""사고 대응용 프라이버시 승인 보류·해제 (WP5, ADR 0039·0042, F-REV-18).

라벨러가 노출(블러되지 않은 얼굴 등)을 신고하면 블러 재검수를 수집하기 전까지는 블러 라벨이 그대로라
승인도 렌더도 그대로다. 그 사이 같은 렌더로 새 라벨링 작업을 만들 수 있었다. 보류는 그 틈을 닫는다:

- `hold_session` (`dlp privacy hold <세션> --reason`): 세션 잠금 → 보류 기록(추가만) → 승인 상태면
  AUTO_BLURRED로 → 모든 영상 스트림의 배포된 라벨링 작업에 해시와 무관하게 회수 요청(사유
  `hold:<행위자>:<사유>`, 블러 검수 작업 제외). 이 DB 단계는 저장소와 무관하게 커밋된다.
  렌더 기록 무효(`invalidate_hold_renders`)는 커밋 **뒤** 따로 하는 재시도 가능한 단계다
  (F-PMS-14: 무효화 쓰기가 실패해 보류·승인 해제·회수 요청까지 되돌아가지 않게). 무효화가 실패해도
  승인 상태가 이미 AUTO_BLURRED라 렌더 게이트는 거부한다. 승인 상태(APPROVED)를 요구하는
  모든 소비자(블러본 열기 `assert_render_current`, 렌더, 라벨링 작업 생성, 내보내기, 행동 VLM)가
  막힌다. 보류 중에는 `approve_session`이 거부한다 (블러 검수 작업 생성·수집은 막지 않는다:
  보류 중에 블러를 고친다). 외부 삭제는 커밋 뒤 회수 실행기 몫이다 (CLI가 돌린다).
- `release_hold` (`dlp privacy hold <세션> --release --reason`): 해제 기록만 남긴다. 승인을
  돌려놓지 않는다. 재승인 절차를 다시 거친다: (필요하면 블러 검수 → 수집) → `dlp privacy approve`
  (승인 조건 확인) → `dlp privacy render` → 다시 만들기.

기록은 `privacy_holds`(추가만, PostgreSQL 트리거)에 남고 세션의 마지막 기록이 hold면 보류 중이다.
라벨은 쓰지 않는다. 승인 상태 갱신은 기존 `set_privacy_state`, 회수 요청은
`request_media_revocation`, 렌더 무효는 `invalidate_render`를 그대로 쓴다.

공개 이름:
- `HoldResult`: 보류 결과 (이번 기록, 지금 유효한 보류, 이전 승인 상태, 회수 요청한 작업).
- `hold_session`, `release_hold`: DB 단계 (호출자 트랜잭션).
- `invalidate_hold_renders`: 커밋 뒤 렌더 기록 무효 (멱등, 실패하면 다시 실행).
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import sqlalchemy as sa

from dlp_media.storage import ObjectStore
from dlp_privacy.errors import PrivacyGateError
from dlp_privacy.generation import VIDEO_KINDS
from dlp_privacy.policy import PrivacyPolicy
from dlp_privacy.rendering import invalidate_render
from dlp_privacy.revocation import request_media_revocation
from dlp_schema.db.repository import (
    active_privacy_hold,
    get_session,
    lock_session,
    record_privacy_hold,
    set_privacy_state,
)
from dlp_schema.review import ReviewTask
from dlp_schema.session import PrivacyHoldEvent, PrivacyState


@dataclass(frozen=True)
class HoldResult:
    """`hold_session` 결과."""

    # 이번에 남긴 보류 기록 (이미 보류 중이었으면 None, 멱등)
    event: PrivacyHoldEvent | None
    # 지금 유효한 보류 기록 (이번 것 또는 이전 것)
    held_since: PrivacyHoldEvent
    # 보류 전 프라이버시 상태 (APPROVED였으면 AUTO_BLURRED로 바꿨다)
    previous_state: PrivacyState
    # 이번에 회수를 요청한 작업 (요청 전 사본, 열린 작업은 revoked가 됐다)
    revocations: list[ReviewTask] = field(default_factory=list[ReviewTask])
    # 커밋 뒤 렌더 기록을 무효로 둘 영상 스트림 (`invalidate_hold_renders`에 넘긴다)
    video_streams: list[str] = field(default_factory=list[str])


def _need(reason: str, actor: str) -> None:
    """사유·행위자가 비었으면 ValueError (사고 대응 기록은 누가·왜가 필요하다)."""
    if not reason.strip() or not actor.strip():
        raise ValueError("승인 보류·해제에는 사유와 행위자가 필요합니다")


def hold_session(
    conn: sa.Connection,
    session_id: str,
    policy: PrivacyPolicy,
    now: datetime,
    *,
    reason: str,
    actor: str,
) -> HoldResult:
    """세션의 프라이버시 승인을 보류한다 (DB만). 이미 보류 중이면 기록은 더하지 않고 나머지를 다시
    한다.

    저장소는 건드리지 않는다: 렌더 기록 무효는 커밋 뒤 `invalidate_hold_renders`로 한다 (F-PMS-14).

    Args:
        conn: 호출자가 연 트랜잭션의 DB 연결 (커밋 뒤 렌더 무효·회수 실행기를 돌린다).
        policy: 프라이버시 정책 (회수 요청의 렌더 해시 계산).
        now: 기록·회수 요청 시각 (시간대 필수).
        reason, actor: 보류 사유와 행위자 (비면 ValueError).

    Returns:
        `HoldResult`.

    Raises:
        ValueError: 사유·행위자가 비었을 때.
        sqlalchemy.exc.NoResultFound: 세션이 없을 때.

    부작용: 세션 행 잠금, `privacy_holds` 추가, `sessions.privacy_state` 갱신(APPROVED였으면),
    `review_tasks` 회수 요청. 저장소 쓰기·외부 삭제는 하지 않는다.
    """
    _need(reason, actor)
    lock_session(conn, session_id)
    session = get_session(conn, session_id)
    current = active_privacy_hold(conn, session_id)
    event = None
    if current is None:
        event = record_privacy_hold(
            conn, session_id, "hold", reason=reason.strip(), actor=actor.strip(), at=now
        )
        current = event
    if session.privacy_state is PrivacyState.APPROVED:
        set_privacy_state(conn, session_id, PrivacyState.AUTO_BLURRED)
    # 블러가 맞는지 모르는 사고 대응이라 해시와 무관하게 모든 영상 스트림의 작업을 요청한다
    requested = request_media_revocation(
        conn, session_id, policy, now, reason=f"hold:{current.actor}:{current.reason}", force=True
    )
    videos = [s.stream_id for s in session.streams if s.kind in VIDEO_KINDS]
    return HoldResult(event, current, session.privacy_state, requested, videos)


def invalidate_hold_renders(
    labeling: ObjectStore, session_id: str, hold: PrivacyHoldEvent, streams: list[str]
) -> list[str]:
    """보류 커밋 뒤 영상 스트림의 렌더 기록을 무효로 둔다. 무효로 바꾼 스트림.

    멱등이다 (이미 무효인 기록을 다시 무효로 쓴다). 저장소 오류는 호출자에게 그대로 전달한다:
    보류·승인 해제·회수 요청은 이미 커밋돼 있고, 같은 `dlp privacy hold`를 다시 실행하면 여기만
    다시 한다. 그 사이에도 승인 상태가 APPROVED가 아니라 렌더 게이트가 거부한다.

    부작용: 라벨링 버킷 렌더 기록 덮어쓰기(무효).
    """
    invalidated: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        for stream_id in streams:
            if invalidate_render(
                labeling, session_id, stream_id, f"hold:{hold.event_id}", Path(tmp)
            ):
                invalidated.append(stream_id)
    return invalidated


def release_hold(
    conn: sa.Connection, session_id: str, now: datetime, *, reason: str, actor: str
) -> PrivacyHoldEvent:
    """보류를 푼다 (해제 기록만). 승인은 돌려놓지 않는다 — 재승인 절차를 다시 거친다.

    Raises:
        ValueError: 사유·행위자가 비었을 때.
        PrivacyGateError: 보류 중이 아닐 때.

    부작용: 세션 행 잠금, `privacy_holds` 추가.
    """
    _need(reason, actor)
    lock_session(conn, session_id)
    if active_privacy_hold(conn, session_id) is None:
        raise PrivacyGateError(f"{session_id}: 승인 보류 중이 아닙니다")
    return record_privacy_hold(
        conn, session_id, "release", reason=reason.strip(), actor=actor.strip(), at=now
    )
