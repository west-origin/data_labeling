"""행동 단계·설명 재생성·번역의 쓰기 직전 확인 (F-PAR-23·80, ADR 0043 3절).

세 단계는 이력 읽기·서버 호출을 세션 잠금 없이 한다 (긴 잠금을 피한다). 그 사이 세션이 사고 대응으로
승인 보류되거나(ADR 0042), 사용 중지되거나, 블러가 바뀌어 받은 블러본이 더는 지금 렌더가 아니면
결과를 쓰면 안 된다. 다른 단계의 최종 게시 확인(내보내기의 잠금 뒤 사용 중지·렌더 해시 재확인)과
같은 방식으로, 쓰기 직전 세션 잠금 아래에서 다시 본다. 막히면 호출자가 아무것도 쓰지 않는다.

공개 이름: `write_blocker`, `fetch_current_blurred` (받을 때의 렌더 해시를 함께 돌려준다).
"""

from __future__ import annotations

from pathlib import Path

import sqlalchemy as sa

from dlp_media.storage import ObjectStore, blurred_key
from dlp_privacy.policy import PrivacyPolicy
from dlp_privacy.runner import (
    RenderNotCurrentError,
    assert_render_current,
    check_fetched,
    expected_render_hash,
)
from dlp_schema.db.repository import active_privacy_hold, get_session, withdrawn_session_ids
from dlp_schema.session import LifecycleState


def write_blocker(
    conn: sa.Connection,
    session_id: str,
    render: tuple[str, str, PrivacyPolicy] | None = None,
) -> str | None:
    """쓰기를 막을 이유 (괜찮으면 None). 호출자가 세션을 잠근 뒤 부른다.

    Args:
        conn: 세션 잠금을 쥔 트랜잭션의 DB 연결.
        session_id: 세션 ID.
        render: 블러본 프레임을 보냈으면 (스트림 ID, 받을 때의 렌더 해시, 프라이버시 정책). 지금
            승인 상태이고 렌더 해시가 같아야 한다. None이면 렌더를 보지 않는다 (번역 등 픽셀 없음).

    Returns:
        사용 중지, 승인 보류, 렌더 변경 중 첫 이유를 사람이 읽을 문장으로. 없으면 None.
    """
    withdrawn = get_session(conn, session_id).lifecycle_state is LifecycleState.WITHDRAWN
    if withdrawn or session_id in withdrawn_session_ids(conn):
        return f"{session_id}: 사용 중지된 세션입니다"
    held = active_privacy_hold(conn, session_id)
    if held is not None:
        return f"{session_id}: 승인 보류 중입니다 ({held.actor}: {held.reason})"
    if render is None:
        return None
    stream_id, before, policy = render
    try:
        now = expected_render_hash(conn, session_id, stream_id, policy)
    except RenderNotCurrentError as exc:
        return str(exc)
    if now != before:
        return (
            f"{session_id}/{stream_id}: 묻는 동안 블러본 렌더가 바뀌었습니다 (dlp privacy render)"
        )
    return None


def fetch_current_blurred(
    conn: sa.Connection,
    labeling: ObjectStore,
    session_id: str,
    stream_id: str,
    work: Path,
    privacy: PrivacyPolicy,
) -> tuple[Path, str]:
    """지금 승인된 렌더의 블러본을 받는다 (ADR 0024, 아니면 RenderNotCurrentError).

    Returns:
        (받은 파일, 받을 때의 렌더 해시). 해시는 쓰기 직전 확인(`write_blocker`의 `render`)이 다시
        비교한다 (F-PAR-23·80).
    """
    video = work / f"{stream_id}.mp4"
    before = expected_render_hash(conn, session_id, stream_id, privacy)
    rendered = assert_render_current(conn, labeling, session_id, stream_id, privacy)
    labeling.get_file(blurred_key(session_id, stream_id), video)
    check_fetched(video, rendered, session_id, stream_id)
    return video, before
