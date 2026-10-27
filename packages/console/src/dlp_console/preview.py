"""결과 보기: 승인·렌더된 블러본 + 라벨 겹쳐 그리기 자료 (ADR 0061 3절).

- 영상은 라벨링 버킷의 블러본만 쓴다. 주소는 라벨러 자격 증명(라벨링 버킷 읽기 전용)으로 서명한다
  (`S3Store.labeler_from_env`). 블러 승인 전·렌더 전에는 주소를 내지 않는다. 원본 버킷은 읽지
  않는다.
- 겹쳐 그리는 라벨은 운영 라벨(`current_labels`)의 박스 트랙과 손·전신 키포인트 트랙이다. 블러
  트랙은 싣지 않는다 (블러는 이미 영상에 들어가 있다). 좌표는 그 스트림 영상 픽셀, 시각은 스트림
  PTS(ms)다. 화면은 재생 시각에서 가장 가까운 앞 키프레임을 그리는 근사 표시다.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import sqlalchemy as sa

from dlp_console.status import SessionState, blurred_key
from dlp_schema.db.repository import get_labels
from dlp_schema.episode import current_labels
from dlp_schema.labels import BoxTrackPayload, KeypointTrackPayload, LabelRecord
from dlp_schema.session import PrivacyState

# 키 → 서명 URL (라벨러 자격 증명)
Signer = Callable[[str], str]


class PreviewUnavailableError(RuntimeError):
    """블러본을 보일 수 없다 (승인 전·렌더 전·모르는 스트림)."""


def blurred_url(state: SessionState, stream_id: str, sign: Signer) -> str:
    """블러본 서명 URL. 예외: PreviewUnavailableError."""
    if stream_id not in state.video_streams:
        raise PreviewUnavailableError(f"영상 스트림이 아닙니다: {stream_id}")
    if state.session.privacy_state is not PrivacyState.APPROVED:
        raise PreviewUnavailableError("블러 승인 전에는 영상을 보일 수 없습니다")
    if not state.rendered.get(stream_id):
        raise PreviewUnavailableError("블러본을 아직 만들지 않았습니다 (블러본 만들기)")
    return sign(blurred_key(state.session.session_id, stream_id))


def _track(label: LabelRecord) -> dict[str, Any] | None:
    p = label.payload
    if isinstance(p, BoxTrackPayload):
        return {
            "type": "box",
            "label": p.class_id,
            "entity": p.entity_id,
            "frames": [[k.t_ms, k.x, k.y, k.w, k.h] for k in p.keyframes if not k.outside],
        }
    if isinstance(p, KeypointTrackPayload):
        return {
            "type": "points",
            "label": p.skeleton if p.hand is None else f"{p.hand.value}_{p.skeleton}",
            "entity": p.entity_id,
            "frames": [
                [f.t_ms, [[q.x, q.y] for q in f.points if q.visibility > 0]] for f in p.keyframes
            ],
        }
    return None


def overlay_tracks(
    conn: sa.Connection, session_id: str, stream_id: str, max_tracks: int
) -> dict[str, Any]:
    """그 스트림의 겹쳐 그릴 트랙 (신뢰도 높은 순 max_tracks개)과 남은 개수."""
    history = get_labels(conn, session_id, kinds=["box_track", "keypoint_track"])
    labels = [x for x in current_labels(history) if x.stream_id == stream_id]
    labels.sort(key=lambda x: -(x.confidence if x.confidence is not None else 1.0))
    tracks: list[dict[str, Any]] = []
    for label in labels[:max_tracks]:
        track = _track(label)
        if track is not None:
            track["state"] = label.verification.state.value
            track["source"] = label.provenance.source.value
            tracks.append(track)
    return {"tracks": tracks, "hidden": max(0, len(labels) - max_tracks)}
