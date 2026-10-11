"""사고 대응용 프라이버시 승인 보류·해제 (F-REV-18, ADR 0042).

시간대 보존 SQLite와 LocalStore로 돈다 (서비스 없음).

정답 근거: 보류는 승인을 AUTO_BLURRED로 되돌리고(렌더 게이트·작업 생성·내보내기가 승인 상태를
요구한다), 렌더 기록을 무효로 두고, 해시와 무관하게 모든 영상 스트림의 배포된 라벨링 작업에 회수를
요청한다(블러 검수 작업 제외). 보류 동안 승인은 거부된다. 해제는 기록만 남기고 승인을 돌려놓지
않는다 — 재승인(`approve_session`의 조건 확인) → 렌더를 다시 거친다. 보류·해제 기록은 추가만 한다.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa

from dlp_media.storage import LocalStore, blurred_key, sha256_file
from dlp_privacy.hold import hold_session, release_hold
from dlp_privacy.policy import PrivacyPolicy
from dlp_privacy.runner import (
    PrivacyGateError,
    RenderMeta,
    RenderNotCurrentError,
    approve_session,
    assert_render_current,
    render_hash,
    render_session,
    write_render_meta,
)
from dlp_schema import load_ontology, repo_root
from dlp_schema.db.repository import (
    active_privacy_hold,
    get_review_task,
    get_session,
    insert_review_task,
    insert_session,
    list_privacy_holds,
    record_privacy_detection,
    register_ontology,
)
from dlp_schema.review import ReviewStage, ReviewTask, ReviewTaskStatus, ReviewTool
from dlp_schema.session import PrivacyState
from dlp_schema.testing import FIXED_TIME, make_session, sqlite_db

SID = "s001"
NOW = FIXED_TIME + timedelta(hours=1)
STREAMS = ("bodycam", "third")


@pytest.fixture
def engine() -> Iterator[sa.Engine]:
    """표·온톨로지 v1을 만든 메모리 SQLite."""
    eng = sqlite_db()
    with eng.begin() as conn:
        register_ontology(conn, load_ontology(repo_root() / "config/ontology/v1"))
    yield eng
    eng.dispose()


def _task(key: str, stream: str, stage: ReviewStage, **kw: Any) -> ReviewTask:
    """검수 작업 (블러 검수면 원본 프록시, 아니면 라벨링 버킷 사본)."""
    privacy = stage is ReviewStage.PRIVACY
    uri = (
        f"local://dlp-raw/sessions/{SID}/derived/{stream}.proxy.mp4"
        if privacy
        else f"local://dlp-labeling/sessions/{SID}/review/labeler01/{stream}.mp4"
    )
    return ReviewTask(
        task_key=key, tool=ReviewTool.CVAT, external_id=key[5:], session_id=SID,
        stream_id=stream, stage=stage, media_uri=uri,
        label_kinds=("blur_track",) if privacy else ("box_track",), created_at=FIXED_TIME,
        assignee="rev01" if privacy else "labeler01", **kw,
    )  # fmt: skip


@pytest.fixture
def approved(engine: sa.Engine, tmp_path: Path, policy: PrivacyPolicy) -> LocalStore:
    """승인 조건을 갖춘 2스트림 세션(블러 0개, 스트림마다 탐지 세대와 수집한 블러 검수)과 렌더 기록.

    라벨링 작업: 바디캠 CVAT(지금 렌더 해시), 3인칭 CVAT(지금 렌더 해시). 라벨링 저장소를 돌려준다.
    """
    raw = LocalStore(tmp_path / "store", "dlp-raw")
    labeling = LocalStore(tmp_path / "store", "dlp-labeling")
    video = tmp_path / "v.bin"
    video.write_bytes(b"synthetic")
    streams: list[dict[str, Any]] = []
    kinds = {"bodycam": ("bodycam", "reference"), "third": ("third_person", "manual")}
    for stream, (kind, sync) in kinds.items():
        key = f"sessions/{SID}/raw/{stream}.mp4"
        raw.put_file(key, video, sha256_file(video))
        streams.append(
            {"stream_id": stream, "kind": kind, "sync_method": sync, "uri": raw.uri(key)}
        )
    current = render_hash([], policy)
    with engine.begin() as conn:
        insert_session(
            conn, make_session(SID, privacy_state=PrivacyState.APPROVED, streams=streams)
        )
        for i, stream in enumerate(STREAMS):
            run = record_privacy_detection(conn, SID, stream, ["m1"], FIXED_TIME)
            insert_review_task(
                conn,
                _task(f"cvat:{10 + i}", stream, ReviewStage.PRIVACY, privacy_detection_id=run,
                      status=ReviewTaskStatus.COLLECTED, collected_at=FIXED_TIME),
            )  # fmt: skip
            insert_review_task(
                conn, _task(f"cvat:{20 + i}", stream, ReviewStage.LABELING, render_hash=current)
            )
    for stream in STREAMS:
        sha = sha256_file(video)
        labeling.put_file(blurred_key(SID, stream), video, sha)
        write_render_meta(labeling, SID, stream, RenderMeta(current, sha), tmp_path)
    return labeling


def test_hold_reverts_approval_invalidates_render_and_requests_revocation(
    engine: sa.Engine, approved: LocalStore, policy: PrivacyPolicy
) -> None:
    """보류: 승인 → AUTO_BLURRED, 렌더 기록 무효, 해시가 같아도 모든 라벨링 작업 회수 요청.

    블러 검수 작업은 회수하지 않는다. 보류 동안 승인·렌더·블러본 열기가 거부된다. 다시 보류해도
    기록을 더하지 않는다 (멱등).
    """
    labeling = approved
    with engine.connect() as conn:  # 보류 전에는 지금 블러본이다
        assert assert_render_current(conn, labeling, SID, "bodycam", policy) is not None
    with engine.begin() as conn:
        result = hold_session(
            conn, SID, policy, NOW, reason="labeler report", actor="ops01", labeling=labeling
        )
    assert result.event is not None and result.event.action == "hold"
    assert (result.event.actor, result.event.reason, result.event.at) == (
        "ops01",
        "labeler report",
        NOW,
    )
    assert result.previous_state is PrivacyState.APPROVED
    assert sorted(t.task_key for t in result.revocations) == ["cvat:20", "cvat:21"]
    assert sorted(result.invalidated) == sorted(STREAMS)
    with engine.connect() as conn:
        assert get_session(conn, SID).privacy_state is PrivacyState.AUTO_BLURRED
        for key in ("cvat:20", "cvat:21"):
            task = get_review_task(conn, key)
            assert task.status is ReviewTaskStatus.REVOKED
            assert task.revoke_reason == "hold:ops01:labeler report"
        assert get_review_task(conn, "cvat:10").revoke_requested_at is None
        with pytest.raises(RenderNotCurrentError):
            assert_render_current(conn, labeling, SID, "bodycam", policy)
    with engine.begin() as conn, pytest.raises(PrivacyGateError, match="보류"):
        approve_session(conn, SID)
    with engine.begin() as conn, pytest.raises(PrivacyGateError):
        render_session(conn, SID, LocalStore(labeling.root.parent, "dlp-raw"), labeling, policy)
    with engine.begin() as conn:
        again = hold_session(conn, SID, policy, NOW, reason="x", actor="ops02", labeling=labeling)
        assert again.event is None and again.held_since == result.event
        assert again.revocations == []
        assert [h.action for h in list_privacy_holds(conn, SID)] == ["hold"]


def test_release_records_only_and_needs_reapproval(
    engine: sa.Engine, approved: LocalStore, policy: PrivacyPolicy
) -> None:
    """해제는 기록만 남긴다. 승인은 그대로 풀려 있고 재승인 절차(조건 확인)를 거쳐야 한다.

    보류가 없으면 해제를 거부하고, 사유·행위자는 필수다.
    """
    with engine.begin() as conn, pytest.raises(PrivacyGateError, match="보류 중이 아닙니다"):
        release_hold(conn, SID, NOW, reason="x", actor="ops01")
    with engine.begin() as conn:
        hold_session(conn, SID, policy, NOW, reason="incident", actor="ops01", labeling=approved)
    with engine.begin() as conn, pytest.raises(ValueError, match="사유"):
        release_hold(conn, SID, NOW, reason=" ", actor="ops01")
    later = NOW + timedelta(hours=1)
    with engine.begin() as conn:
        event = release_hold(conn, SID, later, reason="blur re-reviewed", actor="ops02")
        assert event.action == "release" and event.actor == "ops02"
        assert active_privacy_hold(conn, SID) is None
        assert get_session(conn, SID).privacy_state is PrivacyState.AUTO_BLURRED
        assert [h.action for h in list_privacy_holds(conn, SID)] == ["hold", "release"]
    with engine.begin() as conn:
        assert approve_session(conn, SID).privacy_state is PrivacyState.APPROVED
