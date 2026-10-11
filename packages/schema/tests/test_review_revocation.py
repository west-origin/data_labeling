"""검수 작업 매체 회수 계약·저장소·마이그레이션 0015 (ADR 0039). 서비스 없이 SQLite로 돈다."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa
from pydantic import ValidationError

from dlp_schema.db.migrate import downgrade, upgrade
from dlp_schema.db.repository import (
    get_review_task,
    insert_review_task,
    insert_session,
    mark_review_task_collected,
    mark_task_media_revoked,
    mark_task_tool_revoked,
    pending_revocations,
    request_task_revocation,
)
from dlp_schema.review import ReviewStage, ReviewTask, ReviewTaskStatus
from dlp_schema.testing import FIXED_TIME, make_session, sqlite_db

LATER = FIXED_TIME + timedelta(hours=1)
HASH = "ab" * 32


def _values(**overrides: Any) -> dict[str, Any]:
    """바디캠 라벨링 CVAT 작업의 필드 사전."""
    values: dict[str, Any] = {
        "task_key": "cvat:2",
        "tool": "cvat",
        "external_id": "2",
        "session_id": "s001",
        "stream_id": "bodycam",
        "stage": "labeling",
        "media_uri": "local://dlp-labeling/sessions/s001/review/a/bodycam.mp4",
        "label_kinds": ("box_track",),
        "created_at": FIXED_TIME,
        "render_hash": HASH,
    }
    values.update(overrides)
    return values


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"status": "revoked"}, "revoked 작업은"),
        ({"status": "revoked", "revoke_requested_at": LATER}, "revoked 작업은"),
        (
            {
                "status": "revoked",
                "revoke_requested_at": LATER,
                "revoke_reason": "x",
                "collected_at": LATER,
            },
            "revoked 작업은",
        ),
        ({"revoke_requested_at": LATER, "revoke_reason": "x"}, "열린 작업"),
        ({"status": "collected", "media_revoked_at": LATER}, "회수 완료 표시"),
        ({"status": "collected", "tool_revoked_at": LATER}, "회수 완료 표시"),
        ({"status": "collected", "revoke_requested_at": LATER}, "사유"),
        ({"stage": "privacy"}, "블러 검수 작업"),
        ({"render_hash": "ABC"}, "pattern"),
    ],
)
def test_review_task_revocation_rules(overrides: dict[str, Any], message: str) -> None:
    """회수 필드의 상태 규칙을 어기면 검증 오류다 (계약 검증기, ADR 0039)."""
    with pytest.raises(ValidationError, match=message):
        ReviewTask.model_validate(_values(**overrides))


def test_review_task_revocation_fields_roundtrip() -> None:
    """요청·완료 표시가 있는 회수 작업과 구 작업(None)이 JSON 왕복에서 같다."""
    revoked = ReviewTask.model_validate(
        _values(
            status="revoked", revoke_requested_at=LATER, revoke_reason="blur_changed:cvat:1",
            media_revoked_at=LATER, tool_revoked_at=LATER, revoke_note="ops: 도구 폐기",
        )
    )  # fmt: skip
    assert ReviewTask.model_validate_json(revoked.model_dump_json()) == revoked
    legacy = ReviewTask.model_validate(_values(render_hash=None))
    assert legacy.revoke_requested_at is None and legacy.render_hash is None
    collected = ReviewTask.model_validate(
        _values(
            status="collected", collected_at=LATER, revoke_requested_at=LATER,
            revoke_reason="sweep",
        )
    )  # fmt: skip
    assert collected.status is ReviewTaskStatus.COLLECTED


@pytest.fixture
def engine() -> Any:
    """세션 s001을 넣은 시간대 보존 SQLite."""
    eng = sqlite_db()
    with eng.begin() as conn:
        insert_session(conn, make_session(ontology_version=None))
    yield eng
    eng.dispose()


def _privacy_task() -> ReviewTask:
    """블러 검수 작업 (원본 프록시, 회수 대상이 아니다)."""
    return ReviewTask.model_validate(
        _values(
            task_key="cvat:1", external_id="1", stage="privacy", render_hash=None,
            media_uri="local://dlp-raw/sessions/s001/derived/bodycam.proxy.mp4",
            label_kinds=("blur_track",),
        )
    )  # fmt: skip


def test_request_revocation_marks_open_revoked_and_skips_privacy(engine: sa.Engine) -> None:
    """요청은 미요청 비프라이버시 작업만 바꾼다. 열린 작업은 revoked, 수집한 작업은 상태 유지.

    정답: 블러 검수 작업과 두 번째 요청은 False(멱등), 시간대 없는 시각은 ValueError,
    회수한 작업은 수집 표시할 수 없다(KeyError). 대기 목록은 완료 표시가 둘 다 찰 때까지 남는다.
    """
    open_task = ReviewTask.model_validate(_values())
    done = ReviewTask.model_validate(
        _values(task_key="label_studio:3", tool="label_studio", external_id="3")
    )
    with engine.begin() as conn:
        for task in (_privacy_task(), open_task, done):
            insert_review_task(conn, task)
        mark_review_task_collected(conn, "label_studio:3", LATER)
        with pytest.raises(ValueError, match="시간대"):
            request_task_revocation(conn, "cvat:2", reason="x", at=datetime(2026, 10, 7, 10))
        assert request_task_revocation(conn, "cvat:1", reason="x", at=LATER) is False
        assert request_task_revocation(conn, "cvat:2", reason="blur_changed:cvat:1", at=LATER)
        assert request_task_revocation(conn, "label_studio:3", reason="sweep", at=LATER)
        assert request_task_revocation(conn, "cvat:2", reason="again", at=LATER) is False
        task = get_review_task(conn, "cvat:2")
        assert (
            task.status is ReviewTaskStatus.REVOKED and task.revoke_reason == "blur_changed:cvat:1"
        )
        assert task.revoke_requested_at == LATER
        assert get_review_task(conn, "label_studio:3").status is ReviewTaskStatus.COLLECTED
        assert get_review_task(conn, "cvat:1").revoke_requested_at is None
        with pytest.raises(KeyError):
            mark_review_task_collected(conn, "cvat:2", LATER)
        assert [t.task_key for t in pending_revocations(conn, "s001")] == [
            "cvat:2",
            "label_studio:3",
        ]
        # 두 작업이 같은 매체 URI를 쓰므로 매체 표시는 함께 찬다 (회수 단위는 매체 객체)
        assert mark_task_media_revoked(conn, "s001", open_task.media_uri, LATER) == 2
        assert mark_task_tool_revoked(conn, "cvat:2", LATER)
        assert mark_task_tool_revoked(conn, "cvat:2", LATER) is False
        assert [t.task_key for t in pending_revocations(conn)] == ["label_studio:3"]
        assert mark_task_tool_revoked(conn, "label_studio:3", LATER, note="ops: 폐기")
        assert pending_revocations(conn) == []
        assert get_review_task(conn, "label_studio:3").revoke_note == "ops: 폐기"
        assert get_review_task(conn, "cvat:1").stage is ReviewStage.PRIVACY


def test_migration_0015_downgrade_refuses_revoked_rows(tmp_path: Path) -> None:
    """0015는 올리고 내릴 수 있지만, 회수한 작업이 있으면 내리기를 거부한다 (회수 의도 보존)."""
    url = f"sqlite:///{tmp_path / 'm.db'}"
    upgrade(url)
    downgrade(url, "0014")
    upgrade(url)
    engine = sa.create_engine(url)
    with engine.begin() as conn:
        insert_session(conn, make_session(ontology_version=None))
        insert_review_task(conn, ReviewTask.model_validate(_values()))
        request_task_revocation(conn, "cvat:2", reason="sweep", at=LATER)
    engine.dispose()
    with pytest.raises(RuntimeError, match="0015"):
        downgrade(url, "0014")


@pytest.mark.services
def test_migration_0015_on_postgresql(pg_url: str) -> None:
    """PostgreSQL에서도 0015는 회수 행이 있으면 내리기를 거부하고, 없으면 열만 지운다."""
    engine = sa.create_engine(pg_url)
    with engine.begin() as conn:
        insert_session(conn, make_session(ontology_version=None))
        insert_review_task(conn, ReviewTask.model_validate(_values()))
        request_task_revocation(conn, "cvat:2", reason="sweep", at=LATER)
        assert get_review_task(conn, "cvat:2").revoke_requested_at == LATER
    with pytest.raises(RuntimeError, match="0015"):
        downgrade(pg_url, "0014")
    with engine.begin() as conn:
        conn.execute(sa.text("DELETE FROM review_tasks"))
    engine.dispose()
    downgrade(pg_url, "0014")
    upgrade(pg_url)
