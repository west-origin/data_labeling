"""사용 중지한 세션의 배포된 검수 작업에 매체 회수를 요청한다 (F-REV-17, ADR 0039).

서비스 없이 시간대를 보존하는 SQLite(`dlp_schema.testing.sqlite_db`)로 돈다.

정답 근거: 사용 중지 트랜잭션이 그 세션의 비프라이버시 검수 작업 중 회수 요청이 없는 것마다
`withdrawn:<행위자>` 사유로 요청을 남긴다 (열린 작업은 revoked, 수집한 작업은 상태를 두고 요청만).
블러 검수 작업(원본 권한자 화면)과 이미 요청된 작업·다른 세션의 작업은 그대로다. 다시 불러도 새로
요청하지 않는다. 외부 삭제는 커밋 뒤 회수 실행기 몫이다 (`dlp dataset withdraw`가 돌린다).
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import timedelta

import pytest
import sqlalchemy as sa

from dlp_datasets.lineage import session_lineage, withdraw_session
from dlp_schema import load_ontology, repo_root
from dlp_schema.db.repository import (
    get_review_task,
    insert_review_task,
    insert_session,
    mark_review_task_collected,
    register_ontology,
    request_task_revocation,
)
from dlp_schema.review import ReviewStage, ReviewTask, ReviewTaskStatus, ReviewTool
from dlp_schema.session import LifecycleState
from dlp_schema.testing import FIXED_TIME, make_session, sqlite_db

NOW = FIXED_TIME + timedelta(days=1)


@pytest.fixture
def engine() -> Iterator[sa.Engine]:
    """세션 s001·s002와 검수 작업 다섯을 넣은 메모리 SQLite."""
    eng = sqlite_db()
    with eng.begin() as conn:
        register_ontology(conn, load_ontology(repo_root() / "config/ontology/v1"))
        for sid in ("s001", "s002"):
            insert_session(conn, make_session(sid))
        for task in (
            _task("cvat:1", "s001"),
            _task("label_studio:2", "s001"),
            _task("cvat:3", "s001"),
            _task("cvat:4", "s001", stage=ReviewStage.PRIVACY),
            _task("cvat:5", "s001"),
            _task("cvat:6", "s002"),
        ):
            insert_review_task(conn, task)
        mark_review_task_collected(conn, "cvat:3", FIXED_TIME)
        request_task_revocation(conn, "cvat:5", reason="sweep", at=FIXED_TIME)
    yield eng
    eng.dispose()


def _task(key: str, sid: str, stage: ReviewStage = ReviewStage.LABELING) -> ReviewTask:
    """바디캠 검수 작업 (블러 검수면 원본 프록시, 아니면 라벨링 버킷 사본)."""
    privacy = stage is ReviewStage.PRIVACY
    uri = (
        f"s3://dlp-raw/sessions/{sid}/derived/bodycam.proxy.mp4"
        if privacy
        else f"s3://dlp-labeling/sessions/{sid}/review/labeler01/bodycam.mp4"
    )
    tool = ReviewTool.LABEL_STUDIO if key.startswith("label_studio:") else ReviewTool.CVAT
    return ReviewTask(
        task_key=key, tool=tool, external_id=key.split(":")[1], session_id=sid,
        stream_id="bodycam", stage=stage, media_uri=uri,
        label_kinds=("blur_track",) if privacy else ("box_track",), created_at=FIXED_TIME,
        assignee="rev01" if privacy else "labeler01",
    )  # fmt: skip


def test_withdraw_requests_revocation_of_deployed_review_tasks(engine: sa.Engine) -> None:
    """사용 중지는 같은 트랜잭션에서 그 세션의 배포된 라벨링 작업마다 회수를 요청한다."""
    with engine.begin() as conn:
        lineage = withdraw_session(conn, "s001", "동의 철회", NOW, actor="dpo01")
    assert lineage.lifecycle is LifecycleState.WITHDRAWN
    assert lineage.revocations == ["cvat:1", "label_studio:2", "cvat:3"]
    with engine.connect() as conn:
        tasks = {k: get_review_task(conn, k) for k in ("cvat:1", "label_studio:2", "cvat:3")}
        for task in tasks.values():
            assert task.revoke_requested_at == NOW and task.revoke_reason == "withdrawn:dpo01"
        assert tasks["cvat:1"].status is ReviewTaskStatus.REVOKED
        assert tasks["cvat:3"].status is ReviewTaskStatus.COLLECTED
        # 블러 검수 작업, 이미 요청된 작업, 다른 세션의 작업은 그대로다
        assert get_review_task(conn, "cvat:4").revoke_requested_at is None
        assert get_review_task(conn, "cvat:5").revoke_reason == "sweep"
        assert get_review_task(conn, "cvat:6").revoke_requested_at is None
        # 계보 조회는 회수 요청을 만들지 않는다 (사용 중지 결과에만 있다)
        assert session_lineage(conn, "s001").revocations == []
    with engine.begin() as conn:
        again = withdraw_session(conn, "s001", "다시", NOW + timedelta(hours=1), actor="dpo02")
    assert again.revocations == []
    with engine.connect() as conn:
        assert get_review_task(conn, "cvat:1").revoke_reason == "withdrawn:dpo01"
