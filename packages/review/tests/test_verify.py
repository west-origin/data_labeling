"""검수 완료 판정(dlp review verify) 시험: 남은 일이 있으면 거부하고, 끝나면 전이·시각을 남긴다."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

import pytest
import sqlalchemy as sa

from dlp_review.verify import verify_session
from dlp_schema.db.migrate import upgrade
from dlp_schema.db.repository import (
    get_session,
    insert_labels,
    insert_review_task,
    insert_session,
    list_lifecycle_events,
    mark_review_task_collected,
    record_review,
    register_ontology,
    set_lifecycle,
    set_privacy_state,
)
from dlp_schema.labels import Provenance, Source, VerificationState
from dlp_schema.ontology import load_ontology
from dlp_schema.review import ReviewStage, ReviewTask, ReviewTool
from dlp_schema.session import LifecycleState, PrivacyState
from dlp_schema.testing import FIXED_TIME, action_payload, make_label, make_session

pytestmark = pytest.mark.services
ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def pg() -> Iterator[sa.Engine]:
    url = sa.make_url(
        os.environ.get(
            "DLP_DATABASE_URL", "postgresql+psycopg://dlp:dlp-dev-password@localhost:5432/dlp"
        )
    )
    name = f"dlp_test_{uuid.uuid4().hex[:8]}"
    admin = sa.create_engine(url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(sa.text(f'CREATE DATABASE "{name}"'))
    test_url = url.set(database=name).render_as_string(hide_password=False)
    upgrade(test_url)
    engine = sa.create_engine(test_url)
    with engine.begin() as conn:
        register_ontology(conn, load_ontology(ROOT / "config" / "ontology" / "v1"))
    try:
        yield engine
    finally:
        engine.dispose()
        with admin.connect() as conn:
            conn.execute(sa.text(f'DROP DATABASE "{name}" WITH (FORCE)'))
        admin.dispose()


def test_verify_refuses_until_review_is_done_then_records_transition(pg: sa.Engine) -> None:
    model = Provenance(source=Source.MODEL, model_version="actions-v1")
    later = FIXED_TIME + timedelta(days=3)
    with pg.begin() as conn:
        insert_session(conn, make_session("s001"))
        # 프리라벨 전 단계면 거부
        early = verify_session(conn, "s001", later, "lead")
        assert not early.verified and "프리라벨 전" in early.reasons[0]

        set_privacy_state(conn, "s001", PrivacyState.AUTO_BLURRED)
        set_lifecycle(conn, "s001", LifecycleState.PRIVACY_APPROVED)
        set_lifecycle(conn, "s001", LifecycleState.PRELABELED)
        insert_labels(
            conn, [make_label(action_payload(), label_id="a1", provenance=model, confidence=0.9)]
        )
        insert_review_task(
            conn,
            ReviewTask(
                task_key="label_studio:1", tool=ReviewTool.LABEL_STUDIO, external_id="1",
                session_id="s001", stream_id="bodycam", stage=ReviewStage.LABELING,
                assignee="rev01", media_uri="s3://dlp-labeling/sessions/s001/blurred/bodycam.mp4",
                label_kinds=("action",), created_at=FIXED_TIME,
            ),
        )  # fmt: skip
        blocked = verify_session(conn, "s001", later, "lead")
        text = " ".join(blocked.reasons)
        # 블러 승인 전, 수거 안 한 작업, 검수 안 한 모델 라벨이 모두 이유로 나온다
        assert not blocked.verified
        assert "블러 승인 전" in text and "label_studio:1" in text and "a1" in text
        assert get_session(conn, "s001").lifecycle_state is LifecycleState.PRELABELED

    with pg.begin() as conn:
        set_privacy_state(conn, "s001", PrivacyState.APPROVED)
        mark_review_task_collected(conn, "label_studio:1", FIXED_TIME)
        record_review(conn, "a1", VerificationState.HUMAN_APPROVED, "rev01", FIXED_TIME)
        done = verify_session(conn, "s001", later, "lead")
        assert done.verified and not done.already
        assert get_session(conn, "s001").lifecycle_state is LifecycleState.HUMAN_VERIFIED
        # 전이 시각·실행자가 생애주기 기록에 남는다 (운영 지표의 검증 주)
        last = list_lifecycle_events(conn, "s001")[-1]
        assert last.to_state is LifecycleState.HUMAN_VERIFIED
        assert last.at == later and last.actor == "lead"
        # 멱등: 다시 불러도 전이를 더 남기지 않는다
        count = len(list_lifecycle_events(conn, "s001"))
        again = verify_session(conn, "s001", later + timedelta(days=1), "lead")
        assert again.verified and again.already
        assert len(list_lifecycle_events(conn, "s001")) == count
