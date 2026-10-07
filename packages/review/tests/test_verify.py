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
    """일회용 PostgreSQL DB (마이그레이션 + 온톨로지 v1 등록). 테스트 뒤 지운다.

    `DLP_DATABASE_URL`(기본: 개발 compose의 DB)에 접속해 무작위 이름의 DB를 만든다.
    """
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
    """검수 완료 판정이 남은 일을 모두 이유로 보여 주고, 끝나면 전이·시각을 남기는지 본다.

    시나리오: 프리라벨 전이면 거부 → prelabeled + 블러 승인 전 + 열린 작업 + 미검수 모델 라벨이면
    세 이유가 모두 나오고 생애주기는 그대로 →
    블러 승인·작업 수거·라벨 승인 뒤 human_verified로 전이.
    정답 근거: 전이 시각은 넘긴 `later`, 실행자는 "lead"여야 하고, 다시 불러도 기록이 늘지
    않는다(멱등).
    """
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


def test_cli_verify_actor_defaults_to_current_actor(
    pg: sa.Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """회귀: `dlp review verify`가 --actor·DLP_ACTOR가 없으면 판정자를 "unknown"으로 남겼다.

    정답: 다른 명령처럼 `current_actor()`(DLP_ACTOR, 없으면 OS 사용자)를 쓴다. OS 사용자를
    "os-user"로 바꿔 두면 생애주기 기록의 실행자가 "os-user"다.
    """
    from dlp_cli.main import main

    with pg.begin() as conn:
        insert_session(conn, make_session("s002"))
        set_privacy_state(conn, "s002", PrivacyState.APPROVED)
        set_lifecycle(conn, "s002", LifecycleState.PRIVACY_APPROVED)
        set_lifecycle(conn, "s002", LifecycleState.PRELABELED)
        # 사람이 만든 행동 라벨 하나 (검수 완료 조건: 블러가 아닌 운영 라벨이 있다)
        insert_labels(conn, [make_label(action_payload(), label_id="b1", session_id="s002")])
    monkeypatch.delenv("DLP_ACTOR", raising=False)
    monkeypatch.setattr("getpass.getuser", lambda: "os-user")
    url = pg.url.render_as_string(hide_password=False)
    assert main(["review", "verify", "s002", "--url", url]) == 0
    with pg.connect() as conn:
        last = list_lifecycle_events(conn, "s002")[-1]
    assert last.to_state is LifecycleState.HUMAN_VERIFIED and last.actor == "os-user"
