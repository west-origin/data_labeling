from __future__ import annotations

import itertools
import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
import sqlalchemy as sa

from dlp_actions.clients import OracleVlm
from dlp_actions.policy import load_policy
from dlp_actions.runner import run_actions
from dlp_fixtures.actions import generate_action_scenario
from dlp_schema.db.migrate import upgrade
from dlp_schema.db.repository import get_labels, insert_labels, insert_session, register_ontology
from dlp_schema.episode import current_labels
from dlp_schema.labels import ActionPayload, GapPayload, LabelRecord, Provenance, Source
from dlp_schema.ontology import load_ontology
from dlp_schema.testing import FIXED_TIME, make_session

pytestmark = pytest.mark.services
ROOT = Path(__file__).resolve().parents[3]
SID = "act-0001"
SCENARIO = generate_action_scenario(1, session_id=SID)


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
        register_ontology(conn, load_ontology(ROOT / "config/ontology/v1"))
        insert_session(conn, make_session(SID, duration_ms=SCENARIO.duration_ms))
        # 입력: 손 키포인트와 손 상태만 (정답 행동·사이 구간은 넣지 않는다)
        insert_labels(
            conn, [x for x in SCENARIO.labels if x.kind in ("keypoint_track", "hand_state")]
        )
    try:
        yield engine
    finally:
        engine.dispose()
        with admin.connect() as conn:
            conn.execute(sa.text(f'DROP DATABASE "{name}" WITH (FORCE)'))
        admin.dispose()


def _timeline(conn: sa.Connection) -> list[LabelRecord]:
    labels = current_labels(get_labels(conn, SID, kinds=["action", "gap"]))
    return sorted(labels, key=lambda x: x.t_start_ms)


def test_run_is_idempotent_and_new_version_replaces_old(pg: sa.Engine) -> None:
    ontology = load_ontology(ROOT / "config/ontology/v1")
    policy = load_policy(ROOT)
    truth = [x for x in SCENARIO.labels if isinstance(x.payload, ActionPayload | GapPayload)]
    with pg.begin() as conn:
        first = run_actions(conn, SID, OracleVlm(truth), ontology, policy, FIXED_TIME)
        again = run_actions(conn, SID, OracleVlm(truth), ontology, policy, FIXED_TIME)
        timeline = _timeline(conn)
    counts = first.hands["right"]
    assert counts["actions"] == len(SCENARIO.actions) and counts["unknown_fallbacks"] == 0
    assert again.skipped == ["right"] and not again.hands
    assert timeline[0].t_start_ms == 0 and timeline[-1].t_end_ms == SCENARIO.duration_ms
    assert all(a.t_end_ms == b.t_start_ms for a, b in itertools.pairwise(timeline))
    assert all(x.provenance.source is Source.MODEL for x in timeline)

    # 정책이 바뀌면 이전 버전 레코드는 삭제 표시되고 새 버전으로 다시 만든다
    changed = policy.model_copy(
        update={"boundaries": policy.boundaries.model_copy(update={"merge_ms": 90})}
    )
    with pg.begin() as conn:
        second = run_actions(conn, SID, OracleVlm(truth), ontology, changed, FIXED_TIME)
        timeline2 = _timeline(conn)
        descriptions = current_labels(get_labels(conn, SID, kinds=["description"]))
    assert second.version != first.version
    assert second.retracted == len(timeline) + counts["actions"]  # 행동·사이 구간 + 설명
    assert {x.provenance.model_version for x in timeline2} == {second.version}
    assert {x.provenance.model_version for x in descriptions} == {second.version}


def test_rerun_after_reviewer_deleted_everything_is_skipped(pg: sa.Engine) -> None:
    """감사 회귀 (ADR 0015): 현재 라벨이 아니라 이력으로 멱등을 판단한다 (ID 충돌·되살림 없음)."""
    ontology = load_ontology(ROOT / "config/ontology/v1")
    policy = load_policy(ROOT)
    truth = [x for x in SCENARIO.labels if isinstance(x.payload, ActionPayload | GapPayload)]
    with pg.begin() as conn:
        run_actions(conn, SID, OracleVlm(truth), ontology, policy, FIXED_TIME)
        ours = current_labels(get_labels(conn, SID, kinds=["action", "gap", "description"]))
        insert_labels(
            conn,
            [
                x.model_copy(
                    update={
                        "label_id": f"{x.label_id}:rev",
                        "parent_label_id": x.label_id,
                        "retracted": True,
                        "provenance": Provenance(source=Source.HUMAN),
                        "confidence": None,
                    }
                )
                for x in ours
            ],
        )
        again = run_actions(conn, SID, OracleVlm(truth), ontology, policy, FIXED_TIME)
        assert current_labels(get_labels(conn, SID, kinds=["action", "gap"])) == []
    assert again.skipped == ["right"]
