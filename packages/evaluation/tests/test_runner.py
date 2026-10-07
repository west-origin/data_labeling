from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa

from dlp_eval.gate import decide
from dlp_eval.harness import evaluate
from dlp_eval.policy import load_policy
from dlp_eval.runner import load_golden, write_report
from dlp_fixtures.actions import generate_action_scenario
from dlp_schema.db.migrate import upgrade
from dlp_schema.db.repository import (
    insert_golden_set,
    insert_labels,
    insert_session,
    record_review,
    register_ontology,
)
from dlp_schema.labels import ActionPayload, LabelRecord, Provenance, Source, VerificationState
from dlp_schema.lineage import GoldenSet
from dlp_schema.ontology import load_ontology
from dlp_schema.session import Domain, StreamKind, SyncMethod
from dlp_schema.testing import FIXED_TIME, make_session

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
    try:
        yield engine
    finally:
        engine.dispose()
        with admin.connect() as conn:
            conn.execute(sa.text(f'DROP DATABASE "{name}" WITH (FORCE)'))
        admin.dispose()


def _model(labels: list[LabelRecord], version: str, suffix: str) -> list[LabelRecord]:
    return [
        x.model_copy(
            update={
                "label_id": f"{x.label_id}-{suffix}",
                "provenance": Provenance(source=Source.MODEL, model_version=version),
                "confidence": 0.8,
            }
        )
        for x in labels
    ]


def test_golden_evaluation_from_database(pg: sa.Engine, tmp_path: Path) -> None:
    glove_stream: dict[str, Any] = {
        "stream_id": "glove_right", "kind": StreamKind.GLOVE_RIGHT,
        "uri": "s3://dlp-raw/x/glove.parquet", "sync_method": SyncMethod.SHARED_CLOCK,
    }  # fmt: skip
    with pg.begin() as conn:
        register_ontology(conn, load_ontology(ROOT / "config/ontology/v1"))
        for i, sid in enumerate(("g-0001", "g-0002")):
            session = make_session(sid)
            if i == 0:
                session = session.model_copy(
                    update={
                        "streams": (
                            *session.streams,
                            type(session.streams[0]).model_validate(glove_stream),
                        )
                    }
                )
            insert_session(conn, session)
            truth = [
                x for x in generate_action_scenario(i, session_id=sid).labels if x.kind == "action"
            ]
            good = _model(truth, "actions-good", "good")
            bad = _model(truth[1:], "actions-bad", "bad")  # 첫 행동을 놓침
            # 첫 행동의 정답은 검수자가 승인한 모델 라벨이다 (사람 라벨과 같이 정답으로 본다)
            # 오류 삽입 사본은 모델 버전을 그대로 달고 있어도 예측이 아니다 (감사 회귀, ADR 0015)
            seeded = good[1].model_copy(
                update={
                    "label_id": f"seed-x-{sid}",
                    "seeded_error": True,
                    "t_end_ms": good[1].t_end_ms,
                }
            )
            insert_labels(conn, [*truth[1:], *good, *bad, seeded])
            record_review(
                conn, good[0].label_id, VerificationState.HUMAN_APPROVED, "rev01", FIXED_TIME
            )
        insert_golden_set(
            conn,
            GoldenSet(
                version="cleaning-g1",
                domain=Domain.CLEANING,
                session_ids=("g-0001", "g-0002"),
                created_at=FIXED_TIME,
            ),
        )

    policy = load_policy(ROOT)
    with pg.connect() as conn:
        good = load_golden(conn, "cleaning-g1", {"actions": "actions-good"})
        bad = load_golden(conn, "cleaning-g1", {"actions": "actions-bad"})
    assert [s.groups["glove"] for s in good["actions"]] == ["glove", "bare"]
    good_report = evaluate(
        good, policy, golden_version="cleaning-g1", model_versions={"actions": "actions-good"}
    )
    bad_report = evaluate(
        bad, policy, golden_version="cleaning-g1", model_versions={"actions": "actions-bad"}
    )
    assert good_report.overall["actions"].metrics["segment_f1_0.5"] == 1.0
    assert bad_report.overall["actions"].metrics["segment_f1_0.5"] < 1.0

    decision = decide(bad_report, good_report, policy)
    assert not decision.passed
    write_report(tmp_path / "eval.json", bad_report, decision)
    data = json.loads((tmp_path / "eval.json").read_text("utf-8"))
    assert data["gate"]["passed"] is False and "glove=glove" in data["subgroups"]
    md = (tmp_path / "eval.md").read_text("utf-8")
    assert "배포 게이트: 실패" in md and "segment_f1_0.5" in md
    assert all(isinstance(x.payload, ActionPayload) for s in good["actions"] for x in s.truth)
