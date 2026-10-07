from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
import sqlalchemy as sa

from dlp_fixtures.wiping import generate_wiping_scenario
from dlp_relations.policy import load_policy
from dlp_relations.runner import run_relations
from dlp_schema.db.migrate import upgrade
from dlp_schema.db.repository import get_labels, insert_labels, insert_session, register_ontology
from dlp_schema.episode import current_labels
from dlp_schema.labels import CoveragePayload, LabelRecord, Provenance, RelationPayload, Source
from dlp_schema.ontology import load_ontology
from dlp_schema.testing import FIXED_TIME, make_session

pytestmark = pytest.mark.services
ROOT = Path(__file__).resolve().parents[3]
SID = "wipe-0000"


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
        insert_session(conn, make_session(SID))
        insert_labels(conn, generate_wiping_scenario(0, session_id=SID).labels)
    try:
        yield engine
    finally:
        engine.dispose()
        with admin.connect() as conn:
            conn.execute(sa.text(f'DROP DATABASE "{name}" WITH (FORCE)'))
        admin.dispose()


def _current(conn: sa.Connection) -> list[LabelRecord]:
    return current_labels(get_labels(conn, SID, kinds=["relation", "coverage"]))


def test_rerun_is_idempotent_and_rule_changes_regenerate(pg: sa.Engine) -> None:
    ontology = load_ontology(ROOT / "config/ontology/v1")
    policy = load_policy(ROOT)
    wiping = generate_wiping_scenario(0, session_id=SID)
    with pg.begin() as conn:
        first = run_relations(conn, SID, ontology, policy, FIXED_TIME)
        again = run_relations(conn, SID, ontology, policy, FIXED_TIME)
        current = _current(conn)
    assert first.inserted == len(wiping.truth_relations) + 1  # 관계 + 커버리지 1
    assert (again.inserted, again.retracted, again.kept) == (0, 0, first.inserted)
    [coverage] = [x.payload for x in current if isinstance(x.payload, CoveragePayload)]
    assert coverage.ratio == pytest.approx(wiping.coverage, abs=0.03)
    assert all(x.provenance.model_version == first.version for x in current)

    # 규칙 변경: 손 파지 규칙을 빼면 파지 관계만 삭제 표시되고 나머지는 그대로다
    no_grasp = policy.model_copy(
        update={"rules": tuple(r for r in policy.rules if r.id != "hand_grasp")}
    )
    with pg.begin() as conn:
        changed = run_relations(conn, SID, ontology, no_grasp, FIXED_TIME)
        assert run_relations(conn, SID, ontology, no_grasp, FIXED_TIME).inserted == 0
        current = _current(conn)
    assert (changed.inserted, changed.retracted) == (0, 1)
    assert not any(
        isinstance(x.payload, RelationPayload) and x.payload.derived_by == "hand_grasp"
        for x in current
    )

    # 규칙을 되돌리면 파지 관계가 새 ID로 다시 생긴다
    with pg.begin() as conn:
        restored = run_relations(conn, SID, ontology, policy, FIXED_TIME)
        current = _current(conn)
    assert (restored.inserted, restored.retracted) == (1, 0)
    [grasp] = [
        x for x in current
        if isinstance(x.payload, RelationPayload) and x.payload.derived_by == "hand_grasp"
    ]  # fmt: skip
    assert grasp.label_id.endswith("-1")


def test_reviewer_deleted_relation_is_not_reinserted(pg: sa.Engine) -> None:
    ontology = load_ontology(ROOT / "config/ontology/v1")
    policy = load_policy(ROOT)
    with pg.begin() as conn:
        run_relations(conn, SID, ontology, policy, FIXED_TIME)
        [contact, *_] = [
            x
            for x in _current(conn)
            if isinstance(x.payload, RelationPayload)
            and x.payload.derived_by == "tool_surface_contact"
        ]
        deletion = contact.model_copy(
            update={
                "label_id": f"{contact.label_id}:rev",
                "parent_label_id": contact.label_id,
                "retracted": True,
                "provenance": Provenance(source=Source.HUMAN),
                "confidence": None,
            }
        )
        insert_labels(conn, [deletion])
        summary = run_relations(conn, SID, ontology, policy, FIXED_TIME)
        ids = {x.label_id for x in _current(conn)}
    assert (summary.inserted, summary.skipped_by_review) == (0, 1)
    assert contact.label_id not in ids
