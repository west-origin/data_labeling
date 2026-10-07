from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy.exc import DBAPIError

from dlp_schema.dataset import DatasetVersion, Split
from dlp_schema.db.migrate import downgrade, upgrade
from dlp_schema.db.repository import (
    TransitionError,
    get_dataset_version,
    get_labels,
    get_session,
    insert_dataset_version,
    insert_labels,
    insert_session,
    record_review,
    register_ontology,
    set_lifecycle,
    update_stream_sync,
)
from dlp_schema.db.tables import metadata
from dlp_schema.labels import Provenance, Source, VerificationState
from dlp_schema.ontology import Ontology
from dlp_schema.session import LifecycleState, SyncMethod
from dlp_schema.testing import FIXED_TIME, action_payload, make_label, make_session

DEFAULT_URL = "postgresql+psycopg://dlp:dlp-dev-password@localhost:5432/dlp"


def test_migrations_match_table_definitions(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'm.db'}"
    upgrade(url)
    engine = sa.create_engine(url)
    with engine.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn), metadata)
    engine.dispose()
    assert diff == [], diff
    downgrade(url, "base")


@pytest.fixture
def pg_url() -> Iterator[str]:
    """테스트마다 새 데이터베이스를 만들고 마이그레이션을 적용한다."""
    admin_url = sa.make_url(os.environ.get("DLP_DATABASE_URL", DEFAULT_URL))
    name = f"dlp_test_{uuid.uuid4().hex[:8]}"
    admin = sa.create_engine(admin_url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(sa.text(f'CREATE DATABASE "{name}"'))
    url = admin_url.set(database=name).render_as_string(hide_password=False)
    upgrade(url)
    try:
        yield url
    finally:
        with admin.connect() as conn:
            conn.execute(sa.text(f'DROP DATABASE "{name}" WITH (FORCE)'))
        admin.dispose()


@pytest.fixture
def pg(pg_url: str, ontology: Ontology) -> Iterator[sa.Engine]:
    engine = sa.create_engine(pg_url)
    with engine.begin() as conn:
        register_ontology(conn, ontology)
        insert_session(conn, make_session())
    yield engine
    engine.dispose()


@pytest.mark.services
def test_session_and_label_roundtrip(pg: sa.Engine) -> None:
    model = Provenance(source=Source.MODEL, model_version="vlm-0.1")
    labels = [
        make_label(action_payload(), label_id="a1", provenance=model, confidence=0.7),
        make_label({"kind": "gap", "hand": "right", "gap_type": "idle"}, label_id="g1",
                   t_start_ms=1_000, t_end_ms=2_000),
        make_label({"kind": "blur_track", "target": "reflection",
                    "keyframes": [{"t_ms": 0, "x": 1, "y": 1, "w": 5, "h": 5}]},
                   label_id="b1", stream_id="bodycam"),
    ]  # fmt: skip
    with pg.begin() as conn:
        assert insert_labels(conn, labels) == 3
    with pg.connect() as conn:
        assert get_session(conn, "s001") == make_session()
        assert get_labels(conn, "s001") == sorted(labels, key=lambda x: (x.t_start_ms, x.label_id))
        assert [x.label_id for x in get_labels(conn, "s001", kinds=["gap"])] == ["g1"]


@pytest.mark.services
def test_stream_order_is_preserved(pg: sa.Engine) -> None:
    base = make_session("s002")
    reordered = base.model_copy(update={"streams": tuple(reversed(base.streams))})
    assert [s.stream_id for s in reordered.streams] == ["imu", "bodycam"]
    with pg.begin() as conn:
        insert_session(conn, reordered)
    with pg.connect() as conn:
        assert get_session(conn, "s002") == reordered


@pytest.mark.services
def test_labels_are_immutable_except_review(pg: sa.Engine) -> None:
    with pg.begin() as conn:
        insert_labels(conn, [make_label(action_payload(), label_id="a1")])
    with pg.begin() as conn:
        record_review(conn, "a1", VerificationState.HUMAN_APPROVED, "rev01", FIXED_TIME)
    with pg.connect() as conn:
        [label] = get_labels(conn, "s001")
    assert label.verification.state is VerificationState.HUMAN_APPROVED
    assert label.verification.reviewer_id == "rev01"

    for statement in (
        "UPDATE label_records SET t_end_ms = 5 WHERE label_id = 'a1'",
        "UPDATE label_records SET payload = '{}'::jsonb WHERE label_id = 'a1'",
        "DELETE FROM label_records WHERE label_id = 'a1'",
    ):
        with pytest.raises(DBAPIError, match="label_records"), pg.begin() as conn:
            conn.execute(sa.text(statement))


@pytest.mark.services
def test_lifecycle_transitions_are_enforced(pg: sa.Engine) -> None:
    with pg.begin() as conn:
        set_lifecycle(conn, "s001", LifecycleState.PRIVACY_APPROVED)
        with pytest.raises(TransitionError):
            set_lifecycle(conn, "s001", LifecycleState.EXPORTED)
        set_lifecycle(conn, "s001", LifecycleState.WITHDRAWN)
        assert get_session(conn, "s001").lifecycle_state is LifecycleState.WITHDRAWN


@pytest.mark.services
def test_dataset_version_roundtrip(pg: sa.Engine) -> None:
    version = DatasetVersion(
        version_id="ds-0001", ontology_version="1.0.0", created_at=FIXED_TIME,
        snapshot_uri="lakefs://dlp/main@abc", splits={"s001": Split.GOLDEN},
    )  # fmt: skip
    with pg.begin() as conn:
        insert_dataset_version(conn, version)
    with pg.connect() as conn:
        assert get_dataset_version(conn, "ds-0001") == version


@pytest.mark.services
def test_registering_changed_ontology_under_same_version_fails(
    pg: sa.Engine, ontology: Ontology
) -> None:
    changed = ontology.model_copy(update={"status": "frozen"})
    with pytest.raises(ValueError, match="다른 내용"), pg.begin() as conn:
        register_ontology(conn, changed)


@pytest.mark.services
def test_update_stream_sync_changes_only_sync_fields(pg: sa.Engine) -> None:
    imu = make_session().stream("imu")
    synced = imu.model_copy(
        update={"offset_ms": 12.5, "clock_scale": 1.00005, "sync_method": SyncMethod.TAP_EVENT,
                "sync_confidence": 0.9, "manual_adjustment_ms": -2.0}
    )  # fmt: skip
    with pg.begin() as conn:
        update_stream_sync(conn, "s001", synced)
    with pg.connect() as conn:
        assert get_session(conn, "s001").stream("imu") == synced
    with pytest.raises(KeyError), pg.begin() as conn:
        update_stream_sync(conn, "s001", synced.model_copy(update={"stream_id": "nope"}))
