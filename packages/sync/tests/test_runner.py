from __future__ import annotations

import json
import os
import tempfile
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
import sqlalchemy as sa
import yaml

from dlp_fixtures.sync import generate_sync_scenario
from dlp_media.ingest import ingest_session, load_manifest
from dlp_media.storage import S3Store
from dlp_schema.db.migrate import upgrade
from dlp_schema.db.repository import get_session
from dlp_schema.session import SyncMethod
from dlp_schema.testing import FIXED_TIME
from dlp_sync.policy import SyncPolicy
from dlp_sync.runner import adjust, run_sync

pytestmark = pytest.mark.services


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


def test_ingest_then_sync_updates_db_and_writes_report(
    pg: sa.Engine, policy: SyncPolicy, tmp_path: Path
) -> None:
    scenario = generate_sync_scenario(2, recorded_at=FIXED_TIME, duration_ms=30_000)
    scenario.write(tmp_path, videos=False)
    for name in ("bodycam", "third_person"):
        scenario.write_video(tmp_path / f"{name}.mp4", name, fps=30)
    sid = f"sync-{uuid.uuid4().hex[:8]}"
    manifest = {
        "session_id": sid, "domain": "cleaning", "worker_id": "w01", "site_id": "site01",
        "consent_version": "c1", "recorded_at": FIXED_TIME.isoformat(),
        "streams": [
            {"stream_id": "bodycam", "kind": "bodycam", "path": "bodycam.mp4"},
            {"stream_id": "third_person", "kind": "third_person", "path": "third_person.mp4"},
            {"stream_id": "glove_right", "kind": "glove_right", "path": "glove_right.parquet"},
        ],
    }  # fmt: skip
    (tmp_path / "m.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    store = S3Store.from_env("dlp-raw")
    with pg.begin() as conn:
        ingest_session(*load_manifest(tmp_path / "m.yaml"), store, conn)

    with pg.begin() as conn:
        synced, report = run_sync(conn, sid, store, policy)
    with pg.connect() as conn:
        stored = get_session(conn, sid)
    assert stored == synced
    third = stored.stream("third_person")
    assert third.sync_method is SyncMethod.QR_SLATE
    truth = scenario.clocks["third_person"]
    assert abs(third.to_master_ms(10_000) - float(truth.to_master(10_000))) <= 1000 / 30
    assert stored.stream("glove_right").sync_method is SyncMethod.TAP_EVENT

    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "r.json"
        store.get_file(f"sessions/{sid}/derived/sync_report.json", dest)
        saved = json.loads(dest.read_text(encoding="utf-8"))
    assert saved == json.loads(json.dumps(report.to_dict()))

    with pg.begin() as conn:
        adjust(conn, sid, "third_person", -4.5)
    with pg.begin() as conn:
        again, _ = run_sync(conn, sid, store, policy)
    assert again.stream("third_person").manual_adjustment_ms == -4.5
