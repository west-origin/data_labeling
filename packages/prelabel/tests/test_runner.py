from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import sqlalchemy as sa
import yaml

from dlp_fixtures.actions import ENTITIES, generate_action_scenario
from dlp_fixtures.io import write_parquet
from dlp_fixtures.video import generate_blur_scenario
from dlp_media.ingest import ingest_session, load_manifest
from dlp_media.storage import S3Store
from dlp_prelabel.adapters.stubs import OraclePredictor
from dlp_prelabel.policy import load_policy
from dlp_prelabel.runner import CONTACT_VERSION, run_prelabel
from dlp_schema.db.migrate import upgrade
from dlp_schema.db.repository import (
    get_labels,
    get_session,
    register_ontology,
    set_lifecycle,
    set_privacy_state,
    update_stream_sync,
)
from dlp_schema.labels import BoxKeyframe, BoxTrackPayload, HandStatePayload
from dlp_schema.ontology import load_ontology
from dlp_schema.predictor import Predictor
from dlp_schema.session import LifecycleState, PrivacyState, SyncMethod
from dlp_schema.testing import FIXED_TIME, make_label

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


def test_prelabel_session_with_glove_contacts(pg: sa.Engine, tmp_path: Path) -> None:
    actions = generate_action_scenario(1, n_units=10)
    generate_blur_scenario(1).write(tmp_path / "bodycam.mp4")
    write_parquet(
        tmp_path / "glove_right.parquet",
        {"t_ms": actions.glove_t_ms, "pressure_0": actions.glove_pressure},
    )
    sid = f"pre-{uuid.uuid4().hex[:8]}"
    manifest = {
        "session_id": sid,
        "domain": "cleaning",
        "worker_id": "w01",
        "site_id": "site01",
        "consent_version": "c1",
        "recorded_at": FIXED_TIME.isoformat(),
        "ontology_version": "1.0.0",
        "streams": [
            {"stream_id": "bodycam", "kind": "bodycam", "path": "bodycam.mp4"},
            {"stream_id": "glove_right", "kind": "glove_right", "path": "glove_right.parquet"},
        ],
    }
    (tmp_path / "m.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    raw = S3Store.from_env("dlp-raw")
    with pg.begin() as conn:
        session = ingest_session(*load_manifest(tmp_path / "m.yaml"), raw, conn).session
        glove = session.stream("glove_right").model_copy(
            update={"sync_method": SyncMethod.TAP_EVENT}
        )
        update_stream_sync(conn, sid, glove)  # 장갑이 이미 동기화되었다고 둔다 (오프셋 0)
        set_privacy_state(conn, sid, PrivacyState.APPROVED)
        set_lifecycle(conn, sid, LifecycleState.PRIVACY_APPROVED)

    # 고정 객체 박스 정답을 객체 탐지 stub의 출처로 쓴다
    boxes = [
        make_label(
            BoxTrackPayload(
                entity_id=eid,
                class_id=cls,
                keyframes=tuple(
                    BoxKeyframe(t_ms=t, x=p[0] - 15, y=p[1] - 15, w=30, h=30)
                    for t in actions.frame_times
                ),
            ),
            label_id=f"gt-{eid}",
            session_id=sid,
            stream_id="bodycam",
            t_start_ms=actions.frame_times[0],
            t_end_ms=actions.frame_times[-1],
        )
        for eid, cls, _, p, _ in ENTITIES
        if eid in {"drawer_01", "bucket_01", "sink_01"}
    ]
    predictors: list[Predictor] = [
        OraclePredictor("hands", actions.labels, ("keypoint_track",), now=FIXED_TIME),
        OraclePredictor("objects", boxes, ("box_track",), now=FIXED_TIME),
    ]
    policy, ontology = load_policy(ROOT), load_ontology(ROOT / "config/ontology/v1")
    with pg.begin() as conn:
        summary = run_prelabel(conn, sid, raw, predictors, policy, ontology, FIXED_TIME)
    assert summary.produced == {"bodycam/hands": 1, "bodycam/objects": 3}
    truth = [
        x
        for x in actions.labels
        if isinstance(x.payload, HandStatePayload) and x.payload.contact_target_kind != "none"
    ]
    assert summary.contacts == len(truth)

    with pg.begin() as conn:
        again = run_prelabel(conn, sid, raw, predictors, policy, ontology, FIXED_TIME)
        contacts = [
            x
            for x in get_labels(conn, sid, kinds=["hand_state"])
            if x.provenance.model_version == CONTACT_VERSION
        ]
        assert get_session(conn, sid).lifecycle_state is LifecycleState.PRELABELED
    assert sorted(again.skipped) == ["bodycam/hands", "bodycam/objects"] and again.contacts == 0
    for c, gt in zip(contacts, truth, strict=True):
        assert abs(c.t_start_ms - gt.t_start_ms) <= 33
        gp = gt.payload
        assert isinstance(gp, HandStatePayload) and isinstance(c.payload, HandStatePayload)
        if gp.target_id in {"drawer_01", "bucket_01", "sink_01"}:
            assert c.payload.target_id == gp.target_id
            assert c.payload.contact_target_kind == (
                "fixed_surface" if gp.target_id == "sink_01" else "object"
            )
    assert np.isfinite([c.confidence or 0 for c in contacts]).all()
