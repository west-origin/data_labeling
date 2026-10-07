"""데이터셋 버전 → 내보내기 → 이력 (PostgreSQL, make up).

완료 기준: 기본 정책에서 미검수 라벨 0건, 사용 중지 세션 0건. 결과에 원본 위치가 없다.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
import sqlalchemy as sa

from dlp_datasets.build import build_dataset_version
from dlp_datasets.lineage import session_lineage, withdraw_session
from dlp_datasets.policy import load_policy as load_dataset_policy
from dlp_datasets.snapshot import LocalSnapshotStore
from dlp_export.policy import ExportPolicy
from dlp_export.runner import run_export
from dlp_fixtures.video import vfr_times
from dlp_media.storage import LocalStore
from dlp_schema.db.migrate import upgrade
from dlp_schema.db.repository import insert_labels, insert_session, register_ontology
from dlp_schema.labels import VerificationState
from dlp_schema.ontology import Ontology
from dlp_schema.session import PrivacyState
from dlp_schema.testing import FIXED_TIME, make_session

from .conftest import ROOT, scenario_labels, write_blurred

pytestmark = pytest.mark.services


@pytest.fixture
def engine() -> Iterator[sa.Engine]:
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
    e = sa.create_engine(test_url)
    try:
        yield e
    finally:
        e.dispose()
        with admin.connect() as conn:
            conn.execute(sa.text(f'DROP DATABASE "{name}" WITH (FORCE)'))
        admin.dispose()


def test_export_applies_policy_and_records_history(
    engine: sa.Engine, policy: ExportPolicy, ontology: Ontology, tmp_path: Path
) -> None:
    snapshots = LocalSnapshotStore(tmp_path / "snap")
    labeling = LocalStore(tmp_path / "store", "dlp-labeling")
    datasets = LocalStore(tmp_path / "store", "dlp-datasets")
    sids = [f"s{i}" for i in range(4)]
    times = vfr_times(np.random.default_rng(4), 1000)
    with engine.begin() as conn:
        register_ontology(conn, ontology)
        for i, sid in enumerate(sids):
            insert_session(
                conn,
                make_session(
                    sid,
                    worker_id=f"w{i}",
                    site_id=f"site{i}",
                    streams=[
                        {
                            "stream_id": "bodycam",
                            "kind": "bodycam",
                            "sync_method": "reference",
                            "uri": f"s3://dlp-raw/sessions/{sid}/bodycam.mp4",
                        },
                    ],
                ).model_copy(update={"privacy_state": PrivacyState.APPROVED}),
            )
            insert_labels(conn, scenario_labels(sid, times))
            write_blurred(labeling, sid, times, tmp_path)
        build_dataset_version(
            conn, snapshots, load_dataset_policy(ROOT), version_id="dv1",
            ontology_version="1.0.0", golden_set_version=None, now=FIXED_TIME,
        )  # fmt: skip
        withdraw_session(conn, "s2", "동의 철회", FIXED_TIME + timedelta(days=1))

    def export(fmt: str, include_unreviewed: bool = False):
        with engine.begin() as conn:
            return run_export(
                conn, root=ROOT, version_id="dv1", fmt=fmt, target="buyer-a",  # type: ignore[arg-type]
                snapshots=snapshots, labeling=labeling, datasets=datasets, raw_bucket="dlp-raw",
                policy=policy, ontology=ontology, include_unreviewed=include_unreviewed,
                splits=None, now=FIXED_TIME + timedelta(days=2),
            )  # fmt: skip

    kept = {"s0", "s1", "s3"} & {sid for sid in sids}
    for fmt in ("intervals", "coco"):
        r = export(fmt)
        out = tmp_path / "store" / "dlp-datasets" / "exports" / r.record.export_id
        manifest = json.loads((out / "manifest.json").read_text())
        exported = {s["session_id"] for s in manifest["sessions"]}
        # 사용 중지 세션 0건 (데이터셋 버전에는 있었다)
        assert "s2" not in exported and exported <= kept and set(r.record.session_ids) == exported
        # 미검수 라벨 0건
        assert not [k for k in r.label_counts if k.endswith("/unreviewed")]
        assert manifest["verification_policy"]["include_unreviewed"] is False
        assert r.record.label_states == (
            VerificationState.HUMAN_APPROVED,
            VerificationState.HUMAN_CORRECTED,
            VerificationState.SAMPLE_VERIFIED,
        )
        text = "".join(p.read_text(errors="ignore") for p in out.rglob("*.json"))
        assert "dlp-raw" not in text and "reviewer-7" not in text
        if fmt == "coco":
            coco = json.loads((out / "coco" / "annotations.json").read_text())
            assert {a["verification"] for a in coco["annotations"]} <= {
                "human_approved", "human_corrected", "sample_verified"
            }  # fmt: skip
            assert {i["session_id"] for i in coco["images"]} == exported
        else:
            for sid in exported:
                f = json.loads((out / "intervals" / f"{sid}.json").read_text())
                assert {x["verification"] for x in f["labels"]} != {"unreviewed"}
                assert all(x["payload"]["kind"] != "blur_track" for x in f["labels"])

    # 미검수 포함은 명시적 옵션으로만, 이력에 정책이 남는다
    r = export("coco", include_unreviewed=True)
    assert any(k.endswith("/unreviewed") for k in r.label_counts)
    assert VerificationState.UNREVIEWED in r.record.label_states
    # 계보: 세션 → 내보내기 (사용 중지 세션은 이후 내보내기에 없다)
    with engine.connect() as conn:
        assert len(session_lineage(conn, "s0").exports) == 3
        assert session_lineage(conn, "s2").exports == []


@pytest.mark.isolated_env
def test_lerobot_export_end_to_end(
    engine: sa.Engine, policy: ExportPolicy, ontology: Ontology, tmp_path: Path
) -> None:
    snapshots = LocalSnapshotStore(tmp_path / "snap")
    labeling = LocalStore(tmp_path / "store", "dlp-labeling")
    datasets = LocalStore(tmp_path / "store", "dlp-datasets")
    times = vfr_times(np.random.default_rng(5), 800)
    with engine.begin() as conn:
        register_ontology(conn, ontology)
        for i, sid in enumerate(("a", "b")):
            insert_session(
                conn,
                make_session(sid, worker_id=f"w{i}", site_id=f"x{i}").model_copy(
                    update={"privacy_state": PrivacyState.APPROVED}
                ),
            )
            insert_labels(conn, scenario_labels(sid, times))
            write_blurred(labeling, sid, times, tmp_path)
        build_dataset_version(
            conn, snapshots, load_dataset_policy(ROOT), version_id="dv1",
            ontology_version="1.0.0", golden_set_version=None, now=FIXED_TIME,
        )  # fmt: skip
        r = run_export(
            conn, root=ROOT, version_id="dv1", fmt="lerobot", target="internal",
            snapshots=snapshots, labeling=labeling, datasets=datasets, raw_bucket="dlp-raw",
            policy=policy, ontology=ontology, include_unreviewed=False, splits=None,
            now=FIXED_TIME,
        )  # fmt: skip
    out = tmp_path / "store" / "dlp-datasets" / "exports" / r.record.export_id / "lerobot"
    assert r.details["episodes"] == 2 and r.details["loader_check"]["episodes"] == 2
    assert (out / "meta" / "info.json").exists() and (out / "meta" / "dlp_vocab.json").exists()
    episodes = json.loads((out / "meta" / "dlp_episodes.json").read_text())
    assert [e["session_id"] for e in episodes] == ["a", "b"]
    assert list(out.rglob("*.mp4")) and list(out.rglob("data/**/*.parquet"))
