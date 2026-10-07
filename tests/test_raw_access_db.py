"""완료 기준 (WP16): 원본 열람 이벤트가 빠짐없이 감사 로그에 남는다 (PostgreSQL, S3, make up).

수집 → 블러 탐지 → 프리라벨을 감사 저장소로 돌리고, 실제 원본 저장소 호출 수와 감사 로그를 맞춘다.
작업이 실패해 트랜잭션이 되돌아가도 접근 기록은 남는다.
"""

from __future__ import annotations

import os
import uuid
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import pytest
import sqlalchemy as sa
import yaml
from sqlalchemy.exc import DBAPIError

from dlp_fixtures.actions import generate_action_scenario
from dlp_fixtures.video import generate_blur_scenario
from dlp_media.audit import AuditedStore, DbAccessSink
from dlp_media.ingest import ingest_session, load_manifest
from dlp_media.storage import S3Store, StoredObject
from dlp_prelabel.adapters.stubs import OraclePredictor
from dlp_prelabel.policy import load_policy as load_prelabel_policy
from dlp_prelabel.runner import run_prelabel
from dlp_privacy.detectors.oracle import OracleDetector
from dlp_privacy.policy import TargetPolicy
from dlp_privacy.policy import load_policy as load_privacy_policy
from dlp_privacy.runner import detect_session
from dlp_schema.db.migrate import upgrade
from dlp_schema.db.repository import (
    get_session,
    list_raw_access,
    register_ontology,
    set_privacy_state,
)
from dlp_schema.ontology import load_ontology
from dlp_schema.session import PrivacyState
from dlp_schema.testing import FIXED_TIME

pytestmark = pytest.mark.services
ROOT = Path(__file__).resolve().parents[1]


class Counting:
    """실제 원본 저장소 호출을 센다 (감사 저장소 안쪽)."""

    def __init__(self, inner: S3Store) -> None:
        self.inner, self.bucket = inner, inner.bucket
        self.calls: Counter[tuple[str, str]] = Counter()

    def uri(self, key: str) -> str:
        return self.inner.uri(key)

    def head(self, key: str) -> StoredObject | None:
        return self.inner.head(key)

    def put_file(self, key: str, path: Path, sha256: str) -> None:
        self.calls[("write", key)] += 1
        self.inner.put_file(key, path, sha256)

    def get_file(self, key: str, dest: Path) -> None:
        self.calls[("read", key)] += 1
        self.inner.get_file(key, dest)


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


def test_every_raw_read_is_in_the_audit_log(pg: sa.Engine, tmp_path: Path) -> None:
    sid = f"aud-{uuid.uuid4().hex[:8]}"
    blur = generate_blur_scenario(3, session_id=sid)
    blur.write(tmp_path / "bodycam.mp4")
    manifest = {
        "session_id": sid, "domain": "cleaning", "worker_id": "w", "site_id": "s",
        "consent_version": "c1", "recorded_at": FIXED_TIME.isoformat(), "ontology_version": "1.0.0",
        "streams": [{"stream_id": "bodycam", "kind": "bodycam", "path": "bodycam.mp4"}],
    }  # fmt: skip
    (tmp_path / "m.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    sink = DbAccessSink(pg)
    inner = Counting(S3Store.from_env("dlp-raw"))

    def store(purpose: str) -> AuditedStore:
        return AuditedStore(inner, sink, actor="svc-pipeline", purpose=purpose)

    privacy = load_privacy_policy(ROOT)
    oracle_policy = privacy.model_copy(
        update={"targets": {t: TargetPolicy(margin=tp.margin, detectors=("oracle",))
                            for t, tp in privacy.targets.items()}}
    )  # fmt: skip
    with pg.begin() as conn:
        register_ontology(conn, load_ontology(ROOT / "config/ontology/v1"))
        ingest_session(*load_manifest(tmp_path / "m.yaml"), store("media.ingest"), conn)
        detect_session(
            conn, sid, store("privacy.detect"), {"oracle": OracleDetector("oracle", blur.labels)},
            {}, oracle_policy, FIXED_TIME,
        )  # fmt: skip
        set_privacy_state(conn, sid, PrivacyState.APPROVED)
        actions = generate_action_scenario(1, session_id=sid, n_units=2)
        run_prelabel(
            conn, sid, store("prelabel.run"),
            [OraclePredictor("hands", actions.labels, ("keypoint_track",), now=FIXED_TIME)],
            load_prelabel_policy(ROOT), load_ontology(ROOT / "config/ontology/v1"), FIXED_TIME,
        )  # fmt: skip

    # 실패한 작업 안의 열람도 남는다 (감사 기록은 따로 커밋)
    with pg.connect() as conn:
        uri = get_session(conn, sid).reference_stream.uri
    key = uri.removeprefix(inner.uri(""))
    with pytest.raises(RuntimeError), pg.begin() as conn:
        store("debug.view").get_file(key, tmp_path / "x.mp4")
        conn.execute(sa.text("SELECT 1"))
        raise RuntimeError("작업 실패")

    with pg.connect() as conn:
        events = list_raw_access(conn)
    logged = Counter((e.action, e.key) for e in events if e.action in ("read", "write"))
    assert inner.calls and logged == inner.calls  # 원본 저장소 호출 하나하나가 기록에 있다
    assert {e.purpose for e in events} == {
        "media.ingest", "privacy.detect", "prelabel.run", "debug.view",
    }  # fmt: skip
    assert {e.session_id for e in events} == {sid} and {e.actor for e in events} == {"svc-pipeline"}

    # 감사 기록은 고치거나 지울 수 없다
    with pytest.raises(DBAPIError, match="추가만"), pg.begin() as conn:
        conn.execute(sa.text("DELETE FROM raw_access_log"))
