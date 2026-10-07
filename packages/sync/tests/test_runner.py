"""`dlp sync run`/`adjust`의 DB·저장소 통합 테스트 (WP4, `@pytest.mark.services`).

실행 중인 PostgreSQL과 SeaweedFS S3(`make up`)가 필요하다. 테스트마다 일회용 DB를 만들고 지운다.
수집(`ingest_session`) → 동기화(`run_sync`) → DB·보고서 확인 → 사람 조정 → 재동기화 → 재수집 순서로
전체 흐름을 본다. 정답은 합성 시나리오(seed 2)의 `clocks`다.

주의: 이 테스트는 원본 버킷(`dlp-raw`)을 `S3Store.from_env`로 직접 연다 (감사 저장소를 거치지 않는
테스트 전용 경로).
"""

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
    """마이그레이션을 적용한 일회용 PostgreSQL DB 엔진 (테스트 끝에 DB를 강제로 지운다)."""
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
    """수집한 세션을 동기화하면 DB 스트림 필드와 저장소 보고서가 갱신되는지 종단으로 검증한다.

    - 3인칭은 qr_slate로, 정답 시계와 10 s 지점에서 1프레임(33 ms) 안.
    - 장갑은 tap_event.
    - 저장소 `sessions/<세션>/derived/sync_report.json`이 반환한 보고서와 같다.
    - `adjust`로 넣은 -4.5 ms가 재동기화 뒤에도 남는다.
    - 동기화 결과가 바뀐 세션을 같은 매니페스트로 다시 수집해도 "unchanged"(충돌 아님).
    """
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

    # 동기화 결과가 바뀐 세션을 같은 매니페스트로 다시 수집해도 충돌이 아니다
    with pg.begin() as conn:
        reingest = ingest_session(*load_manifest(tmp_path / "m.yaml"), store, conn)
    assert reingest.db == "unchanged"
