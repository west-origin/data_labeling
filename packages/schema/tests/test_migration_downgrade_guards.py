"""추가 전용 기록이 있으면 오래된 리비전 downgrade를 거부하는지 (F-SCH-28).

0005(검수 배정·measurement 열), 0009(원본 접근·운영 기록), 0011(수명 주기 전이),
0012(탐지 완료 기록)는
기록이 있을 때 표·열을 지우지 않고 `RuntimeError`로 멈춘다 (0017·0022·0024와 같은 방식).
거부되면 downgrade 전체가 한 트랜잭션으로 되돌아가 리비전·표가 그대로다.
"""

from __future__ import annotations

import pytest
import sqlalchemy as sa

from dlp_schema.db.migrate import downgrade
from dlp_schema.db.privacy_repository import record_privacy_detection
from dlp_schema.db.repository import insert_raw_access
from dlp_schema.ops import RawAccessEvent
from dlp_schema.testing import FIXED_TIME

pytestmark = pytest.mark.services


def _revision_and_tables(url: str) -> tuple[str, set[str]]:
    """지금 alembic 리비전과 표 이름 집합."""
    engine = sa.create_engine(url)
    try:
        with engine.connect() as conn:
            rev = conn.execute(sa.text("SELECT version_num FROM alembic_version")).scalar_one()
            return rev, set(sa.inspect(conn).get_table_names())
    finally:
        engine.dispose()


def test_detection_runs_block_downgrade_of_0012(pg: sa.Engine) -> None:
    """탐지 완료 기록이 있으면 0012를 되돌리지 않고, 리비전·표가 그대로 남는다."""
    with pg.begin() as conn:
        record_privacy_detection(conn, "s001", "bodycam", ["yunet-1"], FIXED_TIME)
    url = pg.url.render_as_string(hide_password=False)
    pg.dispose()
    before = _revision_and_tables(url)
    with pytest.raises(RuntimeError, match="0012"):
        downgrade(url, "0011")
    assert _revision_and_tables(url) == before


def test_lifecycle_events_block_downgrade_of_0011(pg: sa.Engine) -> None:
    """세션 등록이 남긴 수명 주기 전이 기록이 있으면 0011을 되돌리지 않는다."""
    url = pg.url.render_as_string(hide_password=False)
    pg.dispose()
    with pytest.raises(RuntimeError, match="0011"):
        downgrade(url, "0010")
    assert "session_lifecycle_events" in _revision_and_tables(url)[1]


def test_raw_access_log_blocks_downgrade_of_0009(pg_url: str) -> None:
    """원본 접근 기록이 있으면 0009를 되돌리지 않는다 (세션 없는 접근 기록 하나)."""
    engine = sa.create_engine(pg_url)
    with engine.begin() as conn:
        insert_raw_access(
            conn,
            RawAccessEvent(
                event_id="ev1",
                at=FIXED_TIME,
                actor="op1",
                purpose="privacy.detect",
                action="read",
                bucket="dlp-raw",
                key="s001/bodycam.mp4",
            ),
        )
    engine.dispose()
    with pytest.raises(RuntimeError, match="0009"):
        downgrade(pg_url, "0008")
    assert "raw_access_log" in _revision_and_tables(pg_url)[1]


def test_measurement_labels_block_downgrade_of_0005(pg_url: str) -> None:
    """measurement가 적힌 라벨이 있으면 0005가 measurement 열을 지우지 않는다.

    0011 이전으로 내린 뒤(세션 등록이 전이 기록을 남기지 않게) SQL로 세션·측정 라벨을 넣는다.
    """
    downgrade(pg_url, "0010")
    engine = sa.create_engine(pg_url)
    with engine.begin() as conn:
        conn.execute(
            sa.text(
                "INSERT INTO ontology_versions (version, status, content)"
                " VALUES ('1.0.0', 'draft', '{}')"
            )
        )
        conn.execute(
            sa.text(
                "INSERT INTO sessions (session_id, domain, worker_id, site_id, consent_version,"
                " recorded_at, duration_ms, calibration, privacy_state, lifecycle_state,"
                " created_at) VALUES ('s1', 'cleaning', 'w1', 'p1', 'c1', :t, 1000, '{}',"
                " 'approved', 'prelabeled', :t)"
            ),
            {"t": FIXED_TIME},
        )
        conn.execute(
            sa.text(
                "INSERT INTO label_records (label_id, session_id, kind, t_start_ms, t_end_ms,"
                " ontology_version, source, evidence, verification_state, retracted,"
                " seeded_error, created_at, payload, measurement) VALUES ('l1', 's1', 'gap', 0,"
                " 1000, '1.0.0', 'human', 'visual', 'unreviewed', false, false, :t,"
                " '{\"kind\": \"gap\"}', 'blind')"
            ),
            {"t": FIXED_TIME},
        )
    engine.dispose()
    with pytest.raises(RuntimeError, match="0005"):
        downgrade(pg_url, "0004")
    engine = sa.create_engine(pg_url)
    with engine.connect() as conn:
        cols = {c["name"] for c in sa.inspect(conn).get_columns("label_records")}
    engine.dispose()
    assert "measurement" in cols
