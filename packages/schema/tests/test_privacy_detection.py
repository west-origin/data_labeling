"""탐지 완료 계약의 DB 왕복·스트림 참조 무결성 (SQLite, ADR 0032)."""

from datetime import timedelta, timezone
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError

from dlp_schema.db.migrate import upgrade
from dlp_schema.db.repository import (
    insert_session,
    latest_privacy_detection,
    list_privacy_detections,
    record_privacy_detection,
)
from dlp_schema.testing import FIXED_TIME, make_session


def test_completed_detection_roundtrips_and_references_existing_stream(tmp_path: Path) -> None:
    """완료 0건 기록도 버전·시간대가 유지되고 없는 스트림에는 기록할 수 없다."""
    url = f"sqlite:///{tmp_path / 'runs.db'}"
    upgrade(url)
    engine = sa.create_engine(url)
    with engine.begin() as conn:
        conn.execute(sa.text("PRAGMA foreign_keys=ON"))
        insert_session(conn, make_session(ontology_version=None))
        record_privacy_detection(
            conn,
            "s001",
            "bodycam",
            ["m2", "m1", "m1"],
            FIXED_TIME.astimezone(timezone(timedelta(hours=9))),
        )
        record_privacy_detection(conn, "s001", "bodycam", ["m3"], FIXED_TIME + timedelta(seconds=1))
        runs = list_privacy_detections(conn, "s001")
        assert len(runs) == 2 and runs[0].model_versions == ("m1", "m2")
        assert runs[0].at == FIXED_TIME and runs[0].event_id < runs[1].event_id
        assert latest_privacy_detection(conn, "s001", "bodycam") == runs[1]
        assert latest_privacy_detection(conn, "s001", "third_person") is None
    with pytest.raises(IntegrityError), engine.begin() as conn:
        record_privacy_detection(conn, "s001", "missing", ["m1"], FIXED_TIME)
    engine.dispose()
