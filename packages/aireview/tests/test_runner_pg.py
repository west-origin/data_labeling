"""실행기의 PostgreSQL 성질 (services, WP17.5, ADR 0040 5.1·I5·I8).

`make up` 뒤 `make test-services`로 돈다. 테스트마다 마이그레이션한 일회용 DB를 만들고 지운다
(개발 DB에는 마이그레이션을 적용하지 않는다).

- 세션 잠금 경합: 다른 연결이 잠금을 쥐고 있으면 실행하지 않고(None → 종료 7), 실행이 끝나면 잠금을
  푼다 (몇 시간짜리 실행이 열린 트랜잭션을 쥐지 않게 잠금 뒤 커밋한다).
- 잠금이 뚫린 두 번째 실행기(기록된 단위를 모르는 기록 대상)가 같은 단위를 다시 써도 `ON CONFLICT DO
  NOTHING`으로 행이 늘지 않고 트리거에 걸리지 않는다.
- 판정해도 `label_records`는 한 행도 바뀌지 않는다.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
import sqlalchemy as sa

from dlp_aireview.oracle import OracleJudge
from dlp_aireview.policy import AiReviewPolicy
from dlp_aireview.runner import (
    DbSink,
    RunOptions,
    load_input,
    run_from_db,
    run_session,
    session_lock,
)
from dlp_schema.db.aireview_repository import list_ai_run_events, list_ai_units
from dlp_schema.db.locks import try_lock_aireview, unlock_aireview
from dlp_schema.db.migrate import upgrade
from dlp_schema.db.tables import label_records

from .blurred import RAW, DbScene, db_scene

pytestmark = pytest.mark.services
DEFAULT_URL = "postgresql+psycopg://dlp:dlp-dev-password@localhost:5432/dlp"


@pytest.fixture
def pg() -> Iterator[sa.Engine]:
    """마이그레이션된 일회용 PostgreSQL DB (끝나면 지운다)."""
    admin_url = sa.make_url(os.environ.get("DLP_DATABASE_URL", DEFAULT_URL))
    name = f"dlp_test_{uuid.uuid4().hex[:8]}"
    admin = sa.create_engine(admin_url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(sa.text(f'CREATE DATABASE "{name}"'))
    url = admin_url.set(database=name).render_as_string(hide_password=False)
    upgrade(url)
    engine = sa.create_engine(url)
    try:
        yield engine
    finally:
        engine.dispose()
        with admin.connect() as conn:
            conn.execute(sa.text(f'DROP DATABASE "{name}" WITH (FORCE)'))
        admin.dispose()


def _labels(engine: sa.Engine) -> list[tuple[object, ...]]:
    """라벨 이력 표 전체."""
    with engine.connect() as conn:
        return [tuple(r) for r in conn.execute(sa.select(label_records).order_by("label_id"))]


def _run(db: DbScene, policy: AiReviewPolicy) -> object:
    """run_from_db (작업자 4개)."""
    return run_from_db(
        db.engine, db.labeling, db.session_id, OracleJudge(list(db.scene.labels)), policy,
        db.privacy, raw_bucket=RAW, options=RunOptions(workers=4),
    )  # fmt: skip


def test_session_lock_contention_and_release(
    pg: sa.Engine, tmp_path: Path, policy: AiReviewPolicy
) -> None:
    """다른 연결이 세션 잠금을 쥐면 실행하지 않는다 (None). 실행이 끝나면 잠금이 풀린다."""
    db = db_scene(tmp_path, engine=pg)
    before = _labels(pg)
    with session_lock(pg, db.session_id) as got:
        assert got
        assert _run(db, policy) is None
        with pg.connect() as conn:
            assert try_lock_aireview(conn, db.session_id) is False
            conn.commit()
    summary = _run(db, policy)
    assert getattr(summary, "status", None) == "completed"
    with pg.connect() as conn:
        assert try_lock_aireview(conn, db.session_id) is True
        unlock_aireview(conn, db.session_id)
        conn.commit()
        assert [e.event.value for e in list_ai_run_events(conn, db.session_id)] == [
            "started", "completed",
        ]  # fmt: skip
    assert _labels(pg) == before


class _Forgetful(DbSink):
    """잠금이 뚫린 두 번째 실행기 흉내: 처음 한 번은 기록된 단위를 모른다."""

    def __init__(self, engine: sa.Engine) -> None:
        super().__init__(engine)
        self.calls = 0

    def done_unit_ids(self, session_id: str, version: str) -> set[str]:
        self.calls += 1
        return set() if self.calls == 1 else super().done_unit_ids(session_id, version)

    def is_complete(self, session_id: str, version: str, plan_digest: str) -> bool:
        return False


def test_duplicate_units_are_ignored_on_conflict(
    pg: sa.Engine, tmp_path: Path, policy: AiReviewPolicy
) -> None:
    """같은 단위를 다시 판정해 써도 행이 늘지 않는다 (ON CONFLICT DO NOTHING, 트리거 오류 없음)."""
    db = db_scene(tmp_path, engine=pg)
    _run(db, policy)
    with pg.connect() as conn:
        first = {u.unit_id: u.run_id for u in list_ai_units(conn, db.session_id)}
    work = tmp_path / "w"
    work.mkdir()
    inp = load_input(pg, db.labeling, db.session_id, policy, db.privacy, work, raw_bucket=RAW)
    judge = OracleJudge(list(db.scene.labels))
    summary = run_session(inp, judge, _Forgetful(pg), policy, options=RunOptions(workers=4))
    assert summary.status == "completed" and summary.written == len(first) and judge.calls
    with pg.connect() as conn:
        again = {u.unit_id: u.run_id for u in list_ai_units(conn, db.session_id)}
    assert again == first  # 먼저 쓴 행(처음 실행 ID)이 남는다
