"""dlp_schema 테스트 공용 픽스처: 저장소 루트와 온톨로지 v1.

마이그레이션된 일회용 PostgreSQL(`pg_url`, `pg`)도 공유한다.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
import sqlalchemy as sa

from dlp_schema.db.migrate import upgrade
from dlp_schema.db.repository import insert_session, register_ontology
from dlp_schema.ontology import Ontology, load_ontology
from dlp_schema.testing import make_session

# 저장소 루트 (packages/schema/tests/ → 세 단계 위)
ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="session")
def repo() -> Path:
    """저장소 루트 경로 (config/, schemas/를 찾는 기준)."""
    return ROOT


@pytest.fixture(scope="session")
def ontology() -> Ontology:
    """실제 저장소의 온톨로지 v1 (config/ontology/v1). 세션 범위로 한 번만 읽는다."""
    return load_ontology(ROOT / "config" / "ontology" / "v1")


DEFAULT_URL = "postgresql+psycopg://dlp:dlp-dev-password@localhost:5432/dlp"


@pytest.fixture
def pg_url() -> Iterator[str]:
    """테스트마다 새 데이터베이스를 만들고 마이그레이션을 적용한다.

    관리 DB(postgres)에 AUTOCOMMIT으로 붙어 `dlp_test_<임의 8자>` DB를 만들고, 끝나면 강제로 지운다.
    """
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
    """마이그레이션된 임시 DB 엔진. 온톨로지 v1과 기본 세션 s001을 미리 등록한다."""
    engine = sa.create_engine(pg_url)
    with engine.begin() as conn:
        register_ontology(conn, ontology)
        insert_session(conn, make_session())
    yield engine
    engine.dispose()


@pytest.fixture
def target_ontology(ontology: Ontology) -> Ontology:
    """이관 테스트 목적지: 기존 ID를 유지하면서 grip·cloth_side·origin을 추가한 v1.1."""
    data = ontology.model_dump(mode="json")
    data["version"] = "1.1.0"
    data["verbs"]["grip"] = {"ko": "잡기", "level": "primitive"}
    data["objects"]["rag"]["tool"]["working_parts"].append("cloth_side")
    data["surface_parts"]["origin"] = {"ko": "원점"}
    return Ontology.model_validate(data)
