"""dlp_schema 테스트 공용 픽스처: 저장소 루트와 온톨로지 v1.

DB 픽스처(`pg_url`, `pg`)는 `test_db.py`에 있다.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dlp_schema.ontology import Ontology, load_ontology

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
