"""픽스처 생성기 테스트 공용 픽스처: 저장소의 온톨로지 v1 (정답 라벨 검증용)."""

from __future__ import annotations

from pathlib import Path

import pytest

from dlp_schema.ontology import Ontology, load_ontology

# 저장소 루트 (packages/fixtures/tests/conftest.py에서 세 단계 위)
ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="session")
def ontology() -> Ontology:
    """`config/ontology/v1` 온톨로지 (세션 범위에서 한 번 읽는다)."""
    return load_ontology(ROOT / "config" / "ontology" / "v1")
