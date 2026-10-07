from __future__ import annotations

from pathlib import Path

import pytest

from dlp_schema.ontology import Ontology, load_ontology

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="session")
def repo() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def ontology() -> Ontology:
    return load_ontology(ROOT / "config" / "ontology" / "v1")
