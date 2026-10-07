from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

from dlp_fixtures.sync import SyncScenario, generate_sync_scenario
from dlp_fixtures.video import BlurScenario, generate_blur_scenario
from dlp_schema.testing import FIXED_TIME


@pytest.fixture(scope="session")
def blur(tmp_path_factory: pytest.TempPathFactory) -> tuple[BlurScenario, Path]:
    scenario = generate_blur_scenario(3)
    path = tmp_path_factory.mktemp("blur") / "bodycam.mp4"
    scenario.write(path)
    return scenario, path


@pytest.fixture(scope="session")
def sync(tmp_path_factory: pytest.TempPathFactory) -> tuple[SyncScenario, Path]:
    scenario = generate_sync_scenario(4, recorded_at=FIXED_TIME, duration_ms=12_000)
    out = tmp_path_factory.mktemp("sync")
    scenario.write(out)
    return scenario, out


def _write_manifest(directory: Path, sync_dir: Path, **overrides: object) -> Path:
    manifest: dict[str, object] = {
        "session_id": "ing-s001",
        "domain": "cleaning",
        "worker_id": "w01",
        "site_id": "site01",
        "consent_version": "c1",
        "recorded_at": FIXED_TIME.isoformat(),
        "streams": [
            {"stream_id": "bodycam", "kind": "bodycam", "path": str(sync_dir / "bodycam.mp4")},
            {"stream_id": "third_person", "kind": "third_person",
             "path": str(sync_dir / "third_person.mp4")},
            {"stream_id": "glove_right", "kind": "glove_right",
             "path": str(sync_dir / "glove_right.parquet")},
        ],
    }  # fmt: skip
    manifest.update(overrides)
    path = directory / "manifest.yaml"
    path.write_text(yaml.safe_dump(manifest, allow_unicode=True), encoding="utf-8")
    return path


ManifestWriter = Callable[..., Path]


@pytest.fixture
def write_manifest() -> ManifestWriter:
    """(디렉터리, 동기화 픽스처 디렉터리, **덮어쓸 값) → 매니페스트 경로."""
    return _write_manifest
