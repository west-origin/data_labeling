"""미디어 수집 테스트 공용 픽스처 (WP2·WP3).

- `blur`: 정답 프레임 시각을 아는 합성 VFR 영상 (`generate_blur_scenario`, seed 3).
- `sync`: 동기화 시나리오 (`generate_sync_scenario`, seed 4, 12초): 바디캠·3인칭 영상(오디오 포함,
  10 fps), IMU 사이드카(200 Hz), 장갑 Parquet(100 Hz, 압력 5채널).
- `write_manifest`: 그 파일들을 가리키는 수집 매니페스트 YAML을 쓰는 함수.
"""

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
    """합성 블러 시나리오(VFR, seed 3)와 영상 파일 경로 (세션 범위)."""
    scenario = generate_blur_scenario(3)
    path = tmp_path_factory.mktemp("blur") / "bodycam.mp4"
    scenario.write(path)
    return scenario, path


@pytest.fixture(scope="session")
def sync(tmp_path_factory: pytest.TempPathFactory) -> tuple[SyncScenario, Path]:
    """합성 동기화 시나리오(seed 4, 12초)와 파일이 쓰인 디렉터리 (세션 범위)."""
    scenario = generate_sync_scenario(4, recorded_at=FIXED_TIME, duration_ms=12_000)
    out = tmp_path_factory.mktemp("sync")
    scenario.write(out)
    return scenario, out


def _write_manifest(directory: Path, sync_dir: Path, **overrides: object) -> Path:
    """수집 매니페스트를 directory/manifest.yaml로 쓴다.

    기본: 세션 ing-s001, 청소 도메인, 바디캠·3인칭·오른손 장갑 스트림(sync_dir의 절대 경로).
    overrides로 최상위 키를 바꾼다 (None을 주면 그 값이 null이 된다).
    """
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
