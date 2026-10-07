from __future__ import annotations

from pathlib import Path

import pytest

from dlp_fixtures.video import BlurScenario, generate_blur_scenario
from dlp_privacy.policy import PrivacyPolicy, load_policy

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="session")
def policy() -> PrivacyPolicy:
    return load_policy(ROOT)


@pytest.fixture(scope="session")
def blur(tmp_path_factory: pytest.TempPathFactory) -> tuple[BlurScenario, Path]:
    scenario = generate_blur_scenario(5)
    path = tmp_path_factory.mktemp("blur") / "bodycam.mp4"
    scenario.write(path)
    return scenario, path
