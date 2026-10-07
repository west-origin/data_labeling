"""프라이버시 패키지 테스트 공용 픽스처 (WP2·WP5).

- `policy`: 저장소의 실제 privacy.yaml + defaults.yaml 정책 (테스트는 model_copy로 일부를 바꾼다).
- `blur`: 정답 블러 트랙을 아는 합성 VFR 바디캠 영상 (`dlp_fixtures.video.generate_blur_scenario`,
  seed 5).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dlp_fixtures.video import BlurScenario, generate_blur_scenario
from dlp_privacy.policy import PrivacyPolicy, load_policy

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="session")
def policy() -> PrivacyPolicy:
    """저장소 루트의 프라이버시 정책 (세션당 한 번 읽는다)."""
    return load_policy(ROOT)


@pytest.fixture(scope="session")
def blur(tmp_path_factory: pytest.TempPathFactory) -> tuple[BlurScenario, Path]:
    """합성 블러 시나리오와 그 영상 파일 경로 (세션 범위, seed 5).

    시나리오에는 프레임 시각(VFR), 대상별 정답 blur_track 라벨, 프레임 이미지가 들어 있다.
    """
    scenario = generate_blur_scenario(5)
    path = tmp_path_factory.mktemp("blur") / "bodycam.mp4"
    scenario.write(path)
    return scenario, path
