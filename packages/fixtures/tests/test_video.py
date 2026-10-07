"""블러 대상 영상 생성기(`generate_blur_scenario`)와 VFR 쓰기 테스트 (WP2, WP5 전제).

VFR 프레임 간격, 프라이버시 대상 6종의 정답 라벨, 화면 진입·이탈·거울 안 반사, 인코딩 후 PTS와 대상
픽셀 색 보존을 확인한다. seed 5.
"""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np
import pytest

from dlp_fixtures.video import (
    TARGET_COLORS,
    BlurScenario,
    generate_blur_scenario,
    read_frame_times,
    target_boxes_at,
)
from dlp_schema.labels import BlurTrackPayload
from dlp_schema.ontology import Ontology
from dlp_schema.validation import check_label


@pytest.fixture(scope="module")
def scenario() -> BlurScenario:
    """seed 5 블러 시나리오 (모듈 범위 캐시)."""
    return generate_blur_scenario(5)


def test_frame_rate_is_variable(scenario: BlurScenario) -> None:
    """프레임 간격이 세 가지 넘게 섞여 있고 25~50 ms 안인지 (VFR)."""
    gaps = set(np.diff(scenario.frame_times).tolist())
    assert len(gaps) > 2
    assert min(gaps) >= 25 and max(gaps) <= 50


def test_labels_cover_every_required_privacy_target(
    scenario: BlurScenario, ontology: Ontology
) -> None:
    """정답 블러 트랙이 필수 대상 6종을 모두 포함하고 온톨로지 검증을 통과하는지 검증한다."""
    targets = {x.payload.target for x in scenario.labels if isinstance(x.payload, BlurTrackPayload)}
    assert {"face", "reflection", "document", "screen", "photo", "shipping_label"} <= targets
    for label in scenario.labels:
        assert check_label(label, ontology) == []


def test_targets_enter_exit_and_stay_inside_mirror(scenario: BlurScenario) -> None:
    """송장은 화면 밖으로 나가고(`outside` 키프레임), 얼굴은 왼쪽 가장자리에 걸쳐 들어오며(첫
    박스가 잘림), 반사는 늘 거울 영역 안에 있는지 검증한다.
    """
    by_target = {
        x.payload.target: x.payload.keyframes
        for x in scenario.labels
        if isinstance(x.payload, BlurTrackPayload)
    }
    assert any(k.outside for k in by_target["shipping_label"])  # 화면 밖으로 나간다
    assert by_target["face"][0].w < 36  # 왼쪽 가장자리에 걸쳐 들어온다
    mx, my, mw, mh = scenario.mirror
    for k in by_target["reflection"]:
        assert not k.outside
        assert mx <= k.x and k.x + k.w <= mx + mw and my <= k.y and k.y + k.h <= my + mh


def test_encoded_video_keeps_vfr_timestamps_and_target_pixels(
    scenario: BlurScenario, tmp_path: Path
) -> None:
    """MP4로 쓴 뒤 읽은 프레임 PTS가 정답 프레임 시각과 같고, 10프레임마다 정답 박스 안 픽셀 색이
    `TARGET_COLORS`와 20 안에서 맞는지(압축 손실 감안) 검증한다.
    """
    path = tmp_path / "bodycam.mp4"
    scenario.write(path)
    assert read_frame_times(path) == scenario.frame_times

    with av.open(str(path)) as container:
        frames = {
            round(float(f.time or 0) * 1000): f.to_ndarray(format="rgb24")
            for f in container.decode(video=0)
        }
    for t_ms in scenario.frame_times[::10]:
        img = frames[t_ms]
        for target, box in target_boxes_at(scenario.labels, t_ms).items():
            if box.w < 8 or box.h < 8:
                continue
            x, y, w, h = int(box.x), int(box.y), int(box.w), int(box.h)
            # 모서리와 경계 압축 손실을 피해 안쪽 행 하나의 중앙값을 본다
            patch = img[y + h - 3, x + 2 : x + w - 2]
            median = np.median(patch, axis=0)
            assert np.allclose(median, TARGET_COLORS[target], atol=20), (target, t_ms, median)
