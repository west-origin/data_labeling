"""`dlp_fixtures.hand3d` 검사: 결정성, 2D가 정답 3D의 투영인지, 합성 깊이가 손바닥에서만 손
깊이인지.
"""

from __future__ import annotations

import numpy as np
import pytest

from dlp_fixtures.hand3d import PALM_JOINTS, generate_hand3d_scenario, hand_model
from dlp_schema.labels import Hand, KeypointTrackPayload


def test_same_seed_same_scenario_and_noise_changes_only_with_seed() -> None:
    """같은 인자면 라벨이 같고, 흔들림이 있으면 seed마다 다르다."""
    assert generate_hand3d_scenario(3).labels == generate_hand3d_scenario(3).labels
    a = generate_hand3d_scenario(1, pixel_noise_px=1.0).labels
    b = generate_hand3d_scenario(2, pixel_noise_px=1.0).labels
    assert a[0].payload != b[0].payload


@pytest.mark.parametrize("hand", [Hand.RIGHT, Hand.LEFT])
def test_points_are_projections_of_the_ground_truth(hand: Hand) -> None:
    """키프레임 2D는 정답 카메라 좌표를 내부 파라미터로 투영한 값이고, 모두 화면 안·카메라
    앞이다.
    """
    s = generate_hand3d_scenario(0, hand=hand)
    [x] = s.labels
    assert isinstance(x.payload, KeypointTrackPayload) and x.payload.hand is hand
    k = s.intrinsics
    for f in x.payload.keyframes:
        cam = s.joints_cam[f.t_ms]
        assert np.all(cam[:, 2] > 0.3)
        for q, p in zip(f.points, cam, strict=True):
            assert q.x == pytest.approx(k.fx * p[0] / p[2] + k.cx)
            assert q.y == pytest.approx(k.fy * p[1] / p[2] + k.cy)
            assert 0 <= q.x < s.width and 0 <= q.y < s.height


def test_hand_model_has_adult_palm_length() -> None:
    """손목 → 중지 MCP 0.09 m (손 크기 사전값 `prelabel.yaml depth.hand_fit.reference_palm_m`과
    같다).
    """
    j = hand_model(0.3)
    assert float(np.sqrt(np.sum(np.diff(j[[0, 9]], axis=0) ** 2))) == pytest.approx(0.09)


def test_scene_depth_is_hand_depth_only_on_palm_joints() -> None:
    """합성 깊이: 손바닥 관절 픽셀은 그 관절 깊이, 손가락 끝(검지 8)은 배경 1.2 m."""
    s = generate_hand3d_scenario(0)
    t, img = next(iter(s.frames()))
    d = s.depth.predict(img)
    [x] = s.labels
    assert isinstance(x.payload, KeypointTrackPayload)
    f = x.payload.keyframes[0]
    for j in PALM_JOINTS:
        u, v = round(f.points[j].x), round(f.points[j].y)
        assert d[v, u] == pytest.approx(s.joints_cam[t][j, 2], rel=1e-6)
    tip = f.points[8]
    assert d[round(tip.y), round(tip.x)] == pytest.approx(1.2)
