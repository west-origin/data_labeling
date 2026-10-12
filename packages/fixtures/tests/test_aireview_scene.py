"""AI 1차 검수 벤치용 합성 장면 테스트 (WP17.3).

정답 근거: 행동 시나리오의 라벨 수·시각, 기준 크기(320x240) 대비 배율, 결정적 픽셀.
"""

from __future__ import annotations

import numpy as np

from dlp_fixtures.actions import ENTITIES, generate_action_scenario
from dlp_fixtures.aireview import THIRD_PERSON, aireview_scene, scene_frame, scene_times
from dlp_schema.labels import (
    BoxTrackPayload,
    Evidence,
    HandStatePayload,
    KeypointTrackPayload,
    Source,
    VerificationState,
)
from dlp_schema.session import StreamKind


def test_scene_labels_are_unreviewed_model_labels_scaled_to_frame() -> None:
    """모든 라벨은 검수 전 모델 출처, 손 상태는 INFERRED, 박스·키포인트는 영상 크기 배율."""
    base = generate_action_scenario(1, session_id="b1", n_units=4)
    scene = aireview_scene(1, session_id="b1", n_units=4, width=640, height=480)
    assert len(scene.labels) == len(base.labels) + len(ENTITIES) + 1
    assert all(x.provenance.source is Source.MODEL for x in scene.labels)
    assert all(x.verification.state is VerificationState.UNREVIEWED for x in scene.labels)
    states = [x for x in scene.labels if isinstance(x.payload, HandStatePayload)]
    assert states and all(x.evidence is Evidence.INFERRED for x in states)
    box = next(x.payload for x in scene.labels if isinstance(x.payload, BoxTrackPayload))
    assert (box.keyframes[0].w, box.keyframes[0].h) == (60.0, 60.0)
    cup = ENTITIES[0][3]
    assert (box.keyframes[0].x, box.keyframes[0].y) == (cup[0] * 2 - 30, cup[1] * 2 - 30)
    hand = next(x.payload for x in scene.labels if isinstance(x.payload, KeypointTrackPayload))
    raw = next(x.payload for x in base.labels if isinstance(x.payload, KeypointTrackPayload))
    assert hand.keyframes[0].points[0].x == raw.keyframes[0].points[0].x * 2
    assert scene.session.duration_ms == base.duration_ms
    kinds = {s.stream_id: s.kind for s in scene.session.streams}
    assert kinds[THIRD_PERSON] is StreamKind.THIRD_PERSON
    assert scene_times(scene) == tuple(base.frame_times)


def test_scene_frames_are_deterministic_and_change_over_time() -> None:
    """같은 입력이면 같은 픽셀, 시각·스트림이 다르면 다른 픽셀. 크기는 장면 크기."""
    scene = aireview_scene(0, n_units=2, width=320, height=180)
    a = scene_frame(scene, "bodycam", 1000)
    assert a.shape == (180, 320, 3) and a.dtype == np.uint8
    assert np.array_equal(a, scene_frame(scene, "bodycam", 1000))
    assert not np.array_equal(a, scene_frame(scene, "bodycam", 1500))
    assert not np.array_equal(a, scene_frame(scene, THIRD_PERSON, 1000))
