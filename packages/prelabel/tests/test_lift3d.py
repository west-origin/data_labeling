from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from dlp_models.depth import Intrinsics
from dlp_prelabel.common import Image, model_label
from dlp_prelabel.lift3d import intrinsics_for, lift_tracks
from dlp_prelabel.policy import load_policy
from dlp_schema.labels import (
    BoxKeyframe,
    BoxTrackPayload,
    Hand,
    Keypoint,
    KeypointFrame,
    KeypointTrackPayload,
    LabelRecord,
)
from dlp_schema.session import CameraIntrinsics

ROOT = Path(__file__).resolve().parents[3]
NOW = datetime(2026, 10, 7, tzinfo=UTC)
W, H = 320, 240


class PlaneDepth:
    """왼쪽 절반 1 m, 오른쪽 절반 2 m인 깊이 맵을 내는 가짜 모델."""

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, image: Image) -> NDArray[np.float32]:
        self.calls += 1
        d = np.full(image.shape[:2], 1.0, dtype=np.float32)
        d[:, W // 2 :] = 2.0
        return d


def _tracks() -> list[LabelRecord]:
    points = tuple(Keypoint(x=80.0, y=120.0, visibility=2) for _ in range(21))
    hand = KeypointTrackPayload(
        entity_id="right_hand",
        skeleton="hand21",
        hand=Hand.RIGHT,
        keyframes=tuple(KeypointFrame(t_ms=t, points=points) for t in range(0, 1_000, 100)),
    )
    box = BoxTrackPayload(
        entity_id="ov_bucket_01",
        class_id="bucket",
        keyframes=(
            BoxKeyframe(t_ms=0, x=200, y=100, w=40, h=40),
            BoxKeyframe(t_ms=500, x=0, y=0, w=0, h=0, outside=True),
        ),
    )
    return [
        model_label(
            label_id=f"l{i}",
            session_id="s",
            stream_id="bodycam",
            t_start_ms=0,
            t_end_ms=900,
            ontology_version="1.0.0",
            model_version="m",
            confidence=0.9,
            payload=p,
            now=NOW,
        )
        for i, p in enumerate((hand, box))
    ]


def test_lifting_unprojects_with_depth_and_respects_stride() -> None:
    policy = load_policy(ROOT).depth.model_copy(update={"frame_stride_ms": 200})
    frames = [(t, np.zeros((H, W, 3), dtype=np.uint8)) for t in range(0, 1_000, 100)]
    depth = PlaneDepth()
    calib = CameraIntrinsics(width=W, height=H, fx=160, fy=160, cx=160, cy=120)
    out = lift_tracks(frames, depth, _tracks(), calib, policy)

    assert depth.calls == 5  # 0, 200, ..., 800
    wrist = out[("right_hand", "wrist")]
    assert [s.t_ms for s in wrist] == [0, 200, 400, 600, 800]
    assert (wrist[0].x, wrist[0].y, wrist[0].z) == pytest.approx((-0.5, 0.0, 1.0))
    assert {p for e, p in out if e == "right_hand"} == {"wrist", "thumb_tip", "index_tip"}
    [bucket] = out[("ov_bucket_01", None)]  # 화면 밖 키프레임은 건너뛴다
    assert (bucket.t_ms, bucket.x, bucket.z) == (0, pytest.approx(0.75), pytest.approx(2.0))


def test_intrinsics_scale_with_resolution_or_fall_back_to_hfov() -> None:
    calib = CameraIntrinsics(width=1920, height=1080, fx=1000, fy=1000, cx=960, cy=540)
    assert intrinsics_for(calib, 960, 540, 90) == Intrinsics(500, 500, 480, 270)
    approx = intrinsics_for(None, 640, 480, 90)
    assert approx.fx == pytest.approx(320) and (approx.cx, approx.cy) == (320, 240)
