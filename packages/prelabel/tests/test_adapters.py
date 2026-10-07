from __future__ import annotations

import ctypes.util
from datetime import UTC, datetime
from pathlib import Path

import pytest

from dlp_fixtures.actions import generate_action_scenario
from dlp_fixtures.video import generate_blur_scenario
from dlp_prelabel.adapters.mediapipe_models import (
    MediaPipeHands,
    MediaPipeObjects,
    MediaPipePose,
    flip_hand,
)
from dlp_prelabel.adapters.stubs import UNAVAILABLE, OraclePredictor
from dlp_prelabel.policy import load_policy
from dlp_schema.labels import Hand
from dlp_schema.ontology import load_ontology
from dlp_schema.predictor import Clip, ModelUnavailableError, Predictor
from dlp_schema.validation import check_label

ROOT = Path(__file__).resolve().parents[3]
NOW = datetime(2026, 10, 7, tzinfo=UTC)
HAS_EGL = ctypes.util.find_library("EGL") is not None
HAS_MODELS = all(
    (ROOT / f"data/models/{n}").is_file()
    for n in ("hand_landmarker.task", "pose_landmarker_lite.task", "efficientdet_lite0.tflite")
)


def test_handedness_is_flipped_for_non_mirrored_bodycam() -> None:
    assert flip_hand("Left", input_is_mirrored=False) is Hand.RIGHT
    assert flip_hand("Right", input_is_mirrored=False) is Hand.LEFT
    assert flip_hand("Left", input_is_mirrored=True) is Hand.LEFT


def test_coco_mapping_targets_exist_in_ontology() -> None:
    ontology = load_ontology(ROOT / "config/ontology/v1")
    for coco, cls in load_policy(ROOT).objects.coco_to_ontology.items():
        assert cls in ontology.objects, coco


def test_oracle_predictors_emit_valid_model_labels() -> None:
    ontology = load_ontology(ROOT / "config/ontology/v1")
    truth = generate_action_scenario(1).labels
    oracle: Predictor = OraclePredictor("hands", truth, ("keypoint_track",), jitter_px=1.0, now=NOW)
    labels = oracle.run(Clip("s1", "bodycam", Path("unused.mp4")))
    assert len(labels) == 1
    [label] = labels
    assert label.provenance.model_version == "oracle-hands-j1.0" and label.confidence == 0.9
    assert check_label(label, ontology) == []


def test_unavailable_features_are_explicit() -> None:
    names = {p.name for p in UNAVAILABLE}
    assert names == {"tool_part_masks", "camera_pose", "mono_depth", "learned_contact"}
    assert all(p.run(Clip("s", "bodycam", Path("x.mp4"))) == [] and p.reason for p in UNAVAILABLE)


def test_missing_model_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ModelUnavailableError, match="make models"):
        MediaPipeHands(tmp_path, load_policy(ROOT), ontology_version="1.0.0", now=NOW)


@pytest.mark.skipif(
    not (HAS_EGL and HAS_MODELS), reason="MediaPipe 모델 또는 libEGL 없음 (make models)"
)
def test_mediapipe_adapters_run_on_cpu(tmp_path: Path) -> None:
    """합성 영상에는 실제 손·사람·물체가 없으므로 개수는 보지 않고, 돌고 형식이 맞는지만 본다."""
    video = tmp_path / "v.mp4"
    generate_blur_scenario(1, duration_ms=1_000).write(video)
    ontology = load_ontology(ROOT / "config/ontology/v1")
    policy = load_policy(ROOT)
    for cls in (MediaPipeHands, MediaPipePose, MediaPipeObjects):
        predictor = cls(ROOT, policy, ontology_version="1.0.0", now=NOW)
        assert predictor.version.startswith("mediapipe-")
        for label in predictor.run(Clip("s1", "bodycam", video)):
            assert check_label(label, ontology) == []
