"""프리라벨 어댑터 테스트: 손 판정 뒤집기, 손별 최고 점수, 정책·레지스트리·온톨로지 일치, stub,
가중치 없음 보고, 실제 모델 CPU 실행(가중치·libEGL이 있을 때만).

실제 모델 테스트는 합성 영상(`dlp_fixtures.video`)에 실제 손·사람·물체가 없으므로 개수는 보지
않고 돌고 라벨 형식이 계약·온톨로지 검사(`check_label`)를 통과하는지만 본다.
"""

from __future__ import annotations

import ctypes.util
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest

from dlp_fixtures.actions import generate_action_scenario
from dlp_fixtures.video import generate_blur_scenario
from dlp_models.registry import load_registry
from dlp_prelabel.adapters.mediapipe_models import (
    MediaPipeHands,
    MediaPipeObjects,
    best_per_hand,
    flip_hand,
)
from dlp_prelabel.adapters.owl_objects import OwlObjects
from dlp_prelabel.adapters.rtmpose import RtmPose
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
    for n in ("hand_landmarker.task", "efficientdet_lite0.tflite")
)
HAS_RTMPOSE = (ROOT / "data/models/rtmpose/rtmpose_m_body7_256x192.onnx").is_file()
HAS_OWL = (ROOT / "data/models/owlv2/model_quantized.onnx").is_file()


def test_handedness_is_flipped_for_non_mirrored_bodycam() -> None:
    """MediaPipe는 거울상 입력을 가정하므로 바디캠(거울상 아님)에서는 Left/Right를 뒤집는지 본다."""
    assert flip_hand("Left", input_is_mirrored=False) is Hand.RIGHT
    assert flip_hand("Right", input_is_mirrored=False) is Hand.LEFT
    assert flip_hand("Left", input_is_mirrored=True) is Hand.LEFT


def test_two_detections_of_the_same_hand_keep_the_highest_score() -> None:
    # 감사 회귀: num_hands=2에서 두 탐지가 모두 같은 손이면 같은 시각 키프레임이 둘 생겼다
    """감사 회귀: num_hands=2에서 두 탐지가 같은 손이면 점수가 높은 것 하나만 남는지 본다 (같은
    시각 키프레임이 두 개 생기던 문제).
    """
    found = best_per_hand([(Hand.LEFT, 0.6, "a"), (Hand.LEFT, 0.9, "b"), (Hand.RIGHT, 0.7, "c")])
    assert found == {Hand.LEFT: (0.9, "b"), Hand.RIGHT: (0.7, "c")}
    assert best_per_hand([(Hand.RIGHT, 0.8, "x"), (Hand.RIGHT, 0.5, "y")]) == {
        Hand.RIGHT: (0.8, "x")
    }


def test_coco_mapping_targets_exist_in_ontology() -> None:
    """`objects.coco_to_ontology`의 모든 대상 클래스가 온톨로지 v1에 있는지 본다."""
    ontology = load_ontology(ROOT / "config/ontology/v1")
    for coco, cls in load_policy(ROOT).objects.coco_to_ontology.items():
        assert cls in ontology.objects, coco


def test_oracle_predictors_emit_valid_model_labels() -> None:
    """Oracle stub이 정답 키포인트 트랙을 모델 출처(버전 `oracle-hands-j1.0`, 신뢰도 0.9)로 내고
    그 라벨이 온톨로지 검사를 통과하는지 본다. 정답 근거: 행동 시나리오 픽스처의 손 트랙 1개.
    """
    ontology = load_ontology(ROOT / "config/ontology/v1")
    truth = generate_action_scenario(1).labels
    oracle: Predictor = OraclePredictor("hands", truth, ("keypoint_track",), jitter_px=1.0, now=NOW)
    labels = oracle.run(Clip("s1", "bodycam", Path("unused.mp4")))
    assert len(labels) == 1
    [label] = labels
    assert label.provenance.model_version == "oracle-hands-j1.0" and label.confidence == 0.9
    assert check_label(label, ontology) == []


def test_unavailable_features_are_explicit() -> None:
    """미연동 기능 stub 세 개(도구 부위 마스크, 카메라 자세, 학습 접촉 분류기)가 이유를 갖고 빈
    결과를 내는지 본다 (TODO(real-model) 자리가 조용히 사라지지 않게).
    """
    names = {p.name for p in UNAVAILABLE}
    assert names == {"tool_part_masks", "camera_pose", "learned_contact"}
    assert all(p.run(Clip("s", "bodycam", Path("x.mp4"))) == [] and p.reason for p in UNAVAILABLE)


def test_missing_model_file_is_reported(tmp_path: Path) -> None:
    """가중치가 없는 저장소 루트에서 실제 어댑터 생성자가 `make models` 안내와 함께
    ModelUnavailableError를 내는지 본다.
    """
    (tmp_path / "config").mkdir()
    shutil.copy(ROOT / "config/models.yaml", tmp_path / "config/models.yaml")
    for cls in (MediaPipeHands, RtmPose, OwlObjects):
        with pytest.raises(ModelUnavailableError, match="make models"):
            cls(tmp_path, load_policy(ROOT), ontology_version="1.0.0", now=NOW)


def test_policy_models_and_queries_exist() -> None:
    """정책 `models`의 모든 이름이 `config/models.yaml`에 있고, OWLv2 질의의 대상 클래스가
    온톨로지에 있는지 본다.
    """
    ontology = load_ontology(ROOT / "config/ontology/v1")
    policy = load_policy(ROOT)
    registry = load_registry(ROOT)
    assert all(name in registry.models for name in policy.models.model_dump().values())
    for query, cls in policy.open_vocab_objects.queries.items():
        assert cls in ontology.objects, query


@pytest.mark.skipif(
    not (HAS_EGL and HAS_MODELS), reason="MediaPipe 모델 또는 libEGL 없음 (make models)"
)
def test_mediapipe_adapters_run_on_cpu(tmp_path: Path) -> None:
    """합성 영상에는 실제 손·사람·물체가 없으므로 개수는 보지 않고, 돌고 형식이 맞는지만 본다.

    가중치와 libEGL이 있을 때만 돈다. 버전 접두사 `mediapipe-`도 본다.
    """
    video = tmp_path / "v.mp4"
    generate_blur_scenario(1, duration_ms=1_000).write(video)
    ontology = load_ontology(ROOT / "config/ontology/v1")
    policy = load_policy(ROOT)
    for cls in (MediaPipeHands, MediaPipeObjects):
        predictor = cls(ROOT, policy, ontology_version="1.0.0", now=NOW)
        assert predictor.version.startswith("mediapipe-")
        for label in predictor.run(Clip("s1", "bodycam", video)):
            assert check_label(label, ontology) == []


@pytest.mark.skipif(not HAS_RTMPOSE, reason="RTMPose 가중치 없음 (make models)")
def test_rtmpose_runs_on_cpu(tmp_path: Path) -> None:
    """RTMPose 가중치가 있을 때 합성 영상에서 돌고 라벨 형식이 맞는지 본다."""
    video = tmp_path / "v.mp4"
    generate_blur_scenario(1, duration_ms=500).write(video)
    ontology = load_ontology(ROOT / "config/ontology/v1")
    predictor = RtmPose(ROOT, load_policy(ROOT), ontology_version="1.0.0", now=NOW)
    assert predictor.version.startswith("rtmpose-")
    for label in predictor.run(Clip("s1", "bodycam", video)):
        assert check_label(label, ontology) == []


@pytest.mark.skipif(not HAS_OWL, reason="OWLv2 가중치 없음 (make models)")
def test_owl_objects_runs_once_per_stride(tmp_path: Path) -> None:
    """CPU에서 프레임당 수 초라 짧은 영상 한 프레임만 돌린다 (stride보다 짧은 영상).

    300 ms 영상은 frame_stride_ms(500)보다 짧아 첫 프레임 한 번만 추론하므로 모든 박스 트랙의
    키프레임이 하나여야 한다.
    """
    video = tmp_path / "v.mp4"
    generate_blur_scenario(1, duration_ms=300).write(video)
    ontology = load_ontology(ROOT / "config/ontology/v1")
    predictor = OwlObjects(ROOT, load_policy(ROOT), ontology_version="1.0.0", now=NOW)
    for label in predictor.run(Clip("s1", "bodycam", video)):
        assert check_label(label, ontology) == []
        assert label.payload.kind == "box_track" and len(label.payload.keyframes) == 1
