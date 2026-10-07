from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from dlp_models.depth import Intrinsics, MetricDepth, input_size, sample_depth
from dlp_models.owlv2 import OwlDetection, Owlv2, decode, nms, preprocess
from dlp_models.registry import load_registry, resolve
from dlp_schema.predictor import ModelUnavailableError

ROOT = Path(__file__).resolve().parents[3]


def _root_with_registry(tmp_path: Path) -> Path:
    (tmp_path / "config").mkdir()
    shutil.copy(ROOT / "config/models.yaml", tmp_path / "config/models.yaml")
    return tmp_path


def test_registry_entries_have_source_and_license() -> None:
    for name, spec in load_registry(ROOT).models.items():
        assert (spec.url is None) != (spec.export is None), name  # 받거나 변환하거나 둘 중 하나
        assert spec.path.startswith("data/models/") and spec.license, name
        if spec.export:
            assert (ROOT / spec.export).is_file(), name


def test_coco_trained_models_are_review() -> None:
    """감사 회귀 (4차): COCO 이미지는 Flickr 개별 CC 라이선스(CC BY-NC 계열 포함)라 직접 학습
    데이터에 비상업 조건이 있다 → review. COCO 표기도 한 가지로 통일한다."""
    registry = load_registry(ROOT)
    coco = {
        name: spec
        for name, spec in registry.models.items()
        if any("COCO" in d for d in spec.training_data)
    }
    assert {"object_detector", "yolox_m_coco", "rtmpose_m_body7"} <= set(coco)
    for name, spec in coco.items():
        assert spec.commercial != "allowed", name
        [entry] = [d for d in spec.training_data if "COCO" in d]
        assert "CC BY-NC" in entry and "Flickr" in entry, name


def test_resolve_reports_missing_and_tampered_weights(tmp_path: Path) -> None:
    root = _root_with_registry(tmp_path)
    with pytest.raises(ModelUnavailableError, match="make models"):
        resolve(root, "yunet")
    with pytest.raises(ModelUnavailableError, match="make export-models"):
        resolve(root, "depth_metric_indoor_small")
    with pytest.raises(ModelUnavailableError, match="목록에"):
        resolve(root, "nope")
    path = root / load_registry(root).models["yunet"].path
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not a model")
    with pytest.raises(ModelUnavailableError, match="해시"):
        resolve(root, "yunet")


def test_owl_preprocess_pads_to_square_and_decode_maps_back_to_pixels() -> None:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    pixels, side = preprocess(image)
    assert pixels.shape == (1, 3, 960, 960) and side == 640
    # 아래쪽 채움 영역(회색 0.5)은 정규화 후 0.5 근처 값이 된다
    assert pixels[0, 0, -1, 0] == pytest.approx((0.5 - 0.48145466) / 0.26862954, abs=1e-3)
    logits = np.array([[3.0, -3.0], [-5.0, -5.0]], dtype=np.float32)
    boxes = np.array([[0.5, 0.5, 0.25, 0.25], [0.1, 0.1, 0.1, 0.1]], dtype=np.float32)
    [det] = decode(logits, boxes, side, [0.5, 0.5], 640, 480)
    assert det.query_index == 0 and det.score == pytest.approx(0.9526, abs=1e-3)
    assert det.box == pytest.approx((240.0, 240.0, 160.0, 160.0))


def test_owl_nms_threshold_is_a_parameter() -> None:
    a = OwlDetection(0, (0.0, 0.0, 10.0, 10.0), 0.9)
    b = OwlDetection(0, (3.0, 0.0, 10.0, 10.0), 0.8)  # IoU 7/13 ≈ 0.54
    assert nms([a, b], 0.5) == [a]
    assert nms([a, b], 0.6) == [a, b]


def test_depth_input_keeps_aspect_ratio_like_the_reference() -> None:
    # 공식 Resize(518, keep_aspect_ratio, ensure_multiple_of=14, lower_bound)
    assert input_size(240, 320) == (518, 686)
    assert input_size(1080, 1920) == (518, 924)
    assert input_size(518, 518) == (518, 518)
    assert input_size(1920, 1080) == (924, 518)
    for h, w in ((240, 320), (720, 1280), (1080, 1440), (333, 777)):
        ih, iw = input_size(h, w)
        assert ih % 14 == 0 and iw % 14 == 0 and min(ih, iw) >= 518
        assert abs(iw / ih - w / h) < 0.03  # 비율 유지 (14 배수 맞춤 오차만)


def test_depth_predict_feeds_aspect_preserving_input_and_restores_size() -> None:
    seen: list[tuple[int, ...]] = []

    class FakeSession:
        def run(self, names: list[str], feeds: dict[str, Any]) -> list[Any]:
            x = feeds["pixel_values"]
            seen.append(x.shape)
            return [np.ones((1, x.shape[2], x.shape[3]), dtype=np.float32)]

    model = MetricDepth.__new__(MetricDepth)
    model.session = FakeSession()  # pyright: ignore[reportAttributeAccessIssue]
    depth = model.predict(np.zeros((240, 320, 3), dtype=np.uint8))
    assert seen == [(1, 3, 518, 686)] and depth.shape == (240, 320)


def test_unproject_and_depth_patch_median() -> None:
    intr = Intrinsics.from_hfov(640, 480, 90)
    assert intr.fx == pytest.approx(320)
    assert intr.unproject(640, 240, 2.0) == pytest.approx((2.0, 0.0, 2.0))
    depth = np.full((10, 10), 1.0, dtype=np.float32)
    depth[5, 5] = 100.0  # 튀는 값 하나는 중앙값이 무시한다
    assert sample_depth(depth, 5, 5, radius=1) == 1.0
    assert np.isnan(sample_depth(depth, -1, 5))


@pytest.mark.skipif(
    not (ROOT / "data/models/owlv2/model_quantized.onnx").is_file(), reason="make models"
)
def test_owlv2_runs_on_cpu() -> None:
    model_path, _ = resolve(ROOT, "owlv2_onnx")
    tokenizer, _ = resolve(ROOT, "owlv2_tokenizer")
    owl = Owlv2(model_path, tokenizer, ["a mirror", "a printed document"])
    image = np.full((240, 320, 3), 255, dtype=np.uint8)
    for d in owl.detect(image, [0.1, 0.1]):
        x, y, w, h = d.box
        assert x >= 0 and y >= 0 and x + w <= 320 and y + h <= 240


@pytest.mark.skipif(
    not (ROOT / "data/models/depth_anything_v2_metric_indoor_small.onnx").is_file(),
    reason="make export-models",
)
def test_metric_depth_runs_on_cpu() -> None:
    path, _ = resolve(ROOT, "depth_metric_indoor_small")
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image[:, :, 1] = np.linspace(0, 255, 320, dtype=np.uint8)[None, :]
    depth = MetricDepth(path).predict(image)
    assert depth.shape == (240, 320) and np.isfinite(depth).all() and (depth >= 0).all()
