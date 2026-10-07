"""dlp_models 단위 테스트: 레지스트리(목록·라이선스·해시), OWLv2·메트릭 깊이 전후처리.

DB·서비스가 필요 없다. 실제 가중치가 필요한 두 테스트(`test_owlv2_runs_on_cpu`,
`test_metric_depth_runs_on_cpu`)는 `make models`·`make export-models`로 받은 파일이 없으면
건너뛴다. 나머지는 가짜 세션·합성 배열로 정답을 손으로 계산할 수 있는 입력을 쓴다. 관련: WP8, ADR
0009, 0010.
"""

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
    """`config/models.yaml`만 있는 빈 저장소 루트를 tmp_path에 만든다 (가중치 파일 없음)."""
    (tmp_path / "config").mkdir()
    shutil.copy(ROOT / "config/models.yaml", tmp_path / "config/models.yaml")
    return tmp_path


def test_registry_entries_have_source_and_license() -> None:
    """모든 로컬 모델 항목이 받는 곳(url)과 변환 스크립트(export) 중 정확히 하나를 갖고,
    `data/models/` 아래 경로와 라이선스를 적었는지 본다. export 스크립트 파일이 실제로
    있는지도 본다.
    """
    for name, spec in load_registry(ROOT).models.items():
        assert (spec.url is None) != (spec.export is None), name  # 받거나 변환하거나 둘 중 하나
        assert spec.path.startswith("data/models/") and spec.license, name
        if spec.export:
            assert (ROOT / spec.export).is_file(), name


def test_coco_trained_models_are_review() -> None:
    """감사 회귀 (4차): COCO 이미지는 Flickr 개별 CC 라이선스(CC BY-NC 계열 포함)라 직접 학습
    데이터에 비상업 조건이 있다 → review. COCO 표기도 한 가지로 통일한다.

    정답 근거: ADR 0010 분류 기준. COCO로 학습한 세 모델이 `allowed`가 아니고 같은 표기를 쓴다.
    """
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
    """`resolve`가 세 실패를 `ModelUnavailableError`로 알리는지 본다.

    시나리오: 가중치 없음(직접 받는 모델은 `make models`, 변환형은 `make export-models` 안내),
    목록에 없는 이름, 내용이 바뀐 파일(해시 불일치).
    """
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
    """OWLv2 전처리가 640x480을 640 정사각형으로 채우고 960 입력을 만들며, 정규화 박스를 원래 픽셀로
    되돌리는지 본다.

    정답 근거: 채움 영역은 회색 0.5라 정규화 값이 (0.5-평균)/표준편차. logit 3 → 시그모이드
    0.9526. 중심 (0.5, 0.5) 크기 0.25 박스 x 640 = 중심 320, 크기 160 → (240, 240, 160, 160).
    """
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
    """NMS IoU 문턱이 인자로 동작하는지 본다. 두 박스 IoU는 7/13 ≈ 0.54라 0.5 문턱에서는 하나,
    0.6 문턱에서는 둘 다 남는다.
    """
    a = OwlDetection(0, (0.0, 0.0, 10.0, 10.0), 0.9)
    b = OwlDetection(0, (3.0, 0.0, 10.0, 10.0), 0.8)  # IoU 7/13 ≈ 0.54
    assert nms([a, b], 0.5) == [a]
    assert nms([a, b], 0.6) == [a, b]


def test_depth_input_keeps_aspect_ratio_like_the_reference() -> None:
    """깊이 입력 크기가 공식 Resize(518, 비율 유지, 14 배수, lower_bound)와 같은지 본다.

    정답 근거: 공식 구현으로 계산한 값 (240x320 → 518x686 등). 여러 해상도에서 14 배수, 짧은 변
    518 이상, 비율 오차가 14 배수 맞춤 정도인지도 본다.
    """
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
    """가짜 ONNX 세션으로 `MetricDepth.predict`가 비율을 지킨 입력(1, 3, 518, 686)을 넣고 출력
    깊이 맵을 원래 크기(240, 320)로 되돌리는지 본다 (실제 가중치 불필요).
    """
    seen: list[tuple[int, ...]] = []

    class FakeSession:
        """입력 shape를 기록하고 입력과 같은 공간 크기의 1 m 깊이를 돌려주는 가짜 세션."""

        def run(self, names: list[str], feeds: dict[str, Any]) -> list[Any]:
            """`OnnxModel.run`과 같은 모양: 입력 shape를 `seen`에 남기고 (1, H', W') 깊이 1.0을
            낸다.
            """
            x = feeds["pixel_values"]
            seen.append(x.shape)
            return [np.ones((1, x.shape[2], x.shape[3]), dtype=np.float32)]

    model = MetricDepth.__new__(MetricDepth)
    model.session = FakeSession()  # pyright: ignore[reportAttributeAccessIssue]
    depth = model.predict(np.zeros((240, 320, 3), dtype=np.uint8))
    assert seen == [(1, 3, 518, 686)] and depth.shape == (240, 320)


def test_unproject_and_depth_patch_median() -> None:
    """화각 근사 내부 파라미터·역투영·패치 중앙값을 본다.

    정답 근거: 640 폭, 화각 90도면 fx = 320. 오른쪽 끝 픽셀(u=640)을 깊이 2 m로 올리면 x =
    (640-320)*2/320 = 2. 튀는 값 하나(100 m)는 3x3 중앙값이 무시한다. 화면 밖은 nan.
    """
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
    """실제 OWLv2 가중치로 CPU 추론이 돌고 박스가 이미지 안에 있는지 본다 (가중치가 있을 때만)."""
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
    """실제 메트릭 깊이 가중치로 CPU 추론이 돌고 원래 크기의 유한한 양수 깊이를 내는지 본다
    (변환한 가중치가 있을 때만). 합성 그라디언트 이미지라 깊이 값 자체는 보지 않는다.
    """
    path, _ = resolve(ROOT, "depth_metric_indoor_small")
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image[:, :, 1] = np.linspace(0, 255, 320, dtype=np.uint8)[None, :]
    depth = MetricDepth(path).predict(image)
    assert depth.shape == (240, 320) and np.isfinite(depth).all() and (depth >= 0).all()
