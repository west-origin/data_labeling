"""YOLOX 전·후처리 단위 테스트 (`dlp_models.yolox`, ADR 0060). 가중치 없이 합성 배열로 정답을
계산한다.

정답 근거는 YOLOX 0.3.0의 `preproc`(비율 유지, 114로 오른쪽·아래 채움, 정규화 없음),
`demo_postprocess`(격자 디코드 `(xy + grid) * stride`, `exp(wh) * stride`, stride 8·16·32),
`multiclass_nms_class_aware`(점수 = 물체 x 클래스, 클래스별 NMS)다.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from dlp_models.yolox import (
    PAD_VALUE,
    Yolox,
    YoloxDetection,
    anchor_count,
    decode,
    grid_decode,
    letterbox,
    nms,
)


def _row(cx: float, cy: float, w: float, h: float, obj: float, cls: list[float]) -> list[float]:
    """출력 한 행 (cx, cy, w, h, 물체 점수, 클래스 점수...)."""
    return [cx, cy, w, h, obj, *cls]


def test_letterbox_keeps_ratio_pads_right_bottom_with_114_and_no_normalization() -> None:
    """320x480(높이x너비) → r = 640/480, 크기 int(480r)=640 x int(320r)=426, 나머지 114.

    색 (10, 20, 30)이 그대로 float로 들어가고(정규화 없음, 채널 순서 유지), 왼쪽 위에 놓인 흰
    사각형의 입력 위치가 원래 위치 x r이다.
    """
    img = np.zeros((320, 480, 3), dtype=np.uint8)
    img[:] = (10, 20, 30)
    img[60:150, 90:210] = 255  # y 60~150, x 90~210 흰 사각형
    x, r = letterbox(img)
    assert x.shape == (1, 3, 640, 640) and x.dtype == np.float32
    assert r == pytest.approx(640 / 480)
    assert tuple(x[0, :, 5, 5]) == (10.0, 20.0, 30.0)
    assert (x[0, :, 426:, :] == PAD_VALUE).all() and (x[0, :, 425, :] != PAD_VALUE).all()
    # 쌍선형 보간 가장자리는 중간값이라 반보다 밝은 칸으로 본다 (끝 칸 + 1 = 오른쪽·아래 변)
    ys, xs = np.nonzero(x[0, 0] > 128)
    assert xs.min() == pytest.approx(90 * r, abs=1.5)
    assert xs.max() + 1 == pytest.approx(210 * r, abs=1.5)
    assert ys.min() == pytest.approx(60 * r, abs=1.5)
    assert ys.max() + 1 == pytest.approx(150 * r, abs=1.5)


def test_decode_maps_letterboxed_box_back_to_original_pixels() -> None:
    """원래 (60, 30, 120, 90) 박스를 입력 픽셀(x r)로 만든 출력 행이 원래 박스로 돌아온다.

    점수 = 물체 0.8 x 클래스 0.9 = 0.72. 채움 영역 밖으로 나간 박스는 이미지 안으로 잘린다.
    """
    r = 640 / 480
    x, y, w, h = 60.0, 30.0, 120.0, 90.0
    pred = np.array(
        [
            _row((x + w / 2) * r, (y + h / 2) * r, w * r, h * r, 0.8, [0.9, 0.1]),
            # 원래 이미지(480x320) 오른쪽 아래로 나간 박스: 400~520, 300~340 → 400~480, 300~320
            _row(460 * r, 320 * r, 120 * r, 40 * r, 1.0, [0.0, 0.7]),
            # 완전히 채움 영역 (원래 y 400~420) → 버린다
            _row(100 * r, 410 * r, 20 * r, 20 * r, 1.0, [0.9, 0.9]),
        ],
        dtype=np.float32,
    )
    dets = decode(pred, r, 480, 320, [0.5, 0.5])
    assert [d.class_index for d in dets] == [0, 1]
    assert dets[0].score == pytest.approx(0.72) and dets[0].box == pytest.approx((x, y, w, h))
    assert dets[1].box == pytest.approx((400.0, 300.0, 80.0, 20.0), abs=1e-3)


def test_class_thresholds_and_column_check() -> None:
    """클래스별 문턱: 같은 점수 0.6이 문턱 0.5 클래스는 남고 0.7 클래스는 빠진다. 열 수가 5 + C가
    아니면(정책 classes와 가중치 불일치) ValueError.
    """
    pred = np.array([_row(50, 50, 20, 20, 1.0, [0.6, 0.6])], dtype=np.float32)
    assert [d.class_index for d in decode(pred, 1.0, 640, 640, [0.5, 0.7])] == [0]
    with pytest.raises(ValueError, match="5 \\+ 클래스 3"):
        decode(pred, 1.0, 640, 640, [0.5, 0.5, 0.5])


def test_grid_decode_matches_demo_postprocess() -> None:
    """격자 디코드 정답: 640 입력은 80x80 + 40x40 + 20x20 = 8400칸.

    stride 16 격자의 (x=3, y=2) 칸(행 6400 + 2x40 + 3)에 xy 오프셋 0.5, wh log 2 → 중심
    (3.5x16, 2.5x16) = (56, 40), 크기 2x16 = 32. stride 32 마지막 칸 (19, 19)에 0 → 중심
    (608, 608), 크기 32. 점수 열은 그대로다.
    """
    assert anchor_count((640, 640)) == 8400
    raw = np.zeros((8400, 7), dtype=np.float32)
    i = 6400 + 2 * 40 + 3
    raw[i, :4] = (0.5, 0.5, math.log(2), math.log(2))
    raw[i, 4:] = (0.9, 0.1, 0.8)
    out = grid_decode(raw)
    assert out[i, :4] == pytest.approx((56.0, 40.0, 32.0, 32.0))
    assert out[i, 4:] == pytest.approx((0.9, 0.1, 0.8))
    assert out[-1, :4] == pytest.approx((608.0, 608.0, 32.0, 32.0))
    assert out[0, :4] == pytest.approx((0.0, 0.0, 8.0, 8.0))  # stride 8 첫 칸, exp(0) x 8
    assert raw[i, 0] == pytest.approx(0.5)  # 입력은 바꾸지 않는다
    with pytest.raises(ValueError, match="격자 칸 8400"):
        grid_decode(raw[:100])


def test_nms_is_per_class() -> None:
    """같은 클래스 두 박스 IoU 7/13 ≈ 0.54 > 0.45 → 점수 높은 것만. 다른 클래스는 겹쳐도 둘 다."""
    a = YoloxDetection(0, (0.0, 0.0, 10.0, 10.0), 0.9)
    b = YoloxDetection(0, (3.0, 0.0, 10.0, 10.0), 0.8)
    c = YoloxDetection(1, (3.0, 0.0, 10.0, 10.0), 0.7)
    assert nms([b, a, c], 0.45) == [a, c]
    assert nms([a, b], 0.6) == [a, b]


def test_yolox_detect_feeds_bgr_letterbox_and_decodes_raw_or_decoded_output() -> None:
    """가짜 ONNX 세션: 입력 이름 `images`, (1, 3, 640, 640), RGB → BGR (빨강이 채널 2).

    격자 상대값 출력(기본)은 격자 디코드를 거치고, decoded=True면 그대로 쓴다. 480x640 영상은
    r = 1이라 stride 32 첫 칸 (0, 0)에 오프셋 0.5, wh log 2 → 중심 (16, 16), 크기 64 →
    (0, 0, 48, 48) (왼쪽 위는 0으로 잘림).
    """
    seen: list[NDArray[np.float32]] = []
    raw = np.zeros((1, 8400, 6), dtype=np.float32)
    k = 6400 + 1600
    raw[0, k] = (0.5, 0.5, math.log(2), math.log(2), 1.0, 0.9)

    class FakeSession:
        """입력을 기록하고 정해 둔 출력을 내는 `OnnxModel` 대역."""

        def __init__(self, out: NDArray[np.float32]) -> None:
            self.out = out

        def run(self, names: list[str], feeds: dict[str, Any]) -> list[Any]:
            """출력 이름 `output`만 받는다."""
            assert names == ["output"] and list(feeds) == ["images"]
            seen.append(feeds["images"])
            return [self.out]

    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    rgb[..., 0] = 255  # 빨강
    model = Yolox.__new__(Yolox)
    model.session = FakeSession(raw)  # pyright: ignore[reportAttributeAccessIssue]
    model.num_classes, model.decoded, model.nms_iou, model.size = 1, False, 0.45, (640, 640)
    [det] = model.detect(rgb, [0.5])
    assert seen[0].shape == (1, 3, 640, 640)
    assert seen[0][0, 2, 0, 0] == 255 and seen[0][0, 0, 0, 0] == 0  # BGR
    assert det.box == pytest.approx((0.0, 0.0, 48.0, 48.0)) and det.score == pytest.approx(0.9)
    decoded = raw.copy()
    decoded[0, k, :4] = (16.0, 16.0, 64.0, 64.0)
    model.session = FakeSession(decoded)  # pyright: ignore[reportAttributeAccessIssue]
    model.decoded = True
    assert model.detect(rgb, [0.5]) == [det]
    with pytest.raises(ValueError, match="문턱 2개"):
        model.detect(rgb, [0.5, 0.5])
