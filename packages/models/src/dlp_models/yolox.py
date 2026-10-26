"""YOLOX 고정 클래스 탐지 (ONNX Runtime, CPU). 공개 데이터로 미세조정한 도구·가사 객체 탐지기
(ADR 0060).

전·후처리는 YOLOX 0.3.0(Megvii-BaseDetection/YOLOX, 태그 0.3.0)과 같다.
- 전처리 (`yolox/data/data_augment.py preproc`): BGR uint8 이미지를 비율 r = min(640/H, 640/W)로
  쌍선형 축소·확대하고(크기는 `int(W*r)`, `int(H*r)`로 버림), 114로 채운 640x640 캔버스의 왼쪽
  위에 놓는다 (오른쪽·아래 채움). HWC → CHW float32이고 0~255 그대로다 (평균·표준편차 정규화 없음,
  0.2.0부터 legacy 정규화가 빠졌다).
- 모델 출력: (1, N, 5 + C). N = 640/8^2 + 640/16^2 + 640/32^2 = 8400 (격자 순서 stride 8 → 16 → 32,
  각 격자 안은 행 우선). 열은 (cx, cy, w, h, 물체 점수, 클래스 점수 C개)이고 물체·클래스 점수는
  추론 모드에서 이미 시그모이드를 지났다 (`yolox/models/yolo_head.py`).
  - `tools/export_onnx.py`는 `--decode_in_inference`를 주지 않으면(기본) 디코드하지 않은 격자
    상대값을 낸다: cx, cy는 격자 칸 안 오프셋, w, h는 log 크기. `grid_decode`가
    `demo_postprocess`와 같이 `(xy + grid) * stride`, `exp(wh) * stride`로 바꾼다.
    `--decode_in_inference`로 내보낸 모델은 이미 640 입력 픽셀 단위다.
- 후처리 (`yolox/utils/demo_utils.py`): 점수 = 물체 점수 x 클래스 점수, 박스를 r로 나눠 원래 픽셀로
  되돌리고, 클래스별(class-aware) NMS를 한다. 공식 데모와 다른 점: 문턱은 이상(>=)으로 비교하고
  클래스마다 문턱을 따로 줄 수 있다. IoU는 연속 좌표(넓이에 +1 없음)로 계산하고, 박스를 원래 이미지
  안으로 자른다 (채움 영역으로 나간 부분).

입력 이름 `images`, 출력 이름 `output`은 `export_onnx.py` 기본값이다.

파이프라인 위치: `dlp prelabel run`의 고정 클래스 도구 박스(`dlp_prelabel.adapters.yolox_tools`,
정책 `prelabel.yaml closed_vocab_tools`). 클래스 이름은 가중치 옆 파일이 아니라 정책 `classes`
목록(출력 열 순서)으로 받는다.

좌표계: 반환하는 `YoloxDetection.box`는 원래 이미지 픽셀의 (x, y, w, h)이고 왼쪽 위가 원점이다.

공개 항목:
- `YoloxDetection`: 탐지 하나 (클래스 번호, 픽셀 박스, 점수).
- `letterbox`, `grid_decode`, `decode`, `nms`: 모델 없이 시험할 수 있는 전·후처리 (순수 numpy·cv2).
- `Yolox`: ONNX 세션을 열어 두고 이미지마다 `detect`를 부르는 추론기.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from dlp_models.onnx import OnnxModel

INPUT_SIZE = 640  # 기본 입력 한 변(px). YOLOX exp test_size (640, 640)
PAD_VALUE = 114  # letterbox 채움 값 (YOLOX preproc)
STRIDES = (8, 16, 32)  # P3~P5 격자 간격 (p6 없음)
INPUT_NAME = "images"  # export_onnx.py --input 기본값
OUTPUT_NAME = "output"  # export_onnx.py --output 기본값
NMS_IOU = 0.45  # 기본값 (공식 ONNX 데모). 호출하는 단계는 정책 값을 넘긴다

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class YoloxDetection:
    """YOLOX 탐지 하나.

    class_index: 출력 클래스 열 번호 (0부터, 정책 `classes` 순서).
    box: 원래 이미지 픽셀 좌표 (x, y, w, h), 이미지 안으로 잘린 값.
    score: 물체 점수 x 클래스 점수 (0~1).
    """

    class_index: int
    box: Box
    score: float


def letterbox(
    bgr: NDArray[np.uint8], size: tuple[int, int] = (INPUT_SIZE, INPUT_SIZE)
) -> tuple[NDArray[np.float32], float]:
    """YOLOX `preproc`와 같은 입력 (1, 3, H, W) float32와 비율 r.

    Args:
        bgr: (H, W, 3) BGR uint8 (cv2.imread와 같은 채널 순서).
        size: 모델 입력 (높이, 너비) px.

    Returns:
        (모델 입력, r). 원래 픽셀 = 입력 픽셀 / r. 채움은 오른쪽·아래이고 값은 114다.
    """
    h, w = bgr.shape[:2]
    r = min(size[0] / h, size[1] / w)
    nw, nh = int(w * r), int(h * r)
    canvas = np.full((size[0], size[1], 3), PAD_VALUE, dtype=np.uint8)
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
    canvas[:nh, :nw] = resized
    chw = np.ascontiguousarray(canvas.transpose(2, 0, 1), dtype=np.float32)
    return chw[None], r


def anchor_count(size: tuple[int, int], strides: Sequence[int] = STRIDES) -> int:
    """입력 (높이, 너비)에서 격자 칸 수 합 N (640x640이면 8400)."""
    return sum((size[0] // s) * (size[1] // s) for s in strides)


def grid_decode(
    raw: NDArray[np.float32],
    size: tuple[int, int] = (INPUT_SIZE, INPUT_SIZE),
    strides: Sequence[int] = STRIDES,
) -> NDArray[np.float32]:
    """디코드하지 않은 출력 (N, 5 + C)을 입력 픽셀 (cx, cy, w, h)로 바꾼 사본 (`demo_postprocess`).

    격자 순서는 stride 순, 각 격자 안은 행 우선(y 바깥, x 안)이다. 점수 열은 그대로 둔다.

    Raises:
        ValueError: 행 수가 `anchor_count(size, strides)`와 다를 때 (입력 크기·stride가 맞지 않음).
    """
    n = anchor_count(size, strides)
    if raw.shape[0] != n:
        raise ValueError(f"YOLOX 출력 행 {raw.shape[0]}개가 입력 {size} 격자 칸 {n}개와 다릅니다")
    grids: list[NDArray[np.float32]] = []
    scale: list[NDArray[np.float32]] = []
    for s in strides:
        hs, ws = size[0] // s, size[1] // s
        xv, yv = np.meshgrid(np.arange(ws), np.arange(hs))
        grids.append(np.stack((xv, yv), 2).reshape(-1, 2).astype(np.float32))
        scale.append(np.full((hs * ws, 1), s, dtype=np.float32))
    grid, stride = np.concatenate(grids), np.concatenate(scale)
    out = raw.astype(np.float32, copy=True)
    out[:, :2] = (out[:, :2] + grid) * stride
    out[:, 2:4] = np.exp(out[:, 2:4]) * stride
    return out


def decode(
    pred: NDArray[np.float32],
    ratio: float,
    width: int,
    height: int,
    thresholds: Sequence[float],
    nms_iou: float = NMS_IOU,
) -> list[YoloxDetection]:
    """디코드된 출력 (N, 5 + C) → 클래스별 문턱을 넘는 원래 픽셀 탐지 (클래스별 NMS 뒤).

    Args:
        pred: 배치 차원을 뺀 (N, 5 + C). 박스는 입력 픽셀 (cx, cy, w, h).
        ratio: `letterbox`의 r. 입력 픽셀을 r로 나눠 원래 픽셀로 되돌린다.
        width, height: 원래 이미지 크기(px). 박스를 이 안으로 자른다.
        thresholds: 클래스별 점수 문턱 (길이 C). 물체 x 클래스 점수가 이상이면 남긴다.
        nms_iou: 같은 클래스 박스끼리의 NMS IoU 문턱.

    Returns:
        점수 내림차순 탐지 목록. 자른 뒤 넓이가 0인 박스(채움 영역)는 버린다.

    Raises:
        ValueError: 열 수가 5 + len(thresholds)가 아닐 때 (정책 classes와 가중치가 맞지 않음).
    """
    c = len(thresholds)
    if pred.ndim != 2 or pred.shape[1] != 5 + c:
        raise ValueError(f"YOLOX 출력 열 {pred.shape[-1]}개가 5 + 클래스 {c}개와 다릅니다")
    scores = pred[:, 4:5] * pred[:, 5:]
    cx, cy, bw, bh = (pred[:, i] / ratio for i in range(4))
    x1 = np.clip(cx - bw / 2, 0.0, float(width))
    y1 = np.clip(cy - bh / 2, 0.0, float(height))
    x2 = np.clip(cx + bw / 2, 0.0, float(width))
    y2 = np.clip(cy + bh / 2, 0.0, float(height))
    out: list[YoloxDetection] = []
    for k, thr in enumerate(thresholds):
        idx = np.flatnonzero((scores[:, k] >= thr) & (x2 > x1) & (y2 > y1))
        out += [
            YoloxDetection(
                k,
                (float(x1[i]), float(y1[i]), float(x2[i] - x1[i]), float(y2[i] - y1[i])),
                float(scores[i, k]),
            )
            for i in idx
        ]
    return nms(out, nms_iou)


def nms(dets: list[YoloxDetection], iou_thr: float) -> list[YoloxDetection]:
    """클래스별 탐욕 NMS. 점수가 높은 것부터 남기고, 이미 남긴 같은 클래스 박스와 IoU가
    iou_thr보다 크면 버린다 (공식 `nms`처럼 IoU <= 문턱이면 남긴다). 다른 클래스끼리는 둘 다 남는다.

    Returns:
        남긴 탐지 (점수 내림차순, 같은 점수면 입력 순서).
    """
    keep: list[YoloxDetection] = []
    for d in sorted(dets, key=lambda d: -d.score):
        if all(k.class_index != d.class_index or _iou(k.box, d.box) <= iou_thr for k in keep):
            keep.append(d)
    return keep


def _iou(a: Box, b: Box) -> float:
    """두 (x, y, w, h) 박스의 IoU. 합집합 넓이가 0이면 0."""
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


class Yolox:
    """YOLOX ONNX 추론기. 세션을 한 번 열고 이미지마다 `detect`를 부른다."""

    def __init__(
        self,
        model: Path,
        num_classes: int,
        *,
        input_size: int = INPUT_SIZE,
        decoded: bool = False,
        nms_iou: float = NMS_IOU,
        threads: int = 0,
    ) -> None:
        """모델을 연다.

        Args:
            model: ONNX 파일 (`registry.resolve`가 돌려준 경로).
            num_classes: 출력 클래스 수 C (정책 `classes` 길이). 출력 열이 5 + C가 아니면 `detect`가
                ValueError를 낸다.
            input_size: 정사각 입력 한 변(px, 32의 배수). 내보낼 때의 test_size와 같아야 한다.
            decoded: True면 `--decode_in_inference`로 내보낸 모델 (출력이 이미 입력 픽셀).
                False(공식 export 기본)면 `grid_decode`를 거친다.
            nms_iou: NMS IoU 문턱 (정책 값을 넘긴다). threads: ONNX Runtime 스레드 수 (0이면 기본).
        """
        self.session = OnnxModel(model, threads)
        self.num_classes, self.decoded, self.nms_iou = num_classes, decoded, nms_iou
        self.size = (input_size, input_size)

    def detect(self, rgb: NDArray[np.uint8], thresholds: Sequence[float]) -> list[YoloxDetection]:
        """이미지 한 장에서 탐지한다.

        Args:
            rgb: (H, W, 3) RGB uint8 (`dlp_prelabel.common.iter_frames` 출력). BGR로 바꿔 넣는다.
            thresholds: 클래스별 점수 문턱 (길이 C).

        Returns:
            원래 이미지 픽셀 좌표의 탐지 목록 (`decode` 참고).

        Raises:
            ValueError: 출력 모양이 (1, N, 5 + C)가 아니거나 문턱 길이가 C가 아닐 때.
        """
        if len(thresholds) != self.num_classes:
            raise ValueError(f"문턱 {len(thresholds)}개가 클래스 {self.num_classes}개와 다릅니다")
        bgr = np.ascontiguousarray(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), dtype=np.uint8)
        pixels, ratio = letterbox(bgr, self.size)
        (out,) = self.session.run([OUTPUT_NAME], {INPUT_NAME: pixels})
        if out.ndim != 3 or out.shape[0] != 1:
            raise ValueError(f"YOLOX 출력 모양 {out.shape}가 (1, N, 5 + C)가 아닙니다")
        pred = out[0] if self.decoded else grid_decode(out[0], self.size)
        h, w = rgb.shape[:2]
        return decode(pred, ratio, w, h, thresholds, self.nms_iou)
