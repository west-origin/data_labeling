"""OWLv2 오픈 보캐뷸러리 탐지 (ONNX Runtime, CPU). 텍스트 질의로 임의의 대상을 찾는다.

전처리는 Hugging Face Owlv2ImageProcessor와 같다: 짧은 변을 회색(0.5)으로 채워 정사각형으로
만들고(오른쪽·아래), 960x960으로 줄인 뒤 CLIP 평균·표준편차로 정규화한다. 출력 박스는 정사각형
기준 정규화된 (cx, cy, w, h)라 원래 이미지 픽셀로 되돌릴 때 긴 변 길이를 곱한다.

CPU에서 프레임당 수 초가 걸린다. 영상에는 프레임 간격(stride)을 두고, GPU 환경에서 간격을 줄인다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from tokenizers import Tokenizer

from dlp_models.onnx import OnnxModel

SIZE = 960
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
MAX_TOKENS = 16


@dataclass(frozen=True)
class OwlDetection:
    query_index: int
    box: tuple[float, float, float, float]  # x, y, w, h 픽셀
    score: float


def preprocess(image: NDArray[np.uint8]) -> tuple[NDArray[np.float32], int]:
    """(1, 3, 960, 960) 입력과 정사각형 한 변 길이."""
    h, w = image.shape[:2]
    side = max(h, w)
    square = np.full((side, side, 3), 0.5, dtype=np.float32)
    square[:h, :w] = image.astype(np.float32) / 255
    resized = cv2.resize(square, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    normalized = (resized - MEAN) / STD
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None], dtype=np.float32), side


def decode(
    logits: NDArray[np.float32],
    boxes: NDArray[np.float32],
    side: int,
    thresholds: list[float],
    width: int,
    height: int,
) -> list[OwlDetection]:
    """logits (N, Q), boxes (N, 4) cxcywh 정규화 → 질의별 문턱을 넘는 탐지.

    박스는 이미지 안으로 자른다 (정사각형 채움 영역은 잘려 나간다)."""
    scores = 1 / (1 + np.exp(-logits))
    out: list[OwlDetection] = []
    for q, thr in enumerate(thresholds):
        for i in np.flatnonzero(scores[:, q] >= thr):
            cx, cy, bw, bh = (float(v) * side for v in boxes[i])
            x1, y1 = max(0.0, cx - bw / 2), max(0.0, cy - bh / 2)
            x2, y2 = min(float(width), cx + bw / 2), min(float(height), cy + bh / 2)
            if x2 > x1 and y2 > y1:
                out.append(OwlDetection(q, (x1, y1, x2 - x1, y2 - y1), float(scores[i, q])))
    return nms(out, 0.5)


def nms(dets: list[OwlDetection], iou_thr: float) -> list[OwlDetection]:
    keep: list[OwlDetection] = []
    for d in sorted(dets, key=lambda d: -d.score):
        if all(k.query_index != d.query_index or _iou(k.box, d.box) < iou_thr for k in keep):
            keep.append(d)
    return keep


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


class Owlv2:
    def __init__(self, model: Path, tokenizer: Path, queries: list[str], threads: int = 0) -> None:
        self.session = OnnxModel(model, threads)
        tok = Tokenizer.from_file(str(tokenizer))
        self.queries = queries
        self.input_ids = np.zeros((len(queries), MAX_TOKENS), dtype=np.int64)
        self.attention = np.zeros((len(queries), MAX_TOKENS), dtype=np.int64)
        for i, enc in enumerate(tok.encode_batch(queries)):
            n = min(len(enc.ids), MAX_TOKENS)
            self.input_ids[i, :n] = enc.ids[:n]
            self.attention[i, :n] = 1

    def detect(self, image: NDArray[np.uint8], thresholds: list[float]) -> list[OwlDetection]:
        pixels, side = preprocess(image)
        logits, boxes = self.session.run(
            ["logits", "pred_boxes"],
            {"input_ids": self.input_ids, "pixel_values": pixels, "attention_mask": self.attention},
        )
        h, w = image.shape[:2]
        return decode(np.asarray(logits)[0], np.asarray(boxes)[0], side, thresholds, w, h)
