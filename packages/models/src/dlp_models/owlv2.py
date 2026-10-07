"""OWLv2 오픈 보캐뷸러리 탐지 (ONNX Runtime, CPU). 텍스트 질의로 임의의 대상을 찾는다.

전처리는 Hugging Face Owlv2ImageProcessor와 같다: 짧은 변을 회색(0.5)으로 채워 정사각형으로
만들고(오른쪽·아래), 960x960으로 줄인 뒤 CLIP 평균·표준편차로 정규화한다. 출력 박스는 정사각형
기준 정규화된 (cx, cy, w, h)라 원래 이미지 픽셀로 되돌릴 때 긴 변 길이를 곱한다.

CPU에서 프레임당 수 초가 걸린다. 영상에는 프레임 간격(stride)을 두고, GPU 환경에서 간격을 줄인다.

파이프라인 위치: `dlp prelabel run`의 도구 박스(`dlp_prelabel.adapters.owl_objects.OwlObjects`,
질의는 `prelabel.yaml open_vocab_objects.queries`)와 `dlp privacy detect`의 오픈 보캐뷸러리 탐지기
(`dlp_privacy.detectors`)가 쓴다. 관련: WP8, ADR 0009, ADR 0010.

모델 입출력 (Xenova/owlv2-base-patch16-ensemble `model_quantized.onnx`, 고정 리비전):
- 입력 `input_ids`, `attention_mask`: (Q, 16) int64. 질의 Q개를 CLIP BPE 토크나이저
  (`tokenizer.json`)로 바꿔 16토큰으로 자르거나 0으로 채운다.
- 입력 `pixel_values`: (1, 3, 960, 960) float32 (`preprocess`).
- 출력 `logits`: (1, N, Q) 패치 N개, 질의 Q개의 점수(시그모이드 전). N = (960/16)^2 = 3600.
- 출력 `pred_boxes`: (1, N, 4) 패딩된 정사각형 기준 0~1 정규화 (cx, cy, w, h).

좌표계: 반환하는 `OwlDetection.box`는 원래 이미지 픽셀의 (x, y, w, h)이고 왼쪽 위가 원점이다.

가중치·라이선스: `config/models.yaml`의 `owlv2_onnx`(Apache-2.0, 학습 데이터에 Objects365 비상업 →
`commercial: review`)와 `owlv2_tokenizer`(allowed). ADR 0010.

공개 항목:
- `OwlDetection`: 탐지 하나 (질의 번호, 픽셀 박스, 점수).
- `preprocess`, `decode`, `nms`: 모델 없이 시험할 수 있는 전·후처리.
- `Owlv2`: 질의를 한 번 토큰화해 두고 이미지마다 `detect`를 부르는 추론기.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from tokenizers import Tokenizer

from dlp_models.onnx import OnnxModel

SIZE = 960  # 모델 입력 한 변(px). OWLv2 base-patch16 고정 해상도
# CLIP 이미지 정규화 평균·표준편차 (0~1 스케일, RGB). Owlv2ImageProcessor 기본값과 같다
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
# 질의 하나의 최대 토큰 수. 내보낸 ONNX 그래프의 텍스트 입력 길이와 같아야 한다 (HF 처리기 기본값
# 16)
MAX_TOKENS = 16
NMS_IOU = (
    0.5  # 기본값. 호출하는 단계는 정책 값(예: prelabel.yaml open_vocab_objects.nms_iou)을 넘긴다
)


@dataclass(frozen=True)
class OwlDetection:
    """OWLv2 탐지 하나.

    query_index: `Owlv2(queries=...)`에 넘긴 질의 목록의 번호 (0부터).
    box: 원래 이미지 픽셀 좌표 (x, y, w, h), 이미지 안으로 잘린 값.
    score: 시그모이드 점수 (0~1). 질의별 문턱은 호출자가 정한다.
    """

    query_index: int
    box: tuple[float, float, float, float]  # x, y, w, h 픽셀
    score: float


def preprocess(image: NDArray[np.uint8]) -> tuple[NDArray[np.float32], int]:
    """(1, 3, 960, 960) 입력과 정사각형 한 변 길이.

    1) 0~1로 나누고 긴 변 크기의 정사각형 왼쪽 위에 놓는다 (나머지는 0.5 회색, 오른쪽·아래 채움).
    2) 960x960으로 쌍선형 축소·확대. 3) CLIP 평균·표준편차 정규화. 4) HWC → NCHW.

    Args:
        image: (H, W, 3) RGB uint8.

    Returns:
        (모델 입력, side). side = max(H, W)이고 `decode`가 정규화 박스를 픽셀로 되돌릴 때 쓴다.
    """
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
    nms_iou: float = NMS_IOU,
) -> list[OwlDetection]:
    """logits (N, Q), boxes (N, 4) cxcywh 정규화 → 질의별 문턱을 넘는 탐지.

    박스는 이미지 안으로 자른다 (정사각형 채움 영역은 잘려 나간다).

    Args:
        logits: 배치 차원을 뺀 (N, Q) 점수 (시그모이드 전).
        boxes: 배치 차원을 뺀 (N, 4) 정규화 (cx, cy, w, h). 패딩된 정사각형 기준이다.
        side: `preprocess`가 돌려준 정사각형 한 변(px). 정규화 좌표에 곱해 픽셀로 바꾼다.
        thresholds: 질의별 점수 문턱 (길이 Q). 문턱 이상이면 남긴다.
        width, height: 원래 이미지 크기(px). 박스를 이 안으로 자른다.
        nms_iou: 같은 질의 박스끼리의 NMS IoU 문턱.

    Returns:
        NMS 뒤 탐지 목록 (점수 내림차순). 자른 뒤 넓이가 0인 박스(완전히 채움 영역)는 버린다.
    """
    scores = 1 / (1 + np.exp(-logits))  # 시그모이드
    out: list[OwlDetection] = []
    for q, thr in enumerate(thresholds):
        for i in np.flatnonzero(scores[:, q] >= thr):
            cx, cy, bw, bh = (float(v) * side for v in boxes[i])
            # 중심·크기 → 모서리. 정사각형 채움 영역(원래 이미지 밖)으로 나간 부분은 자른다
            x1, y1 = max(0.0, cx - bw / 2), max(0.0, cy - bh / 2)
            x2, y2 = min(float(width), cx + bw / 2), min(float(height), cy + bh / 2)
            if x2 > x1 and y2 > y1:
                out.append(OwlDetection(q, (x1, y1, x2 - x1, y2 - y1), float(scores[i, q])))
    return nms(out, nms_iou)


def nms(dets: list[OwlDetection], iou_thr: float) -> list[OwlDetection]:
    """질의별 탐욕 NMS. 점수가 높은 것부터 남기고, 이미 남긴 같은 질의 박스와 IoU가
    iou_thr 이상이면 버린다. 다른 질의끼리는 겹쳐도 둘 다 남는다.

    O(n^2)이지만 문턱을 넘는 탐지는 프레임당 수십 개라 충분하다.
    """
    keep: list[OwlDetection] = []
    for d in sorted(dets, key=lambda d: -d.score):
        if all(k.query_index != d.query_index or _iou(k.box, d.box) < iou_thr for k in keep):
            keep.append(d)
    return keep


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """두 (x, y, w, h) 박스의 IoU. 합집합 넓이가 0이면 0.

    `dlp_prelabel.common.iou`와 같은 계산이다 (패키지 의존 방향 때문에 따로 둔다).
    """
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


class Owlv2:
    """OWLv2 추론기. 질의 목록을 생성할 때 한 번 토큰화해 두고 이미지마다 재사용한다."""

    def __init__(
        self,
        model: Path,
        tokenizer: Path,
        queries: list[str],
        threads: int = 0,
        *,
        nms_iou: float = NMS_IOU,
    ) -> None:
        """모델·토크나이저를 열고 질의를 토큰화한다.

        Args:
            model: ONNX 파일 (`registry.resolve(root, "owlv2_onnx")`).
            tokenizer: Hugging Face `tokenizer.json` (`owlv2_tokenizer`).
            queries: 영어 질의 문장 (예: "a mop"). 순서가 `OwlDetection.query_index`가 된다.
            threads: ONNX Runtime 스레드 수 (0이면 기본값).
            nms_iou: NMS IoU 문턱 (정책 값을 넘긴다).
        """
        self.session = OnnxModel(model, threads)
        self.nms_iou = nms_iou
        tok = Tokenizer.from_file(str(tokenizer))
        self.queries = queries
        # (Q, 16) 토큰 ID와 마스크. 남는 자리는 0(패딩)이고 마스크도 0이다.
        # 16토큰보다 긴 질의는 뒤를 자른다 (질의는 짧은 명사구로 쓴다)
        self.input_ids = np.zeros((len(queries), MAX_TOKENS), dtype=np.int64)
        self.attention = np.zeros((len(queries), MAX_TOKENS), dtype=np.int64)
        for i, enc in enumerate(tok.encode_batch(queries)):
            n = min(len(enc.ids), MAX_TOKENS)
            self.input_ids[i, :n] = enc.ids[:n]
            self.attention[i, :n] = 1

    def detect(self, image: NDArray[np.uint8], thresholds: list[float]) -> list[OwlDetection]:
        """이미지 한 장에서 모든 질의를 찾는다.

        Args:
            image: (H, W, 3) RGB uint8.
            thresholds: 질의별 점수 문턱 (길이는 질의 수와 같다).

        Returns:
            원래 이미지 픽셀 좌표의 탐지 목록 (`decode` 참고).
        """
        pixels, side = preprocess(image)
        logits, boxes = self.session.run(
            ["logits", "pred_boxes"],
            {"input_ids": self.input_ids, "pixel_values": pixels, "attention_mask": self.attention},
        )
        h, w = image.shape[:2]
        return decode(
            np.asarray(logits)[0], np.asarray(boxes)[0], side, thresholds, w, h, self.nms_iou
        )
