"""SAM 2.1 박스·점 프롬프트 분할 (ONNX, CPU, ADR 0059).

공식 SAM 2.1 hiera-tiny 가중치(facebook/sam2.1-hiera-tiny, Apache-2.0)를 onnx-community가 영상
인코더(`vision_encoder.onnx`)와 프롬프트 인코더·마스크 디코더(`prompt_encoder_mask_decoder.onnx`)로
나눠 변환한 것을 쓴다. 두 그래프는 가중치를 외부 데이터 파일(`*.onnx_data`)에 두므로 같은 디렉터리에
그 이름 그대로 있어야 한다 (`config/models.yaml sam2_tiny_*`).

영상 모드(메모리 뱅크로 프레임 사이 전파)는 ONNX 공개본이 없어 쓰지 않는다. 호출자
(`dlp_prelabel.masks`)가 정책 간격의 프레임마다 박스 트랙 키프레임을 프롬프트로 넣는 이미지 모드다.

- 전처리: RGB를 1024x1024로 쌍선형 크기 변경(화면비 무시, 변환본 전처리 설정과 같다) → 0~1 →
  ImageNet 정규화. 프롬프트 좌표도 같은 비율로 바꾼다.
- 출력: 프롬프트마다 마스크 후보 3개(256x256 로짓)와 예측 IoU. 예측 IoU가 가장 큰 후보를 원래
  크기로 쌍선형 보간해 0보다 큰 화소를 마스크로 한다. 대상 점수 로짓이 0 이하면(대상 없음) None.

이 모듈은 라벨·DB를 다루지 않는다. 입력은 메모리의 RGB 이미지다.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from dlp_models.onnx import OnnxModel

# 모델 입력 크기 (정사각) 와 ImageNet 정규화 값 (변환본 preprocessor_config.json)
INPUT_SIZE = 1024
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# 인코더 출력(다단계 특징) 이름. 디코더 입력 이름과 같다
EMBEDDINGS = ("image_embeddings.0", "image_embeddings.1", "image_embeddings.2")

# (x, y, w, h) 픽셀 박스, 왼쪽 위 원점
Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Prompt:
    """프롬프트 하나: 박스와 (선택) 대상 안쪽 점들 (픽셀)."""

    box: Box
    points: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True)
class MaskResult:
    """분할 결과: (H, W) 불리언 마스크와 예측 IoU 점수(0~1)."""

    mask: NDArray[np.bool_]
    score: float


@dataclass(frozen=True)
class Embedding:
    """한 프레임의 인코더 출력과 원래 크기 (높이, 너비)."""

    features: tuple[NDArray[np.float32], ...]
    size: tuple[int, int]


class Sam2Segmenter:
    """SAM 2.1 ONNX 분할기 (CPU). 인코더는 프레임당 한 번, 디코더는 프롬프트마다 돈다."""

    def __init__(self, encoder: Path, decoder: Path, threads: int = 0) -> None:
        """두 ONNX 세션을 연다 (외부 데이터 파일은 같은 디렉터리에서 읽는다).

        Args:
            encoder, decoder: `.onnx` 경로 (`registry.resolve`가 확인한 것).
            threads: ONNX Runtime 연산 스레드 수 (0이면 기본값).
        """
        self.encoder = OnnxModel(encoder, threads)
        self.decoder = OnnxModel(decoder, threads)

    def embed(self, image: NDArray[np.uint8]) -> Embedding:
        """(H, W, 3) RGB uint8 → 인코더 특징."""
        h, w = image.shape[:2]
        x = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        x = ((x.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]
        out = self.encoder.run(list(EMBEDDINGS), {"pixel_values": np.ascontiguousarray(x)})
        return Embedding(tuple(out), (h, w))

    def predict(
        self, image: NDArray[np.uint8], prompts: Sequence[Prompt]
    ) -> list[MaskResult | None]:
        """한 프레임을 인코딩하고 프롬프트마다 마스크를 낸다 (`dlp_prelabel.masks.Segmenter`)."""
        return self.segment(self.embed(image), prompts) if prompts else []

    def segment(self, emb: Embedding, prompts: Sequence[Prompt]) -> list[MaskResult | None]:
        """프롬프트마다 마스크 (대상 없음이면 None). 점 없는 프롬프트는 한 번에 묶어 디코딩한다."""
        out: list[MaskResult | None] = [None] * len(prompts)
        boxed = [i for i, p in enumerate(prompts) if not p.points]
        if boxed:
            for i, r in zip(boxed, self._decode(emb, [prompts[i] for i in boxed], ()), strict=True):
                out[i] = r
        for i, p in enumerate(prompts):
            if p.points:
                [out[i]] = self._decode(emb, [p], p.points)
        return out

    def _decode(
        self, emb: Embedding, prompts: Sequence[Prompt], points: tuple[tuple[float, float], ...]
    ) -> list[MaskResult | None]:
        """박스 여러 개(점은 그 박스들에 공통, 보통 박스 하나에만 씀)를 디코딩한다."""
        h, w = emb.size
        sx, sy = INPUT_SIZE / w, INPUT_SIZE / h
        boxes = np.array(
            [[(x) * sx, (y) * sy, (x + bw) * sx, (y + bh) * sy] for (x, y, bw, bh) in
             (p.box for p in prompts)],
            dtype=np.float32,
        )[None]  # fmt: skip
        pts = np.array([[(px * sx, py * sy) for px, py in points]], dtype=np.float32)
        pts = pts.reshape(1, 1, len(points), 2)
        feeds: dict[str, NDArray[np.generic]] = {
            "input_points": pts,
            "input_labels": np.ones((1, 1, len(points)), dtype=np.int64),
            "input_boxes": boxes,
            **dict(zip(EMBEDDINGS, emb.features, strict=True)),
        }
        scores, masks, objects = self.decoder.run(
            ["iou_scores", "pred_masks", "object_score_logits"], feeds
        )
        results: list[MaskResult | None] = []
        for j in range(len(prompts)):
            if float(objects[0, j, 0]) <= 0.0:
                results.append(None)
                continue
            best = int(np.argmax(scores[0, j]))
            low: NDArray[np.float32] = masks[0, j, best]
            logits = np.asarray(cv2.resize(low, (w, h), interpolation=cv2.INTER_LINEAR), np.float32)
            score = float(np.clip(scores[0, j, best], 0.0, 1.0))
            results.append(MaskResult(logits > 0.0, score))
        return results
