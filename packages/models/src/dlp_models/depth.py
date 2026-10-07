"""단안 메트릭 깊이 (Depth Anything V2 Metric Indoor Small, ONNX Runtime, CPU)와 2D→3D 변환.

깊이 맵은 미터 단위이고 카메라 좌표계(광축 Z 앞쪽, X 오른쪽, Y 아래쪽)로 픽셀을 올린다.
카메라 내부 파라미터가 세션에 없으면 정책의 기본 화각으로 근사한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from dlp_models.onnx import OnnxModel

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
INPUT = 518  # 14의 배수


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_hfov(cls, width: int, height: int, hfov_deg: float) -> Intrinsics:
        f = width / 2 / np.tan(np.deg2rad(hfov_deg) / 2)
        return cls(float(f), float(f), width / 2, height / 2)

    def unproject(self, u: float, v: float, z: float) -> tuple[float, float, float]:
        return ((u - self.cx) * z / self.fx, (v - self.cy) * z / self.fy, z)


class MetricDepth:
    def __init__(self, model: Path, threads: int = 0) -> None:
        self.session = OnnxModel(model, threads)

    def predict(self, image: NDArray[np.uint8]) -> NDArray[np.float32]:
        """원래 이미지 크기의 깊이 맵 (미터)."""
        h, w = image.shape[:2]
        x = (
            cv2.resize(image, (INPUT, INPUT), interpolation=cv2.INTER_CUBIC).astype(np.float32)
            / 255
        )
        x = ((x - MEAN) / STD).transpose(2, 0, 1)[None]
        [out] = self.session.run(["predicted_depth"], {"pixel_values": np.ascontiguousarray(x)})
        depth: NDArray[np.float32] = np.ascontiguousarray(out[0], dtype=np.float32)
        return np.asarray(cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR), np.float32)


def sample_depth(depth: NDArray[np.float32], u: float, v: float, radius: int = 2) -> float:
    """(u, v) 주변 (2r+1)^2 픽셀 깊이의 중앙값. 화면 밖이면 nan."""
    h, w = depth.shape
    x, y = round(u), round(v)
    if not (0 <= x < w and 0 <= y < h):
        return float("nan")
    patch = depth[max(0, y - radius) : y + radius + 1, max(0, x - radius) : x + radius + 1]
    return float(np.median(patch))
