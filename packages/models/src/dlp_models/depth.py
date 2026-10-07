"""단안 메트릭 깊이 (Depth Anything V2 Metric Indoor Small, ONNX Runtime, CPU)와 2D→3D 변환.

전처리는 공식 구현(Depth-Anything-V2 `image2tensor`)과 같다: 가로세로 비율을 지키고, 짧은 변이
518 이상이 되도록(lower_bound) 키운 뒤 두 변을 14의 배수로 맞춘다. 정사각형으로 늘이면 형태가
찌그러져 깊이가 틀어진다.

깊이 맵은 미터 단위이고 카메라 좌표계(광축 Z 앞쪽, X 오른쪽, Y 아래쪽)로 픽셀을 올린다.
카메라 내부 파라미터가 세션에 없으면 정책의 기본 화각으로 근사한다.

파이프라인 위치: `dlp prelabel run`의 3D 궤적 단계(`dlp_prelabel.lift3d.DepthLifter`)가 쓴다.
관련: WP8, ADR 0009(실제 모델), ADR 0010(상업 사용).

모델 입출력 (`scripts/export_depth_onnx.py`로 Hugging Face판에서 변환한 ONNX):
- 입력 `pixel_values`: (1, 3, H', W') float32. RGB, 0~1로 나눈 뒤 ImageNet 평균·표준편차로 정규화.
  H', W'는 `input_size`가 정한다 (14의 배수, 짧은 변 ≥ 518).
- 출력 `predicted_depth`: (1, H', W') float32, 미터. `MetricDepth.predict`가 원래 해상도로 되돌린다.

가중치·라이선스: `config/models.yaml`의 `depth_metric_indoor_small` (가중치 Apache-2.0, 학습
데이터에
비상업 조건이 있어 `commercial: review`, ADR 0010).

공개 항목:
- `Intrinsics`: 핀홀 카메라 내부 파라미터(픽셀)와 역투영.
- `input_size`: 공식 Resize와 같은 모델 입력 크기 계산.
- `MetricDepth`: RGB 이미지 → 같은 크기 깊이 맵(미터).
- `sample_depth`: 한 점 주변 패치의 깊이 중앙값.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from dlp_models.onnx import OnnxModel

# ImageNet RGB 평균·표준편차 (0~1 스케일). 공식 전처리 `NormalizeImage`와 같은 값이다.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
INPUT = 518  # 짧은 변의 하한 (14의 배수)
MULTIPLE = 14  # ViT 패치 크기


@dataclass(frozen=True)
class Intrinsics:
    """핀홀 카메라 내부 파라미터. 모두 픽셀 단위다.

    fx, fy: 초점 거리(px). cx, cy: 주점(px, 왼쪽 위 원점). 렌즈 왜곡은 다루지 않는다
    (광각 액션캠은 왜곡이 커서 화면 가장자리의 3D 위치 오차가 커진다).
    """

    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_hfov(cls, width: int, height: int, hfov_deg: float) -> Intrinsics:
        """가로 화각으로 근사한 내부 파라미터 (캘리브레이션이 없을 때).

        f = (width / 2) / tan(hfov / 2). 세로도 같은 f(정사각 픽셀)로 보고, 주점은 화면 중앙
        (width/2, height/2)으로 둔다.

        Args:
            width, height: 이미지 크기(px).
            hfov_deg: 가로 화각(도, 0 < hfov < 180). 정책 `prelabel.yaml depth.default_hfov_deg`.
        """
        f = width / 2 / np.tan(np.deg2rad(hfov_deg) / 2)
        return cls(float(f), float(f), width / 2, height / 2)

    def unproject(self, u: float, v: float, z: float) -> tuple[float, float, float]:
        """픽셀 (u, v)와 깊이 z(m)를 카메라 좌표 (x, y, z) 미터로 올린다.

        x = (u - cx) * z / fx, y = (v - cy) * z / fy. 좌표계는 Z 앞(광축), X 오른쪽, Y 아래.
        z는 광축 방향 깊이(평면 깊이)로 본다 (카메라 중심까지의 거리가 아니다).
        """
        return ((u - self.cx) * z / self.fx, (v - self.cy) * z / self.fy, z)


def input_size(
    height: int, width: int, target: int = INPUT, multiple: int = MULTIPLE
) -> tuple[int, int]:
    """모델 입력 (높이, 너비).

    공식 Resize(keep_aspect_ratio, ensure_multiple_of=14, resize_method=lower_bound)와 같다.

    두 변 모두 target 이상이 되도록 같은 배율(큰 쪽 배율)로 키운 뒤, 각 변을 가장 가까운 14의
    배수로 반올림한다. 반올림 결과가 target보다 작아지면 올림한다 (짧은 변이 518 아래로 내려가지
    않게). 예: (240, 320) → (518, 686).

    Args:
        height, width: 원래 이미지 크기(px).
        target: 짧은 변 하한 (기본 518).
        multiple: 맞출 배수 (기본 14, ViT 패치 크기).
    """
    scale = max(target / height, target / width)

    def constrain(x: float) -> int:
        """x를 multiple의 배수로 반올림하되 target보다 작아지면 올림한다."""
        y = round(x / multiple) * multiple
        if y < target:
            y = math.ceil(x / multiple) * multiple
        return int(y)

    return constrain(scale * height), constrain(scale * width)


class MetricDepth:
    """메트릭 깊이 추론기 (CPU). 프레임 한 장에 수백 ms~수 초가 걸린다.

    호출자(`lift3d.lift_tracks`)가 `prelabel.yaml depth.frame_stride_ms` 간격으로만 부른다.
    """

    def __init__(self, model: Path, threads: int = 0) -> None:
        """ONNX 세션을 연다.

        Args:
            model: ONNX 파일 경로 (`registry.resolve(root, "depth_metric_indoor_small")`).
            threads: ONNX Runtime 스레드 수 (0이면 기본값).
        """
        self.session = OnnxModel(model, threads)

    def predict(self, image: NDArray[np.uint8]) -> NDArray[np.float32]:
        """원래 이미지 크기의 깊이 맵 (미터).

        Args:
            image: (H, W, 3) RGB uint8 (BGR이 아니다. `dlp_prelabel.common.iter_frames` 출력).

        Returns:
            (H, W) float32 깊이(m). 모델 출력(H', W')을 쌍선형 보간으로 원래 크기로 되돌린다.
        """
        h, w = image.shape[:2]
        ih, iw = input_size(h, w)
        # 공식 구현처럼 바이큐빅으로 키우고 0~1 → ImageNet 정규화 → HWC를 NCHW(배치 1)로 바꾼다
        x = cv2.resize(image, (iw, ih), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255
        x = ((x - MEAN) / STD).transpose(2, 0, 1)[None]
        [out] = self.session.run(["predicted_depth"], {"pixel_values": np.ascontiguousarray(x)})
        depth: NDArray[np.float32] = np.ascontiguousarray(out[0], dtype=np.float32)
        # cv2.resize의 크기 인자는 (너비, 높이) 순서다
        return np.asarray(cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR), np.float32)


def sample_depth(depth: NDArray[np.float32], u: float, v: float, radius: int = 2) -> float:
    """(u, v) 주변 (2r+1)^2 픽셀 깊이의 중앙값. 화면 밖이면 nan.

    키포인트가 물체 경계에 걸리면 한 픽셀 깊이가 배경으로 튈 수 있어 중앙값으로 버틴다.
    화면 가장자리에서는 패치를 화면 안으로 자른다 (픽셀 수가 줄어든다).

    Args:
        depth: (H, W) 깊이 맵(m).
        u, v: 픽셀 좌표 (가로, 세로). 가장 가까운 정수 픽셀로 반올림한다.
        radius: 패치 반지름 r(px). 정책 `prelabel.yaml depth.patch_radius_px`.

    Returns:
        깊이(m). 반올림한 (u, v)가 화면 밖이면 `nan` (호출자가 버린다).
    """
    h, w = depth.shape
    x, y = round(u), round(v)
    if not (0 <= x < w and 0 <= y < h):
        return float("nan")
    patch = depth[max(0, y - radius) : y + radius + 1, max(0, x - radius) : x + radius + 1]
    return float(np.median(patch))
