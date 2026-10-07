"""YuNet 얼굴 탐지 (OpenCV FaceDetectorYN, CPU).

가중치는 `make models`로 받는다 (config/models.yaml yunet, opencv_zoo, MIT 라이선스).

WP5, ADR 0009·0010. 정책: privacy.yaml `detectors.yunet` (model, nms_threshold, top_k).
버전은 `registry.resolve`가 준 `<이름>-<가중치 해시 12자>`다.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from dlp_privacy.detection import Detection, Image
from dlp_privacy.geometry import Box


class YuNetFaceDetector:
    """YuNet 얼굴 탐지 `FrameDetector`. 대상은 항상 "face"."""

    def __init__(
        self, name: str, model_path: Path, version: str, *, nms_threshold: float, top_k: int
    ) -> None:
        """nms_threshold·top_k는 privacy.yaml detectors.<이름>에서 온다. 입력 크기와 점수 문턱은
        프레임마다 detect에서 다시 정한다 (생성자 값은 자리 채움)."""
        self.name = name
        self.version = version
        self.model = cv2.FaceDetectorYN.create(
            str(model_path), "", (320, 320), 0.0, nms_threshold, top_k
        )

    def detect(self, image: Image, t_ms: int, threshold: float) -> list[Detection]:
        """프레임 크기 그대로 얼굴을 찾는다 (리사이즈 없음). t_ms는 쓰지 않는다.

        모델 상태(입력 크기·문턱)를 호출마다 바꾸므로 같은 인스턴스를 여러 스레드에서 동시에
        쓰면 안 된다 (reflection 탐지기가 잘라 낸 이미지로도 부른다).
        """
        h, w = image.shape[:2]
        self.model.setInputSize((w, h))
        self.model.setScoreThreshold(threshold)
        _, faces = self.model.detect(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))  # YuNet은 BGR 입력
        if faces is None:  # pyright: ignore[reportUnnecessaryComparison]  (얼굴이 없으면 None)
            return []
        # 행 형식: [x, y, w, h, 눈·코·입 랜드마크 10개, 점수] → 0~3열이 박스, 14열이 점수
        rows = np.asarray(faces, dtype=np.float32)
        return [
            Detection(
                "face",
                Box(float(r[0]), float(r[1]), float(r[2]), float(r[3])),
                float(r[14]),
                self.name,
            )
            for r in rows
        ]
