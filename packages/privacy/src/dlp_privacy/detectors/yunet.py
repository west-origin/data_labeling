"""YuNet 얼굴 탐지 (OpenCV FaceDetectorYN, CPU).

가중치는 `make models`로 받는다 (config/models.yaml yunet, opencv_zoo, MIT 라이선스).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from dlp_privacy.detection import Detection, Image
from dlp_privacy.geometry import Box


class YuNetFaceDetector:
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
        h, w = image.shape[:2]
        self.model.setInputSize((w, h))
        self.model.setScoreThreshold(threshold)
        _, faces = self.model.detect(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if faces is None:  # pyright: ignore[reportUnnecessaryComparison]  (얼굴이 없으면 None)
            return []
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
