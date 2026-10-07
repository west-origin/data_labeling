"""QR 코드·바코드 탐지 (OpenCV, 가중치 없음). 송장·라벨이 있다는 근거로 shipping_label을 낸다.

코드 영역만 잡는다. 송장 전체 영역은 오픈 보캐뷸러리 탐지기가 잡으며, 코드만 잡힌 구간은
탐지기 불일치 구간으로 검수 화면에 먼저 뜬다.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from dlp_privacy.detection import Detection, Image
from dlp_privacy.geometry import Box


def _box(points: NDArray[np.float32]) -> Box:
    xs, ys = points[:, 0], points[:, 1]
    return Box(
        float(xs.min()), float(ys.min()), float(xs.max() - xs.min()), float(ys.max() - ys.min())
    )


class CodeDetector:
    version = f"opencv-{cv2.__version__}"

    def __init__(self, name: str, score: float) -> None:
        self.name = name
        self.score = (
            score  # 코드는 확실히 찾거나 못 찾으므로 고정 신뢰도 (정책 detectors.codes.score)
        )
        self.qr = cv2.QRCodeDetector()
        self.barcode = cv2.barcode.BarcodeDetector()

    def detect(self, image: Image, t_ms: int, threshold: float) -> list[Detection]:
        if threshold > self.score:
            return []
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        boxes: list[Box] = []
        ok, points = self.qr.detectMulti(gray)
        if ok and points is not None:  # pyright: ignore[reportUnnecessaryComparison]
            boxes += [_box(np.asarray(p, dtype=np.float32)) for p in points]
        ok, points = self.barcode.detect(gray)
        if ok and points is not None:  # pyright: ignore[reportUnnecessaryComparison]
            boxes += [_box(np.asarray(p, dtype=np.float32)) for p in points]
        return [Detection("shipping_label", b, self.score, self.name) for b in boxes if b.area > 0]
