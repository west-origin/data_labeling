"""QR 코드·바코드 탐지 (OpenCV, 가중치 없음). 송장·라벨이 있다는 근거로 shipping_label을 낸다.

코드 영역만 잡는다. 송장 전체 영역은 오픈 보캐뷸러리 탐지기가 잡으며, 코드만 잡힌 구간은
탐지기 불일치 구간으로 검수 화면에 먼저 뜬다.

WP5. 정책: privacy.yaml `detectors.codes` (kind: opencv_codes, score: 고정 신뢰도).
버전은 OpenCV 버전이다 (OpenCV를 올리면 탐지 모델 버전이 바뀌어 다시 탐지한다).
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from dlp_privacy.detection import Detection, Image
from dlp_privacy.geometry import Box


def _box(points: NDArray[np.float32]) -> Box:
    """코드 모서리 점들 (N, 2)의 외접 축 정렬 박스 (회전된 코드도 덮는다)."""
    xs, ys = points[:, 0], points[:, 1]
    return Box(
        float(xs.min()), float(ys.min()), float(xs.max() - xs.min()), float(ys.max() - ys.min())
    )


class CodeDetector:
    """QR 코드·1차원 바코드 위치를 찾아 shipping_label 탐지로 낸다 (`FrameDetector`)."""

    # 탐지기 버전 = 설치된 OpenCV 버전
    version = f"opencv-{cv2.__version__}"

    def __init__(self, name: str, score: float) -> None:
        """name: 정책 탐지기 이름, score: 모든 탐지에 줄 고정 신뢰도 (0~1)."""
        self.name = name
        self.score = (
            score  # 코드는 확실히 찾거나 못 찾으므로 고정 신뢰도 (정책 detectors.codes.score)
        )
        self.qr = cv2.QRCodeDetector()
        self.barcode = cv2.barcode.BarcodeDetector()

    def detect(self, image: Image, t_ms: int, threshold: float) -> list[Detection]:
        """프레임에서 QR·바코드를 찾는다 (내용은 해독하지 않고 위치만).

        고정 신뢰도가 문턱보다 낮으면 아무것도 내지 않는다. t_ms는 쓰지 않는다 (상태 없음).
        넓이가 0인 박스(점·선으로 퇴화)는 버린다.
        """
        if threshold > self.score:
            return []
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        boxes: list[Box] = []
        # OpenCV 스텁은 points를 None이 아닌 배열로 적지만 실제로는 None일 수 있다
        ok, points = self.qr.detectMulti(gray)
        if ok and points is not None:  # pyright: ignore[reportUnnecessaryComparison]
            boxes += [_box(np.asarray(p, dtype=np.float32)) for p in points]
        ok, points = self.barcode.detect(gray)
        if ok and points is not None:  # pyright: ignore[reportUnnecessaryComparison]
            boxes += [_box(np.asarray(p, dtype=np.float32)) for p in points]
        return [Detection("shipping_label", b, self.score, self.name) for b in boxes if b.area > 0]
