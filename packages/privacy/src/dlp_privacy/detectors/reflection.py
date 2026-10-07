"""반사면 속 얼굴: 반사면 영역(거울, 유리, 꺼진 화면)을 찾고 그 안에서 얼굴 탐지 문턱을 낮춘다.

가슴 캠은 착용자 얼굴이 거울에서만 보이므로 화장실 청소에서 필수다. 영역 탐지기는
target이 "reflective_surface"인 탐지를 낸다.
"""

from __future__ import annotations

from dlp_privacy.detection import Detection, FrameDetector, Image
from dlp_privacy.geometry import Box

REGION_TARGET = "reflective_surface"


class ReflectionDetector:
    def __init__(
        self, name: str, region: FrameDetector, face: FrameDetector, threshold_scale: float
    ) -> None:
        self.name = name
        self.version = f"reflection({region.version},{face.version})"
        self.region = region
        self.face = face
        self.threshold_scale = threshold_scale

    def detect(self, image: Image, t_ms: int, threshold: float) -> list[Detection]:
        out: list[Detection] = []
        h, w = image.shape[:2]
        for region in self.region.detect(image, t_ms, threshold):
            if region.target != REGION_TARGET:
                continue
            r = region.box.clipped(w, h)
            if r is None:
                continue
            x, y = int(r.x), int(r.y)
            crop = image[y : y + int(r.h), x : x + int(r.w)]
            for face in self.face.detect(crop, t_ms, threshold * self.threshold_scale):
                b = face.box
                out.append(
                    Detection("reflection", Box(b.x + x, b.y + y, b.w, b.h), face.score, self.name)
                )
        return out
