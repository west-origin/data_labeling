"""반사면 속 얼굴: 반사면 영역(거울, 유리, 꺼진 화면)을 찾고 그 안에서 얼굴 탐지 문턱을 낮춘다.

가슴 캠은 착용자 얼굴이 거울에서만 보이므로 화장실 청소에서 필수다. 영역 탐지기는
target이 "reflective_surface"인 탐지를 낸다.

WP5. 정책: privacy.yaml `detectors.reflection` (region_detector: open_vocab, face_detector: yunet,
threshold_scale). 결과 대상은 "reflection"이며, 그 블러는 항상 reflection 검수 우선 구간이 된다.
버전은 두 하위 탐지기 버전을 묶은 문자열이다.
"""

from __future__ import annotations

from dlp_privacy.detection import Detection, FrameDetector, Image
from dlp_privacy.geometry import Box

# 영역 탐지기가 반사면 영역에 붙이는 대상 ID (온톨로지 대상이 아닌 중간 결과)
REGION_TARGET = "reflective_surface"


class ReflectionDetector:
    """반사면 영역 안의 얼굴을 "reflection" 대상으로 내는 `FrameDetector`."""

    def __init__(
        self, name: str, region: FrameDetector, face: FrameDetector, threshold_scale: float
    ) -> None:
        """
        Args:
            name: 정책 탐지기 이름.
            region: 반사면 영역 탐지기 (보통 OWLv2, 대상 탐지와 같은 인스턴스를 공유).
            face: 얼굴 탐지기 (보통 YuNet).
            threshold_scale: 영역 안 얼굴 문턱 배율 (문턱 x 배율, 거울 속 얼굴은 점수가 낮다).
        """
        self.name = name
        self.version = f"reflection({region.version},{face.version})"
        self.region = region
        self.face = face
        self.threshold_scale = threshold_scale

    def detect(self, image: Image, t_ms: int, threshold: float) -> list[Detection]:
        """반사면 영역마다 잘라 낸 이미지에서 얼굴을 찾고 원 이미지 좌표로 되돌린다.

        영역 탐지에는 원래 문턱을, 얼굴 탐지에는 낮춘 문턱(threshold x threshold_scale)을 쓴다.
        """
        out: list[Detection] = []
        h, w = image.shape[:2]
        for region in self.region.detect(image, t_ms, threshold):
            if region.target != REGION_TARGET:
                continue
            r = region.box.clipped(w, h)
            if r is None:
                continue
            # 영역 좌표를 정수로 내림해 자른다 (crop 원점 = (x, y))
            x, y = int(r.x), int(r.y)
            crop = image[y : y + int(r.h), x : x + int(r.w)]
            for face in self.face.detect(crop, t_ms, threshold * self.threshold_scale):
                b = face.box
                # crop 좌표 → 원 이미지 좌표
                out.append(
                    Detection("reflection", Box(b.x + x, b.y + y, b.w, b.h), face.score, self.name)
                )
        return out
