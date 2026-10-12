"""반사면 속 얼굴: 반사면 영역(거울, 유리, 꺼진 화면)을 찾고 그 안에서 얼굴 탐지 문턱을 낮춘다.

가슴 캠 영상의 거울·유리·화면에 비치는 얼굴을 찾는 데 사용한다. 탐지 성공을 보장하지 않으므로
누락 여부는 사람이 원본 영상으로 검수해야 한다. 영역 탐지기는
target이 "reflective_surface"인 탐지를 낸다.

WP5. 정책: privacy.yaml `detectors.reflection` (region_detector: open_vocab, face_detector: yunet,
threshold_scale). 결과 대상은 "reflection"이며, 그 블러는 항상 reflection 검수 우선 구간이 된다.
버전은 두 하위 탐지기 버전을 묶은 문자열이다.
영역은 프레임 안으로 자르고 바깥쪽으로 정수화해 잘라 낸다 (`crop_bounds`, 1px 미만 영역도 1px).
"""

from __future__ import annotations

import math

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
        crop 범위는 `crop_bounds`로 정하고, 프레임 밖이거나 빈 영역은 건너뛴다.
        """
        out: list[Detection] = []
        for region in self.region.detect(image, t_ms, threshold):
            if region.target != REGION_TARGET:
                continue
            bounds = crop_bounds(region.box, image)
            if bounds is None:
                continue
            x1, y1, x2, y2 = bounds
            crop = image[y1:y2, x1:x2]
            for face in self.face.detect(crop, t_ms, threshold * self.threshold_scale):
                b = face.box
                # crop 좌표 → 원 이미지 좌표 (crop 원점 = (x1, y1))
                box = Box(b.x + x1, b.y + y1, b.w, b.h)
                out.append(Detection("reflection", box, face.score, self.name))
        return out


def crop_bounds(box: Box, image: Image) -> tuple[int, int, int, int] | None:
    """영역 박스를 덮는 정수 crop 범위 (x1, y1, x2, y2). 잘라 낼 화소가 없으면 None.

    박스를 프레임 안으로 자른 뒤 바깥쪽으로 정수화한다 (x1·y1 내림, x2·y2 올림, 부분 화소도 넣는다).
    그래서 폭·높이가 1px 미만인 영역도 1px 이상의 crop이 되고, 소수 좌표 영역의 끝을 놓치지 않는다.
    예전에는 원점만 내림하고 폭·높이를 int로 버려, 1px 미만 영역이 빈 crop이 되고 얼굴 탐지기
    (YuNet의 cv2.cvtColor 단언)가 `privacy detect` 전체를 중단시켰다 (F-PMS-03).
    빈 crop은 탐지기에 넘기지 않는다. YuNet은 1x1 입력도 받으므로 그 밖의 최소 크기는 두지 않는다.

    Args:
        box: 영역 박스 (원 이미지 화소 좌표, 프레임 밖으로 걸칠 수 있다).
        image: 원 이미지 (H, W, 3).

    Returns:
        `image[y1:y2, x1:x2]`로 자를 범위. crop 원점은 (x1, y1)이다.
    """
    h, w = image.shape[:2]
    r = box.clipped(w, h)
    if r is None:
        return None
    x1, y1 = max(0, math.floor(r.x)), max(0, math.floor(r.y))
    x2, y2 = min(w, math.ceil(r.x + r.w)), min(h, math.ceil(r.y + r.h))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2
