"""영상 프레임 기하 변환과 라벨 좌표 변환 (테스트 시점 증강 TTA·학습 증강 공용, ADR 0055·0057).

한 변환은 3x3 투영 행렬 하나(원본 픽셀 → 변환 픽셀)와 출력 크기다. 좌우 뒤집기·배율·회전·자르기·
원근 흔들기를 모두 같은 형식으로 나타내고, 이미지(`warp_image`)·점(`map_points`)·박스(`map_box`)에
같은 행렬을 쓴다. 그래서 픽셀과 라벨이 늘 같이 움직인다 (아래 좌표 규약을 지킬 때).

좌우 뒤집기의 라벨 의미:
- 손 판정: 뒤집은 화면의 오른손은 왼손 모양이다 → `Transform.flipped`이면 왼·오른을 바꾼다
  (`swap_hand`). hand21 관절 번호는 해부학 이름이라 그대로다.
- COCO 17점: 좌우 관절 번호를 맞바꾼다 (`COCO17_FLIP`).
- 카메라 좌표 3D: 영상 x 반전은 카메라 좌표 x 반전과 같다 (주점이 화면 가운데일 때. 아니면 근사).

좌표계: 스트림 픽셀, 왼쪽 위 원점, 박스 (x, y, w, h). 좌표는 연속 픽셀이다: 화면 왼쪽 위 모서리가
(0, 0)이고 화소 (i, j)의 중심은 (i + 0.5, j + 0.5)다. cv2 warp 함수는 화소 중심을 정수로 보므로
행렬을 그대로 넘기면 영상만 반 픽셀~1 픽셀 어긋난다 (좌우 뒤집기는 정확히 1 픽셀, F-PMS-37).
cv2에는 늘 `cv_matrix`로 바꾼 행렬을 넘긴다.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from dlp_schema.common import Contract

Box = tuple[float, float, float, float]
Matrix = NDArray[np.float64]

# COCO 17점 좌우 짝 (뒤집은 화면에서 i번 관절은 COCO17_FLIP[i]번 관절이다)
COCO17_FLIP = (0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15)


@dataclass(frozen=True)
class Transform:
    """원본 (width, height) → 출력 (out_w, out_h) 투영 변환.

    matrix: 3x3 (원본 픽셀 동차 좌표 → 출력 픽셀). flipped: 좌우가 뒤집혔는가 (손 판정·좌우 관절).
    name: 기록용 이름 (예: "hflip", "scale0.8").
    """

    matrix: Matrix
    out_w: int
    out_h: int
    flipped: bool
    name: str

    def inverse(self, width: int, height: int) -> Transform:
        """출력 → 원본 역변환 (원본 크기를 받아 출력 크기로 쓴다)."""
        return Transform(
            np.linalg.inv(self.matrix), width, height, self.flipped, f"inv-{self.name}"
        )

    def then(self, other: Transform) -> Transform:
        """이 변환 뒤에 other를 적용한 합성 변환."""
        return Transform(
            other.matrix @ self.matrix,
            other.out_w,
            other.out_h,
            self.flipped != other.flipped,
            f"{self.name}+{other.name}",
        )


def identity(width: int, height: int) -> Transform:
    """항등 변환."""
    return Transform(np.eye(3), width, height, False, "id")


def hflip(width: int, height: int) -> Transform:
    """좌우 뒤집기 (x → width - x)."""
    m = np.array([[-1.0, 0.0, float(width)], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    return Transform(m, width, height, True, "hflip")


def scale(width: int, height: int, factor: float) -> Transform:
    """전체 배율 (출력 크기도 factor배, 반올림)."""
    out_w, out_h = max(1, round(width * factor)), max(1, round(height * factor))
    m = np.diag([out_w / width, out_h / height, 1.0])
    return Transform(m, out_w, out_h, False, f"scale{factor:g}")


def rotate(width: int, height: int, degrees: float) -> Transform:
    """화면 가운데를 중심으로 반시계 회전 (출력 크기 같음, 모서리는 잘리고 빈 곳은 채움)."""
    m = np.vstack([cv2.getRotationMatrix2D((width / 2, height / 2), degrees, 1.0), [0, 0, 1]])
    return Transform(m.astype(np.float64), width, height, False, f"rot{degrees:g}")


def crop_resize(width: int, height: int, box: Box) -> Transform:
    """원본의 box(x, y, w, h) 영역을 원본 크기로 늘린다 (자르기 + 배율)."""
    x, y, w, h = box
    m = np.array([[width / w, 0.0, -x * width / w], [0.0, height / h, -y * height / h], [0, 0, 1]])
    return Transform(m.astype(np.float64), width, height, False, "crop")


def perspective(width: int, height: int, offsets: NDArray[np.float64]) -> Transform:
    """네 모서리를 offsets(4x2 픽셀, 왼위·오위·오아·왼아 순)만큼 옮기는 원근 변환."""
    src = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    dst = (src + offsets.astype(np.float32)).astype(np.float32)
    m = cv2.getPerspectiveTransform(src, dst).astype(np.float64)
    return Transform(m, width, height, False, "persp")


# 연속 픽셀 좌표 = cv2 화소 좌표 + 0.5
_HALF = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]])


def cv_matrix(t: Transform) -> Matrix:
    """cv2 warp 함수에 넘길 행렬 (화소 중심 = 정수 규약으로 바꾼 것, F-PMS-37).

    연속 좌표 p = c + 0.5(c는 cv2 화소 좌표)이므로 cv2 행렬은 H⁻¹·M·H (H = 0.5 이동)다.
    """
    return np.linalg.inv(_HALF) @ t.matrix @ _HALF


def warp_image(
    image: NDArray[Any], t: Transform, *, border: int = 0, nearest: bool = False
) -> NDArray[Any]:
    """이미지에 변환을 적용한다 (선형 보간, nearest면 최근접(마스크용), 빈 곳은 border 값)."""
    return cv2.warpPerspective(
        image, cv_matrix(t), (t.out_w, t.out_h),
        flags=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=border,
    )  # fmt: skip


def map_points(t: Transform, points: NDArray[np.float64]) -> NDArray[np.float64]:
    """(N, 2) 픽셀 점 → 변환 픽셀 점."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    h = np.hstack([pts, np.ones((pts.shape[0], 1))]) @ t.matrix.T
    return h[:, :2] / h[:, 2:3]


def map_box(t: Transform, box: Box) -> Box:
    """박스 네 모서리를 옮긴 뒤의 외접 박스 (x, y, w, h). 회전·원근이면 원래 박스보다 커진다."""
    x, y, w, h = box
    corners = map_points(t, np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]))
    x0, y0 = corners.min(axis=0)
    x1, y1 = corners.max(axis=0)
    return (float(x0), float(y0), float(x1 - x0), float(y1 - y0))


def clip_box(box: Box, width: int, height: int) -> Box | None:
    """박스를 화면 안으로 자른다. 남는 넓이가 없으면 None."""
    x, y, w, h = box
    x0, y0 = max(0.0, x), max(0.0, y)
    x1, y1 = min(float(width), x + w), min(float(height), y + h)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1 - x0, y1 - y0)


def inside(t: Transform, points: NDArray[np.float64]) -> NDArray[np.bool_]:
    """변환한 점이 출력 화면 안인가 (점마다)."""
    p = map_points(t, points)
    return (p[:, 0] >= 0) & (p[:, 0] < t.out_w) & (p[:, 1] >= 0) & (p[:, 1] < t.out_h)


def swap_hand(hand: str, t: Transform) -> str:
    """좌우 뒤집힌 변환이면 손 판정("left"/"right")을 바꾼다."""
    if not t.flipped:
        return hand
    return {"left": "right", "right": "left"}.get(hand, hand)


def iou(a: Box, b: Box) -> float:
    """두 박스 IoU."""
    ax1, ay1, bx1, by1 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax1, bx1) - max(a[0], b[0]))
    ih = max(0.0, min(ay1, by1) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def tta_support(
    boxes: list[Box],
    views: list[tuple[Transform, list[Box]]],
    width: int,
    height: int,
    iou_match: float,
) -> list[float]:
    """원본 탐지 박스마다, 변환한 화면들에서도 같은 자리에 탐지가 있는 비율 (0~1).

    Args:
        boxes: 원본 화면 탐지 박스.
        views: (변환, 그 변환 화면에서의 탐지 박스) 목록. 탐지 박스는 역변환으로 원본 좌표에 되돌려
            비교한다.
        width, height: 원본 크기.
        iou_match: 같은 탐지로 볼 IoU 하한.

    Returns:
        박스마다 일치한 변환 수 / 변환 수. 변환이 없으면 모두 1.0 (판단하지 않음).
    """
    if not views:
        return [1.0] * len(boxes)
    back = [
        [map_box(t.inverse(width, height), b) for b in found] for t, found in views
    ]  # fmt: skip
    return [
        sum(any(iou(box, b) >= iou_match for b in view) for view in back) / len(views)
        for box in boxes
    ]


def angle_deg(t: Transform) -> float:
    """변환의 대략적인 회전 각도 (도, 기록용)."""
    return math.degrees(math.atan2(t.matrix[1, 0], t.matrix[0, 0]))


class TtaSpec(Contract):
    """테스트 시점 증강 설정 (프라이버시·도구 힌트 정책 공용, ADR 0055).

    transforms: 변환 이름 목록. "hflip"(좌우 뒤집기), "scale:<배율>"(예: "scale:0.75"),
        "rot:<도>"(반시계 회전). 원본 화면은 늘 따로 보므로 넣지 않는다.
    iou: 역변환한 탐지가 원본 박스와 같은 탐지로 볼 IoU 하한.
    max_tracks: 한 스트림에서 TTA를 돌릴 트랙 수 상한 (신뢰도 낮은 트랙부터, 고르는 일은 호출자
        `dlp_cli.hints_cmds`가 한다). 0이면 돌리지 않는다.
    frames_per_track: 트랙마다 볼 키프레임 수 (트랙 안에서 고르게).
    """

    transforms: tuple[str, ...] = Field(min_length=1)
    iou: float = Field(gt=0, le=1)
    max_tracks: int = Field(ge=0)
    frames_per_track: int = Field(ge=1)


def parse_transform(spec: str, width: int, height: int) -> Transform:
    """변환 이름 → 변환 ("hflip", "scale:<배율>", "rot:<도>"). 모르는 이름이면 ValueError."""
    name, _, arg = spec.partition(":")
    if name == "hflip" and not arg:
        return hflip(width, height)
    if name == "scale" and arg:
        return scale(width, height, float(arg))
    if name == "rot" and arg:
        return rotate(width, height, float(arg))
    raise ValueError(f"모르는 TTA 변환: {spec}")


# TTA를 돌릴 트랙: (라벨 ID, 탐지 묶음 키(대상·클래스), [(PTS ms, 박스)])
TtaTrack = tuple[str, str, list[tuple[int, Box]]]
Detector = Callable[[NDArray[np.uint8]], list[tuple[str, Box]]]


def tta_tracks(
    video: Path, tracks: Sequence[TtaTrack], detect: Detector, spec: TtaSpec
) -> dict[str, float]:
    """트랙마다 TTA 지지 비율 평균 (키프레임마다 변환 화면에서 같은 키의 탐지가 같은 자리에 있는가).

    영상은 PTS 인덱스로 필요한 프레임만 한 번 순차 디코드한다 (`frames.decode_frames`, 원본 크기).
    키프레임 시각에 가장 가까운 프레임을 쓴다. 트랙 수·키프레임 수는 호출자가 spec에 맞춰 고른다.

    Returns:
        라벨 ID → 0~1 (키프레임 평균). 변환이 없으면 1.
    """
    from dlp_media.frames import decode_frames
    from dlp_media.pts import build_pts_index

    pts = build_pts_index(video)
    wanted: dict[int, list[tuple[str, str, Box]]] = {}
    for label_id, key, frames in tracks:
        for t_ms, box in frames:
            wanted.setdefault(pts.nearest(t_ms), []).append((label_id, key, box))
    scores: dict[str, list[float]] = {}
    for idx, _ms, img in decode_frames(video, pts, wanted, max_side=1 << 16):
        h, w = img.shape[:2]
        views = [
            (t, detect(warp_image(img, t)))
            for t in (parse_transform(s, w, h) for s in spec.transforms)
        ]
        for label_id, key, box in wanted[idx]:
            same = [(t, [b for k, b in found if k == key]) for t, found in views]
            scores.setdefault(label_id, []).append(tta_support([box], same, w, h, spec.iou)[0])
    return {k: sum(v) / len(v) for k, v in scores.items()}
