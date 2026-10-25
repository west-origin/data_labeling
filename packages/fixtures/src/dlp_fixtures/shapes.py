"""정답 모양을 아는 합성 영상: 객체·손 마스크 단계(ADR 0059) 시험용 (WP2).

1인칭 화면처럼 객체 둘(움직이는 타원 컵, 기운 사각형 양동이)과 화면 아래에서 들어오는 착용자
아래팔(큰 사람 박스로 잡힌다), 그 끝의 왼손(손바닥 타원 + 손가락 다섯)을 그린다. 정답:

- `masks`: 개체 → (PTS ms → (H, W) 불리언 정답 마스크). 손은 `left_hand`, 착용자 팔은 `wearer_arm`
  (마스크로 나오면 안 되는 대상).
- `labels`: 프롬프트가 될 모델 출처 라벨. 객체 박스 트랙(정답 마스크 외접 박스), 착용자 팔을 덮은
  person 박스 트랙(착용자 몸 판정에 걸린다, ADR 0058), 손 hand21 트랙(관절이 손 모양 안에 있다).

객체가 손과 겹치지 않게 두어 정답 마스크끼리 겹치지 않는다. 키프레임 시각은 그 영상의 PTS ms다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from dlp_fixtures.video import write_video
from dlp_schema.labels import (
    BoxKeyframe,
    BoxTrackPayload,
    Hand,
    Keypoint,
    KeypointFrame,
    KeypointTrackPayload,
    LabelPayload,
    LabelRecord,
    Provenance,
    Source,
)
from dlp_schema.testing import FIXED_TIME

Image = NDArray[np.uint8]
Mask = NDArray[np.bool_]
# (개체, 클래스, RGB 색)
OBJECTS = (("cup_01", "cup", (220, 60, 40)), ("bucket_01", "bucket", (40, 90, 220)))
WEARER_COLOR = (205, 170, 140)  # 착용자 소매·팔
HAND_COLOR = (235, 190, 160)  # 손
BACKGROUND = (110, 120, 110)


@dataclass
class ShapeScenario:
    """정답 모양 영상과 프롬프트 라벨."""

    width: int
    height: int
    frames: list[tuple[int, Image]]
    masks: dict[str, dict[int, Mask]] = field(default_factory=dict[str, dict[int, Mask]])
    labels: list[LabelRecord] = field(default_factory=list[LabelRecord])

    @property
    def frame_times(self) -> list[int]:
        """프레임 PTS ms."""
        return [t for t, _ in self.frames]

    def write(self, path: Path) -> None:
        """영상을 MP4로 쓴다 (PTS 그대로)."""
        write_video(path, iter(self.frames), width=self.width, height=self.height)


def _bbox(mask: Mask) -> tuple[float, float, float, float]:
    """정답 마스크 외접 박스 (x, y, w, h)."""
    ys, xs = np.nonzero(mask)
    return (float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1),
            float(ys.max() - ys.min() + 1))  # fmt: skip


def _hand_points(cx: float, cy: float, s: float) -> list[tuple[float, float]]:
    """손 21관절 (마스크는 `_draw_hand`가 이 점으로 그린다).

    손목 (cx, cy+s), 손바닥 중심 (cx, cy), 손가락은 위로 퍼진다. 관절 순서는 MediaPipe hand21.
    """
    pts: list[tuple[float, float]] = [(cx, cy + s)]  # 0 손목
    for k, angle in enumerate((-60.0, -25.0, -5.0, 15.0, 35.0)):  # 엄지 → 새끼
        a = np.deg2rad(angle - 90.0)
        base = (cx + 0.6 * s * np.cos(a), cy + 0.6 * s * np.sin(a))
        length = s * (0.9 if k == 0 else 1.2)
        for j in range(1, 5):  # 뿌리 → 끝 4점
            r = (j - 1) / 3.0
            pts.append((base[0] + r * length * np.cos(a), base[1] + r * length * np.sin(a)))
    return pts


def _draw_hand(size: tuple[int, int], pts: list[tuple[float, float]], s: float) -> Mask:
    """손 정답 마스크: 손바닥 타원 + 손가락 굵은 선."""
    w, h = size
    m = np.zeros((h, w), np.uint8)
    cx, cy = pts[0][0], pts[0][1] - s
    cv2.ellipse(m, (round(cx), round(cy)), (round(0.75 * s), round(0.95 * s)), 0, 0, 360, 1, -1)
    for k in range(5):
        a, b = pts[1 + 4 * k], pts[4 + 4 * k]
        p0, p1 = (round(a[0]), round(a[1])), (round(b[0]), round(b[1]))
        cv2.line(m, p0, p1, 1, max(3, round(0.28 * s)))
    return m.astype(bool)


def _label(payload: LabelPayload, label_id: str, times: list[int]) -> LabelRecord:
    """모델 출처 프롬프트 라벨 (세션 shapes, 스트림 bodycam, 신뢰도 0.9)."""
    return LabelRecord(
        label_id=label_id, session_id="shapes", stream_id="bodycam",
        t_start_ms=times[0], t_end_ms=times[-1], ontology_version="1.0.0",
        provenance=Provenance(source=Source.MODEL, model_version="fixture-shapes"),
        confidence=0.9, created_at=FIXED_TIME, payload=payload,
    )  # fmt: skip


def generate_shape_scenario(
    seed: int = 0, *, n_frames: int = 12, width: int = 320, height: int = 240
) -> ShapeScenario:
    """정답 모양 영상을 만든다 (프레임 간격 33~34 ms, 객체는 프레임마다 조금씩 움직인다)."""
    rng = np.random.default_rng(seed)
    times = [round(i * 1000 / 30) for i in range(n_frames)]
    sc = ShapeScenario(width, height, [])
    boxes: dict[str, list[BoxKeyframe]] = {e: [] for e, _, _ in OBJECTS}
    boxes["wearer_arm"] = []
    hand_frames: list[KeypointFrame] = []
    for e in [*boxes, "left_hand"]:
        sc.masks[e] = {}
    s = 0.07 * height
    for i, t in enumerate(times):
        img = np.empty((height, width, 3), np.uint8)
        img[:] = BACKGROUND
        noise = rng.integers(0, 12, img.shape, dtype=np.uint8)
        img = np.asarray(cv2.add(img, noise), dtype=np.uint8)
        cup = np.zeros((height, width), np.uint8)
        cv2.ellipse(cup, (round(0.22 * width + 2 * i), round(0.3 * height)),
                    (round(0.09 * width), round(0.07 * height)), 0, 0, 360, 1, -1)  # fmt: skip
        bucket = np.zeros((height, width), np.uint8)
        rect = ((0.75 * width - i, 0.3 * height), (0.14 * width, 0.22 * height), 20.0)
        cv2.fillPoly(bucket, [np.round(cv2.boxPoints(rect)).astype(np.int32)], 1)
        arm = np.zeros((height, width), np.uint8)
        cv2.rectangle(arm, (round(0.42 * width), round(0.68 * height)),
                      (round(0.62 * width), height - 1), 1, -1)  # fmt: skip
        pts = _hand_points(0.52 * width + i, 0.62 * height, s)
        hand = _draw_hand((width, height), pts, s)
        for e, m in (("cup_01", cup), ("bucket_01", bucket), ("wearer_arm", arm)):
            sc.masks[e][t] = m.astype(bool)
        for (_, _, color), m in zip(OBJECTS, (cup, bucket), strict=True):
            img[m > 0] = color
        img[arm > 0] = WEARER_COLOR
        img[hand] = HAND_COLOR
        sc.masks["left_hand"][t] = hand
        sc.frames.append((t, img))
        for e in boxes:
            x, y, bw, bh = _bbox(sc.masks[e][t])
            if e == "wearer_arm":  # 사람 박스는 팔과 그 끝 손까지 덮는다 (COCO person이 잡는 모양)
                hx, hy, hw, _ = _bbox(hand)
                y, bh = min(y, hy), y + bh - min(y, hy)
                x, bw = min(x, hx), max(x + bw, hx + hw) - min(x, hx)
            boxes[e].append(BoxKeyframe(t_ms=t, x=x, y=y, w=bw, h=bh))
        hand_frames.append(
            KeypointFrame(
                t_ms=t, points=tuple(Keypoint(x=px, y=py, visibility=2) for px, py in pts)
            )
        )
    classes = {e: c for e, c, _ in OBJECTS} | {"wearer_arm": "person"}
    for e, kfs in boxes.items():
        payload = BoxTrackPayload(entity_id=e, class_id=classes[e], keyframes=tuple(kfs))
        sc.labels.append(_label(payload, f"shapes-box-{e}", times))
    hand_payload = KeypointTrackPayload(
        entity_id="left_hand", skeleton="hand21", hand=Hand.LEFT, keyframes=tuple(hand_frames)
    )
    sc.labels.append(_label(hand_payload, "shapes-hand-left", times))
    return sc
