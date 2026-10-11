"""판정 프레임에 표식·시각을 그리고 확대 조각을 만든다 (WP17.2, ADR 0040 2.3).

- `draw_stamp`: 왼쪽 위 모서리에 `F3 +1.2s` (프레임 번호와 증거 구간 기준 상대 시각, ASCII).
- `draw_marks`: set-of-marks. 공간 대상 트랙에 고대비 테두리(검은 바깥선 + 밝은 안쪽 실선)와 태그
  (`B3`, `H1 R`, `P2`)를 그린다. 블러 모자이크(사각 픽셀 덩어리)와 모양이 다르다. 손 키포인트는 얇은
  골격선, 전신은 점과 외접 상자.
- `crop`: 작은 박스 주변을 잘라 정사각형으로 키운다 (원 해상도 프레임에서).
- `box_at`, `points_at`: 키프레임 선형 보간 (outside를 지킨다).
- `render_unit_images`: 단위의 원 해상도 프레임 → 보낼 이미지(`PromptImage`) 목록.

시간 라벨은 그리지 않는다 (VLM이 화면 글자에 끌리지 않게, 같은 창의 분할 요청끼리 이미지 캐시가
맞게). 입력 픽셀은 승인된 블러본에서만 온다 (`dlp_aireview.media`, WP17.5). 좌표는 그 스트림 영상
픽셀이다 (블러본은 원본과 같은 해상도로 렌더한다).
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from dlp_aireview.plan import PlannedUnit
from dlp_aireview.policy import AiReviewPolicy
from dlp_aireview.prompt import PromptImage, mark_tag
from dlp_media.frames import fit_size
from dlp_models.vlm import encode_jpeg
from dlp_schema.aireview import AiGroup
from dlp_schema.labels import BoxKeyframe, BoxTrackPayload, Hand, KeypointTrackPayload, LabelRecord

Image = NDArray[np.uint8]
Box = tuple[float, float, float, float]
# 표식 색 (RGB): 밝은 안쪽 선과 검은 바깥선
_MARK = (255, 230, 0)
_OUTLINE = (0, 0, 0)
# MediaPipe hand21 골격 연결 (손목 0, 손가락마다 4관절)
HAND_EDGES = tuple(
    [(0, 1 + 4 * f) for f in range(5)]
    + [(1 + 4 * f + k, 2 + 4 * f + k) for f in range(5) for k in range(3)]
)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_stamp(img: Image, text: str) -> Image:
    """왼쪽 위에 검은 바탕 흰 글자 (ASCII). 원본을 바꾸지 않고 사본을 돌려준다."""
    out = img.copy()
    scale = max(0.4, min(out.shape[:2]) / 600)
    (w, h), base = cv2.getTextSize(text, _FONT, scale, 1)
    cv2.rectangle(out, (0, 0), (w + 6, h + base + 6), _OUTLINE, -1)
    cv2.putText(out, text, (3, h + 3), _FONT, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _tag(img: Image, text: str, x: int, y: int) -> None:
    """표식 태그 (밝은 바탕 검은 글자, 상자 위쪽)."""
    scale = max(0.4, min(img.shape[:2]) / 700)
    (w, h), base = cv2.getTextSize(text, _FONT, scale, 1)
    top = max(0, y - h - base - 4)
    cv2.rectangle(img, (x, top), (x + w + 4, top + h + base + 4), _MARK, -1)
    cv2.putText(img, text, (x + 2, top + h + 2), _FONT, scale, _OUTLINE, 1, cv2.LINE_AA)


def _rect(img: Image, box: Box) -> tuple[int, int, int, int]:
    """고대비 테두리 (바깥 검은 3px + 안쪽 밝은 1px). 정수 좌표를 돌려준다."""
    x, y, w, h = (round(v) for v in box)
    cv2.rectangle(img, (x, y), (x + w, y + h), _OUTLINE, 3)
    cv2.rectangle(img, (x, y), (x + w, y + h), _MARK, 1)
    return x, y, w, h


def draw_marks(
    img: Image, marks: Sequence[tuple[str, Box | None, Sequence[tuple[float, float]] | None]]
) -> Image:
    """표식을 그린 사본.

    marks = (태그, 상자 또는 None, 키포인트 점 또는 None) 목록 (보낸 프레임 좌표). 점이 21개면
    손 골격선을 잇는다. 상자가 없고 점만 있으면 외접 상자를 쓴다.
    """
    out = img.copy()
    for tag, box, points in marks:
        if points:
            pts = [(round(px), round(py)) for px, py in points]
            if len(pts) == 21:
                for a, b in HAND_EDGES:
                    cv2.line(out, pts[a], pts[b], _MARK, 1, cv2.LINE_AA)
            for p in pts:
                cv2.circle(out, p, 2, _OUTLINE, -1)
                cv2.circle(out, p, 1, _MARK, -1)
            if box is None:
                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                box = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
        if box is not None:
            x, y, _, _ = _rect(out, box)
            _tag(out, tag, max(0, x), max(0, y))
    return out


def crop(img: Image, box: Box, side: int) -> Image:
    """상자 주변(상자 크기의 2배, 최소 16px)을 잘라 한 변 `side`의 정사각형으로 키운다.

    비율을 지키고 모자라는 쪽은 검은 여백이다.
    """
    x, y, w, h = box
    cx, cy, half = x + w / 2, y + h / 2, max(w, h, 16.0)
    x0, y0 = max(0, round(cx - half)), max(0, round(cy - half))
    x1, y1 = min(img.shape[1], round(cx + half)), min(img.shape[0], round(cy + half))
    piece = img[y0:y1, x0:x1]
    if piece.size == 0:
        return np.zeros((side, side, 3), dtype=np.uint8)
    scale = side / max(piece.shape[:2])
    resized = cv2.resize(
        piece, (max(1, round(piece.shape[1] * scale)), max(1, round(piece.shape[0] * scale))),
        interpolation=cv2.INTER_LINEAR,
    )  # fmt: skip
    out = np.zeros((side, side, 3), dtype=np.uint8)
    out[: resized.shape[0], : resized.shape[1]] = resized[:side, :side]
    return out


def box_at(keyframes: Sequence[BoxKeyframe], t_ms: float) -> Box | None:
    """시각 t의 상자 (앞뒤 키프레임 선형 보간). 첫 키프레임 전·마지막 뒤·outside 구간이면 None."""
    if not keyframes or t_ms < keyframes[0].t_ms or t_ms > keyframes[-1].t_ms:
        return None
    prev = max((k for k in keyframes if k.t_ms <= t_ms), key=lambda k: k.t_ms)
    if prev.outside:
        return None
    nxt = next((k for k in keyframes if k.t_ms > t_ms), None)
    if nxt is None or nxt.outside or nxt.t_ms == prev.t_ms:
        return prev.x, prev.y, prev.w, prev.h
    a = (t_ms - prev.t_ms) / (nxt.t_ms - prev.t_ms)
    return (
        prev.x + a * (nxt.x - prev.x), prev.y + a * (nxt.y - prev.y),
        prev.w + a * (nxt.w - prev.w), prev.h + a * (nxt.h - prev.h),
    )  # fmt: skip


def points_at(p: KeypointTrackPayload, t_ms: float) -> list[tuple[float, float]] | None:
    """시각 t의 보이는 키포인트 (앞뒤 키프레임 선형 보간). 구간 밖이면 None."""
    kfs = p.keyframes
    if t_ms < kfs[0].t_ms or t_ms > kfs[-1].t_ms:
        return None
    prev = max((k for k in kfs if k.t_ms <= t_ms), key=lambda k: k.t_ms)
    nxt = next((k for k in kfs if k.t_ms > t_ms), prev)
    a = 0.0 if nxt.t_ms == prev.t_ms else (t_ms - prev.t_ms) / (nxt.t_ms - prev.t_ms)
    return [
        (q.x + a * (r.x - q.x), q.y + a * (r.y - q.y))
        for q, r in zip(prev.points, nxt.points, strict=True)
        if q.visibility > 0
    ]


def _scaled(box: Box | None, sx: float, sy: float) -> Box | None:
    """원 좌표 상자 → 보낸 프레임 좌표."""
    return None if box is None else (box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy)


def _target_mark(
    x: LabelRecord, n: int, t_ms: int, sx: float, sy: float
) -> tuple[str, Box | None, list[tuple[float, float]] | None, Box | None]:
    """공간 대상 하나의 (태그, 보낸 좌표 상자, 보낸 좌표 점, 원 좌표 상자)."""
    p = x.payload
    if isinstance(p, BoxTrackPayload):
        raw = box_at(p.keyframes, t_ms)
        return mark_tag(x, n), _scaled(raw, sx, sy), None, raw
    if isinstance(p, KeypointTrackPayload):
        pts = points_at(p, t_ms)
        side = " R" if p.hand is Hand.RIGHT else " L" if p.hand is Hand.LEFT else ""
        scaled = None if pts is None else [(px * sx, py * sy) for px, py in pts]
        return mark_tag(x, n) + side, None, scaled, None
    return mark_tag(x, n), None, None, None


def render_unit_images(
    unit: PlannedUnit, frames: Sequence[Image], policy: AiReviewPolicy
) -> tuple[PromptImage, ...]:
    """단위의 원 해상도 프레임(RGB, `unit.frame_idx` 순) → 보낼 이미지 (프레임 F0… 다음 확대 조각).

    프레임은 긴 변 상한·정렬 단위(`image.align_px`)로 줄인다. temporal은 시각 표기만, spatial은 시각
    표기 + 대상 표식을 그린다. 작은 박스(보낸 프레임에서 짧은 변 < `crops.min_box_side_px`)는 원
    해상도에서 확대 조각을 만든다 (요청당 `crops.max`개, 대상마다 처음 보이는 프레임에서 하나).

    Raises:
        ValueError: 프레임 수가 단위 프레임 수와 다를 때.
    """
    if len(frames) != len(unit.frame_ms):
        raise ValueError("프레임 수가 단위 프레임 수와 다릅니다")
    start = unit.span[0]
    spatial = unit.group is AiGroup.SPATIAL
    cp = policy.spatial.crops
    quality = policy.image.jpeg_quality
    images: list[PromptImage] = []
    crops: list[PromptImage] = []
    cropped: set[int] = set()
    for k, (img, t_ms) in enumerate(zip(frames, unit.frame_ms, strict=True)):
        h0, w0 = img.shape[:2]
        w, h = fit_size(w0, h0, unit.window.max_side_px, policy.image.align_px)
        small: Image = np.asarray(cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA), np.uint8)
        sx, sy = w / w0, h / h0
        if spatial and policy.spatial.marks:
            marks = [_target_mark(x, n, t_ms, sx, sy) for n, x in enumerate(unit.targets)]
            small = draw_marks(small, [(tag, box, pts) for tag, box, pts, _ in marks])
            for n, (tag, box, _, raw) in enumerate(marks):
                tiny = box is not None and min(box[2], box[3]) < cp.min_box_side_px
                if (
                    cp.enabled
                    and tiny
                    and raw is not None
                    and n not in cropped
                    and len(crops) < cp.max
                ):
                    cropped.add(n)
                    jpeg = encode_jpeg(crop(img, raw, cp.side_px), quality)
                    crops.append(
                        PromptImage(jpeg, t_ms, cp.side_px, cp.side_px, crop_of=k, label=tag)
                    )
        if policy.temporal.timestamp_overlay or spatial:
            small = draw_stamp(small, f"F{k} +{(t_ms - start) / 1000:.1f}s")
        images.append(PromptImage(encode_jpeg(small, quality), t_ms, w, h))
    return (*images, *crops)
