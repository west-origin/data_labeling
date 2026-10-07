"""접촉 구간 추정 (모델이 아니라 알고리즘).

- 장갑: 압력 합을 히스테리시스 문턱으로 나눈다. 시작은 on 문턱을 넘은 뒤 off 아래였던 마지막 샘플
  다음으로 되짚고, 끝은 off 아래로 내려간 첫 샘플이다 (장갑 접촉 시각이 접촉 모델의 정답 신호다).
- 영상: 손가락 끝 다섯 점과 객체 박스 사이 최소 거리가 문턱 이하인 프레임을 접촉 후보로 본다.
- 융합: 장갑 구간의 시각을 쓰고, 대상은 그 구간과 가장 많이 겹치는 영상 구간의 객체로 정한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from dlp_prelabel.policy import GloveContactPolicy, VideoContactPolicy
from dlp_schema.labels import BoxTrackPayload, KeypointTrackPayload

FINGERTIPS = (4, 8, 12, 16, 20)


@dataclass(frozen=True)
class ContactInterval:
    start_ms: int
    end_ms: int
    target_id: str | None
    source: str  # glove | video | fused


def glove_contact_intervals(
    t_ms: NDArray[np.float64], pressure: NDArray[np.float64], policy: GloveContactPolicy
) -> list[ContactInterval]:
    out: list[ContactInterval] = []
    i, n = 0, t_ms.size
    while i < n:
        if pressure[i] <= policy.on_threshold:
            i += 1
            continue
        start = i
        while start > 0 and pressure[start - 1] > policy.off_threshold:
            start -= 1
        end = i
        while end < n and pressure[end] >= policy.off_threshold:
            end += 1
        s_ms, e_ms = float(t_ms[start]), float(t_ms[min(end, n - 1)])
        if e_ms - s_ms >= policy.min_duration_ms:
            out.append(ContactInterval(round(s_ms), round(e_ms), None, "glove"))
        i = end + 1
    return out


def _box_distance(px: float, py: float, box: tuple[float, float, float, float]) -> float:
    x, y, w, h = box
    dx = max(x - px, 0.0, px - (x + w))
    dy = max(y - py, 0.0, py - (y + h))
    return float(np.hypot(dx, dy))


def video_contact_intervals(
    hand: KeypointTrackPayload, objects: list[BoxTrackPayload], policy: VideoContactPolicy
) -> list[ContactInterval]:
    boxes: dict[int, list[tuple[str, tuple[float, float, float, float]]]] = {}
    for obj in objects:
        for k in obj.keyframes:
            if not k.outside:
                boxes.setdefault(k.t_ms, []).append((obj.entity_id, (k.x, k.y, k.w, k.h)))
    per_frame: list[tuple[int, str | None]] = []
    for f in hand.keyframes:
        best: tuple[float, str] | None = None
        for entity, box in boxes.get(f.t_ms, []):
            d = min(_box_distance(f.points[i].x, f.points[i].y, box) for i in FINGERTIPS)
            if d <= policy.max_distance_px and (best is None or d < best[0]):
                best = (d, entity)
        per_frame.append((f.t_ms, best[1] if best else None))

    raw: list[ContactInterval] = []
    for t, target in per_frame:
        if target is None:
            continue
        if raw and raw[-1].target_id == target and t - raw[-1].end_ms <= policy.merge_gap_ms:
            raw[-1] = ContactInterval(raw[-1].start_ms, t, target, "video")
        else:
            raw.append(ContactInterval(t, t, target, "video"))
    return [c for c in raw if c.end_ms - c.start_ms >= policy.min_duration_ms]


def fuse_contacts(
    glove: list[ContactInterval], video: list[ContactInterval]
) -> list[ContactInterval]:
    out: list[ContactInterval] = []
    for g in glove:
        overlaps = [
            (min(g.end_ms, v.end_ms) - max(g.start_ms, v.start_ms), v.target_id)
            for v in video
            if min(g.end_ms, v.end_ms) > max(g.start_ms, v.start_ms)
        ]
        target = max(overlaps)[1] if overlaps else None
        out.append(ContactInterval(g.start_ms, g.end_ms, target, "fused" if target else "glove"))
    return out
