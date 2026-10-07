"""접촉 구간 추정 (모델이 아니라 알고리즘).

- 장갑: 압력 합을 히스테리시스 문턱으로 나눈다. 시작은 on 문턱을 넘은 뒤 off 아래였던 마지막 샘플
  다음으로 되짚고, 끝은 off 아래로 내려간 첫 샘플이다 (장갑 접촉 시각이 접촉 모델의 정답 신호다).
- 영상: 손가락 끝 다섯 점과 객체 박스 사이 최소 거리가 문턱 이하인 프레임을 접촉 후보로 본다.
  손 키프레임 시각에 박스 키프레임이 없으면 앞뒤 박스 키프레임을 선형 보간한다 (간격이
  box_max_gap_ms 이하일 때만). 도구 박스는 프레임마다가 아니라 frame_stride_ms마다 나온다.
- 융합: 장갑 구간의 시각을 쓰고, 대상은 그 구간과 가장 많이 겹치는 영상 구간의 객체로 정한다.
  장갑 구간과 겹치지 않는 영상 구간은 영상 출처(낮은 신뢰도)로 그대로 낸다 (장갑이 놓친 접촉일 수
  있어 검수자가 먼저 본다).
"""

from __future__ import annotations

import bisect
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


Box = tuple[float, float, float, float]


def box_at(
    track: BoxTrackPayload, t_ms: int, max_gap_ms: float, times: list[int] | None = None
) -> Box | None:
    """t_ms의 박스. 키프레임이 없으면 앞뒤 키프레임을 선형 보간한다.

    앞뒤 중 하나가 화면 밖(outside)이거나 간격이 max_gap_ms보다 길면 없음.
    times: 키프레임 시각 목록 (여러 번 부를 때 미리 만들어 넘긴다).
    """
    frames = track.keyframes
    i = bisect.bisect_left(times if times is not None else [k.t_ms for k in frames], t_ms)
    if i < len(frames) and frames[i].t_ms == t_ms:
        k = frames[i]
        return None if k.outside else (k.x, k.y, k.w, k.h)
    if i == 0 or i == len(frames):
        return None
    a, b = frames[i - 1], frames[i]
    if a.outside or b.outside or b.t_ms - a.t_ms > max_gap_ms:
        return None
    r = (t_ms - a.t_ms) / (b.t_ms - a.t_ms)
    return (
        a.x + (b.x - a.x) * r,
        a.y + (b.y - a.y) * r,
        a.w + (b.w - a.w) * r,
        a.h + (b.h - a.h) * r,
    )


def video_contact_intervals(
    hand: KeypointTrackPayload, objects: list[BoxTrackPayload], policy: VideoContactPolicy
) -> list[ContactInterval]:
    tracks = [(obj, [k.t_ms for k in obj.keyframes]) for obj in objects if obj.keyframes]
    per_frame: list[tuple[int, str | None]] = []
    for f in hand.keyframes:
        best: tuple[float, str] | None = None
        for obj, times in tracks:
            if f.t_ms < times[0] or f.t_ms > times[-1]:
                continue
            box = box_at(obj, f.t_ms, policy.box_max_gap_ms, times)
            if box is None:
                continue
            entity = obj.entity_id
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


def _overlap(a: ContactInterval, b: ContactInterval) -> int:
    return min(a.end_ms, b.end_ms) - max(a.start_ms, b.start_ms)


def fuse_contacts(
    glove: list[ContactInterval], video: list[ContactInterval]
) -> list[ContactInterval]:
    """장갑 구간마다 대상을 붙이고, 어느 장갑 구간과도 겹치지 않는 영상 구간은 영상 출처로 낸다."""
    out: list[ContactInterval] = []
    for g in glove:
        overlaps = [(_overlap(g, v), v.target_id) for v in video if _overlap(g, v) > 0]
        target = max(overlaps)[1] if overlaps else None
        out.append(ContactInterval(g.start_ms, g.end_ms, target, "fused" if target else "glove"))
    out += [v for v in video if not any(_overlap(g, v) > 0 for g in glove)]
    return sorted(out, key=lambda c: (c.start_ms, c.end_ms))
