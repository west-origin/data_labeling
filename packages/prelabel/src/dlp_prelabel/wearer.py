"""3인칭 영상 속 바디캠 착용자 찾기 (알고리즘).

동기화된 마스터 시각에서 바디캠 쪽 운동 신호(바디캠 IMU 가속도 크기 또는 1인칭 손목 속도)와
3인칭 각 인물의 손목 속도(두 손목 중 큰 값)를 같은 격자로 리샘플해 피어슨 상관을 구한다.
상관이 가장 높고 min_correlation 이상인 인물을 착용자로 본다. 검수자가 확인한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from dlp_schema.labels import KeypointTrackPayload

COCO_WRISTS = (9, 10)


@dataclass(frozen=True)
class WearerMatch:
    entity_id: str | None
    correlation: float
    scores: dict[str, float]


def wrist_speed(
    track: KeypointTrackPayload, offset_ms: float = 0.0
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """(마스터 시각, 손목 속도 px/s). offset_ms는 스트림 → 마스터 오프셋."""
    t = np.array([f.t_ms for f in track.keyframes], dtype=float) + offset_ms
    xy = np.array([[[f.points[i].x, f.points[i].y] for i in COCO_WRISTS] for f in track.keyframes])
    if t.size < 3:
        return t, np.zeros(t.size)
    v = np.linalg.norm(np.diff(xy, axis=0), axis=2).max(axis=1) / np.diff(t) * 1000
    return (t[1:] + t[:-1]) / 2, v


def match_wearer(
    reference_t: NDArray[np.float64],
    reference_v: NDArray[np.float64],
    people: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]],
    *,
    rate_hz: float,
    min_correlation: float,
) -> WearerMatch:
    scores: dict[str, float] = {}
    for entity, (t, v) in people.items():
        lo, hi = max(reference_t[0], t[0]), min(reference_t[-1], t[-1])
        if hi - lo < 2000 / rate_hz * 10:
            continue
        grid = np.arange(lo, hi, 1000 / rate_hz)
        a = np.interp(grid, reference_t, reference_v)
        b = np.interp(grid, t, v)
        if a.std() == 0 or b.std() == 0:
            continue
        scores[entity] = float(np.corrcoef(a, b)[0, 1])
    if not scores:
        return WearerMatch(None, 0.0, {})
    best = max(scores, key=lambda e: scores[e])
    if scores[best] < min_correlation:
        return WearerMatch(None, scores[best], scores)
    return WearerMatch(best, scores[best], scores)
