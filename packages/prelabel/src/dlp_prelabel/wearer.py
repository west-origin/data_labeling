"""3인칭 영상 속 바디캠 착용자 찾기 (알고리즘).

동기화된 마스터 시각에서 바디캠 쪽 운동 신호(바디캠 IMU 가속도 크기 또는 1인칭 손목 속도)와 3인칭 각
인물의 손목 속도(두 손목 중 큰 값)를 같은 격자로 리샘플해 피어슨 상관을 구한다. 상관이 가장 높고
min_correlation 이상인 인물을 착용자로 본다. 검수자가 확인한다.

파이프라인 위치: `runner._wearer` (3인칭 스트림과 공유 시계 IMU가 있을 때). 정책: `prelabel.yaml
wearer_matching`. 결과는 entity_id="wearer"인 사본 레코드로 남는다 (`runner.wearer_copy`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from dlp_schema.labels import KeypointTrackPayload

# COCO17 왼손목·오른손목 번호
COCO_WRISTS = (9, 10)


@dataclass(frozen=True)
class WearerMatch:
    """착용자 매칭 결과.

    entity_id: 착용자로 고른 인물 키 (러너는 라벨 ID를 키로 쓴다). 없으면 None.
    correlation: 고른 인물(또는 문턱 미달이면 최고 점수 인물)의 상관. 비교 대상이 없으면 0.
    scores: 비교한 인물별 상관 (겹침이 짧거나 신호가 평평한 인물은 빠진다).
    """

    entity_id: str | None
    correlation: float
    scores: dict[str, float]


def wrist_speed(
    track: KeypointTrackPayload, offset_ms: float = 0.0
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """(마스터 시각, 손목 속도 px/s). offset_ms는 스트림 → 마스터 오프셋.

    COCO17의 두 손목(9, 10번) 중 프레임 사이 이동 거리가 큰 쪽을 쓴다. 속도 시각은 이웃
    키프레임의 중간점이다. 키프레임이 3개 미만이면 (원래 시각, 0 배열)을 돌려준다 (매칭에서
    평평한 신호로 빠진다). 러너는 offset_ms 대신 반환 시각을 `to_master_ms`로 바꾼다 (드리프트
    보정 포함).
    """
    t = np.array([f.t_ms for f in track.keyframes], dtype=float) + offset_ms
    xy = np.array([[[f.points[i].x, f.points[i].y] for i in COCO_WRISTS] for f in track.keyframes])
    if t.size < 3:
        return t, np.zeros(t.size)
    # 이웃 키프레임 사이 이동 거리(px)를 두 손목 중 큰 값으로 고르고 시간(ms)으로 나눠 px/s로 바꾼다
    v = np.linalg.norm(np.diff(xy, axis=0), axis=2).max(axis=1) / np.diff(t) * 1000
    return (t[1:] + t[:-1]) / 2, v


def match_wearer(
    reference_t: NDArray[np.float64],
    reference_v: NDArray[np.float64],
    people: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]],
    *,
    rate_hz: float,
    min_correlation: float,
    min_overlap_samples: int,
) -> WearerMatch:
    """기준 운동 신호와 가장 닮은 인물을 고른다.

    인물마다 두 신호가 겹치는 구간 [lo, hi)를 1000/rate_hz ms 격자로 선형 보간하고 피어슨 상관을
    잰다. 겹침이 min_overlap_samples 격자 길이보다 짧거나 어느 한쪽 표준편차가 0이면 그 인물은
    건너뛴다.

    Args:
        reference_t, reference_v: 기준 신호 (마스터 ms, 값). 러너는 IMU 가속도 크기를 넘긴다.
        people: 인물 키 → (마스터 ms, 손목 속도).
        rate_hz, min_correlation, min_overlap_samples: `prelabel.yaml wearer_matching`.

    Returns:
        `WearerMatch`. 최고 상관이 min_correlation 미만이면 entity_id=None (점수는 남긴다).
    """
    scores: dict[str, float] = {}
    for entity, (t, v) in people.items():
        # 두 신호가 겹치는 시간 [lo, hi]. 그 길이가 min_overlap_samples 격자 칸보다 짧으면 건너뛴다
        lo, hi = max(reference_t[0], t[0]), min(reference_t[-1], t[-1])
        if hi - lo < min_overlap_samples * 1000 / rate_hz:
            continue
        grid = np.arange(lo, hi, 1000 / rate_hz)
        a = np.interp(grid, reference_t, reference_v)
        b = np.interp(grid, t, v)
        # 평평한 신호는 상관이 정의되지 않는다 (nan 대신 건너뛴다)
        if a.std() == 0 or b.std() == 0:
            continue
        scores[entity] = float(np.corrcoef(a, b)[0, 1])
    if not scores:
        return WearerMatch(None, 0.0, {})
    best = max(scores, key=lambda e: scores[e])
    if scores[best] < min_correlation:
        return WearerMatch(None, scores[best], scores)
    return WearerMatch(best, scores[best], scores)
