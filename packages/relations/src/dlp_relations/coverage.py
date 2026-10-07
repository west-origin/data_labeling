"""표면 커버리지: 접촉 중 작용부 경로를 반지름 footprint 원으로 쓸어 덮은 면적 비율.

표면 좌표(가로·세로 비율)에 표면 크기(m, 접촉 샘플들의 중앙값)를 곱해 미터 격자에 칠한다. 같은 접촉
구간 안의 연속 샘플은 선분으로 잇는다 (샘플 간격이 footprint보다 커도 빈틈이 없다).

정책: `relations.yaml coverage` (grid_m, footprint_m). `derive`가 (도구, 표면)마다 부른다.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

from dlp_relations.contact import SurfaceContact


def coverage_ratio(contacts: list[SurfaceContact], footprint_m: float, grid_m: float) -> float:
    """덮은 칸 수 / 전체 칸 수 (0~1).

    Args:
        contacts: 같은 (도구, 표면)의 접촉 구간들. points가 표면 좌표, sizes가 표면 크기.
        footprint_m: 작용부 원 반지름(m). grid_m: 격자 칸 크기(m).

    Returns:
        표면 크기를 알 수 없으면(샘플 없음) 0. 칸 중심이 어느 선분(캡슐)에서 footprint_m 안이면 덮은
        것으로 본다. 표면 밖(비율 0~1 밖) 점은 격자 밖이라 칠하지 않는다.
    """
    sizes = [s for c in contacts for s in c.sizes]
    if not sizes:
        return 0.0
    # 표면 크기(m): 접촉 샘플들의 중앙값 (깊이 잡음으로 시각마다 조금씩 다르다)
    width, height = (float(np.median([s[i] for s in sizes])) for i in (0, 1))
    nx, ny = max(1, math.ceil(width / grid_m)), max(1, math.ceil(height / grid_m))
    # 칸 중심 좌표(m). 칸 수는 grid_m로 올림해 표면 크기에 정확히 맞춘다
    xs = (np.arange(nx) + 0.5) * width / nx
    ys = (np.arange(ny) + 0.5) * height / ny
    covered = np.zeros((ny, nx), dtype=bool)
    for c in contacts:
        pts = [(a * width, b * height) for _, a, b in c.points]
        # 연속 샘플을 선분으로 잇는다. 샘플이 하나면 길이 0 선분(원)으로 칠한다
        segments = list(itertools.pairwise(pts))
        if len(pts) == 1:
            segments = [(pts[0], pts[0])]
        for (ax, ay), (bx, by) in segments:
            # 선분 둘레 footprint_m 경계 상자 안 칸만 계산한다 (전체 격자를 매번 보지 않게)
            x0 = np.searchsorted(xs, min(ax, bx) - footprint_m)
            x1 = np.searchsorted(xs, max(ax, bx) + footprint_m)
            y0 = np.searchsorted(ys, min(ay, by) - footprint_m)
            y1 = np.searchsorted(ys, max(ay, by) + footprint_m)
            if x0 >= x1 or y0 >= y1:
                continue
            gx, gy = np.meshgrid(xs[x0:x1], ys[y0:y1])
            dx, dy = bx - ax, by - ay
            length2 = dx * dx + dy * dy
            # 칸 중심을 선분에 투영한 매개변수 u (0~1로 잘라 끝점 쪽은 원이 된다) → 캡슐까지 거리
            u = np.clip(((gx - ax) * dx + (gy - ay) * dy) / length2, 0, 1) if length2 else 0.0
            near = np.hypot(gx - (ax + u * dx), gy - (ay + u * dy)) <= footprint_m
            covered[y0:y1, x0:x1] |= near
    return float(covered.mean())
