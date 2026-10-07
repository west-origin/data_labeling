"""표면 커버리지: 접촉 중 작용부 경로를 반지름 footprint 원으로 쓸어 덮은 면적 비율.

표면 좌표(가로·세로 비율)에 표면 크기(m, 접촉 샘플들의 중앙값)를 곱해 미터 격자에 칠한다.
같은 접촉 구간 안의 연속 샘플은 선분으로 잇는다 (샘플 간격이 footprint보다 커도 빈틈이 없다).
"""

from __future__ import annotations

import itertools
import math

import numpy as np

from dlp_relations.contact import SurfaceContact


def coverage_ratio(contacts: list[SurfaceContact], footprint_m: float, grid_m: float) -> float:
    sizes = [s for c in contacts for s in c.sizes]
    if not sizes:
        return 0.0
    width, height = (float(np.median([s[i] for s in sizes])) for i in (0, 1))
    nx, ny = max(1, math.ceil(width / grid_m)), max(1, math.ceil(height / grid_m))
    xs = (np.arange(nx) + 0.5) * width / nx
    ys = (np.arange(ny) + 0.5) * height / ny
    covered = np.zeros((ny, nx), dtype=bool)
    for c in contacts:
        pts = [(a * width, b * height) for _, a, b in c.points]
        segments = list(itertools.pairwise(pts))
        if len(pts) == 1:
            segments = [(pts[0], pts[0])]
        for (ax, ay), (bx, by) in segments:
            x0 = np.searchsorted(xs, min(ax, bx) - footprint_m)
            x1 = np.searchsorted(xs, max(ax, bx) + footprint_m)
            y0 = np.searchsorted(ys, min(ay, by) - footprint_m)
            y1 = np.searchsorted(ys, max(ay, by) + footprint_m)
            if x0 >= x1 or y0 >= y1:
                continue
            gx, gy = np.meshgrid(xs[x0:x1], ys[y0:y1])
            dx, dy = bx - ax, by - ay
            length2 = dx * dx + dy * dy
            u = np.clip(((gx - ax) * dx + (gy - ay) * dy) / length2, 0, 1) if length2 else 0.0
            near = np.hypot(gx - (ax + u * dx), gy - (ay + u * dy)) <= footprint_m
            covered[y0:y1, x0:x1] |= near
    return float(covered.mean())
