"""키포인트 PCK: 정답 관절과 예측 관절 거리가 alpha * 기준 길이 이하인 비율.

기준 길이는 정답 키포인트 박스의 긴 변이다 (손 21점, 전신 17점 모두). 정답에서 보이는(visibility>0)
관절만 센다. 예측이 없는 관절은 틀린 것으로 센다.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PckResult:
    pck: float
    correct: int
    total: int


def pck(pairs: Sequence[tuple[np.ndarray, np.ndarray | None]], alpha: float) -> PckResult:
    """pairs: (정답 (K, 3)=[x, y, visibility], 예측 (K, 2) 또는 None)."""
    correct = total = 0
    for gt, pred in pairs:
        visible = gt[:, 2] > 0
        if not visible.any():
            continue
        xs, ys = gt[visible, 0], gt[visible, 1]
        scale = max(float(xs.max() - xs.min()), float(ys.max() - ys.min()), 1e-6)
        total += int(visible.sum())
        if pred is None:
            continue
        d = np.hypot(pred[visible, 0] - xs, pred[visible, 1] - ys)
        correct += int(np.sum(d <= alpha * scale))
    return PckResult(correct / total if total else 0.0, correct, total)
