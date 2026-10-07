"""분류 지표: macro F1, Cohen 카파, 기대 보정 오차(ECE), 클래스별 표본 수."""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Sequence

import numpy as np


def macro_f1(truth: Sequence[Hashable], pred: Sequence[Hashable]) -> float:
    """정답·예측에 나온 모든 클래스의 F1 평균 (scikit-learn f1_score(average="macro")와 같다)."""
    classes = sorted(set(truth) | set(pred), key=repr)
    scores: list[float] = []
    for c in classes:
        tp = sum(t == c and p == c for t, p in zip(truth, pred, strict=True))
        fp = sum(t != c and p == c for t, p in zip(truth, pred, strict=True))
        fn = sum(t == c and p != c for t, p in zip(truth, pred, strict=True))
        scores.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def cohen_kappa(a: Sequence[Hashable], b: Sequence[Hashable]) -> float:
    """두 라벨러의 일치도. 우연 일치를 뺀 비율."""
    n = len(a)
    if n == 0:
        return 0.0
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def ece(confidence: Sequence[float], correct: Sequence[bool], bins: int = 15) -> float:
    """기대 보정 오차: 신뢰도 구간(균등 bins개)별 |정확도 - 평균 신뢰도|의 표본 가중 평균."""
    conf = np.asarray(confidence, dtype=np.float64)
    ok = np.asarray(correct, dtype=np.float64)
    if conf.size == 0:
        return float("nan")  # 예측이 없으면 보정을 잴 수 없다 (0이면 만점처럼 보인다)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf > lo) & (conf <= hi) if i else (conf >= lo) & (conf <= hi)
        if mask.any():
            total += mask.sum() / conf.size * abs(ok[mask].mean() - conf[mask].mean())
    return float(total)


def under_sampled(counts: dict[str, int], minimum: int) -> list[str]:
    """정답 표본이 minimum보다 적은 클래스 (지표를 믿기 어렵다는 표시용)."""
    return sorted(c for c, n in counts.items() if n < minimum)
