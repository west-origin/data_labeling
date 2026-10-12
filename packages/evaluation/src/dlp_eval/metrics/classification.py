"""분류 지표: macro F1, Cohen 카파, 기대 보정 오차(ECE), 클래스별 표본 수.

참조 정의:
- `macro_f1`: scikit-learn `f1_score(average="macro")` (labels=정답·예측 합집합, zero_division=0).
  test_metrics_reference.py가 무작위 입력에서 일치를 검사한다.
- `cohen_kappa`: Cohen(1960) 카파, scikit-learn `cohen_kappa_score`(범주 가중 없음)와 일치 테스트.
  표본 가중치(`weights`)는 `cohen_kappa_score(sample_weight=…)`와 같다.
- `ece`: Guo et al. 2017 "On Calibration of Modern Neural Networks"의 균등 구간 ECE. 표본
  가중치(`weights`)를 주면 구간 비중·정확도·평균 신뢰도를 가중 평균으로 낸다 (정수 가중치는 표본을
  그만큼 반복한 것과 같다, 일치 테스트).
- `under_sampled`: 이 프로젝트의 표시 규칙 (`evaluation.yaml min_samples_per_class`).

사용처: 하네스의 파지 유형·행동 동사 macro F1, 객체 검출 신뢰도 ECE, 검수 운영(dlp_review.ops)의
이중 라벨링 일치도(카파), AI 1차 검수 품질의 대조군 카파·ECE(표본 가중,
`dlp_review.ops.ai_quality`).
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence

import numpy as np


def macro_f1(truth: Sequence[Hashable], pred: Sequence[Hashable]) -> float:
    """정답·예측에 나온 모든 클래스의 F1 평균 (scikit-learn f1_score(average="macro")와 같다).

    Args:
        truth, pred: 같은 길이의 클래스 열 (짝지어진 표본). 길이가 다르면 `ValueError`(zip strict).

    Returns:
        클래스별 F1(2TP / (2TP + FP + FN))의 단순 평균. 표본이 없으면 0.0.

    주의: 예측에만 나온 클래스도 평균에 들어간다 (F1 0). 하네스는 맞출 예측이 없는 정답을
    "missing" 클래스로 넘기므로, 놓친 표본이 하나라도 있으면 "missing" 클래스(F1 0)가 평균에
    더해진다.
    """
    # repr로 정렬해 클래스 타입이 섞여도 순서가 정해지게 한다 (결과에는 영향 없음)
    classes = sorted(set(truth) | set(pred), key=repr)
    scores: list[float] = []
    for c in classes:
        tp = sum(t == c and p == c for t, p in zip(truth, pred, strict=True))
        fp = sum(t != c and p == c for t, p in zip(truth, pred, strict=True))
        fn = sum(t == c and p != c for t, p in zip(truth, pred, strict=True))
        scores.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def cohen_kappa(
    a: Sequence[Hashable], b: Sequence[Hashable], weights: Sequence[float] | None = None
) -> float:
    """두 라벨러의 일치도. 우연 일치를 뺀 비율.

    kappa = (po - pe) / (1 - pe). po는 관측 일치율, pe는 두 라벨러의 클래스 주변 분포로 계산한
    우연 일치율이다. weights를 주면 표본마다 그 무게로 센다 (일치율·주변 분포 모두).

    Args:
        a, b: 같은 표본에 대한 두 라벨러의 클래스 열 (같은 길이).
        weights: 표본 가중치 (같은 길이, 0 이상). None이면 모두 1.

    Returns:
        -1~1. 표본이 없으면(가중치 합 0 포함) 0.0. pe == 1(두 라벨러가 모든 표본에 같은 한 클래스만
        씀)이면 완전 일치이므로 1.0 (scikit-learn은 이때 0/0으로 NaN을 낸다 — 이 점만 다르다).
    """
    w = [1.0] * len(a) if weights is None else [float(x) for x in weights]
    pairs = list(zip(a, b, w, strict=True))
    n = sum(w)
    if not pairs or n == 0:
        return 0.0
    po = sum(wi for x, y, wi in pairs if x == y) / n
    ca: dict[Hashable, float] = {}
    cb: dict[Hashable, float] = {}
    for x, y, wi in pairs:
        ca[x] = ca.get(x, 0.0) + wi
        cb[y] = cb.get(y, 0.0) + wi
    pe = sum(ca.get(k, 0.0) * cb.get(k, 0.0) for k in set(ca) | set(cb)) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def ece(
    confidence: Sequence[float],
    correct: Sequence[bool],
    bins: int = 15,
    weights: Sequence[float] | None = None,
) -> float:
    """기대 보정 오차: 신뢰도 구간(균등 bins개)별 |정확도 - 평균 신뢰도|의 표본 가중 평균.

    Guo et al. 2017 정의: ECE = sum_b (|B_b| / n) * |acc(B_b) - conf(B_b)|.
    구간은 [0, 1/bins], (1/bins, 2/bins], …, ((bins-1)/bins, 1]이다 (첫 구간만 0을 포함).
    weights를 주면 |B_b|·n·acc·conf를 그 무게의 합·가중 평균으로 낸다.

    Args:
        confidence: 예측마다 모델 신뢰도 (0~1).
        correct: 예측마다 맞았는지 (하네스는 객체 예측이 같은 클래스 정답과
            IoU >= track_iou로 맞았는지).
        bins: 구간 수 (`evaluation.yaml ece_bins`, 기본 15).
        weights: 표본 가중치 (같은 길이, 0 이상). None이면 모두 1.

    Returns:
        0~1 (낮을수록 좋다, `harness.LOWER_IS_BETTER`). 예측이 없으면(가중치 합 0 포함) NaN.
    """
    conf = np.asarray(confidence, dtype=np.float64)
    ok = np.asarray(correct, dtype=np.float64)
    w = np.ones_like(conf) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != conf.shape or ok.shape != conf.shape:
        raise ValueError("confidence·correct·weights의 길이가 다르다")
    n = float(w.sum())
    if conf.size == 0 or n == 0:
        return float("nan")  # 예측이 없으면 보정을 잴 수 없다 (0이면 만점처럼 보인다)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        # 구간은 (lo, hi]. 첫 구간만 [lo, hi]로 신뢰도 0을 넣는다
        mask = (conf > lo) & (conf <= hi) if i else (conf >= lo) & (conf <= hi)
        mass = float(w[mask].sum())
        if mass > 0:
            acc = float((w[mask] * ok[mask]).sum()) / mass
            mean_conf = float((w[mask] * conf[mask]).sum()) / mass
            total += mass / n * abs(acc - mean_conf)
    return float(total)


def under_sampled(counts: dict[str, int], minimum: int) -> list[str]:
    """정답 표본이 minimum보다 적은 클래스 (지표를 믿기 어렵다는 표시용).

    Args:
        counts: 클래스 → 정답 표본 수 (하네스가 과제마다 센다).
        minimum: `evaluation.yaml min_samples_per_class`.

    Returns:
        정렬된 클래스 이름 목록. 게이트는 이 목록으로 막지 않고 경고만 남긴다.
    """
    return sorted(c for c, n in counts.items() if n < minimum)
