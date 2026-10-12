"""비율의 신뢰 상한 (Clopper-Pearson 정확 구간). scipy에는 타입 스텁이 없다.

pyright strict에서 scipy 호출의 알 수 없는 타입 경고를 이 모듈 하나에 가둔다. 사용처: AI 1차 검수
품질 지표의 ok 놓침률 상한 (`dlp_review.ops.ai_quality`, ADR 0040 7.2).

참조 정의: 한쪽 Clopper-Pearson 상한 = Beta(confidence; k + 1, n - k)의 분위수 (k = n이면 1).
scipy `binomtest(k, n, alternative="less").proportion_ci(confidence, method="exact").high`와 같다
(test_metrics_reference.py).

표본 가중 비율의 상한(`weighted_upper`)은 Korn-Graubard(1998) 방식의 근사다: 유효 표본 수
n* = (Σw)²/Σw² (Kish)와 x* = p̂·n*를 Clopper-Pearson 식에 넣는다 (x*·n*가 실수여도 Beta 분위수는
정의된다). 가중치가 모두 같으면 n* = n이라 `clopper_pearson_upper`와 같다.
"""

# pyright: reportMissingTypeStubs=false, reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false

from __future__ import annotations

import math
from collections.abc import Sequence

from scipy.stats import beta


def clopper_pearson_upper(k: int, n: int, confidence: float = 0.95) -> float:
    """n번 중 k번 일어난 사건 비율의 한쪽 신뢰 상한 (Clopper-Pearson).

    Args:
        k: 일어난 수 (0 ≤ k ≤ n).
        n: 시행 수.
        confidence: 신뢰 수준 (0~1, 기본 95%).

    Returns:
        0~1. 시행이 없으면 NaN (상한을 모른다. 0이나 1로 두면 통과·실패처럼 보인다).

    Raises:
        ValueError: k가 0~n 밖이거나 confidence가 (0, 1) 밖일 때.
    """
    if not 0 <= k <= n or not 0 < confidence < 1:
        raise ValueError(f"잘못된 입력: k={k}, n={n}, confidence={confidence}")
    if n == 0:
        return math.nan
    if k == n:
        return 1.0
    return float(beta.ppf(confidence, k + 1, n - k))


def effective_upper(p: float, n_eff: float, confidence: float = 0.95) -> float:
    """비율 p, 유효 표본 수 n_eff인 가중 비율의 한쪽 신뢰 상한 (Korn-Graubard 근사, 모듈 docstring).

    x = p·n_eff로 Beta(confidence; x + 1, n_eff - x) 분위수를 낸다 (x = n_eff면 1). n_eff와 x가
    정수면 `clopper_pearson_upper(x, n_eff)`와 같다.

    Returns:
        0~1. n_eff가 0이면 NaN.

    Raises:
        ValueError: p가 0~1 밖, n_eff가 음수, confidence가 (0, 1) 밖일 때.
    """
    if not 0 <= p <= 1 or n_eff < 0 or not 0 < confidence < 1:
        raise ValueError(f"잘못된 입력: p={p}, n_eff={n_eff}, confidence={confidence}")
    if n_eff == 0:
        return math.nan
    x = p * n_eff
    if x >= n_eff:
        return 1.0
    return float(beta.ppf(confidence, x + 1, n_eff - x))


def weighted_upper(hits: Sequence[float], pool: Sequence[float], confidence: float = 0.95) -> float:
    """표본 가중 비율 Σhits / Σpool의 한쪽 신뢰 상한 (`effective_upper`, n_eff = (Σw)²/Σw²).

    Args:
        hits: 사건이 일어난 표본의 가중치 (pool의 부분 집합).
        pool: 모든 표본의 가중치 (0 이상).
        confidence: 신뢰 수준 (0~1, 기본 95%).

    Returns:
        0~1. 표본이 없거나 가중치 합이 0이면 NaN. 사건이 모든 표본이면 1.

    Raises:
        ValueError: 음수 가중치, 사건 가중치 합이 전체보다 클 때, confidence가 (0, 1) 밖일 때.
    """
    total, hit = float(sum(pool)), float(sum(hits))
    if any(w < 0 for w in pool) or hit > total + 1e-9 or not 0 < confidence < 1:
        raise ValueError(f"잘못된 입력: hits={hit}, pool={total}, confidence={confidence}")
    if not pool or total == 0:
        return math.nan
    n_eff = total * total / float(sum(w * w for w in pool))
    return effective_upper(min(hit / total, 1.0), n_eff, confidence)
