"""비율의 신뢰 상한 (Clopper-Pearson 정확 구간). scipy에는 타입 스텁이 없다.

pyright strict에서 scipy 호출의 알 수 없는 타입 경고를 이 모듈 하나에 가둔다. 사용처: AI 1차 검수
품질 지표의 ok 놓침률 상한 (`dlp_review.ops.ai_quality`, ADR 0040 7.2).

참조 정의: 한쪽 Clopper-Pearson 상한 = Beta(confidence; k + 1, n - k)의 분위수 (k = n이면 1).
scipy `binomtest(k, n, alternative="less").proportion_ci(confidence, method="exact").high`와 같다
(test_metrics_reference.py).
"""

# pyright: reportMissingTypeStubs=false, reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false

from __future__ import annotations

import math

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
