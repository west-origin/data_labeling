"""scipy 호출(헝가리안 할당, k-최근접)의 타입 래퍼. scipy에는 타입 스텁이 없다.

pyright strict의 알 수 없는 타입 경고를 이 모듈 하나에 가둔다
(`dlp_eval.metrics.assign`과 같은 방식).
"""

# pyright: reportMissingTypeStubs=false, reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from scipy.spatial import KDTree


def assign(cost: NDArray[np.float64]) -> list[tuple[int, int]]:
    """비용 합이 최소인 일대일 짝 (행, 열). 금지 짝은 호출자가 큰 비용으로 넣고 걸러낸다."""
    rows, cols = linear_sum_assignment(cost)
    r = np.asarray(rows, dtype=np.int64).tolist()
    c = np.asarray(cols, dtype=np.int64).tolist()
    return list(zip(r, c, strict=True))


def knn(
    data: NDArray[np.float64], query: NDArray[np.float64], k: int
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """query 각 행의 data 최근접 k개 (거리 (Q, k), 번호 (Q, k))."""
    dist, idx = KDTree(data).query(query, k=k)
    d = np.asarray(dist, dtype=np.float64).reshape(len(query), k)
    i = np.asarray(idx, dtype=np.int64).reshape(len(query), k)
    return d, i
