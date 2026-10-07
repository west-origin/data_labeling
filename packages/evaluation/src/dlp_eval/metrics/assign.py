"""헝가리안 할당 (scipy)의 타입 래퍼. scipy에는 타입 스텁이 없다.

pyright strict에서 scipy 호출의 알 수 없는 타입 경고를 이 모듈 하나에 가둔다. 사용처: 시점 사건
매칭(`temporal.match_events`), 추적 지표(`tracking.hota`·`clear`·`identity`), 키포인트 매칭
(`harness._assign_pairs`).
"""

# pyright: reportMissingTypeStubs=false, reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment


def assign(cost: NDArray[np.float64]) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """비용 합이 최소인 일대일 할당 (행 인덱스, 열 인덱스).

    Args:
        cost: (행 수, 열 수) 비용 행렬. 직사각형이어도 된다 (짧은 쪽 개수만큼 짝이 나온다).
            최대화가 필요하면 호출자가 부호를 뒤집어 넘긴다 (예: `assign(-score)`).
            "맞추면 안 되는 짝"은 큰 비용을 넣고, 할당 뒤 호출자가 그 짝을 걸러야 한다
            (scipy는 짝 수를 줄이지 않는다).

    Returns:
        (rows, cols): 같은 길이의 int64 배열. `rows`는 오름차순이다 (scipy 규약).
    """
    rows, cols = linear_sum_assignment(cost)
    return np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)
