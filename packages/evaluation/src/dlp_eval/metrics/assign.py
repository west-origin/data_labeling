"""헝가리안 할당 (scipy)의 타입 래퍼. scipy에는 타입 스텁이 없다."""

# pyright: reportMissingTypeStubs=false, reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment


def assign(cost: NDArray[np.float64]) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """비용 합이 최소인 일대일 할당 (행 인덱스, 열 인덱스)."""
    rows, cols = linear_sum_assignment(cost)
    return np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)
