"""키포인트 PCK: 정답 관절과 예측 관절 거리가 alpha * 기준 길이 이하인 비율.

기준 길이는 정답 키포인트 박스의 긴 변이다 (손 21점, 전신 17점 모두). 정답에서 보이는(visibility>0)
관절만 센다. 예측이 없는 관절은 틀린 것으로 센다.

참조 정의와 차이:
- PCK(Yang & Ramanan 2013)는 "거리 <= alpha * 기준 길이"인 관절 비율이다. 기준 길이는 데이터셋마다
  다르다 (MPII PCKh는 머리 길이, 손 데이터셋은 손 박스 긴 변 등). 여기서는 손·전신 모두 **정답에서
  보이는 관절로 만든 박스의 긴 변**을 쓴다 (`config/policies/evaluation.yaml pck_alpha`, 기본 0.1).
- 보이는 관절이 하나뿐이면 박스 크기가 0이라 기준 길이를 1e-6으로 막는다 → 사실상 거리 0만 맞음.
- 예측의 visibility는 보지 않는다 (예측 좌표만 쓴다). 예측 좌표가 NaN이면 비교가 거짓이라 틀림이
  된다.
- 정답·예측 짝 맞추기(누구의 손인지)는 이 함수 밖, `harness.eval_keypoints`가 한다.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PckResult:
    """PCK 결과."""

    pck: float  # correct / total (0~1). 보이는 정답 관절이 하나도 없으면 0.0
    correct: int  # 허용 거리 안에 든 관절 수
    total: int  # 센 관절 수 (정답에서 visibility>0인 관절, 예측 없음 포함)


def pck(pairs: Sequence[tuple[np.ndarray, np.ndarray | None]], alpha: float) -> PckResult:
    """pairs: (정답 (K, 3)=[x, y, visibility], 예측 (K, 2) 또는 None).

    Args:
        pairs: 정답·예측 짝 목록. 정답은 (K, 3) 배열 [x px, y px, visibility], 예측은 (K, 2) 배열
            [x px, y px]로 같은 관절 순서여야 한다. 예측이 None이면 그 정답의 보이는 관절을 모두
            틀린 것으로 센다 (맞출 예측이 없는 정답).
        alpha: 기준 길이에 곱하는 비율 (`evaluation.yaml pck_alpha`). 0보다 커야 한다.

    Returns:
        모든 짝을 합친 `PckResult` (관절 단위 미시 평균, 짝·클래스별 평균이 아니다).
    """
    correct = total = 0
    for gt, pred in pairs:
        visible = gt[:, 2] > 0
        if not visible.any():
            # 보이는 관절이 없는 정답은 분모에도 넣지 않는다
            continue
        xs, ys = gt[visible, 0], gt[visible, 1]
        # 기준 길이: 보이는 정답 관절 박스의 긴 변 (0 나눗셈을 막으려 1e-6 하한)
        scale = max(float(xs.max() - xs.min()), float(ys.max() - ys.min()), 1e-6)
        total += int(visible.sum())
        if pred is None:
            continue
        d = np.hypot(pred[visible, 0] - xs, pred[visible, 1] - ys)
        correct += int(np.sum(d <= alpha * scale))
    return PckResult(correct / total if total else 0.0, correct, total)
