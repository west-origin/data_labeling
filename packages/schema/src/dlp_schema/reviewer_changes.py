"""검수자가 직접 낸 검수 변화 (물려받기만 한 라벨 제외, F-COE-23·31·32).

`dlp_schema.history.review_changes`는 현재 모델 라벨이 검수 상태면 "그대로 승인(accepted)"으로
센다. 그런데 설명(ADR 0043 3절)과 3D 궤적(ADR 0048·0050·0051)은 검수 도구 경로가 없어 행동·2D
트랙 검수를 물려받는다. 행동 하나·손 트랙 하나 승인에 따라온 물려받은 라벨 여러 개를 검수자의
승인으로 세면 운영 지표(자동 승인율·수정률), 액티브 러닝 수정률이 희석되고, 원본 보관 기산점
(마지막 사람 검수 시각)이 자동 단계가 3D를 다시 만든 시각으로 밀린다 (물려받는 검수 시각은
max(주인 검수 시각, 라벨 생성 시각)이다).

그래서 지금 검수 상태가 명시 물려받기 기록(`description_inheritance`·`trajectory_inheritance`)의
마지막 기록에서 온 라벨은 뺀다. 물려준 행동·2D 트랙의 사람 검수는 그 자체로 세어지므로 빠지는
사람 행동은 없다. 기록 인자는 기본값이 없다 (빠뜨리면 물려받은 라벨을 직접 검수로 센다,
`tests/test_inheritance_required.py`).

호출자: `dlp_ops.metrics`(주간 지표), `dlp_ops.retention`(보관 기산점), `dlp_active.rates`(수정률).
순수 함수다.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence

from dlp_schema.description_follow import active_inheritance
from dlp_schema.history import ReviewChange, review_changes
from dlp_schema.labels import LabelRecord, VerificationState
from dlp_schema.review import DescriptionInheritance, TrajectoryInheritance
from dlp_schema.trajectory_follow import active_trajectory_inheritance


def reviewer_changes(
    history: list[LabelRecord],
    states: Collection[VerificationState],
    inheritance: Sequence[DescriptionInheritance],
    trajectory_inheritance: Sequence[TrajectoryInheritance],
) -> list[ReviewChange]:
    """`review_changes` 중 검수를 물려받기만 한 라벨(설명·3D 궤적)을 뺀 것.

    Args:
        history: 세션 하나의 전체 라벨 이력 (`get_labels`).
        states: "검수됨"으로 볼 검증 상태 (`review_changes`와 같다).
        inheritance: 세션의 설명 물려받기 기록 (`get_description_inheritance`).
        trajectory_inheritance: 세션의 3D 궤적 물려받기 기록 (`get_trajectory_inheritance`).
    """
    return [
        c
        for c in review_changes(history, states)
        if active_inheritance(c.label, inheritance) is None
        and active_trajectory_inheritance(c.label, trajectory_inheritance) is None
    ]
