"""상태 전이 정확도: 개체·속성별 상태 값이 바뀌는 시점(이전 값 → 다음 값)을 정답과 맞춘다.

정답 전이 하나는 같은 개체·속성·이전 값·다음 값을 가진 예측 전이가 허용 오차 안에 있으면
맞은 것이다.

이 프로젝트 고유 지표다 (공개 참조 구현 없음, ADR 0013). 시점 매칭은
`temporal.match_events`(허용 오차 안 일대일, 오차 합 최소)를 같은 (개체, 속성, 이전 값, 다음 값)
묶음 안에서 쓴다. 허용 오차는 `evaluation.yaml tolerance_ms.state`(ms, 마스터 타임라인 시각).
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass

from dlp_eval.metrics.temporal import match_events

# (개체, 속성, 시작 ms, 끝 ms, 값)
StateSpan = tuple[str, str, int, int, str]
Transition = tuple[str, str, str, str, int]  # 개체, 속성, 이전 값, 다음 값, 시각


def transitions(spans: Sequence[StateSpan]) -> list[Transition]:
    """상태 구간 목록에서 값이 바뀌는 전이를 뽑는다.

    (개체, 속성)별로 시작 시각순으로 정렬해, 이웃한 두 구간의 값이 다르면 뒤 구간의 시작 시각을
    전이 시각으로 본다. 구간 사이 공백·겹침은 따지지 않는다 (이웃 순서만 본다). 값이 같은 이웃은
    전이가 아니다.

    Returns:
        (개체, 속성, 이전 값, 다음 값, 전이 시각 ms) 목록. 순서는 묶음 등장 순서 → 시각순.
    """
    out: list[Transition] = []
    groups: dict[tuple[str, str], list[StateSpan]] = {}
    for s in spans:
        groups.setdefault((s[0], s[1]), []).append(s)
    for (entity, attr), items in groups.items():
        items = sorted(items, key=lambda s: s[2])
        for a, b in itertools.pairwise(items):
            if a[4] != b[4]:
                out.append((entity, attr, a[4], b[4], b[2]))
    return out


@dataclass(frozen=True)
class TransitionResult:
    """상태 전이 매칭 결과."""

    accuracy: float  # 맞힌 정답 전이 / 정답 전이 (정답 전이가 없으면 1.0)
    precision: float  # 맞힌 예측 전이 / 예측 전이 (예측 전이가 없으면 1.0)
    matched: int  # 맞힌 전이 수 (일대일)
    truth: int  # 정답 전이 수
    pred: int  # 예측 전이 수


def transition_accuracy(
    truth: Sequence[StateSpan], pred: Sequence[StateSpan], tolerance_ms: int
) -> TransitionResult:
    """정답·예측 상태 구간의 전이를 맞춘다.

    Args:
        truth, pred: (개체, 속성, 시작 ms, 끝 ms, 값) 구간. 한 세션 안의 것이어야 한다
            (개체 ID가 세션 범위이므로 하네스가 세션마다 부른다).
        tolerance_ms: 전이 시각 허용 오차 (ms, 양쪽 포함).

    Returns:
        `TransitionResult`. 정답 전이가 없으면 accuracy 1.0, 예측 전이가 없으면 precision 1.0
        (0 나눗셈 대신 "틀린 것이 없음"으로 본다).
    """
    tt, pt = transitions(truth), transitions(pred)
    matched = 0
    # 같은 (개체, 속성, 이전 값, 다음 값) 묶음 안에서만 시각을 맞춘다
    for key in {t[:4] for t in tt} | {p[:4] for p in pt}:
        r = match_events(
            [t[4] for t in tt if t[:4] == key], [p[4] for p in pt if p[:4] == key], tolerance_ms
        )
        matched += r.tp
    return TransitionResult(
        matched / len(tt) if tt else 1.0,
        matched / len(pt) if pt else 1.0,
        matched,
        len(tt),
        len(pt),
    )
