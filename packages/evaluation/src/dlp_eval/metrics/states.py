"""상태 전이 정확도: 개체·속성별 상태 값이 바뀌는 시점(이전 값 → 다음 값)을 정답과 맞춘다.

정답 전이 하나는 같은 개체·속성·이전 값·다음 값을 가진 예측 전이가 허용 오차 안에 있으면
맞은 것이다.
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
    accuracy: float  # 맞힌 정답 전이 / 정답 전이
    precision: float
    matched: int
    truth: int
    pred: int


def transition_accuracy(
    truth: Sequence[StateSpan], pred: Sequence[StateSpan], tolerance_ms: int
) -> TransitionResult:
    tt, pt = transitions(truth), transitions(pred)
    matched = 0
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
