"""두 번 두드림 검출과 스트림 사이 대응.

오디오(감쇠 버스트), 장갑 압력·IMU 가속도(짧은 펄스) 모두 같은 방식으로 찾는다.
1. 포락선이 잡음 수준(중앙값 + MAD 배수)을 넘는 구간을 사건으로 묶는다.
2. 너무 긴 사건(행동 중 접촉 등)은 버린다.
3. 간격이 double_tap_gap_ms 안인 연속 사건 두 개를 한 번의 "두 번 두드림"으로 본다.
사건 시각은 문턱을 처음 넘은 샘플이다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from dlp_sync.anchors import Anchor
from dlp_sync.policy import TapPolicy


@dataclass(frozen=True)
class DoubleTap:
    first_ms: float
    second_ms: float


def detect_double_taps(
    t_ms: NDArray[np.float64], envelope: NDArray[np.float64], policy: TapPolicy
) -> list[DoubleTap]:
    env = np.abs(envelope - np.median(envelope))
    mad = float(np.median(np.abs(env - np.median(env)))) or float(np.std(env)) or 1e-12
    threshold = float(np.median(env)) + policy.threshold_mad * mad
    above = np.flatnonzero(env > threshold)
    if above.size == 0:
        return []

    events: list[tuple[float, float]] = []  # (시작, 끝)
    start = prev = float(t_ms[above[0]])
    for i in above[1:]:
        t = float(t_ms[i])
        if t - prev > policy.merge_gap_ms:
            events.append((start, prev))
            start = t
        prev = t
    events.append((start, prev))
    pulses = [s for s, e in events if e - s <= policy.max_pulse_ms]

    lo, hi = policy.double_tap_gap_ms
    taps: list[DoubleTap] = []
    i = 0
    while i < len(pulses) - 1:
        if lo <= pulses[i + 1] - pulses[i] <= hi:
            taps.append(DoubleTap(pulses[i], pulses[i + 1]))
            i += 2
        else:
            i += 1
    return taps


def match_taps(
    reference: list[DoubleTap], target: list[DoubleTap], policy: TapPolicy
) -> list[Anchor]:
    """기준(바디캠)과 대상의 두 번 두드림을 짝짓는다.

    가능한 모든 짝의 오프셋 후보 중 허용 오차 안에서 가장 많은 짝을 설명하는 것을 고른다
    (어느 한쪽이 두드림을 놓치거나 잡음을 두드림으로 잡아도 견딘다).
    """
    best: list[Anchor] = []
    for r in reference:
        for t in target:
            offset = r.first_ms - t.first_ms
            anchors: list[Anchor] = []
            used: set[int] = set()
            for rr in reference:
                for j, tt in enumerate(target):
                    if (
                        j not in used
                        and abs(rr.first_ms - (tt.first_ms + offset)) <= policy.match_tolerance_ms
                    ):
                        used.add(j)
                        anchors.append(Anchor(rr.first_ms, tt.first_ms, "tap1"))
                        anchors.append(Anchor(rr.second_ms, tt.second_ms, "tap2"))
                        break
            if len(anchors) > len(best):
                best = anchors
    return best
