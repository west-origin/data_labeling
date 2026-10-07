"""두 번 두드림 검출과 스트림 사이 대응 (`tap_event` 방법, WP4, ADR 0004·0028).

촬영 프로토콜: 녹화 시작·끝에 착용자가 장갑 낀 손으로 두 번 두드린다. 바디캠 마이크(기준)와
대상 스트림(3인칭 마이크, 장갑 압력, 외부 IMU, 외부 오디오)이 같은 사건을 본다.

오디오(감쇠 버스트), 장갑 압력·IMU 가속도(짧은 펄스) 모두 같은 방식으로 찾는다.
1. 포락선이 잡음 수준(중앙값 + MAD 배수)을 넘는 구간을 사건으로 묶는다.
2. 너무 긴 사건(행동 중 접촉 등)은 버린다.
3. 간격이 double_tap_gap_ms 안인 연속 사건 두 개를 한 번의 "두 번 두드림"으로 본다.
사건 시각은 문턱을 처음 넘은 샘플이다.

정책: `sync.yaml tap` 절 (`double_tap_gap_ms`, `max_pulse_ms`, `merge_gap_ms`, `threshold_mad`,
`match_tolerance_ms`). 짝짓기 허용 오차는 `max_drift_ppm`만큼 넓힌다 (ADR 0028 결정 4).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from dlp_sync.anchors import Anchor
from dlp_sync.policy import TapPolicy


@dataclass(frozen=True)
class DoubleTap:
    """한 번의 "두 번 두드림" 사건. 두 시각 모두 그 스트림 자기 시계 ms (첫 문턱 초과 샘플)."""

    first_ms: float
    second_ms: float


def detect_double_taps(
    t_ms: NDArray[np.float64], envelope: NDArray[np.float64], policy: TapPolicy
) -> list[DoubleTap]:
    """신호에서 두 번 두드림을 찾는다.

    Args:
        t_ms: 샘플 시각 ms (오름차순, 불균일 가능).
        envelope: 같은 길이의 신호 (오디오 샘플, 압력 합, 가속도 크기). 부호는 상관없다
            (중앙값을 뺀 절댓값을 쓴다).
        policy: `sync.yaml tap`.

    Returns:
        시간 순 `DoubleTap` 목록. 사건 두 개가 한 쌍으로 쓰이면 다음 쌍은 그 다음 사건부터 찾는다
        (겹치는 쌍을 만들지 않는다). 없으면 빈 목록.
    """
    # 기준선(중앙값)을 빼고 절댓값: 오디오처럼 양음으로 흔들리는 신호도 크기로 본다
    env = np.abs(envelope - np.median(envelope))
    # MAD(중앙값 절대 편차)로 잡음 크기를 잰다. 0이면(평탄 신호) 표준편차, 그것도 0이면 아주 작은 값
    mad = float(np.median(np.abs(env - np.median(env)))) or float(np.std(env)) or 1e-12
    threshold = float(np.median(env)) + policy.threshold_mad * mad
    above = np.flatnonzero(env > threshold)
    if above.size == 0:
        return []

    # 문턱 초과 샘플을 merge_gap_ms 간격 안이면 한 사건으로 묶는다 (오디오 버스트의 영교차 등)
    events: list[tuple[float, float]] = []  # (시작, 끝)
    start = prev = float(t_ms[above[0]])
    for i in above[1:]:
        t = float(t_ms[i])
        if t - prev > policy.merge_gap_ms:
            events.append((start, prev))
            start = t
        prev = t
    events.append((start, prev))
    # 길이가 max_pulse_ms 이하인 사건만 두드림 후보 (시작 시각만 남긴다)
    pulses = [s for s, e in events if e - s <= policy.max_pulse_ms]

    # 연속 후보의 간격이 [lo, hi] 안이면 한 쌍. 짝지으면 두 칸, 아니면 한 칸 전진
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
    reference: list[DoubleTap],
    target: list[DoubleTap],
    policy: TapPolicy,
    *,
    max_drift_ppm: float = 0.0,
) -> list[Anchor]:
    """기준(바디캠)과 대상의 두 번 두드림을 짝짓는다.

    가능한 모든 짝의 오프셋 후보 중 허용 오차 안에서 가장 많은 짝을 설명하는 것을 고른다
    (어느 한쪽이 두드림을 놓치거나 잡음을 두드림으로 잡아도 견딘다).
    후보를 낸 짝에서 |Δt|만큼 떨어진 두드림은 두 시계의 드리프트가 쌓였을 수 있으므로
    허용 오차에 max_drift_ppm·|Δt|를 더한다 (sync.yaml max_drift_ppm). 20분 녹화의 80 ppm
    드리프트는 끝에서 약 96 ms다.

    Args:
        reference: 기준(바디캠 오디오)의 두 번 두드림.
        target: 대상 스트림의 두 번 두드림.
        policy: `sync.yaml tap` (`match_tolerance_ms`).
        max_drift_ppm: 드리프트 상한 ppm. 0이면 허용 오차를 넓히지 않는다.

    Returns:
        짝마다 앵커 두 개(첫 두드림 `"tap1"`, 두 번째 두드림 `"tap2"`). 같은 수의 짝을 설명하는
        후보가 여럿이면 먼저 찾은 것. 짝이 없으면 빈 목록. 계산량은 O(|ref|²·|tgt|²)지만 두드림은
        보통 몇 개뿐이다.
    """
    best: list[Anchor] = []
    # 모든 (기준, 대상) 쌍을 오프셋 후보로 시험한다 (RANSAC과 비슷한 전수 탐색)
    for r in reference:
        for t in target:
            offset = r.first_ms - t.first_ms
            anchors: list[Anchor] = []
            used: set[int] = set()  # 이미 짝지은 대상 인덱스 (대상 하나는 한 번만 쓴다)
            for rr in reference:
                for j, tt in enumerate(target):
                    # 후보를 낸 기준 두드림에서 멀수록 드리프트가 쌓일 수 있어 허용 오차를 넓힌다
                    tolerance = policy.match_tolerance_ms + max_drift_ppm * 1e-6 * abs(
                        rr.first_ms - r.first_ms
                    )
                    if j not in used and abs(rr.first_ms - (tt.first_ms + offset)) <= tolerance:
                        used.add(j)
                        anchors.append(Anchor(rr.first_ms, tt.first_ms, "tap1"))
                        anchors.append(Anchor(rr.second_ms, tt.second_ms, "tap2"))
                        break
            if len(anchors) > len(best):
                best = anchors
    return best
