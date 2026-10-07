"""1단: 손목 속도와 접촉 신호로 행동 경계 후보를 만든다.

후보의 종류:
- 접촉 시작·끝 (장갑 압력 또는 손 상태 접촉 구간)
- 정지 구간(속도 < still_speed가 min_still_ms 이상)의 시작과 끝: 손이 멈춤 / 다시 움직이기 시작
  (대기 구간, 접근 끝, 잡은 채 멈춤 뒤 옮기기 시작 등)
- 접촉 밖의 속도 골짜기(< valley_speed): 멈추지 않고 다음 목표로 방향을 바꾸는 순간
  (이탈 끝 = 다음 접근 시작). 접촉 중에는 문지르기처럼 왕복하는 움직임이 골짜기를 많이 만들므로
  쓰지 않는다.
가까운 후보(merge_ms 이내)는 하나로 합치며 접촉 경계를 우선한다. 경계는 VLM이 아니라 이 신호로만
정한다.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from dlp_actions.policy import BoundaryPolicy
from dlp_schema.labels import KeypointTrackPayload

WRIST, MIDDLE_MCP = 0, 9


@dataclass(frozen=True)
class Candidate:
    t_ms: int
    reason: str  # contact_start | contact_end | still_start | still_end | valley | edge


def wrist_series(
    track: KeypointTrackPayload,
) -> tuple[NDArray[np.int64], NDArray[np.float64], float]:
    """(시각, 손목 위치 (N, 2), 손바닥 길이 px 중앙값)."""
    times = np.array([f.t_ms for f in track.keyframes], dtype=np.int64)
    xy = np.array([[f.points[WRIST].x, f.points[WRIST].y] for f in track.keyframes])
    palm = [
        float(
            np.hypot(
                f.points[MIDDLE_MCP].x - f.points[WRIST].x,
                f.points[MIDDLE_MCP].y - f.points[WRIST].y,
            )
        )
        for f in track.keyframes
    ]
    return times, xy, float(np.median(palm)) if palm else 1.0


def speed(
    times: NDArray[np.int64], xy: NDArray[np.float64], scale: float, smooth_ms: int
) -> NDArray[np.float64]:
    """손바닥 길이/초 단위 속도 (중앙 차분 + 이동 평균)."""
    if len(times) < 2:
        return np.zeros(len(times))
    t = times.astype(np.float64) / 1000
    v = np.hypot(np.gradient(xy[:, 0], t), np.gradient(xy[:, 1], t)) / max(scale, 1e-6)
    dt = float(np.median(np.diff(times)))
    k = max(1, round(smooth_ms / dt)) if dt > 0 else 1
    if k > 1:
        v = np.convolve(
            np.pad(v, (k // 2, k - 1 - k // 2), mode="edge"), np.ones(k) / k, mode="valid"
        )
    return v


def _inside(t: int, intervals: list[tuple[int, int]]) -> bool:
    return any(s < t < e for s, e in intervals)


def boundary_candidates(
    times: NDArray[np.int64],
    xy: NDArray[np.float64],
    scale: float,
    contacts: list[tuple[int, int]],
    policy: BoundaryPolicy,
) -> list[Candidate]:
    v = speed(times, xy, scale, policy.smooth_ms)
    found: list[Candidate] = []
    for s, e in contacts:
        found += [Candidate(s, "contact_start"), Candidate(e, "contact_end")]

    still = v < policy.still_speed
    long_still = np.zeros(len(times), dtype=bool)  # min_still_ms 이상 이어진 정지
    i = 0
    while i < len(times):
        if not still[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(times) and still[j + 1]:
            j += 1
        if times[j] - times[i] >= policy.min_still_ms:
            long_still[i : j + 1] = True
            if i > 0:
                found.append(Candidate(int(times[i]), "still_start"))
            if j < len(times) - 1:
                found.append(Candidate(int(times[j]), "still_end"))
        i = j + 1

    last_valley = -(10**9)
    for k in range(1, len(times) - 1):
        t = int(times[k])
        if long_still[k] or v[k] >= policy.valley_speed or _inside(t, contacts):
            continue
        window = (times >= t - policy.valley_window_ms) & (times <= t + policy.valley_window_ms)
        before, after = window & (times < t), window & (times > t)
        deep = (
            before.any()
            and after.any()
            and v[k] <= policy.valley_ratio * v[before].max()
            and v[k] <= policy.valley_ratio * v[after].max()
        )
        if (
            deep
            and v[k] <= v[k - 1]
            and v[k] < v[k + 1]
            and t - last_valley >= policy.min_valley_separation_ms
        ):
            found.append(Candidate(t, "valley"))
            last_valley = t

    priority = {"contact_start": 0, "contact_end": 0, "still_start": 1, "still_end": 1, "valley": 2}
    merged: list[Candidate] = []
    for c in sorted(found, key=lambda c: (c.t_ms, priority[c.reason])):
        if merged and c.t_ms - merged[-1].t_ms <= policy.merge_ms:
            if priority[c.reason] < priority[merged[-1].reason]:
                merged[-1] = c
            continue
        merged.append(c)
    return merged


def segments(
    candidates: list[Candidate], start_ms: int, end_ms: int, min_segment_ms: int
) -> list[tuple[int, int]]:
    """후보로 [start_ms, end_ms]를 빈틈 없이 나눈다.

    min_segment_ms보다 짧은 조각은 앞 구간에 붙인다.
    """
    cuts = [start_ms, *(c.t_ms for c in candidates if start_ms < c.t_ms < end_ms), end_ms]
    out: list[tuple[int, int]] = []
    for a, b in itertools.pairwise(cuts):
        if (out and b - a < min_segment_ms) or (out and out[-1][1] - out[-1][0] < min_segment_ms):
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return out
