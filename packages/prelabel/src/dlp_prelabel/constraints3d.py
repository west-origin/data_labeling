"""손 3D 물리 제약: 팔 길이 밖 거리, 속도·가속도 튐, 세션 손끝 거리와 다른 표본 (ADR 0054).

단안 깊이·손 크기 맞춤은 시각마다 따로 풀어 배율이 튈 수 있다 (2026-10-22 파일럿: 손목 거리
0.3~1.3 m가 오가고 표본의 약 5%가 3 m/s를 넘었다). 이 모듈은 `lift3d`가 낸 손 관절 표본에 제약을
검사해 표본마다 어긴 것(`Constraint3DFlag`)을 남기고, 정책(`depth.constraints.actions`)에 따라
신뢰도를 낮추거나(downweight) 표본을 뺀다(drop). 표시는 숨기지 않는다: 남긴 표본의 `flags`에 그대로
있다.

검사 (손 개체 `<left|right>_hand`의 카메라 좌표 표본만, 박스 중심은 보지 않는다):
- reach: 손목의 카메라 거리가 reach_m 범위 밖.
- speed: 손목 표본 i로 들어오는 속도와 나가는 속도가 모두 max_speed_mps 초과 (한 표본만 튄 것.
  이웃 간격이 max_dt_ms를 넘으면 세지 않는다).
- accel: 손목 표본 i 앞뒤 속도 차 / 간격이 max_accel_mps2 초과.
- bone_length: 손끝 표본의 손목 거리가 그 세션 같은 손·같은 손끝 중앙값에서 bone_tolerance 비율
  넘게 다름 (중앙값은 표본 min_bone_samples개 이상일 때만).
손목에 붙은 표시(reach·speed·accel)는 같은 시각 손끝 표본에도 붙인다 (한 시각의 손 맞춤은 손 전체의
배율을 함께 정하므로 손목이 튀면 손끝도 같이 튄다).
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from dlp_prelabel.policy import Constraints3DPolicy
from dlp_schema.labels import Constraint3DFlag, Trajectory3DSample

TIPS = ("thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip")

Key = tuple[str, str | None]
Flags = dict[int, set[Constraint3DFlag]]


def _xyz(s: Trajectory3DSample) -> tuple[float, float, float]:
    """표본 좌표."""
    return (s.x, s.y, s.z)


def _velocity(a: Trajectory3DSample, b: Trajectory3DSample) -> np.ndarray:
    """a→b 속도 벡터 (m/s)."""
    return (np.array(_xyz(b)) - np.array(_xyz(a))) / ((b.t_ms - a.t_ms) / 1000)


def wrist_flags(wrist: Sequence[Trajectory3DSample], policy: Constraints3DPolicy) -> Flags:
    """손목 표본 시각 → 어긴 제약 (reach·speed·accel)."""
    flags: Flags = defaultdict(set)
    low, high = policy.reach_m
    for s in wrist:
        if not low <= math.dist(_xyz(s), (0.0, 0.0, 0.0)) <= high:
            flags[s.t_ms].add(Constraint3DFlag.REACH)
    for a, b, c in zip(wrist, wrist[1:], wrist[2:], strict=False):
        if b.t_ms - a.t_ms > policy.max_dt_ms or c.t_ms - b.t_ms > policy.max_dt_ms:
            continue
        v_in, v_out = _velocity(a, b), _velocity(b, c)
        if min(np.linalg.norm(v_in), np.linalg.norm(v_out)) > policy.max_speed_mps:
            flags[b.t_ms].add(Constraint3DFlag.SPEED)
        dt = (c.t_ms - a.t_ms) / 2000
        if float(np.linalg.norm(v_out - v_in)) / dt > policy.max_accel_mps2:
            flags[b.t_ms].add(Constraint3DFlag.ACCEL)
    return flags


def bone_flags(
    wrist: Sequence[Trajectory3DSample],
    tip: Sequence[Trajectory3DSample],
    policy: Constraints3DPolicy,
) -> Flags:
    """손끝 표본 시각 → bone_length (세션 중앙값에서 bone_tolerance 비율 넘게 다른 손목 거리)."""
    at = {s.t_ms: s for s in wrist}
    lengths = {s.t_ms: math.dist(_xyz(s), _xyz(at[s.t_ms])) for s in tip if s.t_ms in at}
    flags: Flags = defaultdict(set)
    if len(lengths) < policy.min_bone_samples:
        return flags
    median = float(np.median(list(lengths.values())))
    for t, d in lengths.items():
        if median > 0 and abs(d - median) / median > policy.bone_tolerance:
            flags[t].add(Constraint3DFlag.BONE_LENGTH)
    return flags


def _apply(
    samples: Sequence[Trajectory3DSample], flags: Flags, policy: Constraints3DPolicy
) -> list[Trajectory3DSample]:
    """표시를 싣고 정책대로 신뢰도를 낮추거나 뺀 표본 목록."""
    out: list[Trajectory3DSample] = []
    for s in samples:
        found = flags.get(s.t_ms, set())
        if not found:
            out.append(s)
            continue
        actions = {policy.actions[f] for f in found}
        if "drop" in actions:
            continue
        conf = s.confidence
        if conf is not None:
            conf = round(conf * policy.downweight ** sum(a == "downweight" for a in actions), 4)
        out.append(s.model_copy(update={"flags": tuple(sorted(found)), "confidence": conf}))
    return out


def constrain_hands(
    samples: dict[Key, list[Trajectory3DSample]], policy: Constraints3DPolicy
) -> dict[Key, list[Trajectory3DSample]]:
    """`lift3d` 결과에 손 3D 제약을 적용한다. 손 개체가 아닌 키는 그대로. 꺼져 있으면 그대로.

    표본을 모두 뺀 (개체, 부위)는 결과에서 빠진다.
    """
    if not policy.enabled:
        return samples
    out = dict(samples)
    for entity in {e for e, part in samples if e.endswith("_hand") and part == "wrist"}:
        wrist = sorted(samples[(entity, "wrist")], key=lambda s: s.t_ms)
        hand = wrist_flags(wrist, policy)
        for part in ("wrist", *TIPS):
            key = (entity, part)
            if key not in samples:
                continue
            flags: Flags = defaultdict(set, {t: set(f) for t, f in hand.items()})
            if part != "wrist":
                for t, f in bone_flags(wrist, samples[key], policy).items():
                    flags[t] |= f
            kept = _apply(samples[key], flags, policy)
            if kept:
                out[key] = kept
            else:
                del out[key]
    return out


def flag_counts(samples: dict[Key, list[Trajectory3DSample]]) -> dict[str, int]:
    """표시별 표본 수 (요약 출력용)."""
    counts: dict[str, int] = defaultdict(int)
    for ss in samples.values():
        for s in ss:
            for f in s.flags or ():
                counts[f.value] += 1
    return dict(counts)
