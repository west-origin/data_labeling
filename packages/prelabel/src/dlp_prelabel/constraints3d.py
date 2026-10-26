"""손 3D 물리 제약: 팔 길이 밖 거리, 속도·가속도 튐, 세션 손끝 거리와 다른 표본 (ADR 0054).

단안 깊이·손 크기 맞춤은 시각마다 따로 풀어 배율이 튈 수 있다 (2026-10-22 파일럿: 손목 거리
0.3~1.3 m가 오가고 표본의 약 5%가 3 m/s를 넘었다). 이 모듈은 `lift3d`가 낸 손 관절 표본에 제약을
검사해 표본마다 어긴 것(`Constraint3DFlag`)을 남기고, 정책(`depth.constraints.actions`)에 따라
신뢰도를 낮추거나(downweight) 표본을 뺀다(drop). 표시는 숨기지 않는다: 남긴 표본의 `flags`에 그대로
있다.

검사 (손 개체 `<left|right>_hand`의 카메라 좌표 표본만, 박스 중심은 보지 않는다):
- reach: 손목의 카메라 거리가 reach_m 범위 밖.
- speed: 손목 표본 b로 들어오는 속도와 나가는 속도가 모두 max_speed_mps 초과 (한 표본만 튄 것).
- accel: 손목 표본 b의 |v_out - v_in| / ((t_c - t_a) / 2)가 max_accel_mps2 초과 (a·c는 앞뒤 표본).
  speed·accel 모두 앞뒤 이웃 간격 하나라도 max_dt_ms를 넘으면 세지 않는다.
- bone_length: 손끝 표본의 손목 거리가 그 세션 같은 손·같은 손끝 중앙값에서 bone_tolerance 비율
  넘게 다름 (중앙값은 표본 min_bone_samples개 이상일 때만).
손목에 붙은 표시(reach·speed·accel)는 같은 시각 손끝 표본에도 붙인다 (한 시각의 손 맞춤은 손 전체의
배율을 함께 정하므로 손목이 튀면 손끝도 같이 튄다).

착용자 팔 (`constrain_arms`, ADR 0058): 팔꿈치 표본(`<left|right>_arm`, elbow)의 같은 손 손목
(`<left|right>_hand`, wrist) 거리가 forearm_m 범위 밖이거나, 그 세션 같은 손 중앙값에서
forearm_tolerance 비율 넘게 다르면 forearm_length (중앙값은 표본 min_bone_samples개 이상일 때만).
같은 시각 손목 표본이 없으면 검사하지 않는다. 손목은 `constrain_hands`를 거친 뒤의 표본이라
drop된 손목 시각의 팔꿈치는 검사하지 않는다 (`lift3d`가 손 → 팔 순서로 부른다).
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
        # actions는 집합이라 표시가 여럿이어도 downweight는 한 번만 곱한다
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


def forearm_lengths(
    wrist: Sequence[Trajectory3DSample], elbow: Sequence[Trajectory3DSample]
) -> dict[int, float]:
    """같은 시각 손목→팔꿈치 거리 (m). 손목 표본이 없는 시각은 빠진다."""
    at = {s.t_ms: s for s in wrist}
    return {s.t_ms: math.dist(_xyz(s), _xyz(at[s.t_ms])) for s in elbow if s.t_ms in at}


def forearm_flags(
    wrist: Sequence[Trajectory3DSample],
    elbow: Sequence[Trajectory3DSample],
    policy: Constraints3DPolicy,
) -> Flags:
    """팔꿈치 표본 시각 → forearm_length (범위 밖, 또는 세션 중앙값에서 벗어남)."""
    lengths = forearm_lengths(wrist, elbow)
    flags: Flags = defaultdict(set)
    low, high = policy.forearm_m
    median = (
        float(np.median(list(lengths.values())))
        if len(lengths) >= policy.min_bone_samples
        else None
    )
    for t, d in lengths.items():
        off = (
            median is not None
            and median > 0
            and abs(d - median) / median > policy.forearm_tolerance
        )
        if not low <= d <= high or off:
            flags[t].add(Constraint3DFlag.FOREARM_LENGTH)
    return flags


def constrain_arms(
    samples: dict[Key, list[Trajectory3DSample]], policy: Constraints3DPolicy
) -> dict[Key, list[Trajectory3DSample]]:
    """팔꿈치 표본에 아래팔 길이 제약을 적용한다 (ADR 0058). 꺼져 있거나 팔이 없으면 그대로.

    손목은 같은 손 개체 `<left|right>_hand`의 wrist 표본이다 (팔 개체 `<left|right>_arm`).
    `constrain_hands` 뒤의 표본이라 drop된 손목 시각의 팔꿈치는 검사하지 않는다. 표본을 모두
    뺀 팔꿈치 키는 결과에서 빠진다.
    """
    if not policy.enabled:
        return samples
    out = dict(samples)
    for entity, part in [k for k in samples if k[0].endswith("_arm") and k[1] == "elbow"]:
        wrist = samples.get((entity.removesuffix("_arm") + "_hand", "wrist"), [])
        kept = _apply(
            samples[(entity, part)], forearm_flags(wrist, samples[(entity, part)], policy), policy
        )
        if kept:
            out[(entity, part)] = kept
        else:
            del out[(entity, part)]
    return out
