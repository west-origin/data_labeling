"""에피소드 하나를 로봇 관절 궤적으로 옮기고 도달 가능성 보고를 만든다 (ADR 0047).

입력: 에피소드 프레임 시각과 그 시각의 손 점 3개(세션 월드 좌표, 없으면 NaN). 출력: 프레임별 관절 값
(팔 관절 + 그리퍼), 목표 유무, 도달 여부, 위치·회전 오차, 속도 한계 표시와 `FeasibilityReport`.

정직성 규칙: 목표가 없는 프레임은 관절 값을 직전 해로 두되 target_present=0으로 표시하고, 닿지 못한
프레임은 가장 가까운 해를 내되 reachable=0과 오차를 남긴다. 정책 `unreachable: drop_episode`는
비율이 상한을 넘는 에피소드를 버릴 것을 결정(`decision`)만 하고, 실제로 빼는 것은 내보내기가 한다.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from dlp_retarget.ik import IkProblem, IkSolver
from dlp_retarget.policy import RetargetPolicy, UnreachableAction
from dlp_retarget.robot import LoadedRobot
from dlp_retarget.targets import gripper_values, hand_targets
from dlp_retarget.urdf import Chain
from dlp_schema.common import Contract

Vec = NDArray[np.float64]


@dataclass(frozen=True)
class HandPoints:
    """프레임 시각(ms)과 손 점 (n, 3) 세 개 (세션 월드 좌표 m, 없으면 NaN)."""

    times_ms: Vec
    wrist: NDArray[np.float64]
    thumb: NDArray[np.float64]
    index: NDArray[np.float64]


class FeasibilityReport(Contract):
    """에피소드 도달 가능성 보고 (내보내기 meta/dlp_retarget.json, manifest details).

    fraction은 0~1. 오차 통계는 목표가 있는 프레임만 (없으면 None). frames_out_of_reach: 목표가 팔
    첫 관절에서 최대 팔 길이보다 먼 프레임 (어떤 자세로도 닿을 수 없음). longest_unreachable_ms:
    목표가 있는데 닿지 못한 프레임이 이어진 가장 긴 시간. decision: 정책에 따라 내보낼지(keep)
    버릴지(drop).
    """

    frames: int
    frames_with_target: int
    target_fraction: float
    reachable_frames: int
    reachable_fraction: float
    position_error_mean_m: float | None
    position_error_max_m: float | None
    orientation_error_mean_rad: float | None
    orientation_error_max_rad: float | None
    frames_out_of_reach: int
    velocity_limited_frames: int
    longest_unreachable_ms: float
    decision: Literal["keep", "drop"]
    reason: str | None = None


@dataclass
class RetargetResult:
    """프레임별 결과 (n = 프레임 수)와 보고."""

    joint_names: tuple[str, ...]  # 팔 관절 + 그리퍼 관절
    q: NDArray[np.float64]  # (n, len(joint_names)) rad
    target_present: NDArray[np.bool_]
    reachable: NDArray[np.bool_]
    position_error_m: Vec
    orientation_error_rad: Vec
    velocity_limited: NDArray[np.bool_]
    report: FeasibilityReport


def retarget_version(robot: LoadedRobot, policy: RetargetPolicy, solver: IkSolver) -> str:
    """리타기팅 판: `retarget-<풀이기>-<판>+r<로봇 설정 해시>+m<모델 sha256 앞 12자>+p<정책
    해시>`."""
    return (
        f"retarget-{solver.name}-{solver.version}+r{robot.config.digest()}"
        f"+m{robot.model_sha256[:12]}+p{policy.digest()}"
    )


def max_reach(chain: Chain) -> tuple[Vec, float]:
    """(첫 움직이는 관절 위치, 그 점에서 끝까지 링크 길이 합) — 어떤 자세로도 넘을 수 없는 거리."""
    q0 = np.zeros(len(chain.moving))
    f = chain.frames(q0)
    pts = [*f.origins, f.tip[:3, 3]]
    length = sum(float(np.linalg.norm(b - a)) for a, b in itertools.pairwise(pts))
    # prismatic 관절은 이동 범위만큼 더 닿는다
    length += sum(max(abs(j.lower), abs(j.upper)) for j in chain.moving if j.kind == "prismatic")
    return f.origins[0], length


def _longest_run_ms(mask: NDArray[np.bool_], times: Vec) -> float:
    """참이 이어진 가장 긴 구간의 길이 (ms, 첫 프레임 시각 ~ 마지막 프레임 시각)."""
    best, start = 0.0, None
    for k, m in enumerate(mask):
        if m and start is None:
            start = k
        if start is not None and (not m or k == mask.size - 1):
            end = k if m else k - 1
            best = max(best, float(times[end] - times[start]))
            start = None
    return best


def _decide(
    n: int, with_target: int, reachable: int, policy: RetargetPolicy
) -> tuple[Literal["keep", "drop"], str | None]:
    """정책에 따른 내보내기 결정과 이유."""
    target_fraction = with_target / n if n else 0.0
    unreachable = 1 - reachable / with_target if with_target else 1.0
    if policy.unreachable is UnreachableAction.MARK:
        return "keep", None
    if target_fraction < policy.min_target_fraction:
        return (
            "drop",
            f"손 목표가 있는 프레임 비율 {target_fraction:.2f} < {policy.min_target_fraction}",
        )
    if unreachable > policy.max_unreachable_fraction:
        return "drop", f"도달하지 못한 비율 {unreachable:.2f} > {policy.max_unreachable_fraction}"
    return "keep", None


def _stat(values: Vec, fn: str) -> float | None:
    """NaN을 뺀 평균·최대 (값이 없으면 None, 소수 넷째 자리)."""
    v = values[~np.isnan(values)]
    if v.size == 0:
        return None
    return round(float(v.mean() if fn == "mean" else v.max()), 4)


def retarget_episode(
    points: HandPoints, robot: LoadedRobot, policy: RetargetPolicy, solver: IkSolver
) -> RetargetResult:
    """손 점 → 관절 궤적·도달 표시·보고 (모듈 설명)."""
    cfg = robot.config
    tg = hand_targets(points.wrist, points.thumb, points.index, cfg.arm.placement)
    problem = IkProblem(
        chain=robot.chain,
        targets=tg,
        times_ms=points.times_ms,
        approach_axis=np.array(cfg.arm.approach_axis, dtype=np.float64),
        closing_axis=np.array(cfg.arm.closing_axis, dtype=np.float64),
        config=cfg.ik,
        model_path=robot.model_path,
    )
    sol = solver.solve(problem)
    n = tg.present.size
    grip = gripper_values(tg.opening_m, cfg.arm.gripper)
    held = np.empty(n)
    last = cfg.arm.gripper.closed
    for k in range(n):
        last = float(grip[k]) if tg.present[k] else last
        held[k] = last
    q = np.column_stack([sol.q, held])
    origin, reach = max_reach(robot.chain)
    far = tg.present & (np.linalg.norm(np.nan_to_num(tg.position) - origin, axis=1) > reach)
    with_target, reachable = int(tg.present.sum()), int(sol.reachable.sum())
    decision, reason = _decide(n, with_target, reachable, policy)
    report = FeasibilityReport(
        frames=n,
        frames_with_target=with_target,
        target_fraction=round(with_target / n, 4) if n else 0.0,
        reachable_frames=reachable,
        reachable_fraction=round(reachable / with_target, 4) if with_target else 0.0,
        position_error_mean_m=_stat(sol.position_error_m, "mean"),
        position_error_max_m=_stat(sol.position_error_m, "max"),
        orientation_error_mean_rad=_stat(sol.orientation_error_rad, "mean"),
        orientation_error_max_rad=_stat(sol.orientation_error_rad, "max"),
        frames_out_of_reach=int(far.sum()),
        velocity_limited_frames=int(sol.velocity_limited.sum()),
        longest_unreachable_ms=_longest_run_ms(tg.present & ~sol.reachable, points.times_ms),
        decision=decision,
        reason=reason,
    )
    return RetargetResult(
        joint_names=robot.joint_names,
        q=q,
        target_present=tg.present,
        reachable=sol.reachable,
        position_error_m=sol.position_error_m,
        orientation_error_rad=sol.orientation_error_rad,
        velocity_limited=sol.velocity_limited,
        report=report,
    )
