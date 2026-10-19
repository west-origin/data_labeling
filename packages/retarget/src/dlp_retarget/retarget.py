"""에피소드 하나를 로봇 관절 궤적으로 옮기고 도달 가능성 보고를 만든다 (ADR 0047).

입력: 에피소드 프레임 시각과 그 시각의 손 점 3개(세션 월드 좌표, 없으면 NaN). 출력: 프레임별 관절 값
(팔 관절 + 그리퍼), 목표 유무, 도달 여부, 위치·회전 오차, 속도 한계 표시와 `FeasibilityReport`.

정직성 규칙:
- 목표가 없는 프레임은 관절 값을 직전 프레임 값으로 두되 target_present=0으로 표시한다. 첫 목표 앞의
  프레임은 풀 해가 아직 없어 시작 자세(관절 범위 가운데)와 닫힌 그리퍼(`gripper.closed`)다. 어느
  쪽도 사람 동작에서 온 값이 아니다.
- 닿지 못한 프레임은 가장 가까운 해를 내되 reachable=0과 오차를 남긴다.
- 속도 한계: 팔 관절은 풀이기가(`ik.limit_step`), 그리퍼 관절은 여기서 같은 규칙(URDF velocity x
  velocity_scale x 프레임 간격, 직전 프레임 값 기준)으로 자르고 `velocity_limited`에 합친다.
- 목표가 하나도 없는 에피소드는 정책과 관계없이 버리기로 결정한다 (로봇 관절이 모두 시작 자세라 로봇
  데이터로 의미가 없다). 정책 `unreachable: drop_episode`는 비율 기준까지 더 본다.
  결정(`decision`)만 하고 실제로 빼는 것은 내보내기가 한다.
- orientation_mode none은 회전을 재지 않으므로 회전 오차를 NaN(내보내기 -1)으로 둔다 (0이면 방향이
  완벽히 맞은 것과 구분되지 않는다).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from dlp_retarget.ik import IkProblem, IkSolver, limit_step
from dlp_retarget.policy import RetargetPolicy, UnreachableAction
from dlp_retarget.robot import LoadedRobot, OrientationMode
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
    첫 관절에서 최대 팔 길이보다 먼 프레임 (어떤 자세로도 닿을 수 없음). velocity_limited_frames:
    팔·그리퍼 중 하나라도 속도 한계로 잘린 프레임 수. longest_unreachable_ms: 목표가 있는데 닿지
    못한 프레임이 이어진 가장 긴 구간의 첫·마지막 프레임 시각 차 (홀로 떨어진 프레임 하나면 0,
    미도달 유무는 reachable_frames < frames_with_target으로 본다). decision: 내보낼지(keep)
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
    """참이 이어진 가장 긴 구간의 길이 (ms, 첫 프레임 시각 ~ 마지막 프레임 시각, 한 프레임이면
    0)."""
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
    """정책에 따른 내보내기 결정과 이유 (목표가 없는 에피소드는 정책과 관계없이 drop)."""
    target_fraction = with_target / n if n else 0.0
    unreachable = 1 - reachable / with_target if with_target else 1.0
    if with_target == 0:
        return "drop", "손 목표가 있는 프레임이 없습니다 (월드 좌표 손 3D 없음)"
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
    held, grip_limited = _gripper_track(tg.present, tg.opening_m, robot, points.times_ms)
    velocity_limited = sol.velocity_limited | grip_limited
    orientation_error = sol.orientation_error_rad
    if cfg.ik.orientation_mode is OrientationMode.NONE:
        orientation_error = np.full(n, np.nan)  # 재지 않음 (모듈 설명)
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
        orientation_error_mean_rad=_stat(orientation_error, "mean"),
        orientation_error_max_rad=_stat(orientation_error, "max"),
        frames_out_of_reach=int(far.sum()),
        velocity_limited_frames=int(velocity_limited.sum()),
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
        orientation_error_rad=orientation_error,
        velocity_limited=velocity_limited,
        report=report,
    )


def _gripper_track(
    present: NDArray[np.bool_], opening_m: Vec, robot: LoadedRobot, times_ms: Vec
) -> tuple[Vec, NDArray[np.bool_]]:
    """프레임별 그리퍼 관절 값과 속도 한계 표시.

    목표가 있는 프레임은 열림 거리 대응 값(`gripper_values`), 없는 프레임은 직전 값(첫 목표 전에는
    closed)이다. 첫 프레임을 뺀 모든 프레임을 직전 프레임 값 기준 속도 한계(URDF 그리퍼 velocity x
    velocity_scale)로 자른다. 손가락 끝 깊이 잡음으로 열림 거리가 프레임마다 흔들려도 내보낸 값이
    로봇이 낼 수 없는 속도가 되지 않고, 잘린 프레임이 표시된다.
    """
    cfg = robot.config
    want = gripper_values(opening_m, cfg.arm.gripper)
    vmax = np.array([robot.gripper.velocity * cfg.ik.velocity_scale])
    n = present.size
    held = np.empty(n)
    limited = np.zeros(n, dtype=np.bool_)
    last = np.array([cfg.arm.gripper.closed])
    for k in range(n):
        cur = np.array([float(want[k])]) if present[k] else last
        if k > 0:
            cur, limited[k] = limit_step(cur, last, vmax, times_ms, k)
        held[k], last = float(cur[0]), cur
    return held, limited
