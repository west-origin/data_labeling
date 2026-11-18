"""다지 손 역기구학: 손목·손가락 끝 위치를 팔 관절과 손가락 관절로 함께 푼다 (ADR 0068).

`HandIkSolver`: 에피소드 하나의 프레임별 손목·손가락 끝 목표(`targets.HandTargets`) → 프레임별 관절
값(팔 관절 + 손가락 관절)과 손목·손가락별 오차. 팔 IK(`dlp_retarget.ik`)와 같은 규칙을 따른다:
목표를 바꾸지 않고, 닿지 못하면 가장 가까운 해와 그 오차를 내며, 관절 위치·속도 한계를 지킨다.

기구학: 로봇은 팔 끝(손목 링크)에서 손가락들이 갈라지는 나무다 (`urdf.Tree`, 끝 0 = 손목 링크,
끝 i + 1 = 손가락 i의 끝 링크). 관절 값 벡터는 팔 관절 + 손가락 1 관절 + 손가락 2 관절 + ... 이고,
손가락마다 자기 관절이 따로 있다 (로더 `robot.load_any_robot_file`이 확인한다). 손가락 끝 위치는
팔 관절과 그 손가락의 관절에만 따른다.

`DlsHandSolver` (내장, numpy, CPU): 프레임마다 감쇠 최소제곱 반복
    (JᵀJ + (λ + μ) I) Δq = JᵀE,  q ← clip(q + Δq, 하한, 상한)
- E: [w · 손목 위치 오차; 손가락 끝 위치 오차들], J: 같은 행의 위치 야코비안 (w = wrist_weight).
  방향 오차 행은 없다. 사람 손의 방향은 손목에서 본 손가락 끝들의 위치에 들어 있다. 다만 손가락
  끝 위치만으로는 손바닥 방향이 하나로 정해지지 않는다 (손가락 관절이 남는 방향을 메운다). 직전
  해에서 가장 적게 움직이는 해로 이어 가므로 손바닥 방향과 손가락 마디 각은 사람과 다를 수 있다.
- 손가락 끝 점이 없는 손가락은 그 프레임에 목표가 없다: 그 손가락의 오차 행과 관절 열을 빼고
  풀어 그 관절은 직전 프레임 값을 그대로 둔다 (값을 지어내지 않는다). 나머지 손가락과 팔은 푼다.
  손가락 끝이 모두 없으면 팔만 손목 위치를 따라간다.
- μ(smoothness_weight), λ(damping), 관절 한계 자르기, 속도 한계(`ik.limit_step`, 팔·손가락 모든
  관절, 첫 프레임을 뺀 모든 프레임), 시작점(`ik.seeds`)은 팔 IK와 같다.
- 다시 시작: 손목 목표가 없던 프레임 뒤, 또는 어느 손가락의 목표가 새로 생긴 프레임은 여러
  시작점(직전 값 + 관절 범위 가운데 + 결정적 난수)에서 풀어, 도달한 해 가운데 직전 값에 가장
  가까운 것을 쓴다 (없으면 비용이 가장 작은 해). 시작점마다 목표가 없는 손가락의 관절은 직전 값에
  묶어 둔다. 오래 유지한 손가락이 낡은 자세에서 출발해 관절 한계에 걸리는 일을 줄인다.
- 손목 목표가 없는 프레임은 모든 관절을 직전 값으로 두고 오차를 NaN으로 둔다 (첫 목표 전에는 시작
  자세 = 관절 범위 가운데).

mink 풀이기(`dlp_retarget.mink`)는 팔 하나 + 평행 그리퍼만 푼다. 다지 손 로봇에 mink를 고르면
`make_hand_solver`가 거부한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from dlp_retarget.ik import limit_step, seeds
from dlp_retarget.policy import RetargetPolicy, SolverName
from dlp_retarget.robot import HandIkConfig, RobotError
from dlp_retarget.targets import HandTargets
from dlp_retarget.urdf import Tree

Vec = NDArray[np.float64]
Mask = NDArray[np.bool_]
# 반복을 멈추는 관절 변화 크기 (|Δq| 노름, 회전 관절 rad·직선 관절 m)
_STEP_EPS = 1e-7


@dataclass(frozen=True)
class HandIkProblem:
    """한 에피소드의 다지 손 IK 문제.

    tree: 기준 링크 → [손목 링크, 손가락 끝 링크...] 나무 (`robot.LoadedDexterousRobot.tree`).
    targets: 프레임별 손목·손가락 끝 목표 (로봇 기준 좌표, `targets.finger_targets`). 손가락 수와
        순서가 나무의 끝과 같아야 한다.
    times_ms: 프레임 시각 ms (에피소드 마스터 시각, 속도 한계의 프레임 간격에만 쓴다).
    config: 로봇의 다지 손 IK 설정.
    fingertip_tolerance_m: 손가락 끝 도달 허용 오차 (`DexterousHandConfig.fingertip_tolerance_m`).
    """

    tree: Tree
    targets: HandTargets
    times_ms: Vec
    config: HandIkConfig
    fingertip_tolerance_m: float


@dataclass
class HandIkSolution:
    """프레임별 결과 (n = 프레임 수, F = 손가락 수).

    손목 목표가 없는 프레임의 오차는 NaN, reachable은 거짓이다. 손가락 끝 목표가 없는 칸의 오차는
    NaN, tip_reachable은 거짓이다. reachable은 손목과 목표가 있는 손가락 끝이 모두 허용 오차 안일
    때 참이다 (목표가 없는 손가락은 판정에 넣지 않는다, `targets.tip_present`로 가린다).
    """

    q: NDArray[np.float64]  # (n, dof) 회전 관절 rad, 직선 관절 m
    wrist_error_m: Vec  # (n,)
    tip_error_m: NDArray[np.float64]  # (n, F)
    reachable: Mask  # (n,)
    tip_reachable: Mask  # (n, F)
    velocity_limited: Mask  # (n,)


class HandIkSolver(Protocol):
    """다지 손 IK 풀이기 어댑터. name·version은 리타기팅 버전에 들어간다."""

    name: str
    version: str

    def solve(self, problem: HandIkProblem) -> HandIkSolution:
        """프레임별 관절 값과 손목·손가락별 오차를 낸다."""
        ...


@dataclass(frozen=True)
class _Model:
    """에피소드 동안 같은 값: 나무, 관절 한계, 팔 관절 열, 손가락마다 자기 관절 열, 설정."""

    tree: Tree
    lower: Vec
    upper: Vec
    arm: Mask  # (dof,) 손목 경로의 관절
    own: Mask  # (F, dof) 손가락마다 자기 관절 (손목 경로에 없는 것)
    config: HandIkConfig
    tolerance_m: float  # 손가락 끝 허용 오차


@dataclass(frozen=True)
class _Goal:
    """한 프레임 목표.

    points: 손목과 손가락 끝 목표 (F + 1, 3), 목표가 없는 손가락은 0 (rows가 뺀다). tips: 손가락 끝
    목표 유무 (F,). rows: 쓰는 오차 행 (3 (F + 1),). cols: 푸는 관절 열 (dof,). weights: 행 가중치.
    """

    points: NDArray[np.float64]
    tips: Mask
    rows: Mask
    cols: Mask
    weights: Vec


def _model(problem: HandIkProblem) -> _Model:
    """문제 → 풀이에 쓰는 고정 값.

    Raises:
        ValueError: 목표의 손가락 수가 나무의 끝과 다를 때, 손가락의 자기 관절이 없을 때.
    """
    tree = problem.tree
    arm = tree.on_path[0]
    own = tree.on_path[1:] & ~arm
    if problem.targets.tips.shape[1] != own.shape[0]:
        raise ValueError(
            f"손가락 끝 목표 {problem.targets.tips.shape[1]}개: 나무의 손가락 {own.shape[0]}개와 "
            "다릅니다"
        )
    if not own.any(axis=1).all():
        raise ValueError("자기 관절이 없는 손가락이 있습니다 (손목 링크 아래의 끝 링크가 아님)")
    return _Model(
        tree, tree.lower, tree.upper, arm, own, problem.config, problem.fingertip_tolerance_m
    )


def _goal(model: _Model, targets: HandTargets, k: int) -> _Goal:
    """프레임 k의 목표 (손목은 늘 있고, 손가락은 끝 점이 있는 것만)."""
    tips = targets.tip_present[k]
    points = np.vstack([targets.wrist[k], np.where(tips[:, None], targets.tips[k], 0.0)])
    rows = np.repeat(np.concatenate([[True], tips]), 3)
    cols = model.arm | model.own[tips].any(axis=0)
    weights = np.ones(rows.size)
    weights[:3] = model.config.wrist_weight
    return _Goal(points, tips, rows, cols, weights)


def _iterate(model: _Model, q0: Vec, mu: float, goal: _Goal) -> Vec:
    """감쇠 최소제곱 반복 (q0에서 시작, μ는 추가 감쇠). goal.cols 밖의 관절은 q0 그대로다."""
    cfg = model.config
    lo, hi = model.lower[goal.cols], model.upper[goal.cols]
    q = np.clip(q0, model.lower, model.upper)
    eye = np.eye(lo.size)
    for _ in range(cfg.max_iterations):
        pos, jac = model.tree.jacobian(q)
        e = ((goal.points - pos).reshape(-1) * goal.weights)[goal.rows]
        j = (jac * goal.weights[:, None])[goal.rows][:, goal.cols]
        dq = np.linalg.solve(j.T @ j + (cfg.damping + mu) * eye, j.T @ e)
        q[goal.cols] = np.clip(q[goal.cols] + dq, lo, hi)
        if float(np.linalg.norm(dq)) < _STEP_EPS:
            break
    return q


def _errors(model: _Model, q: Vec, goal: _Goal) -> tuple[float, Vec]:
    """(손목 위치 오차 m, 손가락 끝 위치 오차 (F,) m). 목표가 없는 손가락은 NaN."""
    dist = np.linalg.norm(goal.points - model.tree.positions(q), axis=1)
    return float(dist[0]), np.where(goal.tips, dist[1:], np.nan)


def _within(model: _Model, wrist: float, tips: Vec, goal: _Goal) -> tuple[bool, Mask]:
    """(프레임 도달, 손가락별 도달 (F,)).

    손가락은 목표가 있고 오차 ≤ 손가락 끝 허용 오차일 때, 프레임은 손목 오차 ≤
    position_tolerance_m이고 목표가 있는 손가락이 모두 도달일 때 참이다.
    """
    reached = goal.tips & (np.nan_to_num(tips, nan=np.inf) <= model.tolerance_m)
    frame = wrist <= model.config.position_tolerance_m and bool((reached | ~goal.tips).all())
    return frame, reached


class DlsHandSolver:
    """내장 다지 손 감쇠 최소제곱 풀이기 (모듈 설명). version: `dlshand1`."""

    name = "dls"
    version = "dlshand1"

    def _restart(self, model: _Model, goal: _Goal, q_prev: Vec, solved: bool) -> Vec:
        """여러 시작점에서 풀어 고른 해.

        시작점은 직전 값(풀었던 적이 있을 때) + `ik.seeds`다. 목표가 없는 손가락의 관절은 모든
        시작점에서 직전 값이다. 풀었던 적이 있으면 도달한 해 가운데 직전 값에 가장 가까운 것,
        풀었던 적이 없거나 도달한 해가 없으면 비용(w · 손목 오차 + 손가락 끝 오차 합)이 가장
        작은 것.
        """
        cfg = model.config
        starts = ([q_prev] if solved else []) + seeds(model.tree, cfg.restarts)
        best: tuple[float, Vec] | None = None
        near: tuple[float, Vec] | None = None
        for s in starts:
            q = _iterate(model, np.where(goal.cols, s, q_prev), 0.0, goal)
            wrist, tips = _errors(model, q, goal)
            if solved and _within(model, wrist, tips, goal)[0]:
                dist = float(np.linalg.norm(q - q_prev))
                if near is None or dist < near[0]:
                    near = (dist, q)
            cost = cfg.wrist_weight * wrist + float(np.nansum(tips))
            if best is None or cost < best[0]:
                best = (cost, q)
        chosen = near if near is not None else best
        assert chosen is not None
        return chosen[1]

    def solve(self, problem: HandIkProblem) -> HandIkSolution:
        """프레임마다 풀어 `HandIkSolution`을 낸다 (`HandIkSolver.solve`).

        Raises:
            ValueError: 목표의 손가락 수가 나무와 다를 때 (`_model`).
        """
        model, cfg, tg = _model(problem), problem.config, problem.targets
        n, fingers = tg.present.size, tg.tips.shape[1]
        out = HandIkSolution(
            q=np.zeros((n, model.lower.size)),
            wrist_error_m=np.full(n, np.nan),
            tip_error_m=np.full((n, fingers), np.nan),
            reachable=np.zeros(n, dtype=np.bool_),
            tip_reachable=np.zeros((n, fingers), dtype=np.bool_),
            velocity_limited=np.zeros(n, dtype=np.bool_),
        )
        vmax = model.tree.velocity * cfg.velocity_scale
        q_prev = seeds(model.tree, 1)[0]  # 첫 목표 전 시작 자세 (관절 범위 가운데)
        solved = False  # 한 번이라도 풀었는가 (다시 시작할 때 직전 값을 시작점에 넣는다)
        had = np.zeros(fingers, dtype=np.bool_)  # 직전 프레임에 목표가 있던 손가락
        had_target = False
        for k in range(n):
            if not tg.present[k]:
                out.q[k], had_target, had = q_prev, False, np.zeros(fingers, dtype=np.bool_)
                continue
            goal = _goal(model, tg, k)
            if had_target and not bool((goal.tips & ~had).any()):
                q = _iterate(model, q_prev, cfg.smoothness_weight, goal)
            else:
                q = self._restart(model, goal, q_prev, solved)
            if k > 0:
                q, out.velocity_limited[k] = limit_step(q, q_prev, vmax, problem.times_ms, k)
            wrist, tips = _errors(model, q, goal)
            out.q[k], out.wrist_error_m[k], out.tip_error_m[k] = q, wrist, tips
            out.reachable[k], out.tip_reachable[k] = _within(model, wrist, tips, goal)
            q_prev, solved, had_target, had = q, True, True, goal.tips
        return out


def make_hand_solver(policy: RetargetPolicy) -> HandIkSolver:
    """정책의 풀이기로 다지 손을 풀 수 있는지 확인하고 내장 풀이기를 돌려준다.

    Raises:
        RobotError: `retarget.yaml solver`가 mink일 때 (mink 풀이기는 다지 손을 풀지 않는다).
    """
    if policy.solver is not SolverName.DLS:
        raise RobotError(
            f"다지 손 로봇은 내장 풀이기(solver: dls)만 풉니다: retarget.yaml solver가 "
            f"{policy.solver.value}입니다 (mink 풀이기는 팔 하나 + 평행 그리퍼만 지원, F-COE-92)"
        )
    return DlsHandSolver()
