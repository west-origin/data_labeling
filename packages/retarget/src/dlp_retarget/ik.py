"""역기구학 풀이기 인터페이스와 내장 감쇠 최소제곱 풀이기 (ADR 0047).

`IkSolver`: 에피소드 하나의 프레임별 끝단 목표(`targets.Targets`) → 프레임별 관절 값과 오차.
풀이기는 목표를 바꾸지 않는다: 닿지 못하면 가장 가까운 해와 그 오차를 내고 `reachable`을 거짓으로
둔다 (목표를 조용히 작업 공간 안으로 잘라 "도달"로 만들지 않는다).

`DlsIkSolver` (내장, numpy, CPU): 프레임마다 직전 프레임 해에서 시작하는 감쇠
최소제곱(레벤버그-마쿼트)
반복  (JᵀJ + (λ + μ) I) Δq = JᵀE,  q ← clip(q + Δq, 하한, 상한)
- E: [위치 오차; w · 회전 오차], J: [선속도 야코비안; w · 각속도 야코비안]. 회전 오차는
  orientation_mode에 따라 없음(위치만), 접근 축(a x a*, 접근 축 둘레 회전은 자유), 접근·닫힘 축.
- μ (smoothness_weight): 반복마다 걸음을 줄이는 추가 감쇠. 직전 해에서 시작하므로 가까운 해(같은
  자세 가지)로 수렴해 매끄럽다. 목표에 맞는 해 자체를 끌어당기지는 않는다 (끌어당기면 오차가
  남는다).
  λ (damping): 특이점 근처 안정.
- 관절 위치 한계는 매 반복 자르고, 속도 한계(URDF velocity x velocity_scale x 프레임 간격)는 직전 해
  기준으로 자른 뒤 `velocity_limited`로 표시한다.
- 첫 목표 프레임과 목표가 끊긴 뒤 다시 시작하는 프레임은 여러 시작점(관절 범위 가운데 + 결정적 난수
  `restarts`개)에서 풀어 비용이 가장 작은 해를 쓴다.
- 목표가 없는 프레임은 직전 해를 그대로 두고(첫 목표 전에는 범위 가운데) 오차를 NaN으로 둔다.

mink(MuJoCo, Apache-2.0) 풀이기는 격리 환경 어댑터(`dlp_retarget.mink`)로 같은 인터페이스를 따른다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from dlp_retarget.robot import IkConfig, OrientationMode
from dlp_retarget.targets import Targets
from dlp_retarget.urdf import Chain

Vec = NDArray[np.float64]
# 반복을 멈추는 관절 변화 크기 (rad)
_STEP_EPS = 1e-7


@dataclass(frozen=True)
class IkProblem:
    """한 에피소드의 IK 문제.

    approach_axis / closing_axis: 끝 링크 좌표의 접근·닫힘 축.
    times_ms: 프레임 시각 (속도 한계에 쓴다).
    model_path: 로봇 모델 파일 (mink 풀이기가 MuJoCo로 읽는다, 내장 풀이기는 쓰지 않는다).
    """

    chain: Chain
    targets: Targets
    times_ms: Vec
    approach_axis: Vec
    closing_axis: Vec
    config: IkConfig
    model_path: Path | None = None


@dataclass
class IkSolution:
    """프레임별 결과 (n = 프레임 수). 목표가 없는 프레임의 오차는 NaN, reachable은 거짓."""

    q: NDArray[np.float64]  # (n, dof)
    position_error_m: Vec  # (n,)
    orientation_error_rad: Vec  # (n,)
    reachable: NDArray[np.bool_]  # (n,)
    velocity_limited: NDArray[np.bool_]  # (n,)


class IkSolver(Protocol):
    """IK 풀이기 어댑터. name·version은 리타기팅 버전에 들어간다."""

    name: str
    version: str

    def solve(self, problem: IkProblem) -> IkSolution:
        """프레임별 관절 값과 오차를 낸다."""
        ...


@dataclass(frozen=True)
class Goal:
    """한 프레임 목표 (로봇 기준 좌표 위치, 접근 축, 닫힘 축)."""

    position: Vec
    approach: Vec
    closing: Vec


def _angle(u: Vec, v: Vec) -> float:
    """두 단위 벡터 사이 각 (rad)."""
    return math.acos(min(max(float(u @ v), -1.0), 1.0))


def frame_errors(chain: Chain, q: Vec, goal: Goal, problem: IkProblem) -> tuple[float, float]:
    """(위치 오차 m, 회전 오차 rad). 회전 오차는 orientation_mode에 따라 0, 접근 축 각, 두 축 각의
    큰 쪽."""
    tip = chain.fk(q)
    pos = float(np.linalg.norm(goal.position - tip[:3, 3]))
    mode = problem.config.orientation_mode
    if mode is OrientationMode.NONE:
        return pos, 0.0
    a = tip[:3, :3] @ problem.approach_axis
    rot = _angle(a, goal.approach)
    if mode is OrientationMode.FULL:
        rot = max(rot, _angle(tip[:3, :3] @ problem.closing_axis, goal.closing))
    return pos, rot


def _system(
    chain: Chain, q: Vec, goal: Goal, problem: IkProblem
) -> tuple[Vec, NDArray[np.float64]]:
    """(오차 E, 야코비안 J) — 모듈 설명의 식."""
    cfg = problem.config
    tip, jac = chain.jacobian(q)
    e_p = goal.position - tip[:3, 3]
    if cfg.orientation_mode is OrientationMode.NONE:
        return e_p, jac[:3]
    w = cfg.orientation_weight
    rot = tip[:3, :3]
    a = rot @ problem.approach_axis
    if cfg.orientation_mode is OrientationMode.APPROACH:
        e_r = np.cross(a, goal.approach)
        # 접근 축 둘레 회전은 오차에 들지 않는다 (그 성분을 뺀다)
        proj = np.eye(3) - np.outer(goal.approach, goal.approach)
        j_r = proj @ jac[3:]
    else:
        c = rot @ problem.closing_axis
        e_r = 0.5 * (np.cross(a, goal.approach) + np.cross(c, goal.closing))
        j_r = jac[3:]
    return np.concatenate([e_p, w * e_r]), np.vstack([jac[:3], w * j_r])


def _iterate(chain: Chain, q0: Vec, mu: float, goal: Goal, problem: IkProblem) -> Vec:
    """감쇠 최소제곱 반복 (q0에서 시작, μ는 추가 감쇠)."""
    cfg = problem.config
    lo, hi = chain.lower, chain.upper
    q = np.clip(q0, lo, hi)
    eye = np.eye(q.size)
    for _ in range(cfg.max_iterations):
        e, jac = _system(chain, q, goal, problem)
        dq = np.linalg.solve(jac.T @ jac + (cfg.damping + mu) * eye, jac.T @ e)
        q = np.clip(q + dq, lo, hi)
        if float(np.linalg.norm(dq)) < _STEP_EPS:
            break
    return q


def seeds(chain: Chain, count: int) -> list[Vec]:
    """시작점: 관절 범위 가운데(한계 없는 관절은 0) + 결정적 난수 (count개)."""
    lo, hi = chain.lower, chain.upper
    flo = np.where(np.isfinite(lo), lo, -math.pi)
    fhi = np.where(np.isfinite(hi), hi, math.pi)
    rng = np.random.default_rng(0)
    out = [(flo + fhi) / 2]
    out += [rng.uniform(flo, fhi) for _ in range(count - 1)]
    return out


class DlsIkSolver:
    """내장 감쇠 최소제곱 풀이기 (모듈 설명). version: `dls1`."""

    name = "dls"
    version = "dls1"

    def _cost(self, chain: Chain, q: Vec, goal: Goal, problem: IkProblem) -> float:
        """시작점 비교 비용: 위치 오차 + w · 회전 오차."""
        pos, rot = frame_errors(chain, q, goal, problem)
        return pos + problem.config.orientation_weight * rot

    def _restart(self, chain: Chain, goal: Goal, problem: IkProblem) -> Vec:
        """여러 시작점 중 비용이 가장 작은 해."""
        best: tuple[float, Vec] | None = None
        for s in seeds(chain, problem.config.restarts):
            q = _iterate(chain, s, 0.0, goal, problem)
            cost = self._cost(chain, q, goal, problem)
            if best is None or cost < best[0]:
                best = (cost, q)
        assert best is not None
        return best[1]

    def solve(self, problem: IkProblem) -> IkSolution:
        """프레임마다 풀어 `IkSolution`을 낸다 (`IkSolver.solve`)."""
        chain, cfg, tg = problem.chain, problem.config, problem.targets
        n, dof = tg.present.size, len(chain.moving)
        out = IkSolution(
            q=np.zeros((n, dof)),
            position_error_m=np.full(n, np.nan),
            orientation_error_rad=np.full(n, np.nan),
            reachable=np.zeros(n, dtype=np.bool_),
            velocity_limited=np.zeros(n, dtype=np.bool_),
        )
        vmax = chain.velocity * cfg.velocity_scale
        q_prev = seeds(chain, 1)[0]
        had_target = False
        for k in range(n):
            if not tg.present[k]:
                out.q[k], had_target = q_prev, False
                continue
            goal = Goal(tg.position[k], tg.approach[k], tg.closing[k])
            if had_target:
                q = _iterate(chain, q_prev, cfg.smoothness_weight, goal, problem)
                dt = max(float(problem.times_ms[k] - problem.times_ms[k - 1]) / 1000, 0.0)
                step = np.clip(q - q_prev, -vmax * dt, vmax * dt)
                out.velocity_limited[k] = bool(np.any(np.abs(step - (q - q_prev)) > 1e-9))
                q = q_prev + step
            else:
                q = self._restart(chain, goal, problem)
            pos, rot = frame_errors(chain, q, goal, problem)
            out.q[k], out.position_error_m[k], out.orientation_error_rad[k] = q, pos, rot
            out.reachable[k] = pos <= cfg.position_tolerance_m and (
                cfg.orientation_mode is OrientationMode.NONE or rot <= cfg.orientation_tolerance_rad
            )
            q_prev, had_target = q, True
        return out
