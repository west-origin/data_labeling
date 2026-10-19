"""역기구학 풀이기 인터페이스와 내장 감쇠 최소제곱 풀이기 (ADR 0047·0053).

`IkSolver`: 에피소드 하나의 프레임별 끝단 목표(`targets.Targets`) → 프레임별 관절 값과 오차.
풀이기는 목표를 바꾸지 않는다: 닿지 못하면 가장 가까운 해와 그 오차를 내고 `reachable`을 거짓으로
둔다 (목표를 조용히 작업 공간 안으로 잘라 "도달"로 만들지 않는다).

좌표: 목표와 야코비안은 모두 로봇 기준 링크 좌표다. 접근 축 a·닫힘 축 c는 끝 링크 좌표의 고정
벡터(`approach_axis`·`closing_axis`)를 끝 링크 회전 R로 돌린 값(a = R a_local)이고, 목표 축
a*·c*(사람 손에서 만든 방향)와 비교한다.

`DlsIkSolver` (내장, numpy, CPU): 프레임마다 감쇠 최소제곱(레벤버그-마쿼트) 반복
    (JᵀJ + (λ + μ) I) Δq = JᵀE,  q ← clip(q + Δq, 하한, 상한)
- E: [위치 오차; w · 회전 오차], J: [선속도 야코비안; w · 각속도 야코비안] (w = orientation_weight,
  회전 1 rad를 위치 w m로 본다). 회전 오차는 orientation_mode에 따라
  - none: 없음 (위치 3행만).
  - approach: a x a* (작은 각에서 a를 a*로 돌리는 회전 벡터). 각속도 야코비안에 투영 (I - a* a*ᵀ)를
    곱해 접근 축 둘레 회전(롤)이 비용에 들지 않게 한다 (롤은 자유).
  - full: (a x a* + c x c*) / 2. 두 축을 함께 맞추면 끝 링크 회전 전체(롤 포함)가 정해진다.
- 관절이 목표 자유도보다 적으면(5축 팔 + full 등) 정확히 맞출 수 없는 성분은 가중 최소제곱이 위치와
  나눠 진다. 남은 오차는 그대로 보고한다.
- μ (smoothness_weight): 반복마다 걸음을 줄이는 추가 감쇠. 직전 해에서 시작하므로 가까운 해(같은
  자세 가지)로 수렴해 매끄럽다. 목표에 맞는 해 자체를 끌어당기지는 않는다 (끌어당기면 오차가
  남는다). λ (damping): 특이점 근처(JᵀJ가 거의 특이) 안정.
- 관절 위치 한계는 매 반복 자른다 (clip). 반복이 한계에 막히면 그 방향으로는 더 가지 않는다.
- 속도 한계(URDF velocity x velocity_scale x 프레임 간격)는 첫 프레임을 뺀 모든 프레임에서 직전
  프레임 관절 값(내보내는 값) 기준으로 자르고 `velocity_limited`로 표시한다. 목표가 끊긴 뒤 다시
  시작하는 프레임도 예외가 아니다 (직전 프레임은 유지한 값이다): 그렇지 않으면 내보낸 action에 실제
  로봇이 낼 수 없는 한 걸음이 들어간다. 잘린 뒤 남은 오차는 reachable·오차로 드러난다.
- 첫 목표 프레임과 목표가 끊긴 뒤 다시 시작하는 프레임은 여러 시작점(직전 해(있으면) + 관절 범위
  가운데 + 결정적 난수 `restarts`개)에서 풀고, 도달 가능한 해 가운데 직전 해에 가장 가까운 것을 쓴다
  (도달 가능한 해가 없으면 비용이 가장 작은 해). 비용이 거의 같은 해들 사이에서 난수 시작점이
  이겨 관절이 다른 가지로 뛰는 일을 막는다.
- 목표가 없는 프레임은 직전 프레임 값을 그대로 두고(첫 목표 전에는 시작 자세 = 관절 범위 가운데)
  오차를 NaN으로 둔다.

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
    """(위치 오차 m, 회전 오차 rad). 회전 오차는 orientation_mode에 따라 0(none, 재지 않음), 접근 축
    각(approach, 롤은 보지 않음), 접근·닫힘 축 각의 큰 쪽(full)."""
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
    """(오차 E, 야코비안 J) — 모듈 설명의 식.

    a x a*는 a를 a*로 돌리는 회전축 방향, 크기 sin(각)이다. 작은 각에서 각속도 ω로 a가 바뀌는 속도는
    ω x a이므로 J의 각속도 행과 같은 단위(rad)로 맞출 수 있다.
    """
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
    """시작점: 관절 범위 가운데(한계 없는 관절은 0) + 결정적 난수 (count개).

    한계가 없는 관절(continuous)은 [-π, π]에서 고른다. 난수 씨앗이 고정이라 같은 입력이면 같은 해다.
    """
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
    # dls2: 다시 시작할 때 직전 해를 시작점에 넣고 직전 해에 가까운 도달 해를 고르며, 다시 시작하는
    # 프레임에도 속도 한계를 적용한다 (ADR 0053)
    version = "dls2"

    def _restart(self, chain: Chain, goal: Goal, problem: IkProblem, q_prev: Vec | None) -> Vec:
        """여러 시작점에서 풀어 고른 해.

        직전 해가 있으면 도달 해 가운데 직전 해에 가장 가까운 것, 없거나 도달 해가 없으면 비용(위치 오차
        + w · 회전 오차)이 가장 작은 것.
        """
        cfg = problem.config
        starts = ([] if q_prev is None else [q_prev]) + seeds(chain, cfg.restarts)
        best: tuple[float, Vec] | None = None  # 비용 최소 해 (직전 해가 없거나 도달 해가 없을 때)
        near: tuple[float, Vec] | None = None  # 도달 해 중 직전 해와 거리 최소
        for s in starts:
            q = _iterate(chain, s, 0.0, goal, problem)
            pos, rot = frame_errors(chain, q, goal, problem)
            if q_prev is not None and _within(pos, rot, cfg):
                dist = float(np.linalg.norm(q - q_prev))
                if near is None or dist < near[0]:
                    near = (dist, q)
            cost = pos + cfg.orientation_weight * rot
            if best is None or cost < best[0]:
                best = (cost, q)
        chosen = near if near is not None else best
        assert chosen is not None
        return chosen[1]

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
        q_prev = seeds(chain, 1)[0]  # 첫 목표 전 시작 자세 (관절 범위 가운데)
        solved: Vec | None = None  # 마지막으로 푼 해 (다시 시작할 때 첫 시작점)
        had_target = False
        for k in range(n):
            if not tg.present[k]:
                out.q[k], had_target = q_prev, False
                continue
            goal = Goal(tg.position[k], tg.approach[k], tg.closing[k])
            if had_target:
                q = _iterate(chain, q_prev, cfg.smoothness_weight, goal, problem)
            else:
                q = self._restart(chain, goal, problem, solved)
            if k > 0:
                q, out.velocity_limited[k] = limit_step(q, q_prev, vmax, problem.times_ms, k)
            pos, rot = frame_errors(chain, q, goal, problem)
            out.q[k], out.position_error_m[k], out.orientation_error_rad[k] = q, pos, rot
            out.reachable[k] = _within(pos, rot, cfg)
            q_prev, solved, had_target = q, q, True
        return out


def _within(pos: float, rot: float, cfg: IkConfig) -> bool:
    """도달 판정: 위치 오차 ≤ position_tolerance_m, 그리고 (none이 아니면) 회전 오차 ≤
    orientation_tolerance_rad."""
    return pos <= cfg.position_tolerance_m and (
        cfg.orientation_mode is OrientationMode.NONE or rot <= cfg.orientation_tolerance_rad
    )


def limit_step(q: Vec, q_prev: Vec, vmax: Vec, times_ms: Vec, k: int) -> tuple[Vec, bool]:
    """프레임 k의 관절 값을 직전 프레임 값에서 |Δq| ≤ vmax · Δt 안으로 자른다.

    Returns:
        (자른 관절 값, 잘렸는가). vmax가 inf인 관절(URDF에 속도 한계 없음)은 자르지 않는다.
        시각이 거꾸로 가면 Δt = 0으로 본다 (움직이지 못한다).
    """
    dt = max(float(times_ms[k] - times_ms[k - 1]) / 1000, 0.0)
    want = q - q_prev
    # inf x 0 = NaN을 피한다 (속도 한계 없는 관절은 Δt와 무관하게 자르지 않는다)
    bound = np.full(vmax.shape, np.inf)
    finite = np.isfinite(vmax)
    bound[finite] = vmax[finite] * dt
    step = np.clip(want, -bound, bound)
    return q_prev + step, bool(np.any(np.abs(step - want) > 1e-9))
