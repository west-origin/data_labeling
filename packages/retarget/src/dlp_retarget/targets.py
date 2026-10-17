"""사람 손 점 → 로봇 끝단 목표와 그리퍼 열림 (ADR 0047).

손 점 3개(세션 월드 좌표 m, ADR 0046): 손목 W, 엄지 끝 T, 검지 끝 I.
- 잡기 점 g = (T + I) / 2 → 끝단(도구 중심점) 목표 위치.
- 접근 축 a = (g - W) 정규화 → 끝 링크의 approach_axis를 맞출 방향.
- 닫힘 축 c = (I - T)에서 a 성분을 뺀 뒤 정규화 → closing_axis (orientation_mode=full에서만 쓴다).
- 열림 거리 d = |I - T| → 평행 그리퍼 관절 값 (선형 대응, `GripperConfig`).

배치 (`Placement`, 월드 → 로봇 기준 좌표, 요 회전 + 이동 + 위치 배율):
- world_fixed: p_b = s · R_bw (g - base_xyz), R_bw = Rz(base_yaw)ᵀ.
- episode_centroid: p_b = workspace_center + s · R_bw (g - ḡ), ḡ = 그 에피소드 잡기 점 평균
  (목표가 있는 프레임만), R_bw = I (월드 X = 로봇 X, 월드 Z = 로봇 Z).
방향 축은 R_bw만 곱한다.

손 점이 하나라도 없는 프레임은 목표가 없다 (지어내지 않는다).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from dlp_retarget.robot import GripperConfig, Placement, PlacementMode


@dataclass(frozen=True)
class Targets:
    """프레임별 끝단 목표 (n = 프레임 수). present가 거짓인 프레임의 값은 의미가 없다 (NaN)."""

    present: NDArray[np.bool_]  # (n,)
    position: NDArray[np.float64]  # (n, 3) 로봇 기준 좌표 m
    approach: NDArray[np.float64]  # (n, 3) 단위
    closing: NDArray[np.float64]  # (n, 3) 단위
    opening_m: NDArray[np.float64]  # (n,) 엄지-검지 끝 거리


def _rz(yaw: float) -> NDArray[np.float64]:
    """z축 회전."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _unit(v: NDArray[np.float64]) -> NDArray[np.float64]:
    """행마다 단위 벡터 (길이 0이면 NaN)."""
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 1e-9, v / n, np.nan)


def hand_targets(
    wrist: NDArray[np.float64],
    thumb: NDArray[np.float64],
    index: NDArray[np.float64],
    placement: Placement,
) -> Targets:
    """손 점 (n, 3) 세 개(없는 프레임은 NaN) → 로봇 기준 좌표 목표.

    Raises:
        ValueError: 배열 모양이 다를 때.
    """
    if not wrist.shape == thumb.shape == index.shape or wrist.ndim != 2 or wrist.shape[1] != 3:
        raise ValueError("손 점 배열은 모두 (n, 3)이어야 합니다")
    present = ~(
        np.isnan(wrist).any(axis=1) | np.isnan(thumb).any(axis=1) | np.isnan(index).any(axis=1)
    )
    grasp = (thumb + index) / 2
    approach = _unit(grasp - wrist)
    span = index - thumb
    opening = np.linalg.norm(span, axis=1)
    closing = _unit(span - np.sum(span * approach, axis=1, keepdims=True) * approach)
    present &= ~np.isnan(approach).any(axis=1) & ~np.isnan(closing).any(axis=1)
    s = placement.position_scale
    if placement.mode is PlacementMode.WORLD_FIXED:
        r_bw = _rz(placement.base_yaw_rad).T
        position = s * (grasp - np.array(placement.base_xyz)) @ r_bw.T
    else:
        r_bw = np.eye(3)
        center = grasp[present].mean(axis=0) if present.any() else np.zeros(3)
        position = np.array(placement.workspace_center) + s * (grasp - center) @ r_bw.T
    nan = np.full(3, np.nan)
    return Targets(
        present=present,
        position=np.where(present[:, None], position, nan),
        approach=np.where(present[:, None], approach @ r_bw.T, nan),
        closing=np.where(present[:, None], closing @ r_bw.T, nan),
        opening_m=np.where(present, opening, np.nan),
    )


def gripper_values(opening_m: NDArray[np.float64], gripper: GripperConfig) -> NDArray[np.float64]:
    """열림 거리 → 그리퍼 관절 값 (거리 범위 밖은 끝값, NaN은 그대로)."""
    span = gripper.open_distance_m - gripper.closed_distance_m
    frac = np.clip((opening_m - gripper.closed_distance_m) / span, 0.0, 1.0)
    return gripper.closed + frac * (gripper.open - gripper.closed)
