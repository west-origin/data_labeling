"""손 중심 3D 점·2D 픽셀·메트릭 깊이로 손 21관절의 카메라 좌표 3D를 정한다 (ADR 0044).

입력 (한 시각, 손 하나):
- world: 손 모델(MediaPipe hand_world_landmarks)의 손 중심 3D 21점 W_i (미터, 카메라와 같은 축,
  원점은 손의 대략적인 기하 중심). 손 모양과 대략적인 크기를 준다.
- pixels: 같은 관절의 2D 픽셀 u_i (스트림 영상 픽셀).
- 깊이 맵: 메트릭 깊이 모델 출력 (미터). 손바닥 관절(`hand_fit.anchor_points`)에서만 읽는다.
- 내부 파라미터: 세션 캘리브레이션 또는 기본 화각 근사 (`lift3d.intrinsics_for`).

추정 (세 단계):
1. 이동량 맞춤: 손 모양을 그대로 두고 카메라 좌표 Q_i = W_i + T가 2D 픽셀로 투영되도록 T를
   최소제곱으로 푼다. 핀홀 투영 a_i = (u_i - cx) / fx, b_i = (v_i - cy) / fy에서
   (W_ix + Tx) - a_i (W_iz + Tz) = 0, (W_iy + Ty) - b_i (W_iz + Tz) = 0 은 T에 선형이다. 한 번 푼 뒤
   관절 깊이 1/(W_iz + Tz)로 가중해 다시 풀면 잔차가 정규화 재투영 오차와 같아진다.
2. 크기(거리) 정하기: 단안 투영은 Q 전체를 카메라 원점 기준으로 k배 해도 같은 2D를 낸다. k를
   두 근거로 정한다.
   - 깊이: 손바닥 관절 j마다 깊이 맵 값 / Q_jz의 중앙값 k_depth (얇은 손가락 끝은 배경 깊이가 섞여
     넣지 않는다).
   - 손 크기 사전값: k_size = reference_palm_m / |W_0 - W_9| (손목 → 중지 MCP). 사전값이 없으면 1
     (손 모델 크기를 믿는다).
   k = exp(w log k_depth + (1 - w) log k_size) (w = depth_weight). 깊이가 없거나 두 배율 비가
   max_scale_ratio를 넘으면(배경·가림 깊이) k = k_size.
3. 결과: P_i = k Q_i. 2D 투영은 맞춘 그대로이고, 손 안의 상대 위치는 손 모델 모양을 k배 한 것이다.

실패(보이는 관절 부족, 풀 수 없음, 관절이 카메라 뒤, 재투영 RMS 초과)면 None을 돌려주고, 호출자
(`lift3d.lift_tracks`)가 관절마다 깊이만 쓰는 예전 방식으로 올린다.

한계: 단안이라 절대 거리 오차가 남는다 (손 크기 개인차, 깊이 모델 오차). 렌즈 왜곡은 다루지 않는다.
깊이 맵은 손 표면 깊이라 관절 중심보다 약간 앞(수 mm~1 cm)이다. 좌표는 그 시각의 카메라 좌표다.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from dlp_models.depth import Intrinsics, sample_depth
from dlp_prelabel.policy import DepthPolicy, HandFitPolicy

# 손 크기 사전값의 두 끝 관절 (손목, 중지 MCP)
PALM_FROM, PALM_TO = 0, 9
# 이동량 정규방정식의 조건수 상한 (넘으면 풀 수 없다고 본다: 점이 한 줄에 몰림 등)
MAX_CONDITION = 1e8


def solve_translation(
    world: NDArray[np.float64], pixels: NDArray[np.float64], intr: Intrinsics
) -> NDArray[np.float64] | None:
    """손 모양 world(n, 3)를 옮겨 pixels(n, 2)로 투영되게 하는 이동량 T(3,). 풀 수 없으면 None.

    선형 최소제곱 뒤 관절 깊이로 한 번 다시 가중한다 (모듈 설명 1단계). 관절 하나라도 카메라 뒤
    (깊이 ≤ 0)면 None.
    """
    a = (pixels[:, 0] - intr.cx) / intr.fx
    b = (pixels[:, 1] - intr.cy) / intr.fy
    n = len(world)
    rows = np.zeros((2 * n, 3))
    rows[0::2, 0], rows[0::2, 2] = 1.0, -a
    rows[1::2, 1], rows[1::2, 2] = 1.0, -b
    rhs = np.empty(2 * n)
    rhs[0::2] = a * world[:, 2] - world[:, 0]
    rhs[1::2] = b * world[:, 2] - world[:, 1]
    weights = np.ones(2 * n)
    t = np.zeros(3)
    for _ in range(2):
        aw = rows * weights[:, None]
        if np.linalg.cond(aw) > MAX_CONDITION:
            return None
        t = np.linalg.lstsq(aw, rhs * weights, rcond=None)[0]
        z = world[:, 2] + t[2]
        if np.any(z <= 0):
            return None
        # 잔차를 관절 깊이로 나누면 정규화 재투영 오차가 된다
        weights = np.repeat(1.0 / z, 2)
    return t


def reprojection_rms(
    cam: NDArray[np.float64], pixels: NDArray[np.float64], intr: Intrinsics
) -> float:
    """카메라 좌표 점 cam(n, 3)을 투영한 픽셀과 pixels(n, 2)의 RMS 거리(px)."""
    u = intr.fx * cam[:, 0] / cam[:, 2] + intr.cx
    v = intr.fy * cam[:, 1] / cam[:, 2] + intr.cy
    return float(np.sqrt(np.mean((u - pixels[:, 0]) ** 2 + (v - pixels[:, 1]) ** 2)))


def size_scale(world: NDArray[np.float64], policy: HandFitPolicy) -> float:
    """손 크기 사전값 배율 k_size (사전값이 없거나 손바닥 길이가 0이면 1)."""
    palm = float(np.sqrt(np.sum(np.diff(world[[PALM_FROM, PALM_TO]], axis=0) ** 2)))
    if policy.reference_palm_m is None or palm <= 0:
        return 1.0
    return policy.reference_palm_m / palm


def depth_scale(
    cam: NDArray[np.float64],
    pixels: NDArray[np.float64],
    visible: NDArray[np.bool_],
    depth: NDArray[np.float32],
    policy: DepthPolicy,
) -> float | None:
    """손바닥 관절 깊이 / 맞춘 관절 깊이의 중앙값 k_depth. 읽을 깊이가 없으면 None.

    깊이 맵 값이 nan(화면 밖)이거나 [min_depth_m, max_depth_m] 밖인 관절, 보이지 않는 관절은 뺀다.
    """
    ratios: list[float] = []
    for j in policy.hand_fit.anchor_points:
        if not visible[j]:
            continue
        z = sample_depth(depth, float(pixels[j, 0]), float(pixels[j, 1]), policy.patch_radius_px)
        if math.isnan(z) or not policy.min_depth_m <= z <= policy.max_depth_m:
            continue
        ratios.append(z / float(cam[j, 2]))
    return float(np.median(np.asarray(ratios, dtype=np.float64))) if ratios else None


def fuse_scale(k_depth: float | None, k_size: float, policy: HandFitPolicy) -> float:
    """두 배율의 로그 가중 평균 (깊이가 없거나 너무 어긋나면 k_size)."""
    if k_depth is None or k_depth <= 0:
        return k_size
    if max(k_depth / k_size, k_size / k_depth) > policy.max_scale_ratio:
        return k_size
    w = policy.depth_weight
    return math.exp(w * math.log(k_depth) + (1 - w) * math.log(k_size))


def fit_hand(
    world: NDArray[np.float64],
    pixels: NDArray[np.float64],
    visible: NDArray[np.bool_],
    depth: NDArray[np.float32],
    intr: Intrinsics,
    policy: DepthPolicy,
) -> NDArray[np.float64] | None:
    """손 21관절 카메라 좌표 (21, 3) 미터. 맞춤이 안 되면 None (호출자는 깊이만 쓴다).

    Args:
        world: 손 중심 3D 점 (21, 3), 미터.
        pixels: 2D 픽셀 (21, 2).
        visible: 관절마다 맞춤에 쓸지 (visibility > 0).
        depth: 그 시각 메트릭 깊이 맵 (H, W).
        intr: 카메라 내부 파라미터.
        policy: `prelabel.yaml depth` (hand_fit 절과 깊이 범위·패치).
    """
    fit = policy.hand_fit
    if int(visible.sum()) < fit.min_points:
        return None
    t = solve_translation(world[visible], pixels[visible], intr)
    if t is None:
        return None
    cam = world + t
    if (
        np.any(cam[:, 2] <= 0)
        or reprojection_rms(cam[visible], pixels[visible], intr) > fit.max_reprojection_px
    ):
        return None
    k = fuse_scale(depth_scale(cam, pixels, visible, depth, policy), size_scale(world, fit), fit)
    return cam * k
