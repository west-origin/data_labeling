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

품질과 신뢰도 (ADR 0045): 맞춤은 재투영 RMS·보이는 관절 수·깊이와 손 크기 사전값의 배율 비를
`HandFit`에 함께 낸다. `fit_confidence`·`point_confidence`가 `depth.sample_confidence` 정책으로
표본마다 신뢰도를 정한다 (캘리브레이션이 없으면 낮춘다).

한계: 단안이라 절대 거리 오차가 남는다 (손 크기 개인차, 깊이 모델 오차). 렌즈 왜곡은 다루지 않는다.
깊이 맵은 손 표면 깊이라 관절 중심보다 약간 앞(수 mm~1 cm)이다. 좌표는 그 시각의 카메라 좌표다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

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
    palm = _palm_length(world)
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


@dataclass(frozen=True)
class HandFit:
    """한 시각 손 맞춤 결과와 품질 진단 (`fit_hand`, 표본 신뢰도 `fit_confidence`에 쓴다).

    points: 21관절 카메라 좌표 (21, 3) 미터.
    rms_px: 보이는 관절의 재투영 RMS (px).
    n_visible: 맞춤에 쓴 보이는 관절 수.
    scale_ratio: 깊이 배율과 손 크기 사전값 배율의 비 max(kd/ks, ks/kd) (깊이를 섞었을 때만,
        깊이가 없거나 너무 어긋나 버렸으면 None).
    palm_m: 맞춘 손의 손목 → 중지 MCP 길이 (m). 깊이 배율만으로 잰 값은 `palm_from_depth`.
    palm_from_depth: 깊이 배율만 썼을 때의 손바닥 길이 (m, 깊이가 없으면 None). 작업자 손 크기
        추정(`dlp_prelabel.palm.estimate_palm`)에 쓴다.
    """

    points: NDArray[np.float64]
    rms_px: float
    n_visible: int
    scale_ratio: float | None
    palm_m: float
    palm_from_depth: float | None


def fuse_scale(k_depth: float | None, k_size: float, policy: HandFitPolicy) -> float:
    """두 배율의 로그 가중 평균 (깊이가 없거나 너무 어긋나면 k_size)."""
    if k_depth is None or k_depth <= 0:
        return k_size
    if max(k_depth / k_size, k_size / k_depth) > policy.max_scale_ratio:
        return k_size
    w = policy.depth_weight
    return math.exp(w * math.log(k_depth) + (1 - w) * math.log(k_size))


def _palm_length(world: NDArray[np.float64]) -> float:
    """손목 → 중지 MCP 길이 (world와 같은 단위)."""
    return float(np.sqrt(np.sum(np.diff(world[[PALM_FROM, PALM_TO]], axis=0) ** 2)))


def fit_hand(
    world: NDArray[np.float64],
    pixels: NDArray[np.float64],
    visible: NDArray[np.bool_],
    depth: NDArray[np.float32],
    intr: Intrinsics,
    policy: DepthPolicy,
) -> HandFit | None:
    """손 21관절 카메라 좌표와 맞춤 품질. 맞춤이 안 되면 None (호출자는 깊이만 쓴다).

    Args:
        world: 손 중심 3D 점 (21, 3), 미터.
        pixels: 2D 픽셀 (21, 2).
        visible: 관절마다 맞춤에 쓸지 (visibility > 0).
        depth: 그 시각 메트릭 깊이 맵 (H, W).
        intr: 카메라 내부 파라미터.
        policy: `prelabel.yaml depth` (hand_fit 절과 깊이 범위·패치). 작업자 손 크기 사전값은
            호출자가 `hand_fit.reference_palm_m`을 바꾼 사본으로 넘긴다 (ADR 0045).
    """
    fit = policy.hand_fit
    n_visible = int(visible.sum())
    if n_visible < fit.min_points:
        return None
    t = solve_translation(world[visible], pixels[visible], intr)
    if t is None:
        return None
    cam = world + t
    if np.any(cam[:, 2] <= 0):
        return None
    rms = reprojection_rms(cam[visible], pixels[visible], intr)
    if rms > fit.max_reprojection_px:
        return None
    k_depth = depth_scale(cam, pixels, visible, depth, policy)
    k_size = size_scale(world, fit)
    k = fuse_scale(k_depth, k_size, fit)
    ratio: float | None = None
    if k_depth is not None and k_depth > 0:
        r = max(k_depth / k_size, k_size / k_depth)
        ratio = r if r <= fit.max_scale_ratio else None
    palm = _palm_length(world)
    return HandFit(
        points=cam * k,
        rms_px=rms,
        n_visible=n_visible,
        scale_ratio=ratio,
        palm_m=k * palm,
        palm_from_depth=k_depth * palm if k_depth is not None and k_depth > 0 else None,
    )


def _ramp(value: float, lo: float, hi: float, floor: float) -> float:
    """value가 lo면 1, hi면 floor, 그 사이는 선형, 범위 밖은 끝값 (lo < hi)."""
    if hi <= lo:
        return 1.0
    w = min(max((value - lo) / (hi - lo), 0.0), 1.0)
    return 1.0 - w * (1.0 - floor)


def fit_confidence(fit: HandFit, *, calibrated: bool, policy: DepthPolicy) -> float:
    """맞춤 표본 신뢰도 (0~1, 소수 셋째 자리, ADR 0045).

    `sample_confidence.hand_fit` x 재투영 항 x 관절 수 항 x 깊이·사전값 일치 항 x 내부 파라미터
    항. 재투영 항은 RMS 0 → 1, `hand_fit.max_reprojection_px` → `reprojection_floor`. 관절 수 항은
    21 → 1, `hand_fit.min_points` → `points_floor`. 일치 항은 배율 비 1 → 1,
    `hand_fit.max_scale_ratio` → `agreement_floor`이고 깊이를 못 쓰거나 버렸으면 `no_depth`.
    캘리브레이션이 없으면 `hfov_approx`를 곱한다.
    """
    c, fit_policy = policy.sample_confidence, policy.hand_fit
    value = c.hand_fit
    value *= _ramp(fit.rms_px, 0.0, fit_policy.max_reprojection_px, c.reprojection_floor)
    value *= _ramp(-float(fit.n_visible), -21.0, -float(fit_policy.min_points), c.points_floor)
    if fit.scale_ratio is None:
        value *= c.no_depth
    else:
        value *= _ramp(fit.scale_ratio, 1.0, fit_policy.max_scale_ratio, c.agreement_floor)
    if not calibrated:
        value *= c.hfov_approx
    return round(value, 3)


def point_confidence(*, calibrated: bool, policy: DepthPolicy) -> float:
    """관절별 깊이·박스 중심 표본 신뢰도 (`sample_confidence.point_depth`, 화각 근사면 곱)."""
    c = policy.sample_confidence
    return round(c.point_depth * (1.0 if calibrated else c.hfov_approx), 3)
