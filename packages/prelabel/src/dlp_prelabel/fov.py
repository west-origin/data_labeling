"""손 크기로 세션 초점 거리(화각) 추정 (ADR 0056, `depth.focal_estimate`).

왜
    캘리브레이션이 없는 세션은 정책 기본 화각(`default_hfov_deg`)으로 내부 파라미터를 근사한다. 첫
    실영상(iPhone 13 0.5x 초광각, 짧은 변 화각 약 74도)에서 기본 90도를 가로 변에 걸어 초점 거리가
    실제와 크게 달랐고, 손 맞춤 거리가 팔 길이 밖(약 0.7~1.3 m)으로 밀렸다.
어떻게
    손 맞춤(`handfit.fit_hand`)은 메트릭 깊이로 정한 배율로 손바닥 길이(`HandFit.palm_from_depth`)를
    낸다. 픽셀 길이 l, 깊이 z, 초점 f에서 손바닥 길이는 l·z/f라, 근사 초점 f0로 잰 값 p0와 실제
    손바닥 길이 L(작업자 등록값, 없으면 성인 평균 사전값) 사이에는 f = f0·p0/L이 성립한다. 품질
    좋은 맞춤(`palm_estimate`의 재투영·보이는 관절 기준)에서 p0/L의 중앙값으로 f를 정한다.
한계 (정직한 출처 표시 이유)
    - 깊이 모델의 배율 오차와 손 크기 사전값 오차(성인 평균이면 약 ±10%)가 그대로 초점 거리 오차가
      된다. 캘리브레이션이 아니다: 출처는 `estimated_hand_size`, 신뢰도는 calibrated보다 낮다.
    - 렌즈 왜곡·주점은 추정하지 않는다 (주점은 화면 가운데, fx = fy).
    - 실제 캘리브레이션(세션 값·기기 등록값)이 있으면 쓰지 않는다.
    - 화각을 입력받지 않는 메트릭 깊이 모델은 학습 화각을 전제로 거리를 낸다.
      그래서 이 식은 카메라가 아니라 깊이 모델의 화각 사전값을 되돌려 준다
      (pilot-real2 초광각에서 초점 2.4배). 정책 기본은 끈다 (ADR 0056 개정, F-PAR-65).
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable
from dataclasses import dataclass

from dlp_models.depth import Intrinsics
from dlp_prelabel.common import Image
from dlp_prelabel.lift3d import DepthModel, depth_frames, hand_fit_of, hand_frames_by_time
from dlp_prelabel.palm import usable_hand_times
from dlp_prelabel.policy import PrelabelPolicy
from dlp_schema.labels import LabelRecord
from dlp_schema.session import CameraIntrinsics


@dataclass(frozen=True)
class FocalEstimate:
    """추정 초점 거리 (px)와 그 가로 화각, 표본 수, 상대 퍼짐(사분위 범위 / 중앙값)."""

    intrinsics: CameraIntrinsics
    hfov_deg: float
    n_samples: int
    spread: float


def focal_samples(
    frames: Iterable[tuple[int, Image]],
    depth: DepthModel,
    tracks: list[LabelRecord],
    policy: PrelabelPolicy,
) -> tuple[list[float], tuple[int, int] | None]:
    """화각 근사 내부 파라미터로 잰 손바닥 길이 p0 (m) 목록과 프레임 크기 (w, h).

    깊이는 `focal_estimate.frame_stride_ms` 간격으로만 추론한다 (3D 단계 앞의 추가 비용).
    """
    fe = policy.depth.focal_estimate
    est = policy.palm_estimate
    depth_policy = policy.depth.model_copy(update={"frame_stride_ms": fe.frame_stride_ms})
    hands = hand_frames_by_time(tracks)
    out: list[float] = []
    size: tuple[int, int] | None = None
    for t, dmap, intr in depth_frames(frames, usable_hand_times(tracks), depth, None, depth_policy):
        size = (int(dmap.shape[1]), int(dmap.shape[0]))
        for _, frame in hands[t]:
            fit = hand_fit_of(frame, dmap, intr, depth_policy)
            if (
                fit is not None
                and fit.palm_from_depth is not None
                and fit.rms_rel <= est.max_reprojection_ratio
                and fit.n_visible >= est.min_visible
            ):
                out.append(fit.palm_from_depth)
    return out, size


def estimate_focal(
    samples: list[float], prior_palm_m: float, size: tuple[int, int], policy: PrelabelPolicy
) -> FocalEstimate | None:
    """p0 표본 → 초점 거리. 표본이 모자라거나 퍼짐이 크거나 화각이 범위 밖이면 None."""
    fe = policy.depth.focal_estimate
    if len(samples) < fe.min_samples or prior_palm_m <= 0:
        return None
    ratios = sorted(s / prior_palm_m for s in samples)
    median = statistics.median(ratios)
    q = statistics.quantiles(ratios, n=4)
    spread = (q[2] - q[0]) / median if median > 0 else math.inf
    width, height = size
    f0 = Intrinsics.from_hfov(width, height, policy.depth.default_hfov_deg).fx
    fx = f0 * median
    hfov = math.degrees(2 * math.atan(width / (2 * fx)))
    lo, hi = fe.hfov_range_deg
    if spread > fe.max_spread or not lo <= hfov <= hi:
        return None
    intr = CameraIntrinsics(
        width=width, height=height, fx=round(fx, 2), fy=round(fx, 2), cx=width / 2, cy=height / 2
    )
    return FocalEstimate(intr, round(hfov, 2), len(samples), round(spread, 4))
