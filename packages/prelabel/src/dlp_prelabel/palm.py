"""작업자 손 크기(손목 → 중지 MCP 길이) 추정 (ADR 0045, `dlp calib palm-estimate`).

3D 손 맞춤(`handfit.fit_hand`)의 크기는 손 크기 사전값과 메트릭 깊이를 섞어 정한다. 사전값이 성인
평균(`reference_palm_m` 0.09 m)이면 손이 작거나 큰 작업자는 그만큼(약 ±10%) 멀거나 가깝게 놓인다.
캘리브레이션된 내부 파라미터가 있는 세션에서 깊이만으로 정한 배율(`HandFit.palm_from_depth`)은
사전값과 무관하게 손바닥 길이를 잰다. 품질이 좋은 시각(`palm_estimate` 정책: 재투영 RMS, 보이는
관절 수, 값 범위)만 모아 중앙값을 그 작업자의 추정값으로 낸다.

한계: 깊이 모델의 배율 오차(장면마다 다를 수 있다)가 그대로 들어간다. 여러 세션·장면에 걸친
중앙값이라 한 장면의 치우침은 줄지만, 깊이 모델 자체의 체계 오차는 남는다. 잰 값
(`dlp calib palm`)이 있으면 그것을 쓰는 편이 낫다. 캘리브레이션이 없는 세션은 쓰지 않는다 (화각
근사로는 크기를 잴 수 없다).

공개 이름: `palm_samples`, `PalmEstimate`, `estimate_palm`.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable
from dataclasses import dataclass

from dlp_prelabel.common import Image
from dlp_prelabel.lift3d import DepthModel, depth_frames, hand_fit_of, hand_frames_by_time
from dlp_prelabel.policy import PrelabelPolicy
from dlp_schema.labels import LabelRecord
from dlp_schema.session import CameraIntrinsics


def palm_samples(
    frames: Iterable[tuple[int, Image]],
    depth: DepthModel,
    tracks: list[LabelRecord],
    calib: CameraIntrinsics,
    policy: PrelabelPolicy,
) -> list[float]:
    """한 세션의 손 맞춤에서 깊이만으로 잰 손바닥 길이 (m) 목록.

    Args:
        frames: 바디캠 (PTS ms, RGB 프레임) 순회.
        depth: 깊이 추론기.
        tracks: 3D 단계와 같은 입력 트랙 (`runner.pick_lift_tracks`). world가 있는 hand21 키프레임만
            쓴다 (박스는 무시).
        calib: 캘리브레이션된 내부 파라미터 (없는 세션은 호출하지 않는다).
        policy: 프리라벨 정책 (`depth` 절의 깊이 범위·간격·맞춤, `palm_estimate` 절의 품질 기준).

    Returns:
        품질 기준을 통과한 시각마다 손 하나의 값 (등장 순).
    """
    est = policy.palm_estimate
    lo, hi = est.palm_range_m
    hands = hand_frames_by_time(tracks)
    out: list[float] = []
    for t, dmap, intr in depth_frames(frames, hands.keys(), depth, calib, policy.depth):
        for _, frame in hands[t]:
            fit = hand_fit_of(frame, dmap, intr, policy.depth)
            if (
                fit is None
                or fit.palm_from_depth is None
                or fit.rms_px > est.max_reprojection_px
                or fit.n_visible < est.min_visible
            ):
                continue
            if lo <= fit.palm_from_depth <= hi:
                out.append(fit.palm_from_depth)
    return out


@dataclass(frozen=True)
class PalmEstimate:
    """작업자 손바닥 길이 추정 (중앙값, 0.1 mm 반올림)과 표본 수·세션 수."""

    palm_m: float
    n_samples: int
    n_sessions: int


def estimate_palm(
    per_session: Iterable[list[float]], policy: PrelabelPolicy
) -> PalmEstimate | None:
    """세션별 표본을 모아 중앙값을 낸다. 표본이 `palm_estimate.min_samples`보다 적으면 None."""
    sessions = [s for s in per_session if s]
    values = [v for s in sessions for v in s]
    if len(values) < policy.palm_estimate.min_samples:
        return None
    return PalmEstimate(round(statistics.median(values), 4), len(values), len(sessions))
