"""작업자 손 크기(손목 → 중지 MCP 길이) 추정 (ADR 0045·0052, `dlp calib palm-estimate`).

왜 필요한가
    3D 손 맞춤(`handfit.fit_hand`)은 MediaPipe 손 모양(world, m)의 크기를 손 크기 사전값과 메트릭
    깊이를 섞어 정한다. 사전값이 성인 평균(`reference_palm_m` 0.09 m)이면 손이 작거나 큰 작업자는
    그만큼(약 ±10%) 멀거나 가깝게 놓인다. 캘리브레이션된 내부 파라미터가 있는 세션에서 깊이만으로
    정한 배율(`HandFit.palm_from_depth`)은 사전값과 무관하게 손바닥 길이를 잰다. 화각 근사(초점 거리
    추정)로는 크기를 잴 수 없어 캘리브레이션 없는 세션은 쓰지 않는다.

어떻게
    품질이 좋은 시각(`palm_estimate` 정책: 재투영 RMS, 보이는 관절 수, 값 범위)만 모아 중앙값을
    그 작업자의 추정값으로 낸다. 손 3D 점(world)이 있는 키프레임 시각만 깊이를 추론한다 (사람이
    그리거나 고친 트랙은 world가 없어 맞춤을 할 수 없다, `usable_hand_times`).

한계
    - 깊이 모델의 배율 오차(장면마다 다를 수 있다)가 그대로 들어간다. 여러 세션·장면에 걸친
      중앙값이라 한 장면의 치우침은 줄지만, 깊이 모델 자체의 체계 오차는 남는다. 잰 값
      (`dlp calib palm`)이 있으면 그것을 쓰는 편이 낫다 (추정 등록은 잰 값을 덮지 않게 막는다,
      `--override-measured`).
    - MediaPipe 손 트랙은 착용자 손과 다른 사람(돌봄 대상자) 손을 구분하지 않는다. 대상자 손이
      섞이면 중앙값이 치우친다 (합성 실험: 손 크기 1.4배인 손이 30% 이상 섞이면 3~16%). 손 트랙
      검수가 끝난 뒤 `--reviewed-only`(승인·표본 검증한 손 트랙만)로 돌리는 것을 권한다 (F-PMS-27).

공개 이름: `usable_hand_times`, `palm_samples`, `PalmEstimate`, `estimate_palm`.
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


def usable_hand_times(tracks: list[LabelRecord]) -> set[int]:
    """손 맞춤을 할 수 있는 시각 (PTS ms): hand21 키프레임 중 손 3D 점(world)이 있는 것.

    비어 있으면 그 세션에서는 표본을 낼 수 없다. 호출자는 원본 영상을 받기 전에 이것으로 걸러
    불필요한 원본 접근을 피한다 (F-COE-70).
    """
    hands = hand_frames_by_time(tracks)
    return {t for t, frames in hands.items() if any(f.world is not None for _, f in frames)}


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
    # world 없는 시각은 맞춤이 None이라 깊이 추론만 낭비한다
    wanted = usable_hand_times(tracks)
    out: list[float] = []
    for t, dmap, intr in depth_frames(frames, wanted, depth, calib, policy.depth):
        for _, frame in hands[t]:
            fit = hand_fit_of(frame, dmap, intr, policy.depth)
            if (
                fit is None
                or fit.palm_from_depth is None
                or fit.rms_rel > est.max_reprojection_ratio
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
