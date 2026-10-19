"""3D 표본 방식·신뢰도, 작업자 손 크기 사전값과 그 추정 단위 테스트 (ADR 0045).

정답은 `dlp_fixtures.hand3d`가 낸 카메라 좌표 21관절과 픽스처 인자(손 크기 배율 `hand_scale`,
초점 거리 `focal_px`)다. DB·서비스·모델 가중치 불필요.

신뢰도 정답 (저장소 `prelabel.yaml depth.sample_confidence`, `handfit.fit_confidence` 식):
- 흔들림 없음 + 캘리브레이션: 재투영 0·21관절·깊이와 사전값 배율 비 1 → `hand_fit` 그대로.
- 화각 근사(캘리브레이션 없음): 거기에 `hfov_approx`를 곱한다.
- world 없음(관절별 깊이): `point_depth` (화각 근사면 `hfov_approx`도 곱).
- 깊이 10% 편향: 배율 비 1.1 → 일치 항 1 - (0.1 / 0.5) x (1 - agreement_floor).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from dlp_fixtures.hand3d import Hand3DScenario, generate_hand3d_scenario
from dlp_prelabel.handfit import HandFit, fit_confidence
from dlp_prelabel.lift3d import lift_tracks, mean_confidence, with_palm
from dlp_prelabel.palm import estimate_palm, palm_samples
from dlp_prelabel.policy import HAND21_JOINTS, DepthPolicy, PrelabelPolicy, load_policy
from dlp_schema.labels import (
    KeypointTrackPayload,
    LabelRecord,
    Lift3DMethod,
    Trajectory3DSample,
    without_world,
)

ROOT = Path(__file__).resolve().parents[3]
JOINT_OF = {name: i for i, name in HAND21_JOINTS.items()}


def _prelabel() -> PrelabelPolicy:
    """저장소 정책 (모든 프레임에서 깊이)."""
    p = load_policy(ROOT)
    return p.model_copy(update={"depth": p.depth.model_copy(update={"frame_stride_ms": 0})})


def _lift(
    s: Hand3DScenario,
    policy: DepthPolicy,
    *,
    calibrated: bool = True,
    tracks: list[LabelRecord] | None = None,
) -> dict[tuple[str, str | None], list[Trajectory3DSample]]:
    calib = s.intrinsics if calibrated else None
    return lift_tracks(s.frames(), s.depth, tracks or s.labels, calib, policy)


def _all(out: dict[tuple[str, str | None], list[Trajectory3DSample]]) -> list[Trajectory3DSample]:
    return [x for v in out.values() for x in v]


def _max_error_mm(
    s: Hand3DScenario, out: dict[tuple[str, str | None], list[Trajectory3DSample]]
) -> float:
    errs: list[float] = []
    for (_, part), samples in out.items():
        assert part is not None
        for x in samples:
            truth: NDArray[np.float64] = s.joints_cam[x.t_ms][JOINT_OF[part]]
            diff = np.array([x.x, x.y, x.z], dtype=np.float64) - truth
            errs.append(1000 * float(np.sqrt(np.sum(diff**2))))
    return max(errs)


def test_samples_record_hand_fit_and_confidence_from_fit_quality() -> None:
    """흔들림 없는 손: 모든 표본이 hand_fit이고 신뢰도는 정책 최고값(캘리브레이션) 또는 그 x
    hfov_approx(화각 근사)다. world를 빼면 point_depth와 그 신뢰도다. 궤적 신뢰도는 표본 평균.
    """
    policy = _prelabel().depth
    c = policy.sample_confidence
    s = generate_hand3d_scenario(0, n_frames=10)
    calibrated = _all(_lift(s, policy))
    assert {x.method for x in calibrated} == {Lift3DMethod.HAND_FIT}
    assert [x.confidence for x in calibrated] == [pytest.approx(c.hand_fit, abs=2e-3)] * len(
        calibrated
    )
    approx = _all(_lift(s, policy, calibrated=False))
    expected = pytest.approx(c.hand_fit * c.hfov_approx, abs=2e-3)
    assert [x.confidence for x in approx] == [expected] * len(approx)
    [x] = s.labels
    assert isinstance(x.payload, KeypointTrackPayload)
    plain = [x.model_copy(update={"payload": without_world(x.payload)})]
    depth_only = _all(_lift(s, policy, calibrated=False, tracks=plain))
    assert {q.method for q in depth_only} == {Lift3DMethod.POINT_DEPTH}
    assert {q.confidence for q in depth_only} == {round(c.point_depth * c.hfov_approx, 3)}
    assert mean_confidence(depth_only) == round(c.point_depth * c.hfov_approx, 3)


def test_confidence_drops_with_noise_and_depth_disagreement() -> None:
    """2D 흔들림(재투영 오차↑)·깊이 10% 편향(배율 비 1.1)은 신뢰도를 낮춘다. 편향의 일치 항은
    모듈 설명의 식과 같다 (흔들림 없는 손이라 다른 항은 1).
    """
    policy = _prelabel().depth
    c, fit = policy.sample_confidence, policy.hand_fit
    clean = mean_confidence(_all(_lift(generate_hand3d_scenario(1), policy)))
    noisy = mean_confidence(_all(_lift(generate_hand3d_scenario(1, pixel_noise_px=2.0), policy)))
    biased = _all(_lift(generate_hand3d_scenario(1, depth_bias=1.1), policy))
    assert noisy < clean
    agreement = 1 - (0.1 / (fit.max_scale_ratio - 1)) * (1 - c.agreement_floor)
    expected = pytest.approx(c.hand_fit * agreement, abs=2e-3)
    assert [x.confidence for x in biased] == [expected] * len(biased)


def test_fit_confidence_ramps_between_policy_ends() -> None:
    """식의 끝값: RMS = max_reprojection_px, 관절 = min_points, 깊이 없음이면 각 floor·no_depth를
    곱한 값이다. 가장 좋은 맞춤은 정책 최고값 그대로다.
    """
    policy = _prelabel().depth
    c, fp = policy.sample_confidence, policy.hand_fit
    worst = HandFit(np.zeros((21, 3)), fp.max_reprojection_px, fp.min_points, None, 0.09, None)
    expected = c.hand_fit * c.reprojection_floor * c.points_floor * c.no_depth
    assert fit_confidence(worst, calibrated=True, policy=policy) == round(expected, 3)
    best = HandFit(np.zeros((21, 3)), 0.0, 21, 1.0, 0.09, 0.09)
    assert fit_confidence(best, calibrated=True, policy=policy) == c.hand_fit


def test_worker_palm_prior_fixes_a_small_hand() -> None:
    """정답 손이 20% 작으면(손바닥 0.072 m) 정책 기본값 0.09로는 크기가 틀려 40 mm 넘게 멀리 놓이고,
    작업자 값 0.072를 쓰면 1 mm 안이다 (깊이는 정답이라 배율 비 1.25 < 1.5로 섞인다).
    """
    policy = _prelabel().depth
    s = generate_hand3d_scenario(2, hand_scale=0.8)
    assert _max_error_mm(s, _lift(s, policy)) > 40.0
    assert _max_error_mm(s, _lift(s, with_palm(policy, 0.072))) < 1.0
    assert with_palm(policy, None) is policy


def test_palm_estimate_recovers_the_worker_hand_size() -> None:
    """캘리브레이션된 내부 파라미터와 정답 깊이로 깊이 배율만 써서 손바닥 길이를 잰다: 정답
    0.072 m와 0.5 mm 안. 표본이 min_samples보다 적으면 추정하지 않는다. 깊이 10% 편향은 그대로
    10% 들어간다 (모듈 설명의 한계).
    """
    policy = _prelabel()
    s = generate_hand3d_scenario(3, hand_scale=0.8, n_frames=40)
    samples = palm_samples(s.frames(), s.depth, s.labels, s.intrinsics, policy)
    assert len(samples) == 40
    est = estimate_palm([samples], policy)
    assert est is not None and est.n_samples == 40 and est.n_sessions == 1
    assert est.palm_m == pytest.approx(0.072, abs=5e-4)
    assert estimate_palm([samples[:5]], policy) is None
    b = generate_hand3d_scenario(3, hand_scale=0.8, n_frames=40, depth_bias=1.1)
    biased = estimate_palm(
        [palm_samples(b.frames(), b.depth, b.labels, b.intrinsics, policy)], policy
    )
    assert biased is not None and biased.palm_m == pytest.approx(0.072 * 1.1, abs=1e-3)


class _CountingDepth:
    """깊이 추론 횟수를 센다 (정답 깊이를 그대로 낸다)."""

    def __init__(self, s: Hand3DScenario) -> None:
        self.s, self.calls = s, 0

    def predict(self, image: NDArray[np.uint8]) -> NDArray[np.float32]:
        self.calls += 1
        return self.s.depth.predict(image)


def test_palm_samples_skip_depth_where_hand_has_no_world() -> None:
    """손 3D 점(world)이 없는 키프레임(사람이 그리거나 고친 트랙) 시각에는 깊이를 추론하지 않는다.

    world가 없으면 손 맞춤을 할 수 없어 표본도 없다 (F-COE-70). world가 있는 트랙은 표본을 낸다.
    """
    policy = _prelabel()
    s = generate_hand3d_scenario(3, hand_scale=0.8, n_frames=20)
    bare = [
        x.model_copy(update={"payload": without_world(x.payload)})
        if isinstance(x.payload, KeypointTrackPayload)
        else x
        for x in s.labels
    ]
    depth = _CountingDepth(s)
    assert palm_samples(s.frames(), depth, bare, s.intrinsics, policy) == []
    assert depth.calls == 0
    assert len(palm_samples(s.frames(), depth, s.labels, s.intrinsics, policy)) == 20
