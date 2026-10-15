"""손가락 끝 3D 추정(`dlp_prelabel.handfit`, `lift3d.lift_tracks`의 손 맞춤) 단위 테스트 (ADR 0044).

정답은 `dlp_fixtures.hand3d`가 낸 카메라 좌표 21관절이다. 픽스처의 합성 깊이는 손바닥 관절 둘레만
손 깊이이고 나머지는 배경(1.2 m)이라, 관절별 깊이 방식은 얇은 손가락 끝에서 배경 깊이를 읽는다.
DB·서비스·모델 가중치 불필요.

허용 오차 (근거):
- 흔들림 없음: 손 모델 3D·2D·깊이가 모두 정답과 맞으므로 1 mm (반올림 0.1 mm).
- 2D 1 px + 손 중심 3D 3 mm 흔들림: 손목 0.4~0.55 m, fx 160이면 1 px ≈ 3 mm라 크기·위치 오차가
  1~2 cm 생긴다 → 평균 20 mm, 관절별 깊이의 손가락 끝(수백 mm)보다 열 배 이상 낫다.
- 깊이 10% 편향: 손 크기 사전값(정답과 같은 0.09 m)과 반반 섞어 크기 오차 약 5% → 0.5 m에서 약
  25 mm → 35 mm. 손 안 상대 위치(손목 기준 손가락 끝)는 5% x 0.17 m ≈ 8 mm → 10 mm.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from dlp_fixtures.hand3d import Hand3DScenario, generate_hand3d_scenario
from dlp_models.depth import Intrinsics
from dlp_prelabel.handfit import fuse_scale, solve_translation
from dlp_prelabel.lift3d import lift_tracks
from dlp_prelabel.policy import HAND21_JOINTS, DepthPolicy, load_policy
from dlp_schema.labels import (
    KeypointTrackPayload,
    LabelRecord,
    Trajectory3DSample,
    WorldPoint,
    without_world,
)

ROOT = Path(__file__).resolve().parents[3]
JOINT_OF = {name: i for i, name in HAND21_JOINTS.items()}
TIPS = ("thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip")


def _policy(**hand_fit: object) -> DepthPolicy:
    """저장소 정책의 depth 절 (모든 프레임에서 깊이, hand_fit 일부 값 바꾸기)."""
    depth = load_policy(ROOT).depth
    return depth.model_copy(
        update={"frame_stride_ms": 0, "hand_fit": depth.hand_fit.model_copy(update=hand_fit)}
    )


def _lift(
    s: Hand3DScenario, policy: DepthPolicy, tracks: list[LabelRecord] | None = None
) -> dict[tuple[str, str | None], list[Trajectory3DSample]]:
    """픽스처 프레임·합성 깊이·내부 파라미터로 `lift_tracks`를 돈다."""
    return lift_tracks(s.frames(), s.depth, tracks or s.labels, s.intrinsics, policy)


def _errors_mm(
    s: Hand3DScenario, out: dict[tuple[str, str | None], list[Trajectory3DSample]]
) -> dict[str, float]:
    """관절 이름 → 정답 카메라 좌표와의 평균 거리(mm)."""
    errs: dict[str, float] = {}
    for (_, part), samples in out.items():
        assert part is not None
        truth: NDArray[np.float64] = np.take(
            np.stack([s.joints_cam[x.t_ms] for x in samples]), JOINT_OF[part], axis=1
        )
        got = np.array([[x.x, x.y, x.z] for x in samples], dtype=np.float64)
        errs[part] = 1000 * float(np.mean(np.sqrt(np.sum((got - truth) ** 2, axis=1))))
    return errs


def _without_world(s: Hand3DScenario) -> list[LabelRecord]:
    """world를 뺀 같은 트랙 (사람이 고친 트랙·예전 버전처럼)."""
    [x] = s.labels
    assert isinstance(x.payload, KeypointTrackPayload)
    return [x.model_copy(update={"payload": without_world(x.payload)})]


def test_noise_free_fingertips_match_ground_truth() -> None:
    """흔들림이 없으면 손목·다섯 손가락 끝 모두 정답과 1 mm 안이고, 모든 프레임에 표본이 있다."""
    s = generate_hand3d_scenario(0)
    out = _lift(s, _policy())
    assert {p for _, p in out} == {"wrist", *TIPS}
    assert all(len(v) == len(s.frame_times) for v in out.values())
    assert max(_errors_mm(s, out).values()) < 1.0


def test_fit_beats_per_joint_depth_under_noise() -> None:
    """2D 1 px·손 중심 3D 3 mm 흔들림에서도 손가락 끝 평균 오차 20 mm 안이다. 같은 트랙에서
    world를 빼면(관절별 깊이) 손가락 끝이 배경 깊이를 읽어 300 mm 넘게 틀린다 (손목은 손바닥 원판
    위라 맞다).
    """
    s = generate_hand3d_scenario(1, pixel_noise_px=1.0, world_noise_m=0.003)
    fit = _errors_mm(s, _lift(s, _policy()))
    depth_only = _errors_mm(s, _lift(s, _policy(), _without_world(s)))
    assert max(fit.values()) < 20.0
    assert min(depth_only[t] for t in TIPS) > 300.0 and depth_only["wrist"] < 10.0


def test_hand_size_prior_corrects_a_wrong_hand_model_scale() -> None:
    """손 모델이 손을 10% 크게 내도(world_scale 1.1) 깊이와 손 크기 사전값(0.09 m, 정답과 같음)이
    크기를 바로잡아 1 mm 안이다. 사전값을 끄고(null) 깊이 가중치도 0이면 손 모델 크기를 믿어 10%
    멀리 놓인다 (0.4~0.55 m에서 40 mm 넘게).
    """
    s = generate_hand3d_scenario(2, world_scale=1.1)
    assert max(_errors_mm(s, _lift(s, _policy())).values()) < 1.0
    trusting = _errors_mm(s, _lift(s, _policy(reference_palm_m=None, depth_weight=0.0)))
    assert min(trusting.values()) > 40.0


def test_biased_depth_is_tempered_by_the_size_prior() -> None:
    """깊이가 10% 멀게 나와도 절대 위치 오차는 35 mm, 손목 기준 손가락 끝 상대 위치 오차는 10 mm
    안이다 (근거는 모듈 설명).
    """
    s = generate_hand3d_scenario(3, depth_bias=1.1)
    out = _lift(s, _policy())
    assert max(_errors_mm(s, out).values()) < 35.0
    wrist = {x.t_ms: np.array([x.x, x.y, x.z]) for x in out[("right_hand", "wrist")]}
    for tip in TIPS:
        for x in out[("right_hand", tip)]:
            truth = s.joints_cam[x.t_ms]
            rel = np.array([x.x, x.y, x.z]) - wrist[x.t_ms]
            assert np.linalg.norm(rel - (truth[JOINT_OF[tip]] - truth[0])) < 0.010


def test_inconsistent_world_points_fall_back_to_per_joint_depth() -> None:
    """손 중심 3D 점이 2D와 맞지 않으면(관절 순서가 뒤섞임) 재투영 오차가 상한을 넘어 맞춤을 버리고,
    world가 없는 트랙과 같은 결과(관절별 깊이)를 낸다.
    """
    s = generate_hand3d_scenario(4)
    [x] = s.labels
    assert isinstance(x.payload, KeypointTrackPayload)
    frames = tuple(
        f.model_copy(update={"world": tuple(reversed(f.world or ()))}) for f in x.payload.keyframes
    )
    scrambled = x.model_copy(update={"payload": x.payload.model_copy(update={"keyframes": frames})})
    assert _lift(s, _policy(), [scrambled]) == _lift(s, _policy(), _without_world(s))


def test_too_few_visible_joints_fall_back() -> None:
    """보이는 관절이 min_points보다 적으면 맞추지 않는다 (min_points 22는 정책이 막으므로 21에서
    한 관절을 가린다: 보이는 20개 < 21).
    """
    s = generate_hand3d_scenario(5)
    [x] = s.labels
    assert isinstance(x.payload, KeypointTrackPayload)
    frames = tuple(
        f.model_copy(
            update={"points": (f.points[0].model_copy(update={"visibility": 0}), *f.points[1:])}
        )
        for f in x.payload.keyframes
    )
    hidden = x.model_copy(update={"payload": x.payload.model_copy(update={"keyframes": frames})})
    out = _lift(s, _policy(min_points=21), [hidden])
    # 손목(가림)은 빠지고, 손가락 끝은 관절별 깊이(배경)로 올라 크게 틀린다
    assert ("right_hand", "wrist") not in out
    assert min(_errors_mm(s, out).values()) > 300.0


def test_solve_translation_recovers_a_known_shift_and_rejects_points_behind() -> None:
    """한 직선 위가 아닌 점 넷을 (0.1, -0.05, 0.6)만큼 옮겨 투영하면 그 이동량을 되찾는다. 이동량이
    점을 카메라 뒤로 보내야 맞는 2D(모두 같은 점)면 풀 수 없어 None.
    """
    intr = Intrinsics(200.0, 200.0, 160.0, 120.0)
    world = np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.01], [0.0, 0.06, -0.01], [0.03, 0.03, 0.02]])
    shift = np.array([0.1, -0.05, 0.6])
    cam = world + shift
    pixels = np.stack(
        [intr.fx * cam[:, 0] / cam[:, 2] + intr.cx, intr.fy * cam[:, 1] / cam[:, 2] + intr.cy],
        axis=1,
    )
    t = solve_translation(world, pixels, intr)
    assert t is not None and t == pytest.approx(shift, abs=1e-9)
    same = np.repeat(world[:1], 4, axis=0)
    assert solve_translation(same, pixels, intr) is None


def test_fuse_scale_uses_log_mean_and_drops_disagreeing_depth() -> None:
    """깊이 배율 1.21·크기 배율 1.0, 가중치 0.5 → 1.1 (로그 평균). 비가 1.5를 넘으면(2.0) 깊이를
    버리고 크기 배율, 깊이가 없으면 크기 배율.
    """
    fit = load_policy(ROOT).depth.hand_fit
    assert fuse_scale(1.21, 1.0, fit) == pytest.approx(1.1)
    assert fuse_scale(2.0, 1.0, fit) == 1.0
    assert fuse_scale(None, 0.9, fit) == 0.9


def test_world_points_are_hand_centred_in_the_fixture() -> None:
    """픽스처의 손 중심 3D 점은 정답 관절에서 관절 평균을 뺀 것이다 (MediaPipe 원점 흉내)."""
    s = generate_hand3d_scenario(6)
    [x] = s.labels
    assert isinstance(x.payload, KeypointTrackPayload)
    f = x.payload.keyframes[0]
    truth = s.joints_cam[f.t_ms]
    assert f.world is not None
    got = np.array([[w.x, w.y, w.z] for w in f.world])
    assert got == pytest.approx(truth - truth.mean(axis=0))
    assert isinstance(f.world[0], WorldPoint)
