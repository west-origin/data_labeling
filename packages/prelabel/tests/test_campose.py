"""바디캠 자세 추정과 월드 좌표 변환 (ADR 0046, F-PAR-26). 서비스 없이 돈다.

정답은 `dlp_fixtures.camera_motion`(정답 자세·IMU·렌더 프레임·해석적 깊이·월드 좌표 손)이다. 수치
상한은 ADR 0046 "정확도" 표와 같다. 실제 영상의 정확도는 측정하지 않았다 (F-PAR-29).
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from dlp_fixtures.camera_motion import (
    HAND_PARTS,
    CameraMotionScenario,
    MotionSpec,
    generate_camera_motion,
)
from dlp_prelabel.campose import (
    ImuSeries,
    ImuVisualOdometry,
    OraclePoseEstimator,
    PoseSample,
    PoseTrack,
    TranslationFit,
    imu_orientation,
    is_pose_label,
    pose_label,
    pose_samples,
    solve_translation,
)
from dlp_prelabel.policy import PrelabelPolicy, load_policy
from dlp_prelabel.posemath import (
    ate_rmse,
    exp_so3,
    log_so3,
    matrix_from_quat,
    quat_from_matrix,
    rotation_angle,
    rpe,
    slerp,
)
from dlp_prelabel.worldframe import pose_at, to_world_samples, world_label
from dlp_schema.labels import CoordinateFrame, IntrinsicsSource, Lift3DMethod, Trajectory3DPayload
from dlp_schema.ontology import load_ontology
from dlp_schema.session import CameraIntrinsics
from dlp_schema.testing import FIXED_TIME
from dlp_schema.validation import check_label

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def policy() -> PrelabelPolicy:
    """저장소 prelabel.yaml."""
    return load_policy(ROOT)


def _imu(sc: CameraMotionScenario, *, apply_offset: bool = True) -> ImuSeries:
    """픽스처 IMU를 바디캠(마스터) 시각으로 바꾼 것 (`Stream.to_master_ms`와 같은 식).

    apply_offset=False는 동기화 오프셋을 잊은 경우를 흉내 낸다.
    """
    shift = sc.imu_offset_ms if apply_offset else 0.0
    return ImuSeries(sc.imu_t_ms + shift, sc.imu_acc, sc.imu_gyro)


def _r_ci(sc: CameraMotionScenario) -> NDArray[np.float64]:
    """IMU → 카메라 회전 (픽스처 4x4의 회전 부분)."""
    return np.array(sc.imu_to_camera, dtype=np.float64)[:3, :3]


def _odometry(policy: PrelabelPolicy, hfov_deg: float | None = None) -> ImuVisualOdometry:
    """저장소 정책의 내장 오도메트리 (화각·화면비 허용 오차는 depth 절)."""
    return ImuVisualOdometry(
        policy.camera_pose,
        hfov_deg=policy.depth.default_hfov_deg if hfov_deg is None else hfov_deg,
        max_aspect_mismatch=policy.depth.max_intrinsics_aspect_mismatch,
    )


def _run(sc: CameraMotionScenario, policy: PrelabelPolicy, **kw: bool) -> PoseTrack:
    """내장 오도메트리로 자세 궤적을 낸다 (정답 깊이)."""
    est = _odometry(policy)
    return est.estimate(
        sc.frames(),
        imu=_imu(sc, **kw),
        imu_to_camera=_r_ci(sc),
        intrinsics=sc.intrinsics,
        depth=sc,
    )


def _errors(sc: CameraMotionScenario, track: PoseTrack) -> tuple[float, float, float, float]:
    """(ATE m, RPE 이동 m, RPE 회전 rad, 최대 회전 오차 rad)."""
    est = [(s.rotation, s.position) for s in track.samples]
    truth = [sc.pose(s.t_ms) for s in track.samples]
    ate = ate_rmse([p for _, p in est], [p for _, p in truth])
    rpe_t, rpe_r = rpe(est, truth)
    worst = max(rotation_angle(r.T @ t) for (r, _), (t, _) in zip(est, truth, strict=True))
    return ate, rpe_t, rpe_r, worst


def _path_length(points: list[NDArray[np.float64]]) -> float:
    """연속 위치 사이 거리 합 (m)."""
    return float(np.sum(np.linalg.norm(np.diff(np.array(points), axis=0), axis=1)))


# ---------------------------------------------------------------- 수학 도우미


def test_rotation_helpers_round_trip() -> None:
    """회전 벡터·쿼터니언 왕복과 구면 보간 (작은 각·180도 근처 포함)."""
    rng = np.random.default_rng(0)
    for w in [np.array([1e-10, 0, 0]), np.array([0, 0, np.pi - 1e-7]), *rng.normal(0, 1, (20, 3))]:
        r = exp_so3(w)
        assert np.allclose(r @ r.T, np.eye(3), atol=1e-9)
        assert np.allclose(exp_so3(log_so3(r)), r, atol=1e-6)
        q = quat_from_matrix(r)
        assert q[0] >= 0 and np.allclose(matrix_from_quat(*q), r, atol=1e-9)
    a, b = exp_so3(np.array([0.0, 0.0, 0.2])), exp_so3(np.array([0.0, 0.0, 0.6]))
    assert np.allclose(slerp(a, b, 0.5), exp_so3(np.array([0.0, 0.0, 0.4])), atol=1e-12)


def test_translation_solver_recovers_center_with_outliers(policy: PrelabelPolicy) -> None:
    """회전을 알 때 두 점 RANSAC 이동량 풀이가 30% 이상치에도 카메라 중심을 1 mm 안으로 찾는다."""
    sc = generate_camera_motion(1, duration_s=1.0)
    k = sc.intrinsics
    rng = np.random.default_rng(1)
    r_wc = exp_so3(np.array([0.3, -0.2, 0.1]))
    center = np.array([0.4, -0.3, 0.2])
    cam = np.column_stack(
        [rng.uniform(-1, 1, 120), rng.uniform(-0.7, 0.7, 120), rng.uniform(1, 4, 120)]
    )
    world = cam @ r_wc.T + center
    uv = np.column_stack([k.fx * cam[:, 0] / cam[:, 2] + k.cx, k.fy * cam[:, 1] / cam[:, 2] + k.cy])
    uv[:40] += rng.uniform(-40, 40, (40, 2))  # 이상치 (특징점 추적 실패 흉내)
    fit = solve_translation(world, uv, r_wc, k, policy.camera_pose)
    assert isinstance(fit, TranslationFit)
    assert np.linalg.norm(fit.position - center) < 1e-3
    assert 80 <= fit.inliers <= 85 and fit.rms_px < 0.5
    # 인라이어가 모자라면 자세를 내지 않는다 (지어내지 않는다)
    assert solve_translation(world[:10], uv[:10], r_wc, k, policy.camera_pose) is None


def test_imu_orientation_is_gravity_aligned_and_uses_extrinsics(policy: PrelabelPolicy) -> None:
    """IMU 회전 적분이 정답 카메라 회전과 맞고(최대 0.2도), 외부 파라미터를 무시하면 크게 틀린다."""
    sc = generate_camera_motion(2)
    imu = _imu(sc)
    orient = imu_orientation(imu, _r_ci(sc), 0.0, policy.camera_pose)
    assert orient.static_init
    errs: list[float] = []
    wrong: list[float] = []
    for t in sc.frame_times:
        r_wi = orient.at(t)
        assert r_wi is not None
        errs.append(rotation_angle((r_wi @ _r_ci(sc).T).T @ sc.pose(t)[0]))
        wrong.append(rotation_angle(r_wi.T @ sc.pose(t)[0]))  # 외부 파라미터를 단위 행렬로 본 것
    assert np.degrees(max(errs)) < 0.2
    assert np.degrees(min(wrong)) > 30
    # 월드 Z는 중력 반대: 첫 자세에서 카메라 아래(Y_c)는 월드 -Z 쪽으로 기운다
    first = orient.at(0)
    assert first is not None
    down_w = first @ _r_ci(sc).T @ np.array([0.0, 1.0, 0.0])
    assert down_w[2] < -0.8


# ---------------------------------------------------------------- 내장 오도메트리 정확도


def test_imu_visual_odometry_tracks_known_trajectory(policy: PrelabelPolicy) -> None:
    """잡음 없는 합성: 1.5 m 경로에서 ATE 1 cm 미만, RPE(100 ms) 이동 2 mm·회전 0.05도 미만,
    최대 회전 오차 0.2도 미만. 모든 처리 프레임에 자세가 있고 신뢰도는 정책 범위다."""
    sc = generate_camera_motion(0)
    track = _run(sc, policy)
    assert track.lost_at_ms is None and track.stats["failed"] == 0
    assert len(track.samples) == track.stats["frames"] == 40
    ate, rpe_t, rpe_r, worst = _errors(sc, track)
    truth_path = _path_length([sc.pose(s.t_ms)[1] for s in track.samples])
    assert truth_path > 1.0
    assert ate < 0.01 and rpe_t < 0.002 and np.degrees(rpe_r) < 0.05
    assert np.degrees(worst) < 0.2
    assert np.allclose(track.samples[0].position, 0) and track.stats["static_init"] == 1
    assert all(0 < s.confidence <= policy.camera_pose.confidence.base for s in track.samples)


def test_odometry_with_sensor_noise_and_gyro_bias(policy: PrelabelPolicy) -> None:
    """자이로 편향 0.01 rad/s(정지 구간에서 뺌)·흰잡음, 가속도 잡음: ATE 1.5 cm, 회전 0.5도 미만."""
    sc = generate_camera_motion(3, gyro_bias=(0.01, -0.01, 0.005), gyro_noise=0.01, acc_noise=0.1)
    ate, _, _, worst = _errors(sc, _run(sc, policy))
    assert ate < 0.015 and np.degrees(worst) < 0.5


def test_metric_scale_follows_depth_bias(policy: PrelabelPolicy) -> None:
    """이동 배율은 깊이 배율을 그대로 따른다 (깊이 10% 멀게 → 경로 약 10% 길게). 단안 깊이의
    한계."""
    sc = generate_camera_motion(0, depth_bias=1.1)
    track = _run(sc, policy)
    est = _path_length([s.position for s in track.samples])
    truth = _path_length([sc.pose(s.t_ms)[1] for s in track.samples])
    assert 1.05 < est / truth < 1.15


def test_imu_sync_offset_matters(policy: PrelabelPolicy) -> None:
    """IMU 시계가 바디캠보다 120 ms 늦게 시작: 오프셋을 적용하면 정확도가 같고, 잊으면 회전이
    틀린다."""
    sc = generate_camera_motion(4, imu_offset_ms=120.0, motion=MotionSpec(yaw_rad=0.5))
    good = _errors(sc, _run(sc, policy))
    bad = _errors(sc, _run(sc, policy, apply_offset=False))
    assert np.degrees(good[3]) < 0.3
    assert np.degrees(bad[3]) > 3 * np.degrees(good[3]) and np.degrees(bad[3]) > 1.0


def test_moving_start_lowers_confidence(policy: PrelabelPolicy) -> None:
    """처음 정지 구간이 없으면 자이로 편향을 빼지 못한 것으로 보고 신뢰도에 no_static_init을
    곱한다."""
    sc = generate_camera_motion(5, static_s=0.0)
    track = _run(sc, policy)
    c = policy.camera_pose.confidence
    assert track.stats["static_init"] == 0
    assert max(s.confidence for s in track.samples) <= round(c.base * c.no_static_init, 3)


def test_tracking_loss_stops_instead_of_inventing_poses(policy: PrelabelPolicy) -> None:
    """가려진(무늬 없는) 프레임이 max_lost_ms보다 길게 이어지면 그 뒤 자세를 내지 않는다."""
    sc = generate_camera_motion(6)
    blind_from = 1500

    def frames() -> Iterator[tuple[int, NDArray[np.uint8]]]:
        for t, img in sc.frames():
            yield t, (np.full_like(img, 90) if t >= blind_from else img)

    est = _odometry(policy, hfov_deg=90)
    track = est.estimate(
        frames(), imu=_imu(sc), imu_to_camera=_r_ci(sc), intrinsics=sc.intrinsics, depth=sc
    )
    assert track.lost_at_ms is not None
    assert track.lost_at_ms - blind_from >= policy.camera_pose.max_lost_ms
    assert all(s.t_ms < blind_from for s in track.samples)


def _imu_without(sc: CameraMotionScenario, lo_ms: float, hi_ms: float) -> ImuSeries:
    """마스터 시각 [lo_ms, hi_ms) IMU 표본을 뺀 IMU (늦은 시작·기록 끊김 흉내)."""
    imu = _imu(sc)
    keep = (imu.t_ms < lo_ms) | (imu.t_ms >= hi_ms)
    return ImuSeries(imu.t_ms[keep], imu.acc[keep], imu.gyro[keep])


def test_constant_rotation_start_is_not_mistaken_for_static(policy: PrelabelPolicy) -> None:
    """0.2 rad/s로 일정하게 돌며 녹화를 시작하면(F-PAR-46 재현) 자이로 크기는 거의 상수지만 평균이
    커서 정지가 아니다: 편향을 빼지 않아 회전이 맞고(2도 안), 신뢰도에 no_static_init을 곱한다.
    예전에는 그 회전을 편향으로 빼 2초 뒤 회전 오차가 20도를 넘었다.
    """
    sc = generate_camera_motion(0, duration_s=2.0, start_yaw_rate=0.2)
    track = _run(sc, policy)
    c = policy.camera_pose.confidence
    assert track.stats["static_init"] == 0
    assert max(s.confidence for s in track.samples) <= round(c.base * c.no_static_init, 3)
    assert math.degrees(_errors(sc, track)[3]) < 2.0


def test_imu_starting_after_the_first_frame_still_gives_poses(policy: PrelabelPolicy) -> None:
    """IMU 첫 표본이 첫 프레임보다 늦으면(사이드카 동기화 오프셋, F-PAR-47 재현) IMU가 덮는 첫
    프레임부터 자세를 낸다. 예전에는 첫 프레임에서 'IMU가 끝났다'로 멈춰 자세가 0개였다. 초기화
    구간(500 ms)이 움직임 시작 전에 끝나도록 정지 구간을 1초로 둔다 (그 구간이 움직임에 걸치면
    중력 방향이 가속도로 틀어지는 것은 이 결함과 별개다).
    """
    for late in (10.0, 300.0):
        sc = generate_camera_motion(4, duration_s=2.0, static_s=1.0)
        imu = _imu_without(sc, -1e9, late)
        track = _odometry(policy).estimate(
            sc.frames(), imu=imu, imu_to_camera=_r_ci(sc), intrinsics=sc.intrinsics, depth=sc
        )
        assert len(track.samples) > 10, late
        assert track.samples[0].t_ms >= late and track.lost_at_ms is None
        assert math.degrees(_errors(sc, track)[3]) < 1.0


def test_imu_gap_stops_instead_of_bridging(policy: PrelabelPolicy) -> None:
    """IMU가 1.0~2.0초를 놓치면(F-PAR-48 재현) 빈틈 앞에서 자세를 멈추고 lost_at_ms를 남긴다 (빈틈
    너머 회전을 지어내지 않는다). 표본 하나(5 ms)만 빠진 짧은 빈틈은 그대로 끝까지 잇는다. 초기화
    구간 안의 빈틈은 거부한다.
    """
    sc = generate_camera_motion(0, duration_s=3.0)
    track = _odometry(policy).estimate(
        sc.frames(), imu=_imu_without(sc, 1000, 2000), imu_to_camera=_r_ci(sc),
        intrinsics=sc.intrinsics, depth=sc,
    )  # fmt: skip
    assert track.lost_at_ms is not None and track.lost_at_ms >= 1000
    assert track.samples and all(s.t_ms < 1000 for s in track.samples)
    assert math.degrees(_errors(sc, track)[3]) < 0.5
    short = _odometry(policy).estimate(
        sc.frames(), imu=_imu_without(sc, 1000, 1006), imu_to_camera=_r_ci(sc),
        intrinsics=sc.intrinsics, depth=sc,
    )  # fmt: skip
    full = _run(sc, policy)
    assert short.lost_at_ms is None and len(short.samples) == len(full.samples)
    with pytest.raises(ValueError, match="끊겼"):
        _odometry(policy).estimate(
            sc.frames(), imu=_imu_without(sc, 200, 300), imu_to_camera=_r_ci(sc),
            intrinsics=sc.intrinsics, depth=sc,
        )  # fmt: skip


def test_calibration_of_another_aspect_ratio_is_not_used_for_poses(
    policy: PrelabelPolicy,
) -> None:
    """화면비가 다른 캘리브레이션(16:9)을 4:3 영상에 주면 화각 근사로 떨어지고 신뢰도에
    hfov_approx를 곱한다 (`lift3d`와 같은 규칙, F-PAR-44). 라벨의 내부 파라미터 출처는 러너가
    `calibrated`로 정한다 (F-PAR-51)."""
    sc = generate_camera_motion(0, duration_s=1.0)
    wide = CameraIntrinsics(width=640, height=360, fx=320, fy=320, cx=320, cy=180)
    est = _odometry(policy)
    track = est.estimate(sc.frames(), imu=_imu(sc), imu_to_camera=_r_ci(sc), intrinsics=wide,
                         depth=sc)  # fmt: skip
    c = policy.camera_pose.confidence
    assert not est.calibrated
    assert track.samples[0].confidence == round(c.base * c.hfov_approx, 3)
    label = pose_label(
        track, session_id="s", stream_id="bodycam", version="v", ontology_version="1.0.0",
        now=FIXED_TIME, intrinsics=IntrinsicsSource.HFOV_APPROX,
    )  # fmt: skip
    assert label is not None and isinstance(label.payload, Trajectory3DPayload)
    assert label.payload.intrinsics is IntrinsicsSource.HFOV_APPROX


def test_lk_termination_criteria_are_policy_values(policy: PrelabelPolicy) -> None:
    """LK 반복 종료 기준은 정책 값이라 바꾸면 camera_pose 절 해시(자세 단계 버전)가 바뀐다
    (F-PAR-55). 저장소 값은 OpenCV 기본값(30회, 0.01 px)과 같다."""
    f = policy.camera_pose.features
    assert (f.lk_max_iterations, f.lk_epsilon) == (30, 0.01)
    features = f.model_copy(update={"lk_max_iterations": 1})
    changed = policy.model_copy(
        update={"camera_pose": policy.camera_pose.model_copy(update={"features": features})}
    )
    assert changed.digest("camera_pose") != policy.digest("camera_pose")


def test_depth_is_required_for_metric_translation(policy: PrelabelPolicy) -> None:
    """깊이가 없으면 내장 오도메트리는 거부한다 (배율을 지어내지 않는다)."""
    sc = generate_camera_motion(0, duration_s=1.0)
    est = _odometry(policy, hfov_deg=90)
    with pytest.raises(ValueError, match="깊이"):
        est.estimate(
            sc.frames(), imu=_imu(sc), imu_to_camera=_r_ci(sc), intrinsics=None, depth=None
        )


# ---------------------------------------------------------------- 라벨과 월드 좌표


def _oracle_track(sc: CameraMotionScenario, stride: int = 100) -> PoseTrack:
    """정답 자세 stub 결과."""
    est = OraclePoseEstimator(
        {t: sc.pose(t) for t in sc.frame_times}, frame_stride_ms=stride, confidence=0.9
    )
    return est.estimate(
        sc.frames(), imu=_imu(sc), imu_to_camera=_r_ci(sc), intrinsics=None, depth=None
    )


def test_pose_label_round_trip_and_ontology() -> None:
    """자세 궤적 라벨: entity camera, frame world, 쿼터니언 왕복(1e-5), 온톨로지 검사 통과."""
    sc = generate_camera_motion(0, duration_s=1.5)
    track = _oracle_track(sc)
    label = pose_label(
        track, session_id="s1", stream_id="bodycam", version="campose-oracle+pabc+i123",
        ontology_version="1.0.0", now=FIXED_TIME,
    )  # fmt: skip
    assert label is not None and is_pose_label(label)
    assert label.label_id.startswith("s1-bodycam-campose-")
    assert check_label(label, load_ontology(ROOT / "config/ontology/v1")) == []
    back = pose_samples(label)
    for a, b in zip(track.samples, back, strict=True):
        assert a.t_ms == b.t_ms
        assert rotation_angle(a.rotation.T @ b.rotation) < 1e-5
        assert np.allclose(a.position, b.position, atol=1e-4)
    empty = PoseTrack(samples=[])
    assert (
        pose_label(
            empty,
            session_id="s1",
            stream_id="bodycam",
            version="v",
            ontology_version="1.0.0",
            now=FIXED_TIME,
        )
        is None
    )


def test_world_samples_match_truth_with_oracle_and_estimated_poses(policy: PrelabelPolicy) -> None:
    """카메라 좌표 손 6점 → 월드: 정답 자세면 자세 표본과 같은 시각은 0.2 mm(반올림) 안, 자세 사이
    보간 시각을 합쳐 평균 4 mm 안. 내장 오도메트리 자세면 평균 1.5 cm·최대 3 cm 안. 신뢰도 = 원래
    표본 x 자세, 방식은 그대로, derived_from이 남는다."""
    sc = generate_camera_motion(0)
    cams = sc.camera_labels(stride_ms=66)
    gap = policy.world_frame.max_pose_gap_ms
    for track, mean_tol, max_tol in (
        (_oracle_track(sc), 0.004, 0.015),
        (_run(sc, policy), 0.015, 0.03),
    ):
        pose = pose_label(
            track, session_id=sc.session_id, stream_id="bodycam", version="campose-x",
            ontology_version="1.0.0", now=FIXED_TIME,
        )  # fmt: skip
        assert pose is not None
        errors: list[float] = []
        for cam in cams:
            out = world_label(
                cam, pose, track.samples, version="world3d-1+px", max_gap_ms=gap, now=FIXED_TIME
            )
            assert out is not None
            p = out.payload
            assert isinstance(p, Trajectory3DPayload) and p.frame is CoordinateFrame.WORLD
            assert p.derived_from == (cam.label_id, pose.label_id) and p.part in HAND_PARTS
            j = HAND_PARTS.index(p.part or "")
            for s in p.samples:
                truth: NDArray[np.float64] = sc.hand_world(s.t_ms)[j]
                err = float(np.linalg.norm(np.array([s.x, s.y, s.z]) - truth))
                errors.append(err)
                if mean_tol < 0.01 and s.t_ms in {q.t_ms for q in track.samples}:
                    assert err < 2e-4
                assert s.method is Lift3DMethod.HAND_FIT
                assert s.confidence is not None and s.confidence <= 0.8 * 0.9 + 1e-9
        assert np.mean(errors) < mean_tol and max(errors) < max_tol


def test_world_transform_does_not_extrapolate() -> None:
    """자세 범위 밖·자세 간격이 max_pose_gap_ms보다 큰 시각의 표본은 바꾸지 않는다."""
    eye = np.eye(3)
    poses = [
        PoseSample(100, eye, np.zeros(3), 0.9),
        PoseSample(200, eye, np.array([1.0, 0.0, 0.0]), 0.5),
        PoseSample(600, eye, np.array([2.0, 0.0, 0.0]), 0.9),
    ]
    got = pose_at(poses, 150, 250)
    assert got is not None and np.allclose(got[1], [0.5, 0, 0]) and got[2] == 0.5
    assert pose_at(poses, 400, 250) is None  # 200~600은 400 ms 간격
    assert pose_at(poses, 50, 250) is None and pose_at(poses, 700, 250) is None
    from dlp_schema.labels import Trajectory3DSample

    samples = [Trajectory3DSample(t_ms=t, x=0, y=0, z=1) for t in (50, 150, 400, 600)]
    out = to_world_samples(samples, poses, 250)
    assert [s.t_ms for s in out] == [150, 600]
    assert out[0].confidence == 0.5  # 표본 신뢰도가 없으면 자세 신뢰도만
