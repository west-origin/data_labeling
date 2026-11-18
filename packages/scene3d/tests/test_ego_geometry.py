"""위치 추정의 순수 계산 시험: 기술자 대응기, 닮음 변환을 거친 자세, PnP, 내부 파라미터 맞춤, 보간.

정답은 시험 안에서 만든 카메라·점이다 (저장소·DB·모델 없음, ADR 0067).
"""

from __future__ import annotations

from datetime import UTC, datetime

import cv2
import numpy as np
import pytest

from dlp_scene3d.geometry import (
    DescriptorMatcher,
    Similarity,
    interpolate_pose,
    look_at,
    match_descriptors,
    rigid_through_similarity,
)
from dlp_scene3d.localize import EgoCamera, PoseTrack, fit_intrinsics, pose_at
from dlp_scene3d.pipeline import Scene3dError
from dlp_scene3d.pnp import solve_pnp
from dlp_schema.scene3d import Alignment, EgoLocalization, FramePose
from dlp_schema.session import CameraIntrinsics

K = np.array([[400.0, 0.0, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]])
UP = np.array([0.0, 0.0, 1.0])


def _rot(axis: tuple[float, float, float], deg: float) -> np.ndarray:
    v = np.asarray(axis, dtype=np.float64)
    rvec = (v / float(np.linalg.norm(v)) * float(np.deg2rad(deg))).reshape(3, 1)
    r, _ = cv2.Rodrigues(rvec)
    return np.asarray(r, dtype=np.float64)


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    cos = float(np.clip((np.trace(a @ b.T) - 1) / 2, -1.0, 1.0))
    return float(np.rad2deg(np.arccos(cos)))


def _project(points: np.ndarray, r: np.ndarray, t: np.ndarray, dist: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(r)
    uv, _ = cv2.projectPoints(points, rvec, t.reshape(3, 1), K, dist)
    return np.asarray(uv, dtype=np.float64).reshape(-1, 2)


def test_descriptor_matcher_agrees_with_match_descriptors() -> None:
    """전수 비교 대응기는 k-d 트리 대응(`match_descriptors`)과 같은 쌍을 낸다 (상호 최근접 +
    비율 검사)."""
    rng = np.random.default_rng(0)
    scene = rng.normal(size=(700, 32))
    picked = rng.choice(len(scene), 300, replace=False)
    query = np.concatenate(
        [scene[picked] + rng.normal(scale=0.3, size=(300, 32)), rng.normal(size=(150, 32))]
    )
    matcher = DescriptorMatcher.build(scene)
    pairs = matcher.match(query, 0.8)
    assert np.array_equal(pairs, match_descriptors(query, scene, 0.8))
    # 같은 장면 점을 닮은 질의 둘 중에는 그 점에서 더 가까운 하나만 남는다 (상호 검사)
    assert len(set(pairs[:, 1].tolist())) == len(pairs) > 200
    assert matcher.match(query[:1], 0.8).shape == (0, 2)
    assert DescriptorMatcher.build(scene[:1]).match(query, 0.8).shape == (0, 2)


def test_rigid_pose_through_similarity_keeps_pixels_and_scales_depth() -> None:
    """after 좌표의 자세를 닮음 변환으로 before 좌표에 옮기면 같은 점이 같은 화소에 찍히고 깊이만
    척도 배가 된다. 회전에 척도를 곱하지 않는다 (R은 회전 행렬 그대로)."""
    rng = np.random.default_rng(1)
    a_to_b = Similarity(1.25, _rot((0.2, 0.3, 0.9), 35.0), np.array([0.4, -0.7, 0.2]))
    centre = np.array([0.3, -1.0, 1.4])
    r_ca, t_ca = look_at(centre, np.array([0.0, 0.5, 0.3]), UP)
    x_a = rng.uniform(-2, 2, size=(50, 3))
    r_cb, t_cb = rigid_through_similarity(r_ca, t_ca, a_to_b)

    cam_a = x_a @ r_ca.T + t_ca
    cam_b = a_to_b.apply(x_a) @ r_cb.T + t_cb
    assert np.allclose(cam_b, a_to_b.s * cam_a, atol=1e-12)
    assert np.allclose(r_cb @ r_cb.T, np.eye(3), atol=1e-12) and np.linalg.det(r_cb) > 0.999
    # 카메라 중심은 점처럼 옮겨진다
    assert np.allclose(-r_cb.T @ t_cb, a_to_b.apply(centre[None])[0], atol=1e-12)
    # 척도를 그대로 곱한 "자세"는 틀리다: 회전 행렬이 아니게 된다
    assert not np.allclose((a_to_b.s * r_cb) @ (a_to_b.s * r_cb).T, np.eye(3))


@pytest.mark.parametrize("distortion", [(), (-0.28, 0.09, 0.001, -0.0006, 0.0)])
def test_pnp_recovers_pose_with_noise_outliers_and_distortion(
    distortion: tuple[float, ...],
) -> None:
    """화소 잡음 0.5와 틀린 대응 60%가 있어도 자세를 1 cm·0.2도 안에서 되찾고, 틀린 대응은 내점이
    아니다. 왜곡 계수가 있는 카메라도 같다. 같은 입력이면 같은 결과다."""
    rng = np.random.default_rng(2)
    dist = np.asarray(distortion, dtype=np.float64)
    points = rng.uniform(-3, 3, size=(600, 3))
    centre = np.array([0.2, -0.4, 0.3])
    r, t = look_at(centre, centre + np.array([1.0, 0.3, 0.1]), UP)
    uv = _project(points, r, t, dist)
    depth = (points @ r.T + t)[:, 2]
    seen = np.flatnonzero(
        (depth > 0.3) & (uv[:, 0] > 0) & (uv[:, 0] < 640) & (uv[:, 1] > 0) & (uv[:, 1] < 480)
    )[:60]
    n_wrong = 90
    obj = np.concatenate([points[seen], points[rng.choice(len(points), n_wrong)]])
    img = np.concatenate(
        [
            uv[seen] + rng.normal(scale=0.5, size=(len(seen), 2)),
            rng.uniform(0, 1, size=(n_wrong, 2)) * np.array([640.0, 480.0]),
        ]
    )
    sol = solve_pnp(obj, img, K, dist, threshold_px=2.0, iterations=5000, confidence=0.999)
    assert sol is not None and sol.matches == len(obj)
    assert np.linalg.norm(-sol.r.T @ sol.t - centre) < 0.01
    assert _angle_deg(sol.r, r) < 0.2
    assert len(sol.inliers) >= 0.9 * len(seen) and sol.inliers.max() < len(seen) + 3
    assert sol.reproj_px < 1.0
    again = solve_pnp(obj, img, K, dist, threshold_px=2.0, iterations=5000, confidence=0.999)
    assert again is not None and np.array_equal(again.r, sol.r) and np.array_equal(again.t, sol.t)


def test_pnp_returns_none_without_enough_points() -> None:
    """대응이 4개 미만이면 풀지 않는다."""
    points = np.eye(3)
    assert solve_pnp(points, np.zeros((3, 2)), K, np.zeros(0), threshold_px=2.0, iterations=100,
                     confidence=0.99) is None  # fmt: skip


def test_fit_intrinsics_keeps_scales_or_refuses() -> None:
    """같은 크기·홀수 변 덧댐은 값 그대로, 같은 화면비의 축소는 비례 조정, 다른 화면비는 거부."""
    calib = CameraIntrinsics(
        width=1919, height=1079, fx=1000.0, fy=1001.0, cx=960.0, cy=540.0, distortion=(-0.1, 0.02)
    )
    padded = fit_intrinsics(calib, 1920, 1080, 0.02)
    assert (padded.width, padded.height) == (1920, 1080)
    assert (padded.fx, padded.fy, padded.cx, padded.cy) == (1000.0, 1001.0, 960.0, 540.0)
    full = CameraIntrinsics(width=1920, height=1080, fx=1000.0, fy=1000.0, cx=960.0, cy=540.0)
    assert fit_intrinsics(full, 1920, 1080, 0.02) == full
    half = fit_intrinsics(full, 960, 540, 0.02)
    assert (half.fx, half.fy, half.cx, half.cy) == (500.0, 500.0, 480.0, 270.0)
    assert half.distortion == full.distortion
    with pytest.raises(Scene3dError, match="화면비"):
        fit_intrinsics(full, 1440, 1080, 0.02)
    cam = EgoCamera.of(padded)
    assert cam.k[0, 0] == 1000.0 and cam.k[1, 2] == 540.0 and cam.dist.tolist() == [-0.1, 0.02]


def test_interpolate_pose_moves_centre_linearly_and_rotation_on_the_arc() -> None:
    """카메라 중심은 두 중심을 잇는 직선 위, 회전은 두 회전 사이 호 위에 있다."""
    r0, t0 = look_at(np.array([1.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.5]), UP)
    r1, t1 = look_at(np.array([0.0, 2.0, 1.4]), np.array([0.0, 0.0, 0.5]), UP)
    total = _angle_deg(r0, r1)
    assert total > 30
    for alpha in (0.0, 0.25, 0.5, 1.0):
        r, t = interpolate_pose(r0, t0, r1, t1, alpha)
        centre = -r.T @ t
        want = (1 - alpha) * np.array([1.0, 0.0, 1.0]) + alpha * np.array([0.0, 2.0, 1.4])
        assert np.allclose(centre, want, atol=1e-9)
        assert _angle_deg(r0, r) == pytest.approx(alpha * total, abs=1e-6)
        assert _angle_deg(r, r1) == pytest.approx((1 - alpha) * total, abs=1e-6)


def _record(times: list[int]) -> EgoLocalization:
    """시각마다 x축으로 0.1 m씩 가고 z축 둘레로 10도씩 도는 위치 기록."""
    poses: list[FramePose] = []
    for i, t_ms in enumerate(times):
        r = _rot((0.0, 0.0, 1.0), 10.0 * i)
        t = -r @ np.array([0.1 * i, 0.0, 1.0])
        rv, tv = [float(v) for v in r.reshape(-1)], [float(v) for v in t]
        poses.append(
            FramePose(
                t_ms=t_ms,
                reference="before",
                rotation=(rv[0], rv[1], rv[2], rv[3], rv[4], rv[5], rv[6], rv[7], rv[8]),
                translation=(tv[0], tv[1], tv[2]),
                matches=40,
                inliers=30,
                reproj_px=0.5,
            )
        )
    identity = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    return EgoLocalization(
        session_id="s",
        stream_id="bodycam",
        model_version="v",
        before_reconstruction="b",
        after_reconstruction="a",
        alignment_tag="al",
        alignment=Alignment(
            model_version="v",
            scale=1.0,
            rotation=identity,
            translation=(0.0, 0.0, 0.0),
            matches=10,
            inliers=10,
            rms_m=0.0,
            accepted=True,
        ),
        render_hash="0" * 64,
        intrinsics=CameraIntrinsics(width=640, height=480, fx=400, fy=400, cx=320, cy=240),
        intrinsics_origin="session",
        metric=True,
        frame_count=len(times) + 2,
        localized=len(times),
        poses=tuple(poses),
        created_at=datetime(2026, 11, 14, tzinfo=UTC),
    )


def test_pose_at_interpolates_between_neighbours_and_refuses_gaps() -> None:
    """기록된 시각은 그 자세, 사이 시각은 이웃 둘의 보간. 이웃 간격이 한계를 넘거나(풀지 못한
    프레임이 사이에 있다) 기록 밖이면 None이다 (밖으로 늘리지 않는다)."""
    record = _record([0, 200, 400, 1200, 1400])
    track = PoseTrack.of(record)
    exact = track.at(200, 500)
    assert exact is not None and np.allclose(exact.center, [0.1, 0.0, 1.0])
    mid = track.at(300.0, 500)
    assert mid is not None and np.allclose(mid.center, [0.15, 0.0, 1.0], atol=1e-9)
    assert _angle_deg(mid.r, _rot((0.0, 0.0, 1.0), 15.0)) < 1e-6
    quarter = pose_at(record, 250, 500)
    assert quarter is not None and np.allclose(quarter.center, [0.125, 0.0, 1.0], atol=1e-9)
    # 400~1200 ms 사이는 간격 800 ms: 한계 500이면 없음, 한계를 800으로 올리면 보간한다
    assert track.at(800, 500) is None
    wide = track.at(800, 800)
    assert wide is not None and np.allclose(wide.center, [0.25, 0.0, 1.0], atol=1e-9)
    # 간격이 큰 구간의 끝점 자체는 기록된 자세다
    assert track.at(1200, 500) is not None
    assert track.at(-1, 500) is None and track.at(1401, 500) is None
    assert PoseTrack.of(_record([])).at(0, 500) is None
