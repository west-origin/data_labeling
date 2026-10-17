"""바디캠 6자유도 자세 추정 (ADR 0046, F-PAR-26).

좌표계 규약
    - 카메라 좌표: X 오른쪽, Y 아래, Z 앞(광축), 미터 (`lift3d`와 같다).
    - 세션 월드 좌표: Z 위(중력 반대), 원점 = 첫 자세의 카메라 중심, X = 첫 자세 광축의 수평 성분,
      Y = Z x X (왼쪽). 미터, 오른손 좌표계.
    - 자세 표본: 카메라 → 월드 회전 R_wc와 카메라 중심 p (월드). 월드 점 = R_wc · 카메라 점 + p.
    - IMU → 카메라 외부 파라미터: `Session.calibration.camera_imu_extrinsics`(4x4, IMU 점 → 카메라
    점).
      회전만 쓴다 (이동 팔 길이는 이동을 영상으로 풀어 필요 없다).
    - 시각: 자세 표본은 바디캠 PTS ms다. IMU 표본 시각은 호출자가 `Stream.to_master_ms`로
    바디캠(마스터)
      시각으로 바꿔 넘긴다 (`ImuSeries`). 바디캠은 마스터라 PTS ms = 마스터 ms다 (ADR 0019).

추정기 (`CameraPoseEstimator` 어댑터)
    - `ImuVisualOdometry` (내장, CPU): 회전은 IMU 자이로 적분 + 가속도 기울기 보정(상보 필터, 첫
    구간의
      평균 가속도로 중력 정렬, 정지면 자이로 편향 제거), 이동은 이전 자세 프레임의 특징점(Shi-Tomasi
      +
      피라미드 LK, 앞뒤 왕복 검사)을 메트릭 깊이로 월드 3D로 올리고, 지금 프레임 2D 투영식에서
      회전을 고정한 채 카메라 중심 3개 미지수를 선형으로 푼다 (두 점 RANSAC → 인라이어 깊이 가중
      재풀이). 루프 닫힘·묶음 조정은 없다 (오도메트리 드리프트가 쌓인다).
    - `OraclePoseEstimator` (CPU stub): 정답 자세를 그대로 돌려준다 (합성 픽스처, 러너 시험).
    - TODO(real-model): 루프 닫힘·묶음 조정이 있는 외부 시각-관성 SLAM 엔진(Basalt, BSD-3)은
    연동하지
      않았다. 같은 `CameraPoseEstimator` 어댑터로 붙인다 (ADR 0046 라이선스 조사).

파이프라인 위치: `runner._camera_pose`가 바디캠 영상·IMU·깊이 모델로 부르고 결과를 trajectory3d
(entity_id="camera", frame=world)로 저장한다 (`pose_label`). 월드 좌표 손 궤적은 `worldframe`이
만든다. 정책: `prelabel.yaml camera_pose`.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from dlp_models.depth import sample_depth
from dlp_prelabel.common import Image, model_label
from dlp_prelabel.lift3d import DepthModel
from dlp_prelabel.policy import CameraPosePolicy
from dlp_prelabel.posemath import exp_so3, matrix_from_quat, orthonormalize, quat_from_matrix, slerp
from dlp_schema.episode import version_tag
from dlp_schema.labels import (
    CoordinateFrame,
    LabelRecord,
    Source3D,
    Trajectory3DPayload,
    Trajectory3DSample,
)
from dlp_schema.session import CameraIntrinsics

Mat3 = NDArray[np.float64]
Vec3 = NDArray[np.float64]
# 카메라 자세 궤적의 개체 ID (계약 `Trajectory3DPayload`, 에피소드 `EntityKind.CAMERA`)
CAMERA_ENTITY = "camera"
_UP = np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class ImuSeries:
    """바디캠(마스터) 시각으로 바꾼 IMU 표본 (IMU 좌표계). 시각은 증가해야 한다."""

    t_ms: NDArray[np.float64]
    acc: NDArray[np.float64]  # (N, 3) 비력 m/s²
    gyro: NDArray[np.float64]  # (N, 3) rad/s


@dataclass(frozen=True)
class PoseSample:
    """자세 표본: 바디캠 PTS ms, 카메라 → 월드 회전, 카메라 중심 (월드 m), 신뢰도."""

    t_ms: int
    rotation: Mat3
    position: Vec3
    confidence: float


@dataclass
class PoseTrack:
    """추정 결과. samples는 시각 순. lost_at_ms: 추적을 잃어 그 뒤 자세를 내지 않은 시각 (없으면
    None).

    stats: 처리 프레임 수(frames), 자세를 낸 수(tracked), 추적 실패 수(failed), 정지 초기화 여부
    (static_init 0/1) 등 요약 (CLI 출력용).
    """

    samples: list[PoseSample]
    lost_at_ms: int | None = None
    stats: dict[str, float] = field(default_factory=dict[str, float])


class DepthSource(Protocol):
    """프레임 시각과 프레임 → 메트릭 깊이 맵 (H, W) m. 합성 픽스처는 시각으로 정답 깊이를 낸다."""

    def depth_at(self, t_ms: int, image: Image) -> NDArray[np.float32]:
        """그 프레임의 깊이 맵."""
        ...


@dataclass
class ModelDepth:
    """깊이 추론기(`lift3d.DepthModel`)를 `DepthSource`로 쓴다 (시각은 쓰지 않는다)."""

    model: DepthModel

    def depth_at(self, t_ms: int, image: Image) -> NDArray[np.float32]:
        """`model.predict(image)`."""
        return self.model.predict(image)


class CameraPoseEstimator(Protocol):
    """바디캠 자세 추정기 어댑터 (내장 오도메트리, stub, 외부 SLAM 엔진이 같은 모양으로 붙는다).

    name: 단계 이름. version: 모델 버전에 들어가는 추정기 판 (알고리즘·가중치가 바뀌면 바꾼다).
    """

    name: str
    version: str

    def estimate(
        self,
        frames: Iterable[tuple[int, Image]],
        *,
        imu: ImuSeries,
        imu_to_camera: Mat3,
        intrinsics: CameraIntrinsics | None,
        depth: DepthSource | None,
    ) -> PoseTrack:
        """바디캠 프레임·IMU(마스터 시각)·깊이로 자세 궤적을 낸다."""
        ...


# ---------------------------------------------------------------- IMU 회전


@dataclass(frozen=True)
class Orientation:
    """IMU 회전 적분 결과: 표본 시각(ms)마다 IMU → 월드 회전 R_wi (N, 3, 3), 정지 초기화 여부."""

    t_ms: NDArray[np.float64]
    r_wi: NDArray[np.float64]
    static_init: bool

    def at(self, t_ms: float) -> Mat3 | None:
        """시각 t의 R_wi (두 표본 사이 구면 보간). 범위 밖이면 None."""
        if t_ms < self.t_ms[0] or t_ms > self.t_ms[-1]:
            return None
        i = int(np.searchsorted(self.t_ms, t_ms))
        if self.t_ms[i] == t_ms or i == 0:
            return self.r_wi[i]
        t0, t1 = self.t_ms[i - 1], self.t_ms[i]
        return slerp(self.r_wi[i - 1], self.r_wi[i], float((t_ms - t0) / (t1 - t0)))


def initial_imu_rotation(up_i: Vec3, forward_i: Vec3) -> Mat3:
    """중력 반대(위) 방향과 카메라 광축(둘 다 IMU 좌표) → 첫 R_wi (모듈 설명의 월드 규약).

    Raises:
        ValueError: 광축이 거의 수직이라 수평 방향(X)을 정할 수 없을 때.
    """
    z = up_i / np.linalg.norm(up_i)
    horizontal = forward_i - float(forward_i @ z) * z
    if np.linalg.norm(horizontal) < 1e-3:
        raise ValueError("첫 자세의 광축이 수직이라 월드 X를 정할 수 없습니다")
    x = horizontal / np.linalg.norm(horizontal)
    y = np.cross(z, x)
    # 열 = IMU 좌표로 본 월드 축 → R_iw. R_wi는 그 전치
    return np.stack([x, y, z], axis=1).T


def imu_orientation(
    imu: ImuSeries, r_ci: Mat3, start_ms: float, policy: CameraPosePolicy
) -> Orientation:
    """IMU 자이로 적분 + 가속도 기울기 보정으로 R_wi를 낸다 (start_ms 이후 표본만).

    1. start_ms(그 앞 마지막 표본)부터 `init_window_ms`의 평균 가속도(비력, 위 방향)로 중력을
    정하고, 카메라 광축의 수평
       성분을 월드 X로 둔다 (`initial_imu_rotation`). 그 구간이 정지(`static_*_std` 이하)면 자이로
       평균을 편향으로 뺀다.
    2. 표본마다 R_wi ← R_wi · exp(ω Δt) (두 표본 평균 각속도), 가속도 크기가 중력과
       `accel_tolerance` 안이면 측정 위 방향을 월드 Z 쪽으로 `accel_gain` x Δt만큼 돌린다 (수평축
       회전이라 요는 바꾸지 않는다).

    Raises:
        ValueError: start_ms 뒤 초기 구간 IMU 표본이 2개보다 적을 때, 광축이 수직일 때.
    """
    p = policy.imu
    # start_ms를 덮도록 그 앞 마지막 표본부터 쓴다
    first = max(int(np.searchsorted(imu.t_ms, start_ms, side="right")) - 1, 0)
    t, acc, gyro = imu.t_ms[first:], imu.acc[first:], imu.gyro[first:]
    win = t <= start_ms + p.init_window_ms
    if int(win.sum()) < 2:
        raise ValueError("자세 초기화 구간에 IMU 표본이 부족합니다 (IMU 동기화·범위를 확인하세요)")
    acc_norm = np.linalg.norm(acc[win], axis=1)
    gyro_norm = np.linalg.norm(gyro[win], axis=1)
    static = bool(acc_norm.std() <= p.static_acc_std and gyro_norm.std() <= p.static_gyro_std)
    bias = gyro[win].mean(axis=0) if static else np.zeros(3)
    forward_i = r_ci.T @ np.array([0.0, 0.0, 1.0])
    r = initial_imu_rotation(acc[win].mean(axis=0), forward_i)
    out = np.empty((t.size, 3, 3))
    out[0] = r
    for k in range(1, t.size):
        dt = (t[k] - t[k - 1]) / 1000
        omega = (gyro[k] + gyro[k - 1]) / 2 - bias
        r = r @ exp_so3(omega * dt)
        a: NDArray[np.float64] = acc[k]
        norm = float(np.linalg.norm(a))
        if p.accel_gain > 0 and abs(norm - p.gravity) < p.accel_tolerance:
            v = r @ (a / norm)  # 지금 추정으로 본 측정 위 방향 (월드)
            r = exp_so3(np.cross(v, _UP) * p.accel_gain * dt) @ r
        out[k] = orthonormalize(r) if k % 100 == 0 else r
    return Orientation(t, out, static)


# ---------------------------------------------------------------- 이동량


@dataclass(frozen=True)
class TranslationFit:
    """회전을 고정한 이동량 풀이: 카메라 중심(월드), 인라이어 수, 인라이어 재투영 RMS(px)."""

    position: Vec3
    inliers: int
    rms_px: float


def _project_errors(
    y: NDArray[np.float64], c: Vec3, uv: NDArray[np.float64], k: CameraIntrinsics
) -> NDArray[np.float64]:
    """회전된 점 y(= R_cw X)와 c(= R_cw p)의 재투영 오차(px). 카메라 뒤 점은 inf."""
    z = y[:, 2] - c[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = k.fx * (y[:, 0] - c[0]) / z + k.cx
        v = k.fy * (y[:, 1] - c[1]) / z + k.cy
    err = np.hypot(u - uv[:, 0], v - uv[:, 1])
    return np.where(z > 1e-3, err, np.inf)


def _linear_center(
    y: NDArray[np.float64], xn: NDArray[np.float64], weights: NDArray[np.float64] | None = None
) -> Vec3 | None:
    """투영식 (y_x - c_x) - x̂ (y_z - c_z) = 0 (x, y 두 줄)을 c에 대해 최소제곱으로 푼다."""
    n = y.shape[0]
    a = np.zeros((2 * n, 3))
    b = np.zeros(2 * n)
    a[0::2, 0], a[0::2, 2] = -1.0, xn[:, 0]
    a[1::2, 1], a[1::2, 2] = -1.0, xn[:, 1]
    b[0::2] = xn[:, 0] * y[:, 2] - y[:, 0]
    b[1::2] = xn[:, 1] * y[:, 2] - y[:, 1]
    if weights is not None:
        w = np.repeat(weights, 2)
        a, b = a * w[:, None], b * w
    sol, _, rank, _ = np.linalg.lstsq(a, b, rcond=None)
    return sol if rank == 3 else None


def solve_translation(
    points_w: NDArray[np.float64],
    uv: NDArray[np.float64],
    r_wc: Mat3,
    k: CameraIntrinsics,
    policy: CameraPosePolicy,
) -> TranslationFit | None:
    """월드 3D 점과 지금 프레임 2D, 아는 회전 R_wc로 카메라 중심을 푼다 (두 점 RANSAC + 재풀이).

    Returns:
        인라이어가 `min_inliers` 이상이면 `TranslationFit`, 아니면 None.
    """
    tp = policy.translation
    n = points_w.shape[0]
    if n < max(tp.min_inliers, 2):
        return None
    r_cw = r_wc.T
    y = points_w @ r_cw.T
    xn = np.stack([(uv[:, 0] - k.cx) / k.fx, (uv[:, 1] - k.cy) / k.fy], axis=1)
    rng = np.random.default_rng(tp.seed)
    best: NDArray[np.bool_] | None = None
    for _ in range(tp.ransac_iterations):
        pick = rng.choice(n, 2, replace=False)
        c = _linear_center(y[pick], xn[pick])
        if c is None:
            continue
        inl = _project_errors(y, c, uv, k) < tp.inlier_px
        if best is None or int(inl.sum()) > int(best.sum()):
            best = inl
    if best is None or int(best.sum()) < tp.min_inliers:
        return None
    c = _linear_center(y[best], xn[best])
    for _ in range(3):
        if c is None:
            return None
        # 대수 오차를 정규화 영상 오차에 가깝게: 깊이의 역수로 가중한 뒤 인라이어를 다시 고른다
        err = _project_errors(y, c, uv, k)
        best = err < tp.inlier_px
        if int(best.sum()) < tp.min_inliers:
            return None
        c = _linear_center(y[best], xn[best], 1.0 / np.maximum(y[best, 2] - c[2], 1e-3))
    if c is None:
        return None
    err = _project_errors(y, c, uv, k)
    best = err < tp.inlier_px
    if int(best.sum()) < tp.min_inliers:
        return None
    rms = float(np.sqrt(np.mean(err[best] ** 2)))
    return TranslationFit(r_wc @ c, int(best.sum()), rms)


def pose_confidence(
    fit: TranslationFit, *, static_init: bool, calibrated: bool, policy: CameraPosePolicy
) -> float:
    """자세 표본 신뢰도 (`camera_pose.confidence`, 소수 셋째 자리)."""
    c = policy.confidence
    inlier_term = min(1.0, fit.inliers / c.good_inliers)
    reproj = 1 - (1 - c.reprojection_floor) * min(fit.rms_px / policy.translation.inlier_px, 1.0)
    value = c.base * inlier_term * reproj
    if not static_init:
        value *= c.no_static_init
    if not calibrated:
        value *= c.hfov_approx
    return round(value, 3)


# ---------------------------------------------------------------- 특징점


@dataclass
class _Keyframe:
    """마지막으로 자세를 낸 프레임: 시각, 회색 영상, 특징점 2D (n, 2)와 그 월드 3D (n, 3)."""

    t_ms: int
    gray: NDArray[np.uint8]
    uv: NDArray[np.float32]
    points_w: NDArray[np.float64]


def _gray(image: Image) -> NDArray[np.uint8]:
    """RGB → 회색."""
    return np.asarray(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), dtype=np.uint8)


def detect_features(gray: NDArray[np.uint8], policy: CameraPosePolicy) -> NDArray[np.float32]:
    """Shi-Tomasi 모서리 (n, 2) float32 (없으면 (0, 2))."""
    f = policy.features
    # 모서리가 없으면 OpenCV가 None을 낸다 (타입 스텁에는 없다)
    pts = cast(
        "NDArray[np.float32] | None",
        cv2.goodFeaturesToTrack(
            gray,
            maxCorners=f.max_corners,
            qualityLevel=f.quality_level,
            minDistance=f.min_distance_px,
        ),
    )
    if pts is None:
        return np.zeros((0, 2), np.float32)
    return np.asarray(pts, dtype=np.float32).reshape(-1, 2)


def track_features(
    prev: NDArray[np.uint8],
    cur: NDArray[np.uint8],
    uv: NDArray[np.float32],
    policy: CameraPosePolicy,
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """피라미드 LK로 prev의 uv를 cur로 잇고, 거꾸로 다시 이어 왕복 오차가 `fb_max_px` 안인 것만
    남긴다.

    Returns:
        (cur의 2D (n, 2), 쓸 수 있는지 (n,)).
    """
    f = policy.features
    if uv.shape[0] == 0:
        return uv, np.zeros(0, dtype=np.bool_)
    win = (f.lk_window_px, f.lk_window_px)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    p0 = uv.reshape(-1, 1, 2)
    p1, st1, _ = cv2.calcOpticalFlowPyrLK(
        prev, cur, p0, np.empty_like(p0), winSize=win, maxLevel=f.lk_levels, criteria=crit
    )
    back, st2, _ = cv2.calcOpticalFlowPyrLK(
        cur, prev, p1, np.empty_like(p0), winSize=win, maxLevel=f.lk_levels, criteria=crit
    )
    p1a = np.asarray(p1, dtype=np.float32).reshape(-1, 2)
    fb = np.linalg.norm(np.asarray(back, dtype=np.float32).reshape(-1, 2) - uv, axis=1)
    ok = (np.asarray(st1, dtype=np.uint8).reshape(-1) == 1) & (
        np.asarray(st2, dtype=np.uint8).reshape(-1) == 1
    )
    ok &= fb < f.fb_max_px
    h, w = cur.shape
    ok &= (p1a[:, 0] >= 0) & (p1a[:, 0] < w) & (p1a[:, 1] >= 0) & (p1a[:, 1] < h)
    return p1a, ok


def lift_features(
    uv: NDArray[np.float32],
    dmap: NDArray[np.float32],
    k: CameraIntrinsics,
    r_wc: Mat3,
    p_wc: Vec3,
    policy: CameraPosePolicy,
) -> tuple[NDArray[np.float32], NDArray[np.float64]]:
    """특징점을 깊이로 월드 3D에 올린다 (깊이가 nan이거나 범위 밖이면 버린다). (2D, 월드 3D)."""
    tp = policy.translation
    keep: list[int] = []
    pts: list[tuple[float, float, float]] = []
    for i, (u, v) in enumerate(uv):
        z = sample_depth(dmap, float(u), float(v), tp.depth_patch_px)
        if math.isnan(z) or not tp.min_depth_m <= z <= tp.max_depth_m:
            continue
        keep.append(i)
        pts.append(((float(u) - k.cx) * z / k.fx, (float(v) - k.cy) * z / k.fy, z))
    if not keep:
        return np.zeros((0, 2), np.float32), np.zeros((0, 3))
    cam = np.array(pts, dtype=np.float64)
    return uv[keep], cam @ r_wc.T + p_wc


def scaled_intrinsics(
    k: CameraIntrinsics | None, width: int, height: int, hfov_deg: float
) -> CameraIntrinsics:
    """해상도에 맞춘 내부 파라미터 (없으면 가로 화각 근사, `lift3d.intrinsics_for`와 같은 규칙)."""
    if k is None:
        f = width / 2 / math.tan(math.radians(hfov_deg) / 2)
        return CameraIntrinsics(width=width, height=height, fx=f, fy=f, cx=width / 2, cy=height / 2)
    sx, sy = width / k.width, height / k.height
    return CameraIntrinsics(
        width=width, height=height, fx=k.fx * sx, fy=k.fy * sy, cx=k.cx * sx, cy=k.cy * sy
    )


# ---------------------------------------------------------------- 추정기


class ImuVisualOdometry:
    """내장 IMU 시각 오도메트리 (모듈 설명). CPU, 외부 엔진·가중치 없음 (깊이 모델은 넘겨받는다).

    version: `imuvo1` (알고리즘을 바꾸면 숫자를 올린다).
    """

    name = "campose"
    version = "imuvo1"

    def __init__(self, policy: CameraPosePolicy, *, hfov_deg: float) -> None:
        """policy: `prelabel.yaml camera_pose`. hfov_deg: 내부 파라미터가 없을 때 가로 화각
        (`depth.default_hfov_deg`)."""
        self.policy = policy
        self.hfov_deg = hfov_deg

    def estimate(
        self,
        frames: Iterable[tuple[int, Image]],
        *,
        imu: ImuSeries,
        imu_to_camera: Mat3,
        intrinsics: CameraIntrinsics | None,
        depth: DepthSource | None,
    ) -> PoseTrack:
        """자세 궤적 (`CameraPoseEstimator.estimate`).

        frame_stride_ms 간격 프레임마다: IMU 회전으로 R_wc를 정하고, 마지막 자세 프레임(키프레임)의
        특징점을 지금 프레임으로 추적해 이동량을 푼다. 첫 프레임은 원점이다. 풀이가 실패하면 그
        프레임은 자세가 없고, 키프레임에서 `max_lost_ms`가 지나도록 실패하면 멈춘다.

        Raises:
            ValueError: 깊이가 없을 때 (이동 배율을 정할 수 없다), IMU 초기화 실패.
        """
        if depth is None:
            raise ValueError("내장 오도메트리에는 메트릭 깊이가 필요합니다")
        policy = self.policy
        r_ic = imu_to_camera.T
        track = PoseTrack(samples=[], stats={"frames": 0, "tracked": 0, "failed": 0})
        orient: Orientation | None = None
        key: _Keyframe | None = None
        k: CameraIntrinsics | None = None
        last_t: int | None = None
        for t, img in frames:
            if last_t is not None and t - last_t < max(policy.frame_stride_ms, 1):
                continue
            if orient is None:
                orient = imu_orientation(imu, imu_to_camera, t, policy)
                track.stats["static_init"] = float(orient.static_init)
            r_wi = orient.at(t)
            if r_wi is None:
                break  # IMU가 끝났다
            last_t = t
            track.stats["frames"] += 1
            if k is None:
                k = scaled_intrinsics(intrinsics, img.shape[1], img.shape[0], self.hfov_deg)
            r_wc = r_wi @ r_ic
            gray = _gray(img)
            if key is None:
                position = np.zeros(3)
                conf = round(
                    policy.confidence.base
                    * (1.0 if orient.static_init else policy.confidence.no_static_init)
                    * (1.0 if intrinsics is not None else policy.confidence.hfov_approx),
                    3,
                )
            else:
                fit = self._step(key, gray, r_wc, k)
                if fit is None:
                    track.stats["failed"] += 1
                    if t - key.t_ms > policy.max_lost_ms:
                        track.lost_at_ms = t
                        break
                    continue
                position = fit.position
                conf = pose_confidence(
                    fit, static_init=orient.static_init, calibrated=intrinsics is not None,
                    policy=policy,
                )  # fmt: skip
            track.samples.append(PoseSample(t, r_wc, position, conf))
            track.stats["tracked"] += 1
            uv, pw = lift_features(
                detect_features(gray, policy), depth.depth_at(t, img), k, r_wc, position, policy
            )
            key = _Keyframe(t, gray, uv, pw)
        return track

    def _step(
        self, key: _Keyframe, gray: NDArray[np.uint8], r_wc: Mat3, k: CameraIntrinsics
    ) -> TranslationFit | None:
        """키프레임 특징점을 지금 프레임으로 추적해 이동량을 푼다 (실패하면 None)."""
        uv, ok = track_features(key.gray, gray, key.uv, self.policy)
        if int(ok.sum()) < self.policy.translation.min_inliers:
            return None
        return solve_translation(key.points_w[ok], uv[ok].astype(np.float64), r_wc, k, self.policy)


class OraclePoseEstimator:
    """정답 자세를 돌려주는 CPU stub (합성 픽스처·러너 시험). 영상·IMU·깊이를 읽지 않는다.

    poses: 바디캠 PTS ms → (R_wc, p). frame_stride_ms 간격 프레임 중 정답이 있는 시각만 낸다.
    """

    name = "campose"

    def __init__(
        self,
        poses: Mapping[int, tuple[Mat3, Vec3]],
        *,
        frame_stride_ms: int,
        confidence: float = 0.9,
        version: str = "oracle",
    ) -> None:
        """version: 모델 버전에 들어가는 판 (바꾸면 다시 돈다, 재실행 시험용)."""
        self.poses = dict(poses)
        self.stride = frame_stride_ms
        self.confidence = confidence
        self.version = version

    def estimate(
        self,
        frames: Iterable[tuple[int, Image]],
        *,
        imu: ImuSeries,
        imu_to_camera: Mat3,
        intrinsics: CameraIntrinsics | None,
        depth: DepthSource | None,
    ) -> PoseTrack:
        """stride 간격 프레임 시각의 정답 자세."""
        out: list[PoseSample] = []
        last: int | None = None
        for t, _ in frames:
            if t not in self.poses or (last is not None and t - last < max(self.stride, 1)):
                continue
            last = t
            r, p = self.poses[t]
            out.append(PoseSample(t, r, np.asarray(p, dtype=np.float64), self.confidence))
        return PoseTrack(out, stats={"frames": len(out), "tracked": len(out), "failed": 0})


# ---------------------------------------------------------------- 라벨 왕복


def pose_label(
    track: PoseTrack,
    *,
    session_id: str,
    stream_id: str,
    version: str,
    ontology_version: str,
    now: datetime,
) -> LabelRecord | None:
    """자세 궤적 → trajectory3d 라벨 (entity_id="camera", frame=world, 표본 쿼터니언). 표본이 없으면
    None.

    ID: `<세션>-<스트림>-campose-<version_tag>`. 라벨 신뢰도는 표본 신뢰도의 평균.
    """
    if not track.samples:
        return None
    samples: list[Trajectory3DSample] = []
    for s in track.samples:
        qw, qx, qy, qz = quat_from_matrix(s.rotation)
        x, y, z = (round(float(v), 4) for v in s.position)
        samples.append(
            Trajectory3DSample(
                t_ms=s.t_ms,
                x=x,
                y=y,
                z=z,
                qw=round(qw, 6),
                qx=round(qx, 6),
                qy=round(qy, 6),
                qz=round(qz, 6),
                confidence=s.confidence,
            )
        )
    payload = Trajectory3DPayload(
        entity_id=CAMERA_ENTITY,
        frame=CoordinateFrame.WORLD,
        source_3d=Source3D.VISUAL_INERTIAL_SLAM,
        samples=tuple(samples),
    )
    mean = round(sum(s.confidence for s in track.samples) / len(track.samples), 3)
    return model_label(
        label_id=f"{session_id}-{stream_id}-campose-{version_tag(version)}",
        session_id=session_id,
        stream_id=stream_id,
        t_start_ms=samples[0].t_ms,
        t_end_ms=samples[-1].t_ms,
        ontology_version=ontology_version,
        model_version=version,
        confidence=mean,
        payload=payload,
        now=now,
    )


def is_pose_label(label: LabelRecord) -> bool:
    """카메라 자세 궤적 라벨인가 (entity_id="camera", frame=world, 회전이 있음)."""
    p = label.payload
    return (
        isinstance(p, Trajectory3DPayload)
        and p.entity_id == CAMERA_ENTITY
        and p.frame is CoordinateFrame.WORLD
        and all(s.qw is not None for s in p.samples)
    )


def pose_samples(label: LabelRecord) -> list[PoseSample]:
    """자세 궤적 라벨 → 자세 표본 (시각 순). 표본 신뢰도가 없으면 라벨 신뢰도(없으면 0)를 쓴다.

    Raises:
        ValueError: 카메라 자세 라벨이 아닐 때 (`is_pose_label`).
    """
    if not is_pose_label(label):
        raise ValueError(f"{label.label_id}: 카메라 자세 궤적이 아닙니다")
    p = label.payload
    assert isinstance(p, Trajectory3DPayload)
    default = label.confidence or 0.0
    out: list[PoseSample] = []
    for s in sorted(p.samples, key=lambda q: q.t_ms):
        r = matrix_from_quat(s.qw or 0.0, s.qx or 0.0, s.qy or 0.0, s.qz or 0.0)
        conf = s.confidence if s.confidence is not None else default
        out.append(PoseSample(s.t_ms, r, np.array([s.x, s.y, s.z]), conf))
    return out


def poses_by_time(samples: Sequence[PoseSample]) -> dict[int, tuple[Mat3, Vec3]]:
    """표본 → {시각: (R_wc, p)} (시험·평가용)."""
    return {s.t_ms: (s.rotation, s.position) for s in samples}
