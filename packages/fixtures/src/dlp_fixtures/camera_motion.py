"""정답 자세를 아는 머리 카메라 움직임: 렌더 프레임, 해석적 깊이, IMU 신호, 월드 좌표 손 (ADR 0046).

장면은 중력 정렬 월드 좌표(Z 위, 미터)의 무늬 있는 두 평면(탁자 z = -0.7, 벽 x = 2.5)이다. 머리
카메라는 처음 `static_ms` 동안 가만히 있다가(자이로 편향·중력 초기화 구간) 앞뒤·좌우로 움직이며
고개를 돌리고 숙인다. 월드 좌표는 추정기 규약과 같다: 원점 = 첫 자세의 카메라 중심, X = 첫 자세
광축의 수평 성분, Z = 위 (`dlp_prelabel.campose`).

정답과 출력 (`CameraMotionScenario`)
- `R_wc[t]`, `p_wc[t]`: 프레임 시각(바디캠 PTS ms)마다 카메라 → 월드 회전과 카메라 중심 (정답).
  `pose_at(t_s)`로 임의 시각의 정답도 얻는다.
- `frames()`: (PTS ms, 렌더 RGB 프레임). 평면 무늬를 광선-평면 교차로 그린다 (특징점 추적 입력).
- `depth_at(t_ms, image)`: 그 프레임의 정답 메트릭 깊이 (광축 방향, m). `depth_bias`로 배율 오차를
넣는다.
- `imu`: IMU 시계(ms)의 가속도(비력, m/s²)·자이로(rad/s), IMU 좌표계. 바디캠 시각 = IMU 시각 +
  `imu_offset_ms` (`Stream.to_master_ms`와 같은 식). IMU → 카메라 회전 `imu_to_camera`(4x4, 이동 0).
  `gyro_bias`, `gyro_noise`, `acc_noise`로 센서 오차를 넣는다.
- `hand_world[t]`: 월드 좌표 손 6점 (손목, 엄지·검지·중지·약지·새끼 끝) (6, 3) m — 탁자 위를 닦는
손.
  `hand_cam[t]`: 같은 점의 카메라 좌표. `camera_labels(...)`: 그것을 3D 단계(`lift3d`) 출력처럼 쓴
  카메라 좌표 trajectory3d 라벨 (선택 흔들림).

테스트 사용처: `dlp_prelabel`(카메라 자세 추정 ATE·RPE, 월드 좌표 손 오차, 러너의 자세·월드 단계).
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, field

import cv2
import numpy as np
from numpy.typing import NDArray

from dlp_schema.labels import (
    CoordinateFrame,
    LabelRecord,
    Lift3DMethod,
    Provenance,
    Source,
    Source3D,
    Trajectory3DPayload,
    Trajectory3DSample,
)
from dlp_schema.session import CameraIntrinsics
from dlp_schema.testing import FIXED_TIME

# 중력 가속도 (m/s²). 월드 좌표 중력 = (0, 0, -G)
G = 9.81
# 손 6점의 3D 궤적 부위 이름 (온톨로지 hand_joints, `prelabel.yaml depth.hand_points` 순서)
HAND_PARTS = ("wrist", "thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip")
# 손목 기준 손가락 끝 위치 (월드 좌표 m, 손등이 위인 오른손이 앞(+X)을 향함)
_TIP_OFFSETS = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.06, 0.05, -0.03],
        [0.10, 0.015, -0.035],
        [0.105, -0.01, -0.04],
        [0.095, -0.03, -0.035],
        [0.08, -0.045, -0.03],
    ]
)
# 첫 자세(고개를 숙이기 전) 카메라 → 월드 회전: 광축(Z_c) = 월드 +X, 오른쪽(X_c) = 월드 -Y,
# 아래(Y_c) = 월드 -Z
_R_LEVEL = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
# 장면 평면 (법선 n, 거리 d: n·X = d), 무늬 좌표에 쓰는 두 축 번호, 무늬 밝기 배율
_PLANES = (
    (np.array([0.0, 0.0, 1.0]), -0.7, (0, 1), 1.0),  # 탁자 윗면 z = -0.7
    (np.array([1.0, 0.0, 0.0]), 2.5, (1, 2), 0.8),  # 앞 벽 x = 2.5
)
# 무늬 이미지 크기(px)와 1 m당 무늬 픽셀 수
_TEXTURE_PX = 1024
_TEXTURE_SCALE = 300.0
# 정답 각속도·가속도 수치 미분 간격 (s)
_H = 1e-3


def _rx(a: float) -> NDArray[np.float64]:
    """x축 회전 행렬 (라디안)."""
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rz(a: float) -> NDArray[np.float64]:
    """z축 회전 행렬 (라디안)."""
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _log(r: NDArray[np.float64]) -> NDArray[np.float64]:
    """회전 행렬 → 회전 벡터 (축 x 각, 라디안)."""
    vec, _ = cv2.Rodrigues(r)
    return np.asarray(vec, dtype=np.float64).reshape(3)


def _ramp(t: float, static_s: float) -> float:
    """정지 구간 뒤 0.5초 동안 0 → 1로 부드럽게 오르는 움직임 크기 (smoothstep)."""
    x = min(max((t - static_s) / 0.5, 0.0), 1.0)
    return x * x * (3 - 2 * x)


@dataclass(frozen=True)
class MotionSpec:
    """머리 움직임 크기 (기본값: 걸으며 둘러보는 정도)."""

    sway_m: float = 0.2  # 앞뒤 흔들림 진폭 (m)
    walk_m: float = 0.3  # 끝까지 앞으로 걷는 거리 (m)
    side_m: float = 0.12  # 좌우 진폭 (m)
    bob_m: float = 0.04  # 위아래 진폭 (m)
    yaw_rad: float = 0.3  # 좌우 고개 돌림 진폭
    pitch_rad: float = 0.12  # 숙임 진폭 (기본 숙임 -0.6 rad 둘레)
    roll_rad: float = 0.08  # 기울임 진폭


@dataclass
class CameraMotionScenario:
    """정답 자세를 아는 머리 카메라 시퀀스 (모듈 설명 참고)."""

    intrinsics: CameraIntrinsics
    frame_times: list[int]
    duration_s: float
    static_s: float
    motion: MotionSpec
    imu_to_camera: tuple[tuple[float, ...], ...]
    imu_offset_ms: float
    imu_t_ms: NDArray[np.float64]  # IMU 시계 ms
    imu_acc: NDArray[np.float64]  # (N, 3) IMU 좌표 비력 m/s²
    imu_gyro: NDArray[np.float64]  # (N, 3) IMU 좌표 각속도 rad/s
    depth_bias: float = 1.0
    texture: NDArray[np.float32] = field(default_factory=lambda: np.zeros((1, 1), np.float32))
    width: int = 320
    height: int = 240
    session_id: str = "campose-000"

    # ---------------------------------------------------------------- 정답

    def pose_at(self, t_s: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """바디캠 시각 t_s(초)의 정답 (카메라 → 월드 회전 (3, 3), 카메라 중심 (3,))."""
        m, s = self.motion, _ramp(t_s, self.static_s)
        u = max(t_s - self.static_s, 0.0)
        span = max(self.duration_s - self.static_s, 1e-6)
        p = s * np.array(
            [
                m.sway_m * math.sin(2 * math.pi * u / 2.2) + m.walk_m * u / span,
                m.side_m * math.sin(2 * math.pi * u / 3.1),
                m.bob_m * math.sin(2 * math.pi * u / 0.9),
            ]
        )
        yaw = s * m.yaw_rad * math.sin(2 * math.pi * u / 2.7)
        pitch = -0.6 + s * m.pitch_rad * math.sin(2 * math.pi * u / 1.9)
        roll = s * m.roll_rad * math.sin(2 * math.pi * u / 1.3)
        # 월드 Z로 고개 돌림, 카메라 x로 숙임(음수 = 아래), 카메라 z로 기울임
        r = _rz(yaw) @ _R_LEVEL @ _rx(pitch) @ _rz(roll)
        return r, p

    def pose(self, t_ms: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """바디캠 PTS ms의 정답 자세."""
        return self.pose_at(t_ms / 1000)

    def hand_world_at(self, t_s: float) -> NDArray[np.float64]:
        """시각 t_s의 월드 좌표 손 6점 (6, 3): 탁자 위를 좌우·앞뒤로 닦는다."""
        wrist = np.array(
            [
                0.55 + 0.08 * math.sin(2 * math.pi * t_s / 0.9),
                -0.05 + 0.12 * math.sin(2 * math.pi * t_s / 1.5),
                -0.62 + 0.015 * math.sin(2 * math.pi * t_s / 0.7),
            ]
        )
        return wrist + _TIP_OFFSETS

    def hand_world(self, t_ms: int) -> NDArray[np.float64]:
        """바디캠 PTS ms의 월드 좌표 손 6점."""
        return self.hand_world_at(t_ms / 1000)

    def hand_cam(self, t_ms: int) -> NDArray[np.float64]:
        """바디캠 PTS ms의 카메라 좌표 손 6점 (정답 자세로 바꾼 것)."""
        r, p = self.pose(t_ms)
        return (r.T @ (self.hand_world(t_ms) - p).T).T

    # ---------------------------------------------------------------- 렌더·깊이

    def _cast(
        self, t_ms: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float32], NDArray[np.float32]]:
        """프레임마다 광선-평면 교차: (깊이 (H, W), 무늬 x (H, W), 무늬 y (H, W)) (무늬는 px)."""
        k = self.intrinsics
        r, p = self.pose(t_ms)
        vv, uu = np.mgrid[0 : self.height, 0 : self.width].astype(np.float64)
        ray_c = np.stack([(uu - k.cx) / k.fx, (vv - k.cy) / k.fy, np.ones_like(uu)], axis=-1)
        ray_w = ray_c @ r.T
        depth = np.full((self.height, self.width), np.inf)
        tx = np.zeros((self.height, self.width), np.float32)
        ty = np.zeros((self.height, self.width), np.float32)
        for i, (n, d, (a, b), _) in enumerate(_PLANES):
            denom = ray_w @ n
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (d - float(n @ p)) / denom
            hit = (t > 0.05) & (t < depth)
            point = p + ray_w * t[..., None]
            depth = np.where(hit, t, depth)
            # 평면마다 무늬 원점을 달리해 두 평면이 같은 무늬로 보이지 않게 한다
            tx = np.where(hit, point[..., a] * _TEXTURE_SCALE + 311 * i, tx).astype(np.float32)
            ty = np.where(hit, point[..., b] * _TEXTURE_SCALE + 173 * i, ty).astype(np.float32)
        return depth, tx, ty

    def render(self, t_ms: int) -> NDArray[np.uint8]:
        """그 시각의 RGB 프레임 (H, W, 3)."""
        depth, tx, ty = self._cast(t_ms)
        img = cv2.remap(
            self.texture,
            np.mod(tx, _TEXTURE_PX),
            np.mod(ty, _TEXTURE_PX),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP,
        )
        img = np.where(np.isfinite(depth), img, 128.0)
        gray = np.clip(img, 0, 255).astype(np.uint8)
        return np.repeat(gray[..., None], 3, axis=2)

    def frames(self) -> Iterator[tuple[int, NDArray[np.uint8]]]:
        """(바디캠 PTS ms, RGB 프레임)."""
        for t in self.frame_times:
            yield t, self.render(t)

    def depth_at(self, t_ms: int, image: NDArray[np.uint8]) -> NDArray[np.float32]:
        """그 프레임의 메트릭 깊이 (광축 방향 m, 평면이 없는 픽셀은 nan). image는 쓰지 않는다."""
        depth = self._cast(t_ms)[0]
        out = np.where(np.isfinite(depth), depth * self.depth_bias, np.nan)
        return out.astype(np.float32)

    # ---------------------------------------------------------------- 라벨

    def camera_labels(
        self,
        *,
        stride_ms: int = 100,
        noise_m: float = 0.0,
        seed: int = 0,
        stream_id: str = "bodycam",
        model_version: str = "oracle-depth3d",
    ) -> list[LabelRecord]:
        """정답 손 6점의 카메라 좌표 trajectory3d 라벨 (3D 단계 출력 흉내, 부위마다 하나).

        stride_ms 간격으로 고른 프레임 시각에 표본을 두고, noise_m 표준편차 흔들림을 넣는다. 표본
        방식은 hand_fit, 신뢰도 0.8.
        """
        rng = np.random.default_rng(seed)
        times: list[int] = []
        for t in self.frame_times:
            if not times or t - times[-1] >= stride_ms:
                times.append(t)
        cams = {t: self.hand_cam(t) for t in times}
        out: list[LabelRecord] = []
        for j, part in enumerate(HAND_PARTS):
            samples: list[Trajectory3DSample] = []
            for t in times:
                xyz = cams[t][j] + (rng.normal(0, noise_m, 3) if noise_m else 0.0)
                samples.append(
                    Trajectory3DSample(
                        t_ms=t,
                        x=round(float(xyz[0]), 4),
                        y=round(float(xyz[1]), 4),
                        z=round(float(xyz[2]), 4),
                        method=Lift3DMethod.HAND_FIT,
                        confidence=0.8,
                    )
                )
            out.append(
                LabelRecord(
                    label_id=f"{self.session_id}-{stream_id}-3d-oracle-right_hand-{part}",
                    session_id=self.session_id,
                    stream_id=stream_id,
                    t_start_ms=times[0],
                    t_end_ms=times[-1],
                    ontology_version="1.0.0",
                    provenance=Provenance(source=Source.MODEL, model_version=model_version),
                    confidence=0.8,
                    created_at=FIXED_TIME,
                    payload=Trajectory3DPayload(
                        entity_id="right_hand",
                        part=part,
                        frame=CoordinateFrame.CAMERA,
                        source_3d=Source3D.MONO_DEPTH,
                        samples=tuple(samples),
                    ),
                )
            )
        return out

    def imu_columns(self) -> dict[str, NDArray[np.float64]]:
        """IMU 사이드카·정규화 Parquet 열 (t_ms, ax..gz)."""
        cols = {"t_ms": self.imu_t_ms}
        for i, name in enumerate(("ax", "ay", "az")):
            cols[name] = self.imu_acc[:, i]
        for i, name in enumerate(("gx", "gy", "gz")):
            cols[name] = self.imu_gyro[:, i]
        return cols


# IMU → 카메라 회전 기본값 (축을 바꿔 단 IMU: IMU x = 카메라 z, IMU y = 카메라 x, IMU z = 카메라 y).
# 열 = 카메라 좌표로 본 IMU 축
DEFAULT_IMU_TO_CAMERA = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])


def _texture(rng: np.random.Generator) -> NDArray[np.float32]:
    """특징점이 많은 무늬 (여러 크기 흐린 잡음 + 사각형), 0~255."""
    base = np.zeros((_TEXTURE_PX, _TEXTURE_PX), np.float32)
    for sigma, weight in ((1.5, 0.5), (4.0, 0.8), (12.0, 1.0)):
        noise = rng.normal(0, 1, (_TEXTURE_PX, _TEXTURE_PX)).astype(np.float32)
        blurred = cv2.GaussianBlur(noise, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
        base += weight * blurred / (blurred.std() + 1e-6)
    for _ in range(120):
        x, y = (int(v) for v in rng.integers(0, _TEXTURE_PX - 40, 2))
        w, h = (int(v) for v in rng.integers(8, 40, 2))
        base[y : y + h, x : x + w] += float(rng.choice([-2.0, 2.0]))
    base = (base - base.mean()) / (base.std() + 1e-6)
    return np.clip(128 + 45 * base, 0, 255).astype(np.float32)


def _imu_signals(
    sc: CameraMotionScenario,
    r_ci: NDArray[np.float64],
    t_master_s: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """정답 자세의 수치 미분 → IMU 좌표 (비력, 각속도) (센서 오차 전)."""
    r_ic = r_ci.T
    acc = np.zeros((t_master_s.size, 3))
    gyro = np.zeros((t_master_s.size, 3))
    gravity_up = np.array([0.0, 0.0, G])
    for i, t in enumerate(t_master_s):
        r, p = sc.pose_at(float(t))
        r0, p0 = sc.pose_at(float(t) - _H)
        r1, p1 = sc.pose_at(float(t) + _H)
        omega_c = _log(r0.T @ r1) / (2 * _H)
        a_w = (p1 - 2 * p + p0) / _H**2
        f_c = r.T @ (a_w + gravity_up)
        acc[i] = r_ic @ f_c
        gyro[i] = r_ic @ omega_c
    return acc, gyro


def generate_camera_motion(
    seed: int = 0,
    *,
    duration_s: float = 4.0,
    static_s: float = 0.6,
    fps: float = 30.0,
    focal_px: float = 160.0,
    imu_rate_hz: float = 200.0,
    imu_offset_ms: float = 0.0,
    imu_to_camera: NDArray[np.float64] | None = None,
    gyro_bias: tuple[float, float, float] = (0.0, 0.0, 0.0),
    gyro_noise: float = 0.0,
    acc_noise: float = 0.0,
    depth_bias: float = 1.0,
    motion: MotionSpec | None = None,
) -> CameraMotionScenario:
    """정답 자세를 아는 머리 카메라 시퀀스. 같은 인자면 같은 결과다 (결정적).

    Args:
        seed: 무늬·센서 잡음 난수 시드.
        duration_s: 길이 (초). static_s: 처음 정지 구간 (초, 자이로 편향·중력 초기화).
        fps: 프레임률 (프레임 시각은 round(i * 1000 / fps) ms).
        focal_px: fx = fy (320x240, 주점 중앙).
        imu_rate_hz: IMU 표본률. imu_offset_ms: 바디캠 시각 = IMU 시각 + 이 값.
        imu_to_camera: IMU → 카메라 회전 (3, 3) (기본 `DEFAULT_IMU_TO_CAMERA`).
        gyro_bias, gyro_noise, acc_noise: 자이로 편향(rad/s), 자이로·가속도 흰잡음 표준편차.
        depth_bias: `depth_at` 배율 오차 (1.1이면 10% 멀게).
        motion: 움직임 크기 (기본 `MotionSpec()`).
    """
    rng = np.random.default_rng(seed)
    r_ci = DEFAULT_IMU_TO_CAMERA if imu_to_camera is None else imu_to_camera
    extr = np.eye(4)
    extr[:3, :3] = r_ci
    k = CameraIntrinsics(width=320, height=240, fx=focal_px, fy=focal_px, cx=160.0, cy=120.0)
    n = math.floor(duration_s * fps)
    frame_times = [round(i * 1000 / fps) for i in range(n)]
    sc = CameraMotionScenario(
        intrinsics=k,
        frame_times=frame_times,
        duration_s=duration_s,
        static_s=static_s,
        motion=motion or MotionSpec(),
        imu_to_camera=tuple(tuple(float(v) for v in row) for row in extr),
        imu_offset_ms=imu_offset_ms,
        imu_t_ms=np.zeros(0),
        imu_acc=np.zeros((0, 3)),
        imu_gyro=np.zeros((0, 3)),
        depth_bias=depth_bias,
        texture=_texture(rng),
        session_id=f"campose-{seed:03d}",
    )
    # IMU는 바디캠 영상 전체를 조금 넘게 덮는다 (앞뒤 0.2초)
    t_master_s = np.arange(-0.2, duration_s + 0.2, 1 / imu_rate_hz)
    acc, gyro = _imu_signals(sc, r_ci, t_master_s)
    acc = acc + (rng.normal(0, acc_noise, acc.shape) if acc_noise else 0.0)
    gyro = (
        gyro
        + np.asarray(gyro_bias)
        + (rng.normal(0, gyro_noise, gyro.shape) if gyro_noise else 0.0)
    )
    sc.imu_t_ms = t_master_s * 1000 - imu_offset_ms
    sc.imu_acc, sc.imu_gyro = acc, gyro
    return sc
