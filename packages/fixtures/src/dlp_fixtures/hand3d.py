"""정답 3D를 아는 손 시퀀스: 카메라 좌표 21관절, 그 투영 2D, 손 중심 3D 점, 합성 깊이 (ADR 0044).

한 손(기본 오른손)이 카메라 앞 0.4~0.55 m에서 움직이며 천천히 돌고 손가락을 조금씩 굽힌다. 관절
위치는 성인 손 크기(손목 → 중지 MCP 0.09 m)의 손 모형으로 만들고, 알려진 핀홀 내부 파라미터로
투영해 2D 픽셀을 얻는다.

정답과 출력 (`Hand3DScenario`)
- `joints_cam[t_ms]`: 카메라 좌표 21관절 (21, 3) 미터 (정답, X 오른쪽, Y 아래, Z 앞).
- `labels`: 손 hand21 키포인트 트랙 하나 (출처 HUMAN, 바디캠 30 fps 프레임마다). 키프레임 points는
  정답 3D의 투영(+ 선택 흔들림), world는 손 모델 출력처럼 정답 3D에서 관절 평균을 뺀 손 중심 3D 점
  (+ 선택 크기 오차 `world_scale`, 흔들림 `world_noise_m`). 프리라벨 Oracle stub이 이 트랙을 모델
  라벨처럼 돌려준다 (MediaPipe hand_world_landmarks 흉내).
- `intrinsics`: 투영에 쓴 내부 파라미터. 기본값은 320x240, 가로 화각 90도 근사와 같다
  (fx = fy = 160, 주점 중앙) — 캘리브레이션 없는 세션의 기본 근사(`prelabel.yaml depth`)와 맞다.
  `focal_px`로 바꿀 수 있다. 손 크기도 `hand_scale`로 바꿀 수 있다 (작업자 손 크기 사전값 시험).
  `resolution_scale`은 영상 크기·초점 거리·주점을 함께 늘린다 (같은 장면의 1080p·4K판, 해상도와
  무관한 맞춤 문턱 시험, F-PAR-43). `world_rotation_deg`는 손 모델 3D 점의 자세를 정답과 그 각도만큼
  어긋나게 한다 (MediaPipe world가 카메라 축과 정확히 같지 않은 경우, F-PAR-42).
- `depth`: 합성 메트릭 깊이 모델 (`SceneDepth`). 배경 평면 + 손바닥 관절 둘레 원판만 관절 깊이를
  갖는다. 얇은 손가락 끝은 배경 깊이가 나오므로 관절별 깊이 방식은 손가락 끝에서 틀린다.
- `frames()`: (PTS ms, 작은 RGB 프레임). 프레임 번호를 첫 픽셀에 적어 `SceneDepth`가 그 시각의
  깊이를 고른다 (영상 파일로 쓰면 압축으로 번호가 깨지므로 메모리 순회에서만 쓴다).

테스트 사용처: `dlp_prelabel`(손가락 끝 3D 추정 오차, 맞춤 실패 시 깊이 방식), `dlp_review`(world를
가진 모델 트랙의 CVAT 왕복).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from dlp_schema.labels import (
    Hand,
    Keypoint,
    KeypointFrame,
    KeypointTrackPayload,
    LabelRecord,
    Provenance,
    Source,
    WorldPoint,
)
from dlp_schema.session import CameraIntrinsics
from dlp_schema.testing import FIXED_TIME

# 바디캠 프레임 간격 ms (30 fps). 프레임 시각은 round(i * FRAME_MS)
FRAME_MS = 1000 / 30
# 깊이 원판을 그리는 손바닥 관절 (손목, 엄지 CMC, 네 손가락 MCP)
PALM_JOINTS = (0, 1, 5, 9, 13, 17)
# 손가락 (뿌리 관절 번호, 뿌리 위치(손 좌표 m), 마디 길이 3개(m), 펼침 방향(도, -y에서 x쪽))
_FINGERS: tuple[tuple[int, tuple[float, float, float], tuple[float, float, float], float], ...] = (
    (5, (-0.022, -0.085, 0.0), (0.040, 0.025, 0.020), -8.0),
    (9, (0.0, -0.090, 0.0), (0.045, 0.028, 0.022), 0.0),
    (13, (0.020, -0.085, 0.0), (0.042, 0.026, 0.020), 8.0),
    (17, (0.038, -0.075, 0.0), (0.032, 0.020, 0.018), 16.0),
)
# 엄지: CMC(1) 위치와 이후 마디 길이 3개, 방향
_THUMB_ROOT = (-0.025, -0.025, -0.005)
_THUMB_LENGTHS = (0.040, 0.032, 0.026)
_THUMB_DIR_DEG = -50.0


def _rot(rx: float, ry: float, rz: float) -> NDArray[np.float64]:
    """x→y→z 순 회전 행렬 (라디안)."""
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    mx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    my = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    mz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return mz @ my @ mx


def _axis_angle(axis: NDArray[np.float64], angle: float) -> NDArray[np.float64]:
    """단위 축·각(라디안) → 회전 행렬 (로드리게스 식)."""
    kx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * kx + (1 - np.cos(angle)) * (kx @ kx)


def _chain(
    root: tuple[float, float, float], lengths: tuple[float, ...], spread_deg: float, curl: float
) -> list[NDArray[np.float64]]:
    """뿌리에서 마디마다 curl(라디안)씩 카메라 반대쪽(+z)으로 굽는 손가락 관절 3개 위치."""
    a = np.deg2rad(spread_deg)
    pos = np.array(root, dtype=np.float64)
    out: list[NDArray[np.float64]] = []
    bend = 0.0
    for length in lengths:
        bend += curl
        # 손가락 방향: 손바닥 평면에서 -y(위)로 spread만큼 벌리고, 굽힌 만큼 +z로 숙인다
        d = np.array([np.sin(a) * np.cos(bend), -np.cos(a) * np.cos(bend), np.sin(bend)])
        pos = pos + length * d
        out.append(pos)
    return out


def hand_model(curl: float) -> NDArray[np.float64]:
    """손 좌표(손목 원점, 카메라와 같은 축) 21관절 (21, 3) 미터. MediaPipe 관절 순서.

    curl: 손가락 마디마다 굽힘(라디안). 손목 → 중지 MCP는 0.09 m.
    """
    joints = np.zeros((21, 3))
    joints[1] = _THUMB_ROOT
    joints[2:5] = _chain(_THUMB_ROOT, _THUMB_LENGTHS, _THUMB_DIR_DEG, curl / 2)
    for root_index, root, lengths, spread in _FINGERS:
        joints[root_index] = root
        joints[root_index + 1 : root_index + 4] = _chain(root, lengths, spread, curl)
    return joints


@dataclass
class SceneDepth:
    """합성 메트릭 깊이: 배경 평면 + 손바닥 관절 원판 (`Hand3DScenario.depth`).

    `predict`는 프레임 첫 픽셀에 적힌 프레임 번호로 그 시각 정답 관절을 고른다. bias는 깊이 모델의
    배율 오차 (1.1이면 10% 멀게).
    """

    joints: list[NDArray[np.float64]]
    pixels: list[NDArray[np.float64]]
    background_m: float = 1.2
    disc_px: int = 5
    bias: float = 1.0
    calls: int = 0

    def predict(self, image: NDArray[np.uint8]) -> NDArray[np.float32]:
        """(H, W, 3) 프레임 → (H, W) 깊이 맵 (미터)."""
        self.calls += 1
        i = int(image[0, 0, 0]) + 256 * int(image[0, 0, 1])
        h, w = image.shape[:2]
        d = np.full((h, w), self.background_m * self.bias, dtype=np.float32)
        yy, xx = np.mgrid[0:h, 0:w]
        for j in PALM_JOINTS:
            u, v = self.pixels[i][j]
            disc = (xx - u) ** 2 + (yy - v) ** 2 <= self.disc_px**2
            d[disc] = self.joints[i][j, 2] * self.bias
        return d


@dataclass
class Hand3DScenario:
    """정답 3D를 아는 손 시퀀스 (모듈 설명 참고)."""

    session_id: str
    hand: Hand
    intrinsics: CameraIntrinsics
    frame_times: list[int]
    joints_cam: dict[int, NDArray[np.float64]]
    labels: list[LabelRecord]
    depth: SceneDepth
    width: int = 320
    height: int = 240

    def frames(self) -> Iterator[tuple[int, NDArray[np.uint8]]]:
        """(PTS ms, RGB 프레임). 첫 픽셀에 프레임 번호를 적는다 (`SceneDepth.predict`)."""
        for i, t in enumerate(self.frame_times):
            img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            img[0, 0, 0], img[0, 0, 1] = i % 256, i // 256
            yield t, img


def _project(p: NDArray[np.float64], k: CameraIntrinsics) -> NDArray[np.float64]:
    """카메라 좌표 (n, 3) → 픽셀 (n, 2)."""
    return np.stack([k.fx * p[:, 0] / p[:, 2] + k.cx, k.fy * p[:, 1] / p[:, 2] + k.cy], axis=1)


def _pose(s: float) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """진행률 s(0~1) → (회전, 손목 위치 m, 굽힘 rad). 천천히 돌며 다가왔다 멀어진다."""
    wave = np.sin(2 * np.pi * s)
    rot = _rot(0.35 * wave, 0.5 * np.sin(np.pi * s) - 0.25, 0.3 * np.cos(2 * np.pi * s))
    wrist = np.array([-0.04 + 0.08 * s, 0.07 + 0.02 * wave, 0.40 + 0.15 * s])
    return rot, wrist, 0.15 + 0.25 * (0.5 + 0.5 * wave)


def generate_hand3d_scenario(
    seed: int = 0,
    *,
    n_frames: int = 30,
    hand: Hand = Hand.RIGHT,
    pixel_noise_px: float = 0.0,
    world_noise_m: float = 0.0,
    world_scale: float = 1.0,
    depth_bias: float = 1.0,
    hand_scale: float = 1.0,
    focal_px: float = 160.0,
    resolution_scale: float = 1.0,
    world_rotation_deg: float = 0.0,
) -> Hand3DScenario:
    """정답 3D를 아는 손 시퀀스. 같은 인자면 같은 결과다 (결정적).

    Args:
        seed: 흔들림 난수 시드 (세션 ID에도 들어간다).
        n_frames: 프레임 수 (30 fps).
        hand: 손 (왼손이면 모형을 x로 뒤집는다).
        pixel_noise_px: 2D 점 흔들림 표준편차(px).
        world_noise_m: 손 중심 3D 점 흔들림 표준편차(m).
        world_scale: 손 중심 3D 점 크기 배율 (손 모델이 실제 손 크기를 틀리는 정도, 1.1이면 10% 큼).
        depth_bias: 합성 깊이 배율 오차 (`SceneDepth.bias`).
        hand_scale: 정답 손 크기 배율 (손목 → 중지 MCP = 0.09 m x hand_scale, ADR 0045 작업자 손
            크기). 손목 위치는 그대로다.
        focal_px: 투영 초점 거리 fx = fy (px). 기본 160은 가로 화각 90도 근사와 같다. 다르면
            캘리브레이션 없이 화각 근사로 올린 3D가 틀린다 (ADR 0045 캘리브레이션 시험).
        resolution_scale: 영상 크기 배율 (320x240 x 배율, 초점 거리·주점·깊이 원판 반지름도 같은
            배율). 6이면 1920x1440. 장면(3D)은 그대로다.
        world_rotation_deg: 손 중심 3D 점을 손 중심 기준으로 이 각도(도)만큼 돌린다 (축은 시드로
            정하는 무작위 단위 벡터, 프레임마다 같다). 정답 3D·2D는 그대로다.
    """
    rng = np.random.default_rng(seed)
    w, h = round(320 * resolution_scale), round(240 * resolution_scale)
    f = focal_px * resolution_scale
    k = CameraIntrinsics(width=w, height=h, fx=f, fy=f, cx=w / 2, cy=h / 2)
    axis = rng.normal(size=3)
    tilt = _axis_angle(axis / np.linalg.norm(axis), np.deg2rad(world_rotation_deg))
    mirror = np.array([-1.0, 1.0, 1.0]) if hand is Hand.LEFT else np.ones(3)
    frame_times = [round(i * FRAME_MS) for i in range(n_frames)]
    joints: list[NDArray[np.float64]] = []
    pixels: list[NDArray[np.float64]] = []
    frames: list[KeypointFrame] = []
    for i, t in enumerate(frame_times):
        rot, wrist, curl = _pose(i / max(n_frames - 1, 1))
        cam = (rot @ (hand_model(curl) * mirror * hand_scale).T).T + wrist
        px = _project(cam, k)
        joints.append(cam)
        pixels.append(px)
        seen = px + rng.normal(0, pixel_noise_px, px.shape) if pixel_noise_px else px
        world = (cam - cam.mean(axis=0)) @ tilt.T * world_scale
        if world_noise_m:
            world = world + rng.normal(0, world_noise_m, world.shape)
        frames.append(
            KeypointFrame(
                t_ms=t,
                points=tuple(Keypoint(x=float(x), y=float(y), visibility=2) for x, y in seen),
                world=tuple(WorldPoint(x=float(x), y=float(y), z=float(z)) for x, y, z in world),
            )
        )
    session_id = f"hand3d-{seed:03d}"
    track = LabelRecord(
        label_id=f"{session_id}-kp-{hand.value}",
        session_id=session_id,
        stream_id="bodycam",
        t_start_ms=frame_times[0],
        t_end_ms=frame_times[-1],
        ontology_version="1.0.0",
        provenance=Provenance(source=Source.HUMAN),
        created_at=FIXED_TIME,
        payload=KeypointTrackPayload(
            entity_id=f"{hand.value}_hand", skeleton="hand21", hand=hand, keyframes=tuple(frames)
        ),
    )
    return Hand3DScenario(
        session_id=session_id,
        hand=hand,
        intrinsics=k,
        frame_times=frame_times,
        joints_cam=dict(zip(frame_times, joints, strict=True)),
        labels=[track],
        depth=SceneDepth(
            joints=joints,
            pixels=pixels,
            bias=depth_bias,
            disc_px=max(5, round(5 * resolution_scale)),
        ),
        width=w,
        height=h,
    )
