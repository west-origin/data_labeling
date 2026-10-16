"""정답 내부 파라미터를 아는 체커보드 캘리브레이션 영상 (ADR 0045, `dlp calib intrinsics` 시험).

알려진 핀홀 내부 파라미터(왜곡 없음)의 카메라 앞에서 평면 체커보드가 여러 자세(기울기·회전·거리)로
보이는 영상을 렌더한다. 픽셀마다 보드 평면으로 역투영(호모그래피 H = K [r1 r2 t]의 역)해 칸 색을
정하고, 2x2 초표본으로 경계를 부드럽게 한다. 보드 밖은 회색 배경, 보드 둘레에는 흰 여백 한 칸을 둔다
(OpenCV 코너 검출이 바깥 테두리를 찾기 쉽게).

정답과 출력 (`CheckerboardScenario`)
- `intrinsics`: 렌더에 쓴 내부 파라미터 (정답).
- `inner_corners`: 보드 안쪽 코너 수 (열, 행). `square_m`: 칸 한 변 (m).
- `frame_times`: 영상 PTS ms. 자세 하나를 `hold_frames` 프레임 동안 유지한다 (실제 촬영처럼).

테스트 사용처: `dlp_media.calibration`(정답 복원 오차), `dlp calib intrinsics` 명령.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from dlp_fixtures.video import write_video
from dlp_schema.session import CameraIntrinsics

# 30 fps 프레임 간격 (ms). 프레임 시각은 round(i * FRAME_MS)
FRAME_MS = 1000 / 30


def _rot(rx: float, ry: float, rz: float) -> NDArray[np.float64]:
    """x→y→z 순 회전 행렬 (라디안)."""
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    mx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    my = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    mz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return mz @ my @ mx


@dataclass
class CheckerboardScenario:
    """정답 내부 파라미터를 아는 체커보드 영상 (모듈 설명 참고)."""

    intrinsics: CameraIntrinsics
    inner_corners: tuple[int, int]
    square_m: float
    frame_times: list[int]
    n_poses: int


def render_board(
    k: CameraIntrinsics,
    rot: NDArray[np.float64],
    center: NDArray[np.float64],
    inner_corners: tuple[int, int],
    square_m: float,
) -> NDArray[np.uint8]:
    """보드 하나를 그 자세로 렌더한 RGB 프레임 (k.height, k.width, 3).

    보드 좌표: 안쪽 코너 격자의 중심이 원점, X 오른쪽(열), Y 아래(행), 평면 Z=0. 카메라 좌표
    P = rot @ [X, Y, 0] + center.
    """
    cols, rows = inner_corners
    kmat = np.array([[k.fx, 0, k.cx], [0, k.fy, k.cy], [0, 0, 1.0]])
    h = kmat @ np.column_stack([rot[:, 0], rot[:, 1], center])
    hinv = np.linalg.inv(h)
    ss = 2  # 초표본 배율
    ys, xs = np.mgrid[0 : k.height * ss, 0 : k.width * ss]
    pix = np.stack([(xs + 0.5) / ss - 0.5, (ys + 0.5) / ss - 0.5, np.ones_like(xs, float)], -1)
    board = pix @ hinv.T
    bx, by = board[..., 0] / board[..., 2], board[..., 1] / board[..., 2]
    # 안쪽 코너가 cols x rows이면 칸은 (cols+1) x (rows+1), 그 둘레에 흰 여백 한 칸
    gx, gy = bx / square_m + (cols + 1) / 2, by / square_m + (rows + 1) / 2
    inside = (gx >= 0) & (gx < cols + 1) & (gy >= 0) & (gy < rows + 1)
    margin = (gx >= -1) & (gx < cols + 2) & (gy >= -1) & (gy < rows + 2)
    dark = inside & ((np.floor(gx) + np.floor(gy)) % 2 == 0)
    value = np.where(margin, 235.0, 110.0)
    value = np.where(dark, 20.0, value)
    value = np.where(board[..., 2] > 0, value, 110.0)
    small = value.reshape(k.height, ss, k.width, ss).mean(axis=(1, 3))
    gray = np.clip(np.round(small), 0, 255).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


def _poses(
    n: int, rng: np.random.Generator
) -> list[tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """기울기(±35도)·회전·거리(0.45~0.75 m)·화면 위치가 다른 보드 자세 n개."""
    out: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []
    for i in range(n):
        a = 2 * np.pi * i / n
        rx = np.deg2rad(30) * np.sin(a) + rng.uniform(-0.05, 0.05)
        ry = np.deg2rad(30) * np.cos(a) + rng.uniform(-0.05, 0.05)
        rz = rng.uniform(-0.3, 0.3)
        z = rng.uniform(0.45, 0.75)
        center = np.array([rng.uniform(-0.06, 0.06), rng.uniform(-0.04, 0.04), z])
        out.append((_rot(rx, ry, rz), center))
    return out


def generate_checkerboard_video(
    path: Path,
    seed: int = 0,
    *,
    intrinsics: CameraIntrinsics | None = None,
    n_poses: int = 12,
    hold_frames: int = 9,
    inner_corners: tuple[int, int] = (8, 6),
    square_m: float = 0.025,
) -> CheckerboardScenario:
    """체커보드 영상을 path(MP4)에 쓰고 정답을 돌려준다. 같은 인자면 같은 영상이다 (결정적).

    Args:
        path: 출력 MP4.
        seed: 자세 난수 시드.
        intrinsics: 정답 내부 파라미터 (기본 640x480, fx 520, fy 515, 주점이 중앙에서 조금 벗어남).
        n_poses: 보드 자세 수. hold_frames: 자세마다 프레임 수.
        inner_corners: 안쪽 코너 (열, 행). square_m: 칸 한 변 (m).
    """
    k = intrinsics or CameraIntrinsics(
        width=640, height=480, fx=520.0, fy=515.0, cx=326.0, cy=236.0
    )
    rng = np.random.default_rng(seed)
    poses = _poses(n_poses, rng)
    times = [round(i * FRAME_MS) for i in range(n_poses * hold_frames)]
    images = [render_board(k, rot, c, inner_corners, square_m) for rot, c in poses]
    write_video(
        path,
        ((t, images[i // hold_frames]) for i, t in enumerate(times)),
        width=k.width,
        height=k.height,
    )
    return CheckerboardScenario(k, inner_corners, square_m, times, n_poses)
