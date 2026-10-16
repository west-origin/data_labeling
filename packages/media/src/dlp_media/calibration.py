"""카메라 내부 파라미터 캘리브레이션: 체커보드 영상 → OpenCV calibrateCamera, 또는 파일 가져오기
(ADR 0045, `dlp calib intrinsics|import`).

입력은 운영자가 그 카메라로 체커보드만 찍은 로컬 영상(또는 이미 아는 캘리브레이션 파일)이다. 세션
영상(원본 버킷)을 읽지 않으므로 원본 접근이 아니다 (보드 외의 사람·장소를 찍지 않는다, 절차 문서
`docs/handover/camera-calibration.md`).

- `calibrate_video`: PTS 간격(`sample_every_ms`)마다 프레임을 검사해 보드 코너를 찾고
  (`cv2.findChessboardCornersSB`), 앞서 채택한 장면과 거의 같은 장면은 건너뛴 뒤
  `cv2.calibrateCamera`로 푼다. 장면 수·재투영 RMS가 정책(`config/policies/calibration.yaml`)을
  못 맞추면 `CalibrationError`.
- `load_intrinsics_file`: 이 저장소 형식(YAML/JSON, `CameraIntrinsics` 필드), OpenCV FileStorage
  (`%YAML:1.0` 또는 XML: `camera_matrix`, `distortion_coefficients`, `image_width`, `image_height`),
  ROS camera_info YAML(같은 키, 행렬은 `data` 목록)을 읽는다.

한계: 핀홀 + OpenCV 방사·접선 왜곡 모델만 다룬다 (어안 렌즈는 `cv2.fisheye`가 필요하다). 3D 단계는
아직 왜곡 계수를 쓰지 않는다 (fx·fy·cx·cy만, `dlp_prelabel.lift3d.intrinsics_for`).
"""

# PyAV·OpenCV 타입 스텁이 일부 반환 타입을 비워 두어 이 모듈에서만 해당 경고를 끈다.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np
import yaml
from numpy.typing import NDArray
from pydantic import Field

from dlp_schema.common import Contract
from dlp_schema.session import CameraIntrinsics


class CalibrationError(RuntimeError):
    """캘리브레이션을 받아들일 수 없다 (보드 장면 부족, 재투영 오차 초과, 파일 형식)."""


class IntrinsicsCalibrationPolicy(Contract):
    """`calibration.yaml intrinsics` 절 (값의 뜻은 YAML 주석)."""

    sample_every_ms: int = Field(ge=1)
    min_views: int = Field(ge=3)
    max_views: int = Field(ge=3)
    min_motion_px: float = Field(ge=0)
    max_rms_px: float = Field(gt=0)
    fix_k3: bool
    zero_tangent: bool


class CalibrationPolicy(Contract):
    version: int
    intrinsics: IntrinsicsCalibrationPolicy


def load_policy(root: Path) -> IntrinsicsCalibrationPolicy:
    """`<root>/config/policies/calibration.yaml`의 intrinsics 절."""
    data: Any = yaml.safe_load((root / "config/policies/calibration.yaml").read_text("utf-8"))
    return CalibrationPolicy.model_validate(data).intrinsics


@dataclass(frozen=True)
class BoardSpec:
    """체커보드: 안쪽 코너 (열, 행)과 칸 한 변 (m)."""

    cols: int
    rows: int
    square_m: float

    def object_points(self) -> NDArray[np.float32]:
        """보드 좌표 코너 (cols*rows, 3), OpenCV 검출 순서 (행마다 왼쪽 → 오른쪽)."""
        grid = np.mgrid[0 : self.cols, 0 : self.rows].T.reshape(-1, 2)
        pts = np.zeros((self.cols * self.rows, 3), np.float32)
        pts[:, :2] = grid * self.square_m
        return pts


@dataclass(frozen=True)
class CalibrationResult:
    """캘리브레이션 결과: 내부 파라미터, 재투영 RMS(px), 쓴 장면 수와 그 PTS ms."""

    intrinsics: CameraIntrinsics
    rms_px: float
    n_views: int
    view_times_ms: tuple[int, ...]


def _gray_frames(path: Path, every_ms: int) -> Iterator[tuple[int, NDArray[np.uint8]]]:
    """(PTS ms, 회색조 프레임). 마지막으로 낸 프레임에서 every_ms가 지난 프레임만 낸다."""
    last: int | None = None
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            t = round(float(frame.pts * frame.time_base) * 1000)
            if last is not None and t - last < every_ms:
                continue
            last = t
            yield t, np.asarray(frame.to_ndarray(format="gray"), np.uint8)


def find_views(
    path: Path, board: BoardSpec, policy: IntrinsicsCalibrationPolicy
) -> tuple[list[tuple[int, NDArray[np.float32]]], tuple[int, int]]:
    """영상에서 보드 코너를 찾은 장면 (PTS ms, 코너 (n, 1, 2))과 프레임 크기 (가로, 세로).

    앞서 채택한 장면과 코너 평균 이동이 `min_motion_px`보다 작으면 건너뛴다 (같은 자세를 여러 번
    넣으면 그 자세에 치우친다).
    """
    views: list[tuple[int, NDArray[np.float32]]] = []
    size = (0, 0)
    for t, gray in _gray_frames(path, policy.sample_every_ms):
        size = (gray.shape[1], gray.shape[0])
        found, corners = cv2.findChessboardCornersSB(gray, (board.cols, board.rows))
        if not found:
            continue
        pts = np.asarray(corners, np.float32).reshape(-1, 1, 2)
        if views:
            moved = float(np.mean(np.linalg.norm(pts - views[-1][1], axis=2)))
            if moved < policy.min_motion_px:
                continue
        views.append((t, pts))
    return views, size


def _spread(items: list[Any], k: int) -> list[Any]:
    """목록에서 고르게 k개 (k 이상이면 그대로)."""
    if len(items) <= k:
        return items
    idx = np.linspace(0, len(items) - 1, k).round().astype(int)
    return [items[i] for i in sorted(set(idx.tolist()))]


def calibrate_video(
    path: Path, board: BoardSpec, policy: IntrinsicsCalibrationPolicy
) -> CalibrationResult:
    """체커보드 영상으로 내부 파라미터를 구한다 (같은 영상·정책이면 같은 결과, 결정적).

    Raises:
        CalibrationError: 보드 장면이 `min_views`보다 적거나 재투영 RMS가 `max_rms_px`를 넘을 때.
    """
    views, (width, height) = find_views(path, board, policy)
    views = _spread(views, policy.max_views)
    if len(views) < policy.min_views:
        raise CalibrationError(
            f"보드를 찾은 장면 {len(views)}개 < 최소 {policy.min_views}개 "
            f"(보드 {board.cols}x{board.rows} 안쪽 코너가 맞는지, 여러 각도로 찍었는지 확인)"
        )
    obj = board.object_points()
    flags = (cv2.CALIB_FIX_K3 if policy.fix_k3 else 0) | (
        cv2.CALIB_ZERO_TANGENT_DIST if policy.zero_tangent else 0
    )
    rms, kmat, dist, _, _ = cv2.calibrateCamera(
        [obj] * len(views), [v for _, v in views], (width, height), None, None, flags=flags
    )
    if rms > policy.max_rms_px:
        raise CalibrationError(f"재투영 RMS {rms:.3f} px > 상한 {policy.max_rms_px} px")
    k = np.asarray(kmat, np.float64)
    intrinsics = CameraIntrinsics(
        width=width,
        height=height,
        fx=round(float(k[0, 0]), 3),
        fy=round(float(k[1, 1]), 3),
        cx=round(float(k[0, 2]), 3),
        cy=round(float(k[1, 2]), 3),
        distortion_model="opencv",
        distortion=tuple(round(float(d), 6) for d in np.asarray(dist).reshape(-1)),
    )
    return CalibrationResult(
        intrinsics, round(float(rms), 4), len(views), tuple(t for t, _ in views)
    )


def _from_opencv_storage(path: Path) -> CameraIntrinsics:
    """OpenCV FileStorage(YAML/XML)의 내부 파라미터.

    키: camera_matrix, distortion_coefficients, image_width, image_height.
    """
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    try:
        k_node, d_node = fs.getNode("camera_matrix"), fs.getNode("distortion_coefficients")
        kmat = None if k_node.empty() else k_node.mat()
        dist = None if d_node.empty() else d_node.mat()
        width = int(fs.getNode("image_width").real())
        height = int(fs.getNode("image_height").real())
    finally:
        fs.release()
    if kmat is None or width <= 0 or height <= 0:
        raise CalibrationError(
            f"{path}: camera_matrix·image_width·image_height가 없습니다 "
            "(이 저장소 형식이면 width·height·fx·fy·cx·cy 키를 씁니다)"
        )
    k = np.asarray(kmat, np.float64)
    coeffs = () if dist is None else tuple(float(d) for d in np.asarray(dist).reshape(-1))
    return CameraIntrinsics(
        width=width,
        height=height,
        fx=float(k[0, 0]),
        fy=float(k[1, 1]),
        cx=float(k[0, 2]),
        cy=float(k[1, 2]),
        distortion=coeffs,
    )


def _from_camera_info(data: dict[str, Any], path: Path) -> CameraIntrinsics:
    """ROS camera_info 형식 YAML (camera_matrix.data 9개, distortion_coefficients.data)."""
    try:
        k = [float(v) for v in data["camera_matrix"]["data"]]
        dist = [float(v) for v in (data.get("distortion_coefficients") or {}).get("data", [])]
        width, height = int(data["image_width"]), int(data["image_height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CalibrationError(f"{path}: camera_info 형식이 아닙니다 ({exc})") from exc
    if len(k) != 9:
        raise CalibrationError(f"{path}: camera_matrix.data는 9개여야 합니다")
    return CameraIntrinsics(
        width=width, height=height, fx=k[0], fy=k[4], cx=k[2], cy=k[5], distortion=tuple(dist)
    )


def load_intrinsics_file(path: Path) -> CameraIntrinsics:
    """캘리브레이션 파일을 읽는다 (모듈 설명의 두 형식).

    Raises:
        CalibrationError: 어느 형식도 아닐 때. pydantic.ValidationError: 값이 계약에 맞지 않을 때.
    """
    text = path.read_text("utf-8")
    if text.lstrip().startswith("%YAML") or path.suffix.lower() == ".xml":
        return _from_opencv_storage(path)
    data: Any = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    if isinstance(data, dict) and "fx" in data:
        return CameraIntrinsics.model_validate(data)
    if isinstance(data, dict) and "camera_matrix" in data:
        return _from_camera_info(data, path)
    raise CalibrationError(f"{path}: 알 수 없는 캘리브레이션 파일 형식")
