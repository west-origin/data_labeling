"""카메라 내부 파라미터 캘리브레이션 단위 테스트 (ADR 0045, `dlp_media.calibration`).

정답은 `dlp_fixtures.calibration`이 체커보드를 렌더할 때 쓴 내부 파라미터(왜곡 없음)다.

허용 오차 (근거): 640x480 렌더, 보드 12자세, 초표본 2x2라 코너 위치 오차는 0.1 px 안팎이다. 이
정도 잡음과 자세 수에서 초점 거리는 0.5% 안팎, 주점은 수 px 안으로 복원된다 → fx·fy 1%, cx·cy 4 px,
재투영 RMS 0.3 px, 왜곡 계수 절댓값 0.05.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from dlp_fixtures.calibration import generate_checkerboard_video
from dlp_media.calibration import (
    BoardSpec,
    CalibrationError,
    calibrate_video,
    load_intrinsics_file,
    load_policy,
)
from dlp_schema.session import CameraIntrinsics

ROOT = Path(__file__).resolve().parents[3]


def test_checkerboard_video_recovers_known_intrinsics(tmp_path: Path) -> None:
    """렌더한 체커보드 영상에서 정답 내부 파라미터를 허용 오차 안으로 되찾는다.

    같은 영상은 같은 결과다 (결정적).
    """
    s = generate_checkerboard_video(tmp_path / "board.mp4", 0)
    board = BoardSpec(*s.inner_corners, s.square_m)
    policy = load_policy(ROOT)
    got = calibrate_video(tmp_path / "board.mp4", board, policy)
    k, truth = got.intrinsics, s.intrinsics
    assert (k.width, k.height) == (truth.width, truth.height)
    assert k.fx == pytest.approx(truth.fx, rel=0.01) and k.fy == pytest.approx(truth.fy, rel=0.01)
    assert abs(k.cx - truth.cx) < 4 and abs(k.cy - truth.cy) < 4
    assert got.rms_px < 0.3 and got.n_views >= policy.min_views
    assert max(abs(d) for d in k.distortion) < 0.05 and k.distortion[4] == 0.0  # k3 고정
    assert calibrate_video(tmp_path / "board.mp4", board, policy) == got


def test_too_few_board_views_or_wrong_board_fail(tmp_path: Path) -> None:
    """자세가 min_views보다 적거나 안쪽 코너 수를 틀리게 주면 등록할 결과를 내지 않는다."""
    policy = load_policy(ROOT)
    s = generate_checkerboard_video(tmp_path / "few.mp4", 1, n_poses=4)
    with pytest.raises(CalibrationError, match="최소"):
        calibrate_video(tmp_path / "few.mp4", BoardSpec(*s.inner_corners, s.square_m), policy)
    s = generate_checkerboard_video(tmp_path / "b.mp4", 2)
    with pytest.raises(CalibrationError):
        calibrate_video(tmp_path / "b.mp4", BoardSpec(7, 7, s.square_m), policy)


def test_known_calibration_files_are_imported(tmp_path: Path) -> None:
    """이 저장소 형식(YAML), OpenCV FileStorage(YAML), ROS camera_info 형식이 같은 값을 준다.
    모르는 형식은 CalibrationError.
    """
    truth = CameraIntrinsics(
        width=1920, height=1080, fx=1400.5, fy=1398.25, cx=961.0, cy=541.5,
        distortion=(-0.1, 0.02, 0.0, 0.0, 0.0),
    )  # fmt: skip
    own = tmp_path / "cam.yaml"
    own.write_text(yaml.safe_dump(truth.model_dump(mode="json")), "utf-8")
    assert load_intrinsics_file(own) == truth

    fs_path = tmp_path / "opencv.yaml"
    fs = cv2.FileStorage(str(fs_path), cv2.FILE_STORAGE_WRITE)
    kmat = np.array([[truth.fx, 0, truth.cx], [0, truth.fy, truth.cy], [0, 0, 1.0]])
    fs.write("camera_matrix", kmat)
    fs.write("distortion_coefficients", np.array([truth.distortion]))
    fs.write("image_width", truth.width)
    fs.write("image_height", truth.height)
    fs.release()
    assert load_intrinsics_file(fs_path) == truth

    ros = tmp_path / "camera_info.yaml"
    ros.write_text(yaml.safe_dump({
        "image_width": truth.width, "image_height": truth.height,
        "camera_matrix": {"rows": 3, "cols": 3, "data": kmat.reshape(-1).tolist()},
        "distortion_coefficients": {"rows": 1, "cols": 5, "data": list(truth.distortion)},
    }), "utf-8")  # fmt: skip
    assert load_intrinsics_file(ros) == truth

    bad = tmp_path / "bad.yaml"
    bad.write_text("focal: 3\n", "utf-8")
    with pytest.raises(CalibrationError):
        load_intrinsics_file(bad)
