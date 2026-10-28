"""카메라 내부 파라미터 캘리브레이션 단위 테스트 (ADR 0045, `dlp_media.calibration`).

정답은 `dlp_fixtures.calibration`이 체커보드를 렌더할 때 쓴 내부 파라미터(왜곡 없음)다.

허용 오차 (근거): 640x480 렌더, 보드 12자세, 초표본 2x2라 코너 위치 오차는 0.1 px 안팎이다. 이
정도 잡음과 자세 수에서 초점 거리는 0.5% 안팎, 주점은 수 px 안으로 복원된다 → fx·fy 1%, cx·cy 4 px,
재투영 RMS 0.3 px, 왜곡 계수 절댓값 0.05.
"""

from __future__ import annotations

import shutil
import subprocess
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
    check_intrinsics,
    load_intrinsics_file,
    load_policy,
)
from dlp_media.prepare import load_policy as load_prepare_policy
from dlp_media.prepare import prepare_video
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


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg 필요")
def test_display_rotation_tagged_board_video_is_rejected_until_prepared(tmp_path: Path) -> None:
    """표시 회전 태그가 있는 보드 영상은 캘리브레이션하지 않는다 (F-PMS-41).

    수집·prepare는 회전 영상을 거부하거나 똑바른 픽셀로 바꾸는데, 캘리브레이션만 저장 픽셀로 풀어
    세션 영상과 축이 다른(90°면 가로·세로가 바뀐) 내부 파라미터를 등록했다. 정답: 태그 영상은
    CalibrationError("dlp media prepare" 안내), prepare한 사본은 표시 방향 크기(480x640)로 풀린다.
    """
    s = generate_checkerboard_video(tmp_path / "board.mp4", 0)
    board = BoardSpec(*s.inner_corners, s.square_m)
    policy = load_policy(ROOT)
    tagged = tmp_path / "tagged.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-display_rotation", "90", "-i", str(tmp_path / "board.mp4"),
         "-c", "copy", str(tagged)],
        check=True,
    )  # fmt: skip
    with pytest.raises(CalibrationError, match="dlp media prepare"):
        calibrate_video(tagged, board, policy)
    ready = tmp_path / "ready.mp4"
    prepare_video(tagged, ready, load_prepare_policy(ROOT))
    k = calibrate_video(ready, board, policy).intrinsics
    assert (k.width, k.height) == (s.intrinsics.height, s.intrinsics.width)


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
        "distortion_model": "plumb_bob",
        "distortion_coefficients": {"rows": 1, "cols": 5, "data": list(truth.distortion)},
    }), "utf-8")  # fmt: skip
    assert load_intrinsics_file(ros) == truth

    bad = tmp_path / "bad.yaml"
    bad.write_text("focal: 3\n", "utf-8")
    with pytest.raises(CalibrationError):
        load_intrinsics_file(bad)


@pytest.mark.parametrize(("tilt_deg", "seed"), [(0, 0), (0, 1), (3, 0), (3, 2)])
def test_frontal_only_board_video_is_rejected_as_degenerate(
    tmp_path: Path, tilt_deg: float, seed: int
) -> None:
    """보드를 기울이지 않은(정면만) 영상은 RMS가 작아도 등록할 결과를 내지 않는다 (F-PMS-20).

    정답: 기울기 진폭 0도면 초점 거리가 관측되지 않아 정답 fx 520에 대해 10배 넘게 틀린 해가 나오고,
    3도면 몇 % 틀린다 (재현). 둘 다 기울기 부족(또는 화각 범위 밖)으로 거부해야 한다.
    """
    s = generate_checkerboard_video(tmp_path / "flat.mp4", seed, tilt_deg=tilt_deg)
    board = BoardSpec(*s.inner_corners, s.square_m)
    with pytest.raises(CalibrationError, match=r"기울|화각"):
        calibrate_video(tmp_path / "flat.mp4", board, load_policy(ROOT))


def test_tilted_board_video_passes_and_reports_max_tilt(tmp_path: Path) -> None:
    """기울기 진폭 10도 영상은 통과하고 정답 fx와 1% 안이다 (기울기 하한의 반대쪽 경계)."""
    s = generate_checkerboard_video(tmp_path / "t.mp4", 1, tilt_deg=10)
    policy = load_policy(ROOT)
    got = calibrate_video(tmp_path / "t.mp4", BoardSpec(*s.inner_corners, s.square_m), policy)
    assert got.intrinsics.fx == pytest.approx(s.intrinsics.fx, rel=0.01)
    assert got.max_tilt_deg >= policy.min_max_view_tilt_deg


def test_board_square_size_does_not_change_intrinsics(tmp_path: Path) -> None:
    """칸 크기를 반이나 두 배로 틀리게 주어도 내부 파라미터·RMS는 같다 (F-PMS-26, 문서 주장 고정).

    정답: 보드 좌표를 같은 비율로 늘이면 외부 파라미터의 이동량만 바뀐다 (핀홀 투영의 축척 불변).
    """
    s = generate_checkerboard_video(tmp_path / "b.mp4", 0)
    policy = load_policy(ROOT)
    results = [
        calibrate_video(tmp_path / "b.mp4", BoardSpec(*s.inner_corners, s.square_m * f), policy)
        for f in (1.0, 0.5, 2.0)
    ]
    for r in results[1:]:
        for name in ("fx", "fy", "cx", "cy"):
            want = getattr(results[0].intrinsics, name)
            assert getattr(r.intrinsics, name) == pytest.approx(want, abs=1e-3)
        assert r.rms_px == pytest.approx(results[0].rms_px, abs=1e-3)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"fx": 5000.0, "fy": 5000.0}, "화각"),
        ({"cx": 600.0}, "중앙"),
        ({"fy": 400.0}, "fx/fy"),
        ({"distortion": (0.1, -488.0, 0.0, 0.0, 0.0)}, "왜곡 계수 절댓값"),
        ({"fx": -520.0}, "양수"),
        ({"cx": 700.0}, "밖"),
        ({"fx": float("nan")}, "유한"),
        ({"distortion": (0.1, 0.2, 0.3)}, "길이"),
        ({"distortion_model": "equidistant"}, "지원하지"),
    ],
)
def test_implausible_intrinsics_are_rejected(change: dict[str, object], match: str) -> None:
    """타당 범위 밖·틀린 값은 CalibrationError (F-PMS-20 보조 기준, F-PMS-21)."""
    base = CameraIntrinsics(width=640, height=480, fx=520.0, fy=515.0, cx=326.0, cy=236.0)
    check_intrinsics(base, load_policy(ROOT), "ok")
    with pytest.raises(CalibrationError, match=match):
        check_intrinsics(base.model_copy(update=change), load_policy(ROOT), "bad")


_ROS_K = [500.0, 0, 320.0, 0, 500.0, 240.0, 0, 0, 1.0]
_OWN = "width: 640\nheight: 480\nfx: 500\nfy: 500\ncx: 320\ncy: 240\n"


def _ros(model: str | None, k: list[float], dist: list[float]) -> str:
    data: dict[str, object] = {
        "image_width": 640, "image_height": 480,
        "camera_matrix": {"rows": 3, "cols": 3, "data": k},
        "distortion_coefficients": {"rows": 1, "cols": len(dist), "data": dist},
    }  # fmt: skip
    if model is not None:
        data["distortion_model"] = model
    return yaml.safe_dump(data)


@pytest.mark.parametrize(
    ("name", "text", "match"),
    [
        ("ros_uncal.yaml", _ros("plumb_bob", [0.0] * 9, [0.0] * 5), "미보정"),
        ("ros_fish.yaml", _ros("equidistant", _ROS_K, [0.1, 0.0, 0.0, 0.0]), "왜곡 모델"),
        ("neg.yaml", _OWN.replace("fx: 500", "fx: -160"), "양수"),
        ("nan.yaml", _OWN.replace("fx: 500", "fx: .nan"), "유한|finite"),
        ("out.yaml", _OWN.replace("cx: 320", "cx: 3200"), "밖"),
        (
            "inf.json",
            '{"width": 640, "height": 480, "fx": Infinity, "fy": 1, "cx": 1, "cy": 1}',
            "유한|finite",
        ),
        ("fish.yaml", _OWN + "distortion_model: fisheye\n", "지원하지"),
        ("broken.yaml", "fx: [1, 2\n", "파싱"),
        ("broken.json", '{"fx": 1', "파싱"),
        ("extra.yaml", _OWN + "camera_name: x\n", "계약"),
        ("width.yaml", _OWN.replace("width: 640", "width: -1"), "계약"),
        ("notfs.xml", "<root><a>1</a></root>", "FileStorage|camera_matrix"),
    ],
)
def test_invalid_calibration_files_are_rejected(
    tmp_path: Path, name: str, text: str, match: str
) -> None:
    """잘못된 파일은 처리되지 않은 예외가 아니라 CalibrationError다 (F-PMS-21·22·23).

    ROS 미보정(K=0), 음수·nan·inf·화면 밖 주점, 어안 모델, 파싱 오류, 모르는 키, FileStorage가 아닌
    XML. CLI는 이것을 종료 코드 2로 낸다.
    """
    path = tmp_path / name
    path.write_text(text, "utf-8")
    with pytest.raises(CalibrationError, match=match):
        load_intrinsics_file(path, load_policy(ROOT))


def test_bom_and_rational_polynomial_files_are_imported(tmp_path: Path) -> None:
    """BOM이 붙은 OpenCV YAML, 계수 8개 rational_polynomial camera_info는 opencv 모델로 읽힌다."""
    fs_path = tmp_path / "opencv.yaml"
    fs = cv2.FileStorage(str(fs_path), cv2.FILE_STORAGE_WRITE)
    fs.write("camera_matrix", np.array(_ROS_K).reshape(3, 3))
    fs.write("distortion_coefficients", np.zeros((1, 5)))
    fs.write("image_width", 640)
    fs.write("image_height", 480)
    fs.release()
    bom = tmp_path / "bom.yaml"
    bom.write_bytes(b"\xef\xbb\xbf" + fs_path.read_bytes())
    try:
        got: CameraIntrinsics | None = load_intrinsics_file(bom)
    except CalibrationError:
        got = None  # OpenCV가 BOM을 못 읽으면 형식 오류로 알린다 (처리되지 않은 예외는 아니다)
    assert got is None or got.fx == 500.0
    ros = tmp_path / "rat.yaml"
    dist = [0.1, -0.02, 0.0, 0.0, 0.0, 0.3, 0.1, 0.01]
    ros.write_text(_ros("rational_polynomial", _ROS_K, dist), "utf-8")
    k = load_intrinsics_file(ros, load_policy(ROOT))
    assert k.distortion_model == "opencv" and k.distortion == tuple(dist)
