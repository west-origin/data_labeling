"""촬영 점검 경고(`dlp_media.capture_check`) 시험.

정답: 프레임레이트·방향 태그·세로·길이·기기 정보·IMU 정지 구간을 알고 만든 합성 probe 결과·
PTS 인덱스·
IMU 표본. 수집 연동은 합성 픽스처 영상(`dlp_fixtures`)으로 본다.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from dlp_media.capture_check import (
    CapturePolicy,
    CaptureWarning,
    calibration_warnings,
    fps,
    imu_warnings,
    load_capture_policy,
    prepare_warnings,
    video_warnings,
)
from dlp_media.imu import ImuData
from dlp_media.probe import MediaInfo, VideoInfo
from dlp_media.pts import PtsIndex
from dlp_schema.session import Calibration

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def policy() -> CapturePolicy:
    """저장소 정책 (min_fps 50, max_session_min 30, 정지 1초)."""
    return load_capture_policy(ROOT)


def _info(width: int, height: int, rotation: int = 0, duration_ms: float = 10_000) -> MediaInfo:
    """합성 probe 결과."""
    video = VideoInfo(
        index=0,
        codec="h264",
        width=width,
        height=height,
        time_base=Fraction(1, 600),
        frames=0,
        rotation_degrees=rotation,
    )
    return MediaInfo(
        path=Path("v.mp4"),
        format_name="mp4",
        duration_ms=duration_ms,
        creation_time=None,
        video=video,
        audio=None,
        data_streams=(),
    )


def _index(rate: int, seconds: int = 2) -> PtsIndex:
    """고정 프레임레이트 PTS 인덱스 (time_base 1/600)."""
    pts = np.arange(rate * seconds, dtype=np.int64) * (600 // rate)
    return PtsIndex(Fraction(1, 600), pts, np.zeros(pts.size, dtype=bool))


def _codes(warnings: Sequence[CaptureWarning]) -> list[str]:
    """경고 코드 목록."""
    return [w.code for w in warnings]


def test_low_fps_portrait_and_long(policy: CapturePolicy) -> None:
    """30 fps 세로 40분 영상 → low_fps, portrait, long_session. 60 fps 가로 10초 → 경고 없음."""
    assert fps(_index(30)) == pytest.approx(30.0)
    bad = video_warnings(_info(1080, 1920, duration_ms=40 * 60_000), _index(30), policy)
    assert _codes(bad) == ["low_fps", "portrait", "long_session"]
    assert video_warnings(_info(1920, 1080), _index(60), policy) == []


def test_rotation_tag_only_without_override(policy: CapturePolicy) -> None:
    """방향 태그 -90°: 지정 각도가 없으면 rotation_tag, 지정하면 없음. 태그대로 돌면 세로라
    portrait.

    준비 점검은 원본의 태그 경고와 준비한 사본(가로 60 fps)의 점검을 합친다.
    """
    src = _info(1920, 1080, rotation=-90)
    assert _codes(video_warnings(src, None, policy)) == ["rotation_tag", "portrait"]
    assert _codes(video_warnings(src, None, policy, rotation_override=180)) == []
    out = prepare_warnings(src, _info(1920, 1080), _index(60), policy, rotation_override=None)
    assert _codes(out) == ["rotation_tag"]


def test_device_info(policy: CapturePolicy) -> None:
    """기기 ID도 내부 파라미터도 없으면 no_device, 기기 ID가 있으면 없음."""
    assert _codes(calibration_warnings(Calibration())) == ["no_device"]
    assert calibration_warnings(Calibration(camera_id="gopro12-wide-60")) == []


def _imu(moving_from_s: float) -> ImuData:
    """200 Hz 3초 IMU: moving_from_s 전에는 정지(중력만), 뒤에는 회전 1 rad/s."""
    t = np.arange(0, 3000, 5, dtype=np.float64)
    acc = np.tile([0.0, 9.81, 0.0], (t.size, 1))
    gyro = np.zeros((t.size, 3))
    gyro[t >= moving_from_s * 1000, 2] = 1.0
    return ImuData(t, acc, gyro, "sidecar:csv")


def test_imu_still_start(policy: CapturePolicy) -> None:
    """정답: 1.5초 정지 뒤 움직임 → 경고 없음, 0.5초부터 → no_still_start, IMU 없음 → no_imu."""
    assert imu_warnings(_imu(1.5), policy) == []
    assert _codes(imu_warnings(_imu(0.5), policy)) == ["no_still_start"]
    assert _codes(imu_warnings(None, policy)) == ["no_imu"]
