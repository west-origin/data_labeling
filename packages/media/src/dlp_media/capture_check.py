"""촬영 조건 점검: 준비(`dlp media prepare`)·수집(`dlp ingest`) 때 내는 경고.

막지 않고 알린다. 수집을 멈출 만큼 확실한 문제가 아니라, 프리라벨 품질을 떨어뜨리는 촬영 습관을 일찍
알리는 것이 목적이다 (2026-10-22 첫 실영상 파일럿: 30 fps, 방향 태그 오류, 렌즈 미확인, IMU 없음).
권장 촬영 조건은 `docs/handover/capture-guide.md`. 기준 값은 `config/policies/media.yaml capture`.

수집 때 영상 점검(`video_warnings`)은 바디캠 스트림만 본다 (3인칭 영상은 점검하지 않는다).

점검 항목 (코드):
- `low_fps`: 프레임레이트가 `min_fps` 미만 (빠른 손동작이 흐려지고 손 추적이 끊긴다).
- `variable_fps`: 가변 프레임레이트 (PTS로 시각을 계산해 오류는 아니나 보간 간격이 고르지 않다).
- `rotation_tag`: 표시 방향 태그가 0이 아니다 (태그가 장착 방향과 다를 수 있다, F-PMS-33.
  준비 때만).
- `portrait`: 바로 세운 영상이 세로다 (가슴·머리 장착 기본은 가로. 장착 방향 확인).
- `long_session`: 세션이 `max_session_min`분보다 길다 (CPU 처리 시간·검수 단위).
- `no_device`: 매니페스트에 기기(`calibration.camera_id`)도 내부 파라미터도 없다 (3D가 화각 근사로
  떨어진다, ADR 0045).
- `no_imu`: IMU가 없다 (바디캠 6자유도 자세·월드 좌표 3D를 만들 수 없다, ADR 0046).
- `no_still_start`: IMU 처음 `still_start_s`초가 정지가 아니다 (자이로 바이어스·중력 방향
  추정에 쓴다).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray
from pydantic import Field

from dlp_media.imu import ImuData
from dlp_media.probe import MediaInfo
from dlp_media.pts import PtsIndex
from dlp_schema.common import Contract
from dlp_schema.session import Calibration


class CapturePolicy(Contract):
    """촬영 점검 기준 (`media.yaml capture`)."""

    min_fps: float = Field(gt=0)  # 이 미만이면 low_fps
    max_session_min: float = Field(gt=0)  # 이보다 긴 세션은 long_session
    still_start_s: float = Field(gt=0)  # 처음 이 시간(초) 동안 정지여야 한다
    still_gyro_rad_s: float = Field(gt=0)  # 정지 판정: 자이로 크기 상한 (rad/s)
    still_acc_std_mps2: float = Field(gt=0)  # 정지 판정: 가속도 크기의 표준편차 상한 (m/s²)


@dataclass(frozen=True)
class CaptureWarning:
    """경고 하나 (코드는 모듈 설명의 목록)."""

    code: str
    message: str


def load_capture_policy(root: Path) -> CapturePolicy:
    """저장소의 `config/policies/media.yaml capture` 절."""
    data: Any = yaml.safe_load((root / "config/policies/media.yaml").read_text("utf-8"))
    return CapturePolicy.model_validate(data["capture"])


def fps(index: PtsIndex) -> float | None:
    """평균 프레임레이트 (프레임 수 - 1) / 길이. 프레임이 하나면 None."""
    ms = index.ms
    span = float(ms[-1] - ms[0])
    return (len(index) - 1) / (span / 1000) if span > 0 else None


def video_warnings(
    info: MediaInfo,
    index: PtsIndex | None,
    policy: CapturePolicy,
    *,
    rotation_override: int | None = None,
) -> list[CaptureWarning]:
    """영상 하나의 점검: 프레임레이트·가변·방향 태그·세로·길이.

    Args:
        info: probe 결과 (준비 전 원본이면 방향 태그를 본다).
        index: PTS 인덱스 (없으면 프레임레이트·가변 점검을 건너뛴다).
        rotation_override: 운영자가 준 표시 각도 (`--display-rotation`). 주었으면 태그 경고를 내지
            않는다 (사람이 영상을 보고 정한 값이다).
    """
    out: list[CaptureWarning] = []
    video = info.video
    if video is None:
        return out
    rate = fps(index) if index is not None else None
    if rate is not None and rate < policy.min_fps:
        out.append(
            CaptureWarning(
                "low_fps",
                f"프레임레이트 {rate:.1f} fps < 권장 {policy.min_fps:g} fps (손 흐림·끊김)",
            )
        )
    if index is not None and index.is_vfr:
        out.append(CaptureWarning("variable_fps", "가변 프레임레이트 (보간 간격이 고르지 않다)"))
    rotation = video.rotation_degrees if rotation_override is None else rotation_override
    if rotation_override is None and video.rotation_degrees % 360:
        out.append(
            CaptureWarning(
                "rotation_tag",
                f"표시 방향 태그 {video.rotation_degrees}°: 장착 방향과 맞는지 영상을 보고 확인 "
                "(틀리면 --display-rotation)",
            )
        )
    width, height = (video.height, video.width) if rotation % 180 else (video.width, video.height)
    if height > width:
        out.append(CaptureWarning("portrait", f"세로 영상 {width}x{height}: 장착 방향 확인"))
    duration = info.duration_ms
    if duration is not None and duration > policy.max_session_min * 60000:
        out.append(
            CaptureWarning(
                "long_session",
                f"길이 {duration / 60000:.1f}분 > {policy.max_session_min:g}분 (나눠 수집 권장)",
            )
        )
    return out


def prepare_warnings(
    source: MediaInfo,
    prepared: MediaInfo,
    index: PtsIndex,
    policy: CapturePolicy,
    *,
    rotation_override: int | None,
) -> list[CaptureWarning]:
    """준비(`dlp media prepare`) 점검: 원본의 방향 태그 + 준비한 사본의 프레임레이트·세로·길이."""
    tag = [
        w
        for w in video_warnings(source, None, policy, rotation_override=rotation_override)
        if w.code == "rotation_tag"
    ]
    return tag + video_warnings(prepared, index, policy, rotation_override=0)


def calibration_warnings(calibration: Calibration | None) -> list[CaptureWarning]:
    """기기·렌즈 정보 점검: camera_id도 내부 파라미터도 없으면 no_device."""
    if calibration is not None and (
        calibration.camera_id is not None or calibration.intrinsics is not None
    ):
        return []
    return [
        CaptureWarning(
            "no_device",
            "매니페스트에 기기(calibration.camera_id)·내부 파라미터가 없다: "
            "3D가 화각 근사로 떨어진다",
        )
    ]


def imu_warnings(imu: ImuData | None, policy: CapturePolicy) -> list[CaptureWarning]:
    """IMU 점검: 없으면 no_imu, 처음 still_start_s초가 정지가 아니면 no_still_start.

    정지 판정: 그 구간 자이로 크기 최댓값 <= still_gyro_rad_s, 가속도 크기 표준편차 <=
    still_acc_std_mps2. 구간에 표본이 2개 미만이면 정지로 보지 않는다.

    값이 NaN인 표본(자이로 채널이 없는 GPMF 기종은 `imu_from_gpmf_payloads`가 자이로를 NaN으로
    채운다)은 "모름"이라 그 채널 판정에서 뺀다. 한 채널에 유한한 표본이 2개 미만이면 그 채널로는
    움직임을 주장하지 않고 남은 채널로만 정한다 (두 채널 다 모르면 경고하지 않는다, F-PMS-43).
    """
    if imu is None or imu.t_ms.size == 0:
        return [CaptureWarning("no_imu", "IMU 없음: 바디캠 자세·월드 좌표 3D를 만들 수 없다")]
    head = imu.t_ms <= imu.t_ms[0] + policy.still_start_s * 1000
    if int(np.count_nonzero(head)) < 2 or not _still(imu, head, policy):
        return [
            CaptureWarning(
                "no_still_start",
                f"처음 {policy.still_start_s:g}초가 정지가 아니다 "
                "(자이로 바이어스·중력 방향 추정이 약해진다)",
            )
        ]
    return []


def _still(imu: ImuData, head: NDArray[np.bool_], policy: CapturePolicy) -> bool:
    """처음 구간(head)이 정지인지. 유한한 표본이 2개 미만인 채널은 판정에 쓰지 않는다."""
    gyro = np.linalg.norm(imu.gyro[head], axis=1)
    gyro = gyro[np.isfinite(gyro)]
    if gyro.size >= 2 and float(gyro.max()) > policy.still_gyro_rad_s:
        return False
    acc = np.linalg.norm(imu.acc[head], axis=1)
    acc = acc[np.isfinite(acc)]
    return not (acc.size >= 2 and float(acc.std()) > policy.still_acc_std_mps2)
