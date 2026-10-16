"""카메라 내부 파라미터 등록과 작업자 손 크기 사전값 (ADR 0045).

역할
    - `CameraCalibration`: 카메라(바디캠 기기) 하나의 캘리브레이션 결과 한 건. DB
      `camera_calibrations`(추가만)에 쌓인다. 세션은 `Session.calibration.camera_id`로 기기를
      가리키고, 3D 단계는 `resolve_intrinsics`로 쓸 내부 파라미터를 정한다.
    - `WorkerHandSize`: 작업자 손바닥 길이(손목 → 중지 MCP, m) 한 건. DB
      `worker_hand_sizes`(추가만). 3D 손 맞춤의 손 크기 사전값을 작업자 값 → 정책 기본값 순으로
      정한다 (`resolve_palm`).

결정 규칙 (순수 함수, DB를 읽지 않는다)
    - 내부 파라미터: 세션에 직접 적힌 값(`Session.calibration.intrinsics`, 수집 매니페스트) →
      등록된 카메라 캘리브레이션 중 그 세션에 유효한 것 → 없음(정책 화각 근사). 유효한 것: 같은
      camera_id이고 valid_from이 없거나 녹화 시각 이하인 것 중 valid_from이 가장 늦은 것(valid_from
      없는 것보다 있는 것이 앞선다), 같으면 가장 늦게 등록한 것.
    - 손 크기: 그 작업자의 가장 늦게 등록한 값 → 정책
      `prelabel.yaml depth.hand_fit.reference_palm_m`.

작업자 ID는 이 기록과 DB 안에만 있고 어떤 내보내기에도 넣지 않는다. 3D 단계 모델 버전에는 사전값
출처 글자(`+hpw`/`+hpp`)와 입력 해시만 들어간다 (작업자 ID·값을 문자열로 싣지 않는다).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from pydantic import AwareDatetime, Field

from dlp_schema.common import Contract, Identifier
from dlp_schema.session import CameraIntrinsics, Session


class CalibrationMethod(StrEnum):
    """캘리브레이션을 얻은 방법."""

    CHECKERBOARD = (
        "checkerboard"  # 체커보드 영상에서 OpenCV calibrateCamera (`dlp calib intrinsics`)
    )
    IMPORTED = "imported"  # 알려진 캘리브레이션 파일을 가져옴 (`dlp calib import`)


class CameraCalibration(Contract):
    """카메라 하나의 내부 파라미터 등록 한 건 (추가만, 늦게 등록한 것이 앞선다)."""

    # calibration_id: 등록 ID (`calibration_id_for`, 내용·등록 시각 해시).
    # camera_id: 기기 ID (바디캠 일련번호 등, 수집 매니페스트 calibration.camera_id와 같은 값).
    # intrinsics: 핀홀 내부 파라미터와 왜곡 계수 (계수는 기록만 하고 3D 단계는 아직 쓰지 않는다).
    # method: 얻은 방법. rms_px: 캘리브레이션 재투영 RMS(px, 가져온 값은 없을 수 있다).
    # n_views: 쓴 보드 장면 수. source_sha256: 입력 파일(영상·캘리브레이션 파일) sha256.
    # valid_from: 이 시각 이후 녹화한 세션에 쓴다 (None이면 모든 세션, 렌즈 교체 등에 쓴다).
    # created_at: 등록 시각.

    calibration_id: Identifier
    camera_id: Identifier
    intrinsics: CameraIntrinsics
    method: CalibrationMethod
    rms_px: float | None = Field(default=None, ge=0)
    n_views: int | None = Field(default=None, ge=1)
    source_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    valid_from: AwareDatetime | None = None
    created_at: AwareDatetime


class HandSizeSource(StrEnum):
    """작업자 손 크기 값의 출처."""

    MEASURED = "measured"  # 사람이 잰 값 (`dlp calib palm`)
    ESTIMATED = "estimated"  # 캘리브레이션된 세션의 손 맞춤에서 추정 (`dlp calib palm-estimate`)


class WorkerHandSize(Contract):
    """작업자 손바닥 길이(손목 → 중지 MCP) 등록 한 건 (추가만, 늦게 등록한 것이 앞선다)."""

    # record_id: 등록 ID. worker_id: 가명 작업자 ID (내보내기에 넣지 않는다).
    # palm_m: 손바닥 길이 (m, 0.04~0.16). source: 출처. n_samples: 추정에 쓴 표본 수 (잰 값은 None).

    record_id: Identifier
    worker_id: Identifier
    palm_m: float = Field(gt=0.04, lt=0.16)
    source: HandSizeSource
    n_samples: int | None = Field(default=None, ge=1)
    created_at: AwareDatetime


def _digest(*parts: str) -> str:
    """등록 ID 해시 조각 (sha256 앞 12자)."""
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]


def calibration_id_for(
    camera_id: str, intrinsics: CameraIntrinsics, valid_from: datetime | None, now: datetime
) -> str:
    """등록 ID `cal-<해시>` (기기·내부 파라미터·유효 시작·등록 시각)."""
    content = json.dumps(intrinsics.model_dump(mode="json"), sort_keys=True)
    since = valid_from.isoformat() if valid_from else "-"
    return f"cal-{_digest(camera_id, content, since, now.isoformat())}"


def hand_size_id_for(worker_id: str, palm_m: float, source: HandSizeSource, now: datetime) -> str:
    """등록 ID `palm-<해시>` (작업자·값·출처·등록 시각). 작업자 ID는 해시로만 들어간다."""
    return f"palm-{_digest(worker_id, repr(palm_m), source.value, now.isoformat())}"


def effective_calibration(
    records: Iterable[CameraCalibration], camera_id: str, recorded_at: datetime
) -> CameraCalibration | None:
    """그 기기로 recorded_at에 녹화한 세션에 쓸 등록 (모듈 설명의 규칙, 없으면 None)."""
    usable = [
        r
        for r in records
        if r.camera_id == camera_id and (r.valid_from is None or r.valid_from <= recorded_at)
    ]
    if not usable:
        return None
    return max(
        usable,
        key=lambda r: (
            r.valid_from is not None,
            r.valid_from or r.created_at,
            r.created_at,
            r.calibration_id,
        ),
    )


@dataclass(frozen=True)
class ResolvedIntrinsics:
    """3D 단계가 쓸 내부 파라미터와 출처 (`resolve_intrinsics`).

    intrinsics: 쓸 값 (None이면 정책 화각 근사). origin: "session"(세션에 직접 적힘),
    "registry:<calibration_id>"(등록된 카메라 캘리브레이션), "hfov"(없음).
    """

    intrinsics: CameraIntrinsics | None
    origin: str

    def digest_input(self) -> str:
        """3D 단계 입력 해시에 넣을 문자열 (출처와 값이 바뀌면 다시 돈다)."""
        value = self.intrinsics.model_dump(mode="json") if self.intrinsics else None
        return json.dumps({"origin": self.origin, "intrinsics": value}, sort_keys=True)


def resolve_intrinsics(
    session: Session, records: Iterable[CameraCalibration]
) -> ResolvedIntrinsics:
    """세션 바디캠 내부 파라미터: 세션 값 → 등록된 기기 캘리브레이션 → 없음 (모듈 설명)."""
    if session.calibration.intrinsics is not None:
        return ResolvedIntrinsics(session.calibration.intrinsics, "session")
    camera = session.calibration.camera_id
    found = effective_calibration(records, camera, session.recorded_at) if camera else None
    if found is not None:
        return ResolvedIntrinsics(found.intrinsics, f"registry:{found.calibration_id}")
    return ResolvedIntrinsics(None, "hfov")


@dataclass(frozen=True)
class ResolvedPalm:
    """3D 손 맞춤의 손 크기 사전값과 출처 (`resolve_palm`).

    palm_m: 작업자 값 (None이면 정책 기본값을 그대로 쓴다). source: "w"(작업자 등록값) 또는
    "p"(정책 기본값). 모델 버전 접미사 `+hp<source>`에 들어간다.
    """

    palm_m: float | None
    source: str

    def digest_input(self) -> str:
        """3D 단계 입력 해시에 넣을 문자열 (작업자 ID는 넣지 않는다)."""
        return f"palm:{self.source}:{self.palm_m!r}"


def latest_hand_size(records: Iterable[WorkerHandSize], worker_id: str) -> WorkerHandSize | None:
    """그 작업자의 가장 늦게 등록한 손 크기 (없으면 None)."""
    mine = [r for r in records if r.worker_id == worker_id]
    return max(mine, key=lambda r: (r.created_at, r.record_id)) if mine else None


def resolve_palm(records: Iterable[WorkerHandSize], worker_id: str) -> ResolvedPalm:
    """손 크기 사전값: 작업자의 최신 등록값 → 정책 기본값 (모듈 설명)."""
    found = latest_hand_size(records, worker_id)
    return ResolvedPalm(found.palm_m, "w") if found else ResolvedPalm(None, "p")
