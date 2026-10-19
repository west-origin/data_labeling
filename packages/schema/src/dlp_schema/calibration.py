"""카메라 내부 파라미터 등록과 작업자 손 크기 사전값 (ADR 0045, 0052).

역할
    - `CameraCalibration`: 카메라(바디캠 기기) 하나의 캘리브레이션 결과 한 건. DB
      `camera_calibrations`(추가만)에 쌓인다. 세션은 수집 매니페스트 `Session.calibration.camera_id`
      또는 수집 뒤의 기기 연결 기록(`SessionCameraAssignment`, `session_camera_assignments`, 추가만,
      ADR 0052)으로 기기를 가리키고, 3D 단계는 `resolve_intrinsics`로 쓸 내부 파라미터를 정한다.
    - `WorkerHandSize`: 작업자 손바닥 길이(손목 → 중지 MCP, m) 한 건. DB
      `worker_hand_sizes`(추가만). 추정값은 표본을 낸 세션 ID(`source_session_ids`,
      `worker_hand_size_sources`)와 추정기 판을 남긴다.
    - `HandSizeInvalidation`: 손 크기 등록의 무효 기록 (추가만, `worker_hand_size_invalidations`,
      ADR 0052). 작업자 동의 철회·삭제 요구, 잘못 등록한 값에 쓴다. 행을 지우거나 고치지 않고 무효
      기록을 더해 그 등록을 결정에서 뺀다.

결정 규칙 (순수 함수, DB를 읽지 않는다)
    - 기기: 매니페스트 `calibration.camera_id` → 그 세션의 가장 늦은 기기 연결 기록 → 없음.
      매니페스트 값이 있으면 연결 기록은 쓰지 않는다 (연결 명령도 다른 기기를 거부한다).
    - 내부 파라미터: 세션에 직접 적힌 값(`Session.calibration.intrinsics`, 수집 매니페스트) →
      그 기기의 등록 중 그 세션에 유효한 것 → 없음(정책 화각 근사). 유효한 것: 같은 camera_id이고
      valid_from이 없거나 녹화 시각 이하인 것 중 valid_from이 가장 늦은 것(valid_from 있는
      것이 없는 것보다 앞선다), 같으면 가장 늦게 등록한 것. 그래서 잘못 등록한 값은 같은
      valid_from으로 다시 등록해야 대체된다 (`dlp calib intrinsics|import --supersede`).
    - 손 크기: 그 작업자의 유효한 등록 중 가장 늦게 등록한 값 → 정책
      `prelabel.yaml depth.hand_fit.reference_palm_m`. 유효하지 않은 등록: 무효 기록이 가리키는
      것(등록 하나, 또는 작업자 전체 무효 기록 시각 이전의 모든 등록)과, 표본을 낸 세션 중 하나라도
      사용 중지된 추정값.

작업자 ID는 이 기록과 DB 안에만 있고 어떤 내보내기에도 넣지 않는다. 3D 단계 모델 버전에는 사전값
출처 글자(`+hpw`/`+hpp`)와 입력 해시만 들어간다 (작업자 ID·값을 문자열로 싣지 않는다). 무효 기록이
결정을 바꾸면 입력 해시가 바뀌어 다음 `dlp prelabel run`이 검수 전 3D만 다시 만든다.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Set
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from pydantic import AwareDatetime, Field, model_validator

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


class SessionCameraAssignment(Contract):
    """수집 뒤에 세션을 바디캠 기기에 잇는 기록 한 건 (추가만, ADR 0052, `dlp calib assign`).

    매니페스트에 camera_id가 없던 세션(ADR 0045 이전 수집 등)에 쓴다. 세션 행(수집 JSON)을 바꾸면
    수집 재실행의 충돌 검사와 부딪히므로 세션 밖 표에 둔다. 가장 늦은 기록이 쓰인다.
    """

    # assignment_id: 기록 ID (`assignment_id_for`). session_id: 세션. camera_id: 기기 ID.
    # reason: 운영자가 적은 근거 (예: 촬영 기록부). created_at: 기록 시각.

    assignment_id: Identifier
    session_id: Identifier
    camera_id: Identifier
    reason: str = Field(min_length=1, max_length=500)
    created_at: AwareDatetime


class HandSizeSource(StrEnum):
    """작업자 손 크기 값의 출처."""

    MEASURED = "measured"  # 사람이 잰 값 (`dlp calib palm`)
    ESTIMATED = "estimated"  # 캘리브레이션된 세션의 손 맞춤에서 추정 (`dlp calib palm-estimate`)


class WorkerHandSize(Contract):
    """작업자 손바닥 길이(손목 → 중지 MCP) 등록 한 건 (추가만, 늦게 등록한 유효한 것이 앞선다)."""

    # record_id: 등록 ID. worker_id: 가명 작업자 ID (내보내기에 넣지 않는다).
    # palm_m: 손바닥 길이 (m, 0.04~0.16). source: 출처. n_samples: 추정에 쓴 표본 수 (잰 값은 None).
    # source_session_ids: 추정 표본을 낸 세션 ID (ID 순, 잰 값은 빈 튜플). 0052 이전 추정은 비어
    #   있다(출처 모름). 이 중 하나라도 사용 중지되면 이 등록은 결정에서 빠진다.
    # estimator: 추정기 판 (`<깊이 3D 판>+pe<palm_estimate 절 해시>`, 잰 값은 None).

    record_id: Identifier
    worker_id: Identifier
    palm_m: float = Field(gt=0.04, lt=0.16)
    source: HandSizeSource
    n_samples: int | None = Field(default=None, ge=1)
    source_session_ids: tuple[Identifier, ...] = Field(
        default=(), description="추정 표본을 낸 세션 ID (사용 중지 전파·계보)"
    )
    estimator: str | None = Field(default=None, max_length=256)
    created_at: AwareDatetime

    @model_validator(mode="after")
    def _measured_has_no_estimate_provenance(self) -> WorkerHandSize:
        """잰 값에는 추정 출처(세션·추정기)가 없다."""
        if self.source is HandSizeSource.MEASURED and (self.source_session_ids or self.estimator):
            raise ValueError("measured 등록에는 source_session_ids·estimator를 두지 않는다")
        return self


class HandSizeInvalidation(Contract):
    """손 크기 등록 무효 기록 한 건 (추가만, ADR 0052, `dlp calib palm-revoke`).

    record_id가 있으면 그 등록 하나, None이면 그 작업자의 created_at 이전(같은 시각 포함) 모든
    등록을 무효로 한다. 무효 뒤에 새로 등록한 값(동의를 다시 받은 뒤 잰 값 등)은 다시 쓰인다.
    """

    # invalidation_id: 기록 ID. worker_id: 가명 작업자 ID. record_id: 대상 등록 (None: 작업자 전체).
    # reason: 근거 (예: 동의 철회, 삭제 요청 번호, 잘못 잰 값). created_at: 기록 시각.

    invalidation_id: Identifier
    worker_id: Identifier
    record_id: Identifier | None = None
    reason: str = Field(min_length=1, max_length=500)
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


def invalidation_id_for(worker_id: str, record_id: str | None, now: datetime) -> str:
    """무효 기록 ID `palmx-<해시>` (작업자·대상 등록·기록 시각)."""
    return f"palmx-{_digest(worker_id, record_id or '*', now.isoformat())}"


def assignment_id_for(session_id: str, camera_id: str, now: datetime) -> str:
    """기기 연결 기록 ID `camasg-<해시>` (세션·기기·기록 시각)."""
    return f"camasg-{_digest(session_id, camera_id, now.isoformat())}"


def effective_calibration(
    records: Iterable[CameraCalibration], camera_id: str, recorded_at: datetime
) -> CameraCalibration | None:
    """그 기기로 recorded_at에 녹화한 세션에 쓸 등록 (모듈 설명의 규칙, 없으면 None).

    정렬 키 (큰 것이 앞선다): valid_from이 있는가 → valid_from(없으면 등록 시각) → 등록 시각 → ID.
    """
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


def session_camera_id(
    session: Session, assignments: Iterable[SessionCameraAssignment]
) -> str | None:
    """세션 바디캠 기기 ID: 매니페스트 값 → 그 세션의 가장 늦은 연결 기록 → None."""
    if session.calibration.camera_id:
        return session.calibration.camera_id
    mine = [a for a in assignments if a.session_id == session.session_id]
    return max(mine, key=lambda a: (a.created_at, a.assignment_id)).camera_id if mine else None


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
    session: Session,
    records: Iterable[CameraCalibration],
    assignments: Iterable[SessionCameraAssignment] = (),
) -> ResolvedIntrinsics:
    """세션 바디캠 내부 파라미터: 세션 값 → 등록된 기기 캘리브레이션 → 없음 (모듈 설명)."""
    if session.calibration.intrinsics is not None:
        return ResolvedIntrinsics(session.calibration.intrinsics, "session")
    camera = session_camera_id(session, assignments)
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


def hand_size_invalid_reason(
    record: WorkerHandSize,
    invalidations: Iterable[HandSizeInvalidation],
    withdrawn: Set[str] = frozenset(),
) -> str | None:
    """그 등록이 결정에서 빠지는 이유 (유효하면 None).

    무효 기록(등록 하나, 또는 그 시각 이전 작업자 전체), 또는 추정 표본을 낸 세션의 사용 중지.
    """
    for inv in invalidations:
        if inv.worker_id != record.worker_id:
            continue
        if inv.record_id == record.record_id or (
            inv.record_id is None and record.created_at <= inv.created_at
        ):
            return f"무효 {inv.invalidation_id} ({inv.reason})"
    gone = sorted(set(record.source_session_ids) & set(withdrawn))
    return f"출처 세션 사용 중지 {gone}" if gone else None


def latest_hand_size(records: Iterable[WorkerHandSize], worker_id: str) -> WorkerHandSize | None:
    """그 작업자의 가장 늦게 등록한 손 크기 (유효 여부와 무관, 없으면 None)."""
    mine = [r for r in records if r.worker_id == worker_id]
    return max(mine, key=lambda r: (r.created_at, r.record_id)) if mine else None


def effective_hand_size(
    records: Iterable[WorkerHandSize],
    worker_id: str,
    invalidations: Iterable[HandSizeInvalidation] = (),
    withdrawn: Set[str] = frozenset(),
) -> WorkerHandSize | None:
    """그 작업자의 유효한 등록 중 가장 늦은 것 (없으면 None, 모듈 설명)."""
    invs = list(invalidations)
    valid = [
        r
        for r in records
        if r.worker_id == worker_id and hand_size_invalid_reason(r, invs, withdrawn) is None
    ]
    return latest_hand_size(valid, worker_id)


def resolve_palm(
    records: Iterable[WorkerHandSize],
    worker_id: str,
    invalidations: Iterable[HandSizeInvalidation] = (),
    withdrawn: Set[str] = frozenset(),
) -> ResolvedPalm:
    """손 크기 사전값: 작업자의 유효한 최신 등록값 → 정책 기본값 (모듈 설명)."""
    found = effective_hand_size(records, worker_id, invalidations, withdrawn)
    return ResolvedPalm(found.palm_m, "w") if found else ResolvedPalm(None, "p")
