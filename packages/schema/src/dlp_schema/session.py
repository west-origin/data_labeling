"""세션과 스트림."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import AwareDatetime, Field, StringConstraints, model_validator

from dlp_schema.common import Confidence, Contract, Identifier, Ms, SemVer


class Domain(StrEnum):
    CLEANING = "cleaning"
    CAREGIVING = "caregiving"
    NURSING = "nursing"


class StreamKind(StrEnum):
    BODYCAM = "bodycam"
    IMU = "imu"
    THIRD_PERSON = "third_person"
    GLOVE_LEFT = "glove_left"
    GLOVE_RIGHT = "glove_right"
    AUDIO = "audio"


class SyncMethod(StrEnum):
    REFERENCE = "reference"  # 기준 스트림(바디캠) 자신
    SHARED_CLOCK = "shared_clock"  # 바디캠과 같은 시계 (IMU, 내장 오디오)
    QR_SLATE = "qr_slate"
    TAP_EVENT = "tap_event"
    AUDIO_XCORR = "audio_xcorr"
    MOTION_XCORR = "motion_xcorr"
    MANUAL = "manual"
    UNSYNCED = "unsynced"


class PrivacyState(StrEnum):
    PENDING = "pending"
    AUTO_BLURRED = "auto_blurred"
    APPROVED = "approved"


class LifecycleState(StrEnum):
    """한 방향으로만 이동한다. 어느 단계에서든 withdrawn으로 빠질 수 있다."""

    RAW_INGESTED = "raw_ingested"
    PRIVACY_APPROVED = "privacy_approved"
    PRELABELED = "prelabeled"
    HUMAN_VERIFIED = "human_verified"
    SPLIT_ASSIGNED = "split_assigned"
    EXPORTED = "exported"
    WITHDRAWN = "withdrawn"


LIFECYCLE_ORDER: tuple[LifecycleState, ...] = (
    LifecycleState.RAW_INGESTED,
    LifecycleState.PRIVACY_APPROVED,
    LifecycleState.PRELABELED,
    LifecycleState.HUMAN_VERIFIED,
    LifecycleState.SPLIT_ASSIGNED,
    LifecycleState.EXPORTED,
)


def can_transition(current: LifecycleState, target: LifecycleState) -> bool:
    """생애주기 전이 허용 여부. 같은 상태 유지는 허용(멱등), 되돌아가기는 금지."""
    if current is LifecycleState.WITHDRAWN:
        return target is LifecycleState.WITHDRAWN
    if target is LifecycleState.WITHDRAWN:
        return True
    return LIFECYCLE_ORDER.index(target) - LIFECYCLE_ORDER.index(current) in (0, 1)


class CameraIntrinsics(Contract):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fx: float
    fy: float
    cx: float
    cy: float
    distortion_model: str = "opencv"
    distortion: tuple[float, ...] = ()


class Calibration(Contract):
    """세션 단위 캘리브레이션. 장착 위치가 바뀔 수 있어 세션마다 기록한다."""

    mount_position: str | None = None
    intrinsics: CameraIntrinsics | None = None
    camera_imu_extrinsics: tuple[tuple[float, ...], ...] | None = Field(
        default=None, description="IMU→카메라 4x4 변환 행렬"
    )
    glove_model: str | None = None
    glove_calibration: dict[str, float] = Field(default_factory=dict)


class Stream(Contract):
    """세션 안의 스트림 하나. 기준 시각 = offset_ms + 스트림 시각 * clock_scale."""

    stream_id: Identifier
    kind: StreamKind
    uri: str
    sample_rate_hz: float | None = Field(default=None, gt=0)
    pts_index_uri: str | None = None
    offset_ms: float = 0.0
    clock_scale: float = Field(default=1.0, gt=0)
    sync_method: SyncMethod = SyncMethod.UNSYNCED
    sync_confidence: Confidence | None = None
    manual_adjustment_ms: float = 0.0

    @property
    def is_identity_clock(self) -> bool:
        """마스터 시계와 같은 시계인가 (오프셋 0, 배율 1, 사람 조정 0)."""
        return self.offset_ms == 0 and self.clock_scale == 1 and self.manual_adjustment_ms == 0

    def to_master_ms(self, stream_ms: float) -> float:
        return self.offset_ms + self.manual_adjustment_ms + stream_ms * self.clock_scale


# 동의서 버전. DB sessions.consent_version(VARCHAR(64))과 길이를 맞춘다 (ADR 0028)
ConsentVersion = Annotated[str, StringConstraints(min_length=1, max_length=64)]


class Session(Contract):
    session_id: Identifier
    domain: Domain
    worker_id: Identifier = Field(description="가명 작업자 ID")
    site_id: Identifier = Field(description="가명 장소 ID")
    consent_version: ConsentVersion = Field(description="동의서 버전 (1~64자)")
    recorded_at: AwareDatetime
    duration_ms: Ms
    streams: tuple[Stream, ...]
    calibration: Calibration = Calibration()
    privacy_state: PrivacyState = PrivacyState.PENDING
    lifecycle_state: LifecycleState = LifecycleState.RAW_INGESTED
    ontology_version: SemVer | None = None

    @model_validator(mode="after")
    def _check_streams(self) -> Session:
        ids = [s.stream_id for s in self.streams]
        if len(ids) != len(set(ids)):
            raise ValueError("stream_id가 중복되었습니다")
        bodycams = [s for s in self.streams if s.kind is StreamKind.BODYCAM]
        if len(bodycams) != 1:
            raise ValueError("세션에는 기준 스트림인 바디캠이 정확히 하나 있어야 합니다")
        ref = bodycams[0]
        if ref.sync_method is not SyncMethod.REFERENCE:
            raise ValueError("바디캠 스트림의 sync_method는 reference여야 합니다")
        if not ref.is_identity_clock:
            raise ValueError(
                "기준 스트림(바디캠)은 offset_ms 0, clock_scale 1, "
                "manual_adjustment_ms 0이어야 합니다"
            )
        if any(s.sync_method is SyncMethod.REFERENCE for s in self.streams if s is not ref):
            raise ValueError("reference 동기화 방법은 바디캠 스트림만 쓸 수 있습니다")
        return self

    @property
    def reference_stream(self) -> Stream:
        return next(s for s in self.streams if s.kind is StreamKind.BODYCAM)

    def stream(self, stream_id: str) -> Stream:
        for s in self.streams:
            if s.stream_id == stream_id:
                return s
        raise KeyError(stream_id)


class LifecycleEvent(Contract):
    """세션 생애주기 전이 기록 (DB session_lifecycle_events, 추가만 한다. ADR 0028).

    세션 등록 때 처음 상태(from_state 없음)를 한 번, 이후 상태가 바뀔 때마다 한 번 남는다.
    같은 상태로의 멱등 호출은 남기지 않는다. 예: 사람 검수 완료 시각 = human_verified로 처음
    전이한 기록의 at.
    """

    event_id: int = Field(ge=1, description="DB가 매기는 일련번호 (같은 세션 안에서 시간 순)")
    session_id: Identifier
    from_state: LifecycleState | None = Field(description="이전 상태. 세션 등록 기록이면 None")
    to_state: LifecycleState
    at: AwareDatetime
    actor: str | None = Field(default=None, description="전이를 일으킨 사람·단계 (모르면 None)")
