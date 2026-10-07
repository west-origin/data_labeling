"""세션과 스트림.

역할
    녹화 세션 하나(작업자 한 명의 한 번 작업)와 그 안의 스트림(바디캠·IMU·3인칭·장갑·오디오), 세션
    단위 캘리브레이션, 생애주기 상태의 계약 (WP1, WP3, WP4, ADR 0002·0004·0028·0029).

파이프라인
    `dlp ingest`가 세션을 등록하고(`db.repository.insert_session`), `dlp sync run`이 스트림의
    오프셋·배율(드리프트)을 채운다(`update_stream_sync`). 각 단계가 끝나면 생애주기 상태를 한 칸씩
    올린다 (`set_lifecycle`).

시계 규약 (ADR 0019)
    - 기준 스트림은 바디캠 하나다. 바디캠 시계가 곧 마스터 타임라인이다 (offset 0, scale 1, 조정 0).
    - 다른 스트림의 시각 → 마스터 시각: `master = offset_ms + manual_adjustment_ms + stream_ms *
      clock_scale` (`Stream.to_master_ms`). 결과는 실수 ms다. 정수로 반올림하는 것은 호출자 몫이다.
    - 공간 라벨은 스트림 PTS 시각 그대로 저장하므로 다시 동기화해도 바뀌지 않는다.

주요 이름
    - 열거형: `Domain`, `StreamKind`, `SyncMethod`, `PrivacyState`, `LifecycleState`.
    - `LIFECYCLE_ORDER`, `can_transition`: 생애주기 순서와 전이 규칙.
    - `CameraIntrinsics`, `Calibration`, `Stream`, `Session`, `LifecycleEvent`.
    - `ConsentVersion`: 동의서 버전 문자열 타입.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import AwareDatetime, Field, StringConstraints, model_validator

from dlp_schema.common import Confidence, Contract, Identifier, Ms, SemVer


# 작업 도메인. 온톨로지 domains의 키와 같다 (cleaning=청소, caregiving=요양돌봄, nursing=간호).
class Domain(StrEnum):
    CLEANING = "cleaning"
    CAREGIVING = "caregiving"
    NURSING = "nursing"


# 스트림 종류. 세션에는 bodycam이 정확히 하나 있어야 한다 (`Session` 검증기).
class StreamKind(StrEnum):
    BODYCAM = "bodycam"  # 1인칭 바디캠 영상 (기준 스트림)
    IMU = "imu"  # 관성 센서 (보통 바디캠 내장, GPMF 등에서 추출)
    THIRD_PERSON = "third_person"  # 3인칭 고정 카메라 영상
    GLOVE_LEFT = "glove_left"  # 왼손 센서 장갑 (압력 등)
    GLOVE_RIGHT = "glove_right"  # 오른손 센서 장갑
    AUDIO = "audio"  # 별도 녹음 오디오


# 스트림을 마스터 타임라인에 맞춘 방법 (`dlp sync run`이
# 정책 `config/policies/sync.yaml` 우선순위로 고른다).
class SyncMethod(StrEnum):
    REFERENCE = "reference"  # 기준 스트림(바디캠) 자신
    SHARED_CLOCK = "shared_clock"  # 바디캠과 같은 시계 (IMU, 내장 오디오)
    QR_SLATE = "qr_slate"  # 화면에 띄운 시각 QR(슬레이트)을 두 영상에서 읽어 맞춤
    TAP_EVENT = "tap_event"  # 두드림(오디오 피크·장갑 압력 스파이크) 시점을 맞춤
    AUDIO_XCORR = "audio_xcorr"  # 오디오 상호상관
    MOTION_XCORR = "motion_xcorr"  # 운동 신호(IMU·장갑) 상호상관
    MANUAL = "manual"  # 사람이 직접 맞춤 (`dlp sync adjust`)
    UNSYNCED = "unsynced"  # 아직 맞추지 못함 (기본값)


# 세션의 프라이버시 게이트 상태 (`dlp privacy detect|approve|render`).
class PrivacyState(StrEnum):
    PENDING = "pending"  # 탐지 전
    AUTO_BLURRED = "auto_blurred"  # 자동 탐지·블러 트랙 생성, 사람 승인 전
    # 원본 접근 권한자가 블러를 승인함 (이후 블러본을 일반 라벨러에게 보여 줄 수 있다)
    APPROVED = "approved"


class LifecycleState(StrEnum):
    """한 방향으로만 이동한다. 어느 단계에서든 withdrawn으로 빠질 수 있다."""

    # 클래스 docstring은 JSON Schema description이므로 바꾸지 않고 여기 주석으로 보강한다.
    RAW_INGESTED = "raw_ingested"  # 원본 수집·등록 완료 (`dlp ingest`)
    PRIVACY_APPROVED = "privacy_approved"  # 블러 사람 승인 (`dlp privacy approve`)
    PRELABELED = "prelabeled"  # 자동 프리라벨 완료 (`dlp prelabel run`)
    HUMAN_VERIFIED = "human_verified"  # 사람 검수 완료 판정 (`dlp review verify`, ADR 0029)
    SPLIT_ASSIGNED = "split_assigned"  # 데이터셋 버전의 분할에 배정 (`dlp dataset build`)
    EXPORTED = "exported"  # 내보내기에 포함됨 (`dlp export ...`)
    WITHDRAWN = "withdrawn"  # 사용 중지 (`dlp dataset withdraw`). 끝 상태, 되돌릴 수 없다


# withdrawn을 뺀 정상 진행 순서. `can_transition`이 인덱스 차이로 "한 칸 앞"을 판정한다.
LIFECYCLE_ORDER: tuple[LifecycleState, ...] = (
    LifecycleState.RAW_INGESTED,
    LifecycleState.PRIVACY_APPROVED,
    LifecycleState.PRELABELED,
    LifecycleState.HUMAN_VERIFIED,
    LifecycleState.SPLIT_ASSIGNED,
    LifecycleState.EXPORTED,
)


def can_transition(current: LifecycleState, target: LifecycleState) -> bool:
    """생애주기 전이 허용 여부. 같은 상태 유지는 허용(멱등), 되돌아가기는 금지.

    규칙:
        - withdrawn에서는 withdrawn(멱등)으로만 갈 수 있다.
        - 어느 상태에서든 withdrawn으로 갈 수 있다.
        - 그 밖에는 같은 상태 또는 `LIFECYCLE_ORDER`에서
          바로 다음 상태로만 갈 수 있다 (건너뛰기 금지).

    Returns:
        허용이면 True. DB 갱신은 하지 않는다 (`db.repository.set_lifecycle`이 이 함수로 검사한다).
    """
    if current is LifecycleState.WITHDRAWN:
        return target is LifecycleState.WITHDRAWN
    if target is LifecycleState.WITHDRAWN:
        return True
    return LIFECYCLE_ORDER.index(target) - LIFECYCLE_ORDER.index(current) in (0, 1)


# 카메라 내부 파라미터 (핀홀 모델, 픽셀 단위).
#   width / height: 영상 해상도(px). fx, fy: 초점 거리(px). cx, cy: 주점(px).
#   distortion_model: 왜곡 모델 이름 (기본 "opencv").
#   distortion: 그 모델의 왜곡 계수 (예: k1, k2, p1, p2, k3).
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

    # mount_position: 바디캠 장착 위치 (예: 가슴, 머리). 모르면 None.
    # intrinsics: 바디캠 내부 파라미터. 없으면 깊이·3D 단계가 정책 기본값을 쓴다.
    # camera_imu_extrinsics: IMU 좌표계 → 카메라 좌표계 4x4 동차 변환 (행 우선 중첩 튜플).
    # glove_model: 장갑 기종 이름. glove_calibration: 장갑 보정 값 (이름 → 값, 기종별).

    mount_position: str | None = None
    intrinsics: CameraIntrinsics | None = None
    camera_imu_extrinsics: tuple[tuple[float, ...], ...] | None = Field(
        default=None, description="IMU→카메라 4x4 변환 행렬"
    )
    glove_model: str | None = None
    glove_calibration: dict[str, float] = Field(default_factory=dict)


class Stream(Contract):
    """세션 안의 스트림 하나. 기준 시각 = offset_ms + 스트림 시각 * clock_scale."""

    # 위 docstring(JSON Schema description)은 manual_adjustment_ms를 생략했다. 실제 변환은
    # `to_master_ms`: master = offset_ms + manual_adjustment_ms + stream_ms * clock_scale.
    # stream_id: 세션 안에서 고유한 스트림 이름 (예: bodycam, imu, tp1).
    # kind: 스트림 종류.
    # uri: 원본 위치 (원본 버킷 `dlp-raw`. 일반 라벨러 경로에 노출 금지).
    # sample_rate_hz: 표본화 주파수 (IMU·장갑·오디오). 영상은 VFR이라 보통 None (PTS 인덱스를 쓴다).
    # pts_index_uri: 영상 PTS 인덱스 위치 (`dlp ingest`가 만든다). 영상 시각은 이것으로만 계산한다.
    # offset_ms: 동기화 오프셋 (ms, 실수). clock_scale: 시계 배율(드리프트 보정, 0보다 큼).
    # sync_method / sync_confidence: 동기화 방법과 신뢰도(0~1, 모르면 None).
    # manual_adjustment_ms: 사람 미세 조정 (`dlp sync adjust`,
    # ms). 자동 동기화를 다시 해도 따로 보존된다.

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
        """마스터 시계와 같은 시계인가 (오프셋 0, 배율 1, 사람 조정 0).

        기준 스트림(바디캠)은 늘 True여야 한다 (`Session` 검증기, `update_stream_sync`).
        """
        return self.offset_ms == 0 and self.clock_scale == 1 and self.manual_adjustment_ms == 0

    def to_master_ms(self, stream_ms: float) -> float:
        """스트림 시각(ms) → 마스터 타임라인 시각(ms, 실수).

        Args:
            stream_ms: 이 스트림 시계의 시각 (영상이면 PTS ms).

        Returns:
            `offset_ms + manual_adjustment_ms + stream_ms * clock_scale`. 반올림하지 않는다.
            역변환(마스터 → 스트림)은 `dlp_export.frames.stream_ms` 등 호출자 쪽에 있다.
        """
        return self.offset_ms + self.manual_adjustment_ms + stream_ms * self.clock_scale


# 동의서 버전. DB sessions.consent_version(VARCHAR(64))과 길이를 맞춘다 (ADR 0028)
ConsentVersion = Annotated[str, StringConstraints(min_length=1, max_length=64)]


# 녹화 세션 하나.
#   session_id: 세션 ID. domain: 도메인.
#   worker_id / site_id: 가명 작업자·장소 ID (분할 누수 방지 단위).
#   consent_version: 동의서 버전. recorded_at: 녹화 시작 시각 (시간대 필수).
#   duration_ms: 세션 길이 (마스터 타임라인 ms).
#   streams: 스트림 목록 (순서는 DB streams.position으로 보존된다). 바디캠 정확히 하나.
#   calibration: 세션 캘리브레이션 (기본 빈 값).
#   privacy_state / lifecycle_state: 현재 상태. DB에서는 전용 함수로만 바꾼다
#     (`set_privacy_state`, `set_lifecycle`).
#   ontology_version: 이 세션 라벨이 따르는 온톨로지 버전 (DB FK). 아직 없으면 None.
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
        """스트림 구성 규칙을 검사한다.

        - stream_id 중복 금지
        - 바디캠 정확히 하나, 그 sync_method는 reference, 시계는 항등 (offset 0·배율 1·조정 0)
        - reference 방법은 바디캠만 쓸 수 있다

        Raises:
            ValueError: 규칙 위반 (Pydantic이 ValidationError로 감싼다).
        """
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
        """기준 스트림(바디캠). 검증기가 정확히 하나를 보장한다."""
        return next(s for s in self.streams if s.kind is StreamKind.BODYCAM)

    def stream(self, stream_id: str) -> Stream:
        """ID로 스트림을 찾는다.

        Raises:
            KeyError: 그런 스트림이 없을 때.
        """
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
