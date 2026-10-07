"""라벨 레코드와 종류별 페이로드.

라벨은 덮어쓰지 않는다. 수정은 새 레코드를 만들고 parent_label_id로 이전 레코드를 가리킨다.
삭제(오탐 제거)는 retracted=True인 새 레코드로 표현한다.

역할
    플랫폼의 모든 라벨(사람·모델·센서)을 담는 단일 레코드 형식 `LabelRecord`와, `kind`로 구별되는
    페이로드 14종을 정의한다 (WP1, ADR 0002). DB `label_records`에 한 행으로 저장된다
    (`db.repository.label_to_row`/`row_to_label`). 검수 상태(verification) 외의 열은 DB 트리거가
    수정을 막는다.

페이로드 종류 (`kind`)
    공간 라벨 (`SPATIAL_KINDS`, stream_id 필수, 시각 = 그 스트림 PTS ms, ADR 0019·0022):
        box_track, mask_track, keypoint_track, blur_track, trajectory3d
    시간 구간 라벨 (시각 = 마스터 타임라인 ms):
        hand_state, action, segment, gap, object_state, coverage, event, relation, description

주요 이름
    - 메타데이터: `Source`, `Evidence`, `VerificationState`, `Provenance`, `Verification`.
    - 좌표·3D: `CoordinateFrame`, `Source3D`, `Hand`.
    - 키프레임: `BoxKeyframe`, `Rle`, `MaskKeyframe`, `Keypoint`, `KeypointFrame`,
      `Trajectory3DSample`.
    - 페이로드: `*Payload` 14종과 판별 공용체 `LabelPayload`.
    - `SKELETON_SIZES`, `SPATIAL_KINDS`, `LabelRecord`.

주의
    - 클래스 docstring과 Field description은 `schemas/label_record.schema.json` 등에 들어간다.
      바꾸면 `make schemas`가 필요하므로 설명 보강은 `#` 주석으로 한다.
    - 좌표 단위: 박스·키포인트 x, y, w, h는 그 스트림 영상의 픽셀 좌표(왼쪽 위 원점)로 쓴다.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, model_validator

from dlp_schema.common import Confidence, Contract, Identifier, Ms, OntologyId, SemVer

# ---------------------------------------------------------------- 공통 메타데이터


# 라벨 출처.
class Source(StrEnum):
    HUMAN = "human"  # 사람이 만들거나 고친 라벨
    MODEL = "model"  # 자동 모델 (model_version·confidence 필수)
    SENSOR = "sensor"  # 센서에서 바로 얻은 라벨 (예: 장갑 압력 접촉, sensor_id 필수)


# 근거 수준.
class Evidence(StrEnum):
    OBSERVED = "observed"  # 영상·센서에서 직접 보임 (기본값)
    INFERRED = "inferred"  # 가려짐 등으로 앞뒤 맥락에서 추정
    UNKNOWN = "unknown"  # 판단 불가


# 검증 상태. DB에서 바꿀 수 있는 유일한 라벨 열 묶음이다 (`db.repository.record_review`).
class VerificationState(StrEnum):
    UNREVIEWED = "unreviewed"  # 검수 전
    # 높은 신뢰도 묶음의 표본이 합격해 묶음 전체가 검증된 것으로 봄
    SAMPLE_VERIFIED = "sample_verified"
    HUMAN_APPROVED = "human_approved"  # 사람이 그대로 승인
    HUMAN_CORRECTED = "human_corrected"  # 사람이 고친 결과 레코드 (또는 고친 사람이 확인한 레코드)


# 3D 궤적 좌표계.
class CoordinateFrame(StrEnum):
    CAMERA = "camera"  # 그 시각 카메라 좌표계 (단안 깊이 결과)
    BODY = "body"  # 작업자 몸 기준 좌표계
    WORLD = "world"  # 세션 고정 월드 좌표계 (SLAM 결과)


# 3D 값을 얻은 방법.
class Source3D(StrEnum):
    MONO_DEPTH = "mono_depth"  # 단안 메트릭 깊이 추정
    VISUAL_SLAM = "visual_slam"  # 영상 SLAM
    VISUAL_INERTIAL_SLAM = "visual_inertial_slam"  # 영상 + IMU SLAM
    MULTI_VIEW = "multi_view"  # 여러 카메라 삼각측량


# 작업자의 왼손·오른손 (작업자 기준. 영상의 좌우와 다를 수 있다).
class Hand(StrEnum):
    LEFT = "left"
    RIGHT = "right"


# 라벨 출처 정보.
#   source: 출처.
#   model_version: 모델 버전 (source=model이면 필수, 정책 해시 포함 가능, 길이 제한 없음).
#   sensor_id: 센서 ID (source=sensor면 필수).
class Provenance(Contract):
    source: Source
    model_version: str | None = None
    sensor_id: str | None = None

    @model_validator(mode="after")
    def _check(self) -> Provenance:
        """출처별 필수 필드를 검사한다. Raises: ValueError (모델에 버전 없음, 센서에 ID 없음)."""
        if self.source is Source.MODEL and not self.model_version:
            raise ValueError("모델 출처 라벨에는 model_version이 필요합니다")
        if self.source is Source.SENSOR and not self.sensor_id:
            raise ValueError("센서 출처 라벨에는 sensor_id가 필요합니다")
        return self


# 검수 정보. state가 unreviewed가 아니면 reviewer_id와 reviewed_at(시간대 필수)이 있어야 한다.
class Verification(Contract):
    state: VerificationState = VerificationState.UNREVIEWED
    reviewer_id: Identifier | None = None
    reviewed_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def _check(self) -> Verification:
        """검수된 상태에 검수자·시각이 있는지 검사한다. Raises: ValueError."""
        if self.state is not VerificationState.UNREVIEWED and (
            self.reviewer_id is None or self.reviewed_at is None
        ):
            raise ValueError("검수된 라벨에는 reviewer_id와 reviewed_at이 필요합니다")
        return self


# ---------------------------------------------------------------- 공간 트랙


# 박스 키프레임. t_ms: 그 스트림 PTS 시각(ms). x, y: 왼쪽 위 꼭짓점(px). w, h: 폭·높이(px, 0 이상).
# outside: 이 시각부터 대상이 화면 밖·안 보임 (CVAT의 outside와 같은 뜻, 좌표는 무시).
# 키프레임 사이 값은 소비자가 보간한다 (선형 등, 각 단계 정책).
class BoxKeyframe(Contract):
    t_ms: Ms
    x: float
    y: float
    w: float = Field(ge=0)
    h: float = Field(ge=0)
    outside: bool = False


class Rle(Contract):
    """COCO 형식 RLE 마스크."""

    # size: (높이, 너비) px — COCO 순서. counts: COCO 압축 RLE 문자열.

    size: tuple[int, int]
    counts: str


# 마스크 키프레임. t_ms: 스트림 PTS ms. rle: 마스크 (outside가 아니면 필수). outside: 안 보임.
class MaskKeyframe(Contract):
    t_ms: Ms
    rle: Rle | None = None
    outside: bool = False

    @model_validator(mode="after")
    def _check(self) -> MaskKeyframe:
        """보이는(outside=False) 키프레임에 rle가 있는지 검사한다. Raises: ValueError."""
        if not self.outside and self.rle is None:
            raise ValueError("보이는 마스크 키프레임에는 rle가 필요합니다")
        return self


# 키포인트 하나. x, y: 픽셀 좌표. visibility: COCO 규약 (0 없음, 1 가려짐, 2 보임).
# confidence: 모델 신뢰도 (선택).
class Keypoint(Contract):
    x: float
    y: float
    visibility: Literal[0, 1, 2] = Field(description="0 라벨 없음, 1 가려짐, 2 보임")
    confidence: Confidence | None = None


# 한 시각의 키포인트 묶음. points 개수는 골격 크기(`SKELETON_SIZES`)와 같아야 한다.
class KeypointFrame(Contract):
    t_ms: Ms
    points: tuple[Keypoint, ...]


# 골격 이름 → 키포인트 수.
# hand21=MediaPipe 손, coco17=COCO 전신, wholebody133=COCO-WholeBody(RTMPose).
SKELETON_SIZES: dict[str, int] = {"hand21": 21, "coco17": 17, "wholebody133": 133}


# 박스 트랙 (객체·도구 추적). entity_id: 개체 ID. class_id: 온톨로지 객체 클래스.
# keyframes: 1개 이상.
class BoxTrackPayload(Contract):
    kind: Literal["box_track"] = "box_track"
    entity_id: Identifier
    class_id: OntologyId
    keyframes: tuple[BoxKeyframe, ...] = Field(min_length=1)


# 마스크 트랙. part가 있으면 도구의 작용부·파지부 등 부분 마스크다
# (그 클래스의 tool_parts 안이어야 한다, `validation.check_label`).
class MaskTrackPayload(Contract):
    kind: Literal["mask_track"] = "mask_track"
    entity_id: Identifier
    class_id: OntologyId
    part: OntologyId | None = Field(default=None, description="도구 작용부·파지부 등 부분 마스크")
    keyframes: tuple[MaskKeyframe, ...] = Field(min_length=1)


# 키포인트 트랙 (손·전신 포즈). skeleton: 골격. hand: hand21이면 필수, 전신이면 None.
class KeypointTrackPayload(Contract):
    kind: Literal["keypoint_track"] = "keypoint_track"
    entity_id: Identifier
    skeleton: Literal["hand21", "coco17", "wholebody133"]
    hand: Hand | None = None
    keyframes: tuple[KeypointFrame, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> KeypointTrackPayload:
        """프레임별 키포인트 수와 손 골격의 hand 필드를 검사한다. Raises: ValueError."""
        n = SKELETON_SIZES[self.skeleton]
        if any(len(f.points) != n for f in self.keyframes):
            raise ValueError(f"{self.skeleton} 골격은 프레임마다 키포인트 {n}개가 필요합니다")
        if self.skeleton == "hand21" and self.hand is None:
            raise ValueError("손 골격에는 hand(left/right)가 필요합니다")
        return self


# 블러 트랙 (프라이버시 게이트 출력). target: 프라이버시 대상 ID. keyframes: 가릴 박스.
# 어떤 내보내기에도 넣지 않는다 (`export.ExportedLabel`). 블러 검수는 원본 접근 권한자만 한다.
class BlurTrackPayload(Contract):
    kind: Literal["blur_track"] = "blur_track"
    target: OntologyId = Field(description="프라이버시 사전의 대상 ID")
    keyframes: tuple[BoxKeyframe, ...] = Field(min_length=1)


# 3D 궤적 표본. t_ms: 그 영상 스트림 PTS ms. x, y, z: 위치 (미터, frame 좌표계).
# qw, qx, qy, qz: 방향 쿼터니언 (카메라 자세처럼 회전이 있을 때만, 없으면 모두 None).
class Trajectory3DSample(Contract):
    t_ms: Ms
    x: float
    y: float
    z: float
    qw: float | None = None
    qx: float | None = None
    qy: float | None = None
    qz: float | None = None


class Trajectory3DPayload(Contract):
    """3D 궤적. 카메라 자세는 entity_id="camera"에 회전(쿼터니언)까지 기록한다."""

    # entity_id: 개체 ID ("camera"면 카메라 자세). part: 부분 ID (작용부, 표면 꼭짓점, 손 관절 등,
    #   `Ontology.known_parts` 안). None이면 개체 전체(중심).
    # frame: 좌표계. source_3d: 3D를 얻은 방법. samples: 1개 이상.

    kind: Literal["trajectory3d"] = "trajectory3d"
    entity_id: Identifier
    part: OntologyId | None = None
    frame: CoordinateFrame
    source_3d: Source3D
    samples: tuple[Trajectory3DSample, ...] = Field(min_length=1)


# ---------------------------------------------------------------- 시간 구간


class HandStatePayload(Contract):
    """손 상태 구간. 값이 바뀌는 시각(변화점)마다 새 구간을 만든다."""

    # hand: 어느 손. contact_target_kind: 접촉 대상 종류 (온톨로지 contact_target_kinds).
    # target_id: 접촉한 개체 ID (object·tool·fixed_surface면 필수, none이면 비워야 한다).
    # body_part: 대상자 신체 부위 (person이면 필수). grasp_type: 파지 유형 (none이면 비워야 한다).
    # role: 손 역할 (active 주동 / assist 보조 / inactive).

    kind: Literal["hand_state"] = "hand_state"
    hand: Hand
    contact_target_kind: OntologyId
    target_id: Identifier | None = None
    body_part: OntologyId | None = None
    grasp_type: OntologyId | None = None
    role: OntologyId

    @model_validator(mode="after")
    def _check(self) -> HandStatePayload:
        """접촉 대상 종류별 필수·금지 필드를 검사한다 (none/object/tool/fixed_surface/person).

        self(자기 몸) 등 그 밖의 종류는 추가 제약이 없다. Raises: ValueError.
        """
        k = self.contact_target_kind
        if k == "none" and (self.target_id or self.body_part or self.grasp_type):
            raise ValueError("접촉 대상이 없으면 target_id·body_part·grasp_type을 비워야 합니다")
        if k in ("object", "tool", "fixed_surface") and not self.target_id:
            raise ValueError(f"접촉 대상 {k}에는 target_id가 필요합니다")
        if k == "person" and not self.body_part:
            raise ValueError("대상자 접촉에는 body_part가 필요합니다")
        return self


class ActionPayload(Contract):
    """원시 동작 하나 (손별). 양손 행동은 두 손 트랙에 같은 action_id로 기록한다."""

    # action_id: 행동 ID (양손이면 두 레코드가 공유). hand: 어느 손. verb: 원시 동작 동사 ID.
    # target_id / target_body_part / tool_id: 대상 개체, 대상자 신체 부위, 쓴 도구 개체 (선택).
    # direction: 자유 텍스트 방향 (선택).
    # 시각 4개 (마스터 ms): t_approach_ms(접근 시작 = 라벨 t_start_ms) <= t_contact_start_ms <=
    #   t_contact_end_ms <= t_end_ms(= 라벨 t_end_ms).
    #   접촉 시각은 비접촉 동작·연속 접촉 중간이면 None.
    # contact_held: Field description 참고.
    # pre_state / post_state: 행동 전·후 대상 상태 (상태 속성 ID → 값 ID).

    kind: Literal["action"] = "action"
    action_id: Identifier
    hand: Hand
    verb: OntologyId
    target_id: Identifier | None = None
    target_body_part: OntologyId | None = None
    tool_id: Identifier | None = None
    direction: str | None = None
    t_approach_ms: Ms
    t_contact_start_ms: Ms | None = None
    t_contact_end_ms: Ms | None = None
    t_end_ms: Ms
    contact_held: bool = Field(
        default=False, description="연속 접촉 묶음의 중간·끝 행동 (접촉 시작이 앞 행동에 있음)"
    )
    pre_state: dict[OntologyId, OntologyId] = Field(default_factory=dict)
    post_state: dict[OntologyId, OntologyId] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> ActionPayload:
        """접근 <= 접촉 시작 <= 접촉 종료 <= 종료 순서를 검사한다 (None은 건너뜀).

        Raises:
            ValueError: 순서 위반.
        """
        start, end = self.t_contact_start_ms, self.t_contact_end_ms
        if start is not None and end is not None and start > end:
            raise ValueError("접촉 시작이 접촉 종료보다 늦습니다")
        times = [t for t in (self.t_approach_ms, start, end, self.t_end_ms) if t is not None]
        if times != sorted(times):
            raise ValueError("접근 시작 ≤ 접촉 시작 ≤ 접촉 종료 ≤ 종료 순서여야 합니다")
        return self


# 상위 구간 수준.
class SegmentLevel(StrEnum):
    SKILL = "skill"  # 기술 (ref_id = skill·human_skill 동사)
    SUBSTEP = "substep"  # 작업의 하위 단계 (ref_id = 하위 단계 ID)
    TASK = "task"  # 작업 (ref_id = 작업 ID)


class SegmentPayload(Contract):
    """원시 동작을 묶는 상위 구간 (기술, 하위 단계, 작업)."""

    # segment_id: 구간 ID (description 라벨이 참조). level / ref_id: 수준과 참조 사전 ID.
    # child_ids: 묶인 하위 구간·행동의 ID (action_id 또는 segment_id).

    kind: Literal["segment"] = "segment"
    segment_id: Identifier
    level: SegmentLevel
    ref_id: OntologyId = Field(description="기술은 동사 ID, 하위 단계·작업은 작업 사전 ID")
    child_ids: tuple[Identifier, ...] = ()


# 공백 구간 (행동 사이를 채운다. 타임라인에 빈 시간을 두지 않는다).
# hand: 어느 손 (양손 공통이면 None). gap_type: idle(대기) / unknown(미상) / out_of_scope(범위 외).
class GapPayload(Contract):
    kind: Literal["gap"] = "gap"
    hand: Hand | None = None
    gap_type: OntologyId


class ObjectStatePayload(Contract):
    """객체 상태가 유지되는 구간. 상태 전이는 같은 객체·속성의 연속 구간에서 도출한다."""

    # entity_id: 개체 ID. class_id: 객체 클래스. attribute: 상태 속성 (그 클래스의 states 안).
    # value: 그 속성의 값.

    kind: Literal["object_state"] = "object_state"
    entity_id: Identifier
    class_id: OntologyId
    attribute: OntologyId
    value: OntologyId


# 표면 커버리지 (`dlp relations run`). surface_id: 표면 개체. tool_id: 처리한 도구 (선택).
# ratio: 라벨 구간 동안 도구 작용부가 지나간 표면 면적 비율 (0~1).
class CoveragePayload(Contract):
    kind: Literal["coverage"] = "coverage"
    surface_id: Identifier
    tool_id: Identifier | None = None
    ratio: float = Field(ge=0.0, le=1.0)


# 이벤트 (실패·안전·감염 관리). event_type: 이벤트 ID. severity: 심각도 1~3 (사전이 요구하면 필수).
# related_action_id / related_entity_id: 관련 행동·개체
# (사전 requires_action / requires_object면 필수).
# 시점 이벤트(form=point)는 라벨 t_start_ms == t_end_ms.
class EventPayload(Contract):
    kind: Literal["event"] = "event"
    event_type: OntologyId
    severity: Literal[1, 2, 3] | None = None
    related_action_id: Identifier | None = None
    related_entity_id: Identifier | None = None


# 관계 술어 (주체 -술어-> 대상).
class RelationPredicate(StrEnum):
    GRASP = "grasp"
    CONTACT = "contact"
    SUPPORT = "support"
    CONTAIN = "contain"


# 관계 (`dlp relations run`의 규칙 엔진 출력 등). subject_id/object_id: 주체·대상 개체.
# subject_part/object_part: 부분 ID (예: 걸레의 cloth_face).
# derived_by: 도출 규칙 ID (사람이 그렸으면 None).
class RelationPayload(Contract):
    kind: Literal["relation"] = "relation"
    subject_id: Identifier
    subject_part: OntologyId | None = None
    predicate: RelationPredicate
    object_id: Identifier
    object_part: OntologyId | None = None
    derived_by: str | None = Field(default=None, description="관계를 도출한 규칙 ID")


# 자연어 설명 (VLM 또는 사람). segment_id: 설명하는 상위 구간·행동 ID. text: 설명.
# language: 언어 코드 (기본 ko).
class DescriptionPayload(Contract):
    kind: Literal["description"] = "description"
    segment_id: Identifier
    text: str = Field(min_length=1)
    language: str = "ko"


# 페이로드 판별 공용체: JSON의 "kind" 값으로 어느 타입인지 정한다. 새 종류를 더하면 여기,
# SPATIAL_KINDS(공간이면), history.label_class, episode.entity_refs, validation.check_label,
# migration._FIELDS를 함께 본다.
LabelPayload = Annotated[
    BoxTrackPayload
    | MaskTrackPayload
    | KeypointTrackPayload
    | BlurTrackPayload
    | Trajectory3DPayload
    | HandStatePayload
    | ActionPayload
    | SegmentPayload
    | GapPayload
    | ObjectStatePayload
    | CoveragePayload
    | EventPayload
    | RelationPayload
    | DescriptionPayload,
    Field(discriminator="kind"),
]

# 공간 라벨: 키프레임 시각이 그 스트림 영상의 PTS 시각이고 stream_id가 필수다 (ADR 0019).
# 3D 궤적도 특정 영상(보통 바디캠)에서 들어 올린 것이라 공간 라벨이다 (ADR 0022).
SPATIAL_KINDS = frozenset(
    {"box_track", "mask_track", "keypoint_track", "blur_track", "trajectory3d"}
)


# ---------------------------------------------------------------- 라벨 레코드


# 라벨 레코드 하나 (DB label_records 한 행).
#   label_id: 라벨 ID. 모델 출처는 `version_tag(모델 버전)`을 넣어 버전마다 다르게 만든다.
#   session_id: 세션. stream_id: Field description 참고.
#   t_start_ms / t_end_ms: 라벨 구간 (시작 <= 끝). 공간 라벨은 스트림 PTS ms, 시간 구간 라벨은
#     마스터 ms.
#   ontology_version: 따르는 온톨로지 버전 (DB FK).
#   provenance / evidence / confidence: 출처, 근거, 신뢰도 (모델 출처면 confidence 필수).
#   verification: 검수 상태 (DB에서 이것만 바꿀 수 있다).
#   parent_label_id: 고친(또는 지운) 이전 레코드. 처음 레코드면 None.
#   retracted / seeded_error / measurement: Field description 참고.
#     운영 라벨 판정은 `episode.current_labels`.
#   created_at: 레코드 생성 시각 (시간대 필수).
#   payload: 종류별 내용.
class LabelRecord(Contract):
    label_id: Identifier
    session_id: Identifier
    stream_id: Identifier | None = Field(
        default=None, description="공간 라벨이 그려진 스트림. 공간 라벨에는 필수"
    )
    t_start_ms: Ms
    t_end_ms: Ms
    ontology_version: SemVer
    provenance: Provenance
    evidence: Evidence = Evidence.OBSERVED
    confidence: Confidence | None = None
    verification: Verification = Verification()
    parent_label_id: Identifier | None = None
    retracted: bool = Field(default=False, description="parent 라벨을 삭제(오탐 제거)하는 레코드")
    seeded_error: bool = Field(default=False, description="오류 삽입 과제용. 학습에서 제외")
    measurement: Literal["blind", "double"] | None = Field(
        default=None,
        description="측정용 레코드 (블라인드 과제, 이중 라벨링의 두 번째 라벨). 운영 라벨이 아니다",
    )
    created_at: AwareDatetime
    payload: LabelPayload

    @property
    def kind(self) -> str:
        """페이로드 종류 이름 (DB label_records.kind 열)."""
        return self.payload.kind

    @model_validator(mode="after")
    def _check(self) -> LabelRecord:
        """레코드 수준 규칙을 검사한다.

        - t_start_ms <= t_end_ms
        - retracted면 parent_label_id 필수, 자기 자신을 부모로 가리키지 않음
        - 공간 라벨은 stream_id 필수
        - 모델 출처는 confidence 필수
        - 종류별 시각 규칙 (`_check_times`)

        Raises:
            ValueError: 위반 (Pydantic이 ValidationError로 감싼다).
        """
        if self.t_start_ms > self.t_end_ms:
            raise ValueError("t_start_ms가 t_end_ms보다 늦습니다")
        if self.retracted and self.parent_label_id is None:
            raise ValueError("retracted 레코드에는 parent_label_id가 필요합니다")
        if self.parent_label_id == self.label_id:
            raise ValueError("라벨이 자기 자신을 parent로 가리킵니다")
        p = self.payload
        if p.kind in SPATIAL_KINDS and self.stream_id is None:
            raise ValueError(f"{p.kind} 라벨에는 stream_id가 필요합니다")
        if self.provenance.source is Source.MODEL and self.confidence is None:
            raise ValueError("모델 출처 라벨에는 confidence가 필요합니다")
        self._check_times(p)
        return self

    def _check_times(self, p: LabelPayload) -> None:
        """페이로드 시각이 라벨 구간과 맞는지 검사한다.

        - 트랙·궤적: 키프레임(표본) 시각이 중복 없이 오름차순이고 [t_start_ms, t_end_ms] 안.
        - 행동: (접근 시작, 종료)가 라벨 (t_start_ms, t_end_ms)와 같아야 한다.

        Raises:
            ValueError: 위반.
        """
        stamps: list[int] = []
        match p:
            case BoxTrackPayload() | MaskTrackPayload() | BlurTrackPayload():
                stamps = [k.t_ms for k in p.keyframes]
            case KeypointTrackPayload():
                stamps = [k.t_ms for k in p.keyframes]
            case Trajectory3DPayload():
                stamps = [s.t_ms for s in p.samples]
            case ActionPayload():
                if (p.t_approach_ms, p.t_end_ms) != (self.t_start_ms, self.t_end_ms):
                    raise ValueError("행동 라벨의 구간은 접근 시작~종료와 같아야 합니다")
            case _:
                pass
        if stamps:
            if stamps != sorted(stamps) or len(stamps) != len(set(stamps)):
                raise ValueError("키프레임 시각은 중복 없이 오름차순이어야 합니다")
            if stamps[0] < self.t_start_ms or stamps[-1] > self.t_end_ms:
                raise ValueError("키프레임 시각이 라벨 구간을 벗어났습니다")
