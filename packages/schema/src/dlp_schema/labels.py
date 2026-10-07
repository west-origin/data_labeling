"""라벨 레코드와 종류별 페이로드.

라벨은 덮어쓰지 않는다. 수정은 새 레코드를 만들고 parent_label_id로 이전 레코드를 가리킨다.
삭제(오탐 제거)는 retracted=True인 새 레코드로 표현한다.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, model_validator

from dlp_schema.common import Confidence, Contract, Identifier, Ms, OntologyId, SemVer

# ---------------------------------------------------------------- 공통 메타데이터


class Source(StrEnum):
    HUMAN = "human"
    MODEL = "model"
    SENSOR = "sensor"


class Evidence(StrEnum):
    OBSERVED = "observed"
    INFERRED = "inferred"
    UNKNOWN = "unknown"


class VerificationState(StrEnum):
    UNREVIEWED = "unreviewed"
    SAMPLE_VERIFIED = "sample_verified"
    HUMAN_APPROVED = "human_approved"
    HUMAN_CORRECTED = "human_corrected"


class CoordinateFrame(StrEnum):
    CAMERA = "camera"
    BODY = "body"
    WORLD = "world"


class Source3D(StrEnum):
    MONO_DEPTH = "mono_depth"
    VISUAL_SLAM = "visual_slam"
    VISUAL_INERTIAL_SLAM = "visual_inertial_slam"
    MULTI_VIEW = "multi_view"


class Hand(StrEnum):
    LEFT = "left"
    RIGHT = "right"


class Provenance(Contract):
    source: Source
    model_version: str | None = None
    sensor_id: str | None = None

    @model_validator(mode="after")
    def _check(self) -> Provenance:
        if self.source is Source.MODEL and not self.model_version:
            raise ValueError("모델 출처 라벨에는 model_version이 필요합니다")
        if self.source is Source.SENSOR and not self.sensor_id:
            raise ValueError("센서 출처 라벨에는 sensor_id가 필요합니다")
        return self


class Verification(Contract):
    state: VerificationState = VerificationState.UNREVIEWED
    reviewer_id: Identifier | None = None
    reviewed_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def _check(self) -> Verification:
        if self.state is not VerificationState.UNREVIEWED and (
            self.reviewer_id is None or self.reviewed_at is None
        ):
            raise ValueError("검수된 라벨에는 reviewer_id와 reviewed_at이 필요합니다")
        return self


# ---------------------------------------------------------------- 공간 트랙


class BoxKeyframe(Contract):
    t_ms: Ms
    x: float
    y: float
    w: float = Field(ge=0)
    h: float = Field(ge=0)
    outside: bool = False


class Rle(Contract):
    """COCO 형식 RLE 마스크."""

    size: tuple[int, int]
    counts: str


class MaskKeyframe(Contract):
    t_ms: Ms
    rle: Rle | None = None
    outside: bool = False

    @model_validator(mode="after")
    def _check(self) -> MaskKeyframe:
        if not self.outside and self.rle is None:
            raise ValueError("보이는 마스크 키프레임에는 rle가 필요합니다")
        return self


class Keypoint(Contract):
    x: float
    y: float
    visibility: Literal[0, 1, 2] = Field(description="0 라벨 없음, 1 가려짐, 2 보임")
    confidence: Confidence | None = None


class KeypointFrame(Contract):
    t_ms: Ms
    points: tuple[Keypoint, ...]


SKELETON_SIZES: dict[str, int] = {"hand21": 21, "coco17": 17, "wholebody133": 133}


class BoxTrackPayload(Contract):
    kind: Literal["box_track"] = "box_track"
    entity_id: Identifier
    class_id: OntologyId
    keyframes: tuple[BoxKeyframe, ...] = Field(min_length=1)


class MaskTrackPayload(Contract):
    kind: Literal["mask_track"] = "mask_track"
    entity_id: Identifier
    class_id: OntologyId
    part: OntologyId | None = Field(default=None, description="도구 작용부·파지부 등 부분 마스크")
    keyframes: tuple[MaskKeyframe, ...] = Field(min_length=1)


class KeypointTrackPayload(Contract):
    kind: Literal["keypoint_track"] = "keypoint_track"
    entity_id: Identifier
    skeleton: Literal["hand21", "coco17", "wholebody133"]
    hand: Hand | None = None
    keyframes: tuple[KeypointFrame, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> KeypointTrackPayload:
        n = SKELETON_SIZES[self.skeleton]
        if any(len(f.points) != n for f in self.keyframes):
            raise ValueError(f"{self.skeleton} 골격은 프레임마다 키포인트 {n}개가 필요합니다")
        if self.skeleton == "hand21" and self.hand is None:
            raise ValueError("손 골격에는 hand(left/right)가 필요합니다")
        return self


class BlurTrackPayload(Contract):
    kind: Literal["blur_track"] = "blur_track"
    target: OntologyId = Field(description="프라이버시 사전의 대상 ID")
    keyframes: tuple[BoxKeyframe, ...] = Field(min_length=1)


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

    kind: Literal["trajectory3d"] = "trajectory3d"
    entity_id: Identifier
    part: OntologyId | None = None
    frame: CoordinateFrame
    source_3d: Source3D
    samples: tuple[Trajectory3DSample, ...] = Field(min_length=1)


# ---------------------------------------------------------------- 시간 구간


class HandStatePayload(Contract):
    """손 상태 구간. 값이 바뀌는 시각(변화점)마다 새 구간을 만든다."""

    kind: Literal["hand_state"] = "hand_state"
    hand: Hand
    contact_target_kind: OntologyId
    target_id: Identifier | None = None
    body_part: OntologyId | None = None
    grasp_type: OntologyId | None = None
    role: OntologyId

    @model_validator(mode="after")
    def _check(self) -> HandStatePayload:
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
        start, end = self.t_contact_start_ms, self.t_contact_end_ms
        if start is not None and end is not None and start > end:
            raise ValueError("접촉 시작이 접촉 종료보다 늦습니다")
        times = [t for t in (self.t_approach_ms, start, end, self.t_end_ms) if t is not None]
        if times != sorted(times):
            raise ValueError("접근 시작 ≤ 접촉 시작 ≤ 접촉 종료 ≤ 종료 순서여야 합니다")
        return self


class SegmentLevel(StrEnum):
    SKILL = "skill"
    SUBSTEP = "substep"
    TASK = "task"


class SegmentPayload(Contract):
    """원시 동작을 묶는 상위 구간 (기술, 하위 단계, 작업)."""

    kind: Literal["segment"] = "segment"
    segment_id: Identifier
    level: SegmentLevel
    ref_id: OntologyId = Field(description="기술은 동사 ID, 하위 단계·작업은 작업 사전 ID")
    child_ids: tuple[Identifier, ...] = ()


class GapPayload(Contract):
    kind: Literal["gap"] = "gap"
    hand: Hand | None = None
    gap_type: OntologyId


class ObjectStatePayload(Contract):
    """객체 상태가 유지되는 구간. 상태 전이는 같은 객체·속성의 연속 구간에서 도출한다."""

    kind: Literal["object_state"] = "object_state"
    entity_id: Identifier
    class_id: OntologyId
    attribute: OntologyId
    value: OntologyId


class CoveragePayload(Contract):
    kind: Literal["coverage"] = "coverage"
    surface_id: Identifier
    tool_id: Identifier | None = None
    ratio: float = Field(ge=0.0, le=1.0)


class EventPayload(Contract):
    kind: Literal["event"] = "event"
    event_type: OntologyId
    severity: Literal[1, 2, 3] | None = None
    related_action_id: Identifier | None = None
    related_entity_id: Identifier | None = None


class RelationPredicate(StrEnum):
    GRASP = "grasp"
    CONTACT = "contact"
    SUPPORT = "support"
    CONTAIN = "contain"


class RelationPayload(Contract):
    kind: Literal["relation"] = "relation"
    subject_id: Identifier
    subject_part: OntologyId | None = None
    predicate: RelationPredicate
    object_id: Identifier
    object_part: OntologyId | None = None
    derived_by: str | None = Field(default=None, description="관계를 도출한 규칙 ID")


class DescriptionPayload(Contract):
    kind: Literal["description"] = "description"
    segment_id: Identifier
    text: str = Field(min_length=1)
    language: str = "ko"


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

SPATIAL_KINDS = frozenset({"box_track", "mask_track", "keypoint_track", "blur_track"})


# ---------------------------------------------------------------- 라벨 레코드


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
        return self.payload.kind

    @model_validator(mode="after")
    def _check(self) -> LabelRecord:
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
