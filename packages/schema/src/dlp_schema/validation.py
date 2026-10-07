"""라벨을 온톨로지에 대조해 검증한다. 스키마 검증(타입)과 별개로, 값이 사전 안에 있는지 본다.

역할
    `LabelRecord`의 Pydantic 검증은 형식(타입·범위·시간 순서)만 본다. 여기서는 그 값이 지정한
    온톨로지 버전의 사전 안에 있는지, 사전이 정한 규칙(원시 동작만 행동 라벨, 비접촉 동사, 이벤트
    필수 필드, 시점 이벤트 등)을 지키는지 본다 (WP1).

주요 이름
    - `OntologyViolationError`: 위반이 있을 때 `validate_label`이 던진다.
    - `check_label`: 위반 메시지 목록 (예외 없음).
    - `validate_label`: 위반이 있으면 예외.

쓰는 곳
    현재는 각 패키지의 테스트(프리라벨·행동·픽스처 출력이 사전을 지키는지)에서만 부른다. 운영 경로
    (라벨 저장, 검수 결과 수집)는 이 검사를 거치지 않는다. 행동 단계의 VLM 출력 검사는
    `dlp_actions`가 따로 한다.

주의
    - 개체 ID(`entity_id`, `target_id` 등) 참조는 여기서 보지 않는다. 에피소드 그래프
      (`episode.EpisodeGraph`) 검증이 본다.
    - 라벨의 `ontology_version`이 사전 버전과 다르면 그 자체로 위반이다.
"""

from __future__ import annotations

from collections.abc import Mapping

from dlp_schema.labels import (
    ActionPayload,
    BlurTrackPayload,
    BoxTrackPayload,
    CoveragePayload,
    EventPayload,
    GapPayload,
    HandStatePayload,
    LabelRecord,
    MaskTrackPayload,
    ObjectStatePayload,
    RelationPayload,
    SegmentLevel,
    SegmentPayload,
    Trajectory3DPayload,
)
from dlp_schema.ontology import Ontology, VerbLevel


class OntologyViolationError(ValueError):
    """라벨이 온톨로지 사전을 어겼다.

    Attributes:
        label_id: 위반한 라벨 ID.
        problems: 위반 메시지 목록 (`check_label` 결과 그대로).
    """

    def __init__(self, label_id: str, problems: list[str]) -> None:
        """메시지는 `"<label_id>: <문제1>; <문제2>"` 형태다."""
        super().__init__(f"{label_id}: " + "; ".join(problems))
        self.label_id = label_id
        self.problems = problems


def check_label(label: LabelRecord, ontology: Ontology) -> list[str]:
    """위반 사항 목록을 돌려준다. 비어 있으면 통과.

    Args:
        label: 검사할 라벨 (Pydantic 검증은 이미 통과한 것).
        ontology: 대조할 온톨로지 사전.

    Returns:
        사람이 읽는 한국어 위반 메시지 목록. 예외를 던지지 않는다.

    종류별 검사:
        - box/mask_track: 객체 클래스가 사전에 있는지, mask의 part가 그 클래스 도구의
          작용부·파지부인지.
        - blur_track: 프라이버시 대상이 사전에 있는지.
        - hand_state: 접촉 대상 종류·신체 부위·파지 유형·손 역할.
        - action: 동사가 있고 원시 동작(primitive)인지, 비접촉 동사에 접촉 시각이 없는지, 대상 신체
          부위, pre/post 상태 속성과 값.
        - segment: skill은 기술 동사(primitive가 아닌 동사),
          task는 작업, substep은 어느 작업의 하위 단계.
        - gap: 사이 구간 유형. object_state: 클래스, 그 클래스가 가진 속성인지, 속성 값.
        - event: 이벤트 유형, 사전이 요구하는 심각도·관련
          행동·관련 개체, 시점 이벤트는 t_start == t_end.
        - trajectory3d / relation: part ID가 알려진 부분(`Ontology.known_parts`)인지.
        - coverage / description / keypoint_track: 사전 대조 없음.
    """
    o = ontology
    problems: list[str] = []
    if label.ontology_version != o.version:
        problems.append(f"온톨로지 버전 불일치: 라벨 {label.ontology_version}, 사전 {o.version}")

    def need(value: str | None, table: Mapping[str, object], what: str) -> None:
        """값이 있으면(None이 아니면) 사전 `table`의 키여야 한다. 아니면 problems에 추가."""
        if value is not None and value not in table:
            problems.append(f"알 수 없는 {what}: {value}")

    p = label.payload
    match p:
        case BoxTrackPayload() | MaskTrackPayload():
            need(p.class_id, o.objects, "객체 클래스")
            # 부분 마스크(part)는 mask_track에만 있다. 그 클래스가
            # 도구가 아니면 tool_parts가 비어 위반이다.
            part = p.part if isinstance(p, MaskTrackPayload) else None
            if part is not None and part not in o.tool_parts(p.class_id):
                problems.append(f"{p.class_id}에 없는 부분: {part}")
        case BlurTrackPayload():
            need(p.target, o.privacy_targets, "프라이버시 대상")
        case HandStatePayload():
            need(p.contact_target_kind, o.contact_target_kinds, "접촉 대상 종류")
            need(p.body_part, o.body_parts, "신체 부위")
            need(p.grasp_type, o.grasp_types, "파지 유형")
            need(p.role, o.hand_roles, "손 역할")
        case ActionPayload():
            verb = o.verbs.get(p.verb)
            if verb is None:
                problems.append(f"알 수 없는 동사: {p.verb}")
            else:
                if verb.level is not VerbLevel.PRIMITIVE:
                    problems.append(f"행동 라벨은 원시 동작이어야 합니다: {p.verb}({verb.level})")
                if verb.contactless and p.t_contact_start_ms is not None:
                    problems.append(f"비접촉 동사 {p.verb}에 접촉 시각이 있습니다")
            need(p.target_body_part, o.body_parts, "신체 부위")
            for attr, value in (*p.pre_state.items(), *p.post_state.items()):
                _check_state(o, attr, value, problems)
        case SegmentPayload():
            if p.level is SegmentLevel.SKILL:
                verb = o.verbs.get(p.ref_id)
                if verb is None or verb.level is VerbLevel.PRIMITIVE:
                    problems.append(f"기술 구간의 ref_id는 기술 동사여야 합니다: {p.ref_id}")
            elif p.level is SegmentLevel.TASK:
                need(p.ref_id, o.tasks, "작업")
            # 남은 경우는 SUBSTEP: 어느 작업의 substeps에든 있으면 된다
            elif o.substep_task(p.ref_id) is None:
                problems.append(f"알 수 없는 하위 단계: {p.ref_id}")
        case GapPayload():
            need(p.gap_type, o.gap_types, "사이 구간 유형")
        case ObjectStatePayload():
            need(p.class_id, o.objects, "객체 클래스")
            obj = o.objects.get(p.class_id)
            if obj is not None and p.attribute not in obj.states:
                problems.append(f"{p.class_id}에 없는 상태 속성: {p.attribute}")
            _check_state(o, p.attribute, p.value, problems)
        case EventPayload():
            ev = o.events.get(p.event_type)
            if ev is None:
                problems.append(f"알 수 없는 이벤트: {p.event_type}")
            else:
                if ev.severity and p.severity is None:
                    problems.append(f"이벤트 {p.event_type}에는 심각도가 필요합니다")
                if ev.requires_action and p.related_action_id is None:
                    problems.append(f"이벤트 {p.event_type}에는 related_action_id가 필요합니다")
                if ev.requires_object and p.related_entity_id is None:
                    problems.append(f"이벤트 {p.event_type}에는 related_entity_id가 필요합니다")
                if ev.form == "point" and label.t_start_ms != label.t_end_ms:
                    problems.append(f"시점 이벤트 {p.event_type}는 t_start와 t_end가 같아야 합니다")
        case Trajectory3DPayload():
            # 개체 ID 참조는 에피소드 그래프에서 검증한다. 부분 ID는 사전 안에 있어야 한다.
            if p.part is not None and p.part not in o.known_parts():
                problems.append(f"알 수 없는 부분: {p.part}")
        case RelationPayload():
            for part in (p.subject_part, p.object_part):
                if part is not None and part not in o.known_parts():
                    problems.append(f"알 수 없는 부분: {part}")
        case CoveragePayload():
            pass  # 개체 ID 참조는 에피소드 그래프에서 검증한다
        case _:
            pass
    return problems


def _check_state(o: Ontology, attr: str, value: str, problems: list[str]) -> None:
    """상태 속성 `attr`이 사전에 있고 `value`가 그 속성의 허용 값인지 본다. 위반은 problems에 추가.

    객체 클래스가 그 속성을 갖는지는 보지 않는다 (object_state는 호출 전에 따로 본다. action의
    pre/post_state는 대상 클래스를 모르므로 보지 않는다).
    """
    sa = o.state_attributes.get(attr)
    if sa is None:
        problems.append(f"알 수 없는 상태 속성: {attr}")
    elif value not in sa.values:
        problems.append(f"상태 속성 {attr}에 없는 값: {value}")


def validate_label(label: LabelRecord, ontology: Ontology) -> None:
    """`check_label`과 같지만 위반이 있으면 예외를 던진다.

    Raises:
        OntologyViolationError: 위반이 하나라도 있을 때.
    """
    problems = check_label(label, ontology)
    if problems:
        raise OntologyViolationError(label.label_id, problems)
