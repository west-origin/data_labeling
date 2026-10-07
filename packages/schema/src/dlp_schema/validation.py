"""라벨을 온톨로지에 대조해 검증한다. 스키마 검증(타입)과 별개로, 값이 사전 안에 있는지 본다."""

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
    def __init__(self, label_id: str, problems: list[str]) -> None:
        super().__init__(f"{label_id}: " + "; ".join(problems))
        self.label_id = label_id
        self.problems = problems


def check_label(label: LabelRecord, ontology: Ontology) -> list[str]:
    """위반 사항 목록을 돌려준다. 비어 있으면 통과."""
    o = ontology
    problems: list[str] = []
    if label.ontology_version != o.version:
        problems.append(f"온톨로지 버전 불일치: 라벨 {label.ontology_version}, 사전 {o.version}")

    def need(value: str | None, table: Mapping[str, object], what: str) -> None:
        if value is not None and value not in table:
            problems.append(f"알 수 없는 {what}: {value}")

    p = label.payload
    match p:
        case BoxTrackPayload() | MaskTrackPayload():
            need(p.class_id, o.objects, "객체 클래스")
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
    sa = o.state_attributes.get(attr)
    if sa is None:
        problems.append(f"알 수 없는 상태 속성: {attr}")
    elif value not in sa.values:
        problems.append(f"상태 속성 {attr}에 없는 값: {value}")


def validate_label(label: LabelRecord, ontology: Ontology) -> None:
    problems = check_label(label, ontology)
    if problems:
        raise OntologyViolationError(label.label_id, problems)
