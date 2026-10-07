"""규칙 엔진: 입력 구간(손 상태, 도구-표면 접촉)에 YAML 규칙을 적용해 관계 구간을 만든다.

규칙 ID는 관계의 derived_by에 남는다. 같은 관계(주어·술어·목적어·부분·규칙)가 merge_gap_ms보다 짧게
끊기면 하나로 잇는다.

규칙 형식은 `relations.yaml rules` 주석 참고. 한 입력에 맞는 규칙이 여럿이면 모두 관계를 낸다.
"""

from __future__ import annotations

import string
from dataclasses import dataclass

from dlp_relations.contact import SurfaceContact
from dlp_relations.policy import RelationsPolicy, RelationTemplate, Rule
from dlp_schema.labels import HandStatePayload, LabelRecord, RelationPayload

# 규칙 필드 이름 → 값 (None은 값 없음)
Fields = dict[str, str | None]


@dataclass(frozen=True)
class RelationDraft:
    """관계 초안: 페이로드, 구간(마스터 ms), 신뢰도."""

    payload: RelationPayload
    start_ms: int
    end_ms: int
    confidence: float


def hand_state_fields(p: HandStatePayload, unresolved: tuple[str, ...] = ()) -> Fields:
    """unresolved: 대상을 모른다는 표시 ID. target_id가 이 값이면 없음(None)으로 본다.

    손 상태 페이로드 → 규칙 필드 (hand, contact_target_kind, target_id, body_part, grasp_type,
    role).
    """
    return {
        "hand": p.hand.value,
        "contact_target_kind": p.contact_target_kind,
        "target_id": None if p.target_id in unresolved else p.target_id,
        "body_part": p.body_part,
        "grasp_type": p.grasp_type,
        "role": p.role,
    }


def contact_fields(c: SurfaceContact) -> Fields:
    """도구-표면 접촉 → 규칙 필드 (tool_id, tool_part, surface_id)."""
    return {"tool_id": c.tool_id, "tool_part": c.tool_part, "surface_id": c.surface_id}


def _matches(rule: Rule, fields: Fields) -> bool:
    """규칙의 when 조건이 모두 맞는가 (없는 필드는 None으로 본다)."""
    return all(fields.get(name) in allowed for name, allowed in rule.when.items())


def _render(template: str | None, fields: Fields) -> tuple[bool, str | None]:
    """(성공, 값). 자리에 넣을 값이 없으면 실패.

    틀이 None이면 (성공, None). 틀의 `{이름}` 자리 중 하나라도 필드 값이 None이면 (실패, None).
    """
    if template is None:
        return True, None
    # 틀의 `{이름}` 자리 목록
    names = [n for _, n, _, _ in string.Formatter().parse(template) if n]
    if any(fields.get(n) is None for n in names):
        return False, None
    return True, template.format(**{n: fields[n] for n in names})


def _emit(rule: Rule, fields: Fields) -> RelationPayload | None:
    """규칙 틀에 필드를 넣어 관계 페이로드를 만든다. 자리가 비거나 주어·목적어가 없으면 None."""
    t: RelationTemplate = rule.emit
    values: dict[str, str | None] = {}
    for name in ("subject", "subject_part", "object", "object_part"):
        ok, value = _render(getattr(t, name), fields)
        if not ok:
            return None
        values[name] = value
    subject, obj = values["subject"], values["object"]
    if subject is None or obj is None:
        return None
    return RelationPayload(
        subject_id=subject,
        subject_part=values["subject_part"],
        predicate=t.predicate,
        object_id=obj,
        object_part=values["object_part"],
        derived_by=rule.id,
    )


def apply_rules(
    policy: RelationsPolicy,
    hand_states: list[LabelRecord],
    contacts: list[SurfaceContact],
) -> list[RelationDraft]:
    """손 상태·도구-표면 접촉 구간에 규칙을 적용하고 병합한 관계 초안.

    손 상태 신뢰도는 라벨 신뢰도(없으면 사람 라벨로 보고 1.0), 접촉 신뢰도는
    `tool_surface.confidence`.
    """
    inputs: list[tuple[str, Fields, int, int, float]] = []
    for x in hand_states:
        if isinstance(x.payload, HandStatePayload):
            # 사람 라벨은 신뢰도가 없다 → 1.0
            conf = x.confidence if x.confidence is not None else 1.0
            fields = hand_state_fields(x.payload, policy.unresolved_target_ids)
            inputs.append(("hand_state", fields, x.t_start_ms, x.t_end_ms, conf))
    for c in contacts:
        conf = policy.tool_surface.confidence
        inputs.append(("tool_surface", contact_fields(c), c.start_ms, c.end_ms, conf))

    drafts: list[RelationDraft] = []
    for source, fields, start, end, conf in inputs:
        for rule in policy.rules:
            if rule.source == source and _matches(rule, fields):
                payload = _emit(rule, fields)
                if payload is not None:
                    drafts.append(RelationDraft(payload, start, end, conf))
    return merge(drafts, policy.merge_gap_ms)


def merge(drafts: list[RelationDraft], gap_ms: int) -> list[RelationDraft]:
    """같은 페이로드(JSON이 같은) 초안이 gap_ms 이하로 끊기면 하나로 잇는다.

    이은 구간의 신뢰도는 더 낮은 쪽이다. 결과는 (시작, 페이로드) 순으로 정렬한다.
    """

    def key(d: RelationDraft) -> str:
        """병합 키: 페이로드 JSON (주어·술어·목적어·부위·규칙 ID가 모두 같아야 같은 관계)."""
        return d.payload.model_dump_json()

    out: list[RelationDraft] = []
    # 같은 관계끼리 시각 순으로 모아 바로 앞 구간과만 비교한다
    for d in sorted(drafts, key=lambda d: (key(d), d.start_ms, d.end_ms)):
        last = out[-1] if out else None
        if last is not None and key(last) == key(d) and d.start_ms - last.end_ms <= gap_ms:
            out[-1] = RelationDraft(
                d.payload,
                last.start_ms,
                max(last.end_ms, d.end_ms),
                min(last.confidence, d.confidence),
            )
        else:
            out.append(d)
    return sorted(out, key=lambda d: (d.start_ms, key(d)))
