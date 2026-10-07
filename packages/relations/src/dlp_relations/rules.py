"""규칙 엔진: 입력 구간(손 상태, 도구-표면 접촉)에 YAML 규칙을 적용해 관계 구간을 만든다.

규칙 ID는 관계의 derived_by에 남는다. 같은 관계(주어·술어·목적어·부분·규칙)가 merge_gap_ms보다
짧게 끊기면 하나로 잇는다.
"""

from __future__ import annotations

import string
from dataclasses import dataclass

from dlp_relations.contact import SurfaceContact
from dlp_relations.policy import RelationsPolicy, RelationTemplate, Rule
from dlp_schema.labels import HandStatePayload, LabelRecord, RelationPayload

Fields = dict[str, str | None]


@dataclass(frozen=True)
class RelationDraft:
    payload: RelationPayload
    start_ms: int
    end_ms: int
    confidence: float


def hand_state_fields(p: HandStatePayload) -> Fields:
    return {
        "hand": p.hand.value,
        "contact_target_kind": p.contact_target_kind,
        "target_id": p.target_id,
        "body_part": p.body_part,
        "grasp_type": p.grasp_type,
        "role": p.role,
    }


def contact_fields(c: SurfaceContact) -> Fields:
    return {"tool_id": c.tool_id, "tool_part": c.tool_part, "surface_id": c.surface_id}


def _matches(rule: Rule, fields: Fields) -> bool:
    return all(fields.get(name) in allowed for name, allowed in rule.when.items())


def _render(template: str | None, fields: Fields) -> tuple[bool, str | None]:
    """(성공, 값). 자리에 넣을 값이 없으면 실패."""
    if template is None:
        return True, None
    names = [n for _, n, _, _ in string.Formatter().parse(template) if n]
    if any(fields.get(n) is None for n in names):
        return False, None
    return True, template.format(**{n: fields[n] for n in names})


def _emit(rule: Rule, fields: Fields) -> RelationPayload | None:
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
    inputs: list[tuple[str, Fields, int, int, float]] = []
    for x in hand_states:
        if isinstance(x.payload, HandStatePayload):
            conf = x.confidence if x.confidence is not None else 1.0
            fields = hand_state_fields(x.payload)
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
    def key(d: RelationDraft) -> str:
        return d.payload.model_dump_json()

    out: list[RelationDraft] = []
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
