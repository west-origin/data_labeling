"""검증된 제안 패치를 완성 페이로드로 만들고 온톨로지로 검사한다 (WP17.2, ADR 0040 3.2 5단계).

`materialize(original, patch, entities, o)`
    - delete: 페이로드 없는 삭제 제안 (이벤트·공간 트랙만, 검증 단계에서 이미 거른다).
    - change_kind(action↔gap): 새 페이로드를 만든다. gap→action의 action_id는 Label Studio에서
      새로 그린 행동과 같은 규칙 `h-<손>-<시작 ms>`다 (AI 기원 표시가 판매 데이터에 섞이지 않게.
      AI 기원은 ai_review_outcomes에만 남는다). 행동의 접근 시작·끝은 라벨 구간과 같고 접촉 시각은
      비운다.
    - set: 원 페이로드에 바뀐 값을 넣고 Pydantic 검증을 다시 거친다 (`model_copy`는 검증하지
      않으므로 `model_validate`). 비접촉 동사로 바꾸면 접촉 시각을 비우고, 손 상태 접촉 대상이
      none이면 대상·파지·신체 부위를 비운다. 경계(시각)는 바꾸지 않는다.
    - 바꾸는 것이 없으면 `NO_CHANGE`를 돌려준다 (재검사가 항목 위반으로 센다).
    - 메모리 속 `LabelRecord`로 `check_label`을 통과해야 한다. 실패하면 제안만 버리고(문자열 사유를
      돌려준다) 판정은 남긴다. 세션 단위 검사(상위 구간 자식 연결 등)는 사람이 적용할 때 한다
      (WP17.7).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from pydantic import ValidationError

from dlp_schema.aireview import AiSuggestion
from dlp_schema.labels import (
    ActionPayload,
    BoxTrackPayload,
    EventPayload,
    GapPayload,
    HandStatePayload,
    KeypointTrackPayload,
    LabelPayload,
    LabelRecord,
    ObjectStatePayload,
    SegmentPayload,
)
from dlp_schema.ontology import Ontology
from dlp_schema.validation import check_label

Entities = Sequence[tuple[str, str | None]]
# 원 페이로드와 같은 제안 (재검사에서 위반으로 센다)
NO_CHANGE = "바꾸는 값이 없는 제안"


def _entity(patch: dict[str, Any], key: str, entities: Entities) -> str | None:
    """패치의 개체 번호 → 개체 ID (없으면 None)."""
    v = patch.get(key)
    return entities[v][0] if isinstance(v, int) else None


def _action_update(
    p: ActionPayload, patch: dict[str, Any], e: Entities, o: Ontology
) -> dict[str, Any]:
    """행동 set: 동사·대상·도구. 비접촉 동사면 접촉 시각을 비운다."""
    up: dict[str, Any] = {}
    if "verb" in patch:
        up["verb"] = patch["verb"]
        spec = o.verbs.get(patch["verb"])
        if spec is not None and spec.contactless:
            up |= {"t_contact_start_ms": None, "t_contact_end_ms": None}
    for key, field in (("target", "target_id"), ("tool", "tool_id")):
        if key in patch:
            up[field] = _entity(patch, key, e)
    return up


def _hand_update(
    p: HandStatePayload, patch: dict[str, Any], e: Entities, o: Ontology
) -> dict[str, Any]:
    """손 상태 set: 접촉 대상 종류·대상·파지·신체 부위. none이면 나머지를 비운다."""
    up: dict[str, Any] = {}
    if "contact_kind" in patch:
        up["contact_target_kind"] = patch["contact_kind"]
    if "target" in patch:
        up["target_id"] = _entity(patch, "target", e)
    if "grasp" in patch:
        up["grasp_type"] = patch["grasp"]
    if "body_part" in patch:
        up["body_part"] = patch["body_part"]
    if up.get("contact_target_kind") == "none":
        up |= {"target_id": None, "grasp_type": None, "body_part": None}
    return up


def _simple(
    mapping: dict[str, str],
) -> Callable[[Any, dict[str, Any], Entities, Ontology], dict[str, Any]]:
    """패치 키 → 페이로드 필드 이름을 그대로 옮기는 갱신 함수."""

    def update(p: Any, patch: dict[str, Any], e: Entities, o: Ontology) -> dict[str, Any]:
        return {field: patch[key] for key, field in mapping.items() if key in patch}

    return update


_UPDATES: dict[type, Callable[[Any, dict[str, Any], Entities, Ontology], dict[str, Any]]] = {
    ActionPayload: _action_update,
    HandStatePayload: _hand_update,
    GapPayload: _simple({"gap_type": "gap_type"}),
    EventPayload: _simple({"event_type": "event_type", "severity": "severity"}),
    SegmentPayload: _simple({"ref_id": "ref_id"}),
    ObjectStatePayload: _simple({"value": "value"}),
    BoxTrackPayload: _simple({"class_id": "class_id"}),
    KeypointTrackPayload: _simple({"hand": "hand"}),
}


def _changed_kind(original: LabelRecord, patch: dict[str, Any], e: Entities) -> LabelPayload | str:
    """action↔gap 종류 변경 페이로드 (gap→action의 손이 없으면 사유 문자열)."""
    p = original.payload
    if patch["kind"] == "gap" and isinstance(p, ActionPayload):
        if "gap_type" not in patch:
            return "사이 구간으로 바꾸는 제안에 gap_type이 없습니다"
        return GapPayload(hand=p.hand, gap_type=patch["gap_type"])
    if patch["kind"] == "action" and isinstance(p, GapPayload):
        if p.hand is None or "verb" not in patch:
            return "행동으로 바꾸는 제안에 손 또는 verb가 없습니다"
        return ActionPayload(
            action_id=f"h-{p.hand.value}-{original.t_start_ms}",
            hand=p.hand,
            verb=patch["verb"],
            target_id=_entity(patch, "target", e),
            tool_id=_entity(patch, "tool", e),
            t_approach_ms=original.t_start_ms,
            t_end_ms=original.t_end_ms,
        )
    return f"{original.kind}을(를) {patch['kind']}(으)로 바꿀 수 없습니다"


def materialize(
    original: LabelRecord, patch: dict[str, Any], entities: Entities, o: Ontology
) -> AiSuggestion | str:
    """제안 패치 → 완성 제안 (`AiSuggestion`), 실패하면 사유 문자열 (모듈 docstring).

    Args:
        original: 판정한 원 라벨.
        patch: 값이 있는 제안 키 (target·tool은 창 개체표 번호, 검증 단계를 통과한 것).
        entities: 창 개체표 (개체 ID, 클래스).
        o: 세션 온톨로지.
    """
    fields = tuple(sorted(patch))
    if patch.get("delete") is True:
        return AiSuggestion(op="delete", fields=("delete",))
    try:
        if "kind" in patch and patch["kind"] != original.kind:
            payload = _changed_kind(original, patch, entities)
            op = "change_kind"
        else:
            p = original.payload
            update = _UPDATES.get(type(p))
            if update is None:
                return f"{original.kind}에는 제안을 받지 않습니다"
            changes = update(p, patch, entities, o)
            payload = type(p).model_validate(p.model_dump() | changes)
            op = "set"
        if isinstance(payload, str):
            return payload
        if payload == original.payload:
            return NO_CHANGE
        candidate = original.model_validate(
            original.model_dump() | {"payload": payload.model_dump()}
        )
    except ValidationError as exc:
        return f"제안이 계약을 어깁니다: {exc.errors()[0]['msg']}"
    problems = check_label(candidate, o)
    if problems:
        return "온톨로지 위반: " + "; ".join(problems)
    return AiSuggestion(op=op, fields=fields, payload=payload)
