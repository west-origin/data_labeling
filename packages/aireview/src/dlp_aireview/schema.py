"""판정 응답 JSON Schema와 라벨 종류별 문제 유형·제안 필드 (WP17.2, ADR 0040 3.1).

응답 스키마는 온톨로지 버전·묶음·정책이 같으면 같은 사전이다. 개체·라벨은 창 지역 정수 번호로
가리키므로 세션·창과 무관하다 (서버가 문법(xgrammar, GBNF)을 다시 컴파일하지 않는다). 시간 묶음
(temporal, temporal_long)은 같은 스키마를 쓰고, 공간 묶음은 따로다.

응답 모양 (키는 모두 필수, 값이 없을 수 있으면 null)
    {"ok": [번호], "items": [{"i", "v", "p", "r", "c", "f", "s"}], "missing": [...]}
    - ok: 맞는 대상 번호 (이유 없음, 저장할 때 "AI: 문제 없음").
    - items: 맞지 않거나 판단할 수 없는 대상. v = suspect | wrong | unsure, p = 문제 유형,
      r = 한국어 이유, c = 확신도, f = 근거 프레임 번호(F#), s = 제안 또는 null.
    - missing: 누락 후보 (`response.max_findings`, 기본 0 = 끔).

스키마 프로필
    - portable(기본): type, enum, required, additionalProperties, items, maxItems, anyOf만
      쓴다. 숫자 범위·문자열 길이·minItems는 서버마다 지원이 달라 400 오류나 느린 대체 백엔드를
      부를 수 있으므로 넣지 않고 재검사(`validate`)에 맡긴다.
    - full: 범위·길이까지 넣는다 (probe가 서버 지원을 확인한 뒤에만).

주요 이름: `COMMON_PROBLEMS`, `PROBLEMS_BY_KIND`, `SUGGEST_FIELDS_BY_KIND`, `TEMPORAL_FIELDS`,
`SPATIAL_FIELDS`, `group_problems`, `response_schema`.
"""

from __future__ import annotations

from typing import Any

from dlp_aireview.policy import AiReviewPolicy
from dlp_schema.aireview import AiGroup, AiProblem
from dlp_schema.ontology import Ontology, VerbLevel

P = AiProblem
# 모든 종류에 쓸 수 있는 문제 유형 (response_invalid는 시스템 전용이라 응답에 넣지 않는다)
COMMON_PROBLEMS = (P.NOT_VISIBLE, P.INSUFFICIENT_FRAMES, P.OTHER)
# 라벨 종류 → 응답이 쓸 수 있는 문제 유형 (ADR 0040 2.3)
PROBLEMS_BY_KIND: dict[str, tuple[AiProblem, ...]] = {
    "action": (
        P.WRONG_VERB, P.WRONG_TARGET, P.WRONG_TOOL, P.WRONG_HAND, P.SHOULD_BE_GAP,
        P.BOUNDARY_EARLY, P.BOUNDARY_LATE, *COMMON_PROBLEMS,
    ),
    "gap": (P.SHOULD_BE_ACTION, P.WRONG_GAP_TYPE, *COMMON_PROBLEMS),
    "hand_state": (
        P.CONTACT_MISMATCH_SENSOR, P.WRONG_CONTACT_KIND, P.WRONG_CONTACT_TARGET, P.WRONG_GRASP,
        *COMMON_PROBLEMS,
    ),
    "event": (P.FALSE_EVENT, P.WRONG_EVENT_TYPE, P.WRONG_SEVERITY, *COMMON_PROBLEMS),
    "segment": (P.WRONG_REF, P.HIERARCHY_INCONSISTENT, *COMMON_PROBLEMS),
    "object_state": (P.WRONG_STATE_VALUE, *COMMON_PROBLEMS),
    "box_track": (
        P.WRONG_CLASS, P.FALSE_POSITIVE, P.BOX_MISALIGNED, P.IDENTITY_SWITCH, P.TRACK_DRIFT,
        *COMMON_PROBLEMS,
    ),
    "keypoint_track": (
        P.HAND_SIDE_SWAPPED, P.KEYPOINT_MISPLACED, P.FALSE_POSITIVE, *COMMON_PROBLEMS,
    ),
}  # fmt: skip
# 시간 묶음 제안 객체의 키 (스키마 순서)
TEMPORAL_FIELDS = (
    "kind", "verb", "gap_type", "target", "tool", "contact_kind", "grasp", "body_part",
    "event_type", "severity", "ref_id", "value", "delete",
)  # fmt: skip
# 공간 묶음 제안 객체의 키
SPATIAL_FIELDS = ("class_id", "hand", "delete")
# 라벨 종류 → 제안에 값을 넣을 수 있는 키. kind는 action↔gap 종류 변경이고, 그때 새 종류의 키를
# 함께 쓴다. 경계(시각)는 제안하지 않는다 (VLM은 경계를 정하지 않는다). delete는 정책 delete_kinds
# 안에서만.
SUGGEST_FIELDS_BY_KIND: dict[str, frozenset[str]] = {
    "action": frozenset({"verb", "target", "tool", "kind", "gap_type"}),
    "gap": frozenset({"gap_type", "kind", "verb", "target", "tool"}),
    "hand_state": frozenset({"contact_kind", "target", "grasp", "body_part"}),
    "event": frozenset({"event_type", "severity", "delete"}),
    "segment": frozenset({"ref_id"}),
    "object_state": frozenset({"value"}),
    "box_track": frozenset({"class_id", "delete"}),
    "keypoint_track": frozenset({"hand", "delete"}),
}


def group_problems(group: AiGroup, policy: AiReviewPolicy) -> tuple[AiProblem, ...]:
    """묶음이 판정하는 종류들의 문제 유형 합집합 (정렬, 공통 포함)."""
    kinds = policy.kinds.spatial if group is AiGroup.SPATIAL else policy.kinds.temporal
    found = {p for k in kinds for p in PROBLEMS_BY_KIND[k]} | set(COMMON_PROBLEMS)
    return tuple(sorted(found, key=lambda p: p.value))


def _enum(values: list[Any]) -> dict[str, Any]:
    """값 목록 + null 열거."""
    return {"enum": [*values, None]}


def _int_or_null(full: bool) -> dict[str, Any]:
    """번호(정수) 또는 null. full이면 0 이상."""
    return {"type": ["integer", "null"], **({"minimum": 0} if full else {})}


def temporal_values(o: Ontology) -> dict[str, list[Any]]:
    """시간 제안 키 → 허용 값 (온톨로지 ID 정렬). raw_access 이벤트는 넣지 않는다."""
    states = sorted({v for a in o.state_attributes.values() for v in a.values})
    refs = sorted(
        {v for v, s in o.verbs.items() if s.level is not VerbLevel.PRIMITIVE}
        | set(o.tasks)
        | {s for t in o.tasks.values() for s in t.substeps}
    )
    return {
        "kind": ["action", "gap"],
        "verb": sorted(v for v, s in o.verbs.items() if s.level is VerbLevel.PRIMITIVE),
        "gap_type": sorted(o.gap_types),
        "contact_kind": sorted(o.contact_target_kinds),
        "grasp": sorted(o.grasp_types),
        "body_part": sorted(o.body_parts),
        "event_type": sorted(e for e, s in o.events.items() if not s.raw_access),
        "severity": [1, 2, 3],
        "ref_id": refs,
        "value": states,
    }


def _suggestion(o: Ontology, group: AiGroup, full: bool) -> dict[str, Any]:
    """제안 객체 스키마 (null 또는 키 전부 필수 객체)."""
    if group is AiGroup.SPATIAL:
        props: dict[str, Any] = {
            "class_id": _enum(sorted(o.objects)),
            "hand": _enum(["left", "right"]),
            "delete": {"type": "boolean"},
        }
        keys = SPATIAL_FIELDS
    else:
        values = temporal_values(o)
        props = {k: _enum(values[k]) for k in values}
        props["target"] = props["tool"] = _int_or_null(full)
        props["delete"] = {"type": "boolean"}
        props = {k: props[k] for k in TEMPORAL_FIELDS}
        keys = TEMPORAL_FIELDS
    obj = {
        "type": "object",
        "additionalProperties": False,
        "required": list(keys),
        "properties": props,
    }
    return {"anyOf": [{"type": "null"}, obj]}


def _finding(o: Ontology, group: AiGroup, policy: AiReviewPolicy, full: bool) -> dict[str, Any]:
    """누락 후보 항목 스키마 (temporal: 이벤트, spatial: 물체)."""
    reason = {"type": "string", **({"maxLength": policy.response.reason_max_chars} if full else {})}
    if group is AiGroup.SPATIAL:
        coord = {"type": "integer", **({"minimum": 0, "maximum": 1000} if full else {})}
        props: dict[str, Any] = {
            "class_id": {"enum": sorted(o.objects)},
            "f": {"type": "integer"},
            "box": {"type": "array", "maxItems": 4, "items": coord},
            "r": reason,
        }
    else:
        props = {
            "event_type": {"enum": temporal_values(o)["event_type"]},
            "f0": {"type": "integer"},
            "f1": {"type": "integer"},
            "r": reason,
        }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(props),
        "properties": props,
    }


def response_schema(o: Ontology, group: AiGroup, policy: AiReviewPolicy) -> dict[str, Any]:
    """묶음의 응답 JSON Schema (모듈 docstring의 모양, 정책 `schema_profile`).

    Args:
        o: 세션 온톨로지.
        group: 요청 묶음 (temporal과 temporal_long은 같은 스키마).
        policy: 정책 (대상 상한, 문제 상한, 누락 상한, 프로필).
    """
    full = policy.backend.schema_profile == "full"
    limit = policy.spatial.max_targets if group is AiGroup.SPATIAL else policy.temporal.max_targets
    item_props: dict[str, Any] = {
        "i": {"type": "integer", **({"minimum": 0, "maximum": limit - 1} if full else {})},
        "v": {"enum": ["suspect", "wrong", "unsure"]},
        "p": {
            "type": "array",
            "maxItems": policy.response.max_problems,
            "items": {"enum": [p.value for p in group_problems(group, policy)]},
            **({"minItems": 1} if full else {}),
        },
        "r": {
            "type": "string",
            **({"minLength": 2, "maxLength": policy.response.reason_max_chars} if full else {}),
        },
        "c": {"type": ["number", "null"], **({"minimum": 0, "maximum": 1} if full else {})},
        "f": _int_or_null(full),
        "s": _suggestion(o, group, full),
    }
    item = {
        "type": "object",
        "additionalProperties": False,
        "required": list(item_props),
        "properties": item_props,
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["ok", "items", "missing"],
        "properties": {
            "ok": {"type": "array", "maxItems": limit, "items": {"type": "integer"}},
            "items": {"type": "array", "maxItems": limit, "items": item},
            "missing": {
                "type": "array",
                "maxItems": policy.response.max_findings,
                "items": _finding(o, group, policy, full),
            },
        },
    }
