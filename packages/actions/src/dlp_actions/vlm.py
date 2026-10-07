"""2단: VLM이 후보 구간을 온톨로지 목록 안에서 분류하고 한 문장 설명을 쓴다.

- 프롬프트에 온톨로지(원시 동작, 사이 구간 종류, 화면 속 개체)를 넣는다.
- 출력은 JSON Schema로 강제한다 (OpenAI 호환 서버의 response_format=json_schema). 서버가 강제하지
  못해도 여기서 다시 검사한다: JSON 파싱, 스키마, 온톨로지 밖 값, 개체 목록 밖 대상.
- 위반하면 위반 내용을 붙여 max_retries번까지 다시 묻고, 그래도 안 되면 미상(gap unknown)이다.
  모르는 구간을 대기(idle)로 처리하지 않는다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import Field, ValidationError

from dlp_schema.common import Contract
from dlp_schema.labels import Hand
from dlp_schema.ontology import Ontology


@dataclass(frozen=True)
class SegmentRequest:
    session_id: str
    stream_id: str
    video: Path | None
    hand: Hand
    start_ms: int
    end_ms: int
    in_contact: bool  # 구간 안에 접촉이 있는가 (신호 기준)
    entities: tuple[str, ...]  # 화면 속 개체 ID (대상 후보)


class VlmAnswer(Contract):
    """VLM 응답 형식. 온톨로지 값 검사는 validate_answer가 한다."""

    label: Literal["action", "gap"]
    verb: str | None = None
    gap_type: str | None = None
    target_id: str | None = None
    tool_id: str | None = None
    description: str | None = Field(default=None, max_length=200)
    confidence: float | None = Field(default=None, ge=0, le=1)


@dataclass
class Classified:
    request: SegmentRequest
    answer: VlmAnswer
    attempts: int
    errors: list[str] = field(default_factory=list[str])
    fallback: bool = False  # 재시도 후에도 실패해 미상으로 둠


class VlmClient(Protocol):
    """프롬프트·스키마를 받아 원문 응답(JSON 문자열)을 돌려준다."""

    version: str

    def complete(self, request: SegmentRequest, prompt: str, schema: dict[str, Any]) -> str: ...


def response_schema(ontology: Ontology, entities: tuple[str, ...]) -> dict[str, Any]:
    primitives = sorted(v for v, spec in ontology.verbs.items() if spec.level == "primitive")
    targets: list[str | None] = [*sorted(entities), None]
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["label", "description"],
        "properties": {
            "label": {"enum": ["action", "gap"]},
            "verb": {"enum": [*primitives, None]},
            "gap_type": {"enum": [*sorted(ontology.gap_types), None]},
            "target_id": {"enum": targets},
            "tool_id": {"enum": targets},
            "description": {"type": ["string", "null"], "maxLength": 200},
            "confidence": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
        },
    }


def build_prompt(request: SegmentRequest, ontology: Ontology) -> str:
    verbs = "\n".join(
        f"- {v}: {spec.ko}"
        for v, spec in sorted(ontology.verbs.items())
        if spec.level == "primitive"
    )
    gaps = "\n".join(
        f"- {g}: {spec.ko} ({spec.description})" for g, spec in sorted(ontology.gap_types.items())
    )
    entities = ", ".join(request.entities) or "(없음)"
    hand = "오른손" if request.hand is Hand.RIGHT else "왼손"
    contact = "있음" if request.in_contact else "없음"
    return (
        f"1인칭 돌봄·청소 영상의 {request.start_ms}~{request.end_ms} ms 구간 프레임이다. "
        f"{hand}이 이 구간에서 한 일을 아래 목록 안에서만 고른다.\n"
        f"접촉 센서 기준 접촉: {contact}\n\n"
        f"원시 동작 (label=action, verb):\n{verbs}\n\n"
        f"행동이 아닌 구간 (label=gap, gap_type):\n{gaps}\n"
        "판단할 수 없으면 gap_type=unknown이다. 모르는 구간을 idle로 고르지 않는다.\n\n"
        f"대상·도구 후보 개체 ID: {entities}\n"
        "description은 이 구간의 행동을 한국어 한 문장 지시문으로 쓴다 "
        "(예: '걸레로 세면대를 문지른다').\n"
        "JSON 하나만 출력한다."
    )


def validate_answer(
    raw: str, ontology: Ontology, request: SegmentRequest
) -> tuple[VlmAnswer | None, list[str]]:
    try:
        data: Any = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, [f"JSON이 아닙니다: {exc.msg}"]
    try:
        answer = VlmAnswer.model_validate(data)
    except ValidationError as exc:
        return None, [f"스키마 위반: {e['loc']} {e['msg']}" for e in exc.errors()]
    errors: list[str] = []
    if answer.label == "action":
        spec = ontology.verbs.get(answer.verb or "")
        if spec is None or spec.level != "primitive":
            errors.append(f"원시 동작 목록에 없는 verb: {answer.verb}")
        if answer.gap_type is not None:
            errors.append("action에는 gap_type을 쓰지 않습니다")
    else:
        if answer.gap_type not in ontology.gap_types:
            errors.append(f"사이 구간 종류 목록에 없는 gap_type: {answer.gap_type}")
        if answer.verb is not None or answer.target_id is not None:
            errors.append("gap에는 verb·target_id를 쓰지 않습니다")
    for name in ("target_id", "tool_id"):
        value = getattr(answer, name)
        if value is not None and value not in request.entities:
            errors.append(f"개체 목록에 없는 {name}: {value}")
    return (None, errors) if errors else (answer, [])


UNKNOWN = VlmAnswer(label="gap", gap_type="unknown", description=None)


def classify(
    client: VlmClient, request: SegmentRequest, ontology: Ontology, max_retries: int
) -> Classified:
    schema = response_schema(ontology, request.entities)
    prompt = build_prompt(request, ontology)
    errors: list[str] = []
    for attempt in range(1, max_retries + 2):
        raw = client.complete(request, prompt, schema)
        answer, problems = validate_answer(raw, ontology, request)
        if answer is not None:
            return Classified(request, answer, attempt, errors)
        errors += problems
        prompt = (
            build_prompt(request, ontology)
            + "\n\n앞 응답이 규칙을 어겼다: "
            + "; ".join(problems)
            + "\n목록 안의 값으로 JSON 하나만 다시 출력한다."
        )
    return Classified(request, UNKNOWN, max_retries + 1, errors, fallback=True)
