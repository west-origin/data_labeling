"""2단: VLM이 후보 구간을 온톨로지 목록 안에서 분류하고 한 문장 설명을 쓴다 (WP10, ADR 0012·0026).

- 프롬프트에 온톨로지(원시 동작, 사이 구간 종류, 화면 속 개체)를 넣는다.
- 출력은 JSON Schema로 강제한다 (OpenAI 호환 서버의 response_format=json_schema). 서버가 강제하지
  못해도 여기서 다시 검사한다: JSON 파싱, 스키마, 온톨로지 밖 값, 개체 목록 밖 대상.
- 위반하면 위반 내용을 붙여 max_retries번까지 다시 묻고, 그래도 안 되면 미상(gap unknown)이다.
- 서버 오류·시간 초과(VlmUnavailableError)는 응답 내용 문제가 아니다. 정책의 백오프
  (vlm.unavailable_backoff_s)만큼 기다리며 다시 묻고, 끝내 실패하면 미상으로 두지 않고
  VlmUnavailableError를 올린다 (세션 실행을 멈춘다: 장애를 미상으로 쓰면 같은 버전이라 재실행이
  건너뛰어 영구히 미상으로 남는다). 모르는 구간을 대기(idle)로 처리하지 않는다.

주요 항목: `SegmentRequest`(구간 요청), `VlmAnswer`(응답 형식), `Classified`(결과), `VlmClient`
(클라이언트 프로토콜), `response_schema`, `build_prompt`, `validate_answer`, `classify`.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import Field, ValidationError

from dlp_schema.common import Contract
from dlp_schema.labels import Hand
from dlp_schema.ontology import Ontology


@dataclass(frozen=True)
class SegmentRequest:
    """VLM에 물을 후보 구간 하나.

    Attributes:
        session_id, stream_id: 세션과 영상 스트림 (바디캠).
        video: 프레임을 뽑을 블러본 로컬 경로 (None이면 프레임 없이 묻는다).
        hand: 어느 손의 행동인지.
        start_ms, end_ms: 구간 ms (바디캠 PTS = 마스터 타임라인).
        in_contact: 구간 안에 접촉이 있는가 (신호 기준). 프롬프트에 힌트로 준다.
        entities: 화면 속 개체 ID (대상·도구 후보). 응답의 target_id·tool_id는 이 안이어야 한다.
    """

    session_id: str
    stream_id: str
    video: Path | None
    hand: Hand
    start_ms: int
    end_ms: int
    in_contact: bool  # 구간 안에 접촉이 있는가 (신호 기준)
    entities: tuple[str, ...]  # 화면 속 개체 ID (대상 후보)


class VlmAnswer(Contract):
    """VLM 응답 형식. 온톨로지 값 검사는 validate_answer가 한다.

    label=action이면 verb(원시 동작)·target_id·tool_id, label=gap이면 gap_type을 쓴다.
    description은 한국어 한 문장(200자 이하), confidence는 0~1 또는 없음.
    """

    label: Literal["action", "gap"]
    verb: str | None = None
    gap_type: str | None = None
    target_id: str | None = None
    tool_id: str | None = None
    description: str | None = Field(default=None, max_length=200)
    confidence: float | None = Field(default=None, ge=0, le=1)


@dataclass
class Classified:
    """구간 하나의 분류 결과.

    Attributes:
        request: 원래 요청.
        answer: 검증을 통과한 응답, 또는 미상 대체(`UNKNOWN`).
        attempts: 응답 검증 시도 횟수 (서버 오류 재시도는 세지 않는다).
        errors: 서버 오류·응답 위반 기록 (사람이 읽을 문장).
        fallback: 재시도 후에도 실패해 미상으로 둠.
    """

    request: SegmentRequest
    answer: VlmAnswer
    attempts: int
    errors: list[str] = field(default_factory=list[str])
    fallback: bool = False  # 재시도 후에도 실패해 미상으로 둠


class VlmUnavailableError(RuntimeError):
    """VLM 서버 오류·시간 초과. classify가 백오프하며 재시도하고, 끝내 실패하면 다시 올린다."""


class VlmClient(Protocol):
    """프롬프트·스키마를 받아 원문 응답(JSON 문자열)을 돌려준다.

    `version`은 모델 버전 문자열에 들어간다 (바뀌면 다시 만든다). `complete`는 서버 오류·시간 초과면
    `VlmUnavailableError`를 내야 한다 (응답 내용 위반과 구분).
    """

    version: str

    def complete(self, request: SegmentRequest, prompt: str, schema: dict[str, Any]) -> str:
        """구간 하나를 물어 원문 응답을 돌려준다.

        Args:
            request: 구간 요청 (영상 경로가 있으면 프레임을 뽑는다).
            prompt: 프롬프트 (재시도면 위반 내용이 붙어 있다).
            schema: 응답 JSON Schema (`response_schema`).

        Raises:
            VlmUnavailableError: 서버 오류·시간 초과.
        """
        ...


def response_schema(ontology: Ontology, entities: tuple[str, ...]) -> dict[str, Any]:
    """응답 JSON Schema: 동사는 온톨로지 원시 동작, gap_type은 온톨로지 사이 구간 종류,
    target_id·tool_id는 `entities` 또는 null로 제한한다.

    `label`과 `description`은 필수다. 기술(skill) 수준 동사는 넣지 않는다 (2단 분류 대상은 원시
    동작뿐).
    """
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
    """한국어 프롬프트: 구간, 손, 접촉 여부, 원시 동작 목록(ID: 한국어), 사이 구간 종류 목록, 개체
    후보, 설명 작성 규칙, "JSON 하나만" 지시. "모르면 unknown, idle로 고르지 않는다"를 명시한다.
    """
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
    """원문 응답을 검사한다.

    검사: JSON 파싱 → `VlmAnswer` 스키마 → action이면 원시 동작인지·gap_type 없음, gap이면 온톨로지
    사이 구간 종류인지·verb/target_id 없음 → target_id·tool_id가 개체 목록 안인지.

    Returns:
        (응답, []) 또는 (None, 위반 목록). 위반 목록은 재시도 프롬프트에 그대로 붙는다.
    """
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


# 재시도 후에도 위반일 때 쓰는 미상 응답 (대기 idle이 아니다)
UNKNOWN = VlmAnswer(label="gap", gap_type="unknown", description=None)


def _complete(
    client: VlmClient,
    request: SegmentRequest,
    prompt: str,
    schema: dict[str, Any],
    backoff_s: Sequence[float],
    sleep: Callable[[float], None],
    errors: list[str],
) -> str:
    """서버 오류·시간 초과면 backoff_s의 대기마다 다시 묻는다. 끝내 실패하면 그 오류를 올린다.

    시도 횟수 = len(backoff_s) + 1. 실패할 때마다 `errors`에 기록하고 `sleep(대기 초)`.

    Raises:
        VlmUnavailableError: 마지막 시도도 실패했을 때.
    """
    for wait in backoff_s:
        try:
            return client.complete(request, prompt, schema)
        except VlmUnavailableError as exc:
            errors.append(f"VLM 서버 오류: {exc}")
            sleep(wait)
    return client.complete(request, prompt, schema)


def classify(
    client: VlmClient,
    request: SegmentRequest,
    ontology: Ontology,
    max_retries: int,
    backoff_s: Sequence[float] = (),
    sleep: Callable[[float], None] = time.sleep,
) -> Classified:
    """후보 구간 하나를 VLM에 물어 검증된 분류를 얻는다.

    응답 위반은 max_retries번까지 다시 묻고, 서버 오류는 backoff_s로 따로 다시 묻는다. 서버가 끝내
    응답하지 않으면 VlmUnavailableError를 올린다 (미상으로 두지 않는다).

    Args:
        client: VLM 클라이언트.
        request: 구간 요청.
        ontology: 온톨로지.
        max_retries: 응답 위반 시 재시도 횟수 (총 시도 = max_retries + 1).
        backoff_s: 서버 오류 시 대기 초 목록 (시도마다 처음부터 다시 쓴다).
        sleep: 대기 함수 (테스트에서 바꾼다).

    Returns:
        `Classified`. 끝까지 위반이면 `UNKNOWN`(gap unknown)과 fallback=True.
    """
    schema = response_schema(ontology, request.entities)
    prompt = build_prompt(request, ontology)
    errors: list[str] = []
    for attempt in range(1, max_retries + 2):
        raw = _complete(client, request, prompt, schema, backoff_s, sleep, errors)
        answer, problems = validate_answer(raw, ontology, request)
        if answer is not None:
            return Classified(request, answer, attempt, errors)
        errors += problems
        # 다음 시도: 같은 프롬프트에 이번 위반 내용을 붙여 다시 묻는다
        prompt = (
            build_prompt(request, ontology)
            + "\n\n앞 응답이 규칙을 어겼다: "
            + "; ".join(problems)
            + "\n목록 안의 값으로 JSON 하나만 다시 출력한다."
        )
    return Classified(request, UNKNOWN, max_retries + 1, errors, fallback=True)
