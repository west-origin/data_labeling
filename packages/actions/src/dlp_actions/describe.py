"""사람이 의미를 고친 행동의 한국어 설명을 행동 VLM으로 다시 만든다 (`dlp actions describe`). 멱등.

배경 (ADR 0043 3절)
    검수자가 행동의 의미(동사·대상·도구·손, 양손 중 한 손 삭제)를 고치면 모델 설명은 다른 행동을
    말하므로 검수 전 설명을 지운다 (`dlp_schema.description_follow`). 그 행동에는 설명이 없게 된다.
    사용자 결정("추천안대로"): 행동 단계와 같은 사내 VLM 전송·정책으로 설명만 다시 묻는다.

대상 (`plan_describe`, 순수 계산)
    현재 운영 행동(주인 ID = action_id)마다:
    - 그 행동에 사람 레코드가 이력에 있다 (사람이 고쳤거나 지웠거나 그렸다). 고치지 않은 행동은
      이 모듈의 대상이 아니다 (행동 단계 몫).
    - 현재 원문 설명(번역이 아닌 설명)이 없다. 설명 따르기 규칙이 지울 검수 전 설명은 없는 것으로
      본다. 이 단계가 이전 입력으로 만든 검수 전 설명만 있으면 교체 대상이다 (그것만 지운다).
    - 현재 주인 레코드가 모두 사람이 확인한 상태다(`human_confirmed`). 아니면 기다린다 (`waiting`).
    - 이 버전·입력의 설명 ID가 이력에 이미 있으면 되살리지 않는다 (`deleted`).

요청
    행동 단계의 `VlmClient`(`OpenAICompatibleVlm`, CPU·CI는 `OracleVlm`)로 그 행동 구간의 블러본
    프레임과 함께 묻는다. 프롬프트와 응답 스키마는 손마다 검수자가 확정한 의미 필드 전부
    (`MEANING_FIELDS`: 동사·대상·대상 부위·도구·방향·전후 상태, 설명 따르기가 의미 변경으로 보는
    필드)를 싣는다. 응답은
    label=action, 손마다 그 조합을 그대로 되돌린 `hands`, 필수 description이다 (`DescribeAnswer`).
    검증은 손 단위 조합으로 한다: 양손 행동에서 한 손의 동사와 다른 손의 대상·도구를 섞으면 값 집합
    안이어도 위반이다 (F-PAR-21). 위반이면 다시 묻고(정책 `vlm.max_retries`), 끝내 위반이면 만들지
    않는다 (`fallbacks`, 다음 실행이 다시 묻는다).
    서버 장애는 정책 백오프 뒤 `VlmUnavailableError`를 올린다 (호출자가 트랜잭션을 되돌린다).

기록과 출처 (감사)
    설명 레코드 = 모델 출처, `model_version` = `describe-<행동 정책 표시 stamp>+<VLM 버전>`
    (`describe_version`, 접두사 `dlp_schema.description_follow.REGENERATED_PREFIX`), 근거 INFERRED,
    stream_id 없음(마스터 타임라인), 구간 = 주인 레코드 구간의 합. 라벨 ID =
    `derived_id(action_id, "ko<version_tag>-i<주인 라벨 ID 해시>")`라 같은 버전·같은 고친 레코드면
    같다. 쓴 뒤 `apply_description_follow`로 고친 행동의 검수를 물려받는다 (사용자 정책 "행동을
    승인할 때 같이"): 검수자 = 행동 검수자, 검수 시각 = 설명 생성 시각. 그래서 이 설명은
    "검수 뒤 기계가 만든 글"임이 출처(`describe-` 버전)·생성 시각(행동 검수 뒤)·검수 시각 = 생성
    시각으로 남고, 물려준 행동은 물려받기 기록(`description_inheritance`, 경로 describe)에 남는다.
    영어 번역은 번역 단계가 평소대로 만들고 같은 규칙으로 물려받는다.

잠금
    이력 읽기·VLM 호출은 세션을 잠그지 않는다. 쓰기 직전에 세션을 잠그고, 사용 중지·승인
    보류·(프레임을 보냈으면) 렌더 해시를 다시 본다 (`dlp_actions.write_check.write_blocker`,
    F-PAR-23). 막히면 아무것도 쓰지 않고 `blocked`로 알린다. 아니면 설명 따르기를 먼저 적용한 뒤
    다시 계획해, 그 사이 바뀐 행동의 결과는 버린다 (번역 단계와 같은 방식).

DB: `label_records` 읽기·쓰기(추가만, 검수 상태 갱신은 설명 따르기만). 트랜잭션은 호출자(CLI)가
연다.
저장소: 라벨링 버킷의 블러본 읽기 (지금 승인된 렌더인지 확인한다, ADR 0024).
"""

from __future__ import annotations

import json
import tempfile
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import sqlalchemy as sa
from pydantic import Field, ValidationError

from dlp_actions.policy import ActionsPolicy
from dlp_actions.runner import input_digest
from dlp_actions.vlm import SegmentRequest, VlmClient
from dlp_actions.write_check import fetch_current_blurred, write_blocker
from dlp_media.storage import ObjectStore
from dlp_models.vlm import ask_validated, with_backoff
from dlp_privacy.policy import PrivacyPolicy
from dlp_privacy.policy import load_policy as load_privacy_policy
from dlp_schema import repo_root
from dlp_schema.common import Contract, derived_id
from dlp_schema.db.description_repository import apply_description_follow
from dlp_schema.db.locks import lock_session
from dlp_schema.db.repository import (
    get_description_inheritance,
    get_labels,
    get_session,
    insert_labels,
)
from dlp_schema.description_follow import (
    REGENERATED_PREFIX,
    human_confirmed,
    plan_description_follow,
)
from dlp_schema.episode import current_labels, non_operational_ids, retractions, version_tag
from dlp_schema.labels import (
    ActionPayload,
    DescriptionPayload,
    Evidence,
    Hand,
    LabelRecord,
    Provenance,
    Source,
    VerificationState,
)
from dlp_schema.ontology import Ontology
from dlp_schema.review import DescriptionInheritance, InheritanceMechanism


@dataclass(frozen=True)
class DescribeTarget:
    """설명을 다시 만들 고친 행동 하나.

    Attributes:
        action_id: 행동 ID (설명의 segment_id).
        owners: 현재 주인 레코드 (모두 사람이 확인, 손 순).
        label_id: 만들 설명 라벨 ID (`description_id`).
        stale: 성공하면 지울 이 단계의 이전 검수 전 설명.
    """

    action_id: str
    owners: tuple[LabelRecord, ...]
    label_id: str
    stale: tuple[LabelRecord, ...] = ()


@dataclass(frozen=True)
class DescribePlan:
    """세션의 계획 (`plan_describe`)."""

    targets: list[DescribeTarget]
    waiting: int
    deleted: int


@dataclass
class DescribeSummary:
    """실행 요약 (CLI가 출력한다).

    Attributes:
        version: 이번 설명 버전.
        described: 새 설명을 만든 행동 ID.
        fallbacks: 재시도 후에도 응답이 규칙을 어겨 만들지 못한 행동 ID (다음 실행이 다시 묻는다).
        waiting: 사람 확인을 기다리는 고친 행동 수.
        deleted: 이 버전 설명이 이력에 있으나 현재가 아니라 되살리지 않은 행동 수.
        retracted: 쓴 삭제 레코드 수 (이전 검수 전 설명, 설명 따르기 포함).
        inherited: 행동 검수를 물려받은 설명 수.
        blocked: 쓰기 직전 확인(사용 중지·승인 보류·렌더 변경)이 막은 이유. 있으면 아무것도 쓰지
            않았다 (F-PAR-23).
    """

    version: str
    described: list[str] = field(default_factory=list[str])
    fallbacks: list[str] = field(default_factory=list[str])
    waiting: int = 0
    deleted: int = 0
    retracted: int = 0
    inherited: int = 0
    blocked: str | None = None


def describe_version(policy: ActionsPolicy, vlm_version: str) -> str:
    """설명 버전 `describe-<행동 정책 해시>+<VLM 버전>`."""
    return f"{REGENERATED_PREFIX}{policy.stamp}+{vlm_version}"


def description_id(action_id: str, owners: Sequence[LabelRecord], version: str) -> str:
    """버전·고친 주인 레코드마다 결정적인 설명 라벨 ID."""
    return derived_id(action_id, f"ko{version_tag(version)}-i{input_digest(list(owners))}")


def _ours(x: LabelRecord) -> bool:
    """이 모듈이 만든 설명인지 (모델 출처, `describe-` 버전)."""
    return x.provenance.source is Source.MODEL and (x.provenance.model_version or "").startswith(
        REGENERATED_PREFIX
    )


def _replaceable(x: LabelRecord) -> bool:
    """이 모듈이 만든 검수 전 설명 (새 설명으로 바꿀 수 있다)."""
    return _ours(x) and x.verification.state is VerificationState.UNREVIEWED


def _touched(labels: Sequence[LabelRecord], excluded: set[str]) -> set[str]:
    """사람 레코드(수정·삭제·새로 그림)가 이력에 있는 행동 ID."""
    return {
        x.payload.action_id
        for x in labels
        if x.label_id not in excluded
        and x.provenance.source is Source.HUMAN
        and isinstance(x.payload, ActionPayload)
    }


def _originals(current: Sequence[LabelRecord], gone: set[str]) -> dict[str, list[LabelRecord]]:
    """행동 ID → 현재 원문 설명 (번역이 아님). 설명 따르기가 지울 것(gone)은 뺀다."""
    out: dict[str, list[LabelRecord]] = defaultdict(list)
    for x in current:
        p = x.payload
        if (
            isinstance(p, DescriptionPayload)
            and p.source_label_id is None
            and x.label_id not in gone
        ):
            out[p.segment_id].append(x)
    return out


def plan_describe(
    history: Sequence[LabelRecord],
    version: str,
    inheritance: Sequence[DescriptionInheritance],
) -> DescribePlan:
    """다시 만들 설명 (모듈 docstring "대상"). 순수 함수.

    Args:
        history: 세션의 라벨 이력 전체 (`get_labels`).
        version: 이번 설명 버전 (`describe_version`).
        inheritance: 세션의 물려받기 기록 (설명 따르기가 지울 물려받은 설명을 가린다). 기본값이
            없다: 빠뜨리면 물려받은 설명을 직접 검수로 보아 대상을 놓친다 (F-PAR-24). DB에서는
            `plan_session_describe`를 쓴다.
    """
    labels = list(history)
    excluded = non_operational_ids(labels)
    current = [x for x in current_labels(labels) if x.label_id not in excluded]
    gone = {d.label_id for d in plan_description_follow(labels, inheritance).retract}
    owners: dict[str, list[LabelRecord]] = defaultdict(list)
    for x in current:
        if isinstance(x.payload, ActionPayload):
            owners[x.payload.action_id].append(x)
    touched, originals = _touched(labels, excluded), _originals(current, gone)
    ever = {x.label_id for x in labels}
    targets: list[DescribeTarget] = []
    waiting = deleted = 0
    for action_id, found in sorted(owners.items()):
        mine = tuple(sorted(found, key=lambda o: (_hand(o).value, o.label_id)))
        ds = originals.get(action_id, [])
        if action_id not in touched or any(not _replaceable(d) for d in ds):
            continue
        new_id = description_id(action_id, mine, version)
        if any(d.label_id == new_id for d in ds):
            continue
        if not all(human_confirmed(o) for o in mine):
            waiting += 1
        elif new_id in ever:
            deleted += 1
        else:
            targets.append(DescribeTarget(action_id, mine, new_id, tuple(ds)))
    return DescribePlan(targets, waiting, deleted)


def plan_session_describe(conn: sa.Connection, session_id: str, version: str) -> DescribePlan:
    """DB의 세션 이력과 물려받기 기록으로 `plan_describe` (명령·재생성이 함께 쓴다, F-PAR-24)."""
    return plan_describe(
        get_labels(conn, session_id), version, get_description_inheritance(conn, session_id)
    )


def _hand(x: LabelRecord) -> Hand:
    """행동 레코드의 손."""
    assert isinstance(x.payload, ActionPayload)
    return x.payload.hand


def _actions(target: DescribeTarget) -> list[ActionPayload]:
    """주인 레코드의 행동 페이로드 (손 순)."""
    return [o.payload for o in target.owners if isinstance(o.payload, ActionPayload)]


def _span(target: DescribeTarget) -> tuple[int, int]:
    """주인 레코드 구간의 합 (마스터 ms)."""
    return min(o.t_start_ms for o in target.owners), max(o.t_end_ms for o in target.owners)


def describe_request(
    target: DescribeTarget, session_id: str, stream_id: str, video: Path | None
) -> SegmentRequest:
    """고친 행동 구간의 VLM 요청 (첫 손 기준, 개체 후보는 확정한 대상·도구)."""
    acts = _actions(target)
    start, end = _span(target)
    entities = sorted({e for a in acts for e in (a.target_id, a.tool_id) if e is not None})
    return SegmentRequest(
        session_id=session_id,
        stream_id=stream_id,
        video=video,
        hand=acts[0].hand,
        start_ms=start,
        end_ms=end,
        in_contact=any(a.t_contact_start_ms is not None or a.contact_held for a in acts),
        entities=tuple(entities),
    )


# 손마다 응답이 되돌려야 하는 행동 의미 필드 (설명 따르기 `meaning`이 비교하는 필드 중 손·시각
# 밖 전부)
MEANING_FIELDS = (
    "verb", "target_id", "target_body_part", "tool_id", "direction", "pre_state", "post_state",
)  # fmt: skip


class HandMeaning(Contract):
    """응답이 손마다 되돌리는 확정 의미 (`MEANING_FIELDS` + 손)."""

    hand: Hand
    verb: str
    target_id: str | None = None
    target_body_part: str | None = None
    tool_id: str | None = None
    direction: str | None = None
    pre_state: dict[str, str] = Field(default_factory=dict[str, str])
    post_state: dict[str, str] = Field(default_factory=dict[str, str])


class DescribeAnswer(Contract):
    """설명 재생성 응답: label=action, 손마다 확정 의미, 한국어 한 문장 설명(1~200자), 신뢰도."""

    label: Literal["action"]
    hands: list[HandMeaning]
    description: str = Field(min_length=1, max_length=200)
    confidence: float | None = Field(default=None, ge=0, le=1)


def confirmed(target: DescribeTarget) -> list[dict[str, Any]]:
    """손 순으로 검수자가 확정한 의미 (손 + `MEANING_FIELDS`, JSON 값)."""
    out: list[dict[str, Any]] = []
    for a in _actions(target):
        data = a.model_dump(mode="json")
        out.append({"hand": data["hand"], **{k: data[k] for k in MEANING_FIELDS}})
    return out


def _const(value: Any) -> dict[str, Any]:
    """값 하나만 허용하는 스키마 (사전은 키마다 값 하나)."""
    if isinstance(value, dict):
        items: dict[str, Any] = dict(value)  # pyright: ignore[reportUnknownArgumentType]
        return {
            "type": "object",
            "additionalProperties": False,
            "required": sorted(items),
            "properties": {k: {"enum": [v]} for k, v in sorted(items.items())},
        }
    return {"enum": [value]}


def describe_schema(target: DescribeTarget) -> dict[str, Any]:
    """응답 JSON Schema: label=action, hands = 손마다 확정한 의미 조합(순서·개수 고정), description
    필수(1~200자)."""
    hands = confirmed(target)
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["label", "hands", "description"],
        "properties": {
            "label": {"enum": ["action"]},
            "hands": {
                "type": "array",
                "minItems": len(hands),
                "maxItems": len(hands),
                "prefixItems": [
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "required": sorted(h),
                        "properties": {k: _const(v) for k, v in h.items()},
                    }
                    for h in hands
                ],
                "items": False,
            },
            "description": {"type": "string", "minLength": 1, "maxLength": 200},
            "confidence": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
        },
    }


def _name(table: dict[str, Any], key: str | None) -> str:
    """온톨로지 ID를 `ID: 한국어`로 (사전에 없으면 ID 그대로, None은 "없음")."""
    if key is None:
        return "없음"
    term = table.get(key)
    return f"{key}: {term.ko}" if term is not None else key


def _states(ontology: Ontology, states: dict[str, str]) -> str:
    """상태 속성 → 값 (없으면 "없음")."""
    if not states:
        return "없음"
    return ", ".join(
        f"{_name(ontology.state_attributes, k)} = {v}" for k, v in sorted(states.items())
    )


def describe_prompt(target: DescribeTarget, ontology: Ontology) -> str:
    """한국어 프롬프트: 검수자가 확정한 손마다 의미 필드 전부, 설명 작성 규칙.

    문구(`ask_description` 재시도 문구 포함)를 바꾸면 `dlp_actions.policy.PROMPT_VERSION`과
    actions.yaml prompt_version을 올린다 (설명 버전에 들어간다, F-PAR-84).
    """
    start, end = _span(target)
    lines: list[str] = []
    for a in _actions(target):
        hand = "오른손" if a.hand is Hand.RIGHT else "왼손"
        lines.append(
            f"- {hand}: 동작 {_name(ontology.verbs, a.verb)}, 대상 {a.target_id or '없음'}, "
            f"대상 부위 {_name(ontology.body_parts, a.target_body_part)}, "
            f"도구 {a.tool_id or '없음'}, 방향 {a.direction or '없음'}, "
            f"전 상태 {_states(ontology, a.pre_state)}, 후 상태 {_states(ontology, a.post_state)}"
        )
    return (
        f"1인칭 돌봄·청소 영상의 {start}~{end} ms 구간 프레임이다. 검수자가 이 구간의 행동을 "
        "다음과 같이 확정했다:\n" + "\n".join(lines) + "\n\n"
        "이 확정된 행동(동작·대상·대상 부위·도구·방향·전후 상태)을 한국어 한 문장 지시문으로 "
        "description에 쓴다 (예: '걸레로 세면대를 문지른다'). label은 action이고 hands에는 "
        "손마다 위 값을 그대로 쓴다. 값을 바꾸거나 손 사이에 섞지 않는다.\nJSON 하나만 출력한다."
    )


def validate_description(
    raw: str, target: DescribeTarget
) -> tuple[DescribeAnswer | None, list[str]]:
    """응답 검사: JSON → 형식 → 손마다 확정한 의미 조합 그대로 (손 단위, 섞임·누락·중복 위반)."""
    try:
        answer = DescribeAnswer.model_validate(json.loads(raw))
    except json.JSONDecodeError as exc:
        return None, [f"JSON이 아닙니다: {exc.msg}"]
    except ValidationError as exc:
        return None, [f"스키마 위반: {e['loc']} {e['msg']}" for e in exc.errors()]
    want = confirmed(target)
    got = [h.model_dump(mode="json") for h in answer.hands]
    errors: list[str] = []
    if sorted(map(_key, got)) != sorted(map(_key, want)):
        errors.append(f"손마다 확정한 행동과 다릅니다: {got} (확정 {want})")
    if not answer.description.strip():
        errors.append("description이 비었습니다")
    return (None, errors) if errors else (answer, [])


def _key(hand: dict[str, Any]) -> str:
    """손 조합 비교 키 (정렬된 JSON)."""
    return json.dumps(hand, sort_keys=True, ensure_ascii=False)


def ask_description(
    client: VlmClient,
    target: DescribeTarget,
    request: SegmentRequest,
    ontology: Ontology,
    policy: ActionsPolicy,
) -> DescribeAnswer | None:
    """설명만 묻는다 (위반은 `vlm.max_retries`번, 장애는 `vlm.unavailable_backoff_s`로 다시).

    Returns:
        검증된 응답, 끝내 위반이면 None.

    Raises:
        VlmUnavailableError: 백오프 후에도 서버가 응답하지 않을 때.
        VlmRequestError: 설정 오류 (4xx).
    """
    schema, base = describe_schema(target), describe_prompt(target, ontology)
    errors: list[str] = []

    def ask(problems: Sequence[str]) -> str:
        prompt = base
        if problems:
            prompt += "\n\n앞 응답이 규칙을 어겼다: " + "; ".join(problems)
            prompt += "\n확정한 값으로 JSON 하나만 다시 출력한다."
        return with_backoff(
            lambda: client.complete(request, prompt, schema), policy.vlm.unavailable_backoff_s
        )

    answer, _ = ask_validated(
        ask, lambda raw: validate_description(raw, target), policy.vlm.max_retries, errors
    )
    return answer


def _record(
    target: DescribeTarget,
    session_id: str,
    answer: DescribeAnswer,
    policy: ActionsPolicy,
    version: str,
    now: datetime,
) -> LabelRecord:
    """설명 레코드 (모듈 docstring "기록과 출처")."""
    start, end = _span(target)
    confidence = (
        answer.confidence if answer.confidence is not None else policy.vlm.default_confidence
    )
    return LabelRecord(
        label_id=target.label_id,
        session_id=session_id,
        t_start_ms=start,
        t_end_ms=end,
        ontology_version=target.owners[0].ontology_version,
        provenance=Provenance(source=Source.MODEL, model_version=version),
        evidence=Evidence.INFERRED,
        confidence=round(confidence, 4),
        created_at=now,
        payload=DescriptionPayload(segment_id=target.action_id, text=answer.description.strip()),
    )


def describe_session(
    conn: sa.Connection,
    session_id: str,
    client: VlmClient,
    ontology: Ontology,
    policy: ActionsPolicy,
    now: datetime,
    labeling: ObjectStore | None = None,
) -> DescribeSummary:
    """세션의 고친 행동마다 설명을 다시 만든다 (모듈 docstring).

    Args:
        conn: 열린 트랜잭션의 DB 연결 (호출자가 커밋·롤백한다).
        client: 행동 단계 VLM 클라이언트 (`version`이 설명 버전에 들어간다).
        ontology: 세션 온톨로지 (동작 한국어 이름).
        policy: 행동 정책 (해시가 설명 버전에 들어간다, VLM 재시도·백오프·기본 신뢰도).
        now: 생성 시각 (시간대 필수).
        labeling: 라벨링 버킷. 주면 블러본 프레임을 보낸다. None이면 프레임 없이 묻는다 (테스트).

    Raises:
        VlmUnavailableError, VlmRequestError: 서버 장애·설정 오류 (호출자가 되돌린다).
        RenderNotCurrentError: 블러본이 지금 승인된 렌더가 아닐 때.

    사용 중지·승인 보류 세션은 묻지 않고, 묻는 동안 그렇게 됐거나 렌더가 바뀌었으면 쓰기 직전
    확인(`write_blocker`, 세션 잠금 아래)이 막아 아무것도 쓰지 않는다 (`blocked`, F-PAR-23).

    부작용: `label_records`에 설명·삭제 레코드 추가, 설명 따르기(검수 상태 기록·삭제). VLM 호출.
    """
    version = describe_version(policy, client.version)
    summary = DescribeSummary(version)
    plan = plan_session_describe(conn, session_id, version)
    summary.waiting, summary.deleted = plan.waiting, plan.deleted
    if not plan.targets:
        return summary
    summary.blocked = write_blocker(conn, session_id)
    if summary.blocked is not None:
        return summary
    stream_id = get_session(conn, session_id).reference_stream.stream_id
    answers: dict[str, DescribeAnswer] = {}
    render: tuple[str, str, PrivacyPolicy] | None = None
    with tempfile.TemporaryDirectory() as tmp:
        video: Path | None = None
        if labeling is not None:
            privacy = load_privacy_policy(repo_root())
            video, before = fetch_current_blurred(
                conn, labeling, session_id, stream_id, Path(tmp), privacy
            )
            render = (stream_id, before, privacy)
        for t in plan.targets:
            request = describe_request(t, session_id, stream_id, video)
            answer = ask_description(client, t, request, ontology, policy)
            if answer is None:
                summary.fallbacks.append(t.action_id)
            else:
                answers[t.label_id] = answer
    _write(conn, session_id, answers, policy, version, now, summary, render)
    return summary


def _write(
    conn: sa.Connection,
    session_id: str,
    answers: dict[str, DescribeAnswer],
    policy: ActionsPolicy,
    version: str,
    now: datetime,
    summary: DescribeSummary,
    render: tuple[str, str, PrivacyPolicy] | None,
) -> None:
    """잠그고 쓰기 직전 확인을 한 뒤(막히면 아무것도 쓰지 않음), 설명 따르기를 먼저 적용하고 다시
    계획해 아직 대상인 것만 쓰고 검수를 물려받는다."""
    lock_session(conn, session_id)
    summary.blocked = write_blocker(conn, session_id, render)
    if summary.blocked is not None:
        return
    first = apply_description_follow(conn, session_id, now, InheritanceMechanism.DESCRIBE)
    fresh = plan_session_describe(conn, session_id, version)
    new: list[LabelRecord] = []
    stale: list[LabelRecord] = []
    for t in fresh.targets:
        answer = answers.get(t.label_id)
        if answer is not None:
            new.append(_record(t, session_id, answer, policy, version, now))
            stale += t.stale
            summary.described.append(t.action_id)
    removed = retractions(stale, version, now)
    insert_labels(conn, [*removed, *new])
    follow = apply_description_follow(conn, session_id, now, InheritanceMechanism.DESCRIBE)
    summary.inherited = len(first.approve) + len(follow.approve)
    summary.retracted = len(removed) + len(first.retract) + len(follow.retract)
