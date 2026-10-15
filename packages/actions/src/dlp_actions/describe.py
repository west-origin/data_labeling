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
    프레임과 함께 묻는다. 응답 스키마는 행동 단계와 같은 모양이지만 label=action, verb·target_id·
    tool_id를 검수자가 확정한 값 하나로 묶고 description을 필수로 한다. 그 값과 다르면 위반으로 다시
    묻고(정책 `vlm.max_retries`), 끝내 위반이면 만들지 않는다 (`fallbacks`, 다음 실행이 다시
    묻는다).
    서버 장애는 정책 백오프 뒤 `VlmUnavailableError`를 올린다 (호출자가 트랜잭션을 되돌린다).

기록과 출처 (감사)
    설명 레코드 = 모델 출처, `model_version` = `describe-<행동 정책 해시>+<VLM 버전>`
    (`describe_version`, 접두사 `dlp_schema.description_follow.REGENERATED_PREFIX`), 근거 INFERRED,
    stream_id 없음(마스터 타임라인), 구간 = 주인 레코드 구간의 합. 라벨 ID =
    `derived_id(action_id, "ko<version_tag>-i<주인 라벨 ID 해시>")`라 같은 버전·같은 고친 레코드면
    같다. 쓴 뒤 `apply_description_follow`로 고친 행동의 검수를 물려받는다 (사용자 정책 "행동을
    승인할 때 같이"): 검수자 = 행동 검수자, 검수 시각 = 설명 생성 시각. 그래서 이 설명은
    "검수 뒤 기계가 만든 글"임이 출처(`describe-` 버전)·생성 시각(행동 검수 뒤)·검수 시각 = 생성
    시각으로 남고, 물려준 행동은 `inherited_owner`로 다시 계산한다. 영어 번역은 번역 단계가 평소대로
    만들고 같은 규칙으로 물려받는다.

잠금
    이력 읽기·VLM 호출은 세션을 잠그지 않는다. 쓰기 직전에 세션을 잠그고 설명 따르기를 먼저 적용한
    뒤 다시 계획해, 그 사이 바뀐 행동의 결과는 버린다 (번역 단계와 같은 방식).

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
from typing import Any

import sqlalchemy as sa
from pydantic import ValidationError

from dlp_actions.policy import ActionsPolicy
from dlp_actions.runner import input_digest
from dlp_actions.vlm import SegmentRequest, VlmAnswer, VlmClient
from dlp_media.storage import ObjectStore, blurred_key
from dlp_models.vlm import ask_validated, with_backoff
from dlp_privacy.policy import load_policy as load_privacy_policy
from dlp_privacy.runner import assert_render_current, check_fetched
from dlp_schema import repo_root
from dlp_schema.common import derived_id
from dlp_schema.db.description_repository import apply_description_follow
from dlp_schema.db.locks import lock_session
from dlp_schema.db.repository import get_labels, get_session, insert_labels
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
    """

    version: str
    described: list[str] = field(default_factory=list[str])
    fallbacks: list[str] = field(default_factory=list[str])
    waiting: int = 0
    deleted: int = 0
    retracted: int = 0
    inherited: int = 0


def describe_version(policy: ActionsPolicy, vlm_version: str) -> str:
    """설명 버전 `describe-<행동 정책 해시>+<VLM 버전>`."""
    return f"{REGENERATED_PREFIX}{policy.digest}+{vlm_version}"


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


def plan_describe(history: Sequence[LabelRecord], version: str) -> DescribePlan:
    """다시 만들 설명 (모듈 docstring "대상"). 순수 함수.

    Args:
        history: 세션의 라벨 이력 전체 (`get_labels`).
        version: 이번 설명 버전 (`describe_version`).
    """
    labels = list(history)
    excluded = non_operational_ids(labels)
    current = [x for x in current_labels(labels) if x.label_id not in excluded]
    gone = {d.label_id for d in plan_description_follow(labels).retract}
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


def _allowed(acts: Sequence[ActionPayload]) -> tuple[list[str], list[str | None], list[str | None]]:
    """확정한 동사·대상·도구 값 (응답이 고를 수 있는 값)."""
    verbs = sorted({a.verb for a in acts})
    targets = sorted({a.target_id for a in acts}, key=lambda v: (v is None, v or ""))
    tools = sorted({a.tool_id for a in acts}, key=lambda v: (v is None, v or ""))
    return verbs, targets, tools


def describe_schema(target: DescribeTarget) -> dict[str, Any]:
    """응답 JSON Schema: label=action, 동사·대상·도구는 확정한 값, description 필수(1~200자)."""
    verbs, targets, tools = _allowed(_actions(target))
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["label", "verb", "description"],
        "properties": {
            "label": {"enum": ["action"]},
            "verb": {"enum": verbs},
            "gap_type": {"enum": [None]},
            "target_id": {"enum": targets},
            "tool_id": {"enum": tools},
            "description": {"type": "string", "minLength": 1, "maxLength": 200},
            "confidence": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
        },
    }


def describe_prompt(target: DescribeTarget, ontology: Ontology) -> str:
    """한국어 프롬프트: 검수자가 확정한 손마다 동작(ID: 한국어)·대상·도구, 설명 작성 규칙."""
    start, end = _span(target)
    lines: list[str] = []
    for a in _actions(target):
        hand = "오른손" if a.hand is Hand.RIGHT else "왼손"
        verb = ontology.verbs.get(a.verb)
        name = f"{a.verb}: {verb.ko}" if verb is not None else a.verb
        lines.append(
            f"- {hand}: 동작 {name}, 대상 {a.target_id or '없음'}, 도구 {a.tool_id or '없음'}"
        )
    return (
        f"1인칭 돌봄·청소 영상의 {start}~{end} ms 구간 프레임이다. 검수자가 이 구간의 행동을 "
        "다음과 같이 확정했다:\n" + "\n".join(lines) + "\n\n"
        "이 확정된 행동을 한국어 한 문장 지시문으로 description에 쓴다 "
        "(예: '걸레로 세면대를 문지른다'). label은 action이고 verb·target_id·tool_id는 위 값 "
        "그대로 쓴다. 동작·대상·도구를 바꾸지 않는다.\nJSON 하나만 출력한다."
    )


def validate_description(raw: str, target: DescribeTarget) -> tuple[VlmAnswer | None, list[str]]:
    """응답 검사: JSON → 형식 → label=action → 확정한 동사·대상·도구 → 설명이 비지 않음."""
    try:
        answer = VlmAnswer.model_validate(json.loads(raw))
    except json.JSONDecodeError as exc:
        return None, [f"JSON이 아닙니다: {exc.msg}"]
    except ValidationError as exc:
        return None, [f"스키마 위반: {e['loc']} {e['msg']}" for e in exc.errors()]
    verbs, targets, tools = _allowed(_actions(target))
    errors: list[str] = []
    if answer.label != "action" or answer.verb not in verbs:
        errors.append(f"확정한 동작과 다릅니다: {answer.label}/{answer.verb} (허용 {verbs})")
    if answer.target_id not in targets or answer.tool_id not in tools:
        errors.append(f"확정한 대상·도구와 다릅니다: {answer.target_id}, {answer.tool_id}")
    if not (answer.description or "").strip():
        errors.append("description이 비었습니다")
    return (None, errors) if errors else (answer, [])


def ask_description(
    client: VlmClient,
    target: DescribeTarget,
    request: SegmentRequest,
    ontology: Ontology,
    policy: ActionsPolicy,
) -> VlmAnswer | None:
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
    answer: VlmAnswer,
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
        payload=DescriptionPayload(
            segment_id=target.action_id, text=(answer.description or "").strip()
        ),
    )


def _fetch_video(
    conn: sa.Connection, labeling: ObjectStore, session_id: str, stream_id: str, work: Path
) -> Path:
    """지금 승인된 렌더의 블러본을 받는다 (ADR 0024, 아니면 RenderNotCurrentError)."""
    video = work / f"{stream_id}.mp4"
    rendered = assert_render_current(
        conn, labeling, session_id, stream_id, load_privacy_policy(repo_root())
    )
    labeling.get_file(blurred_key(session_id, stream_id), video)
    check_fetched(video, rendered, session_id, stream_id)
    return video


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

    부작용: `label_records`에 설명·삭제 레코드 추가, 설명 따르기(검수 상태 기록·삭제). VLM 호출.
    """
    version = describe_version(policy, client.version)
    summary = DescribeSummary(version)
    plan = plan_describe(get_labels(conn, session_id), version)
    summary.waiting, summary.deleted = plan.waiting, plan.deleted
    if not plan.targets:
        return summary
    stream_id = get_session(conn, session_id).reference_stream.stream_id
    answers: dict[str, VlmAnswer] = {}
    with tempfile.TemporaryDirectory() as tmp:
        video = (
            None
            if labeling is None
            else _fetch_video(conn, labeling, session_id, stream_id, Path(tmp))
        )
        for t in plan.targets:
            request = describe_request(t, session_id, stream_id, video)
            answer = ask_description(client, t, request, ontology, policy)
            if answer is None:
                summary.fallbacks.append(t.action_id)
            else:
                answers[t.label_id] = answer
    _write(conn, session_id, answers, policy, version, now, summary)
    return summary


def _write(
    conn: sa.Connection,
    session_id: str,
    answers: dict[str, VlmAnswer],
    policy: ActionsPolicy,
    version: str,
    now: datetime,
    summary: DescribeSummary,
) -> None:
    """잠그고 설명 따르기를 먼저 적용한 뒤 다시 계획해, 아직 대상인 것만 쓰고 검수를 물려받는다."""
    lock_session(conn, session_id)
    first = apply_description_follow(conn, session_id, now)
    fresh = plan_describe(get_labels(conn, session_id), version)
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
    follow = apply_description_follow(conn, session_id, now)
    summary.inherited = len(first.approve) + len(follow.approve)
    summary.retracted = len(removed) + len(first.retract) + len(follow.retract)
