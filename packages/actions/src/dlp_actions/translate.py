"""한국어 설명 라벨의 영어 번역 단계 (`dlp translate run`, ADR 0043). 멱등이다.

세션의 현재 운영 설명 라벨(`current_labels`) 중 원문 언어(`translate.yaml source_language`, ko)이고
번역이 아닌 것(source_label_id 없음)마다 번역 언어(en) 형제 레코드가 있도록 한다.

데이터 모델
    번역 레코드 = 모델 출처 `DescriptionPayload(segment_id=원문과 같음, text=영어,
    language=en, source_label_id=원문 라벨 ID)`. parent_label_id는 쓰지 않는다 (parent로
    이으면 `current_labels`가 원문을 "고쳐진 이전 레코드"로 보고 운영 라벨에서 뺀다).
    원문과 번역이 둘 다 현재 운영 라벨이다. 라벨 ID는
    `derived_id(원문 ID, "<언어><version_tag(번역 버전)>")`라 원문·버전마다 결정적이다.
    번역 버전 = `translate-<정책 해시>+<번역기 버전>+g<용어집 해시>` (`translation_version`).

다시 돌릴 때 (원문 S마다, `plan_translation`)
    - S를 가리키는 현재 번역이 이 버전이거나 검수된 것이면(사람 출처·승인·표본 검증) 됐다.
    - 같은 구간(segment_id)에 검수된 번역 언어 설명이 있으면 새로 만들지 않는다 (검수된
      라벨과 겹치는 새 모델 결과를 버리는 행동 단계와 같은 규칙. 원문이 바뀌어 낡았을 수
      있어 `kept_reviewed`로 알린다).
    - 이 버전 번역 ID가 이력에 이미 있으면(검수자가 지움) 되살리지 않는다.
    - 그 밖은 번역한다. 성공하면 S를 가리키던 다른 버전의 검수 전 번역을 삭제 레코드로
      지운다 (실패하면 이전 번역을 그대로 둔다).
    - 원문이 더는 현재 원문이 아닌(고쳐졌거나 지워진) 검수 전 번역은 지운다 (`stale`).
      검수된 번역은 어떤 경우에도 지우지 않는다 (ADR 0015).

실패 처리
    번역기 예외는 이 함수 밖으로 내지 않는다. 서버 장애·설정 오류면 그 세션의 남은 원문을
    묻지 않고 `pending`에 남기고, 응답 위반이면 그 원문만 남긴다 (`invalid`). 나중에
    `dlp translate run --pending`이 채운다. DB 오류는 그대로 올린다.

잠금
    이력 읽기·번역(서버 호출)은 세션을 잠그지 않는다. 쓰기 직전에 세션을 잠그고 이력을 다시
    읽어 계획을 다시 세운 뒤, 그 사이 낡은 결과는 버리고 쓴다 (긴 세션 잠금을 피한다, R3).

검수 물려받기 (F-PAR-15)
    쓴 뒤 `apply_description_follow`로 설명이 행동·상위 구간 검수를 따르게 한다. 행동이 이미 승인·
    표본 검증됐으면 새 번역도 같은 검수자·상태로 기록한다(검수 시각은 번역 생성 시각). 행동이
    검수 전이면 번역도 미검수로 남는다.

DB: `label_records` 읽기·쓰기(추가만, 검수 상태 갱신은 위 물려받기만). 트랜잭션은 호출자(CLI)가
연다.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

import sqlalchemy as sa

from dlp_actions.translate_policy import TranslatePolicy
from dlp_actions.translators import (
    GlossaryEntry,
    TranslationConfigError,
    TranslationInvalidError,
    TranslationUnavailableError,
    Translator,
)
from dlp_schema.common import derived_id
from dlp_schema.db.description_repository import apply_description_follow
from dlp_schema.db.locks import lock_session
from dlp_schema.db.repository import get_labels, insert_labels
from dlp_schema.episode import current_labels, is_protected, retractions, version_tag
from dlp_schema.labels import DescriptionPayload, LabelRecord, Provenance, Source
from dlp_schema.ontology import Ontology
from dlp_schema.review import InheritanceMechanism

# 이 단계 모델 버전의 접두사
PREFIX = "translate-"
# 이 단계가 읽고 쓰는 라벨 종류
KINDS = ("description",)


@dataclass(frozen=True)
class TranslationPlan:
    """한 세션의 번역 계획 (`plan_translation`, 순수 계산).

    Attributes:
        todo: (번역할 원문, 성공하면 지울 그 원문의 다른 버전 검수 전 번역) 목록.
        stale: 원문이 더는 현재 원문이 아닌 검수 전 번역 (지운다).
        covered: 이미 이 버전 또는 검수된 번역이 있는 원문 수.
        kept_reviewed: 같은 구간에 검수된 번역 언어 설명이 있어 새로 만들지 않은 원문.
        deleted: 이 버전 번역이 이력에 있으나 현재 라벨이 아닌(검수자가 지운) 원문 수.
    """

    todo: list[tuple[LabelRecord, list[LabelRecord]]]
    stale: list[LabelRecord]
    covered: int
    kept_reviewed: list[LabelRecord]
    deleted: int


@dataclass
class TranslateSummary:
    """실행 요약 (CLI가 출력한다).

    Attributes:
        version: 번역 버전.
        translated: 새로 만든 번역 수.
        covered: 이미 번역이 있던 원문 수.
        retracted: 쓴 삭제 레코드 수 (낡은 번역·버전 교체).
        kept_reviewed: 검수된 번역이 있어 새로 만들지 않은 원문 ID.
        deleted: 검수자가 지워 되살리지 않은 원문 수.
        pending: 번역하지 못해 남은 원문 라벨 ID (서버 장애·설정 오류·응답 위반).
        invalid: 그중 응답 위반인 원문 라벨 ID.
        stopped: 남은 원문을 묻지 않고 멈춘 이유 ("unavailable" | "config", 없으면 None).
        error: 멈춘 원인 메시지 (서버 응답 앞부분, 키는 들어가지 않는다).
        inherited: 행동·상위 구간 검수를 물려받은 설명 수 (F-PAR-15, 승인된 행동의 새 번역 등).
    """

    version: str
    translated: int = 0
    covered: int = 0
    retracted: int = 0
    kept_reviewed: list[str] = field(default_factory=list[str])
    deleted: int = 0
    pending: list[str] = field(default_factory=list[str])
    invalid: list[str] = field(default_factory=list[str])
    stopped: str | None = None
    error: str | None = None
    inherited: int = 0


def is_machine_translation(x: LabelRecord) -> bool:
    """이 단계가 만든 번역 레코드인지 (모델 출처, `translate-` 버전, 원문 참조 있음)."""
    return (
        isinstance(x.payload, DescriptionPayload)
        and x.payload.source_label_id is not None
        and x.provenance.source is Source.MODEL
        and (x.provenance.model_version or "").startswith(PREFIX)
    )


def glossary_entries(ontology: Ontology) -> list[GlossaryEntry]:
    """온톨로지 표시 이름 전체 → 용어집 (한국어 이름 기준 중복 제거, 이름순, 결정적)."""
    out: dict[str, str] = {}
    for _, name in sorted(ontology.display_names().items()):
        out.setdefault(name.ko, name.en)
    return [GlossaryEntry(ko, en) for ko, en in sorted(out.items())]


def glossary_for(text: str, entries: Sequence[GlossaryEntry], mode: str) -> list[GlossaryEntry]:
    """요청에 넣을 용어집: matched면 원문에 나온 한국어 이름만, all이면 전부, off면 없음."""
    if mode == "off":
        return []
    if mode == "all":
        return list(entries)
    return [e for e in entries if e.ko in text]


def translation_version(
    policy: TranslatePolicy, translator_version: str, ontology: Ontology
) -> str:
    """번역 버전 `translate-<정책 해시>+<번역기 버전>+g<용어집 해시 8자>`.

    용어집(온톨로지 한·영 이름)이 바뀌면 번역 결과가 달라지므로 버전에 넣는다. 용어집을 쓰지
    않는 정책(off)이면 용어집 해시는 고정값이다.
    """
    h = hashlib.sha256()
    if policy.backend.glossary != "off":
        for e in glossary_entries(ontology):
            h.update(f"{e.ko}\t{e.en}\n".encode())
    return f"{PREFIX}{policy.digest}+{translator_version}+g{h.hexdigest()[:8]}"


def translation_id(source: LabelRecord, policy: TranslatePolicy, version: str) -> str:
    """원문·버전마다 결정적인 번역 라벨 ID (`<원문 ID>:<언어><버전 태그>`, 128자 넘으면 축약)."""
    return derived_id(source.label_id, f"{policy.target_language}{version_tag(version)}")


def _descriptions(labels: Sequence[LabelRecord], language: str) -> list[LabelRecord]:
    """그 언어의 설명 라벨."""
    return [
        x
        for x in labels
        if isinstance(x.payload, DescriptionPayload) and x.payload.language == language
    ]


def _source_of(x: LabelRecord) -> str | None:
    """번역 레코드가 가리키는 원문 라벨 ID."""
    return x.payload.source_label_id if isinstance(x.payload, DescriptionPayload) else None


def _segment_of(x: LabelRecord) -> str:
    """설명 라벨의 구간 ID."""
    assert isinstance(x.payload, DescriptionPayload)
    return x.payload.segment_id


def plan_translation(
    history: Sequence[LabelRecord], policy: TranslatePolicy, version: str
) -> TranslationPlan:
    """번역 계획 (모듈 docstring "다시 돌릴 때"의 규칙). 순수 함수.

    Args:
        history: 세션의 설명 라벨 이력 전체 (`get_labels(kinds=["description"])`).
        policy: 번역 정책 (언어).
        version: 이번 번역 버전.
    """
    current = current_labels(list(history))
    sources = [x for x in _descriptions(current, policy.source_language) if _source_of(x) is None]
    targets = _descriptions(current, policy.target_language)
    by_source: dict[str, list[LabelRecord]] = defaultdict(list)
    for t in targets:
        if (sid := _source_of(t)) is not None:
            by_source[sid].append(t)
    reviewed_segments = {_segment_of(t) for t in targets if is_protected(t)}
    ever = {x.label_id for x in history}
    todo: list[tuple[LabelRecord, list[LabelRecord]]] = []
    kept: list[LabelRecord] = []
    covered = deleted = 0
    for s in sources:
        linked = by_source.get(s.label_id, [])
        if any(t.provenance.model_version == version or is_protected(t) for t in linked):
            covered += 1
        elif _segment_of(s) in reviewed_segments:
            kept.append(s)
        elif translation_id(s, policy, version) in ever:
            deleted += 1
        else:
            todo.append((s, [t for t in linked if is_machine_translation(t)]))
    source_ids = {s.label_id for s in sources}
    stale = [
        t
        for t in targets
        if is_machine_translation(t) and not is_protected(t) and _source_of(t) not in source_ids
    ]
    return TranslationPlan(todo, stale, covered, kept, deleted)


def untranslated(
    history: Sequence[LabelRecord], policy: TranslatePolicy, version: str
) -> list[LabelRecord]:
    """번역이 하나도 없는 현재 원문 (대기 목록, `dlp translate pending`). 순수 함수.

    다른 버전 번역만 있는 원문은 넣지 않는다 (번역은 있다. `dlp translate run`이 새 버전으로
    바꾼다).
    """
    return [s for s, olds in plan_translation(history, policy, version).todo if not olds]


def _record(
    source: LabelRecord, text: str, policy: TranslatePolicy, version: str, now: datetime
) -> LabelRecord:
    """번역 레코드 (구간·스트림·온톨로지 버전·근거는 원문과 같다)."""
    assert isinstance(source.payload, DescriptionPayload)
    return LabelRecord(
        label_id=translation_id(source, policy, version),
        session_id=source.session_id,
        stream_id=source.stream_id,
        t_start_ms=source.t_start_ms,
        t_end_ms=source.t_end_ms,
        ontology_version=source.ontology_version,
        provenance=Provenance(source=Source.MODEL, model_version=version),
        evidence=source.evidence,
        confidence=source.confidence
        if source.confidence is not None
        else policy.default_confidence,
        created_at=now,
        payload=DescriptionPayload(
            segment_id=source.payload.segment_id,
            text=text,
            language=policy.target_language,
            source_label_id=source.label_id,
        ),
    )


def _translate_all(
    plan: TranslationPlan,
    translator: Translator,
    entries: Sequence[GlossaryEntry],
    policy: TranslatePolicy,
    summary: TranslateSummary,
) -> dict[str, str]:
    """계획의 원문을 차례로 번역한다. 원문 라벨 ID → 번역문. 실패는 summary에 남긴다."""
    done: dict[str, str] = {}
    for source, _ in plan.todo:
        assert isinstance(source.payload, DescriptionPayload)
        if summary.stopped is not None:
            summary.pending.append(source.label_id)
            continue
        text = source.payload.text
        try:
            done[source.label_id] = translator.translate(
                text, glossary_for(text, entries, policy.backend.glossary)
            )
        except TranslationInvalidError:
            summary.invalid.append(source.label_id)
            summary.pending.append(source.label_id)
        except (TranslationUnavailableError, TranslationConfigError) as exc:
            unavailable = isinstance(exc, TranslationUnavailableError)
            summary.stopped = "unavailable" if unavailable else "config"
            summary.error = str(exc)[:200]
            summary.pending.append(source.label_id)
    return done


def run_translate(
    conn: sa.Connection,
    session_id: str,
    translator: Translator,
    ontology: Ontology,
    policy: TranslatePolicy,
    now: datetime,
) -> TranslateSummary:
    """세션의 한국어 설명마다 영어 번역 레코드를 만든다 (모듈 docstring의 규칙).

    Args:
        conn: 열린 트랜잭션의 DB 연결 (호출자가 커밋한다).
        session_id: 세션 ID.
        translator: 번역기 (`version`이 번역 버전에 들어간다).
        ontology: 세션 온톨로지 (용어집).
        policy: 번역 정책.
        now: 생성 시각 (시간대 필수).

    Returns:
        `TranslateSummary`. `pending`은 쓰고 난 뒤 다시 계산한 실제 대기다: 번역하지 못한 원문과
        번역하는 동안 새로 들어온 원문 (F-PAR-17, 예외를 내지 않는다).

    부작용: `label_records`에 번역 레코드와 삭제 레코드를 추가한다. 번역기(사내 서버)를 부른다.
    """
    version = translation_version(policy, translator.version, ontology)
    summary = TranslateSummary(version)
    plan = plan_translation(get_labels(conn, session_id, kinds=KINDS), policy, version)
    done = _translate_all(plan, translator, glossary_entries(ontology), policy, summary)
    # 쓰기 직전에 잠그고 다시 계획한다: 번역하는 동안 원문이 바뀌었거나 검수됐으면 낡은 결과를
    # 버린다
    lock_session(conn, session_id)
    fresh = plan_translation(get_labels(conn, session_id, kinds=KINDS), policy, version)
    new: list[LabelRecord] = []
    replaced: list[LabelRecord] = []
    for source, olds in fresh.todo:
        text = done.get(source.label_id)
        if text is not None:
            new.append(_record(source, text, policy, version, now))
            replaced += olds
    removed = retractions([*fresh.stale, *replaced], version, now)
    insert_labels(conn, [*removed, *new])
    # 행동이 이미 검수됐으면 새 번역도 그 검수를 물려받는다. 행동을 지웠거나 의미를 고친 검수 전
    # 설명은 지운다 (F-PAR-15, ADR 0043 3절. 같은 잠금·트랜잭션)
    follow = apply_description_follow(conn, session_id, now, InheritanceMechanism.TRANSLATE)
    # 쓰고 물려받은 뒤의 실제 대기로 요약을 정한다 (F-PAR-17): 번역하는 동안 새로 들어온 원문은
    # 이번 실행이 묻지 않았으므로 대기다. 실패했지만 그 사이 지워진 원문은 대기가 아니다
    left = plan_translation(get_labels(conn, session_id, kinds=KINDS), policy, version)
    summary.pending = [s.label_id for s, _ in left.todo]
    summary.invalid = [i for i in summary.invalid if i in summary.pending]
    summary.inherited = len(follow.approve)
    summary.translated = len(new)
    summary.retracted = len(removed) + len(follow.retract)
    summary.covered = fresh.covered
    summary.kept_reviewed = [x.label_id for x in fresh.kept_reviewed]
    summary.deleted = fresh.deleted
    return summary
