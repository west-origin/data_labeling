"""판정 응답 재검사 (WP17.2, ADR 0040 3.2).

서버가 구조화 출력을 강제해도 여기서 늘 다시 검사한다.

1. JSON 파싱과 구조(`ok`, `items`, `missing`) 검사. 실패하면 파싱 실패(→ 재시도).
2. 번호 검사: ok와 items.i를 합치면 대상 번호와 같아야 하고 둘이 겹치면 안 된다. 목록 밖 번호·
   중복은 위반이다. 빠진 번호는 unsure + response_invalid, answered=False로 둔다 (시스템 오류를 모델
   판단처럼 기록하지 않는다).
3. 항목 정합: 판정 값, 문제 유형(그 종류의 허용 목록, unsure는 판단 불가 사유만), wrong·suspect는
   문제 1개 이상과 한글 이유, 이유 2자 이상, 확신도 0~1, 근거 프레임 번호가 보낸 프레임 안.
   위반 항목은 unsure + response_invalid로 낮춘다.
4. 제안 검사: 그 종류가 받지 않는 키, 개체표 밖 번호, 온톨로지 밖 값, 삭제 불가 종류의 삭제, 정책이
   끈 종류 변경, 바꾸는 것이 없는 제안은 위반이다 (항목 위반).
5. `reason_guard`: URL·@·숫자 4자리 이상·글자 수 초과·프레임 참조(F3)·금지어가 있으면 이유를
   문제 유형 기반 고정 문장으로 바꾸고 판정은 유지한다 (위반으로 세지 않는다).
6. 누락 후보(`missing`)는 정책이 끄면(`max_findings` 0) 버린다.

재시도 판단(`ValidatedAnswer.retry_worthy`): 파싱 실패, 또는 (번호 위반 + 항목 위반) / 대상 수가
`retry_invalid_ratio`를 넘을 때. 위반 목록은 재시도 프롬프트에 그대로 붙는다.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from dlp_aireview.policy import AiReviewPolicy
from dlp_aireview.prompt import PROBLEM_KO
from dlp_aireview.schema import (
    PROBLEMS_BY_KIND,
    SPATIAL_FIELDS,
    SUGGEST_FIELDS_BY_KIND,
    TEMPORAL_FIELDS,
    temporal_values,
)
from dlp_aireview.suggest import NO_CHANGE, materialize
from dlp_schema.aireview import UNSURE_PROBLEMS, AiGroup, AiProblem, AiSuggestion, AiVerdictValue
from dlp_schema.labels import KeypointTrackPayload, LabelRecord
from dlp_schema.ontology import Ontology

# 이유 안전 검사 (개인정보·출처 노출 방지)
_URL = re.compile(r"https?://|s3://|@", re.I)
_DIGITS = re.compile(r"\d{4,}")
_FRAME_REF = re.compile(r"(?<![A-Za-z])[FC]\d+")
_HANGUL = re.compile(r"[가-힣]")
OK_REASON = "AI: 문제 없음"
INVALID_REASON = "AI: 응답 형식 오류"


class _Loose(BaseModel):
    """응답 파싱용 기반 (알 수 없는 키는 오류, 값 검사는 항목별로 따로 한다)."""

    model_config = ConfigDict(extra="forbid")


class RawItem(_Loose):
    """응답 항목 (값 범위는 `validate_answer`가 항목별로 본다)."""

    i: int
    v: str
    p: list[str] = []
    r: str = ""
    c: float | None = None
    f: int | None = None
    s: dict[str, Any] | None = None


class RawAnswer(_Loose):
    """응답 전체."""

    ok: list[int]
    items: list[RawItem]
    missing: list[dict[str, Any]] = []


@dataclass(frozen=True)
class ItemDraft:
    """대상 하나의 검증된 판정 초안.

    Attributes:
        position: 요청 안 대상 번호.
        verdict, problems, reason, confidence: 판정.
        frame: 근거 프레임의 요청 안 번호 (F#) 또는 None.
        suggestion: 완성 페이로드로 만든 제안 (`suggest.materialize`) 또는 None.
        suggestion_error: 온톨로지·계약 검사에 걸려 버린 제안의 사유 (판정은 남긴다).
        answered: 모델이 형식에 맞게 답했는가 (False면 unsure + response_invalid).
    """

    position: int
    verdict: AiVerdictValue
    problems: tuple[AiProblem, ...]
    reason: str
    confidence: float | None = None
    frame: int | None = None
    suggestion: AiSuggestion | None = None
    suggestion_error: str | None = None
    answered: bool = True


@dataclass(frozen=True)
class RawFinding:
    """누락 후보 초안 (요청 안 프레임 번호, 정규화 상자)."""

    value: str
    f0: int
    f1: int
    box: tuple[int, int, int, int] | None
    reason: str


@dataclass(frozen=True)
class ValidatedAnswer:
    """응답 재검사 결과.

    Attributes:
        parsed: JSON·구조 파싱에 성공했는가.
        items: 대상마다 하나 (요청 안 번호 순). 파싱 실패면 빈 튜플.
        violations: 재시도 프롬프트에 붙일 위반 문장.
        invalid: 번호 위반 + 항목 위반 수.
        findings: 받아들인 누락 후보.
    """

    parsed: bool
    items: tuple[ItemDraft, ...]
    violations: tuple[str, ...]
    invalid: int
    findings: tuple[RawFinding, ...] = ()

    def retry_worthy(self, n_targets: int, ratio: float) -> bool:
        """다시 물을 만한가: 파싱 실패, 또는 위반 비율이 `ratio`를 넘음."""
        return not self.parsed or self.invalid / max(1, n_targets) > ratio


@dataclass(frozen=True)
class RequestView:
    """재검사에 필요한 요청 정보.

    Attributes:
        group: 묶음.
        targets: 이 요청의 대상 (번호 순).
        n_frames: 보낸 프레임 수 (확대 조각 제외).
        entities: 창 개체표 (개체 ID, 클래스). 제안의 target·tool 번호가 가리킨다.
    """

    group: AiGroup
    targets: tuple[LabelRecord, ...]
    n_frames: int
    entities: tuple[tuple[str, str | None], ...]


def missing_item(position: int) -> ItemDraft:
    """응답에서 빠졌거나 형식이 맞지 않는 대상 (시스템이 채운 unsure)."""
    return ItemDraft(
        position,
        AiVerdictValue.UNSURE,
        (AiProblem.RESPONSE_INVALID,),
        INVALID_REASON,
        answered=False,
    )


def reason_guard(
    text: str, problems: Sequence[AiProblem], policy: AiReviewPolicy
) -> tuple[str, bool]:
    """이유 안전 검사. 걸리면 (문제 유형 기반 고정 문장, True), 아니면 (원문 정리, False).

    검사: URL(`http://`, `https://`, `s3://`)과 `@`, 숫자 4자리 이상 연속, 글자 수 상한 초과,
    프레임·조각 참조(F3, C1), 정책 금지어.
    """
    clean = " ".join(text.split())
    r = policy.response
    bad = (
        _URL.search(clean) is not None
        or _DIGITS.search(clean) is not None
        or len(clean) > r.reason_max_chars
        or _FRAME_REF.search(clean) is not None
        or any(term in clean for term in r.reason_banned_terms)
    )
    if not bad:
        return clean, False
    first = next((p for p in problems if p is not AiProblem.RESPONSE_INVALID), None)
    return (f"AI: {PROBLEM_KO[first]}" if first is not None else "AI: 판단 불가"), True


def parse_answer(raw: str) -> RawAnswer | str:
    """원문 → `RawAnswer`, 실패하면 위반 문장."""
    try:
        data: Any = json.loads(raw)
    except json.JSONDecodeError as exc:
        return f"JSON이 아닙니다: {exc.msg}"
    try:
        return RawAnswer.model_validate(data)
    except ValidationError as exc:
        first = exc.errors()[0]
        return f"응답 구조 위반: {first['loc']} {first['msg']}"


def _allowed_values(o: Ontology, group: AiGroup) -> dict[str, set[Any]]:
    """제안 키 → 허용 값 (target·tool·delete 제외)."""
    if group is AiGroup.SPATIAL:
        return {"class_id": set(o.objects), "hand": {"left", "right"}}
    return {k: set(v) for k, v in temporal_values(o).items()}


def _check_patch(
    s: dict[str, Any], label: LabelRecord, view: RequestView, o: Ontology, policy: AiReviewPolicy
) -> tuple[dict[str, Any] | None, list[str]]:
    """제안 객체 검사 → (값이 있는 키만 담은 패치 또는 None, 위반)."""
    keys = SPATIAL_FIELDS if view.group is AiGroup.SPATIAL else TEMPORAL_FIELDS
    unknown = set(s) - set(keys)
    if unknown:
        return None, [f"제안에 모르는 키 {sorted(unknown)}"]
    patch = {k: v for k, v in s.items() if v is not None and not (k == "delete" and v is False)}
    if not patch:
        return None, []
    errors: list[str] = []
    allowed_keys = SUGGEST_FIELDS_BY_KIND[label.kind]
    if set(patch) - allowed_keys:
        errors.append(f"{label.kind}에 쓸 수 없는 제안 키 {sorted(set(patch) - allowed_keys)}")
    for k in ("target", "tool"):
        v = patch.get(k)
        if v is not None and (not isinstance(v, int) or not 0 <= v < len(view.entities)):
            errors.append(f"개체표 밖의 {k} 번호 {v}")
    values = _allowed_values(o, view.group)
    errors += [
        f"목록 밖 {k} 값 {v}" for k, v in patch.items() if k in values and v not in values[k]
    ]
    if patch.get("delete") is True and label.kind not in policy.response.suggest.delete_kinds:
        errors.append(f"{label.kind}은 삭제 제안을 받지 않습니다")
    if "kind" in patch and not policy.response.suggest.kind_change:
        errors.append("종류 변경 제안은 받지 않습니다")
    if "hand" in patch and not (
        isinstance(label.payload, KeypointTrackPayload) and label.payload.skeleton == "hand21"
    ):
        errors.append("손 키포인트가 아닌 트랙에 hand 제안")
    return (None if errors else patch), errors


def _allowed_problems(verdict: AiVerdictValue, kind: str) -> frozenset[AiProblem]:
    """판정 값·라벨 종류에 쓸 수 있는 문제 유형 (unsure는 판단 불가 사유만, 시스템 사유 제외)."""
    if verdict is AiVerdictValue.UNSURE:
        return UNSURE_PROBLEMS - {AiProblem.RESPONSE_INVALID}
    return frozenset(PROBLEMS_BY_KIND[kind])


def _check_values(
    item: RawItem,
    label: LabelRecord,
    verdict: AiVerdictValue,
    view: RequestView,
    policy: AiReviewPolicy,
) -> tuple[list[AiProblem], list[str]]:
    """문제 유형·이유·확신도·근거 프레임 검사 → (문제 유형, 위반)."""
    errors: list[str] = []
    problems: list[AiProblem] = []
    allowed = {a.value for a in _allowed_problems(verdict, label.kind)}
    for p in item.p:
        if p not in allowed or AiProblem(p) in problems:
            errors.append(f"#{item.i}: {label.kind}에 쓸 수 없는 문제 유형 {p}")
        else:
            problems.append(AiProblem(p))
    if verdict is not AiVerdictValue.UNSURE and not problems:
        errors.append(f"#{item.i}: {verdict.value} 판정에는 문제 유형이 필요하다")
    if len(problems) > policy.response.max_problems:
        errors.append(f"#{item.i}: 문제 유형은 {policy.response.max_problems}개까지")
    reason = item.r.strip()
    if len(reason) < 2 or (verdict is not AiVerdictValue.UNSURE and not _HANGUL.search(reason)):
        errors.append(f"#{item.i}: 이유는 한국어 한 문장")
    if item.c is not None and not 0 <= item.c <= 1:
        errors.append(f"#{item.i}: 확신도는 0~1")
    if item.f is not None and not 0 <= item.f < view.n_frames:
        errors.append(f"#{item.i}: 근거 프레임 번호 {item.f}이 보낸 프레임 밖")
    return problems, errors


def _check_suggestion(
    item: RawItem, label: LabelRecord, view: RequestView, o: Ontology, policy: AiReviewPolicy
) -> tuple[AiSuggestion | None, str | None, list[str]]:
    """제안 검사·완성 → (제안, 버린 제안 사유, 위반). unsure의 제안은 무시한다.

    바꾸는 것이 없는 제안은 위반이다. 온톨로지·계약 검사에 걸린 제안은 위반이 아니라 제안만 버린다.
    """
    if item.s is None or item.v == AiVerdictValue.UNSURE.value:
        return None, None, []
    patch, errors = _check_patch(item.s, label, view, o, policy)
    if errors or patch is None:
        return None, None, [f"#{item.i}: {e}" for e in errors]
    made = materialize(label, patch, view.entities, o)
    if made == NO_CHANGE:
        return None, None, [f"#{item.i}: {NO_CHANGE}"]
    if isinstance(made, str):
        return None, made[:300], []
    return made, None, []


def _check_item(
    item: RawItem, view: RequestView, o: Ontology, policy: AiReviewPolicy
) -> tuple[ItemDraft | None, list[str]]:
    """항목 하나 검사 → (초안 또는 None, 위반 문장)."""
    label = view.targets[item.i]
    try:
        verdict = AiVerdictValue(item.v)
    except ValueError:
        return None, [f"#{item.i}: 판정 값 {item.v}이 목록 밖"]
    if verdict is AiVerdictValue.OK:
        return None, [f"#{item.i}: ok는 items가 아니라 ok 목록에 넣는다"]
    problems, errors = _check_values(item, label, verdict, view, policy)
    suggestion, dropped, s_errors = _check_suggestion(item, label, view, o, policy)
    errors += s_errors
    if errors:
        return None, errors
    text, _ = reason_guard(item.r.strip() or PROBLEM_KO[AiProblem.OTHER], problems, policy)
    draft = ItemDraft(item.i, verdict, tuple(problems), text, item.c, item.f, suggestion, dropped)
    return draft, []


def _box(value: Any) -> tuple[int, int, int, int] | None:
    """누락 후보 상자 [x, y, w, h] (정수 4개가 아니면 None)."""
    if not isinstance(value, list):
        return None
    items: list[Any] = value  # pyright: ignore[reportUnknownVariableType]
    if len(items) != 4 or not all(isinstance(v, int) for v in items):
        return None
    return (int(items[0]), int(items[1]), int(items[2]), int(items[3]))


def _findings(
    answer: RawAnswer, view: RequestView, policy: AiReviewPolicy
) -> tuple[RawFinding, ...]:
    """누락 후보 (정책이 끄면 버린다, 형식이 맞지 않는 후보도 버린다)."""
    out: list[RawFinding] = []
    for m in answer.missing[: policy.response.max_findings]:
        value = m.get("class_id") or m.get("event_type")
        f0 = m.get("f0", m.get("f"))
        f1 = m.get("f1", f0)
        box = m.get("box")
        reason = str(m.get("r", "")).strip()
        if not isinstance(value, str) or not isinstance(f0, int) or not isinstance(f1, int):
            continue
        if not (0 <= f0 <= f1 < view.n_frames) or len(reason) < 2:
            continue
        out.append(RawFinding(value, f0, f1, _box(box), reason))
    return tuple(out)


def validate_answer(
    raw: str, view: RequestView, o: Ontology, policy: AiReviewPolicy
) -> ValidatedAnswer:
    """원문 응답을 재검사한다 (모듈 docstring 1~6단계).

    Returns:
        `ValidatedAnswer`. 파싱에 성공하면 대상마다 초안이 정확히 하나 있다 (빠지거나 위반인 대상은
        unsure + response_invalid).
    """
    answer = parse_answer(raw)
    if isinstance(answer, str):
        return ValidatedAnswer(False, (), (answer,), len(view.targets))
    n = len(view.targets)
    violations: list[str] = []
    seen: dict[int, ItemDraft] = {}
    claimed: list[int] = [*answer.ok, *(it.i for it in answer.items)]
    range_errors = [f"대상 목록 밖 번호 #{i}" for i in claimed if not 0 <= i < n]
    violations += range_errors
    dup = sorted({i for i in claimed if claimed.count(i) > 1 and 0 <= i < n})
    violations += [f"번호 #{i}가 두 번 나옴" for i in dup]
    for i in answer.ok:
        if 0 <= i < n and i not in dup:
            seen[i] = ItemDraft(i, AiVerdictValue.OK, (), OK_REASON)
    rejected: set[int] = set(dup)
    for item in answer.items:
        if not 0 <= item.i < n or item.i in dup:
            continue
        draft, errors = _check_item(item, view, o, policy)
        violations += errors
        if draft is None:
            rejected.add(item.i)
        else:
            seen[item.i] = draft
    invalid = len(range_errors) + len(dup) + len(rejected - set(dup))
    missing = [i for i in range(n) if i not in seen and i not in rejected]
    violations += [f"번호 #{i}가 빠짐" for i in missing]
    items = tuple(seen.get(i) or missing_item(i) for i in range(n))
    return ValidatedAnswer(
        True, items, tuple(violations), invalid + len(missing), _findings(answer, view, policy)
    )
