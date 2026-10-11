"""판정 응답 재검사 (WP17.2, ADR 0040 3.2).

서버가 구조화 출력을 강제해도 여기서 늘 다시 검사한다 (`schema_enforcement: prompt`나 enum을
강제하지 않는 서버도 있다). 응답의 어떤 값도 예외를 내지 않는다: 형식이 맞지 않는 값은 위반이나
버림이 된다.

1. JSON 파싱과 구조(`ok`, `items`, `missing`) 검사. 실패하면 파싱 실패(→ 재시도).
2. 번호 검사: ok와 items.i를 합치면 대상 번호와 같아야 하고 둘이 겹치면 안 된다. 목록 밖 번호·
   중복은 위반이다. 빠진 번호는 unsure + response_invalid, answered=False로 둔다 (시스템 오류를 모델
   판단처럼 기록하지 않는다).
3. 항목 정합: 판정 값, 문제 유형(그 종류의 허용 목록, unsure는 판단 불가 사유만), wrong·suspect는
   문제 1개 이상과 한글 이유, 이유 2자 이상, 확신도 0~1, 근거 프레임 번호가 이 요청에 보낸 프레임
   (단위 프레임 번호 F#) 중 하나. 위반 항목은 unsure + response_invalid로 낮춘다.
4. 제안 검사: 그 종류가 받지 않는 키, 개체표 밖 번호, 온톨로지 밖 값(문자열·정수가 아닌 값 포함),
   삭제 불가 종류의 삭제, 정책이 끈 종류 변경, 바꾸는 것이 없는 제안은 위반이다 (항목 위반).
5. `reason_guard`: URL·@·숫자 4자리 이상·글자 수 초과·프레임 참조(F3)·금지어가 있으면 이유를
   문제 유형 기반 고정 문장으로 바꾸고 판정은 유지한다 (위반으로 세지 않는다).
6. 누락 후보(`missing`, `max_findings` 0이면 끔): 값이 온톨로지 안(시간: 원본 접근이 아닌 이벤트,
   공간: 객체 클래스)이고, 프레임 번호가 보낸 프레임 안이고, 공간 상자가 0~1000 정수 4개(넓이 > 0,
   화면 안)인 후보만 받는다. 이유는 `reason_guard`를 거친다.
7. 누락 후보가 같은 종류 라벨과 겹치면 버린다 (시간: 같은 이벤트 유형 라벨과 구간이 겹침, 공간: 그
   프레임에 같은 클래스 박스가 `finding_overlap_iou` 이상 겹침). 창 밖 후보는 6의 프레임 검사가
   막는다.

위반 문장(`Violation`)은 둘이다: 재시도 프롬프트에 붙이는 `text`(모델이 쓴 값을 담을 수 있다. 사내
서버에만 보낸다)와 단위 기록(`AiReviewUnit.errors`, 추가 전용 표)에 남기는 `record`(번호·위반
종류만, 모델이 쓴 값 없음). 응답 원문을 저장하지 않는 이유(블러본에 글자 블러가 없어 명찰·번호가
옮겨 적힐 수 있음)와 같다.

재시도 판단(`ValidatedAnswer.retry_worthy`): 파싱 실패, 또는 (번호 위반 + 항목 위반) / 대상 수가
`retry_invalid_ratio`를 넘을 때.
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
from dlp_aireview.tracks import Box, box_at
from dlp_schema.aireview import UNSURE_PROBLEMS, AiGroup, AiProblem, AiSuggestion, AiVerdictValue
from dlp_schema.labels import BoxTrackPayload, EventPayload, KeypointTrackPayload, LabelRecord
from dlp_schema.ontology import Ontology

# 이유 안전 검사 (개인정보·출처 노출 방지)
_URL = re.compile(r"https?://|s3://|@", re.I)
_DIGITS = re.compile(r"\d{4,}")
_FRAME_REF = re.compile(r"(?<![A-Za-z])[FC]\d+")
_HANGUL = re.compile(r"[가-힣]")
OK_REASON = "AI: 문제 없음"
INVALID_REASON = "AI: 응답 형식 오류"
# 누락 후보 정규화 상자 범위
_NORM = 1000


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
class Violation:
    """응답 위반 하나.

    Attributes:
        text: 재시도 프롬프트에 붙이는 문장 (모델이 쓴 값을 담을 수 있다. 사내 서버에만 보낸다).
        record: 단위 기록용 문장 (대상 번호·위반 종류만, 모델이 쓴 값·목록 밖 번호 없음).
    """

    text: str
    record: str


def _v(text: str, record: str | None = None) -> Violation:
    """위반 하나 (record를 주지 않으면 text에 모델이 쓴 값이 없는 문장)."""
    return Violation(text, text if record is None else record)


@dataclass(frozen=True)
class ItemDraft:
    """대상 하나의 검증된 판정 초안.

    Attributes:
        position: 요청 안 대상 번호.
        verdict, problems, reason, confidence: 판정.
        frame: 근거 프레임의 단위 프레임 번호 (F#) 또는 None.
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
    """누락 후보 초안 (단위 프레임 번호, 정규화 상자, 안전 검사를 거친 이유)."""

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
        violations: 위반 (재시도 프롬프트 문장과 기록 문장).
        invalid: 번호 위반 + 항목 위반 수.
        findings: 받아들인 누락 후보.
    """

    parsed: bool
    items: tuple[ItemDraft, ...]
    violations: tuple[Violation, ...]
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
        frame_nos: 이 요청에 보낸 프레임의 단위 프레임 번호 (F#, 확대 조각 제외). 응답의 f·누락 후보
            프레임은 이 안이어야 한다.
        entities: 창 개체표 (개체 ID, 클래스). 제안의 target·tool 번호가 가리킨다.
        frame_ms: 단위 프레임 시각 (단위 프레임 번호 순, 누락 후보 시각).
        known: 누락 후보 겹침 검사에 쓰는 단위의 대상·맥락 라벨.
        src_size: 원 프레임 크기 (너비, 높이) px (공간 누락 후보 상자를 라벨 좌표와 비교).
    """

    group: AiGroup
    targets: tuple[LabelRecord, ...]
    frame_nos: tuple[int, ...]
    entities: tuple[tuple[str, str | None], ...]
    frame_ms: tuple[int, ...] = ()
    known: tuple[LabelRecord, ...] = ()
    src_size: tuple[int, int] | None = None


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


def parse_answer(raw: str) -> RawAnswer | Violation:
    """원문 → `RawAnswer`, 실패하면 위반 (기록 문장에는 응답의 키 이름·값을 넣지 않는다)."""
    try:
        data: Any = json.loads(raw)
    except json.JSONDecodeError as exc:
        return _v(f"JSON이 아닙니다: {exc.msg}", "JSON이 아님")
    try:
        return RawAnswer.model_validate(data)
    except ValidationError as exc:
        first = exc.errors()[0]
        return _v(f"응답 구조 위반: {first['loc']} {first['msg']}", "응답 구조 위반")


def _is_int(v: Any) -> bool:
    """bool이 아닌 정수인가 (JSON true가 1로 섞이지 않게)."""
    return isinstance(v, int) and not isinstance(v, bool)


def _scalar(v: Any) -> bool:
    """온톨로지 값 후보(문자열 또는 bool이 아닌 정수)인가. 목록·객체·실수는 아니다."""
    return isinstance(v, str) or _is_int(v)


def _allowed_values(o: Ontology, group: AiGroup) -> dict[str, set[Any]]:
    """제안 키 → 허용 값 (target·tool·delete 제외)."""
    if group is AiGroup.SPATIAL:
        return {"class_id": set(o.objects), "hand": {"left", "right"}}
    return {k: set(v) for k, v in temporal_values(o).items()}


def _patch_value_errors(patch: dict[str, Any], view: RequestView, o: Ontology) -> list[Violation]:
    """제안 값 검사: 개체 번호 범위, 삭제 표시 형식, 온톨로지 목록 (해시할 수 없는 값도 위반)."""
    errors: list[Violation] = []
    for k in ("target", "tool"):
        v = patch.get(k)
        if v is not None and (not _is_int(v) or not 0 <= v < len(view.entities)):
            errors.append(_v(f"개체표 밖의 {k} 번호 {v}", f"개체표 밖의 {k} 번호"))
    if "delete" in patch and not isinstance(patch["delete"], bool):
        errors.append(_v(f"delete는 true 또는 false다: {patch['delete']}", "delete 값 형식"))
    values = _allowed_values(o, view.group)
    errors += [
        _v(f"목록 밖 {k} 값 {v}", f"목록 밖 {k} 값")
        for k, v in patch.items()
        if k in values and not (_scalar(v) and v in values[k])
    ]
    return errors


def _check_patch(
    s: dict[str, Any], label: LabelRecord, view: RequestView, o: Ontology, policy: AiReviewPolicy
) -> tuple[dict[str, Any] | None, list[Violation]]:
    """제안 객체 검사 → (값이 있는 키만 담은 패치 또는 None, 위반)."""
    keys = SPATIAL_FIELDS if view.group is AiGroup.SPATIAL else TEMPORAL_FIELDS
    unknown = set(s) - set(keys)
    if unknown:
        return None, [_v(f"제안에 모르는 키 {sorted(unknown)}", "제안에 모르는 키")]
    patch = {k: v for k, v in s.items() if v is not None and not (k == "delete" and v is False)}
    if not patch:
        return None, []
    errors: list[Violation] = []
    extra = sorted(set(patch) - SUGGEST_FIELDS_BY_KIND[label.kind])
    if extra:
        errors.append(_v(f"{label.kind}에 쓸 수 없는 제안 키 {extra}"))
    errors += _patch_value_errors(patch, view, o)
    if patch.get("delete") is True and label.kind not in policy.response.suggest.delete_kinds:
        errors.append(_v(f"{label.kind}은 삭제 제안을 받지 않습니다"))
    if "kind" in patch and not policy.response.suggest.kind_change:
        errors.append(_v("종류 변경 제안은 받지 않습니다"))
    if "hand" in patch and not (
        isinstance(label.payload, KeypointTrackPayload) and label.payload.skeleton == "hand21"
    ):
        errors.append(_v("손 키포인트가 아닌 트랙에 hand 제안"))
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
) -> tuple[list[AiProblem], list[Violation]]:
    """문제 유형·이유·확신도·근거 프레임 검사 → (문제 유형, 위반)."""
    errors: list[Violation] = []
    problems: list[AiProblem] = []
    allowed = {a.value for a in _allowed_problems(verdict, label.kind)}
    n = item.i
    for p in item.p:
        if p not in allowed or AiProblem(p) in problems:
            errors.append(
                _v(f"#{n}: {label.kind}에 쓸 수 없는 문제 유형 {p}", f"#{n}: 쓸 수 없는 문제 유형")
            )
        else:
            problems.append(AiProblem(p))
    if verdict is not AiVerdictValue.UNSURE and not problems:
        errors.append(_v(f"#{n}: {verdict.value} 판정에는 문제 유형이 필요하다"))
    if len(problems) > policy.response.max_problems:
        errors.append(_v(f"#{n}: 문제 유형은 {policy.response.max_problems}개까지"))
    reason = item.r.strip()
    if len(reason) < 2 or (verdict is not AiVerdictValue.UNSURE and not _HANGUL.search(reason)):
        errors.append(_v(f"#{n}: 이유는 한국어 한 문장"))
    if item.c is not None and not 0 <= item.c <= 1:
        errors.append(_v(f"#{n}: 확신도는 0~1"))
    if item.f is not None and item.f not in view.frame_nos:
        sent = ", ".join(f"F{k}" for k in view.frame_nos)
        errors.append(
            _v(
                f"#{n}: 근거 프레임 번호 {item.f}은 보낸 프레임이 아니다 (보낸 프레임: {sent})",
                f"#{n}: 근거 프레임 번호가 보낸 프레임 밖",
            )
        )
    return problems, errors


def _check_suggestion(
    item: RawItem, label: LabelRecord, view: RequestView, o: Ontology, policy: AiReviewPolicy
) -> tuple[AiSuggestion | None, str | None, list[Violation]]:
    """제안 검사·완성 → (제안, 버린 제안 사유, 위반). unsure의 제안은 무시한다.

    바꾸는 것이 없는 제안은 위반이다. 온톨로지·계약 검사에 걸린 제안은 위반이 아니라 제안만 버린다
    (사유 문장에는 온톨로지 목록을 통과한 값만 들어간다).
    """
    if item.s is None or item.v == AiVerdictValue.UNSURE.value:
        return None, None, []
    patch, errors = _check_patch(item.s, label, view, o, policy)
    if errors or patch is None:
        n = item.i
        return None, None, [Violation(f"#{n}: {e.text}", f"#{n}: {e.record}") for e in errors]
    made = materialize(label, patch, view.entities, o)
    if made == NO_CHANGE:
        return None, None, [_v(f"#{item.i}: {NO_CHANGE}")]
    if isinstance(made, str):
        return None, made[:300], []
    return made, None, []


def _check_item(
    item: RawItem, view: RequestView, o: Ontology, policy: AiReviewPolicy
) -> tuple[ItemDraft | None, list[Violation]]:
    """항목 하나 검사 → (초안 또는 None, 위반)."""
    label = view.targets[item.i]
    try:
        verdict = AiVerdictValue(item.v)
    except ValueError:
        return None, [_v(f"#{item.i}: 판정 값 {item.v}이 목록 밖", f"#{item.i}: 판정 값이 목록 밖")]
    if verdict is AiVerdictValue.OK:
        return None, [_v(f"#{item.i}: ok는 items가 아니라 ok 목록에 넣는다")]
    problems, errors = _check_values(item, label, verdict, view, policy)
    suggestion, dropped, s_errors = _check_suggestion(item, label, view, o, policy)
    errors += s_errors
    if errors:
        return None, errors
    text, _ = reason_guard(item.r.strip() or PROBLEM_KO[AiProblem.OTHER], problems, policy)
    draft = ItemDraft(item.i, verdict, tuple(problems), text, item.c, item.f, suggestion, dropped)
    return draft, []


# ---------------------------------------------------------------- 누락 후보


def _box(value: Any) -> tuple[int, int, int, int] | None:
    """누락 후보 상자 [x, y, w, h] (0~1000 정규화 정수 4개, 넓이 > 0, 화면 안이 아니면 None)."""
    if not isinstance(value, list):
        return None
    items: list[Any] = value  # pyright: ignore[reportUnknownVariableType]
    if len(items) != 4 or not all(_is_int(v) for v in items):
        return None
    x, y, w, h = (int(v) for v in items)
    if min(x, y) < 0 or w <= 0 or h <= 0 or x + w > _NORM or y + h > _NORM:
        return None
    return x, y, w, h


def _iou(a: Box, b: Box) -> float:
    """두 상자 (x, y, w, h)의 IoU."""
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def _event_labeled(value: str, t0: int, t1: int, view: RequestView) -> bool:
    """같은 이벤트 유형 라벨이 [t0, t1]과 겹치는가 (시점 라벨은 구간 안이면)."""
    return any(
        isinstance(x.payload, EventPayload)
        and x.payload.event_type == value
        and x.t_start_ms <= t1
        and x.t_end_ms >= t0
        for x in view.known
    )


def _object_labeled(
    value: str, t_ms: int, box: tuple[int, int, int, int], view: RequestView, iou: float
) -> bool:
    """그 프레임에 같은 클래스 박스가 누락 후보 상자와 `iou` 이상 겹치는가.

    원 프레임 크기를 모르면 겹침을 잴 수 없어 같은 클래스 박스가 보이기만 해도 겹친 것으로 본다
    (누락 후보는 버리는 쪽이 안전하다).
    """
    target: Box = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
    for x in view.known:
        p = x.payload
        if not isinstance(p, BoxTrackPayload) or p.class_id != value:
            continue
        drawn = box_at(p.keyframes, t_ms)
        if drawn is None:
            continue
        if view.src_size is None:
            return True
        sw, sh = view.src_size
        norm = (
            drawn[0] * _NORM / sw, drawn[1] * _NORM / sh,
            drawn[2] * _NORM / sw, drawn[3] * _NORM / sh,
        )  # fmt: skip
        if _iou(norm, target) >= iou:
            return True
    return False


def _finding_frames(m: dict[str, Any], view: RequestView) -> tuple[int, int] | None:
    """누락 후보의 (시작, 끝) 단위 프레임 번호 (공간은 f 하나). 보낸 프레임이 아니면 None."""
    spatial = view.group is AiGroup.SPATIAL
    f0 = m.get("f" if spatial else "f0")
    f1 = f0 if spatial else m.get("f1")
    if not (_is_int(f0) and _is_int(f1)) or f0 not in view.frame_nos or f1 not in view.frame_nos:
        return None
    if f0 > f1 or not 0 <= f1 < len(view.frame_ms):
        return None
    return f0, f1


def _finding_from(
    m: dict[str, Any], view: RequestView, o: Ontology, policy: AiReviewPolicy
) -> RawFinding | None:
    """누락 후보 하나 검사 (모듈 docstring 6·7단계). 받지 않으면 None."""
    spatial = view.group is AiGroup.SPATIAL
    value = m.get("class_id" if spatial else "event_type")
    allowed = set(o.objects) if spatial else set(temporal_values(o)["event_type"])
    frames = _finding_frames(m, view)
    if not isinstance(value, str) or value not in allowed or frames is None:
        return None
    f0, f1 = frames
    t0, t1 = view.frame_ms[f0], view.frame_ms[f1]
    problem = AiProblem.MISSING_OBJECT if spatial else AiProblem.MISSING_EVENT
    raw = m.get("r")
    text, _ = reason_guard(raw if isinstance(raw, str) else "", (problem,), policy)
    reason = text if len(text) >= 2 else f"AI: {PROBLEM_KO[problem]}"
    if not spatial:
        labeled = _event_labeled(value, t0, t1, view)
        return None if labeled else RawFinding(value, f0, f1, None, reason)
    box = _box(m.get("box"))
    if box is None or _object_labeled(value, t0, box, view, policy.response.finding_overlap_iou):
        return None
    return RawFinding(value, f0, f1, box, reason)


def _findings(
    answer: RawAnswer, view: RequestView, o: Ontology, policy: AiReviewPolicy
) -> tuple[RawFinding, ...]:
    """누락 후보 (정책 상한까지, 검사를 통과한 것만)."""
    limit = policy.response.max_findings
    found = [_finding_from(m, view, o, policy) for m in answer.missing[:limit]]
    return tuple(f for f in found if f is not None)


def validate_answer(
    raw: str, view: RequestView, o: Ontology, policy: AiReviewPolicy
) -> ValidatedAnswer:
    """원문 응답을 재검사한다 (모듈 docstring 1~7단계).

    Returns:
        `ValidatedAnswer`. 파싱에 성공하면 대상마다 초안이 정확히 하나 있다 (빠지거나 위반인 대상은
        unsure + response_invalid).
    """
    answer = parse_answer(raw)
    if isinstance(answer, Violation):
        return ValidatedAnswer(False, (), (answer,), len(view.targets))
    n = len(view.targets)
    violations: list[Violation] = []
    seen: dict[int, ItemDraft] = {}
    claimed: list[int] = [*answer.ok, *(it.i for it in answer.items)]
    out_of_range = [
        _v(f"대상 목록 밖 번호 #{i}", "대상 목록 밖 번호") for i in claimed if not 0 <= i < n
    ]
    violations += out_of_range
    dup = sorted({i for i in claimed if claimed.count(i) > 1 and 0 <= i < n})
    violations += [_v(f"번호 #{i}가 두 번 나옴") for i in dup]
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
    invalid = len(out_of_range) + len(dup) + len(rejected - set(dup))
    missing = [i for i in range(n) if i not in seen and i not in rejected]
    violations += [_v(f"번호 #{i}가 빠짐") for i in missing]
    items = tuple(seen.get(i) or missing_item(i) for i in range(n))
    return ValidatedAnswer(
        True, items, tuple(violations), invalid + len(missing), _findings(answer, view, o, policy)
    )
