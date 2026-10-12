# 외부 도구(Label Studio) 결과 JSON을 다루는 부분이 있어 알 수 없는 타입 경고를 이 모듈에서만 끈다.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false

"""AI 1차 검수 제안의 사람 적용과 수집 결과 기록 (WP17.7, ADR 0040 6.6, I6).

`collect.collect_task`가 도구 결과를 받은 뒤 이 모듈을 차례로 부른다. 모든 단계는 AI가 꺼져
있거나(`ReviewSetup.ai`가 None) 작업에 노출 기록이 없으면(AI 단계 전에 만든 작업, 블러 검수
작업) 아무것도 하지 않고 도구 결과를 그대로 돌려준다 (분기 없이 부른다).

1. `begin`: 작업의 노출 기록(`ai_review_exposures`)과 그 기록이 가리키는 판정 단위를 DB에서 읽는다.
2. 결정 읽기: Label Studio는 `ls_decisions`(라벨 ID별 결정, `ai_display.ls_label_decisions`.
   AI 구간에 붙은 결정은 원 라벨로 되돌린다. 검수자가 AI 라벨로 그린 구간은 무시하고 경고 수만
   남긴다). CVAT 1단계는 결정 칸이 없어 빈 사전이다 (F-AIR-06).
3. `apply_decisions`: 적용 결정을 검수 항목에 반영한다. 제안 내용은 도구가 돌려준 메모·메타에서 읽지
   않고 노출 기록 → 판정 단위 → 판정의 `suggestion.payload`를 DB에서 다시 읽어, 세션 온톨로지로
   다시 검사한다 (I6). 비교 기준은 작업에 보낸 라벨과 `reconcile.same_as_sent`다.
4. `settle`: 적용본의 새 레코드가 세션 단위 검증(`validate_session_labels`: 상위 구간 자식
   연결 등)을 통과하지 못하면 AI 적용만 되돌린다. 하나씩 더해 보며(라벨 ID 순) 통과하는 적용만
   남기고 나머지는 `invalid_now`로 기록한다. 검수자 자신의 결과도 통과하지 못하면 지금과 똑같이
   수집이 실패한다 (`insert_labels`). AI는 흐름을 막지 않는다.
5. `record_outcome`: 라벨 이력 기록 뒤 작업마다 한 번 `ai_review_outcomes`에 사람 결과·결정·적용
   결과·제안 일치를 쓴다 (`finish_assignment` 앞, 같은 트랜잭션).

적용 표 (검수자 행동 → 처리 → 결과 레코드 → `AiOutcomeItem`)
    - 아무것도 안 함: 그대로 → 승인 → human=approved, decision=none.
    - 직접 고침(결정 없음): 그대로 → 수정 레코드 → corrected, match=제안 필드 기준
      exact|partial|none.
    - 적용, 구간 그대로, op=set: 항목 페이로드를 제안 페이로드로 → parent=원 라벨, HUMAN,
      human_corrected, 검수자=담당자 → corrected, applied, match=exact.
    - 적용, op=change_kind(action↔gap): 원 항목을 빼고 원 라벨 없는 새 항목(새 action_id는
      `h-<손>-<시작>`) → 원 라벨 삭제 레코드 + parent 없는 새 사람 레코드 → replaced, applied.
    - 적용, op=delete(이벤트·공간 트랙): 항목을 뺀다 → 원 라벨 삭제 레코드 → deleted, applied.
    - 적용했지만 검수자가 구간이나 값도 고침(또는 지움): 검수자 결과가 우선 → conflict_edited.
      검수자가 고친 결과가 제안과 같으면 applied다: 삭제 제안인데 검수자가 지움, set 제안과 같은
      구간·같은 페이로드로 고침, change_kind 제안인데 Label Studio에서 같은 구간의 종류를 제안과
      같은 내용으로 바꿔 붙임(원 라벨 없는 새 항목이 된다).
    - 적용, op=change_kind인데 검수자가 같은 구간·같은 내용의 새 항목을 직접 그려 둠: 원 항목만 빼고
      새 항목은 더하지 않는다 (같은 레코드 둘 방지) → replaced, applied.
    - 적용, 제안 없음 → no_suggestion. 지금 온톨로지 검사·세션 검증 실패 → invalid_now (적용하지
      않음). 세션 검증에는 새 레코드 ID 중복도 든다 (`insert_labels`의 원시 DB 오류 대신 되돌림).
    - 기각: 그대로 → decision=dismiss.
    - 측정이 아닌 작업(표준·QA·오류 삽입)에서 보낸 뒤 다른 단계가 바꾼 라벨(자식 레코드가 생김,
      `collect.drop_retracted`와 같은 기준): 적용 결정이 있어도 항목을 바꾸지 않는다 → invalid_now.
      결과는 `drop_retracted`가 빼므로 human=stale. (종류 변경 제안을 그대로 적용하면 parent 없는
      새 레코드가 drop_retracted를 지나 다른 단계가 지운 라벨이 다른 종류로 되살아난다.) 측정
      작업(블라인드·이중)은 이 검사를 하지 않는다: 비교 기준이 빈 목록이라 drop_retracted가 아무것도
      빼지 않고 결과가 parent 없는 측정 레코드라 운영 라벨을 되살리지 않는다.
    - 집단(arm)이 shown이 아니거나 그 라벨의 판정이 화면에 따로 보이지 않았으면 결정을 무시한다 →
      not_shown.
    - 이중 작업(arm=shown): 적용은 같게 반영하고 결과는 측정 레코드가 된다 → human=na. 짝 표준
      작업이 먼저 수집되어 그 라벨을 고쳤어도(자식 레코드가 생김) 적용한다 (위의 stale 검사는 측정이
      아닌 작업만. 막으면 보인 집단의 이중 일치 κ가 AI가 지적해 고친 라벨에서 체계적으로 낮아진다).

AI가 제안하고 사람이 적용했다는 사실은 라벨 계약에 넣지 않고 이 결과 표에만 남긴다 (결정: 내보내기·
학습에서 구별하는 계약 변경은 하지 않는다). 적용 결과 레코드는 사람이 그린 것과 같은 형식이다 (출처
HUMAN, human_corrected, 검수자 = 작업 담당자. Label Studio는 담당자 계정의 주석만 받는다, F-REV-06).
경계 이동 제안은 적용하지 않는다 (표시만, F-AIR-07).

`review.yaml ai_review.mode = off`로 수집하면 AI 표를 읽지 않으므로 검수자가 고른 결정도 반영하지
않고 결과 기록도 남기지 않는다 (off의 뜻: AI 표를 읽지 않는다).

공개 이름: `AiCollect`, `begin`, `ls_decisions`, `apply_decisions`, `settle`, `record_outcome`,
`human_actions`, `summary_counts`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import sqlalchemy as sa
from pydantic import ValidationError

from dlp_review.ai_display import AiDisplayConfig, flag_region_id, ls_label_decisions
from dlp_review.labelstudio import AI_FLAGS
from dlp_review.reconcile import ReviewedItem, ReviewOutcome, same_as_sent
from dlp_schema.aireview import (
    AiApplyResult,
    AiArm,
    AiDecision,
    AiExposure,
    AiHumanAction,
    AiOutcome,
    AiOutcomeItem,
    AiSuggestion,
    AiVerdict,
)
from dlp_schema.db.aireview_repository import get_ai_exposure, get_ai_unit, insert_ai_outcome
from dlp_schema.db.ontology_repository import get_session_ontology
from dlp_schema.label_validation import validate_session_labels
from dlp_schema.labels import LabelPayload, LabelRecord, Provenance, Source
from dlp_schema.ontology import Ontology
from dlp_schema.review import ReviewMode, ReviewTask
from dlp_schema.validation import OntologyViolationError, check_label

# 삭제 제안을 적용할 수 없는 종류 (손별 타임라인 공백 0, ADR 0015). 판정 재검사가 이미 거르지만
# DB에서 다시 읽은 제안도 다시 막는다 (I6)
_NO_DELETE = frozenset({"action", "gap", "hand_state"})
# 결과 레코드가 측정 레코드가 되는 방식 (사람 결과를 운영 라벨과 비교하지 않는다)
_MEASURED = (ReviewMode.BLIND, ReviewMode.DOUBLE)
Match = Literal["exact", "partial", "none", "na"]
# 적용한 라벨의 항목 변경: ("set", 바꾼 항목) | ("change_kind", 새 항목) | ("delete", None)
Change = tuple[str, ReviewedItem | None]


@dataclass
class AiCollect:
    """작업 하나의 수집 중 AI 상태 (`begin`이 만든다).

    Attributes:
        task: 수집하는 작업.
        exposure: 작업 생성 때 고정한 노출 기록.
        sent: 작업에 보낸 라벨 (ID → 레코드, 보낸 순서).
        verdicts: 노출 기록의 판정 (보인 것 + 숨긴 것) 라벨 ID → (판정, 단위 ID). 단위를 DB에서 다시
            읽은 값이다.
        ontology: 세션 온톨로지 (제안 재검사·세션 검증).
        history: 세션 라벨 이력 (세션 잠금 아래에서 읽은 것, 세션 검증의 기준).
        decisions: 라벨 ID → 검수자 결정 (`ls_decisions`가 채운다).
        results: 적용 결정을 받은 라벨 ID → 처리 결과.
        changes: 실제로 항목을 바꾸는 적용 (라벨 ID → 변경). `settle`이 되돌릴 수 있다.
        warnings: 수집 출력·결과 기록에 남길 경고 문장.
        base: 도구 결과 항목 (`apply_decisions`가 채운다, 적용 전).
        superseded: 이력에서 자식 레코드가 있는 라벨 ID (보낸 뒤 다른 단계가 지웠거나 다른 작업이
            먼저 고침. `collect.drop_retracted`와 같은 기준, `begin`이 채운다). 측정 작업(블라인드·
            이중)은 빈 집합이다 (drop_retracted가 빼는 결과가 없다).
    """

    task: ReviewTask
    exposure: AiExposure
    sent: dict[str, LabelRecord]
    verdicts: dict[str, tuple[AiVerdict, str]]
    ontology: Ontology
    history: list[LabelRecord]
    decisions: dict[str, AiDecision] = field(default_factory=dict[str, AiDecision])
    results: dict[str, AiApplyResult] = field(default_factory=dict[str, AiApplyResult])
    changes: dict[str, Change] = field(default_factory=dict[str, Change])
    warnings: list[str] = field(default_factory=list[str])
    base: list[ReviewedItem] = field(default_factory=list[ReviewedItem])
    superseded: frozenset[str] = frozenset()

    def items(self, keep: Sequence[str] | None = None) -> list[ReviewedItem]:
        """도구 결과(`base`)에 적용(`changes` 중 keep, None이면 전부)을 반영한 검수 항목."""
        chosen = {k: v for k, v in self.changes.items() if keep is None or k in keep}
        out: list[ReviewedItem] = []
        for item in self.base:
            change = chosen.get(item.origin_label_id or "")
            if change is None:
                out.append(item)
            elif change[1] is not None:  # set: 바꾼 항목, change_kind: 새 항목 (원 항목은 빠진다)
                out.append(change[1])
        return out


def begin(
    conn: sa.Connection,
    task: ReviewTask,
    cfg: AiDisplayConfig | None,
    sent: Sequence[LabelRecord],
    history: list[LabelRecord],
) -> AiCollect | None:
    """작업의 AI 상태를 읽는다 (AI 끔이거나 노출 기록이 없으면 None). DB 읽기만.

    Args:
        conn: 수집 트랜잭션 (세션·작업 잠금 아래).
        task: 수집하는 작업.
        cfg: `ReviewSetup.ai` (None이면 AI 표를 읽지 않는다).
        sent: 작업에 보낸 라벨 (블라인드·이중도 측정 전에 보낸 레코드).
        history: 세션 라벨 이력 (`get_labels`).
    """
    if cfg is None:
        return None
    exposure = get_ai_exposure(conn, task.task_key)
    if exposure is None:
        return None
    units: dict[str, list[AiVerdict]] = {}
    verdicts: dict[str, tuple[AiVerdict, str]] = {}
    for lid, uid in (*exposure.shown.items(), *exposure.hidden.items()):
        if uid not in units:
            unit = get_ai_unit(conn, uid)
            units[uid] = list(unit.verdicts) if unit is not None else []
        found = next((v for v in units[uid] if v.label_id == lid), None)
        if found is not None:
            verdicts[lid] = (found, uid)
    # 측정 작업은 비교 기준이 빈 목록이라 drop_retracted가 아무것도 빼지 않고, 결과는 parent 없는
    # 측정 레코드라 운영 라벨을 되살리지 않는다: stale 검사를 측정이 아닌 작업에만 둔다
    superseded = (
        frozenset[str]()
        if task.mode in _MEASURED
        else frozenset(x.parent_label_id for x in history if x.parent_label_id)
    )
    return AiCollect(
        task=task, exposure=exposure, sent={x.label_id: x for x in sent}, verdicts=verdicts,
        ontology=get_session_ontology(conn, task.session_id), history=history,
        superseded=superseded,
    )  # fmt: skip


def ls_decisions(
    results: Sequence[Mapping[str, Any]], ai: AiCollect | None
) -> dict[str, AiDecision]:
    """Label Studio 검수자 결과의 AI 결정 → 라벨 ID별 (AI가 없으면 빈 사전).

    `ai_display.ls_label_decisions`의 경고(작업 라벨이 아닌 구간의 결정, 한 라벨의 서로 다른 결정)와
    검수자가 AI 라벨(`ai_flags`)로 새로 그린 구간 수를 `ai.warnings`에 남긴다. AI 구간은 읽지 않는다
    (예측으로 보낸 AI 구간 id 밖의 것은 검수자가 그린 것이다).
    """
    if ai is None:
        return {}
    rows = [dict(r) for r in results]
    decisions, warnings = ls_label_decisions(rows, ai.exposure, list(ai.sent))
    ai.warnings += warnings
    sent_flags = {flag_region_id(uid, lid) for lid, uid in ai.exposure.shown.items()}
    drawn = {
        str(r.get("id"))
        for r in rows
        if r.get("from_name") == AI_FLAGS and str(r.get("id")) not in sent_flags
    }
    if drawn:
        ai.warnings.append(
            f"검수자가 AI 표시 라벨(AI:…)로 그린 구간 {len(drawn)}개를 무시했다 "
            "(라벨로 받지 않는다)"
        )
    return decisions


def _candidate(original: LabelRecord, s: AiSuggestion, o: Ontology) -> list[str]:
    """제안을 원 라벨에 적용한 사람 레코드의 온톨로지 위반 (없으면 빈 목록, I6 재검사)."""
    if s.op == "delete":
        return [f"{original.kind}은(는) 삭제할 수 없습니다"] if original.kind in _NO_DELETE else []
    if s.payload is None:
        return ["제안 페이로드가 없습니다"]
    kinds = {original.kind, s.payload.kind}
    if (s.op == "set" and len(kinds) != 1) or (
        s.op == "change_kind" and kinds != {"action", "gap"}
    ):
        return [f"{original.kind}에 {s.op} 제안({s.payload.kind})을 적용할 수 없습니다"]
    try:
        candidate = original.model_validate(
            original.model_dump()
            | {"payload": s.payload.model_dump(), "provenance": Provenance(source=Source.HUMAN)}
        )
    except ValidationError as exc:
        return [f"계약 위반: {exc.errors()[0]['msg']}"]
    return check_label(candidate, o)


def _same_span(a: LabelRecord | ReviewedItem, b: LabelRecord | ReviewedItem) -> bool:
    """같은 구간인가 (스트림은 둘 다 있을 때만 비교한다).

    Label Studio에서 새로 그린 시간 구간은 메타가 없어 스트림이 None이다 (종류를 바꿔 붙인 구간은
    원 라벨의 스트림을 잇는다, `labelstudio._reviewed_item`). 원 라벨(행동은 기준 스트림)과 스트림만
    다른 같은 구간을 다른 곳으로 보지 않는다.
    """
    streams = (a.stream_id, b.stream_id)
    return (a.t_start_ms, a.t_end_ms) == (b.t_start_ms, b.t_end_ms) and (
        None in streams or streams[0] == streams[1]
    )


def _drawn(ai: AiCollect, place: LabelRecord | ReviewedItem, payload: LabelPayload) -> bool:
    """검수자가 원 라벨 없이 같은 구간·같은 페이로드의 항목을 이미 냈는가 (도구 결과 기준)."""
    return any(
        x.origin_label_id is None and _same_span(x, place) and x.payload == payload for x in ai.base
    )


def _gone(ai: AiCollect, original: LabelRecord, s: AiSuggestion) -> AiApplyResult:
    """검수자가 원 라벨 구간을 지웠거나 다른 종류로 바꿔 붙였을 때의 적용 결과 (항목은 그대로).

    검수자 결과가 제안과 같으면 applied: 삭제 제안을 지움, change_kind 제안과 같은 구간·같은
    내용으로 종류를 바꿔 붙임. 그 밖은 검수자 결과가 우선한 것이다 (conflict_edited).
    """
    if s.op == "delete":
        return AiApplyResult.APPLIED
    if s.op == "change_kind" and s.payload is not None and _drawn(ai, original, s.payload):
        return AiApplyResult.APPLIED
    return AiApplyResult.CONFLICT_EDITED


def _apply_one(
    ai: AiCollect,
    label_id: str,
    item: ReviewedItem | None,
    normalize: Callable[[LabelRecord], LabelRecord] | None,
) -> tuple[AiApplyResult, Change | None]:
    """적용 결정 하나의 처리 결과와 항목 변경 (모듈 docstring 적용 표)."""
    if ai.exposure.arm is not AiArm.SHOWN or label_id not in ai.exposure.shown:
        return AiApplyResult.NOT_SHOWN, None
    judged = ai.verdicts.get(label_id)
    original = ai.sent.get(label_id)
    if judged is None or original is None:
        ai.warnings.append(f"{label_id}: 보인 판정의 단위·원 라벨을 찾지 못해 적용하지 않았다")
        return AiApplyResult.INVALID_NOW, None
    if label_id in ai.superseded:
        # 측정이 아닌 작업에서 보낸 뒤 다른 단계가 바꾼 라벨: 결과는 drop_retracted가 빼고(stale),
        # 적용도 하지 않는다 (측정 작업은 superseded가 비어 있다)
        ai.warnings.append(f"{label_id}: 보낸 뒤 다른 단계가 바꾼 라벨이라 제안을 적용하지 않았다")
        return AiApplyResult.INVALID_NOW, None
    s = judged[0].suggestion
    if s is None:
        return AiApplyResult.NO_SUGGESTION, None
    if item is None:  # 검수자가 그 구간을 지웠다 (또는 다른 종류로 바꿔 붙였다)
        return _gone(ai, original, s), None
    if not same_as_sent(normalize(original) if normalize else original, item):
        same = (
            s.payload is not None
            and item.payload == s.payload
            and (item.t_start_ms, item.t_end_ms) == (original.t_start_ms, original.t_end_ms)
        )
        return (AiApplyResult.APPLIED if same else AiApplyResult.CONFLICT_EDITED), None
    problems = _candidate(original, s, ai.ontology)
    if problems:
        ai.warnings.append(
            f"{label_id}: 제안이 지금 검사를 통과하지 못해 적용하지 않았다 ({problems[0]})"
        )
        return AiApplyResult.INVALID_NOW, None
    if s.op == "delete" or s.payload is None:
        return AiApplyResult.APPLIED, ("delete", None)
    if s.op == "change_kind" and _drawn(ai, item, s.payload):
        # 검수자가 같은 내용을 직접 그려 뒀다: 원 항목만 빼고 새 항목은 더하지 않는다 (같은 레코드
        # 둘이면 label_id가 같아 저장이 실패한다)
        return AiApplyResult.APPLIED, ("change_kind", None)
    # 종류 변경은 원 라벨 없는 새 항목이다 (Label Studio에서 종류를 바꿔 붙인 것과 같은 의미)
    origin = label_id if s.op == "set" else None
    new = ReviewedItem(origin, item.stream_id, item.t_start_ms, item.t_end_ms, s.payload)
    return AiApplyResult.APPLIED, (s.op, new)


def apply_decisions(
    ai: AiCollect | None,
    reviewed: list[ReviewedItem],
    decisions: Mapping[str, AiDecision],
    *,
    normalize: Callable[[LabelRecord], LabelRecord] | None = None,
) -> list[ReviewedItem]:
    """검수자 결정을 검수 항목에 반영한다 (AI가 없으면 그대로). 순수 함수 (DB는 `begin`이 읽었다).

    Args:
        ai: `begin` 결과.
        reviewed: 도구 결과 항목 (`from_ls_results`·`from_cvat_tracks`).
        decisions: 라벨 ID → 결정 (`ls_decisions`).
        normalize: 도구로 보낼 때 적용한 변환 (CVAT 좌표 반올림, reconcile과 같은 값).

    Returns:
        적용을 반영한 항목. 기각·결정 없음·적용하지 못한 라벨은 도구 결과 그대로다.
    """
    if ai is None:
        return reviewed
    ai.base = list(reviewed)
    ai.decisions = {k: v for k, v in decisions.items() if v is not AiDecision.NONE}
    by_origin = {x.origin_label_id: x for x in reviewed if x.origin_label_id is not None}
    for label_id in sorted(ai.decisions):
        if ai.decisions[label_id] is not AiDecision.APPLY:
            continue
        result, change = _apply_one(ai, label_id, by_origin.get(label_id), normalize)
        ai.results[label_id] = result
        if change is not None:
            ai.changes[label_id] = change
    return ai.items()


def _valid(ai: AiCollect, outcome: ReviewOutcome) -> bool:
    """새 레코드가 세션 단위 검증을 통과하는가 (`insert_labels`가 할 검사와 같다).

    레코드 ID가 겹치면(서로 또는 이력과) 저장이 원시 DB 오류(IntegrityError)로 실패하므로 그것도
    검증 실패로 본다.
    """
    ids = [r.label_id for r in outcome.new_records]
    if len(set(ids)) != len(ids) or not {x.label_id for x in ai.history}.isdisjoint(ids):
        return False
    try:
        validate_session_labels(ai.ontology, ai.history, outcome.new_records)
    except OntologyViolationError:
        return False
    return True


def settle(
    ai: AiCollect | None,
    outcome: ReviewOutcome,
    replay: Callable[[list[ReviewedItem]], ReviewOutcome],
) -> ReviewOutcome:
    """적용본이 세션 단위 검증에 걸리면 AI 적용만 되돌린다 (모듈 docstring 4단계).

    Args:
        ai: `begin` 결과 (None이면 outcome 그대로).
        outcome: 적용본 항목으로 만든 수집 결과.
        replay: 항목 → 수집 결과 (`collect._outcome_for`, 같은 비교 기준·시각).

    Returns:
        쓸 수집 결과. 되돌린 적용은 `ai.results`에서 invalid_now가 된다.
    """
    if ai is None or not ai.changes or _valid(ai, outcome):
        return outcome
    base = replay(ai.items([]))
    if not _valid(ai, base):
        # 검수자 자신의 결과도 통과하지 못한다: 지금과 같이 저장 검증이 수집을 멈추게 둔다
        return base
    keep: list[str] = []
    for label_id in sorted(ai.changes):
        if _valid(ai, replay(ai.items([*keep, label_id]))):
            keep.append(label_id)
            continue
        ai.results[label_id] = AiApplyResult.INVALID_NOW
        ai.warnings.append(
            f"{label_id}: 제안을 적용하면 세션 검증(상위 구간 연결 등)에 걸려 적용하지 않았다"
        )
    ai.changes = {k: v for k, v in ai.changes.items() if k in keep}
    return replay(ai.items())


# ---------------------------------------------------------------- 결과 기록


def _replacement(
    ai: AiCollect, original: LabelRecord, outcome: ReviewOutcome
) -> LabelRecord | None:
    """종류를 바꿔 다시 그린 새 레코드 (적용한 change_kind의 새 항목, 아니면 같은 손·같은 구간의
    다른 종류 새 레코드). 없으면 None."""
    fresh = [r for r in outcome.new_records if r.parent_label_id is None and not r.retracted]
    change = ai.changes.get(original.label_id)
    if change is not None and change[0] == "change_kind" and change[1] is not None:
        new = change[1]
        hits = [r for r in fresh if r.payload == new.payload and _same_span(r, new)]
        return hits[0] if hits else None
    hand = getattr(original.payload, "hand", None)
    return next(
        (
            r for r in fresh
            if _same_span(r, original) and r.kind != original.kind
            and getattr(r.payload, "hand", None) == hand
        ),
        None,
    )  # fmt: skip


def human_actions(
    ai: AiCollect, outcome: ReviewOutcome
) -> dict[str, tuple[AiHumanAction, LabelRecord | None, tuple[str, ...]]]:
    """보낸 라벨마다 사람 결과 (결과, 결과 페이로드를 담은 레코드, 결과 레코드 ID들).

    이중·블라인드 작업은 결과가 측정 레코드라 na다. 그 밖은 stale → 승인 → 수정 → 삭제(같은 구간에
    다른 종류 새 레코드가 있으면 replaced) 순으로 본다.
    """
    out: dict[str, tuple[AiHumanAction, LabelRecord | None, tuple[str, ...]]] = {}
    approved, stale = set(outcome.approved), set(outcome.stale)
    for lid, original in ai.sent.items():
        if ai.task.mode in _MEASURED:
            out[lid] = (AiHumanAction.NA, None, ())
            continue
        if lid in stale:
            out[lid] = (AiHumanAction.STALE, None, ())
            continue
        if lid in approved:
            out[lid] = (AiHumanAction.APPROVED, original, ())
            continue
        children = [r for r in outcome.new_records if r.parent_label_id == lid]
        ids = tuple(r.label_id for r in children)
        kept = [r for r in children if not r.retracted]
        if kept:
            out[lid] = (AiHumanAction.CORRECTED, kept[0], ids)
        elif children:
            new = _replacement(ai, original, outcome)
            if new is not None:
                out[lid] = (AiHumanAction.REPLACED, new, (*ids, new.label_id))
            else:
                out[lid] = (AiHumanAction.DELETED, None, ids)
        else:
            out[lid] = (AiHumanAction.NA, None, ())
    return out


def _changed_fields(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[str]:
    """두 페이로드 사전에서 값이 다른 키."""
    return sorted(k for k in after if before.get(k) != after.get(k))


def _match(
    s: AiSuggestion | None,
    original: LabelRecord,
    human: AiHumanAction,
    result: LabelRecord | None,
) -> Match:
    """사람 결과가 제안과 일치하는 정도 (제안이 바꾸자고 한 필드 기준).

    - delete: 사람이 지웠으면 exact.
    - change_kind: 사람이 같은 페이로드로 바꿔 그렸으면 exact, 같은 종류로만 바꿨으면 partial.
    - set: 사람이 고친 결과에서 제안이 바꾼 필드가 모두 같으면 exact, 일부면 partial.
    - 그 밖(승인, 다른 행동)은 none. 제안이 없거나 사람 결과를 비교하지 않으면(stale·na) na.
    """
    if s is None or human in (AiHumanAction.STALE, AiHumanAction.NA):
        return "na"
    if s.op == "delete":
        return "exact" if human is AiHumanAction.DELETED else "none"
    if s.payload is None or result is None:
        return "none"
    if s.op == "change_kind":
        if human is not AiHumanAction.REPLACED or result.kind != s.payload.kind:
            return "none"
        return "exact" if result.payload == s.payload else "partial"
    if human is not AiHumanAction.CORRECTED:
        return "none"
    want = s.payload.model_dump()
    keys = _changed_fields(original.payload.model_dump(), want)
    got = result.payload.model_dump()
    hits = sum(1 for k in keys if got.get(k) == want[k])
    if keys and hits == len(keys):
        return "exact"
    return "partial" if hits else "none"


def record_outcome(
    conn: sa.Connection,
    ai: AiCollect | None,
    outcome: ReviewOutcome,
    reviewer_id: str,
    now: datetime,
) -> AiOutcome | None:
    """작업 하나의 AI 결과 기록을 쓴다 (AI가 없으면 None). 수집 트랜잭션에서 한 번 부른다.

    항목은 보낸 라벨마다 하나다 (판정이 없는 라벨은 unit_id·verdict가 None). 보낸 뒤 다른 단계가
    바꾼(stale) 라벨의 적용은 `apply_decisions`가 이미 하지 않고 invalid_now로 둔다 (여기서는 사람
    결과가 stale인데 applied로 남는 일이 없게 한 번 더 맞춘다). 기록은 `outcome.ai`에도 싣는다 (수집
    출력).

    부작용: `ai_review_outcomes` 1행 (같은 작업에 두 번 쓰면 IntegrityError).
    """
    if ai is None:
        return None
    humans = human_actions(ai, outcome)
    items: list[AiOutcomeItem] = []
    for lid, original in ai.sent.items():
        human, result, ids = humans[lid]
        judged = ai.verdicts.get(lid)
        verdict = judged[0] if judged is not None else None
        decision = ai.decisions.get(lid, AiDecision.NONE)
        applied = ai.results.get(lid)
        if human is AiHumanAction.STALE and applied is AiApplyResult.APPLIED:
            applied = AiApplyResult.INVALID_NOW
        items.append(
            AiOutcomeItem(
                label_id=lid, unit_id=judged[1] if judged is not None else None,
                verdict=verdict.verdict if verdict is not None else None, arm=ai.exposure.arm,
                human=human, decision=decision, apply_result=applied,
                match=_match(verdict.suggestion if verdict else None, original, human, result),
                result_label_ids=ids,
            )
        )  # fmt: skip
    record = AiOutcome(
        task_key=ai.task.task_key, session_id=ai.task.session_id, reviewer_id=reviewer_id,
        collected_at=now, items=tuple(items), warnings=tuple(ai.warnings),
    )  # fmt: skip
    insert_ai_outcome(conn, record)
    outcome.ai = record
    return record


def summary_counts(record: AiOutcome) -> dict[str, int]:
    """수집 출력용 집계: 적용·기각·적용 못 함(사유별)."""
    out = {"applied": 0, "dismissed": 0}
    for x in record.items:
        if x.decision is AiDecision.DISMISS:
            out["dismissed"] += 1
        if x.apply_result is AiApplyResult.APPLIED:
            out["applied"] += 1
        elif x.apply_result is not None:
            out[x.apply_result.value] = out.get(x.apply_result.value, 0) + 1
    return out
