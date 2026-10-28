# 외부 도구(CVAT) JSON을 다루는 부분이 있어 알 수 없는 타입 경고를 이 모듈에서만 끈다.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false

"""AI 1차 검수 판정을 사람 검수에 얹는다: 노출 집단·표시·우선순위·표본 제외·필수 게이트 (WP17.6).

판정(`dlp_schema.aireview`, `dlp aireview run`이 추가 전용 표에 쓴다)은 참고용이고 라벨도 검수
상태도 아니다 (ADR 0040). 이 모듈은 판정을 읽어 검수 화면에 **표시만** 하고, 사람 검수를 **늘리는**
쪽(먼저 볼 단위, 표본 제외, 늦은 지적의 재검수)으로만 쓴다. 채우는 내용(Label Studio 예측의 라벨
결과, CVAT 트랙)은 원 자동 라벨 그대로라서 아무것도 누르지 않고 제출하면 결과가 AI 없을 때와 같다.

유효한 판정의 두 프로필 (`dlp_schema.aireview.valid_verdicts`, `load_ai_view`)
    - 표시(display): 단위의 렌더 해시가 작업 매체의 렌더 해시와 같아야 한다 (I10, fail-closed).
    - 증가(escalate): 렌더 해시를 보지 않는다. 다른 스트림의 블러 변경으로 세션이 AUTO_BLURRED가
      돼도 표본 제외·마무리 보호·우선순위가 꺼지지 않는다 (I9).
    허용 버전은 CLI가 넘긴다 (`AiDisplayConfig.versions`: review.yaml `ai_review.versions`, 비면
    aireview.yaml 지금 버전). 검수 패키지는 판정 정책(aireview.yaml)을 읽지 않는다 (의존 표).

노출 집단 (`exposure_for`, 작업 생성 때 고정해 `ai_review_exposures`에 쓴다)
    - 블라인드·오류 삽입·QA·`show_modes` 밖 방식: mode_excluded (판정은 hidden).
    - 대조군: 루트 배정 ID(재검수 접미사 `:resample`·`:ai-flagged`를 뗀 것), 이중 배정은 짝 표준
      배정의 루트, 배정 없는 직접 작업은 작업 키로 sha256(키|seed) < holdout_ratio → holdout.
    - 작업 라벨에 유효한 표시 판정이 없으면 no_verdicts, 있으면 shown. shown이어도 맞음(ok), 최소
      확신도 미만, 표시 보정이 숨긴 칸(`AiDisplayConfig.calibration`, ADR 0062), 10분당 상한·CVAT
      이슈 상한을 넘은 판정은 따로 표시하지 않는다 (hidden). 보정은 증가 프로필을 바꾸지 않는다.
    - 화면 요약은 holdout·no_verdicts·mode_excluded 모두 "AI 1차 검수: 없음"으로 같다.
    - 판정률(coverage) = 작업 라벨 중 판정 대상(`dlp_schema.aireview.is_judge_target`, 판정 실행과
      같은 조건·같은 정책 값 `AiDisplayConfig.targets`)에서 유효한 표시 판정이 있는 비율. 장갑 센서
      출처 라벨·너무 긴 상위 구간·원본 접근 이벤트처럼 판정하지 않는 라벨은 분모에 들지 않는다.

공개 이름
    - `AiDisplayConfig`, `AiView`, `load_ai_view`.
    - `root_assignment_id`, `arm_key`, `is_holdout`, `exposure_for`.
    - 글·범위: `summary_text`, `suggestion_text`, `note_text`, `master_clock`, `flag_range`,
      `flag_region_id`. 상수 `VERDICT_KO`, `PROBLEM_KO`, `FIELD_KO`, `REASONS`(판정 → 사유).
    - Label Studio: `ls_ai_results`(예측), `ls_label_decisions`(검수자 결정 → 라벨 ID별, AI 구간에
      붙은 결정을 원 라벨로 되돌린다).
    - CVAT(1단계, 프로젝트 변경 없음): `issue_text`, `issue_frame`, `issue_position`,
      `cvat_ai_issues`.
    - 운영: `ai_flagged_spans`(우선순위 사유), `escalated_ids`(표본 제외·마무리 보호 대상),
      판정단 불일치(`AiView.disagree`, `ai_review.disagreement`, 사유 AI_DISAGREEMENT, ADR 0063),
      `GateDecision`·`required_gate`(required 모드의 배정 게이트, CLI `dlp review plan`).

시각: 시간 라벨 판정의 근거 시각은 마스터 ms, 공간 라벨은 그 스트림 PTS ms다 (ADR 0019). 화면의
근거 시각 표기는 세션 기준 mm:ss.s(Label Studio)와 스트림 기준 초(CVAT)다.
"""

from __future__ import annotations

import bisect
import hashlib
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import sqlalchemy as sa

from dlp_review.ai_calibration import AiCalibration
from dlp_review.clients import CvatClient
from dlp_review.labelstudio import (
    AI_FLAGS,
    AI_LABELS,
    AI_NOTE,
    AI_SUMMARY,
    TO_NAME,
    ls_ai_decisions,
)
from dlp_review.ops.policy import AiReviewDisplayPolicy
from dlp_schema.aireview import (
    AiArm,
    AiDecision,
    AiExposure,
    AiProblem,
    AiTargetRule,
    AiVerdict,
    AiVerdictValue,
    is_judge_target,
    split_verdict_ids,
    valid_verdicts,
    valid_verdicts_by_version,
)
from dlp_schema.db.aireview_repository import AiRunState, list_ai_units
from dlp_schema.history import label_class
from dlp_schema.labels import (
    BoxTrackPayload,
    KeypointTrackPayload,
    LabelPayload,
    LabelRecord,
)
from dlp_schema.ontology import Ontology, ontology_digest
from dlp_schema.review import FlaggedSpan, ReviewAssignment, ReviewMode, ReviewReason, ReviewTask

# 라벨 하나의 유효한 판정: (가장 나쁜 판정, 그 단위 ID)
Judged = tuple[AiVerdict, str]
# 재검수 배정 접미사 (`ops.results.finish_assignment`가 붙인다). 대조군은 접미사를 뗀 루트로 정한다
RESAMPLE_SUFFIXES = (":resample", ":ai-flagged")
# 따로 표시하는 판정 값 (ok는 요약 개수로만 보인다)
FLAG_VALUES = (AiVerdictValue.WRONG, AiVerdictValue.SUSPECT, AiVerdictValue.UNSURE)
# 표시 순서 (나쁜 것부터)
_SEVERITY = {AiVerdictValue.WRONG: 0, AiVerdictValue.SUSPECT: 1, AiVerdictValue.UNSURE: 2}
# 판정 값 → 화면 글자·우선순위 사유
VERDICT_KO = {
    AiVerdictValue.OK: "맞음",
    AiVerdictValue.SUSPECT: "의심",
    AiVerdictValue.WRONG: "틀림",
    AiVerdictValue.UNSURE: "판단 불가",
}
REASONS = {
    AiVerdictValue.WRONG: ReviewReason.AI_WRONG,
    AiVerdictValue.SUSPECT: ReviewReason.AI_SUSPECT,
    AiVerdictValue.UNSURE: ReviewReason.AI_UNSURE,
}
# 문제 유형 → 짧은 한국어 (제안이 없는 지적의 메모)
PROBLEM_KO = {
    AiProblem.WRONG_VERB: "동사가 다름", AiProblem.WRONG_TARGET: "대상이 다름",
    AiProblem.WRONG_TOOL: "도구가 다름", AiProblem.WRONG_HAND: "손이 다름",
    AiProblem.SHOULD_BE_GAP: "사이 구간이어야 함", AiProblem.BOUNDARY_EARLY: "경계가 이름",
    AiProblem.BOUNDARY_LATE: "경계가 늦음", AiProblem.SHOULD_BE_ACTION: "행동이어야 함",
    AiProblem.WRONG_GAP_TYPE: "사이 구간 유형이 다름",
    AiProblem.CONTACT_MISMATCH_SENSOR: "장갑 센서와 접촉이 다름",
    AiProblem.WRONG_CONTACT_KIND: "접촉 대상 종류가 다름",
    AiProblem.WRONG_CONTACT_TARGET: "접촉 대상이 다름", AiProblem.WRONG_GRASP: "파지가 다름",
    AiProblem.FALSE_EVENT: "없는 이벤트", AiProblem.WRONG_EVENT_TYPE: "이벤트 유형이 다름",
    AiProblem.WRONG_SEVERITY: "심각도가 다름", AiProblem.WRONG_REF: "상위 구간이 다름",
    AiProblem.HIERARCHY_INCONSISTENT: "하위 행동과 맞지 않음",
    AiProblem.WRONG_STATE_VALUE: "상태 값이 다름", AiProblem.WRONG_CLASS: "클래스가 다름",
    AiProblem.FALSE_POSITIVE: "없는 대상", AiProblem.BOX_MISALIGNED: "상자가 어긋남",
    AiProblem.IDENTITY_SWITCH: "개체가 바뀜", AiProblem.TRACK_DRIFT: "추적이 밀림",
    AiProblem.HAND_SIDE_SWAPPED: "손이 바뀜", AiProblem.KEYPOINT_MISPLACED: "관절 위치가 어긋남",
    AiProblem.MISSING_EVENT: "빠진 이벤트", AiProblem.MISSING_OBJECT: "빠진 물체",
    AiProblem.NOT_VISIBLE: "보이지 않음", AiProblem.INSUFFICIENT_FRAMES: "프레임 부족",
    AiProblem.OTHER: "기타", AiProblem.RESPONSE_INVALID: "응답 형식 오류",
}  # fmt: skip
# 페이로드 필드 → 화면 이름 (제안 메모에 바뀌는 값만 쓴다. 시각·좌표는 제안하지 않는다)
FIELD_KO = {
    "verb": "동사", "gap_type": "사이 구간", "target_id": "대상", "tool_id": "도구", "hand": "손",
    "contact_target_kind": "접촉 대상 종류", "grasp_type": "파지", "body_part": "부위",
    "event_type": "이벤트", "severity": "심각도", "ref_id": "상위 구간", "value": "상태 값",
    "class_id": "클래스",
}  # fmt: skip
KIND_KO = {"action": "행동", "gap": "사이 구간"}
HAND_KO = {"right": "오른손", "left": "왼손"}
# 10분 (ms): 표시 상한 묶음 폭
_TEN_MIN_MS = 600_000


@dataclass(frozen=True)
class AiDisplayConfig:
    """검수에 AI 판정을 쓰는 설정 (`ReviewSetup.ai`). None이면 끔(off)이다.

    CLI(`dlp_cli.review_config.ai_display_config`)가 고른 검수 정책의 `ai_review`와 허용 판정 버전
    (정책 `versions`, 비면 aireview.yaml 지금 버전), 판정 대상 조건의 정책 값(aireview.yaml)으로
    만든다. mode=off면 만들지 않는다.

    Attributes:
        policy: review.yaml `ai_review`.
        versions: 허용 판정 버전 (비어 있으면 판정을 하나도 쓰지 않는다).
        targets: 판정 대상 조건의 정책 값 (`AiReviewPolicy.target_rule`). 판정률의 분모를 판정
            실행과 같은 조건(`is_judge_target`)으로 센다.
        calibration: 판정 표시 보정 (`dlp_review.ai_calibration`, ADR 0062). None이면 보정 없음.
            CLI가 review.yaml `ai_review.calibration.artifact`로 고정한 산출물을 읽어 넘긴다. 따로
            표시할 의심·틀림만 거르고 순서를 정한다 (증가 프로필·ok는 바꾸지 않는다, I9).
    """

    policy: AiReviewDisplayPolicy
    versions: tuple[str, ...]
    targets: AiTargetRule
    calibration: AiCalibration | None = None


@dataclass(frozen=True)
class AiView:
    """세션의 유효한 판정 두 프로필 (`load_ai_view`).

    Attributes:
        display: 표시 프로필 (렌더 해시 일치). 렌더 해시를 모르면 빈 사전.
        escalate: 증가 프로필 (렌더 무관).
        disagree: 판정단이 갈린 라벨 (증가 프로필, ADR 0063). `ai_review.disagreement.enabled`가
            아니면 비어 있다. 갈린 라벨은 ok 아닌 판정이 있어 `escalate`에도 있다.
    """

    display: dict[str, Judged]
    escalate: dict[str, Judged]
    disagree: frozenset[str] = frozenset()


def load_ai_view(
    conn: sa.Connection,
    session_id: str,
    current: Sequence[LabelRecord],
    render_hashes: Mapping[str, str] | None,
    cfg: AiDisplayConfig,
    *,
    ontology: Ontology | None = None,
) -> AiView:
    """세션의 판정 단위(허용 버전)를 읽어 두 프로필의 유효한 판정을 계산한다 (DB 읽기만).

    Args:
        current: 지금 운영 라벨 (`current_labels(get_labels(...))`).
        render_hashes: 스트림 → 지금 렌더 해시. None이면 표시 프로필을 비운다 (승인 상태가 아니거나
            렌더를 모를 때: fail-closed). 증가 프로필은 늘 계산한다.
        ontology: 세션 온톨로지. 주면 표시 프로필은 지금 온톨로지 내용 해시로 판정한 단위만 쓴다
            (F-AIR-32). 증가 프로필은 보지 않는다.
    """
    units = list_ai_units(conn, session_id, versions=cfg.versions)
    strict = cfg.policy.strict_context
    display = (
        valid_verdicts(
            units, current, versions=cfg.versions, render_hashes=render_hashes,
            strict_context=strict,
            ontology_digest=ontology_digest(ontology) if ontology is not None else None,
        )
        if render_hashes is not None
        else {}
    )  # fmt: skip
    escalate = valid_verdicts(
        units, current, versions=cfg.versions, render_hashes=None, strict_context=strict
    )
    disagree: frozenset[str] = frozenset()
    if cfg.policy.disagreement.enabled:
        # 판정단 불일치 (ADR 0063): 증가 프로필과 같은 유효성 조건으로 버전마다 따로 본다
        per_version = valid_verdicts_by_version(
            units, current, versions=cfg.versions, render_hashes=None, strict_context=strict
        )
        disagree = frozenset(split_verdict_ids(per_version))
    return AiView(display=display, escalate=escalate, disagree=disagree)


# ---------------------------------------------------------------- 노출 집단


def root_assignment_id(assignment_id: str) -> str:
    """재검수 접미사(`:resample`, `:ai-flagged`, 겹쳐도)를 뗀 루트 배정 ID."""
    root = assignment_id
    while root.endswith(RESAMPLE_SUFFIXES):
        root = root.rsplit(":", 1)[0]
    return root


def arm_key(task: ReviewTask, assignment: ReviewAssignment | None) -> str:
    """대조군 판정 키: 이중은 짝 표준 배정의 루트, 그 밖의 배정은 루트, 직접 작업은 작업 키."""
    if assignment is None:
        return task.task_key
    if assignment.mode is ReviewMode.DOUBLE and assignment.pair_id is not None:
        return root_assignment_id(assignment.pair_id)
    return root_assignment_id(assignment.assignment_id)


def is_holdout(key: str, policy: AiReviewDisplayPolicy) -> bool:
    """sha256(`<키>|<seed>`) 앞 8바이트 / 2^64 < holdout_ratio (같은 키는 늘 같은 결과)."""
    digest = hashlib.sha256(f"{key}|{policy.holdout_seed}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < policy.holdout_ratio


def _floor_ok(verdict: AiVerdict, policy: AiReviewDisplayPolicy) -> bool:
    """최소 확신도를 넘는가 (확신도가 없으면 하한이 0일 때만)."""
    floor = getattr(policy.show_min_confidence, verdict.verdict.value)
    if verdict.confidence is None:
        return floor <= 0
    return verdict.confidence >= floor


def _flag_time(label: LabelRecord, verdict: AiVerdict) -> int:
    """표시 묶음·정렬에 쓰는 시각: 근거 시각, 없으면 라벨 시작."""
    return verdict.evidence_t_ms if verdict.evidence_t_ms is not None else label.t_start_ms


def _calibrated(
    calibration: AiCalibration | None, label: LabelRecord, verdict: AiVerdict
) -> tuple[bool, float]:
    """보정 (표시 허용, 표시 순서 가중). 보정이 없으면 (참, 0) = 지금 동작."""
    if calibration is None:
        return True, 0.0
    v, kind, cls = verdict.verdict.value, label.kind, label_class(label)
    return calibration.allows(v, kind, cls, verdict.confidence), calibration.weight(v, kind, cls)


def _pick_shown(
    valid: dict[str, Judged],
    labels: Mapping[str, LabelRecord],
    policy: AiReviewDisplayPolicy,
    limit: int | None,
    calibration: AiCalibration | None = None,
) -> dict[str, str]:
    """따로 표시할 판정 {라벨 ID: 단위 ID}: 틀림·의심·판단 불가, 최소 확신도 이상, 보정이 허용한 것,
    10분당 상한, 작업당 상한(limit, CVAT 이슈) 안. 나쁜 것부터, 같은 심각도는 보정 가중이 큰 것부터
    (보정이 없으면 모두 0), 그다음 시각·라벨 ID 순."""
    flags: list[tuple[str, AiVerdict, str, float]] = []
    for lid, (v, uid) in valid.items():
        if v.verdict not in FLAG_VALUES or not _floor_ok(v, policy):
            continue
        allowed, weight = _calibrated(calibration, labels[lid], v)
        if allowed:
            flags.append((lid, v, uid, weight))
    flags.sort(key=lambda f: (_SEVERITY[f[1].verdict], -f[3], _flag_time(labels[f[0]], f[1]), f[0]))
    per_bucket: dict[int, int] = {}
    shown: dict[str, str] = {}
    for lid, v, uid, _weight in flags:
        if limit is not None and len(shown) >= limit:
            break
        bucket = _flag_time(labels[lid], v) // _TEN_MIN_MS
        if per_bucket.get(bucket, 0) >= policy.max_flags_per_10min:
            continue
        per_bucket[bucket] = per_bucket.get(bucket, 0) + 1
        shown[lid] = uid
    return shown


def exposure_for(
    task: ReviewTask,
    assignment: ReviewAssignment | None,
    view: AiView,
    sent: Sequence[LabelRecord],
    cfg: AiDisplayConfig,
    o: Ontology,
    *,
    limit: int | None = None,
) -> AiExposure:
    """검수 작업 하나의 노출 기록 (모듈 docstring "노출 집단").

    Args:
        task: 만든 작업 (작업 키·방식·배정·렌더 해시·생성 시각을 쓴다).
        assignment: 작업의 배정 (직접 작업이면 None).
        view: 작업 매체 렌더 해시로 계산한 판정 (`load_ai_view`).
        sent: 작업에 보낸 라벨.
        o: 세션 온톨로지 (판정률 분모의 원본 접근 이벤트 판별).
        limit: 따로 표시할 판정 수 상한 (CVAT 이슈 상한, Label Studio는 None).
    """
    policy = cfg.policy
    labels = {x.label_id: x for x in sent}
    valid = {lid: view.display[lid] for lid in labels if lid in view.display}
    # 판정률: 판정 실행과 같은 대상 조건의 라벨 중 유효한 표시 판정이 있는 비율
    candidates = {x.label_id for x in sent if is_judge_target(x, o, cfg.targets)}
    judged = len(candidates & set(valid))
    key = arm_key(task, assignment)
    shown: dict[str, str] = {}
    if task.mode.value not in policy.show_modes:
        arm = AiArm.MODE_EXCLUDED
    elif is_holdout(key, policy):
        arm = AiArm.HOLDOUT
    elif not valid:
        arm = AiArm.NO_VERDICTS
    else:
        arm = AiArm.SHOWN
        shown = _pick_shown(valid, labels, policy, limit, cfg.calibration)
    return AiExposure(
        task_key=task.task_key, session_id=task.session_id, assignment_id=task.assignment_id,
        arm=arm, arm_key=key, versions=cfg.versions, render_hash=task.render_hash,
        coverage=judged / len(candidates) if candidates else 0.0,
        shown=shown, hidden={lid: uid for lid, (_, uid) in valid.items() if lid not in shown},
        created_at=task.created_at,
    )  # fmt: skip


# ---------------------------------------------------------------- 화면 글


def _valid_of(exposure: AiExposure, view: AiView) -> dict[str, Judged]:
    """노출 기록의 판정 (보인 것 + 숨긴 것)."""
    ids = (*exposure.shown, *exposure.hidden)
    return {lid: view.display[lid] for lid in ids if lid in view.display}


def summary_text(exposure: AiExposure, view: AiView) -> str:
    """작업 단위 요약. shown이 아니면 집단과 상관없이 같은 "AI 1차 검수: 없음"이다."""
    if exposure.arm is not AiArm.SHOWN:
        return "AI 1차 검수: 없음"
    counts = {v: 0 for v in VERDICT_KO}
    for verdict, _ in _valid_of(exposure, view).values():
        counts[verdict.verdict] += 1
    parts = [f"{VERDICT_KO[v]} {counts[v]}" for v in (*FLAG_VALUES, AiVerdictValue.OK)]
    text = "AI 1차 검수(참고용·사람 검수 아님): " + " · ".join(parts)
    text += f" · 판정률 {exposure.coverage:.0%}"
    hidden_flags = sum(counts[v] for v in FLAG_VALUES) - len(exposure.shown)
    if hidden_flags > 0:
        text += f" · 표시 상한·확신도로 숨긴 지적 {hidden_flags}"
    return text


def _terms(field: str, o: Ontology) -> Mapping[str, Any]:
    """필드 → 그 값의 온톨로지 사전 (한국어 이름 `ko`가 있는 항목). 개체 ID 필드는 빈 사전."""
    if field == "ref_id":
        substeps = {s: t for task in o.tasks.values() for s, t in task.substeps.items()}
        return {**o.verbs, **o.tasks, **substeps}
    tables: dict[str, Mapping[str, Any]] = {
        "verb": o.verbs, "gap_type": o.gap_types, "contact_target_kind": o.contact_target_kinds,
        "grasp_type": o.grasp_types, "body_part": o.body_parts, "event_type": o.events,
        "class_id": o.objects,
    }  # fmt: skip
    return tables.get(field, {})


def _ko_value(field: str, value: Any, o: Ontology) -> str:
    """페이로드 값 → 화면 글 (온톨로지 한국어 이름, 없으면 값 그대로, 손은 오른손·왼손)."""
    if value is None:
        return "없음"
    text = str(getattr(value, "value", value))
    if field == "hand":
        return HAND_KO.get(text, text)
    term = _terms(field, o).get(text)
    return str(term.ko) if term is not None else text


def suggestion_text(label: LabelRecord, verdict: AiVerdict, o: Ontology) -> str:
    """제안 한 줄 (예: `동사 잡다→밀다 제안`, `행동→사이 구간(대기) 제안`, `삭제 제안`).

    제안이 없으면 문제 유형 글 (예: `동사가 다름`).
    """
    s = verdict.suggestion
    if s is None:
        return ", ".join(PROBLEM_KO[p] for p in verdict.problems) or VERDICT_KO[verdict.verdict]
    if s.op == "delete" or s.payload is None:
        return "삭제 제안"
    new: LabelPayload = s.payload
    if s.op == "change_kind":
        detail = _ko_value("gap_type", getattr(new, "gap_type", None), o)
        if new.kind == "action":
            detail = _ko_value("verb", getattr(new, "verb", None), o)
        return f"{KIND_KO.get(label.kind, label.kind)}→{KIND_KO[new.kind]}({detail}) 제안"
    old, after = label.payload.model_dump(), new.model_dump()
    changed = [
        f"{name} {_ko_value(f, old.get(f), o)}→{_ko_value(f, after[f], o)}"
        for f, name in FIELD_KO.items()
        if f in after and after[f] != old.get(f)
    ]
    return " · ".join(changed) + " 제안" if changed else "값 확인 제안"


def master_clock(ms: int) -> str:
    """마스터 ms → `mm:ss.s` (세션 기준 절대 시각, 한 시간이 넘으면 `h:mm:ss.s`)."""
    tenths = int(ms) // 100
    h, rest = divmod(tenths, 36_000)
    m, rest = divmod(rest, 600)
    sec = f"{rest // 10:02d}.{rest % 10}"
    return f"{h}:{m:02d}:{sec}" if h else f"{m:02d}:{sec}"


def note_text(
    label: LabelRecord, verdict: AiVerdict, o: Ontology, hotkeys: Mapping[str, str] | None
) -> str:
    """라벨 구간 메모: `[틀림 · 참고] <제안·문제> · <이유> · 근거 mm:ss.s · r 적용 / h 기각`.

    hotkeys: 결정 단축키 (`review_ui.yaml ai.hotkeys`, 없으면 단축키 안내를 뺀다). 제안이 없으면
    적용 안내도 뺀다.
    """
    parts = [f"[{VERDICT_KO[verdict.verdict]} · 참고] {suggestion_text(label, verdict, o)}"]
    if verdict.reason_ko:
        parts.append(verdict.reason_ko)
    if verdict.evidence_t_ms is not None:
        parts.append(f"근거 {master_clock(verdict.evidence_t_ms)}")
    if hotkeys:
        keys = [f"{hotkeys['dismiss']} 기각"]
        if verdict.suggestion is not None:
            keys.insert(0, f"{hotkeys['apply']} 적용")
        parts.append(" / ".join(keys))
    return " · ".join(parts)


def flag_range(
    label: LabelRecord, verdict: AiVerdict, cap_ms: int, window_ms: int
) -> tuple[int, int]:
    """AI 구간 범위: 라벨이 cap_ms 이하이면 라벨 구간 전체, 길면 근거 시각(없으면 라벨 시작) ±
    window_ms/2를 라벨 구간 안으로 자른 것."""
    start, end = label.t_start_ms, label.t_end_ms
    if end - start <= cap_ms:
        return start, end
    center = min(max(_flag_time(label, verdict), start), end)
    lo = max(start, center - window_ms // 2)
    return lo, min(end, lo + window_ms)


def flag_region_id(unit_id: str, label_id: str) -> str:
    """AI 구간 결과 id `ai-<sha256(단위+라벨) 앞 8자>` (라벨 ID와 겹치지 않는다)."""
    return "ai-" + hashlib.sha256(f"{unit_id}|{label_id}".encode()).hexdigest()[:8]


def ls_ai_results(
    exposure: AiExposure,
    view: AiView,
    sent: Sequence[LabelRecord],
    o: Ontology,
    policy: AiReviewDisplayPolicy,
    hotkeys: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Label Studio 예측에 더할 AI 결과: 작업 요약 하나 + 보인 판정마다 AI 구간(readonly)과 라벨
    구간 메모. 프리라벨 결과(`to_ls_results`)는 건드리지 않는다 (더하기만 한다)."""
    out: list[dict[str, Any]] = [{
        "id": "ai-summary", "from_name": AI_SUMMARY, "to_name": TO_NAME, "type": "textarea",
        "value": {"text": [summary_text(exposure, view)]},
    }]  # fmt: skip
    labels = {x.label_id: x for x in sent}
    for lid, uid in exposure.shown.items():
        verdict, _ = view.display[lid]
        label = labels[lid]
        start, end = flag_range(label, verdict, policy.span_cap_ms, policy.evidence_window_ms)
        out.append({
            "id": flag_region_id(uid, lid), "from_name": AI_FLAGS, "to_name": TO_NAME,
            "type": "timeserieslabels", "readonly": True,
            "value": {"start": start, "end": end, "instant": start == end,
                      "timeserieslabels": [AI_LABELS[verdict.verdict.value]]},
        })  # fmt: skip
        out.append({
            "id": lid, "from_name": AI_NOTE, "to_name": TO_NAME, "type": "textarea",
            "value": {"text": [note_text(label, verdict, o, hotkeys)]},
        })  # fmt: skip
    return out


def ls_label_decisions(
    results: list[dict[str, Any]], exposure: AiExposure, known_ids: Collection[str]
) -> tuple[dict[str, AiDecision], list[str]]:
    """Label Studio 검수자 결과의 AI 결정(`labelstudio.ls_ai_decisions`, 구간 id별) → 라벨 ID별.

    결정은 화면에서 라벨 구간을 고른 뒤 고르지만, 결정 단축키는 어느 구간을 골랐든 눌린다. 검수자가
    빨간 AI 구간(`ai_flags`)을 고른 채 단축키를 누르면 결정이 AI 구간 id(`flag_region_id`)에
    붙는다. 그 결정은 노출 기록의 보인 판정(`exposure.shown` {라벨 ID: 단위 ID})으로 AI 구간 id를
    다시 계산해 원 라벨로 되돌린다 (검수자가 그 지적을 보고 누른 결정이다).

    Args:
        results: 검수자가 제출한 최신 주석의 결과.
        exposure: 그 작업의 노출 기록 (`ai_review_exposures`).
        known_ids: 작업에 보낸 라벨 ID (`ReviewTask.sent_label_ids`).

    Returns:
        (라벨 ID → 결정, 경고 문장 목록). 버리는 경우는 경고로 남긴다:
        - 작업 라벨도 이 작업의 AI 구간도 아닌 구간의 결정 (새로 그린 구간 등).
        - 한 라벨에 라벨 구간과 AI 구간의 결정이 서로 다름 (어느 쪽인지 알 수 없어 결정 없음).
        보인 지적이 없는 라벨의 결정은 그대로 돌려준다 (적용 단계가 no_suggestion·not_shown으로
        기록한다, WP17.7).
    """
    flags = {flag_region_id(uid, lid): lid for lid, uid in exposure.shown.items()}
    known = set(known_ids)
    found: dict[str, set[AiDecision]] = {}
    unknown = 0
    for rid, decision in ls_ai_decisions(results).items():
        lid = rid if rid in known else flags.get(rid)
        if lid is None:
            unknown += 1
            continue
        found.setdefault(lid, set()).add(decision)
    out = {lid: next(iter(ds)) for lid, ds in found.items() if len(ds) == 1}
    warnings: list[str] = []
    if unknown:
        warnings.append(
            f"작업 라벨이 아닌 구간의 AI 결정 {unknown}개를 무시했다 (새로 그린 구간 등)"
        )
    conflicts = sorted(lid for lid, ds in found.items() if len(ds) > 1)
    if conflicts:
        warnings.append(
            f"라벨 구간과 AI 구간의 결정이 서로 다른 라벨 {len(conflicts)}개는 결정 없음으로 본다 "
            f"(예: {conflicts[0]})"
        )
    return out, warnings


# ---------------------------------------------------------------- CVAT (1단계: 이슈 메모)


def issue_frame(times: Sequence[int], t_ms: int) -> int:
    """스트림 PTS ms → 그 시각에 화면에 떠 있는 CVAT 프레임 번호 (PTS 인덱스, 처음 앞이면 0)."""
    return max(0, min(bisect.bisect_right(times, t_ms) - 1, len(times) - 1))


def _box_at(p: BoxTrackPayload, t: int) -> tuple[float, float, float, float] | None:
    """시각 t의 박스 (앞뒤 키프레임 선형 보간, 화면 밖이면 None)."""
    frames = sorted(p.keyframes, key=lambda k: k.t_ms)
    before = [k for k in frames if k.t_ms <= t]
    after = [k for k in frames if k.t_ms >= t]
    a = before[-1] if before else frames[0]
    b = after[0] if after else frames[-1]
    if a.outside:
        return None
    if b.outside or b.t_ms == a.t_ms:
        return (a.x, a.y, a.w, a.h)
    r = (t - a.t_ms) / (b.t_ms - a.t_ms)

    def mix(va: float, vb: float) -> float:
        return va + (vb - va) * r

    return (mix(a.x, b.x), mix(a.y, b.y), mix(a.w, b.w), mix(a.h, b.h))


def issue_position(label: LabelRecord, t_ms: int) -> list[float]:
    """이슈 표시 위치 [x1, y1, x2, y2] 화소: 그 시각의 박스, 키포인트는 보이는 점의 외접 상자.
    모르면 화면 왼쪽 위 16 * 16 화소."""
    p = label.payload
    if isinstance(p, BoxTrackPayload):
        box = _box_at(p, t_ms)
        if box is not None:
            return [round(box[0], 1), round(box[1], 1), round(box[0] + box[2], 1),
                    round(box[1] + box[3], 1)]  # fmt: skip
    if isinstance(p, KeypointTrackPayload):
        frame = min(p.keyframes, key=lambda f: abs(f.t_ms - t_ms))
        pts = [(k.x, k.y) for k in frame.points if k.visibility > 0]
        if pts:
            xs, ys = [x for x, _ in pts], [y for _, y in pts]
            return [round(min(xs), 1), round(min(ys), 1), round(max(xs), 1), round(max(ys), 1)]
    return [0.0, 0.0, 16.0, 16.0]


def issue_text(label: LabelRecord, verdict: AiVerdict, o: Ontology) -> str:
    """CVAT 이슈 메시지: `[AI 틀림·참고] <제안·문제> · <이유> · 12.4s · 라벨 <ID>`."""
    t = _flag_time(label, verdict)
    parts = [
        f"[AI {VERDICT_KO[verdict.verdict]}·참고] {suggestion_text(label, verdict, o)}",
        verdict.reason_ko,
        f"{t / 1000:.1f}s",
        f"라벨 {label.label_id}",
    ]
    return " · ".join(p for p in parts if p)


def cvat_ai_issues(
    cvat: CvatClient,
    task_id: int,
    exposure: AiExposure,
    view: AiView,
    sent: Sequence[LabelRecord],
    times: Sequence[int],
    o: Ontology,
) -> int:
    """보인 판정(`exposure.shown`, 상한은 노출 계산에서 이미 적용)을 CVAT 이슈로 단다. 단 수.

    프레임은 근거 시각(스트림 PTS)에 화면에 떠 있는 프레임이고, 그 프레임을 담은 job에 단다 (못
    찾으면 첫 job, 블러 검수 이슈 `tasks._segment_issues`와 같은 방식). 수집은 주석만 읽으므로
    이슈는 결과에 영향이 없다. 부작용: CVAT 이슈 생성.
    """
    if not exposure.shown or not times:
        return 0
    jobs = cvat.jobs(task_id)
    if not jobs:
        return 0
    labels = {x.label_id: x for x in sent}
    made = 0
    for lid in exposure.shown:
        verdict, _ = view.display[lid]
        label = labels[lid]
        t = _flag_time(label, verdict)
        frame = issue_frame(times, t)
        job = next(
            (j for j in jobs if int(j["start_frame"]) <= frame <= int(j["stop_frame"])), jobs[0]
        )
        cvat.add_issue(
            int(job["id"]), frame, issue_position(label, t), issue_text(label, verdict, o)
        )
        made += 1
    return made


# ---------------------------------------------------------------- 운영 (증가 프로필)


def escalated_ids(
    escalate: Mapping[str, Judged],
    policy: AiReviewDisplayPolicy,
    disagree: Collection[str] = (),
) -> set[str]:
    """표본 묶음에서 뺄(그리고 마무리에서 표본 검증하지 않을) 라벨: 증가 프로필의 판정 값이
    `sampling_exclude` 안이거나, 판정단이 갈린 라벨(`disagree`, `disagreement.sampling_exclude`일
    때, ADR 0063). AI의 ok는 아무것도 빼지 않는다 (I9)."""
    keep = set(policy.sampling_exclude)
    out = {lid for lid, (v, _) in escalate.items() if v.verdict.value in keep}
    if policy.disagreement.enabled and policy.disagreement.sampling_exclude:
        out |= set(disagree)
    return out


def _span(
    x: LabelRecord, verdict: AiVerdict, reason: ReviewReason, policy: AiReviewDisplayPolicy
) -> FlaggedSpan:
    """라벨 하나의 AI 사유 구간 (`span_cap_ms`보다 길면 근거 시각 주변 그 길이)."""
    start, end = flag_range(x, verdict, policy.span_cap_ms, policy.span_cap_ms)
    return FlaggedSpan(reason=reason, t_start_ms=start, t_end_ms=end, label_ids=(x.label_id,))


def ai_flagged_spans(
    escalate: Mapping[str, Judged],
    labels: Iterable[LabelRecord],
    policy: AiReviewDisplayPolicy,
    disagree: Collection[str] = (),
) -> list[FlaggedSpan]:
    """우선순위 사유 구간 (AI_WRONG·AI_SUSPECT·AI_UNSURE): 증가 프로필 판정이 붙은 라벨의 구간,
    `span_cap_ms`보다 길면 근거 시각 주변 그 길이로 자른다. 가중치는 `priority.weights.ai_*`.

    `disagreement.enabled`면 판정단이 갈린 라벨(`disagree`)에 AI_DISAGREEMENT 구간을 더한다 (같은
    방식, 근거 시각은 증가 프로필의 가장 나쁜 판정, ADR 0063)."""
    split = set(disagree) if policy.disagreement.enabled else set[str]()
    spans: list[FlaggedSpan] = []
    for x in labels:
        judged = escalate.get(x.label_id)
        if judged is None:
            continue
        if judged[0].verdict in REASONS:
            spans.append(_span(x, judged[0], REASONS[judged[0].verdict], policy))
        if x.label_id in split:
            spans.append(_span(x, judged[0], ReviewReason.AI_DISAGREEMENT, policy))
    return spans


@dataclass(frozen=True)
class GateDecision:
    """required 모드의 배정 게이트 결정 (`required_gate`).

    Attributes:
        proceed: 배정을 계획해도 된다.
        waive_timeout: 시간 초과로 진행한다 (호출자가 waived 이벤트를 남긴다).
        reason: 사람에게 보일 이유.
    """

    proceed: bool
    waive_timeout: bool
    reason: str


def required_gate(
    state: AiRunState | None,
    prelabeled_at: datetime | None,
    now: datetime,
    cfg: AiDisplayConfig | None,
) -> GateDecision:
    """`review.yaml ai_review.mode = required`일 때 세션의 배정을 미룰지 정한다 (순수 함수).

    진행: required가 아님 → 허용 버전의 completed 실행이 있음 → 건너뜀(waived) 기록이 있음 →
    prelabeled 전이 뒤 `max_wait_h`가 지남(호출자가 waived:timeout을 남긴다). 그 밖은 미룬다.
    prelabeled 전이 기록이 없으면 시간 초과를 셀 수 없어 미룬다.
    """
    if cfg is None or cfg.policy.mode != "required":
        return GateDecision(True, False, "AI 필수 모드 아님")
    if state is not None and state.completed_versions & set(cfg.versions):
        return GateDecision(True, False, "AI 판정 완료")
    if state is not None and state.waived_at is not None:
        return GateDecision(True, False, f"AI 단계 건너뜀 기록 ({state.waived_at.isoformat()})")
    wait = timedelta(hours=cfg.policy.max_wait_h)
    if prelabeled_at is not None and now - prelabeled_at >= wait:
        return GateDecision(True, True, f"prelabeled 뒤 {cfg.policy.max_wait_h:g}시간 초과")
    since = prelabeled_at.isoformat() if prelabeled_at is not None else "기록 없음"
    return GateDecision(False, False, f"지금 버전의 AI 판정 완료 기록 없음 (prelabeled {since})")
