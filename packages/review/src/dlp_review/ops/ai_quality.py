"""AI 1차 검수 판정의 품질 측정 (WP17.8, ADR 0040 7.1~7.3).

원자료는 노출 기록(`ai_review_exposures`, 작업에 무엇을 보였나)과 수집 결과 기록
(`ai_review_outcomes`, 사람이 무엇을 했나)이다. 순수 함수(`ai_agreement`, `exposure_stats`,
`confidence_floors`)가 계산하고, DB 함수(`load_ai_rows`, `ai_quality`)가 읽어 엮는다. 지표 구현은
`dlp_eval.metrics`(카파·ECE·Clopper-Pearson 상한)를 쓴다. DB에 쓰지 않는다.

정의
    - 사람이 "바꿈" = corrected·deleted·replaced. 사람이 본 결과 = approved + 바꿈.
      stale(보낸 뒤 다른 단계가 바꿈)·na(이중·블라인드 측정)·판정 없는 라벨은 뺀다.
    - 지적 = AI wrong 또는 suspect. unsure는 지적도 맞음도 아니다 (혼동 행렬에만 든다).
    - 지적 정밀도 = P(바꿈 | 지적), 지적 재현율 = P(지적 | 바꿈), ok 놓침률 = P(바꿈 | AI ok),
      틀림인데 승인 = P(approved | AI wrong), 제안 정확 일치 = P(match = exact | 제안 있음).
    - 편향 없는 추정은 대조군(holdout)으로만 낸다. 보인 집단(shown)의 값은 검수자가 AI를 보고 한
      결과라 자동화 편향 감시용이다 (shown 대 holdout의 틀림인데 승인·ok 놓침 비교).
    - 분모에서 뺀다: 재검수 배정(`:resample`·`:ai-flagged`, 선택이 AI에 의존), 이중 작업(별도 표),
      회수한 작업(수집되지 않아 결과 기록이 없다. 노출 집계에서도 뺀다).
    - 표본 가중: 표본 묶음의 표본 라벨(배정 `sample_label_ids`, 무작위) 하나는 그 배정 묶음
      모집단(표본 + 보류)을 대표한다 (`AiRow.weight` = 모집단 / 표본 수, 전수 검수 라벨 = 1). 계획
      단계는 집단과 무관하게 AI가 지적한 라벨을 표본 묶음에서 빼 전수 검수하고
      (`ops.runner.plan_session`의 `escalated_ids`) AI ok 라벨은 일부만 표본으로 보므로,
      행을 그대로 세면 대조군에서도 "바뀐 ok 라벨"이 모집단보다 적게 잡혀 재현율·κ가 부풀려진다.
      그래서 판정·사람 결과 지표(지적 정밀도·재현율·ok 놓침률·틀림인데 승인·제안 정확 일치, 종류별,
      κ, ECE)는 행 가중치로 가중한다 (모집단 추정). 분모 행의 가중치가 모두 같으면 개수 비율이고
      상한은 Clopper-Pearson 정확 구간, 다르면 가중치 합 비율이고 상한은 Korn-Graubard
      근사(`dlp_eval.metrics.proportion`, 유효 표본 수 (Σw)²/Σw²)다. 혼동 행렬과 행 수(`rows`)는
      원자료 확인용 행 개수다 (가중치 없음). 제안 적용 지표(보인 제안 중 적용 비율 등)는 검수자
      행동의 개수 비율이다 (가중치 없음).
    - 층화(ok 놓침률): 표본 층(표본 라벨)과 전수 층으로 나눈다. 층마다 가중 비율과 그 한쪽 95%
      상한을 내고(표본 층 안에서도 배정마다 가중치가 다르므로 가중한다), 층 모집단(가중치 합)으로
      가중해 합친다. 합친 값은 전체 가중 비율과 같고, 합친 상한은 층별 상한의 가중 합이다 (보수적
      근사: 층마다 따로 95%라 동시 신뢰 수준은 더 낮다, F-AIR-11).
    - 확신도 ECE: 대조군에서 확신도가 있는 ok·wrong·suspect 판정의 "맞음" = (지적 = 바꿈), 행 가중.
      자기 보고 확신도는 보정 전에는 결정에 쓰지 않는다 (참고 지표).
    - Cohen κ: 대조군의 (지적 여부, 바꿈 여부), unsure 제외, 행 가중.
    - 오류 삽입 누출: 블라인드가 아닌 작업 중 "AI 없음" 화면(shown이 아닌 집단)에서 오류 삽입 작업의
      비율 대 전체에서의 비율 (F-AIR-03).

공개 이름: `Rate`, `weighted_rate`, `AiRow`, `ArmStats`, `StratumRate`, `ApplyStats`, `AiAgreement`,
`AiExposureStats`, `AiQuality`, `ai_agreement`, `exposure_stats`, `confidence_floors`,
`load_ai_rows`, `ai_quality`, `CHANGED`, `FLAGS`.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import sqlalchemy as sa

from dlp_eval.metrics.classification import cohen_kappa, ece
from dlp_eval.metrics.proportion import clopper_pearson_upper, effective_upper
from dlp_review.ai_display import root_assignment_id
from dlp_schema.aireview import (
    AiApplyResult,
    AiArm,
    AiDecision,
    AiExposure,
    AiHumanAction,
    AiOutcome,
    AiVerdict,
    AiVerdictValue,
)
from dlp_schema.db.aireview_repository import (
    get_ai_exposure,
    get_ai_unit,
    list_ai_exposures,
    list_ai_outcomes,
)
from dlp_schema.db.repository import (
    get_assignment,
    get_labels,
    get_review_task,
    list_review_work,
)
from dlp_schema.review import ReviewAssignment, ReviewMode, ReviewTask, ReviewTaskStatus

# 사람이 바꾼 결과
CHANGED = frozenset({AiHumanAction.CORRECTED, AiHumanAction.DELETED, AiHumanAction.REPLACED})
# 사람이 본 결과 (그 밖은 비교하지 않는다)
_SEEN = CHANGED | {AiHumanAction.APPROVED}
# 지적 (사람 확인이 필요하다고 본 판정)
FLAGS = frozenset({AiVerdictValue.WRONG, AiVerdictValue.SUSPECT})
# 품질 지표를 나눠 내는 집단
ARMS = (AiArm.SHOWN, AiArm.HOLDOUT)
Stratum = Literal["sample", "full"]


@dataclass(frozen=True)
class Rate:
    """비율 num/den과 한쪽 95% 상한 (분모가 0이면 둘 다 NaN).

    n_eff가 None이면 num·den은 개수이고 상한은 Clopper-Pearson 정확 구간이다. 표본 가중 비율이면
    num·den은 가중치 합, n_eff는 유효 표본 수 (Σw)²/Σw²이고 상한은 Korn-Graubard 근사다 (모듈
    docstring "표본 가중").
    """

    num: float
    den: float
    n_eff: float | None = None

    @property
    def weighted(self) -> bool:
        """표본 가중 비율인가."""
        return self.n_eff is not None

    @property
    def value(self) -> float:
        """num / den (분모 0이면 NaN)."""
        return self.num / self.den if self.den else math.nan

    @property
    def upper95(self) -> float:
        """한쪽 95% 상한 (`dlp_eval.metrics.proportion`)."""
        if self.n_eff is None:
            return clopper_pearson_upper(round(self.num), round(self.den))
        return effective_upper(min(self.value, 1.0), self.n_eff) if self.den else math.nan


def weighted_rate(hits: Sequence[float], pool: Sequence[float]) -> Rate:
    """가중치 → 비율 (hits: 사건 표본의 가중치, pool: 모든 표본의 가중치).

    분모 표본의 가중치가 모두 같으면 가중 비율과 값이 같은 개수 비율이다 (상한은 정확 구간).
    """
    if len(set(pool)) <= 1:
        return Rate(len(hits), len(pool))
    total = sum(pool)
    return Rate(sum(hits), total, n_eff=total * total / sum(w * w for w in pool))


def _rate(rows: Iterable[AiRow], hit: str, given: str, *, weighted: bool = True) -> Rate:
    """조건(given)을 만족하는 행 중 hit를 만족하는 비율. 조건 이름은 `AiRow`의 bool 속성.

    weighted면 행 가중치(`AiRow.weight`)로 가중한다 (모듈 docstring "표본 가중").
    """
    pool = [r for r in rows if getattr(r, given)]
    if not weighted:
        return Rate(sum(1 for r in pool if getattr(r, hit)), len(pool))
    return weighted_rate([r.weight for r in pool if getattr(r, hit)], [r.weight for r in pool])


@dataclass(frozen=True)
class AiRow:
    """결과 기록 항목 하나 + 맥락 (라벨 하나, 수집한 작업 하나).

    Attributes:
        task_key / label_id / kind: 작업, 라벨, 라벨 종류 (이력에 없으면 None).
        arm: 작업의 노출 집단. verdict / confidence / has_suggestion: 그 라벨의 판정 (없으면 None).
        shown: 화면에 따로 보인 판정인가 (노출 기록 shown).
        human / decision / apply_result / match: 결과 기록 항목 그대로.
        stratum / weight: 층(표본 라벨이면 sample)과 층 모집단 가중치 (표본 = 모집단/표본 수,
            전수 = 1).
        version: 판정 단위의 판정 버전. followup: 재검수 배정의 작업. mode: 작업 배정 방식.
    """

    task_key: str
    label_id: str
    kind: str | None
    arm: AiArm
    verdict: AiVerdictValue | None
    confidence: float | None
    has_suggestion: bool
    shown: bool
    human: AiHumanAction
    decision: AiDecision
    apply_result: AiApplyResult | None
    match: str
    stratum: Stratum = "full"
    weight: float = 1.0
    version: str | None = None
    followup: bool = False
    mode: ReviewMode = ReviewMode.STANDARD

    @property
    def changed(self) -> bool:
        """사람이 바꿨다."""
        return self.human in CHANGED

    @property
    def flagged(self) -> bool:
        """AI가 지적했다 (wrong·suspect)."""
        return self.verdict in FLAGS

    @property
    def ok(self) -> bool:
        """AI가 맞다고 했다."""
        return self.verdict is AiVerdictValue.OK

    @property
    def wrong(self) -> bool:
        """AI가 틀렸다고 했다."""
        return self.verdict is AiVerdictValue.WRONG

    @property
    def approved(self) -> bool:
        """사람이 그대로 승인했다."""
        return self.human is AiHumanAction.APPROVED

    @property
    def exact(self) -> bool:
        """사람 결과가 제안과 정확히 같다."""
        return self.match == "exact"

    @property
    def suggested(self) -> bool:
        """비교할 수 있는 제안이 있다 (match ≠ na)."""
        return self.has_suggestion and self.match != "na"

    @property
    def shown_suggestion(self) -> bool:
        """제안이 화면에 보였다 (집단 shown, 그 판정이 보임)."""
        return self.arm is AiArm.SHOWN and self.shown and self.has_suggestion

    @property
    def applied(self) -> bool:
        """검수자가 적용을 골랐고 적용됐다."""
        return self.apply_result is AiApplyResult.APPLIED

    @property
    def apply_chosen(self) -> bool:
        """검수자가 적용을 골랐다."""
        return self.decision is AiDecision.APPLY

    @property
    def conflict(self) -> bool:
        """적용을 골랐지만 검수자 결과가 우선했다."""
        return self.apply_result is AiApplyResult.CONFLICT_EDITED

    @property
    def invalid(self) -> bool:
        """적용을 골랐지만 지금 검증을 통과하지 못했다."""
        return self.apply_result is AiApplyResult.INVALID_NOW

    @property
    def comparable(self) -> bool:
        """편향 없는·편향 감시 지표의 분모에 드는 행 (판정 있음, 사람이 봄, 재검수 아님)."""
        return self.verdict is not None and self.human in _SEEN and not self.followup


@dataclass(frozen=True)
class ArmStats:
    """한 집단(또는 한 종류)의 판정 지표.

    Attributes:
        rows: 분모에 든 행 수 (`AiRow.comparable`, 가중치 없음).
        confusion: AI 판정 → 사람 결과 → 행 수 (원자료 확인용, 가중치 없음).
        flag_precision / flag_recall / ok_miss / wrong_approved / suggestion_exact: 모듈 docstring
            (행 가중 비율).
    """

    rows: int
    confusion: dict[str, dict[str, int]]
    flag_precision: Rate
    flag_recall: Rate
    ok_miss: Rate
    wrong_approved: Rate
    suggestion_exact: Rate


def _arm_stats(rows: Sequence[AiRow]) -> ArmStats:
    """행 → 집단 지표 (rows는 이미 `comparable`)."""
    confusion: dict[str, dict[str, int]] = {}
    for r in rows:
        assert r.verdict is not None
        cell = confusion.setdefault(r.verdict.value, {})
        cell[r.human.value] = cell.get(r.human.value, 0) + 1
    return ArmStats(
        rows=len(rows),
        confusion=confusion,
        flag_precision=_rate(rows, "changed", "flagged"),
        flag_recall=_rate(rows, "flagged", "changed"),
        ok_miss=_rate(rows, "changed", "ok"),
        wrong_approved=_rate(rows, "approved", "wrong"),
        suggestion_exact=_rate(rows, "exact", "suggested"),
    )


@dataclass(frozen=True)
class StratumRate:
    """층 하나의 ok 놓침률(행 가중)과 모집단 크기 (가중치 합)."""

    rate: Rate
    population: float


@dataclass(frozen=True)
class ApplyStats:
    """보인 집단의 제안 적용 지표.

    Attributes:
        shown_suggestions: 화면에 보인 제안 수. 비율은 모두 개수 비율이다 (검수자 행동, 가중치
            없음).
        apply_rate: 적용을 고른 비율 (보인 제안 중).
        accept_rate: 실제로 적용된 비율 (보인 제안 중, 주간 `suggestion_accept_rate`).
        conflict_edited / invalid_now: 적용을 고른 것 중 검수자 결과가 우선한·지금 검증 실패 비율.
    """

    shown_suggestions: int
    apply_rate: Rate
    accept_rate: Rate
    conflict_edited: Rate
    invalid_now: Rate


@dataclass
class AiAgreement:
    """판정과 사람 결과의 일치 (`ai_agreement`).

    Attributes:
        arms: 집단(shown·holdout) → 지표.
        by_kind: 라벨 종류 → 대조군 지표.
        ok_miss_strata: 층(sample·full) → 대조군 ok 놓침률.
        ok_miss_rate / ok_miss_upper95: 층을 모집단으로 가중해 합친 대조군 ok 놓침률과 그 보수적
            상한.
        ece / kappa: 대조군 확신도 ECE, Cohen κ (행 가중, 자료가 없으면 NaN).
        apply: 보인 집단의 제안 적용 지표.
        excluded: 분모에서 뺀 행 수 (사유 → 수).
    """

    arms: dict[str, ArmStats] = field(default_factory=dict[str, ArmStats])
    by_kind: dict[str, ArmStats] = field(default_factory=dict[str, ArmStats])
    ok_miss_strata: dict[str, StratumRate] = field(default_factory=dict[str, StratumRate])
    ok_miss_rate: float = math.nan
    ok_miss_upper95: float = math.nan
    ece: float = math.nan
    kappa: float = math.nan
    apply: ApplyStats | None = None
    excluded: dict[str, int] = field(default_factory=dict[str, int])


def _excluded(rows: Sequence[AiRow]) -> dict[str, int]:
    """분모에서 뺀 행의 사유별 수 (판정 없음 → 재검수 → stale → 측정 순으로 하나만 센다)."""
    out: dict[str, int] = {}
    for r in rows:
        if r.comparable:
            continue
        if r.verdict is None:
            key = "no_verdict"
        elif r.followup:
            key = "followup"
        elif r.human is AiHumanAction.STALE:
            key = "stale"
        else:
            key = "measured"
        out[key] = out.get(key, 0) + 1
    return out


def _strata(holdout: Sequence[AiRow]) -> tuple[dict[str, StratumRate], float, float]:
    """대조군 ok 놓침률의 층별 값과 가중 합 (값, 보수적 상한)."""
    strata: dict[str, StratumRate] = {}
    for name in ("sample", "full"):
        oks = [r for r in holdout if r.stratum == name and r.ok]
        if oks:
            # 표본 층 안에서도 배정마다 가중치(묶음 모집단/표본)가 달라 행 가중 비율로 낸다
            rate = weighted_rate([r.weight for r in oks if r.changed], [r.weight for r in oks])
            strata[name] = StratumRate(rate, sum(r.weight for r in oks))
    total = sum(s.population for s in strata.values())
    if not total:
        return strata, math.nan, math.nan
    value = sum(s.population * s.rate.value for s in strata.values()) / total
    upper = sum(s.population * s.rate.upper95 for s in strata.values()) / total
    return strata, value, upper


def ai_agreement(rows: Sequence[AiRow]) -> AiAgreement:
    """판정·사람 결과 행 → 일치 지표 (모듈 docstring 정의). 순수 함수."""
    seen = [r for r in rows if r.comparable]
    out = AiAgreement(excluded=_excluded(rows))
    for arm in ARMS:
        part = [r for r in seen if r.arm is arm and r.mode is not ReviewMode.DOUBLE]
        if part:
            out.arms[arm.value] = _arm_stats(part)
    holdout = [r for r in seen if r.arm is AiArm.HOLDOUT and r.mode is not ReviewMode.DOUBLE]
    for kind in sorted({r.kind for r in holdout if r.kind is not None}):
        out.by_kind[kind] = _arm_stats([r for r in holdout if r.kind == kind])
    out.ok_miss_strata, out.ok_miss_rate, out.ok_miss_upper95 = _strata(holdout)
    scored = [r for r in holdout if r.confidence is not None and (r.flagged or r.ok)]
    if scored:
        conf = [r.confidence or 0.0 for r in scored]
        out.ece = ece(
            conf, [r.flagged == r.changed for r in scored], weights=[r.weight for r in scored]
        )
    judged = [r for r in holdout if r.verdict is not AiVerdictValue.UNSURE]
    if judged:
        out.kappa = cohen_kappa(
            [r.flagged for r in judged], [r.changed for r in judged], [r.weight for r in judged]
        )
    shown = [r for r in rows if r.arm is AiArm.SHOWN and not r.followup]
    # 검수자 행동의 개수 비율 (가중치 없음)
    out.apply = ApplyStats(
        shown_suggestions=sum(1 for r in shown if r.shown_suggestion),
        apply_rate=_rate(shown, "apply_chosen", "shown_suggestion", weighted=False),
        accept_rate=_rate(shown, "applied", "shown_suggestion", weighted=False),
        conflict_edited=_rate(shown, "conflict", "apply_chosen", weighted=False),
        invalid_now=_rate(shown, "invalid", "apply_chosen", weighted=False),
    )
    return out


def _precision(rows: Sequence[AiRow]) -> float:
    """P(바꿈 | 행) (행 가중)."""
    total = sum(r.weight for r in rows)
    return sum(r.weight for r in rows if r.changed) / total if total else math.nan


def confidence_floors(
    rows: Sequence[AiRow], target_precision: float
) -> dict[str, tuple[float | None, int, float]]:
    """판정 값별 최소 확신도 조안 (`review.yaml ai_review.show_min_confidence`, `--ai-thresholds`).

    대조군(편향 없음)에서 확신도가 있는 판정마다, 확신도 ≥ t인 판정을 모두 보일 때의 정밀도
    P(바꿈 | 판정, 확신도 ≥ t)(행 가중)가 목표 이상인 가장 낮은 t를 고른다 (보이는 지적을 가장 많이
    남긴다). t 후보는 관측된 확신도 값이고, 같은 확신도의 판정은 함께 보이거나 함께 숨는다 (자기
    보고 확신도는 0.7·0.8·0.9처럼 이산값이라 동률이 흔하다). 결과는 입력 순서와 무관하다. 그때의
    표본 수와 정밀도를 함께 돌려준다. 목표에 닿는 t가 없으면 (None, 표본 수, 전체 정밀도).

    Returns:
        판정 값(wrong·suspect·unsure) → (최소 확신도 | None, 그 이상 표본 수, 정밀도).
    """
    out: dict[str, tuple[float | None, int, float]] = {}
    pool = [r for r in rows if r.comparable and r.arm is AiArm.HOLDOUT and r.confidence is not None]
    for v in (AiVerdictValue.WRONG, AiVerdictValue.SUSPECT, AiVerdictValue.UNSURE):
        mine = [r for r in pool if r.verdict is v]
        if not mine:
            continue
        best: tuple[float | None, int, float] = (None, len(mine), _precision(mine))
        for t in sorted({r.confidence or 0.0 for r in mine}):
            shown = [r for r in mine if (r.confidence or 0.0) >= t]
            precision = _precision(shown)
            if precision >= target_precision:
                best = (t, len(shown), precision)
                break
        out[v.value] = best
    return out


# ---------------------------------------------------------------- 노출


@dataclass(frozen=True)
class AiExposureStats:
    """노출 기록 집계 (회수한 작업 제외).

    Attributes:
        tasks: 집단 → 작업 수.
        coverage: 판정률 평균 (제외 집단 mode_excluded·off 빼고).
        seeded_share_no_ai / seeded_share_all: 블라인드가 아닌 작업 중 "AI 없음" 화면(shown이 아닌
            집단)에서의 오류 삽입 작업 비율 / 전체에서의 비율 (누출 감시, F-AIR-03).
    """

    tasks: dict[str, int]
    coverage: float
    seeded_share_no_ai: Rate
    seeded_share_all: Rate


def exposure_stats(
    exposures: Sequence[AiExposure], tasks: Mapping[str, ReviewTask]
) -> AiExposureStats:
    """노출 기록 → 집계. tasks: 작업 키 → 작업 (방식·회수 상태). 순수 함수."""
    live = [
        e for e in exposures
        if e.task_key in tasks and tasks[e.task_key].status is not ReviewTaskStatus.REVOKED
    ]  # fmt: skip
    counts: dict[str, int] = {}
    for e in live:
        counts[e.arm.value] = counts.get(e.arm.value, 0) + 1
    covered = [e.coverage for e in live if e.arm not in (AiArm.MODE_EXCLUDED, AiArm.OFF)]
    shown = [e for e in live if tasks[e.task_key].mode is not ReviewMode.BLIND]
    no_ai = [e for e in shown if e.arm is not AiArm.SHOWN]

    def seeded(group: Sequence[AiExposure]) -> Rate:
        hits = sum(1 for e in group if tasks[e.task_key].mode is ReviewMode.SEEDED_ERROR)
        return Rate(hits, len(group))

    return AiExposureStats(
        tasks=counts,
        coverage=sum(covered) / len(covered) if covered else math.nan,
        seeded_share_no_ai=seeded(no_ai),
        seeded_share_all=seeded(shown),
    )


# ---------------------------------------------------------------- DB


def _stratum(assignment: ReviewAssignment | None, label_id: str) -> tuple[Stratum, float]:
    """라벨의 층과 가중치 (표본 라벨 = 배정 묶음 모집단 / 표본 수)."""
    if assignment is None or label_id not in assignment.sample_label_ids:
        return "full", 1.0
    n = len(assignment.sample_label_ids)
    return "sample", (n + len(assignment.withheld_label_ids)) / n


@dataclass
class _Cache:
    """행을 만들며 읽은 DB 값 (작업·배정·단위·라벨 종류)."""

    conn: sa.Connection
    units: dict[str, dict[str, AiVerdict]] = field(default_factory=dict[str, dict[str, AiVerdict]])
    versions: dict[str, str] = field(default_factory=dict[str, str])
    kinds: dict[str, dict[str, str]] = field(default_factory=dict[str, dict[str, str]])

    def verdict(self, unit_id: str | None, label_id: str) -> tuple[AiVerdict | None, str | None]:
        """단위의 그 라벨 판정과 단위 판정 버전."""
        if unit_id is None:
            return None, None
        if unit_id not in self.units:
            unit = get_ai_unit(self.conn, unit_id)
            self.units[unit_id] = {v.label_id: v for v in unit.verdicts} if unit else {}
            if unit is not None:
                self.versions[unit_id] = unit.version
        return self.units[unit_id].get(label_id), self.versions.get(unit_id)

    def kind(self, session_id: str, label_id: str) -> str | None:
        """라벨 종류 (세션 이력에서, 세션마다 한 번 읽는다)."""
        if session_id not in self.kinds:
            self.kinds[session_id] = {x.label_id: x.kind for x in get_labels(self.conn, session_id)}
        return self.kinds[session_id].get(label_id)


def _rows_of(cache: _Cache, outcome: AiOutcome) -> list[AiRow]:
    """결과 기록 하나 → 행."""
    conn = cache.conn
    exposure = get_ai_exposure(conn, outcome.task_key)
    task = get_review_task(conn, outcome.task_key)
    aid = task.assignment_id
    assignment = get_assignment(conn, aid) if aid is not None else None
    followup = aid is not None and root_assignment_id(aid) != aid
    shown = set(exposure.shown) if exposure is not None else set[str]()
    rows: list[AiRow] = []
    for x in outcome.items:
        v, version = cache.verdict(x.unit_id, x.label_id)
        stratum, weight = _stratum(assignment, x.label_id)
        rows.append(
            AiRow(
                task_key=outcome.task_key, label_id=x.label_id,
                kind=cache.kind(outcome.session_id, x.label_id), arm=x.arm, verdict=x.verdict,
                confidence=v.confidence if v is not None else None,
                has_suggestion=v is not None and v.suggestion is not None,
                shown=x.label_id in shown, human=x.human, decision=x.decision,
                apply_result=x.apply_result, match=x.match, stratum=stratum, weight=weight,
                version=version, followup=followup, mode=task.mode,
            )
        )  # fmt: skip
    return rows


def load_ai_rows(conn: sa.Connection, since: datetime, until: datetime) -> list[AiRow]:
    """[since, until) 동안 수집한 작업의 결과 기록 → 행 (DB 읽기만)."""
    cache = _Cache(conn)
    rows: list[AiRow] = []
    for outcome in list_ai_outcomes(conn, since, until):
        rows += _rows_of(cache, outcome)
    return rows


@dataclass
class AiQuality:
    """`dlp review quality --ai`의 AI 절 (`QualityReport.ai`).

    Attributes:
        since / until: 기간 [since, until) (결과 기록은 수집 시각, 노출은 작업 생성 시각).
        agreement: 판정·사람 결과 일치. exposure: 노출 집계.
        review_minutes: 집단 → 영상 1시간당 검수 분 (Label Studio 검수 시간, `review_work`).
        double_kappa_by_arm / blind_bias_by_arm: 짝 표준 배정의 집단 → 이중 일치 카파 평균 /
            블라인드 프리라벨 편향 평균 (자동화 편향 감시).
        rows: 계산에 쓴 행 (임계값 조안에 다시 쓴다).
    """

    since: datetime
    until: datetime
    agreement: AiAgreement
    exposure: AiExposureStats
    review_minutes: dict[str, float] = field(default_factory=dict[str, float])
    double_kappa_by_arm: dict[str, float] = field(default_factory=dict[str, float])
    blind_bias_by_arm: dict[str, float] = field(default_factory=dict[str, float])
    rows: list[AiRow] = field(default_factory=list[AiRow])


def _review_minutes(conn: sa.Connection, since: datetime, until: datetime) -> dict[str, float]:
    """집단 → 영상 1시간당 검수 분 (기간 안의 검수 시간 기록 중 노출 기록이 있는 작업, 재검수 제외).

    검수 시간은 Label Studio lead_time이다 (CVAT는 `dlp ops log-work`로 손으로 기록). 지적 밀도에
    따른 층은 나누지 않는다 (후속: 파일럿 자료로 정한다).
    """
    minutes: dict[str, float] = {}
    hours: dict[str, float] = {}
    arms: dict[str, AiArm | None] = {}
    for w in list_review_work(conn, since, until):
        key = w.task_key
        if key is None:  # 작업 없이 손으로 기록한 검수 시간
            continue
        if key not in arms:
            exposure = get_ai_exposure(conn, key)
            aid = exposure.assignment_id if exposure is not None else None
            followup = aid is not None and root_assignment_id(aid) != aid
            arms[key] = exposure.arm if exposure is not None and not followup else None
        arm = arms[key]
        if arm is None:
            continue
        minutes[arm.value] = minutes.get(arm.value, 0.0) + w.seconds / 60
        hours[arm.value] = hours.get(arm.value, 0.0) + w.video_ms / 3_600_000
    return {k: minutes[k] / hours[k] for k in minutes if hours.get(k)}


def _by_arm(values: Mapping[str, float], arm_of: Mapping[str, AiArm]) -> dict[str, float]:
    """배정 → 값을 집단별 평균으로 (집단을 모르는 배정·NaN은 뺀다)."""
    groups: dict[str, list[float]] = {}
    for aid, value in values.items():
        arm = arm_of.get(aid)
        if arm is not None and not math.isnan(value):
            groups.setdefault(arm.value, []).append(value)
    return {k: sum(v) / len(v) for k, v in groups.items()}


def ai_quality(
    conn: sa.Connection,
    since: datetime,
    until: datetime,
    *,
    double_kappa: Mapping[str, float] | None = None,
    blind_bias: Mapping[str, float] | None = None,
    pairs: Mapping[str, str] | None = None,
) -> AiQuality:
    """기간의 AI 품질 지표 (DB 읽기만).

    Args:
        double_kappa: 이중 배정 → 일치 카파 (`QualityReport.double`의 kappa). 이중 작업의 노출
            집단은 짝 표준 배정의 집단을 따르므로 그 배정 자신의 노출 집단으로 묶는다.
        blind_bias: 블라인드 배정 → 프리라벨 편향. 짝 표준 배정(`pairs`)의 노출 집단으로 묶는다.
        pairs: 배정 → 짝 표준 배정 ID.
    """
    exposures = list_ai_exposures(conn, since, until)
    tasks = {e.task_key: get_review_task(conn, e.task_key) for e in exposures}
    rows = load_ai_rows(conn, since, until)
    arm_of_assignment = {
        e.assignment_id: e.arm for e in exposures if e.assignment_id is not None
    }  # fmt: skip
    pair_arm = {
        aid: arm_of_assignment[p] for aid, p in (pairs or {}).items() if p in arm_of_assignment
    }  # fmt: skip
    return AiQuality(
        since=since, until=until, agreement=ai_agreement(rows),
        exposure=exposure_stats(exposures, tasks),
        review_minutes=_review_minutes(conn, since, until),
        double_kappa_by_arm=_by_arm(double_kappa or {}, arm_of_assignment),
        blind_bias_by_arm=_by_arm(blind_bias or {}, pair_arm),
        rows=rows,
    )  # fmt: skip
