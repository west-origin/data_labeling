"""AI 판정 표시 보정 학습과 효과 측정 (`dlp aireview calibrate`, ADR 0062).

입력은 `ai_quality.load_ai_rows`의 행(사람 판정 기록 `ai_review_outcomes` + 맥락)이다. DB에 쓰지
않는다.

학습 (`fit_calibration`, 순수 함수):
- 표본: 대조군(holdout)이고 비교할 수 있는 행(`AiRow.comparable`: 판정 있음, 사람이 봄, 재검수
  아님), 이중 작업 제외, 판정 버전이 학습 판정 버전 안. 대조군만 쓰는 이유: 보인 집단의 사람 결과는
  검수자가 AI를 보고 한 것이라 지적을 따라 고친 쪽으로 기운다 (자동화 편향, ADR 0040 6절 3).
  대조군의 사람은 판정을 보지 않았다.
- 칸: 보정 판정 값(정책 verdicts, wrong·suspect)마다 (판정), (판정, 종류), (판정, 종류, 클래스).
  칸의 지적 정밀도 = 행 가중 P(사람이 바꿈 | 그 칸의 지적)이고 상한은 `ai_quality.weighted_rate`의
  한쪽 95% 상한이다 (표본 가중이면 Korn-Graubard 근사). 유효 표본 수가 `min_effective_samples`보다
  적은 칸은 만들지 않는다 (더 넓은 칸 또는 보정 없음 = 지금 동작).
- show: 상한 < `hide_below_precision`일 때만 거짓 (확실히 쓸모없는 지적만 숨긴다, 보수적).
- min_confidence: 그 칸에서 확신도 < t인 지적 묶음이 유효 표본 수 하한을 넘고 그 상한이 문턱보다
  낮은 가장 큰 관측 확신도 t (없으면 None).
- weight: (바꿈 가중 + k·p0) / (가중 합 + k), p0 = 그 판정의 대조군 전체 정밀도, k = prior_strength.

효과 (`calibration_effect`): 같은 대조군 행에서 보정 전후의 지적 정밀도 P(바꿈 | 보인 지적)와 재현율
P(보인 지적 | 바꿈) (행 가중, `ai_quality` 정의). "보인 지적" = wrong·suspect 중 보정이 허용한 것.
같은 자료로 학습·평가하면 낙관적이므로 `cross_checked_effect`가 작업 키 해시로 반을 나눠 한쪽으로
학습하고 다른 쪽에서 잰 값을 합친다.

ok·unsure와 증가 프로필은 보정하지 않는다 (I9, `dlp_review.ai_calibration`).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from dlp_review.ai_calibration import AiCalibration, CalibrationCell, content_id
from dlp_review.ops.ai_quality import AiRow, Rate, weighted_rate
from dlp_review.ops.policy import AiCalibrationFit
from dlp_schema.aireview import AiArm
from dlp_schema.review import ReviewMode

Key = tuple[str, str | None, str | None]


def calibration_pool(rows: Sequence[AiRow], judge_versions: Sequence[str]) -> list[AiRow]:
    """학습·효과 측정에 쓰는 행: 대조군, 비교 가능, 이중 작업 아님, 판정 버전이 목록 안."""
    versions = set(judge_versions)
    return [
        r for r in rows
        if r.comparable and r.arm is AiArm.HOLDOUT and r.mode is not ReviewMode.DOUBLE
        and r.version in versions
    ]  # fmt: skip


def _rate(rows: Sequence[AiRow]) -> Rate:
    """P(바꿈 | 행) (행 가중, 상한 포함)."""
    return weighted_rate([r.weight for r in rows if r.changed], [r.weight for r in rows])


def _n_eff(rate: Rate) -> float:
    """유효 표본 수 (개수 비율이면 분모)."""
    return rate.n_eff if rate.n_eff is not None else rate.den


def _floor(rows: Sequence[AiRow], fit: AiCalibrationFit) -> float | None:
    """확신도 아래 묶음이 확실히 쓸모없는 가장 큰 관측 확신도 (모듈 docstring). 없으면 None."""
    scored = [r for r in rows if r.confidence is not None]
    for t in sorted({r.confidence or 0.0 for r in scored}, reverse=True):
        low = [r for r in scored if (r.confidence or 0.0) < t]
        if not low or len(low) == len(scored):
            continue
        rate = _rate(low)
        if _n_eff(rate) >= fit.min_effective_samples and rate.upper95 < fit.hide_below_precision:
            return t
    return None


def _groups(pool: Sequence[AiRow], verdict: str) -> dict[Key, list[AiRow]]:
    """판정 하나의 칸별 행 (판정 / 판정·종류 / 판정·종류·클래스)."""
    out: dict[Key, list[AiRow]] = {}
    for r in pool:
        if r.verdict is None or r.verdict.value != verdict:
            continue
        keys: list[Key] = [(verdict, None, None)]
        if r.kind is not None:
            keys.append((verdict, r.kind, None))
            if r.label_class is not None:
                keys.append((verdict, r.kind, r.label_class))
        for key in keys:
            out.setdefault(key, []).append(r)
    return out


def _cell(key: Key, rows: Sequence[AiRow], fit: AiCalibrationFit, p0: float) -> CalibrationCell:
    """칸 하나 (모듈 docstring의 show·min_confidence·weight)."""
    rate = _rate(rows)
    upper = min(1.0, max(0.0, rate.upper95))
    show = upper >= fit.hide_below_precision
    k = fit.prior_strength
    weight = (rate.num + k * p0) / (rate.den + k) if rate.den + k else p0
    # verdict는 정책 verdicts(wrong·suspect)에서 온다. 계약 검증으로 다시 확인한다
    return CalibrationCell.model_validate(
        {
            "verdict": key[0], "kind": key[1], "label_class": key[2], "n": len(rows),
            "n_eff": round(_n_eff(rate), 6), "precision": round(rate.value, 6),
            "upper95": round(upper, 6), "show": show,
            "min_confidence": _floor(rows, fit) if show else None,
            "weight": round(min(1.0, max(0.0, weight)), 6),
        }
    )  # fmt: skip


def _rows_digest(pool: Sequence[AiRow]) -> str:
    """학습 행의 정렬 해시 (같은 기록이면 같은 값)."""
    items = sorted(
        (r.task_key, r.label_id, r.verdict.value if r.verdict else "", r.human.value, r.weight,
         r.kind or "", r.label_class or "", -1.0 if r.confidence is None else r.confidence,
         r.version or "")
        for r in pool
    )  # fmt: skip
    return hashlib.sha256(json.dumps(items).encode()).hexdigest()


def fit_calibration(
    rows: Sequence[AiRow],
    fit: AiCalibrationFit,
    judge_versions: Sequence[str],
    since: datetime,
    until: datetime,
) -> AiCalibration:
    """사람 판정 기록 행으로 보정 산출물을 만든다 (순수·결정적, 모듈 docstring "학습")."""
    pool = calibration_pool(rows, judge_versions)
    prior: dict[str, float] = {}
    cells: list[CalibrationCell] = []
    for verdict in sorted(set(fit.verdicts)):
        mine = [r for r in pool if r.verdict is not None and r.verdict.value == verdict]
        if not mine:
            continue
        p0 = _rate(mine).value
        prior[verdict] = round(p0, 6)
        for key, group in sorted(_groups(pool, verdict).items(), key=lambda kv: _sort_key(kv[0])):
            if _n_eff(_rate(group)) >= fit.min_effective_samples:
                cells.append(_cell(key, group, fit, p0))
    body: dict[str, Any] = {
        "calibration_id": "0" * 12,
        "judge_versions": sorted(set(judge_versions)),
        "since": since,
        "until": until,
        "fit": fit.model_dump(mode="json"),
        "source": {
            "rows": len(rows),
            "holdout_rows": len(pool),
            "rows_sha256": _rows_digest(pool),
        },
        "prior": prior,
        "cells": cells,
    }
    draft = AiCalibration.model_validate(body)
    data = draft.model_dump(mode="json")
    return draft.model_copy(update={"calibration_id": content_id(data)})


def _sort_key(key: Key) -> tuple[str, str, str]:
    return key[0], key[1] or "", key[2] or ""


@dataclass(frozen=True)
class CalibrationEffect:
    """보정 전후 지적 지표 (대조군, 행 가중).

    flags_before / flags_after: 보인 지적 행 수 (wrong·suspect, 보정 후는 허용된 것).
    precision_* = P(바꿈 | 보인 지적), recall_* = P(보인 지적 | 바꿈).
    """

    rows: int
    flags_before: int
    flags_after: int
    precision_before: Rate
    precision_after: Rate
    recall_before: Rate
    recall_after: Rate

    def to_json(self) -> dict[str, Any]:
        """보고용 사전 (값·상한, NaN은 None)."""

        def r(rate: Rate) -> dict[str, float | None]:
            def num(v: float) -> float | None:
                return None if math.isnan(v) else round(v, 6)

            return {"value": num(rate.value), "upper95": num(rate.upper95), "den": num(rate.den)}

        return {
            "rows": self.rows,
            "flags_before": self.flags_before,
            "flags_after": self.flags_after,
            "precision_before": r(self.precision_before),
            "precision_after": r(self.precision_after),
            "recall_before": r(self.recall_before),
            "recall_after": r(self.recall_after),
        }


def _allowed(calibration: AiCalibration | None) -> Callable[[AiRow], bool]:
    """행의 지적이 보정 후에도 보이는가."""

    def allowed(r: AiRow) -> bool:
        if not r.flagged:
            return False
        if calibration is None or r.verdict is None:
            return True
        return calibration.allows(r.verdict.value, r.kind, r.label_class, r.confidence)

    return allowed


def _effect(pool: Sequence[AiRow], calibration: AiCalibration | None) -> CalibrationEffect:
    """대조군 행(이미 고른 것)에서 보정 전후 지표."""
    after = _allowed(calibration)
    flagged = [r for r in pool if r.flagged]
    shown = [r for r in pool if after(r)]
    changed = [r for r in pool if r.changed]
    return CalibrationEffect(
        rows=len(pool),
        flags_before=len(flagged),
        flags_after=len(shown),
        precision_before=_rate(flagged),
        precision_after=_rate(shown),
        recall_before=weighted_rate(
            [r.weight for r in changed if r.flagged], [r.weight for r in changed]
        ),
        recall_after=weighted_rate(
            [r.weight for r in changed if after(r)], [r.weight for r in changed]
        ),
    )


def calibration_effect(
    rows: Sequence[AiRow], calibration: AiCalibration | None, judge_versions: Sequence[str]
) -> CalibrationEffect:
    """같은 대조군 행에서 보정 전후 지적 정밀도·재현율 (학습 자료와 같으면 낙관적)."""
    return _effect(calibration_pool(rows, judge_versions), calibration)


def _fold(task_key: str) -> int:
    """작업 키 → 0 또는 1 (sha256 첫 바이트의 짝홀)."""
    return hashlib.sha256(task_key.encode()).digest()[0] & 1


def cross_checked_effect(
    rows: Sequence[AiRow],
    fit: AiCalibrationFit,
    judge_versions: Sequence[str],
    since: datetime,
    until: datetime,
) -> CalibrationEffect:
    """작업 키 해시로 반을 나눠 한쪽으로 학습하고 다른 쪽에서 잰 보정 후 지표를 합친다 (2겹).

    보정 전 지표는 전체 대조군 그대로다. 보정 후는 각 행을 자기를 보지 않은 보정으로 판단한다.
    """
    pool = calibration_pool(rows, judge_versions)
    folds = [[r for r in pool if _fold(r.task_key) == i] for i in (0, 1)]
    fitted = [fit_calibration(folds[1 - i], fit, judge_versions, since, until) for i in (0, 1)]

    def after(r: AiRow) -> bool:
        return _allowed(fitted[_fold(r.task_key)])(r)

    base = _effect(pool, None)
    shown = [r for r in pool if after(r)]
    changed = [r for r in pool if r.changed]
    return CalibrationEffect(
        rows=base.rows,
        flags_before=base.flags_before,
        flags_after=len(shown),
        precision_before=base.precision_before,
        precision_after=_rate(shown),
        recall_before=base.recall_before,
        recall_after=weighted_rate(
            [r.weight for r in changed if after(r)], [r.weight for r in changed]
        ),
    )
