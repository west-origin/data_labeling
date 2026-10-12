"""AI 1차 검수 품질 보고 출력 (`dlp review quality --ai`, WP17.8, ADR 0040 7.3).

계산은 `dlp_review.ops.ai_quality`가 하고 여기서는 글·JSON으로 바꾼다. 편향 없는 값은 대조군
(holdout)이고, 보인 집단(shown) 값은 자동화 편향 감시용이다 (같은 줄에 섞지 않는다).

공개 이름: `print_ai_quality`, `ai_quality_json`, `print_thresholds`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from dlp_review.ops.ai_quality import AiQuality, ArmStats, Rate, confidence_floors

# 집단 → 출력 머리 (편향 여부를 함께 적는다)
_ARM_TITLE = {
    "holdout": "대조군(holdout, 편향 없음)",
    "shown": "보인 집단(shown, 자동화 편향 감시)",
}


def _pct(value: float) -> str:
    """비율 → `12.3%` (NaN이면 `-`)."""
    return "-" if math.isnan(value) else f"{value:.1%}"


def _rate(r: Rate, *, upper: bool = False) -> str:
    """비율 → `12.3% (4/32)` (상한을 붙이면 `, 상한 25.0%`)."""
    text = f"{_pct(r.value)} ({r.num}/{r.den})"
    return text + (f", 95% 상한 {_pct(r.upper95)}" if upper and r.den else "")


def _num(value: float, fmt: str = "{:.3f}") -> str:
    """수 → 글 (NaN이면 `-`)."""
    return "-" if math.isnan(value) else fmt.format(value)


def _arm_line(name: str, s: ArmStats) -> str:
    """집단 하나의 판정 지표 한 줄."""
    return (
        f"  [{_ARM_TITLE.get(name, name)}] 행 {s.rows}: 지적 정밀도 {_rate(s.flag_precision)} · "
        f"지적 재현율 {_rate(s.flag_recall)} · ok 놓침률 {_rate(s.ok_miss, upper=True)} · "
        f"틀림인데 승인 {_rate(s.wrong_approved)} · 제안 정확 일치 {_rate(s.suggestion_exact)}"
    )


def _by(values: Mapping[str, float], fmt: str) -> str:
    """집단 → 값 사전을 `shown 0.81 · holdout 0.84`로."""
    return " · ".join(f"{k} {fmt.format(v)}" for k, v in sorted(values.items())) or "-"


def print_ai_quality(q: AiQuality, max_defect_ratio: float) -> None:
    """AI 품질 보고 출력. max_defect_ratio: 비교 기준 `review.yaml sampling.max_defect_ratio`."""
    a, e = q.agreement, q.exposure
    print(f"AI 1차 검수 품질 ({q.since.isoformat()} ~ {q.until.isoformat()})")
    tasks = " · ".join(f"{k} {v}" for k, v in sorted(e.tasks.items())) or "없음"
    print(f"  작업(노출 집단): {tasks} · 판정률 평균 {_pct(e.coverage)}")
    for name in ("holdout", "shown"):
        if name in a.arms:
            print(_arm_line(name, a.arms[name]))
    strata = " · ".join(
        f"{k} {_rate(v.rate, upper=True)} (모집단 {v.population:g})"
        for k, v in sorted(a.ok_miss_strata.items())
    )  # fmt: skip
    print(
        f"  ok 놓침률(대조군, 층화): {strata or '-'} → 합 {_pct(a.ok_miss_rate)} (보수적 95% 상한 "
        f"{_pct(a.ok_miss_upper95)}, 기준 sampling.max_defect_ratio {max_defect_ratio:.1%})"
    )
    print(f"  확신도 ECE(대조군, 참고) {_num(a.ece)} · Cohen κ(지적 여부) {_num(a.kappa)}")
    if a.apply is not None:
        p = a.apply
        print(
            f"  제안 적용(보인 집단): 보인 제안 {p.shown_suggestions} · 적용 고름 "
            f"{_rate(p.apply_rate)} · 적용됨 {_rate(p.accept_rate)} · 검수자 결과 우선 "
            f"{_rate(p.conflict_edited)} · 지금 검증 실패 {_rate(p.invalid_now)}"
        )
    for kind, s in sorted(a.by_kind.items()):
        print(
            f"  종류 {kind}(대조군): 지적 정밀도 {_rate(s.flag_precision)} · 재현율 "
            f"{_rate(s.flag_recall)} · ok 놓침률 {_rate(s.ok_miss)}"
        )
    if "holdout" in a.arms:
        print(f"  혼동 행렬(대조군, AI → 사람): {a.arms['holdout'].confusion}")
    print(f"  검수 분/영상 1시간: {_by(q.review_minutes, '{:.1f}')}")
    print(
        f"  이중 일치 κ(짝 집단별): {_by(q.double_kappa_by_arm, '{:.3f}')} · 블라인드 프리라벨 "
        f"편향(짝 집단별): {_by(q.blind_bias_by_arm, '{:+.3f}')}"
    )
    print(
        f"  오류 삽입 누출: AI 없음 화면 중 오류 삽입 {_rate(e.seeded_share_no_ai)} · 블라인드 "
        f"아닌 전체 {_rate(e.seeded_share_all)}"
    )
    if a.excluded:
        print(f"  분모에서 뺀 행: {', '.join(f'{k} {v}' for k, v in sorted(a.excluded.items()))}")


def print_thresholds(q: AiQuality, target: float) -> None:
    """`review.yaml ai_review.show_min_confidence` 조안 (대조군 기준, `confidence_floors`)."""
    floors = confidence_floors(q.rows, target)
    print(f"show_min_confidence 조안 (목표 정밀도 {target:.0%}, 대조군·자기 보고 확신도):")
    if not floors:
        print("  확신도가 있는 대조군 판정이 없다 (조안 없음)")
    for verdict, (floor, n, precision) in sorted(floors.items()):
        shown = f"{floor:.2f}" if floor is not None else "목표에 닿지 않음 (지금 값 유지)"
        print(f"  {verdict}: {shown} (표본 {n}, 정밀도 {precision:.1%})")


def _rate_json(r: Rate) -> dict[str, Any]:
    """비율 → JSON (NaN은 null)."""

    def clean(v: float) -> float | None:
        return None if math.isnan(v) else v

    return {"num": r.num, "den": r.den, "value": clean(r.value), "upper95": clean(r.upper95)}


def _arm_json(s: ArmStats) -> dict[str, Any]:
    """집단 지표 → JSON."""
    return {
        "rows": s.rows, "confusion": s.confusion,
        "flag_precision": _rate_json(s.flag_precision), "flag_recall": _rate_json(s.flag_recall),
        "ok_miss": _rate_json(s.ok_miss), "wrong_approved": _rate_json(s.wrong_approved),
        "suggestion_exact": _rate_json(s.suggestion_exact),
    }  # fmt: skip


def ai_quality_json(q: AiQuality) -> dict[str, Any]:
    """AI 품질 보고 → JSON 사전 (NaN은 null)."""
    a, e = q.agreement, q.exposure

    def clean(v: float) -> float | None:
        return None if math.isnan(v) else v

    apply = a.apply
    return {
        "since": q.since.isoformat(), "until": q.until.isoformat(),
        "arms": {k: _arm_json(v) for k, v in a.arms.items()},
        "by_kind": {k: _arm_json(v) for k, v in a.by_kind.items()},
        "ok_miss_strata": {
            k: {**_rate_json(v.rate), "population": v.population}
            for k, v in a.ok_miss_strata.items()
        },
        "ok_miss_rate": clean(a.ok_miss_rate), "ok_miss_upper95": clean(a.ok_miss_upper95),
        "ece": clean(a.ece), "kappa": clean(a.kappa),
        "apply": None if apply is None else {
            "shown_suggestions": apply.shown_suggestions,
            "apply_rate": _rate_json(apply.apply_rate),
            "accept_rate": _rate_json(apply.accept_rate),
            "conflict_edited": _rate_json(apply.conflict_edited),
            "invalid_now": _rate_json(apply.invalid_now),
        },
        "excluded": a.excluded,
        "exposure": {
            "tasks": e.tasks, "coverage": clean(e.coverage),
            "seeded_share_no_ai": _rate_json(e.seeded_share_no_ai),
            "seeded_share_all": _rate_json(e.seeded_share_all),
        },
        "review_minutes_per_video_hour": q.review_minutes,
        "double_kappa_by_arm": q.double_kappa_by_arm, "blind_bias_by_arm": q.blind_bias_by_arm,
    }  # fmt: skip
