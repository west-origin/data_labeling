"""배포 게이트: 후보 모델의 골든셋 리포트를 기존 모델과 비교해 배포 여부를 정한다.

과제마다 (config/policies/evaluation.yaml gate)
- 주 지표가 기존보다 min_gain 이상 좋아야 한다.
- 지키는 지표와 주 지표가 전체와 각 하위 집단에서 max_drop보다 많이 나빠지면 안 된다.
- 기존 모델이 없으면 주 지표가 first_deploy 기준을 넘어야 한다.
- 정답 표본이 부족한 클래스는 막지 않고 경고로만 남긴다.
지표 방향: 오차·ECE는 낮을수록 좋다 (harness.LOWER_IS_BETTER).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from dlp_eval.harness import LOWER_IS_BETTER, EvalReport, TaskReport
from dlp_eval.policy import EvaluationPolicy, GateRule


@dataclass
class TaskDecision:
    task: str
    passed: bool
    reasons: list[str] = field(default_factory=list[str])
    warnings: list[str] = field(default_factory=list[str])


@dataclass
class GateDecision:
    passed: bool
    tasks: dict[str, TaskDecision]


def _gain(metric: str, candidate: float, baseline: float) -> float:
    """좋아진 양 (나빠지면 음수)."""
    return baseline - candidate if metric in LOWER_IS_BETTER else candidate - baseline


def _compare(
    rule: GateRule,
    cand: TaskReport,
    base: TaskReport,
    where: str,
    decision: TaskDecision,
    primary: bool,
) -> None:
    metrics = [rule.primary, *rule.guards]
    for m in metrics:
        c, b = cand.metrics.get(m, math.nan), base.metrics.get(m, math.nan)
        if math.isnan(c) and math.isnan(b):
            continue
        if math.isnan(c) or math.isnan(b):
            decision.reasons.append(f"{where} {m}: 비교할 수 없음 (후보 {c}, 기존 {b})")
            continue
        g = _gain(m, c, b)
        if primary and m == rule.primary and g < rule.min_gain:
            decision.reasons.append(
                f"{where} {m}: {b:.4f} → {c:.4f} (향상 {g:+.4f} < {rule.min_gain})"
            )
        elif g < -rule.allowed_drop(m):
            decision.reasons.append(
                f"{where} {m}: {b:.4f} → {c:.4f} (하락 {g:+.4f}, 허용 {rule.allowed_drop(m)})"
            )


def decide(
    candidate: EvalReport, baseline: EvalReport | None, policy: EvaluationPolicy
) -> GateDecision:
    tasks: dict[str, TaskDecision] = {}
    for task, cand in candidate.overall.items():
        rule = policy.gate.get(task)  # type: ignore[call-overload]
        decision = TaskDecision(task, True)
        tasks[task] = decision
        if rule is None:
            decision.warnings.append("게이트 규칙이 없어 판정하지 않음")
            continue
        if cand.under_sampled:
            decision.warnings.append(f"정답 표본 부족 클래스: {', '.join(cand.under_sampled)}")
        base = baseline.overall.get(task) if baseline else None
        if base is None:
            value = cand.metrics.get(rule.primary, math.nan)
            ok = (
                value <= rule.first_deploy
                if rule.primary in LOWER_IS_BETTER
                else value >= rule.first_deploy
            )
            if not ok:
                decision.reasons.append(
                    f"첫 배포 기준 미달: {rule.primary} {value:.4f} (기준 {rule.first_deploy})"
                )
        else:
            _compare(rule, cand, base, "전체", decision, primary=True)
            assert baseline is not None
            for group, reports in candidate.subgroups.items():
                c_sub, b_sub = reports.get(task), baseline.subgroups.get(group, {}).get(task)
                if c_sub is not None and b_sub is not None:
                    _compare(rule, c_sub, b_sub, group, decision, primary=False)
        decision.passed = not decision.reasons
    return GateDecision(all(d.passed for d in tasks.values()), tasks)
