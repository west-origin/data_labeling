"""배포 게이트: 후보 모델의 골든셋 리포트를 기존 모델과 비교해 배포 여부를 정한다.

과제마다 (config/policies/evaluation.yaml gate)
- 주 지표가 전체에서 기존보다 min_gain 이상 좋아야 한다.
- 지키는 지표(guards)는 전체와 각 하위 집단에서, 주 지표는 각 하위 집단에서 허용 하락(지표별
  max_drop_by, 없으면 max_drop)보다 많이 나빠지면 안 된다.
- 기존 모델이 없으면 주 지표가 first_deploy 기준을 넘어야 한다.
- 정답 표본이 부족한 클래스는 막지 않고 경고로만 남긴다.
지표 방향: 오차·ECE는 낮을수록 좋다 (harness.LOWER_IS_BETTER).
값이 NaN(정의되지 않음)이면: 후보만 NaN이면 실패, 기존만 NaN이면 기존을 가장 나쁜 값으로 보아
그 비교는 통과(경고), 둘 다 NaN이면 건너뛴다.

WP11, ADR 0013·0016. 사용처: `dlp eval golden --baseline`(`dlp_cli.eval_cmds`), 재학습 루프
(`dlp_train.loop.run_training_job`). 순수 함수이며 부작용이 없다 (판정 결과는 호출자가 리포트에
남긴다).

공개 이름: `TaskDecision`, `GateDecision`, `decide`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from dlp_eval.harness import LOWER_IS_BETTER, EvalReport, TaskReport
from dlp_eval.policy import EvaluationPolicy, GateRule


@dataclass
class TaskDecision:
    """과제 하나의 게이트 판정."""

    task: str
    passed: bool  # reasons가 비어 있으면 True
    reasons: list[str] = field(default_factory=list[str])  # 실패 이유 (사람이 읽는 문장)
    # 막지 않는 경고 (표본 부족, NaN 기존 값, 규칙 없음 등)
    warnings: list[str] = field(default_factory=list[str])


@dataclass
class GateDecision:
    """전체 판정. 모든 과제가 통과해야 통과한다."""

    passed: bool
    tasks: dict[str, TaskDecision]


def _gain(metric: str, candidate: float, baseline: float) -> float:
    """좋아진 양 (나빠지면 음수).

    높을수록 좋은 지표는 후보 - 기존, `LOWER_IS_BETTER` 지표는 기존 - 후보. 단위는 그 지표 단위다.
    """
    return baseline - candidate if metric in LOWER_IS_BETTER else candidate - baseline


def _compare(
    rule: GateRule,
    cand: TaskReport,
    base: TaskReport,
    where: str,
    decision: TaskDecision,
    primary: bool,
) -> None:
    """후보·기존 리포트 하나(전체 또는 하위 집단)를 규칙으로 비교해 `decision`에 이유·경고를 더한다.

    Args:
        rule: 과제 게이트 규칙.
        cand, base: 같은 범위(전체 또는 같은 하위 집단)의 후보·기존 리포트.
        where: 이유 문장에 붙일 범위 이름 ("전체" 또는 "glove=bare" 등).
        decision: 결과를 덧붙일 판정 (제자리 수정).
        primary: True(전체 비교)면 주 지표에 `min_gain`을 적용한다. False(하위 집단)면 주 지표도
            지키는 지표처럼 허용 하락만 본다.

    리포트에 지표 키가 없으면 NaN으로 본다 (모듈 docstring의 NaN 규칙).
    """
    metrics = [rule.primary, *rule.guards]
    for m in metrics:
        c, b = cand.metrics.get(m, math.nan), base.metrics.get(m, math.nan)
        if math.isnan(c) and math.isnan(b):
            continue
        if math.isnan(c):
            decision.reasons.append(f"{where} {m}: 비교할 수 없음 (후보 {c}, 기존 {b})")
            continue
        if math.isnan(b):
            # 기존 모델은 값이 정의되지 않음 (예: 맞춘 접촉이 없어 오차가 없음) → 가장 나쁜 값으로
            # 보고 후보가 이 비교를 통과한다. 후보만 NaN이면 위에서 실패한다.
            decision.warnings.append(f"{where} {m}: 기존 값이 없어 후보({c:.4f})를 통과로 봄")
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
    """후보 리포트를 기존 리포트(없으면 첫 배포 기준)와 비교해 배포 여부를 정한다.

    Args:
        candidate: 후보 모델의 골든셋 리포트. 판정 대상 과제는 `candidate.overall`에 있는 과제다.
        baseline: 기존(배포 중이거나 대신할 기본 어댑터) 모델의 리포트. None이면 모든 과제가 첫
            배포다. 기존 리포트에 그 과제가 없어도 그 과제는 첫 배포 기준으로 본다.
        policy: 평가 정책 (`gate` 절).

    Returns:
        `GateDecision`. 규칙이 없는 과제는 통과(경고)로 둔다. 하위 집단 비교는 후보·기존 리포트
        양쪽에 같은 하위 집단·과제가 있을 때만 한다.

    첫 배포에서 주 지표가 NaN이면 비교가 거짓이라 실패한다.
    """
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
            # 첫 배포: 절대 기준만 본다 (하위 집단은 보지 않는다)
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
