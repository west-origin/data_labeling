from __future__ import annotations

from pathlib import Path

import pytest

from dlp_eval.gate import decide
from dlp_eval.harness import EvalReport, SessionData, evaluate
from dlp_eval.policy import EvaluationPolicy, Task, load_policy
from dlp_fixtures.actions import generate_action_scenario
from dlp_fixtures.wiping import generate_wiping_scenario
from dlp_schema.labels import (
    ActionPayload,
    BoxKeyframe,
    BoxTrackPayload,
    CoveragePayload,
    LabelRecord,
    ObjectStatePayload,
    Provenance,
    Source,
)
from dlp_schema.testing import make_label

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def policy() -> EvaluationPolicy:
    return load_policy(ROOT)


def as_model(labels: list[LabelRecord], version: str = "m1") -> list[LabelRecord]:
    return [
        x.model_copy(
            update={
                "provenance": Provenance(source=Source.MODEL, model_version=version),
                "confidence": 0.9,
            }
        )
        for x in labels
    ]


def boxes(session: str, shift: float = 0.0) -> list[LabelRecord]:
    """정답을 아는 박스 트랙 두 개 (컵, 걸레)."""
    out: list[LabelRecord] = []
    for i, (cls, x0) in enumerate((("cup", 10.0), ("rag", 200.0))):
        frames = tuple(
            BoxKeyframe(t_ms=t, x=x0 + t / 100 + shift, y=50, w=40, h=40)
            for t in range(0, 1000, 100)
        )
        payload = BoxTrackPayload(entity_id=f"{cls}_01", class_id=cls, keyframes=frames)
        out.append(
            make_label(
                payload,
                label_id=f"{session}-box{i}",
                session_id=session,
                t_end_ms=900,
                stream_id="bodycam",
            )
        )
    return out


def states(session: str, switch_ms: int) -> list[LabelRecord]:
    return [
        make_label(
            ObjectStatePayload(
                entity_id="sink_01", class_id="sink", attribute="cleanliness", value=v
            ),
            label_id=f"{session}-st{i}", session_id=session, t_start_ms=s, t_end_ms=e,
        )
        for i, (s, e, v) in enumerate(((0, switch_ms, "dirty"), (switch_ms, 5000, "clean")))
    ]  # fmt: skip


def scenario_data(seed: int, glove: bool) -> SessionData:
    actions = generate_action_scenario(seed, session_id=f"s{seed}")
    wiping = generate_wiping_scenario(seed, session_id=f"s{seed}")
    coverage = make_label(
        CoveragePayload(surface_id="table_01", tool_id="rag_01", ratio=round(wiping.coverage, 4)),
        label_id=f"s{seed}-cov", session_id=f"s{seed}",
    )  # fmt: skip
    truth = [
        *actions.labels, *wiping.truth_relations, coverage, *boxes(f"s{seed}"),
        *states(f"s{seed}", 2000),
    ]  # fmt: skip
    return SessionData(f"s{seed}", truth, as_model(truth), {"glove": "glove" if glove else "bare"})


def run(data: list[SessionData], policy: EvaluationPolicy, version: str = "m1") -> EvalReport:
    tasks: list[Task] = [
        "objects",
        "hands",
        "contact",
        "actions",
        "relations",
        "states",
        "coverage",
    ]
    return evaluate(
        dict.fromkeys(tasks, data), policy, golden_version="g1",
        model_versions=dict.fromkeys(tasks, version),
    )  # fmt: skip


def test_perfect_predictions_score_perfectly(policy: EvaluationPolicy) -> None:
    report = run([scenario_data(0, glove=True), scenario_data(1, glove=False)], policy)
    m = {task: r.metrics for task, r in report.overall.items()}
    assert m["objects"]["map"] == pytest.approx(1.0) and m["objects"]["hota"] == pytest.approx(1.0)
    assert m["objects"]["idf1"] == 1.0 and m["objects"]["mota"] == 1.0
    assert m["hands"]["pck"] == 1.0
    assert m["contact"]["contact_start_f1"] == 1.0 and m["contact"]["contact_start_error_ms"] == 0
    assert m["contact"]["grasp_macro_f1"] == 1.0
    assert m["actions"]["segment_f1_0.5"] == 1.0 and m["actions"]["temporal_map"] == pytest.approx(
        1.0
    )
    assert m["actions"]["boundary_f1"] == 1.0 and m["actions"]["verb_macro_f1"] == 1.0
    assert m["relations"]["relation_f1"] == 1.0
    assert m["states"]["transition_accuracy"] == 1.0
    assert m["coverage"]["coverage_abs_error"] == 0.0
    # 하위 집단 리포트: 장갑 세션과 맨손 세션
    assert set(report.subgroups) == {"glove=glove", "glove=bare"}
    assert report.subgroups["glove=bare"]["actions"].sessions == 1


def test_degraded_predictions_have_known_scores(policy: EvaluationPolicy) -> None:
    s = scenario_data(0, glove=False)
    actions = [x for x in s.pred if isinstance(x.payload, ActionPayload)]
    dropped = {x.label_id for x in actions[::2]}  # 행동 절반을 빼먹음
    s.pred = [
        x
        for x in s.pred
        if x.label_id not in dropped and x.kind not in ("box_track", "object_state", "coverage")
    ]
    s.pred += as_model(boxes("s0", shift=20.0))  # 박스를 폭의 절반만큼 밀어 IoU = 20/60
    s.pred += as_model(states("s0", 2300))  # 전이 300 ms 늦음 (허용 500 안)
    cov = next(x for x in s.truth if isinstance(x.payload, CoveragePayload))
    assert isinstance(cov.payload, CoveragePayload)
    s.pred.append(
        as_model(
            [
                cov.model_copy(
                    update={
                        "payload": cov.payload.model_copy(update={"ratio": cov.payload.ratio - 0.1})
                    }
                )
            ]
        )[0]
    )
    report = run([s], policy)
    n = len(actions)
    kept = n - len(dropped)
    assert report.overall["actions"].metrics["segment_f1_0.5"] == pytest.approx(
        2 * kept / (n + kept)
    )
    assert report.overall["objects"].metrics["ap50"] == 0.0  # IoU 1/3 < 0.5
    assert report.overall["states"].metrics["transition_accuracy"] == 1.0
    assert report.overall["coverage"].metrics["coverage_abs_error"] == pytest.approx(0.1, abs=1e-4)


def test_under_sampled_classes_are_flagged(policy: EvaluationPolicy) -> None:
    report = run([scenario_data(0, glove=False)], policy)
    # 박스 클래스마다 10개 (< 20)
    assert report.overall["objects"].under_sampled == ["cup", "rag"]
    assert report.overall["objects"].class_counts == {"cup": 10, "rag": 10}


def test_gate_passes_improvement_and_blocks_subgroup_regression(policy: EvaluationPolicy) -> None:
    good = [scenario_data(0, glove=True), scenario_data(1, glove=False)]
    baseline = run(good, policy, "old")

    def worse_actions(data: list[SessionData], session: str) -> list[SessionData]:
        out: list[SessionData] = []
        for s in data:
            pred = s.pred
            if s.session_id == session:
                first = next(x for x in pred if isinstance(x.payload, ActionPayload))
                pred = [x for x in pred if x.label_id != first.label_id]
            out.append(SessionData(s.session_id, s.truth, pred, s.groups))
        return out

    # 같은 성능이면 통과 (min_gain 0)
    same = decide(run(good, policy, "new"), baseline, policy)
    assert same.passed, same
    # 맨손 세션에서만 행동 하나를 놓친 후보: 전체 하락은 작아도 하위 집단에서 막힌다
    regressed = decide(run(worse_actions(good, "s1"), policy, "new"), baseline, policy)
    assert not regressed.passed and not regressed.tasks["actions"].passed
    assert any("glove=bare" in r for r in regressed.tasks["actions"].reasons)
    assert regressed.tasks["objects"].passed
    # 첫 배포: 주 지표가 기준을 넘으면 통과, 표본 부족은 경고만
    first = decide(baseline, None, policy)
    assert first.passed
    # 세션 두 개면 박스는 클래스당 20개(기준 충족), 파지 유형은 기준에 못 미친다
    assert not any("표본 부족" in w for w in first.tasks["objects"].warnings)
    assert any("표본 부족" in w for w in first.tasks["contact"].warnings)


def test_gate_direction_for_error_metrics(policy: EvaluationPolicy) -> None:
    base = run([scenario_data(0, glove=False)], policy, "old")
    s = scenario_data(0, glove=False)
    cov = next(x for x in s.pred if isinstance(x.payload, CoveragePayload))
    assert isinstance(cov.payload, CoveragePayload)
    s.pred = [x for x in s.pred if x is not cov] + [
        cov.model_copy(
            update={"payload": cov.payload.model_copy(update={"ratio": cov.payload.ratio + 0.05})}
        )
    ]
    decision = decide(run([s], policy, "new"), base, policy)
    assert not decision.tasks["coverage"].passed  # 오차가 0 → 0.05로 커졌다


def test_gate_uses_per_metric_drop_for_ms_errors(policy: EvaluationPolicy) -> None:
    from dlp_eval.harness import TaskReport

    def report(error_ms: float) -> EvalReport:
        metrics = {"contact_start_f1": 0.9, "contact_end_f1": 0.9, "grasp_macro_f1": 0.8,
                   "contact_start_error_ms": error_ms}  # fmt: skip
        return EvalReport("g", {"contact": "m"}, {"contact": TaskReport(metrics, {}, [], 1)}, {})

    base = report(40.0)
    assert decide(report(55.0), base, policy).passed  # 15 ms 늘어남 (허용 20 ms)
    worse = decide(report(65.0), base, policy)
    assert not worse.passed and "contact_start_error_ms" in worse.tasks["contact"].reasons[0]


def test_ece_without_predictions_is_not_perfect() -> None:
    import math

    from dlp_eval.metrics.classification import ece

    assert math.isnan(ece([], []))
