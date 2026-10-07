"""평가 하네스(`dlp_eval.harness`)와 배포 게이트(`dlp_eval.gate`)의 단위 테스트 (WP11, ADR
0013·0025).

DB 없이 `SessionData`를 직접 만들어 넣는다. 정답은 합성 픽스처(`generate_action_scenario`:
행동·접촉· 손 키포인트, `generate_wiping_scenario`: 닦기 관계·커버리지)와 이 파일의 작은
박스·상태·블러 생성기다. 예측은 정답을 모델 출처로 복사(완벽)하거나 알려진 만큼 망가뜨린 것이라
기대 지표를 손으로 계산할 수 있다. "감사 회귀" 절은 감사(ADR 0015·0025)에서 고친 버그가 다시 생기지
않는지 본다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from dlp_eval.gate import decide
from dlp_eval.harness import EvalReport, SessionData, evaluate
from dlp_eval.policy import EvaluationPolicy, Task, is_class_metric, load_policy, task_metrics
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
    """저장소의 실제 평가 정책 (config/policies/evaluation.yaml)."""
    return load_policy(ROOT)


def as_model(labels: list[LabelRecord], version: str = "m1") -> list[LabelRecord]:
    """라벨을 모델 `version`의 예측으로 바꾼다 (같은 ID·내용, 신뢰도 0.9)."""
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
    """싱크 청결 상태 dirty → clean 구간 두 개 (전이 시각 = switch_ms)."""
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
    """여러 과제의 정답이 든 세션 하나. 예측은 정답 그대로(완벽). 장소는 seed 홀짝으로 나뉜다."""
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
    groups = {"glove": "glove" if glove else "bare", "site": f"site{seed % 2}"}
    return SessionData(f"s{seed}", truth, as_model(truth), groups)


def run(data: list[SessionData], policy: EvaluationPolicy, version: str = "m1") -> EvalReport:
    """같은 세션들로 여러 과제를 한꺼번에 평가한다 (body·privacy 제외)."""
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
    """예측이 정답과 같으면 모든 지표가 만점(오차 0)이고 하위 집단 리포트가 축·값별로 나온다."""
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
    # 하위 집단 리포트: 장갑 세션과 맨손 세션, 장소별
    assert set(report.subgroups) == {"glove=glove", "glove=bare", "site=site0", "site=site1"}
    assert report.subgroups["site=site1"]["objects"].sessions == 1
    assert report.subgroups["glove=bare"]["actions"].sessions == 1


def test_degraded_predictions_have_known_scores(policy: EvaluationPolicy) -> None:
    """알려진 만큼 망가뜨린 예측의 지표가 손 계산 값과 같다.

    행동 절반 누락 → 구간 F1 = 2·남은 수/(전체+남은 수), 박스 20 px 이동 → IoU 1/3이라 AP50 0,
    전이 300 ms 지연(허용 500 안) → 정확도 1, 커버리지 비율 -0.1 → 절대 오차 0.1.
    """
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
    """정답 표본이 min_samples_per_class(20)보다 적은 클래스가 표본 부족으로 표시된다."""
    report = run([scenario_data(0, glove=False)], policy)
    # 박스 클래스마다 10개 (< 20)
    assert report.overall["objects"].under_sampled == ["cup", "rag"]
    assert report.overall["objects"].class_counts == {"cup": 10, "rag": 10}


def test_gate_passes_improvement_and_blocks_subgroup_regression(policy: EvaluationPolicy) -> None:
    """게이트: 같은 성능은 통과, 한 하위 집단만 나빠져도 실패, 첫 배포는 기준만 보고 표본 부족은
    경고."""
    good = [scenario_data(0, glove=True), scenario_data(1, glove=False)]
    baseline = run(good, policy, "old")

    def worse_actions(data: list[SessionData], session: str) -> list[SessionData]:
        """`session`의 예측에서 첫 행동 하나를 뺀 사본."""
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
    """낮을수록 좋은 지표(커버리지 오차)는 값이 커지면 하락으로 판정한다 (0 → 0.05, 허용 0.01)."""
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
    """ms 오차 지표는 max_drop_by(20 ms)로 판정한다: +15 ms 통과, +25 ms 실패."""
    from dlp_eval.harness import TaskReport

    def report(error_ms: float) -> EvalReport:
        """접촉 지표만 든 최소 리포트 (시작 오차만 바꾼다)."""
        metrics = {"contact_start_f1": 0.9, "contact_end_f1": 0.9, "grasp_macro_f1": 0.8,
                   "contact_start_error_ms": error_ms}  # fmt: skip
        return EvalReport("g", {"contact": "m"}, {"contact": TaskReport(metrics, {}, [], 1)}, {})

    base = report(40.0)
    assert decide(report(55.0), base, policy).passed  # 15 ms 늘어남 (허용 20 ms)
    worse = decide(report(65.0), base, policy)
    assert not worse.passed and "contact_start_error_ms" in worse.tasks["contact"].reasons[0]


def test_ece_without_predictions_is_not_perfect() -> None:
    """예측이 없으면 ECE는 0(만점)이 아니라 NaN이다."""
    import math

    from dlp_eval.metrics.classification import ece

    assert math.isnan(ece([], []))


def _blur(label_id: str, target: str, boxes: list[tuple[int, float, float]], size: float = 40):
    """바디캠 블러 트랙. boxes는 (시각 ms, x, y), 박스는 size x size 정사각형."""
    return make_label(
        {
            "kind": "blur_track",
            "target": target,
            "keyframes": [{"t_ms": t, "x": x, "y": y, "w": size, "h": size} for t, x, y in boxes],
        },
        label_id=label_id,
        stream_id="bodycam",
        t_start_ms=boxes[0][0],
        t_end_ms=boxes[-1][0],
    )


def test_privacy_recall_and_precision(policy: EvaluationPolicy) -> None:
    """블러 재현(대상별 포함)과 정밀을 손 계산 값과 대조한다 (근거는 본문 주석)."""
    truth = [
        _blur("face", "face", [(0, 10, 10), (100, 20, 10), (200, 30, 10)]),
        _blur("doc", "document", [(0, 200, 200), (100, 200, 200)]),
    ]
    # 얼굴은 크게 덮고(3/3), 문서는 첫 시각만 덮는다(1/2). 엉뚱한 곳 블러 하나(오탐).
    pred = as_model(
        [
            _blur("p-face", "face", [(0, 0, 0), (200, 20, 0)], size=60),
            _blur("p-doc", "document", [(0, 200, 200)]),
            _blur("p-fp", "face", [(100, 500, 500)]),
        ]
    )
    report = evaluate(
        {"privacy": [SessionData("s001", truth, pred)]},
        policy,
        golden_version="g",
        model_versions={"privacy": "m1"},
    )
    m = report.overall["privacy"].metrics
    assert m["blur_recall"] == pytest.approx(4 / 5)
    assert m["blur_recall/face"] == 1.0
    assert m["blur_recall/document"] == 0.5
    # 정답 시각별 예측 박스 5개: t0 얼굴·문서, t100 얼굴(보간)·오탐, t200 얼굴.
    # 얼굴 예측(60x60)은 정답(40x40)이 면적의 0.44만 덮어 정밀 기준 0.5 미만 → 맞음은 문서 1개
    # (블러를 넉넉히 키우는 것은 재현에는 좋지만 정밀에서 깎인다)
    assert m["blur_precision"] == pytest.approx(1 / 5)


# ---------------------------------------------------------------- 감사 회귀


def _track(
    label_id: str,
    entity: str,
    cls: str,
    frames: list[tuple[int, float]],
    stream: str = "bodycam",
    outside: frozenset[int] = frozenset(),
) -> LabelRecord:
    """x가 시각에 따라 움직이는 40x40 박스 트랙."""
    keyframes = tuple(
        BoxKeyframe(t_ms=t, x=x, y=50, w=40, h=40, outside=t in outside) for t, x in frames
    )
    return make_label(
        BoxTrackPayload(entity_id=entity, class_id=cls, keyframes=keyframes),
        label_id=label_id, stream_id=stream, t_start_ms=frames[0][0], t_end_ms=frames[-1][0],
    )  # fmt: skip


def _objects(truth: list[LabelRecord], pred: list[LabelRecord], policy: EvaluationPolicy):
    """세션 하나로 objects 과제만 평가해 지표 dict를 돌려준다."""
    report = evaluate(
        {"objects": [SessionData("s1", truth, pred)]}, policy, golden_version="g",
        model_versions={"objects": "m1"},
    )  # fmt: skip
    return report.overall["objects"].metrics


def test_sparse_truth_keyframes_are_interpolated(policy: EvaluationPolicy) -> None:
    """성긴 사람 정답 키프레임은 간격 제한 없이 보간하고, 화면 밖 키프레임 사이는 정답이 없다."""

    # 사람 정답은 키프레임이 성기고 트랙마다 시각이 다르다 (CVAT가 사이를 보간한다).
    # 컵은 0·400·800 ms, 걸레는 200·600·1000 ms에만 키프레임이 있다. 둘 다 등속 이동.
    def cup(t: int) -> float:
        """컵 x 좌표 (등속)."""
        return 10 + t / 10

    def rag(t: int) -> float:
        """걸레 x 좌표 (등속)."""
        return 300 + t / 20

    truth = [
        _track("t-cup", "cup_01", "cup", [(t, cup(t)) for t in (0, 400, 800)]),
        _track("t-rag", "rag_01", "rag", [(t, rag(t)) for t in (200, 600, 1000)]),
    ]
    # 예측은 100 ms마다 정확한 위치 (같은 직선) → 완벽해야 한다
    pred = as_model(
        [
            _track("p-cup", "c", "cup", [(t, cup(t)) for t in range(0, 801, 100)]),
            _track("p-rag", "r", "rag", [(t, rag(t)) for t in range(200, 1001, 100)]),
        ]
    )
    m = _objects(truth, pred, policy)
    assert m["hota"] == pytest.approx(1.0) and m["map"] == pytest.approx(1.0)
    assert m["idf1"] == 1.0 and m["mota"] == 1.0
    # 화면 밖 키프레임이 끼면 그 사이는 정답이 없다: 컵이 400 ms에 화면 밖이면 200·600 ms에는 컵이
    # 없으므로, 그 시각의 컵 예측은 오탐이다
    hidden = [_track("t-cup", "cup_01", "cup", [(t, cup(t)) for t in (0, 400, 800)],
                     outside=frozenset({400})), truth[1]]  # fmt: skip
    m2 = _objects(hidden, pred, policy)
    assert m2["mota"] < 1.0


def test_objects_and_hands_compare_within_the_same_stream(policy: EvaluationPolicy) -> None:
    """공간 라벨은 같은 스트림끼리만 비교한다 (스트림 PTS 시각, ADR 0019). 다른 스트림 예측은 못
    맞춘다."""
    # 같은 개체 ID가 두 스트림(바디캠·3인칭)에 다른 위치로 있다
    truth = [
        _track("t-a", "cup_01", "cup", [(0, 10.0), (100, 20.0)], stream="bodycam"),
        _track("t-b", "cup_01", "cup", [(0, 300.0), (100, 310.0)], stream="third"),
    ]
    pred = as_model(
        [
            _track("p-a", "cup_01", "cup", [(0, 10.0), (100, 20.0)], stream="bodycam"),
            _track("p-b", "cup_01", "cup", [(0, 300.0), (100, 310.0)], stream="third"),
        ]
    )
    m = _objects(truth, pred, policy)
    assert m["hota"] == pytest.approx(1.0) and m["idf1"] == 1.0 and m["map"] == pytest.approx(1.0)
    # 다른 스트림의 예측으로는 맞출 수 없다 (같은 좌표라도)
    swapped = as_model(
        [
            _track("p-a", "cup_01", "cup", [(0, 10.0), (100, 20.0)], stream="third"),
            _track("p-b", "cup_01", "cup", [(0, 300.0), (100, 310.0)], stream="bodycam"),
        ]
    )
    assert _objects(truth, swapped, policy)["ap50"] == 0.0

    def hand(label_id: str, stream: str, x: float) -> LabelRecord:
        """`stream`의 오른손 21관절 (시각 0 하나)."""
        points = [{"x": x + j, "y": 50 + j, "visibility": 2} for j in range(21)]
        return make_label(
            {"kind": "keypoint_track", "entity_id": "hand_r", "skeleton": "hand21",
             "hand": "right", "keyframes": [{"t_ms": 0, "points": points}]},
            label_id=label_id, stream_id=stream, t_start_ms=0, t_end_ms=0,
        )  # fmt: skip

    h_truth = [hand("h-a", "bodycam", 10), hand("h-b", "third", 400)]
    # 3인칭 예측이 먼저 나와도 스트림이 같은 것끼리 맞춘다
    h_pred = as_model([hand("q-b", "third", 400), hand("q-a", "bodycam", 10)])
    report = evaluate(
        {"hands": [SessionData("s1", h_truth, h_pred)]}, policy, golden_version="g",
        model_versions={"hands": "m1"},
    )  # fmt: skip
    assert report.overall["hands"].metrics["pck"] == 1.0


def test_privacy_sparse_truth_is_interpolated(policy: EvaluationPolicy) -> None:
    """블러도 성긴 정답 키프레임을 보간해 비교 시각을 늘리고, 정확한 예측은 재현·정밀 1이다."""
    # 정답 얼굴은 0·400 ms에만 키프레임, 문서는 200 ms에 키프레임. 예측은 100 ms마다 정확하다
    truth = [
        _blur("face", "face", [(0, 10, 10), (400, 50, 10)]),
        _blur("doc", "document", [(0, 200, 200), (200, 200, 200), (400, 200, 200)]),
    ]
    pred = as_model(
        [
            _blur("p-face", "face", [(t, 10 + t / 10, 10) for t in range(0, 401, 100)]),
            _blur("p-doc", "document", [(t, 200, 200) for t in range(0, 401, 100)]),
        ]
    )
    report = evaluate(
        {"privacy": [SessionData("s001", truth, pred)]}, policy, golden_version="g",
        model_versions={"privacy": "m1"},
    )  # fmt: skip
    m = report.overall["privacy"].metrics
    # 200 ms의 얼굴 정답(보간)도 있고 예측과 같은 자리라 재현·정밀 모두 1
    assert report.overall["privacy"].class_counts == {"face": 3, "document": 3}
    assert m["blur_recall"] == 1.0 and m["blur_precision"] == 1.0


def test_coverage_error_hand_computed(policy: EvaluationPolicy) -> None:
    """커버리지 절대 오차 평균: 예측이 없는 정답 쌍은 비율 0으로 본다 (손 계산 0.45)."""

    def cov(label_id: str, surface: str, ratio: float) -> LabelRecord:
        """걸레로 닦은 표면 커버리지 라벨."""
        return make_label(
            CoveragePayload(surface_id=surface, tool_id="rag_01", ratio=ratio), label_id=label_id
        )

    truth = [cov("t1", "table_01", 0.5), cov("t2", "sink_01", 0.8)]
    pred = as_model([cov("p1", "table_01", 0.4)])  # 싱크 예측 없음 → 0으로 본다
    report = evaluate(
        {"coverage": [SessionData("s001", truth, pred)]}, policy, golden_version="g",
        model_versions={"coverage": "m1"},
    )  # fmt: skip
    # |0.4 - 0.5| = 0.1, |0 - 0.8| = 0.8 → 평균 0.45
    assert report.overall["coverage"].metrics["coverage_abs_error"] == pytest.approx(0.45)


def test_gate_with_nan_baseline_metric(policy: EvaluationPolicy) -> None:
    """NaN 규칙: 기존만 NaN이면 통과(경고), 후보만 NaN이면 실패, 둘 다 NaN이면 건너뛴다."""
    from dlp_eval.harness import TaskReport

    def report(error_ms: float) -> EvalReport:
        """접촉 지표만 든 최소 리포트 (시작 오차만 바꾼다)."""
        metrics = {"contact_start_f1": 0.9, "contact_end_f1": 0.9, "grasp_macro_f1": 0.8,
                   "contact_start_error_ms": error_ms}  # fmt: skip
        return EvalReport("g", {"contact": "m"}, {"contact": TaskReport(metrics, {}, [], 1)}, {})

    nan = float("nan")
    # 기존 모델이 맞춘 접촉이 없어 오차가 NaN: 기존을 가장 나쁜 값으로 보고 통과 (경고)
    d = decide(report(30.0), report(nan), policy)
    assert d.passed and any("기존 값이 없어" in w for w in d.tasks["contact"].warnings)
    # 후보가 NaN이면 실패
    assert not decide(report(nan), report(30.0), policy).passed
    # 둘 다 NaN이면 그 지표는 건너뛴다
    assert decide(report(nan), report(nan), policy).passed


# ---------------------------------------------------------------- 감사 회귀 (4차)


def test_sparse_stride_predictions_are_interpolated_per_task(policy: EvaluationPolicy) -> None:
    """성긴 예측(500 ms 간격)은 과제별 max_interp_ms(objects 1200) 안에서 보간한다 (ADR 0025)."""
    # OWLv2 도구 예측은 500 ms마다만 추론하고 트래커가 max_gap_ms(1200) 안의 끊김을 한 트랙으로
    # 잇는다. 정답은 30 fps로 촘촘하다. 같은 박스면 완벽해야 한다 (기본 200 ms 보간이면 0에 가깝다)
    truth = [_track("t-mop", "mop_01", "mop", [(t, 100.0) for t in range(0, 2001, 33)])]
    pred = as_model([_track("p-mop", "m", "mop", [(t, 100.0) for t in range(0, 2001, 500)])])
    m = _objects(truth, pred, policy)
    assert m["hota"] == pytest.approx(1.0) and m["map"] == pytest.approx(1.0)
    # 과제별 보간 간격보다 길게 끊긴 예측은 그 사이를 예측 없음으로 센다
    gaps = policy.max_interp_ms.model_copy(update={"tasks": {"objects": 200}})
    short = policy.model_copy(update={"max_interp_ms": gaps})
    assert _objects(truth, pred, short)["ap50"] < 0.1


def test_interp_gaps_cover_prelabel_tracker_gaps(policy: EvaluationPolicy) -> None:
    """정책 일관성: 평가 보간 간격 >= prelabel.yaml 트래커 max_gap_ms (objects·OWLv2·body)."""
    import yaml

    prelabel = yaml.safe_load((ROOT / "config/policies/prelabel.yaml").read_text("utf-8"))
    gaps = policy.max_interp_ms
    assert gaps.for_task("objects") >= prelabel["open_vocab_objects"]["max_gap_ms"]
    assert gaps.for_task("objects") >= prelabel["objects"]["max_gap_ms"]
    assert gaps.for_task("body") >= prelabel["body"]["max_gap_ms"]


def _person(label_id: str, x0: float, *, hidden: int = 0, stream: str = "third") -> LabelRecord:
    """17관절 사람. 앞쪽 hidden개 관절은 표시하지 않음(visibility 0, 좌표 (0, 0))."""
    points = [
        {"x": 0, "y": 0, "visibility": 0}
        if k < hidden
        else {"x": x0 + (k % 4) * 10, "y": 100 + (k // 4) * 20, "visibility": 2}
        for k in range(17)
    ]
    return make_label(
        {"kind": "keypoint_track", "entity_id": label_id, "skeleton": "coco17",
         "keyframes": [{"t_ms": 0, "points": points}]},
        label_id=label_id, stream_id=stream, t_start_ms=0, t_end_ms=0,
    )  # fmt: skip


def _pck(
    task: Task, truth: list[LabelRecord], pred: list[LabelRecord], policy: EvaluationPolicy
) -> float:
    """세션 하나로 hands 또는 body 과제를 평가해 PCK를 돌려준다."""
    report = evaluate(
        {task: [SessionData("s1", truth, pred)]}, policy, golden_version="g",
        model_versions={task: "m1"},
    )  # fmt: skip
    return report.overall[task].metrics["pck"]


def test_body_pck_matches_people_by_visible_joints(policy: EvaluationPolicy) -> None:
    """전신은 보이는 관절 박스 IoU로 사람을 일대일로 맞춘다 (표시 안 한 (0, 0) 관절 무시, ADR
    0025)."""
    # 정답 B는 관절 두 개를 표시하지 않았다 ((0, 0)). 그 점을 박스에 넣으면 B 박스가 A를 덮어 정답
    # 순서대로 탐욕 매칭할 때 B가 A 예측을 가져간다. 보이는 관절만으로 맞추면 완벽하다
    truth = [_person("tB", 300, hidden=2), _person("tA", 100)]
    pred = as_model([_person("pA", 100), _person("pB", 300)])
    assert _pck("body", truth, pred, policy) == 1.0
    # 예측이 하나뿐이면 겹치는 정답(A)과만 맞춘다 (정답 순서와 상관없이).
    # B의 보이는 관절 15개는 틀림
    assert _pck("body", truth, as_model([_person("pA", 100)]), policy) == pytest.approx(
        17 / (17 + 15)
    )
    # 다른 스트림 예측과는 맞추지 않는다
    other = as_model([_person("pA", 100, stream="bodycam"), _person("pB", 300, stream="bodycam")])
    assert _pck("body", truth, other, policy) == 0.0


def _hand(label_id: str, x0: float, side: str = "left") -> LabelRecord:
    """바디캠 손 21관절 (시각 0 하나, 모두 보임). x0만큼 옆으로 옮긴다."""
    points = [{"x": x0 + k, "y": 50 + k, "visibility": 2} for k in range(21)]
    return make_label(
        {"kind": "keypoint_track", "entity_id": label_id, "skeleton": "hand21", "hand": side,
         "keyframes": [{"t_ms": 0, "points": points}]},
        label_id=label_id, stream_id="bodycam", t_start_ms=0, t_end_ms=0,
    )  # fmt: skip


def test_hands_pck_matches_same_side_hands_by_distance(policy: EvaluationPolicy) -> None:
    """같은 쪽 손이 여럿이면 거리로 일대일 매칭하고, 예측 하나를 두 정답에 쓰지 않는다 (ADR
    0025)."""
    # 바디캠에 왼손이 둘 (착용자, 돌봄 대상). 예측 순서가 정답과 달라도 가까운 손끼리 맞춘다
    truth = [_hand("wearer_l", 100), _hand("recipient_l", 400)]
    pred = as_model([_hand("h0", 400), _hand("h1", 100)])
    assert _pck("hands", truth, pred, policy) == 1.0
    # 예측 트랙 하나는 한 시각에 정답 하나에만 쓴다 (두 정답에 같은 예측을 쓰지 않는다)
    assert _pck("hands", truth, as_model([_hand("h1", 100)]), policy) == 0.5
    # 반대쪽 손 예측과는 맞추지 않는다
    assert _pck("hands", truth, as_model([_hand("h1", 100, "right")]), policy) == 0.0


def _policy_yaml() -> dict[str, Any]:
    """저장소 evaluation.yaml을 검증 전 사전으로 읽는다 (일부를 바꿔 검증기를 시험한다)."""
    data: dict[str, Any] = yaml.safe_load(
        (ROOT / "config" / "policies" / "evaluation.yaml").read_text("utf-8")
    )
    return data


@pytest.mark.parametrize(
    ("task", "key", "value", "bad"),
    [
        ("objects", "primary", "hotaa", "hotaa"),  # 주 지표 오타
        ("contact", "guards", ["contact_end_f1", "grasp_f1"], "grasp_f1"),  # 지키는 지표 오타
        ("privacy", "max_drop_by", {"blur_precison": 0.05}, "blur_precison"),  # 허용 하락 키 오타
    ],
)
def test_gate_metric_typo_fails_at_policy_load(task: str, key: str, value: Any, bad: str) -> None:
    """게이트 규칙에 그 과제 평가기가 내지 않는 지표 이름이 있으면 정책을 읽을 때 실패한다.

    감사 회귀: 예전에는 그대로 읽혀, 후보·기존 모두 NaN인 비교를 게이트가 조용히 건너뛰었다.
    """
    data = _policy_yaml()
    data["gate"][task][key] = value
    with pytest.raises(ValidationError, match=bad):
        EvaluationPolicy.model_validate(data)


def test_segment_metric_names_follow_segment_iou() -> None:
    """segment_iou에서 0.5를 빼면 gate.actions.primary(segment_f1_0.5)는 모르는 이름이 된다."""
    data = _policy_yaml()
    data["segment_iou"] = [0.1, 0.25]
    with pytest.raises(ValidationError, match=r"segment_f1_0\.5"):
        EvaluationPolicy.model_validate(data)


def test_evaluators_emit_exactly_registered_metrics(policy: EvaluationPolicy) -> None:
    """평가기 출력 지표가 `task_metrics`(TASK_METRICS + segment_f1_<문턱>)와 같다.

    게이트 지표 이름 검사의 근거 목록이 실제 출력과 어긋나지 않는지 본다 (`evaluate`도 검사한다).
    """
    report = run([scenario_data(0, glove=True), scenario_data(1, glove=False)], policy)
    assert report.overall
    for task, r in report.overall.items():
        fixed = {m for m in r.metrics if "/" not in m}  # 클래스별 지표(ap/<클래스>)는 접두사만 본다
        assert fixed == task_metrics(task, policy.segment_iou)  # type: ignore[arg-type]
        assert all(is_class_metric(task, m) for m in set(r.metrics) - fixed)  # type: ignore[arg-type]
