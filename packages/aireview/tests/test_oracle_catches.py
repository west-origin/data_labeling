"""정답을 아는 합성 오류 탐지와 판정 경로 테스트 (`perturb`·`oracle`·`judge`, WP17.2 완료 기준).

`perturb`로 넣은 오류를 `OracleJudge`(정답 라벨로 답하는 stub)가 계획 → 프롬프트 → 응답 재검사 →
제안 완성의 실제 경로를 거쳐 100% wrong으로 잡고, 제안이 원래 값과 같고, 손대지 않은 라벨은 ok인지
본다 (배관 검사: 픽셀을 보지 않는다). `faults`로 재시도·부분 수용·미응답·문맥 초과 분할·장애·설정
오류 경로를 확인한다.
"""

from __future__ import annotations

from collections import Counter

import pytest

from dlp_aireview.judge import JudgedUnit, judge_unit
from dlp_aireview.oracle import OracleJudge
from dlp_aireview.perturb import KINDS, perturb
from dlp_aireview.plan import PlannedUnit, plan_session
from dlp_aireview.policy import AiReviewPolicy
from dlp_aireview.prompt import mark_tag
from dlp_aireview.sensors import SensorContext
from dlp_models.vlm import VlmRequestError, VlmUnavailableError
from dlp_schema.aireview import AiGroup, AiProblem, AiVerdict, AiVerdictValue
from dlp_schema.labels import ActionPayload, LabelRecord
from dlp_schema.ontology import Ontology
from dlp_schema.testing import FIXED_TIME
from dlp_schema.validation import check_label

from .conftest import Scene, images_for, with_policy

ABSENT = SensorContext.absent()
RUN = "air-run-" + "1" * 24


def _units(
    scene: Scene, labels: list[LabelRecord], policy: AiReviewPolicy, o: Ontology
) -> list[PlannedUnit]:
    """장면 세션·블러본으로 라벨을 계획한다."""
    return plan_session(scene.session, labels, policy, o, media=scene.media, sensor=ABSENT)


def _judge(
    judge: OracleJudge, unit: PlannedUnit, policy: AiReviewPolicy, o: Ontology
) -> JudgedUnit:
    """단위 하나를 판정한다."""
    return judge_unit(
        judge, unit, images_for(unit, policy), ontology=o, policy=policy, sensor=ABSENT
    )


def _verdicts(judged: list[JudgedUnit]) -> dict[str, list[AiVerdict]]:
    """라벨 ID → 판정 목록."""
    out: dict[str, list[AiVerdict]] = {}
    for j in judged:
        for v in j.verdicts:
            out.setdefault(v.label_id, []).append(v)
    return out


def _with_tool(labels: list[LabelRecord]) -> list[LabelRecord]:
    """첫 행동에 도구를 단 라벨 집합 (tool_swap 대상이 생기게)."""
    out = list(labels)
    k = next(i for i, x in enumerate(out) if isinstance(x.payload, ActionPayload))
    payload = out[k].payload.model_copy(update={"tool_id": "spray_bottle_01"})
    out[k] = out[k].model_copy(update={"payload": payload})
    return out


# (교란 비율, seed). 두 경우를 합치면 9종을 모두 넣는다 (아래 테스트가 확인)
CASES = [
    ({k: 0.3 for k in KINDS}, 3),
    ({"tool_swap": 1.0, "hand_swap": 0.4, "action_to_gap": 0.4}, 0),
]


@pytest.mark.parametrize(("rates", "seed"), CASES)
def test_injected_errors_are_all_caught_with_original_values(
    rates: dict[str, float], seed: int, scene: Scene, policy: AiReviewPolicy, ontology: Ontology
) -> None:
    """넣은 오류는 모두 wrong, 제안은 원래 값, 손대지 않은 라벨은 모두 ok (오지적 0)."""
    base = _with_tool(scene.labels)
    p = perturb(base, rates, seed=seed, o=ontology)
    kinds = Counter(p.injected.values())
    assert set(kinds) <= set(rates) and len(kinds) >= min(3, len(rates)), kinds
    assert all(not check_label(x, ontology) for x in p.labels)
    units = _units(scene, p.labels, policy, ontology)
    judged = [_judge(OracleJudge(base), u, policy, ontology) for u in units]
    found = _verdicts(judged)
    planned = {x.label_id for j in judged for x in j.unit.targets}
    assert set(p.injected) <= planned
    for label_id, verdicts in found.items():
        expected = AiVerdictValue.WRONG if label_id in p.injected else AiVerdictValue.OK
        assert {v.verdict for v in verdicts} == {expected}, (label_id, p.injected.get(label_id))
    for label_id, kind in p.injected.items():
        v = found[label_id][0]
        truth = p.truth.get(label_id)
        s = v.suggestion
        if kind == "false_event":
            assert s is not None and s.op == "delete" and AiProblem.FALSE_EVENT in v.problems
        elif kind == "hand_swap":
            assert AiProblem.WRONG_HAND in v.problems and s is None
        elif kind == "action_to_gap":
            assert truth is not None and isinstance(truth.payload, ActionPayload)
            assert s is not None and s.op == "change_kind" and isinstance(s.payload, ActionPayload)
            assert (s.payload.verb, s.payload.target_id, s.payload.hand) == (
                truth.payload.verb, truth.payload.target_id, truth.payload.hand,
            )  # fmt: skip
            assert s.payload.action_id.startswith("h-")
        else:
            assert truth is not None and s is not None and s.payload == truth.payload, (kind, v)
        assert v.evidence_t_ms is not None and v.answered


def test_cases_cover_all_perturbation_kinds(scene: Scene, ontology: Ontology) -> None:
    """위 매개변수 경우들을 합치면 교란 9종을 모두 넣는다."""
    seen: set[str] = set()
    for rates, seed in CASES:
        seen |= set(
            perturb(_with_tool(scene.labels), rates, seed=seed, o=ontology).injected.values()
        )
    assert seen == set(KINDS)


def test_judged_unit_records_are_valid_and_deterministic(
    scene: Scene, policy: AiReviewPolicy, ontology: Ontology
) -> None:
    """판정 기록은 계약을 지킨다 (대상마다 판정 하나, 응답 sha만 저장).

    같은 입력이면 같은 기록이다.
    """
    unit = _units(scene, scene.labels, policy, ontology)[0]
    a = _judge(OracleJudge(scene.labels), unit, policy, ontology)
    b = _judge(OracleJudge(scene.labels), unit, policy, ontology)
    ra, rb = a.record(RUN, FIXED_TIME), b.record(RUN, FIXED_TIME)
    assert ra == rb
    assert ra.target_label_ids == tuple(x.label_id for x in unit.targets)
    assert ra.response_sha256 is not None and ra.outcome == "answered" and ra.attempts == 1
    assert [f.t_ms for f in ra.frames] == list(unit.frame_ms)
    assert ra.served_model == "oracle-judge" and ra.endpoint == "oracle"


def _small(
    scene: Scene, policy: AiReviewPolicy, ontology: Ontology
) -> tuple[AiReviewPolicy, PlannedUnit]:
    """대상 2개짜리 시간 단위 (위반 비율을 손으로 맞추려고)."""
    small = with_policy(policy, temporal={"max_targets": 2})
    unit = next(
        u
        for u in _units(scene, scene.labels, small, ontology)
        if u.group is AiGroup.TEMPORAL and len(u.targets) == 2
    )
    return small, unit


def test_retry_then_success_and_fallback(
    scene: Scene, policy: AiReviewPolicy, ontology: Ontology
) -> None:
    """JSON이 아니면 위반 목록을 붙여 다시 묻는다.

    끝내 형식이 맞지 않으면 fallback(모두 미응답 unsure).
    """
    small, unit = _small(scene, policy, ontology)
    oracle = OracleJudge(scene.labels, ["not_json"])
    once = _judge(oracle, unit, small, ontology)
    assert once.outcome == "answered" and once.attempts == 2
    assert [v.verdict for v in once.verdicts] == [AiVerdictValue.OK, AiVerdictValue.OK]
    assert oracle.requests[1].history and "JSON" in oracle.requests[1].history[0][1]
    assert any("JSON" in e for e in once.errors)
    failed = _judge(OracleJudge(scene.labels, ["not_json", "not_json"]), unit, small, ontology)
    assert failed.outcome == "fallback" and failed.response_sha256 is None
    assert all(v.verdict is AiVerdictValue.UNSURE and not v.answered for v in failed.verdicts)
    assert failed.record(RUN, FIXED_TIME).outcome == "fallback"


@pytest.mark.parametrize("fault", ["bad_ref", "missing_ref", "bad_value"])
def test_partial_acceptance(
    fault: str, scene: Scene, policy: AiReviewPolicy, ontology: Ontology
) -> None:
    """위반이 대상 수의 절반 이하면 다시 묻지 않는다.

    위반 항목만 unsure + response_invalid로 낮춘다 (부분 수용).
    """
    small, unit = _small(scene, policy, ontology)
    judged = _judge(OracleJudge(scene.labels, [fault]), unit, small, ontology)
    assert judged.attempts == 1 and judged.outcome == "answered" and judged.errors
    first, second = judged.verdicts
    assert second.verdict is AiVerdictValue.OK and second.answered
    if fault == "bad_ref":
        assert first.verdict is AiVerdictValue.OK
    else:
        assert first.verdict is AiVerdictValue.UNSURE and not first.answered
        assert first.problems == (AiProblem.RESPONSE_INVALID,)


def test_context_overflow_splits_inside_the_unit(
    scene: Scene, policy: AiReviewPolicy, ontology: Ontology
) -> None:
    """문맥 초과: 프레임 절반 → 대상 절반으로 단위 안에서 나누고 결과는 한 행이다.

    끝내 안 되면 fallback.
    """
    small, unit = _small(scene, policy, ontology)
    n = len(unit.frame_ms)
    oracle = OracleJudge(scene.labels, ["context_overflow"])
    halved = _judge(oracle, unit, small, ontology)
    assert halved.sub_requests == 1 and halved.attempts == 2
    assert len(oracle.requests[-1].frame_ms) == (n + 1) // 2
    oracle = OracleJudge(scene.labels, ["context_overflow"] * 2)
    split = _judge(oracle, unit, small, ontology)
    assert split.sub_requests == 2 and split.attempts == 4
    assert [len(r.targets) for r in oracle.requests[-2:]] == [1, 1]
    assert [v.label_id for v in split.verdicts] == [x.label_id for x in unit.targets]
    assert all(v.verdict is AiVerdictValue.OK for v in split.verdicts)
    record = split.record(RUN, FIXED_TIME)
    assert record.sub_requests == 2 and record.outcome == "answered"
    gave_up = _judge(OracleJudge(scene.labels, ["context_overflow"] * 4), unit, small, ontology)
    assert gave_up.outcome == "fallback" and gave_up.attempts == 4
    assert all(not v.answered for v in gave_up.verdicts)


def test_server_failures_are_not_recorded(
    scene: Scene, policy: AiReviewPolicy, ontology: Ontology
) -> None:
    """서버 장애와 설정 오류는 결과(unsure)로 기록하지 않고 올린다."""
    small, unit = _small(scene, policy, ontology)
    with pytest.raises(VlmUnavailableError):
        _judge(OracleJudge(scene.labels, ["unavailable"]), unit, small, ontology)
    with pytest.raises(VlmRequestError):
        _judge(OracleJudge(scene.labels, ["bad_request"]), unit, small, ontology)
    with pytest.raises(ValueError, match="고장"):
        OracleJudge(scene.labels, ["no_such_fault"])


def test_perturb_is_deterministic_and_validates_rates(scene: Scene, ontology: Ontology) -> None:
    """같은 seed면 같은 교란, 다른 seed면 다르다. 모르는 종류·범위 밖 비율은 오류."""
    rates = {k: 0.3 for k in KINDS}
    a = perturb(scene.labels, rates, seed=1, o=ontology)
    assert a == perturb(scene.labels, rates, seed=1, o=ontology)
    assert a.injected != perturb(scene.labels, rates, seed=2, o=ontology).injected
    assert perturb(scene.labels, {}, seed=1, o=ontology).labels == scene.labels
    with pytest.raises(ValueError):
        perturb(scene.labels, {"nope": 0.1}, seed=1, o=ontology)
    with pytest.raises(ValueError):
        perturb(scene.labels, {"verb_swap": 1.5}, seed=1, o=ontology)


def test_split_spatial_request_keeps_image_mark_names(
    scene: Scene, policy: AiReviewPolicy, ontology: Ontology
) -> None:
    """공간 단위를 대상 절반으로 나눈 요청에서도 대상 표의 표식 이름은 그림(단위 안 위치)과 같다."""
    unit = next(
        u for u in _units(scene, scene.labels, policy, ontology) if u.group is AiGroup.SPATIAL
    )
    assert len(unit.targets) >= 2
    oracle = OracleJudge(scene.labels, ["context_overflow"] * 2)
    judged = _judge(oracle, unit, policy, ontology)
    assert judged.sub_requests == 2 and all(v.verdict is AiVerdictValue.OK for v in judged.verdicts)
    mid = (len(unit.targets) + 1) // 2
    second = oracle.requests[-1]
    assert second.targets[0].label_id == unit.targets[mid].label_id
    assert f"#0 {mark_tag(unit.targets[mid], mid)} " in second.parts.text
