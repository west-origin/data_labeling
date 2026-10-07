from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest

from dlp_actions.assemble import merge_spans, to_labels
from dlp_actions.boundaries import boundary_candidates, segments, wrist_series
from dlp_actions.clients import OpenAICompatibleVlm, OracleVlm
from dlp_actions.pipeline import HandResult, segment_hand
from dlp_actions.policy import ActionsPolicy, load_policy
from dlp_actions.vlm import SegmentRequest, build_prompt, classify, response_schema, validate_answer
from dlp_fixtures.actions import ActionScenario, generate_action_scenario
from dlp_fixtures.video import generate_blur_scenario
from dlp_schema.labels import (
    ActionPayload,
    DescriptionPayload,
    GapPayload,
    Hand,
    HandStatePayload,
    KeypointTrackPayload,
    LabelRecord,
)
from dlp_schema.ontology import Ontology, load_ontology
from dlp_schema.testing import FIXED_TIME
from dlp_schema.validation import check_label

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def policy() -> ActionsPolicy:
    return load_policy(ROOT)


@pytest.fixture(scope="module")
def ontology() -> Ontology:
    return load_ontology(ROOT / "config/ontology/v1")


def _inputs(
    sc: ActionScenario,
) -> tuple[KeypointTrackPayload, list[tuple[int, int]], list[LabelRecord]]:
    track = next(x.payload for x in sc.labels if isinstance(x.payload, KeypointTrackPayload))
    contacts = [
        (x.t_start_ms, x.t_end_ms)
        for x in sc.labels
        if isinstance(x.payload, HandStatePayload) and x.payload.contact_target_kind != "none"
    ]
    truth = [x for x in sc.labels if isinstance(x.payload, ActionPayload | GapPayload)]
    return track, contacts, truth


def _request(**kw: Any) -> SegmentRequest:
    base: dict[str, Any] = {
        "session_id": "s", "stream_id": "bodycam", "video": None, "hand": Hand.RIGHT,
        "start_ms": 0, "end_ms": 1000, "in_contact": True, "entities": ("sink_01", "cup_01"),
    }  # fmt: skip
    return SegmentRequest(**(base | kw))


def _run(sc: ActionScenario, ontology: Ontology, policy: ActionsPolicy) -> HandResult:
    track, contacts, truth = _inputs(sc)
    return segment_hand(
        session_id=sc.session_id, stream_id="bodycam", video=None, hand=sc.hand, track=track,
        contacts=contacts, entities=tuple(e.entity_id for e in sc.entities), start_ms=0,
        end_ms=sc.duration_ms, client=OracleVlm(truth), ontology=ontology, policy=policy,
        model_version="actions-test", ontology_version="1.0.0", now=FIXED_TIME,
    )  # fmt: skip


def test_boundary_candidates_recall_at_least_95_percent(policy: ActionsPolicy) -> None:
    """완료 기준: 합성 행동 시퀀스에서 경계 후보 재현율 ≥ 95% (기준 문서 허용 오차 안)."""
    tol = policy.tolerance_ms
    hits = total = 0
    for seed in range(30):
        sc = generate_action_scenario(seed)
        track, contacts, _ = _inputs(sc)
        times, xy, scale = wrist_series(track)
        found = np.array(
            [c.t_ms for c in boundary_candidates(times, xy, scale, contacts, policy.boundaries)]
        )
        truth: set[tuple[int, int]] = set()
        for a in sc.actions:
            truth |= {(a.t_approach_ms, tol.approach), (a.t_end_ms, tol.end)}
            truth |= {
                (t, tol.contact_glove)
                for t in (a.t_contact_start_ms, a.t_contact_end_ms)
                if t is not None
            }
        for t, limit in truth:
            total += 1
            hits += bool(np.abs(found - t).min() <= limit)
    assert hits / total >= 0.95


def test_segments_partition_the_timeline() -> None:
    from dlp_actions.boundaries import Candidate

    cands = [Candidate(t, "valley") for t in (100, 150, 500, 990)]
    parts = segments(cands, 0, 1000, 100)
    assert parts[0][0] == 0 and parts[-1][1] == 1000
    assert all(a[1] == b[0] for a, b in itertools.pairwise(parts))
    assert all(e - s >= 100 for s, e in parts)


def test_prompt_and_schema_carry_the_ontology(ontology: Ontology) -> None:
    request = _request()
    prompt = build_prompt(request, ontology)
    assert "rub: 문지르다" in prompt and "unknown: 미상" in prompt and "sink_01" in prompt
    schema = response_schema(ontology, request.entities)
    assert "rub" in schema["properties"]["verb"]["enum"]
    assert "wipe" not in schema["properties"]["verb"]["enum"]  # 기술은 2단 분류 대상이 아니다
    assert set(schema["properties"]["gap_type"]["enum"]) == {
        "idle",
        "unknown",
        "out_of_scope",
        None,
    }


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("문지르기", "JSON이 아닙니다"),
        (json.dumps({"label": "action", "verb": "dance"}), "verb: dance"),
        (json.dumps({"label": "action", "verb": "wipe"}), "verb: wipe"),
        (
            json.dumps({"label": "action", "verb": "rub", "target_id": "moon_01"}),
            "target_id: moon_01",
        ),
        (json.dumps({"label": "gap", "gap_type": "idle", "verb": "rub"}), "gap에는"),
        (json.dumps({"label": "gap", "gap_type": "nap"}), "gap_type: nap"),
        (json.dumps({"label": "jump"}), "스키마 위반"),
    ],
)
def test_invalid_answers_are_rejected(raw: str, message: str, ontology: Ontology) -> None:
    answer, errors = validate_answer(raw, ontology, _request())
    assert answer is None and any(message in e for e in errors)


def test_retry_then_fallback_to_unknown_not_idle(ontology: Ontology) -> None:
    truth = [x for x in generate_action_scenario(0).labels if isinstance(x.payload, ActionPayload)]
    a = truth[0]
    request = _request(
        start_ms=a.t_start_ms,
        end_ms=a.t_end_ms,
        entities=("cup_01", "spray_bottle_01", "drawer_01", "bucket_01", "sink_01"),
    )
    retried = classify(OracleVlm(truth, faults=["not_json"]), request, ontology, max_retries=2)
    assert retried.attempts == 2 and not retried.fallback and retried.answer.label == "action"
    client = OracleVlm(truth, faults=["not_json", "bad_verb", "bad_target"])
    failed = classify(client, request, ontology, max_retries=2)
    assert failed.fallback and client.calls == 3
    assert (failed.answer.label, failed.answer.gap_type) == ("gap", "unknown")


@pytest.mark.parametrize("seed", range(10))
def test_two_stage_segmentation_recovers_truth(
    seed: int, ontology: Ontology, policy: ActionsPolicy
) -> None:
    """오라클 VLM이면 경계 후보 + 병합만으로 정답 행동 순서와 경계를 되찾는다. 타임라인 공백 0."""
    sc = generate_action_scenario(seed)
    result = _run(sc, ontology, policy)
    got = [x.payload for x in result.labels if isinstance(x.payload, ActionPayload)]
    truth = sc.actions
    assert [(a.verb, a.target_id) for a in got] == [(a.verb, a.target_id) for a in truth]
    tol = policy.tolerance_ms
    within = [
        abs(g.t_approach_ms - t.t_approach_ms) <= tol.approach
        and abs(g.t_end_ms - t.t_end_ms) <= tol.end
        for g, t in zip(got, truth, strict=True)
    ]
    assert sum(within) / len(within) >= 0.9

    timeline = sorted(
        (x for x in result.labels if isinstance(x.payload, ActionPayload | GapPayload)),
        key=lambda x: x.t_start_ms,
    )
    assert timeline[0].t_start_ms == 0 and timeline[-1].t_end_ms == sc.duration_ms
    assert all(a.t_end_ms == b.t_start_ms for a, b in itertools.pairwise(timeline))
    descriptions = {
        x.payload.segment_id for x in result.labels if isinstance(x.payload, DescriptionPayload)
    }
    assert descriptions == {a.action_id for a in got}
    assert all(check_label(x, ontology) == [] for x in result.labels)
    assert result.fallbacks == 0


def test_merge_and_fill_rules(ontology: Ontology, policy: ActionsPolicy) -> None:
    """같은 분류 인접 구간 병합, 접촉이 둘이면 사이 경계에서 나눔, 미상은 사이 구간으로 채움."""

    def piece(s: int, e: int, answer: dict[str, Any]) -> Any:
        from dlp_actions.vlm import Classified, VlmAnswer

        return Classified(_request(start_ms=s, end_ms=e), VlmAnswer.model_validate(answer), 1)

    rub = {
        "label": "action",
        "verb": "rub",
        "target_id": "sink_01",
        "description": "세면대를 문지른다",
    }
    pieces = [
        piece(0, 300, rub), piece(300, 900, rub), piece(900, 1200, rub),  # 접촉 1: 300~900
        piece(1200, 1500, rub), piece(1500, 2000, rub),                    # 접촉 2: 1500~1900
        piece(2000, 2400, {"label": "gap", "gap_type": "unknown"}),
    ]  # fmt: skip
    contacts = [(300, 900), (1500, 1900)]
    spans = merge_spans(pieces, 0.5, contacts, {1200: "valley", 1050: "still_start"})
    assert [(s.start_ms, s.end_ms) for s in spans] == [(0, 1200), (1200, 2000), (2000, 2400)]
    labels = to_labels(
        spans, session_id="s", hand=Hand.RIGHT, contacts=contacts, ontology_version="1.0.0",
        model_version="actions-test", now=FIXED_TIME,
    )  # fmt: skip
    actions = [x.payload for x in labels if isinstance(x.payload, ActionPayload)]
    assert [(a.t_contact_start_ms, a.t_contact_end_ms) for a in actions] == [
        (300, 900),
        (1500, 1900),
    ]
    [gap] = [x.payload for x in labels if isinstance(x.payload, GapPayload)]
    assert gap.gap_type == "unknown"


def test_openai_compatible_client_sends_frames_and_schema(
    tmp_path: Path, ontology: Ontology
) -> None:
    video = tmp_path / "v.mp4"
    generate_blur_scenario(1, duration_ms=1_000).write(video)
    seen: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(json.loads(req.content))
        answer = {"label": "gap", "gap_type": "idle", "description": None}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(answer)}}]})

    client = OpenAICompatibleVlm(
        "http://vlm", "test-model", frames=4, max_side=128, timeout_s=5,
        transport=httpx.MockTransport(handler),
    )  # fmt: skip
    request = _request(video=video, start_ms=100, end_ms=900)
    result = classify(client, request, ontology, max_retries=0)
    assert result.answer.gap_type == "idle"
    [body] = seen
    content = body["messages"][0]["content"]
    assert sum(c["type"] == "image_url" for c in content) == 4
    assert body["response_format"]["json_schema"]["schema"]["properties"]["label"]["enum"] == [
        "action",
        "gap",
    ]


def test_vlm_server_errors_are_retried_then_unknown(ontology: Ontology) -> None:
    """감사 회귀: VLM 서버 오류·시간 초과가 세션 전체를 멈추지 않는다. 재시도 후 그 구간만 미상."""
    calls: list[int] = []

    def failing(req: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ReadTimeout("timeout", request=req)
        return httpx.Response(503, text="busy")

    client = OpenAICompatibleVlm(
        "http://vlm", "test-model", frames=1, max_side=64, timeout_s=1,
        transport=httpx.MockTransport(failing),
    )  # fmt: skip
    result = classify(client, _request(), ontology, max_retries=2)
    assert len(calls) == 3 and result.fallback and result.answer.gap_type == "unknown"
    assert all("VLM 서버 오류" in e for e in result.errors)

    # 한 번 실패한 뒤 성공하면 그 답을 쓴다
    calls.clear()

    def flaky(req: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(500)
        answer = {"label": "gap", "gap_type": "idle", "description": None}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(answer)}}]})

    client = OpenAICompatibleVlm(
        "http://vlm", "test-model", frames=1, max_side=64, timeout_s=1,
        transport=httpx.MockTransport(flaky),
    )  # fmt: skip
    ok = classify(client, _request(), ontology, max_retries=2)
    assert not ok.fallback and ok.answer.gap_type == "idle" and ok.attempts == 2
