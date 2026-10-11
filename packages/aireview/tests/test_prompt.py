"""프롬프트·스키마·백엔드 요청 테스트 (`dlp_aireview.prompt`·`schema`·`backend`, WP17.2).

- 고정 부분(system, 묶음별 온톨로지 블록, 응답 스키마)은 세션·시각·개체 ID와 무관하게 바이트까지
  같다.
- 시간 묶음 둘(temporal, temporal_long)은 같은 고정 블록·스키마를 쓴다.
- portable 스키마 프로필에는 숫자 범위·문자열 길이·minItems가 없다.
- 재시도 대화는 요청 끝에만 붙는다 (앞부분 = 캐시 접두사가 그대로).
- 가변 글에 모델 버전·신뢰도·검수자가 들어가지 않고, 원본 접근 이벤트는 고정 블록에 없다.
- 사내 서버 요청: user 메시지 하나(고정 블록 → 이미지 → 가변 글), 추적 헤더, 응답 모델 대조.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import httpx
import pytest

from dlp_aireview.backend import JudgeRequest, OpenAICompatibleJudge, chat_request
from dlp_aireview.plan import PlannedUnit, plan_session
from dlp_aireview.policy import PROMPT_VERSION, AiReviewPolicy
from dlp_aireview.prompt import build_parts, ontology_text, retry_text, system_text
from dlp_aireview.schema import response_schema
from dlp_aireview.sensors import SensorContext, contact_text
from dlp_models.vlm import VlmRequestError, chat_body
from dlp_schema.aireview import AiGroup
from dlp_schema.labels import Hand
from dlp_schema.ontology import Ontology

from .conftest import Scene, images_for, make_scene, with_policy

ABSENT = SensorContext.absent()
# 고정 system 문장의 해시. 문장을 바꾸면 PROMPT_VERSION(코드)과 aireview.yaml prompt_version을 함께
# 올리고 이 값을 갱신한다 (판정 버전이 바뀌어 다시 판정한다).
SYSTEM_SHA = "13d00fc583cd2c523fa85c498a3afe3f921d8a2c8bf8a68871d348543fa7d375"
FORBIDDEN = {"minimum", "maximum", "minLength", "maxLength", "minItems", "exclusiveMinimum"}


def _keys(node: Any) -> set[str]:
    """중첩 사전의 모든 키."""
    if isinstance(node, dict):
        return set(node) | {k for v in node.values() for k in _keys(v)}  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
    if isinstance(node, list):
        return {k for v in node for k in _keys(v)}  # pyright: ignore[reportUnknownVariableType]
    return set()


def _units(scene: Scene, policy: AiReviewPolicy, ontology: Ontology) -> list[PlannedUnit]:
    """장면의 판정 단위 (장갑 없음)."""
    return plan_session(
        scene.session, scene.labels, policy, ontology, media=scene.media, sensor=ABSENT
    )


def test_static_blocks_are_byte_fixed(policy: AiReviewPolicy, ontology: Ontology) -> None:
    """system 문장은 해시로 고정한다. 고정 블록에는 세션 ID·개체 ID·시각이 없고, 시간 묶음 둘은 같은
    블록을 쓴다. 원본 접근 이벤트(discomfort)는 넣지 않는다."""
    system = system_text(policy)
    assert f"v{PROMPT_VERSION}" in system
    assert hashlib.sha256(system.encode()).hexdigest() == SYSTEM_SHA
    temporal = ontology_text(ontology, AiGroup.TEMPORAL, policy)
    assert temporal == ontology_text(ontology, AiGroup.TEMPORAL_LONG, policy)
    spatial = ontology_text(ontology, AiGroup.SPATIAL, policy)
    for block in (system, temporal, spatial):
        assert "s001" not in block and "cup_01" not in block and "discomfort" not in block
    assert "grasp 잡다" in temporal and "cup 컵" in spatial
    assert "JSON Schema" not in temporal
    prompted = with_policy(policy, backend={"schema_enforcement": "prompt"})
    assert "응답 JSON Schema" in ontology_text(ontology, AiGroup.TEMPORAL, prompted)


def test_static_parts_same_across_sessions(policy: AiReviewPolicy, ontology: Ontology) -> None:
    """다른 세션(다른 시나리오)의 단위도 system·고정 블록이 같고, 가변 글만 다르다."""
    a, b = make_scene(0), make_scene(3, n_units=4)
    ua, ub = _units(a, policy, ontology)[0], _units(b, policy, ontology)[0]
    pa = build_parts(ua, images_for(ua, policy), range(len(ua.targets)), ontology, policy, ABSENT)
    pb = build_parts(ub, images_for(ub, policy), range(len(ub.targets)), ontology, policy, ABSENT)
    assert (pa.system, pa.ontology) == (pb.system, pb.ontology)
    assert pa.text != pb.text
    assert pa.blocks()[0] == pa.ontology and pa.blocks()[-1] == pa.text


def test_unit_text_leaves_out_model_details(
    scene: Scene, policy: AiReviewPolicy, ontology: Ontology
) -> None:
    """가변 글: 모델 버전·신뢰도·라벨 ID·개체 ID를 넣지 않는다 (번호로만 가리킨다)."""
    for u in _units(scene, policy, ontology):
        text = build_parts(
            u, images_for(u, policy), range(len(u.targets)), ontology, policy, ABSENT
        ).text
        assert "prelabel-test" not in text and "신뢰" not in text and "confidence" not in text
        assert "s001-" not in text and "cup_01" not in text
        assert text.count("\n#") == len(u.targets)


def test_sensor_sentence_relative_to_evidence_span() -> None:
    """장갑 문장은 증거 구간 기준 상대 초이고, 장갑을 낀 손에 접촉이 없으면 "없음"이다."""
    ctx = SensorContext.of({Hand.RIGHT: [(2_200.4, 4_800.0), (9_000, 9_500)], Hand.LEFT: []})
    assert contact_text(ctx, (1_000, 8_000)) == "장갑 압력 접촉(센서): 오른손 1.2~3.8 s, 왼손 없음"
    assert contact_text(SensorContext.absent(), (0, 1)) is None
    assert (
        ctx.sha256
        == SensorContext.of({Hand.LEFT: [], Hand.RIGHT: [(2_200, 4_800), (9_000, 9_500)]}).sha256
    )
    assert SensorContext.absent().sha256 == "absent"


def test_response_schema_is_constant_and_portable(
    policy: AiReviewPolicy, ontology: Ontology
) -> None:
    """응답 스키마: 묶음·정책·온톨로지가 같으면 같다 (세션 무관). portable에는 숫자 범위·길이·
    minItems가 없고 full에는 있다. 누락 발견은 기본 끔(maxItems 0)."""
    temporal = response_schema(ontology, AiGroup.TEMPORAL, policy)
    assert temporal == response_schema(ontology, AiGroup.TEMPORAL_LONG, policy)
    assert json.dumps(temporal, sort_keys=True) == json.dumps(
        response_schema(ontology, AiGroup.TEMPORAL, policy), sort_keys=True
    )
    spatial = response_schema(ontology, AiGroup.SPATIAL, policy)
    assert spatial != temporal
    for schema in (temporal, spatial):
        assert not _keys(schema) & FORBIDDEN
        assert schema["properties"]["missing"]["maxItems"] == 0
    full = with_policy(policy, backend={"schema_profile": "full"})
    assert _keys(response_schema(ontology, AiGroup.TEMPORAL, full)) & FORBIDDEN
    item = temporal["properties"]["items"]["items"]["properties"]
    verbs = item["s"]["anyOf"][1]["properties"]["verb"]["enum"]
    assert verbs[-1] is None and verbs[:-1] == sorted(verbs[:-1]) and "wipe" not in verbs
    events = item["s"]["anyOf"][1]["properties"]["event_type"]["enum"]
    assert "discomfort" not in events
    spatial_s = spatial["properties"]["items"]["items"]["properties"]["s"]["anyOf"][1]
    assert set(spatial_s["properties"]) == {
        "class_id", "hand", "delete",
    }  # fmt: skip


def _request(
    unit: PlannedUnit,
    policy: AiReviewPolicy,
    ontology: Ontology,
    history: tuple[tuple[str, str], ...] = (),
) -> JudgeRequest:
    """단위 전체 대상을 묻는 판정 요청."""
    images = images_for(unit, policy)
    parts = build_parts(unit, images, range(len(unit.targets)), ontology, policy, ABSENT)
    return JudgeRequest(
        unit_id=unit.unit_id, affinity=unit.window.window_key, group=unit.group, parts=parts,
        history=history, schema=response_schema(ontology, unit.group, policy),
        schema_enforcement="server", max_tokens=policy.backend.max_tokens, targets=unit.targets,
        entities=unit.window.entities, frame_ms=unit.frame_ms,
    )  # fmt: skip


def test_retry_turns_only_append(scene: Scene, policy: AiReviewPolicy, ontology: Ontology) -> None:
    """재시도 대화는 끝에만 붙는다.

    첫 user 메시지(고정 블록 → 이미지 → 가변 글)와 system이 그대로다.
    """
    unit = _units(scene, policy, ontology)[0]
    first = chat_body("m", chat_request(_request(unit, policy, ontology)))
    retry = chat_body(
        "m",
        chat_request(_request(unit, policy, ontology, (("{}", retry_text(["번호 #0가 빠짐"])),))),
    )
    assert retry["messages"][: len(first["messages"])] == first["messages"]
    assert [m["role"] for m in retry["messages"]] == ["system", "user", "assistant", "user"]
    assert "번호 #0가 빠짐" in retry["messages"][-1]["content"][0]["text"]
    content = first["messages"][1]["content"]
    assert content[0]["type"] == "text" and content[-1]["type"] == "text"
    assert all(c["type"] == "image_url" for c in content[1:-1])
    assert all(c["image_url"]["url"].startswith("data:image/jpeg;base64,") for c in content[1:-1])


def test_openai_compatible_judge_request_and_model_check(
    scene: Scene, policy: AiReviewPolicy, ontology: Ontology
) -> None:
    """사내 서버 판정: 단위 키는 헤더(본문 아님), 응답 형식은 json_schema, 서버 모델이 정책 모델과
    다르면 설정 오류다."""
    unit = _units(scene, policy, ontology)[0]
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        body = {"model": policy.backend.model, "choices": [{"message": {"content": '{"ok":[]}'}}]}
        return httpx.Response(200, json=body | {"usage": {"prompt_tokens": 9}})

    judge = OpenAICompatibleJudge(
        "spark-01", "http://10.0.0.11:8000", policy, transport=httpx.MockTransport(handler)
    )
    reply = judge.judge(_request(unit, policy, ontology))
    assert reply.endpoint == "spark-01" and reply.prompt_tokens == 9
    [req] = seen
    assert req.headers["X-DLP-Unit"] == unit.unit_id and unit.unit_id.encode() not in req.content
    body = json.loads(req.content)
    assert body["response_format"]["json_schema"]["name"] == f"aireview_{unit.group.value}"
    assert body["max_tokens"] == policy.backend.max_tokens and body["temperature"] == 0
    assert judge.info.endpoints == ("spark-01",) and judge.info.external is False

    def other(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"model": "other", "choices": [{"message": {"content": "{}"}}]}
        )

    wrong = OpenAICompatibleJudge(
        "spark-02", "http://10.0.0.12:8000", policy, transport=httpx.MockTransport(other)
    )
    with pytest.raises(VlmRequestError):
        wrong.judge(_request(unit, policy, ontology))
