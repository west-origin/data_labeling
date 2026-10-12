"""프롬프트·스키마·백엔드 요청 테스트 (`dlp_aireview.prompt`·`schema`·`backend`, WP17.2).

- 프롬프트 틀(system, 묶음별 온톨로지 블록, 응답 스키마, 가변 글, 재시도 글)은 `golden/`의 골든
  파일과 바이트까지 같다. 틀을 바꾸면 판정 버전이 바뀌어야 하므로 실패 메시지가 PROMPT_VERSION과
  aireview.yaml prompt_version을 함께 올리라고 알린다 (`DLP_UPDATE_GOLDEN=1`로 골든을 다시 쓴다).
- 고정 부분(system, 묶음별 온톨로지 블록, 응답 스키마)은 세션·시각·개체 ID와 무관하게 바이트까지
  같다.
- 시간 묶음 둘(temporal, temporal_long)은 같은 고정 블록·스키마를 쓴다.
- portable 스키마 프로필에는 숫자 범위·문자열 길이·minItems가 없다.
- 재시도 대화는 요청 끝에만 붙는다 (앞부분 = 캐시 접두사가 그대로).
- 가변 글에 모델 버전·신뢰도·검수자가 들어가지 않고, 원본 접근 이벤트는 고정 블록에 없다.
- 사내 서버 요청: user 메시지 하나(고정 블록 → 이미지 → 가변 글), 추적 헤더, 응답 모델 대조.
"""

from __future__ import annotations

import json
import os
from fractions import Fraction
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest

from dlp_aireview.backend import JudgeRequest, OpenAICompatibleJudge, chat_request
from dlp_aireview.plan import PlannedUnit, StreamMedia, plan_session
from dlp_aireview.policy import PROMPT_VERSION, AiReviewPolicy
from dlp_aireview.prompt import build_parts, ontology_text, retry_text, system_text, unit_text
from dlp_aireview.schema import response_schema
from dlp_aireview.sensors import SensorContext, contact_text
from dlp_media.pts import PtsIndex
from dlp_models.vlm import VlmRequestError, chat_body
from dlp_schema.aireview import AiGroup
from dlp_schema.labels import (
    ActionPayload,
    BoxKeyframe,
    BoxTrackPayload,
    EventPayload,
    Evidence,
    GapPayload,
    Hand,
    HandStatePayload,
    LabelRecord,
    Provenance,
    Source,
)
from dlp_schema.ontology import Ontology
from dlp_schema.testing import FIXED_TIME, make_session

from .conftest import Scene, images_for, make_scene, with_policy

ABSENT = SensorContext.absent()
FORBIDDEN = {"minimum", "maximum", "minLength", "maxLength", "minItems", "exclusiveMinimum"}
GOLDEN = Path(__file__).parent / "golden"
# 골든이 다를 때 알리는 절차 (판정 버전에는 정책 값만 들어가므로 틀 변경은 이 버전으로만 드러난다)
BUMP = (
    "프롬프트 틀(dlp_aireview.prompt·schema의 고정 문장·스키마 모양)이 바뀌었다. "
    "판정 버전이 바뀌어야 하므로 dlp_aireview.policy.PROMPT_VERSION과 "
    "config/policies/aireview.yaml prompt_version을 함께 올리고 "
    "DLP_UPDATE_GOLDEN=1로 골든을 다시 쓴다. 온톨로지 v1이나 정책 값(digest에 드는 절)이 "
    "바뀐 것이면 판정 버전이 이미 바뀌므로 골든만 다시 쓴다."
)


def _golden(name: str, text: str) -> None:
    """골든 파일과 바이트 비교 (`DLP_UPDATE_GOLDEN=1`이면 다시 쓴다)."""
    path = GOLDEN / name
    if os.environ.get("DLP_UPDATE_GOLDEN") == "1":
        path.parent.mkdir(exist_ok=True)
        path.write_text(text, "utf-8")
    assert path.read_text("utf-8") == text, f"{name}: {BUMP}"


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
    """고정 블록에는 세션 ID·개체 ID·시각이 없고, 시간 묶음 둘은 같은 블록을 쓴다. 원본 접근
    이벤트(discomfort)는 넣지 않는다 (바이트 고정은 `test_prompt_templates_match_golden_files`)."""
    system = system_text(policy)
    assert f"v{PROMPT_VERSION}" in system
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
        entities=unit.window.entities, frame_nos=tuple(range(len(unit.frame_ms))),
        frame_ms=unit.frame_ms,
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


def _golden_units(policy: AiReviewPolicy, o: Ontology) -> tuple[list[PlannedUnit], SensorContext]:
    """골든용 고정 장면의 단위: 생성기와 무관하게 손으로 만든 라벨 (바디캠 100 ms 간격 프레임).

    시간 대상 4개(행동·사이 구간·이벤트·영상 접촉 손 상태), 사람이 만든 맥락 행동 하나, 박스 트랙
    둘(개체표·공간 대상), 오른손 장갑 접촉 구간.
    """
    session = make_session("s900", duration_ms=12_000)

    def label(label_id: str, payload: Any, start: int, end: int, **kw: Any) -> LabelRecord:
        data: dict[str, Any] = {
            "label_id": label_id, "session_id": "s900", "t_start_ms": start, "t_end_ms": end,
            "ontology_version": "1.0.0", "created_at": FIXED_TIME, "payload": payload,
            "provenance": Provenance(source=Source.MODEL, model_version="m-1"), "confidence": 0.7,
        } | kw  # fmt: skip
        return LabelRecord.model_validate(data)

    def box(label_id: str, entity: str, cls: str, x: float) -> LabelRecord:
        kfs = tuple(BoxKeyframe(t_ms=t, x=x, y=60, w=30, h=30) for t in range(0, 3_001, 500))
        payload = BoxTrackPayload(entity_id=entity, class_id=cls, keyframes=kfs)
        return label(label_id, payload, 0, 3_000, stream_id="bodycam")

    grasp = ActionPayload(
        action_id="act-1", hand=Hand.RIGHT, verb="grasp", target_id="cup_01",
        t_approach_ms=1_000, t_contact_start_ms=1_300, t_contact_end_ms=1_800, t_end_ms=2_000,
    )  # fmt: skip
    push = ActionPayload(
        action_id="act-p", hand=Hand.LEFT, verb="push", target_id="sink_01",
        t_approach_ms=1_000, t_end_ms=2_500,
    )  # fmt: skip
    hand = HandStatePayload(
        hand=Hand.RIGHT, contact_target_kind="object", target_id="cup_01", grasp_type="power",
        role="active",
    )  # fmt: skip
    labels = [
        label("a1", grasp, 1_000, 2_000),
        label("g1", GapPayload(hand=Hand.RIGHT, gap_type="idle"), 2_000, 3_000),
        label("e1", EventPayload(event_type="slip", severity=2), 2_500, 2_900),
        label("h1", hand, 1_300, 1_800, evidence=Evidence.INFERRED),
        label(
            "p1", push, 1_000, 2_500, provenance=Provenance(source=Source.HUMAN), confidence=None
        ),
        box("b-cup", "cup_01", "cup", 100.0),
        box("b-sink", "sink_01", "sink", 200.0),
    ]
    times = np.arange(0, 12_000, 100, dtype=np.int64)
    pts = PtsIndex(Fraction(1, 1000), times, np.ones(len(times), dtype=bool))
    sensor = SensorContext.of({Hand.RIGHT: [(1_300.0, 1_800.0)]})
    units = plan_session(
        session, labels, policy, o, media={"bodycam": StreamMedia(pts, "e" * 64)}, sensor=sensor
    )
    return units, sensor


def test_prompt_templates_match_golden_files(policy: AiReviewPolicy, ontology: Ontology) -> None:
    """프롬프트 틀 전부(system, 묶음별 고정 블록, 응답 스키마, 시간·공간 가변 글, 재시도 글)를 골든
    파일과 바이트 비교한다. 판정 버전(digest)에는 정책 값만 들어가므로, 틀을 바꾸면 이 테스트가
    PROMPT_VERSION을 올리라고 알린다 (같은 버전에 다른 프롬프트의 판정이 섞이지 않게)."""
    _golden("system.txt", system_text(policy))
    _golden("ontology_temporal.txt", ontology_text(ontology, AiGroup.TEMPORAL, policy))
    _golden("ontology_spatial.txt", ontology_text(ontology, AiGroup.SPATIAL, policy))
    plain = with_policy(policy, temporal={"timestamp_overlay": False})
    _golden("ontology_temporal_plain.txt", ontology_text(ontology, AiGroup.TEMPORAL, plain))
    for group in (AiGroup.TEMPORAL, AiGroup.SPATIAL):
        schema = json.dumps(
            response_schema(ontology, group, policy), ensure_ascii=False, indent=1, sort_keys=True
        )
        _golden(f"schema_{group.value}.json", schema + "\n")
    units, sensor = _golden_units(policy, ontology)
    for group in (AiGroup.TEMPORAL, AiGroup.SPATIAL):
        unit = next(u for u in units if u.group is group)
        images = images_for(unit, policy)
        text = unit_text(unit, images, range(len(unit.targets)), ontology, sensor)
        _golden(f"unit_{group.value}.txt", text)
    _golden("retry.txt", retry_text(["번호 #0가 빠짐", "#1: 이유는 한국어 한 문장"]))


def test_golden_scene_covers_every_text_row(policy: AiReviewPolicy, ontology: Ontology) -> None:
    """골든 가변 글이 행 종류(행동·사이 구간·이벤트·손 상태·맥락·개체표·장갑 문장·표식)를 다
    담는다."""
    units, sensor = _golden_units(policy, ontology)
    temporal = next(u for u in units if u.group is AiGroup.TEMPORAL)
    text = unit_text(temporal, images_for(temporal, policy), range(4), ontology, sensor)
    for piece in ("행동 오른손", "사이 구간", "이벤트", "손 상태", "(확정)", "개체:", "장갑 압력"):
        assert piece in text, piece
    spatial = next(u for u in units if u.group is AiGroup.SPATIAL)
    assert "B0" in unit_text(spatial, images_for(spatial, policy), [0], ontology, sensor)


def test_plain_frames_prompt_matches_overlay(
    scene: Scene, policy: AiReviewPolicy, ontology: Ontology
) -> None:
    """`timestamp_overlay: false`면 시간 프레임에 글자를 그리지 않고, 고정 블록도 모서리 표기가
    있다고 말하지 않는다 (프롬프트가 그림과 맞다). 공간 프레임은 늘 표기한다."""
    plain = with_policy(policy, temporal={"timestamp_overlay": False})
    block = ontology_text(ontology, AiGroup.TEMPORAL, plain)
    assert "모서리" not in block and "글자가 없다" in block
    assert "모서리의 F#" in ontology_text(ontology, AiGroup.SPATIAL, plain)
    units = _units(scene, plain, ontology)
    unit = next(u for u in units if u.group is AiGroup.TEMPORAL)
    stamped = images_for(unit, policy)
    bare = images_for(unit, plain)
    assert all(a.image.sha256 != b.image.sha256 for a, b in zip(stamped, bare, strict=True))
    assert plain.verdict_version != policy.verdict_version


def test_served_name_is_sent_as_request_model(
    scene: Scene, policy: AiReviewPolicy, ontology: Ontology
) -> None:
    """`served_name`이 있으면 요청 본문의 model과 응답 대조 모두 그 이름이다 (vLLM은
    --served-model-name이 아닌 이름을 404로 거부한다). 판정 버전·라이선스 이름은 backend.model."""
    aliased = with_policy(policy, backend={"served_name": "judge"})
    unit = _units(scene, aliased, ontology)[0]
    seen: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(json.loads(req.content))
        return httpx.Response(
            200, json={"model": "judge", "choices": [{"message": {"content": "{}"}}]}
        )

    judge = OpenAICompatibleJudge(
        "spark-01", "http://10.0.0.11:8000", aliased, transport=httpx.MockTransport(handler)
    )
    assert judge.judge(_request(unit, aliased, ontology)).served_model == "judge"
    assert seen[0]["model"] == "judge"
    assert aliased.verdict_version.endswith("+vlm-" + policy.backend.model)
    assert judge.info.model == policy.backend.model
