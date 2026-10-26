"""장면 분류와 장면별 질의(`dlp_prelabel.scene`) 시험 (ADR 0055).

정답: 합성 응답(JSON)과 합성 프레임(블러 박스 위치를 아는 단색 화면). 사내 VLM 대신 가짜 채팅
클라이언트를 쓴다 (외부 호출 없음).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dlp_models.vlm import ChatReply, ChatRequest, VlmUnavailableError
from dlp_prelabel.adapters.owl_objects import OwlObjects
from dlp_prelabel.policy import load_policy
from dlp_prelabel.scene import (
    FixedScene,
    SceneResult,
    SceneUnavailableError,
    VlmScene,
    blur_boxes,
    parse_scene,
    queries_for,
    sample_times,
)
from dlp_schema.labels import Provenance, Source
from dlp_schema.testing import make_label

ROOT = Path(__file__).resolve().parents[3]


class _Chat:
    """가짜 사내 VLM: 정해진 내용을 돌려주고 받은 요청을 기록한다 (fail이면 장애)."""

    def __init__(self, content: str, *, fail: bool = False) -> None:
        """content: 응답 내용."""
        self.content, self.fail = content, fail
        self.requests: list[ChatRequest] = []

    def chat(self, req: ChatRequest) -> ChatReply:
        """요청을 기록하고 응답한다."""
        self.requests.append(req)
        if self.fail:
            raise VlmUnavailableError("down")
        return ChatReply(self.content, "m", None, None, None, 1)


def test_parse_scene_rejects_unknown_or_low_confidence() -> None:
    """목록 안·신뢰도 충분 → 그 장면. 목록 밖·낮은 신뢰도·형식 오류 → unknown."""
    sp = load_policy(ROOT).scene
    assert parse_scene('{"scene": "kitchen", "confidence": 0.9}', sp) == SceneResult(
        "kitchen", 0.9, "vlm"
    )
    assert parse_scene('{"scene": "garage", "confidence": 0.9}', sp).scene == "unknown"
    assert parse_scene('{"scene": "kitchen", "confidence": 0.2}', sp).scene == "unknown"
    assert parse_scene("not json", sp).scene == "unknown"


def test_vlm_scene_sends_schema_and_images() -> None:
    """요청에 장면 목록 JSON Schema와 프레임 이미지가 실린다. 장애는 SceneUnavailableError."""
    sp = load_policy(ROOT).scene
    chat = _Chat('{"scene": "bathroom", "confidence": 0.8}')
    frames = [np.zeros((1080, 1920, 3), dtype=np.uint8)] * 2
    assert VlmScene(chat, "m", sp).classify(frames).scene == "bathroom"
    [req] = chat.requests
    assert req.schema is not None and req.schema["properties"]["scene"]["enum"] == list(sp.labels)
    assert sum(not isinstance(b, str) for b in req.turns[0].blocks) == 2
    with pytest.raises(SceneUnavailableError):
        VlmScene(_Chat("", fail=True), "m", sp).classify(frames)
    assert FixedScene().classify(frames) == SceneResult("unknown", 1.0, "fixed")


def test_blur_boxes_mosaics_only_blur_regions() -> None:
    """정답: 블러 박스(100~160, 50~90) 안의 체커 무늬만 뭉개지고 박스 밖은 그대로다."""
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    img[::2, ::2] = 255  # 체커 무늬 (모자이크하면 회색으로 평균된다)
    kf = [{"t_ms": 1000, "x": 100.0, "y": 50.0, "w": 60.0, "h": 40.0}]
    blur = make_label(
        {"kind": "blur_track", "target": "face", "keyframes": kf},
        stream_id="bodycam",
        provenance=Provenance(source=Source.MODEL, model_version="iou-1"),
        confidence=0.5,
    )
    out = blur_boxes(img, 1100, [blur], tol_ms=300, cell_px=16)
    inside = out[50:90, 100:160]
    assert inside.std() < img[50:90, 100:160].std() / 2
    assert np.array_equal(out[100:, :], img[100:, :])
    # 첫 키프레임보다 tol 넘게 앞이면 바꾸지 않는다 (렌더도 블러 없음)
    assert np.array_equal(blur_boxes(img, 500, [blur], tol_ms=300, cell_px=16), img)
    # 마지막 키프레임 뒤는 렌더처럼 박스를 유지한다 (F-PAR-72)
    assert blur_boxes(img, 5000, [blur], tol_ms=300, cell_px=16)[50:90, 100:160].std() < 60


def test_blur_boxes_cover_render_between_far_keyframes() -> None:
    """정답: 키프레임 간격 2 s(허용 오차 300 ms의 두 배 넘음) 가운데 시각에 렌더가 그리는 보간 박스를
    모두 가린다. 예전에는 가장 가까운 키프레임만 써서 가운데 시각은 가리지 않았다 (F-PAR-72).
    끝 키프레임이 outside면 그 뒤는 가리지 않는다."""
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    img[::2, ::2] = 255
    kf = [
        {"t_ms": 1000, "x": 20.0, "y": 50.0, "w": 40.0, "h": 40.0},
        {"t_ms": 3000, "x": 220.0, "y": 50.0, "w": 40.0, "h": 40.0},
        {"t_ms": 4000, "x": 220.0, "y": 50.0, "w": 40.0, "h": 40.0, "outside": True},
    ]
    blur = make_label(
        {"kind": "blur_track", "target": "face", "keyframes": kf},
        stream_id="bodycam",
        provenance=Provenance(source=Source.MODEL, model_version="iou-1"),
        confidence=0.5,
        t_end_ms=4000,
    )
    out = blur_boxes(img, 2000, [blur], tol_ms=300, cell_px=16)
    # 렌더 보간 박스: x 120~160, y 50~90
    assert out[50:90, 120:160].std() < img[50:90, 120:160].std() / 2
    assert np.array_equal(blur_boxes(img, 4500, [blur], tol_ms=300, cell_px=16), img)


def test_scene_queries_and_owl_version() -> None:
    """부엌은 욕조·변기 솔·기저귀를 묻지 않고, unknown은 전체 질의다. 장면이 버전에 들어간다."""
    policy = load_policy(ROOT)
    op = policy.open_vocab_objects
    kitchen = queries_for("kitchen", op)
    assert {"bathtub", "toilet_brush", "diaper"}.isdisjoint(kitchen.values())
    assert queries_for("unknown", op) == dict(op.queries)
    owl = object.__new__(OwlObjects)
    owl.base_version = "owlv2-x+ty+pz"
    owl.set_scene(SceneResult("kitchen", 0.9, "vlm"))
    assert owl.version == "owlv2-x+ty+pz+sckitchen"
    assert sample_times(list(range(100)), 5) == [10, 29, 49, 69, 89]
