"""잘라 낸 박스 AI 확인(`dlp_aireview.crops`) 시험 (ADR 0056).

정답: 합성 프레임(박스 위치를 아는 단색 화면)과 가짜 사내 VLM(`FakeOpenAIServer`, 127.0.0.1)이 내는
정해진 답. 판정은 힌트용 의심 정도로만 바뀐다 (라벨·검수 기록은 만들지 않는다).
"""

from __future__ import annotations

import json

import numpy as np

from dlp_aireview.crops import (
    CropAsk,
    OracleCropVerifier,
    VlmCropVerifier,
    crop_with_margin,
    parse_verdict,
    prompt_for,
    to_doubts,
    verify_crops,
)
from dlp_aireview.media import BlurredFrame
from dlp_aireview.testing import FakeCall, FakeOpenAIServer
from dlp_models.vlm import ChatReply, ChatRequest, OpenAICompatibleChat


def _frame(index: int) -> BlurredFrame:
    """640x480 회색 프레임."""
    return BlurredFrame("bodycam", index, index * 33, np.full((480, 640, 3), 128, dtype=np.uint8))


def test_crop_with_margin_and_prompts() -> None:
    """박스(100, 100, 50, 40)를 각 변 100% 넓히면 150x120, 긴 변 상한으로 줄이고 화면 안으로 자른다.

    블러 질문은 가운데가 모자이크라고 알린다.
    """
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    assert crop_with_margin(rgb, (100, 100, 50, 40), 1.0, 1000).shape[:2] == (121, 151)
    assert max(crop_with_margin(rgb, (100, 100, 50, 40), 1.0, 64).shape[:2]) == 64
    assert crop_with_margin(rgb, (600, 450, 80, 80), 1.0, 1000).shape[:2] == (110, 120)
    face = CropAsk("f1", "privacy", "face", 0, (1, 1, 1, 1))
    assert "pixelated" in prompt_for(face) and "face" in prompt_for(face)
    assert "pixelated" not in prompt_for(CropAsk("t1", "object", "toilet_brush", 0, (1, 1, 1, 1)))
    assert parse_verdict("x", "oops").answer == "unsure"
    assert parse_verdict("x", '{"answer": "no", "confidence": 0.9}').answer == "no"


def test_verify_crops_with_fake_inhouse_server() -> None:
    """가짜 사내 서버가 얼굴 f1에 no(0.9), 도구 t1에 yes(0.8)로 답한다 → 의심은 f1만 (0.9).

    요청마다 조각 이미지 한 장과 응답 JSON Schema가 실린다. 장애는 unsure로 둔다.
    """

    def respond(call: FakeCall) -> str:
        text = json.dumps(call.body)
        return json.dumps(
            {"answer": "no", "confidence": 0.9}
            if "pixelated" in text
            else {"answer": "yes", "confidence": 0.8}
        )

    asks = [
        CropAsk("f1", "privacy", "face", 0, (100.0, 100.0, 40.0, 40.0)),
        CropAsk("t1", "object", "mop", 1, (300.0, 200.0, 60.0, 90.0)),
    ]
    with FakeOpenAIServer("m") as server:
        server.respond_with(respond)
        chat = OpenAICompatibleChat(server.url, "m", timeout_s=5, expected_model="m")
        verdicts = verify_crops(
            asks, [_frame(0), _frame(1)], VlmCropVerifier(chat, "m", 85), margin=1.0, max_side=256
        )
        chats = [c for c in server.calls if c.path == "/v1/chat/completions"]
    assert [(v.label_id, v.answer) for v in verdicts] == [("f1", "no"), ("t1", "yes")]
    assert len(chats) == 2
    content = chats[0].body["messages"][0]["content"]
    assert sum(part["type"] == "image_url" for part in content) == 1
    assert chats[0].body["response_format"]["json_schema"]["name"] == "crop_check"
    assert to_doubts(verdicts, 0.7) == {"f1": 0.9}
    assert to_doubts(verdicts, 0.95) == {}


def test_oracle_stub_and_outage_make_no_hints() -> None:
    """CPU stub은 늘 unsure, 서버 장애도 unsure → 힌트 없음."""

    class Down:
        """늘 장애인 채팅."""

        def chat(self, req: ChatRequest) -> ChatReply:
            """장애를 낸다."""
            raise RuntimeError("down")

    ask = CropAsk("f1", "privacy", "face", 0, (10.0, 10.0, 20.0, 20.0))
    oracle = verify_crops([ask], [_frame(0)], OracleCropVerifier(), margin=1.0, max_side=64)
    down = verify_crops(
        [ask],
        [_frame(0)],
        VlmCropVerifier(Down(), "m", 85),
        margin=1.0,
        max_side=64,
    )
    assert [v.answer for v in oracle + down] == ["unsure", "unsure"]
    assert to_doubts(oracle + down, 0.0) == {}
