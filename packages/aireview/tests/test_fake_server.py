"""가짜 OpenAI 호환 VLM 서버(`dlp_aireview.testing.FakeOpenAIServer`) 테스트 (WP17.3).

벤치·점검·실행기 테스트가 이 서버를 실제 서버 대신 쓰므로, 흉내 내는 성질을 먼저 고정한다.
실제 HTTP(127.0.0.1)로 `dlp_models.vlm.OpenAICompatibleChat`이 보낸다. 정답 근거: 요청에 넣은
이미지 수·크기(토큰 = ⌈w/32⌉·⌈h/32⌉), 같은 요청을 다시 보내면 앞부분이 캐시 적중, 예약한 실패 순서.
"""

from __future__ import annotations

import json
import math
import threading
from typing import Any

import numpy as np
import pytest

from dlp_aireview.testing import FakeOpenAIServer, minimal_instance, text_tokens
from dlp_models.vlm import (
    ChatImage,
    ChatRequest,
    ChatTurn,
    OpenAICompatibleChat,
    VlmContextError,
    VlmRequestError,
    VlmUnavailableError,
    encode_jpeg,
)

SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answer", "n"],
    "properties": {"answer": {"enum": ["예", "아니오"]}, "n": {"type": ["integer", "null"]}},
}


def _image(w: int = 64, h: int = 32) -> ChatImage:
    """단색 시험 이미지."""
    return encode_jpeg(np.full((h, w, 3), 90, dtype=np.uint8), 85)


def _chat(server: FakeOpenAIServer, **kw: Any) -> OpenAICompatibleChat:
    """가짜 서버로 보내는 클라이언트 (정책 모델 이름 = 서버 모델)."""
    return OpenAICompatibleChat(server.url, server.model, timeout_s=5, **kw)


def _req(*blocks: ChatImage | str, schema: dict[str, Any] | None = None, **kw: Any) -> ChatRequest:
    """system + user 하나."""
    return ChatRequest("시스템", (ChatTurn("user", blocks),), schema=schema, **kw)


def test_models_version_tokenize() -> None:
    """/v1/models(모델 id·문맥 길이), /version, /tokenize. 끄면 404 → 클라이언트는 None."""
    with FakeOpenAIServer("m1", context_tokens=4096) as server:
        chat = _chat(server)
        assert chat.model_entries() == [{"id": "m1", "object": "model", "max_model_len": 4096}]
        assert chat.server_version() == "0.0.0-fake"
        assert chat.tokenize("가나다") == text_tokens("가나다") == 3
    with FakeOpenAIServer("m1", tokenize=False, version=None) as server:
        chat = _chat(server)
        assert chat.tokenize("가") is None
        assert chat.server_version() is None


def test_default_answers_follow_request_kind() -> None:
    """판정 요청은 모두 ok, 구조화 출력 강제면 스키마 최소 값, 강제하지 않으면 "모름"."""
    with FakeOpenAIServer("m1") as server:
        chat = _chat(server)
        verdict = chat.chat(_req("판정 대상:\n대상 #0~#2를 판정해 JSON 하나만 출력한다."))
        assert json.loads(verdict.content) == {"ok": [0, 1, 2], "items": [], "missing": []}
        forced = chat.chat(_req("모름이라고만 답하라", schema=SCHEMA))
        assert json.loads(forced.content) == {"answer": "예", "n": None}
        assert forced.model == "m1"
    with FakeOpenAIServer("m1", enforce_schema=False) as server:
        assert _chat(server).chat(_req("모름이라고만 답하라", schema=SCHEMA)).content == "모름"


def test_registered_answer_by_unit_header_and_responder() -> None:
    """`X-DLP-Unit` 헤더에 등록한 답 → 응답 함수 → 기본 답 순서."""
    with FakeOpenAIServer("m1") as server:
        server.answer("air-1", '{"ok": [], "items": [], "missing": []}')
        server.respond_with(lambda call: "응답 함수")
        chat = _chat(server)
        assert chat.chat(_req("x", headers={"X-DLP-Unit": "air-1"})).content.startswith('{"ok": []')
        assert chat.chat(_req("x", headers={"X-DLP-Unit": "air-2"})).content == "응답 함수"
        assert server.chat_calls[0].unit_id == "air-1"
        assert "X-DLP-Unit" not in json.dumps(server.chat_calls[0].body)


def test_usage_counts_images_text_and_prefix_cache() -> None:
    """프롬프트 토큰 = 글(바이트/3) + 이미지(⌈w/32⌉·⌈h/32⌉). 같은 요청을 다시 보내면 전부 캐시 적중,
    뒤쪽 글만 바꾸면 앞부분만 적중한다. report_cached=False면 cached_tokens가 없다."""
    image = _image(64, 40)  # ⌈64/32⌉·⌈40/32⌉ = 2·2 = 4
    with FakeOpenAIServer("m1") as server:
        chat = _chat(server)
        first = chat.chat(_req("고정", image, "가변1"))
        expected = text_tokens("시스템") + text_tokens("고정") + 4 + text_tokens("가변1")
        assert first.prompt_tokens == expected
        assert first.cached_tokens == 0
        again = chat.chat(_req("고정", image, "가변1"))
        assert again.cached_tokens == expected
        changed = chat.chat(_req("고정", image, "가변2"))
        assert changed.cached_tokens == expected - text_tokens("가변1")
    with FakeOpenAIServer("m1", report_cached=False) as server:
        reply = _chat(server).chat(_req("고정", image, "가변"))
        assert reply.prompt_tokens is not None and reply.cached_tokens is None


def test_completion_tokens_and_max_tokens_truncate() -> None:
    """생성 토큰 ≈ 답 글자 수 / 3. max_tokens보다 길면 그만큼 자른다 (첫 토큰 시간 측정용)."""
    with FakeOpenAIServer("m1") as server:
        server.respond_with(lambda call: "x" * 30)
        chat = _chat(server)
        assert chat.chat(_req("q")).completion_tokens == 10
        short = chat.chat(_req("q", max_tokens=2))
        assert short.completion_tokens == 2 and short.content == "x" * 6


def test_rejections_images_keywords_context_and_auth() -> None:
    """이미지 상한·범위 키워드·문맥 길이는 400(문맥은 VlmContextError), 키가 다르면 401."""
    full = dict(SCHEMA, properties={"answer": {"type": "string", "maxLength": 3}})
    with FakeOpenAIServer("m1", max_images=2, full_keywords=False, context_tokens=200) as server:
        chat = _chat(server)
        with pytest.raises(VlmRequestError, match="At most 2"):
            chat.chat(_req(_image(), _image(), _image(), "q"))
        with pytest.raises(VlmRequestError, match="Unsupported") as exc:
            chat.chat(_req("q", schema=full))
        assert not isinstance(exc.value, VlmContextError)
        assert chat.chat(_req("q", schema=SCHEMA)).content  # 범위 키워드가 없으면 받는다
        with pytest.raises(VlmContextError):
            chat.chat(_req("가" * 300, max_tokens=10))
    with FakeOpenAIServer("m1", api_key="k") as server:
        with pytest.raises(VlmRequestError) as denied:
            _chat(server, api_key="wrong").list_models()
        assert denied.value.status == 401
        assert _chat(server, api_key="k").list_models() == ["m1"]


def test_scheduled_failures_and_model_mismatch() -> None:
    """예약한 실패는 `after`번 성공 뒤 `times`번. 429는 Retry-After를 싣는다. served_model이 다르면
    응답 모델 대조(expected_model)가 설정 오류로 막는다."""
    with FakeOpenAIServer("m1") as server:
        server.fail(503, times=2, after=1)
        server.fail(429, retry_after="7")
        chat = _chat(server)
        chat.chat(_req("q"))
        for _ in range(2):
            with pytest.raises(VlmUnavailableError, match="503"):
                chat.chat(_req("q"))
        with pytest.raises(VlmUnavailableError) as limited:
            chat.chat(_req("q"))
        assert limited.value.retry_after_s == 7
        chat.chat(_req("q"))
    with (
        FakeOpenAIServer("m1", served_model="other") as server,
        pytest.raises(VlmRequestError, match="other"),
    ):
        _chat(server, expected_model="m1").chat(_req("q"))


def test_latency_and_in_flight_count() -> None:
    """지연을 두면 동시에 보낸 요청이 겹쳐 처리 중 최대 수가 오른다 (벤치 동시성 확인용)."""
    with FakeOpenAIServer("m1", latency_s=0.2) as server:
        chat = _chat(server)
        threads = [threading.Thread(target=chat.chat, args=(_req("q"),)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert server.max_in_flight >= 2
        assert len(server.chat_calls) == 4


def test_minimal_instance_satisfies_subset_schema() -> None:
    """최소 값: enum 첫 값, anyOf 첫 갈래, 타입 목록의 null, 객체는 required만."""
    schema = {
        "type": "object",
        "required": ["a", "b", "c", "d"],
        "properties": {
            "a": {"enum": [3, 4]},
            "b": {"anyOf": [{"type": "null"}, {"type": "object"}]},
            "c": {"type": ["integer", "null"]},
            "d": {"type": "array", "items": {"type": "integer"}},
            "e": {"type": "string"},
        },
    }
    assert minimal_instance(schema) == {"a": 3, "b": None, "c": None, "d": []}
    assert math.isclose(minimal_instance({"type": "number"}), 0)
