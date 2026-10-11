"""OpenAI 호환 VLM 공통 전송(`dlp_models.vlm`) 테스트 (WP17.1).

서버는 `httpx.MockTransport`로 흉내 낸다 (실제 서버·네트워크 없음). 상태 코드별 오류 분류, 요청 본문
모양(data URI, response_format), 응답 모델 대조, `usage` 파싱, 백오프·재시도 규칙을 본다.
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Callable, Sequence
from typing import Any

import httpx
import numpy as np
import pytest

from dlp_models.vlm import (
    ChatImage,
    ChatRequest,
    ChatTurn,
    OpenAICompatibleChat,
    VlmContextError,
    VlmRequestError,
    VlmUnavailableError,
    ask_validated,
    chat_body,
    encode_jpeg,
    parse_reply,
    with_backoff,
)

Handler = Callable[[httpx.Request], httpx.Response]


def _ok(content: str = "{}", model: str = "m", usage: dict[str, Any] | None = None) -> Any:
    """정상 응답 JSON."""
    data: dict[str, Any] = {"model": model, "choices": [{"message": {"content": content}}]}
    if usage is not None:
        data["usage"] = usage
    return data


def _client(handler: Handler, **kw: Any) -> OpenAICompatibleChat:
    """MockTransport를 쓰는 클라이언트."""
    return OpenAICompatibleChat(
        "http://vlm", "m", timeout_s=5, transport=httpx.MockTransport(handler), **kw
    )


def _request(**kw: Any) -> ChatRequest:
    """이미지 하나 + 글 하나의 최소 요청."""
    data: dict[str, Any] = {
        "system": None,
        "turns": (ChatTurn("user", (ChatImage.of(b"jpg"), "질문")),),
        "schema": {"type": "object"},
        "schema_name": "answer",
    }
    data.update(kw)
    return ChatRequest(**data)


def test_chat_body_shape_with_data_uri_and_schema() -> None:
    """본문: model·temperature 0·messages(이미지 data URI → 글)·response_format(json_schema strict).

    system이 없으면 system 메시지를 넣지 않고, max_tokens가 없으면 키를 넣지 않는다.
    """
    jpeg = b"\xff\xd8abc"
    body = chat_body("m", _request(turns=(ChatTurn("user", (ChatImage.of(jpeg), "질문")),)))
    assert list(body) == ["model", "temperature", "messages", "response_format"]
    [message] = body["messages"]
    assert message["role"] == "user"
    image, text = message["content"]
    assert image == {
        "type": "image_url",
        "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()},
    }
    assert text == {"type": "text", "text": "질문"}
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "answer", "schema": {"type": "object"}, "strict": True},
    }


def test_chat_body_system_turns_max_tokens_and_prompt_enforcement() -> None:
    """system·여러 턴(assistant는 글만)·max_tokens를 순서대로 넣는다. 스키마를 프롬프트로 강제하면
    response_format을 넣지 않는다 (스키마는 호출자가 글에 싣는다)."""
    req = _request(
        system="역할",
        turns=(
            ChatTurn("user", ("고정",)),
            ChatTurn("user", ("가변",)),
            ChatTurn("assistant", ("앞 응답",)),
            ChatTurn("user", ("다시",)),
        ),
        max_tokens=64,
        schema_enforcement="prompt",
    )
    body = chat_body("m", req)
    assert list(body) == ["model", "temperature", "max_tokens", "messages"]
    roles = [m["role"] for m in body["messages"]]
    assert roles == ["system", "user", "user", "assistant", "user"]
    assert body["messages"][0] == {"role": "system", "content": "역할"}
    assert body["messages"][3] == {"role": "assistant", "content": "앞 응답"}
    with pytest.raises(ValueError, match="assistant"):
        ChatTurn("assistant", (ChatImage.of(b"x"),))


def test_chat_image_hash_and_jpeg_is_deterministic() -> None:
    """같은 RGB 입력은 같은 JPEG 바이트(접두사 캐시가 맞으려면 필요), sha256은 바이트의 해시."""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, size=(32, 48, 3), dtype=np.uint8)
    a, b = encode_jpeg(img, 85), encode_jpeg(img.copy(), 85)
    assert a.jpeg == b.jpeg and a.jpeg[:2] == b"\xff\xd8"
    assert a.sha256 == hashlib.sha256(a.jpeg).hexdigest() == b.sha256
    assert encode_jpeg(img, 50).jpeg != a.jpeg


@pytest.mark.parametrize(
    ("status", "text", "error"),
    [
        (500, "boom", VlmUnavailableError),
        (503, "busy", VlmUnavailableError),
        (429, "slow down", VlmUnavailableError),
        (408, "timeout", VlmUnavailableError),
        (400, "This model's maximum context length is 16384 tokens", VlmContextError),
        (413, "prompt is too long", VlmContextError),
        (400, "invalid json schema", VlmRequestError),
        (413, "Request Entity Too Large", VlmRequestError),
        (401, "unauthorized", VlmRequestError),
        (403, "forbidden", VlmRequestError),
        (404, "model not found", VlmRequestError),
        (422, "unprocessable", VlmRequestError),
    ],
)
def test_status_classification(status: int, text: str, error: type[Exception]) -> None:
    """5xx·429·408은 재시도 대상(장애), 400·413 중 문맥 초과는 VlmContextError, 그 밖 4xx는 설정
    오류(VlmRequestError, 재시도 없음)."""
    client = _client(lambda req: httpx.Response(status, text=text))
    with pytest.raises(error) as info:
        client.chat(_request())
    assert type(info.value) is error
    if error is not VlmUnavailableError:
        assert isinstance(info.value, VlmRequestError) and info.value.status == status


def test_retry_after_and_timeouts_and_bad_body_are_unavailable() -> None:
    """429의 Retry-After 초를 담는다. 시간 초과·연결 오류·JSON 아님·형식 오류도 장애다."""
    client = _client(lambda req: httpx.Response(429, headers={"Retry-After": "7"}))
    with pytest.raises(VlmUnavailableError) as info:
        client.chat(_request())
    assert info.value.retry_after_s == 7.0

    def timeout(req: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=req)

    def refused(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=req)

    def body(**kw: Any) -> Handler:
        return lambda req: httpx.Response(200, **kw)

    handlers: list[Handler] = [
        timeout,
        refused,
        body(text="not json"),
        body(json={"choices": []}),
        body(json=[1, 2]),
    ]
    for handler in handlers:
        with pytest.raises(VlmUnavailableError):
            _client(handler).chat(_request())


def test_reply_parses_usage_and_checks_model() -> None:
    """usage(prompt·cached·completion 토큰)를 읽는다. expected_model이 있으면 응답 model이 달라도·
    없어도 VlmRequestError다 (사내 서버가 정책 모델을 띄웠는지 확인, I7)."""
    usage = {
        "prompt_tokens": 100,
        "completion_tokens": 9,
        "prompt_tokens_details": {"cached_tokens": 64},
    }
    reply = parse_reply(_ok("답", usage=usage), None, latency_ms=12)
    assert (reply.content, reply.model, reply.prompt_tokens) == ("답", "m", 100)
    assert (reply.cached_tokens, reply.completion_tokens, reply.latency_ms) == (64, 9, 12)
    bare = parse_reply(_ok(), None, latency_ms=0)
    assert bare.prompt_tokens is None and bare.cached_tokens is None
    assert parse_reply(_ok(model="m"), "m", latency_ms=0).model == "m"
    with pytest.raises(VlmRequestError, match="other"):
        parse_reply(_ok(model="other"), "m", latency_ms=0)
    data = _ok()
    del data["model"]
    with pytest.raises(VlmRequestError):
        parse_reply(data, "m", latency_ms=0)
    assert parse_reply(data, None, latency_ms=0).model is None


def test_client_posts_body_headers_and_api_key() -> None:
    """요청 본문은 chat_body와 같고, 요청 헤더(X-DLP-Unit 등)는 본문이 아니라 HTTP 헤더로 간다.
    API 키는 Authorization 헤더로만 보낸다. 응답 모델이 다르면 설정 오류다."""
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, json=_ok("{}", model="served"))

    client = _client(handler, api_key="secret", expected_model="served")
    req = _request(headers={"X-DLP-Unit": "air-1"})
    reply = client.chat(req)
    assert reply.content == "{}" and reply.latency_ms >= 0
    [sent] = seen
    assert sent.url.path == "/v1/chat/completions"
    assert json.loads(sent.content) == chat_body("m", req)
    assert sent.headers["X-DLP-Unit"] == "air-1"
    assert sent.headers["Authorization"] == "Bearer secret"
    assert b"air-1" not in sent.content and b"secret" not in sent.content
    with pytest.raises(VlmRequestError):
        _client(handler, expected_model="m").chat(req)


def test_list_models_and_tokenize() -> None:
    """/v1/models의 id 목록, /tokenize의 토큰 수 (없는 서버면 None)."""

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "a"}, {"id": "b"}]})
        if req.url.path == "/tokenize":
            body = json.loads(req.content)
            return httpx.Response(200, json={"count": len(body["prompt"])})
        return httpx.Response(404)

    client = _client(handler)
    assert client.list_models() == ["a", "b"]
    assert client.tokenize("abcd") == 4
    assert _client(lambda req: httpx.Response(404)).tokenize("x") is None
    with pytest.raises(VlmUnavailableError):
        _client(lambda req: httpx.Response(503)).list_models()


def test_with_backoff_retries_only_unavailable() -> None:
    """장애만 backoff 대기마다 다시 부른다 (시도 = len + 1). Retry-After가 더 길면 그만큼 기다리되
    backoff 최댓값을 넘지 않는다 (큰 Retry-After에 오래 멈추지 않게). 설정 오류(4xx)는 즉시
    올린다."""
    waits: list[float] = []
    errors: list[str] = []
    calls: list[int] = []

    def flaky() -> str:
        calls.append(1)
        if len(calls) == 1:
            raise VlmUnavailableError("busy", retry_after_s=3.0)
        if len(calls) == 2:
            raise VlmUnavailableError("limited", retry_after_s=3600.0)
        if len(calls) == 3:
            raise VlmUnavailableError("down")
        return "ok"

    out = with_backoff(
        flaky, (1.0, 5.0, 2.0), waits.append, on_error=lambda e: errors.append(str(e))
    )
    assert out == "ok" and waits == [3.0, 5.0, 2.0] and errors == ["busy", "limited", "down"]

    def always() -> str:
        raise VlmUnavailableError("down")

    waits.clear()
    with pytest.raises(VlmUnavailableError):
        with_backoff(always, (1.0, 5.0), waits.append)
    assert waits == [1.0, 5.0]

    def config() -> str:
        calls.append(1)
        raise VlmRequestError("401", status=401)

    calls.clear()
    with pytest.raises(VlmRequestError):
        with_backoff(config, (1.0, 5.0), waits.append)
    assert len(calls) == 1


def test_ask_validated_retries_with_last_violations() -> None:
    """검증에 실패하면 직전 위반 목록을 붙여 다시 묻는다 (최대 max_retries번).

    위반은 errors에 쌓인다.
    """
    asked: list[tuple[str, ...]] = []
    answers = iter(["bad", "worse", "good"])

    def ask(problems: Sequence[str]) -> str:
        asked.append(tuple(problems))
        return next(answers)

    def validate(raw: str) -> tuple[str | None, list[str]]:
        return (raw, []) if raw == "good" else (None, [f"{raw}!"])

    errors: list[str] = []
    value, attempts = ask_validated(ask, validate, 2, errors)
    assert (value, attempts) == ("good", 3)
    assert asked == [(), ("bad!",), ("worse!",)]
    assert errors == ["bad!", "worse!"]

    errors.clear()
    answers = iter(["bad", "bad"])
    value, attempts = ask_validated(ask, validate, 1, errors)
    assert value is None and attempts == 2 and errors == ["bad!", "bad!"]
