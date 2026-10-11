"""OpenAI 호환 VLM 서버 공통 전송 (WP17.1, ADR 0040).

행동 분류(`dlp_actions.clients`)와 AI 1차 검수(`dlp_aireview.backend`)가 같은 요청 형식·오류 분류·
재시도 규칙을 쓰도록 한곳에 모은다. 이 모듈은 라벨·DB·저장소를 다루지 않는다. 입력은 이미 고른
블러본 프레임(JPEG 바이트)과 글이다.

오류 분류 (서버 응답 → 예외)
    - `VlmUnavailableError`: 연결 오류·시간 초과·5xx·408·429(Retry-After, 정책 대기 상한까지)·응답
      형식 오류. 서버가 잠시 응답하지 못한 것이므로 백오프하며 다시 묻는다(`with_backoff`). 결과로
      기록하지 않는다.
    - `VlmRequestError`: 그 밖의 4xx(400·401·403·404·413·422 등)와 응답 모델 불일치. 설정 오류라
      다시 물어도 같으므로 즉시 멈춘다. 행동 단계는 종료 코드 5로 끝난다.
    - `VlmContextError(VlmRequestError)`: 400·413 중 본문이 문맥 길이 초과를 말하는 경우
      ("context", "too long", 또는 "length"와 "token"이 함께 있음). 호출자가 요청을 줄여 다시 물을
      수 있다.

주요 이름
    - 자료형: `ChatImage`, `ChatTurn`, `ChatRequest`, `ChatReply`.
    - 순수 함수: `chat_body`(요청 본문), `parse_reply`(응답 해석), `encode_jpeg`(RGB → JPEG).
    - 재시도: `with_backoff`(장애만), `ask_validated`(응답 위반 → 위반 목록을 붙여 다시 묻기).
    - 클라이언트: `OpenAICompatibleChat` (`/v1/chat/completions`, `/v1/models`, `/tokenize`).

주의
    - 요청 본문은 `httpx`의 `json=` 직렬화를 그대로 쓴다. 행동 단계의 요청 바이트가 리팩토링 전과
      같아야 하므로(`packages/actions/tests/test_vlm_body.py`) 키 순서·직렬화 방식을 바꾸지 않는다.
    - 이미지는 data URI(base64)로만 보낸다. 서명 URL을 쓰지 않는다.
    - API 키는 Authorization 헤더로만 보내고 출력·기록하지 않는다. 추적용 헤더(`X-DLP-Unit` 등)는
      본문에 넣지 않으므로 서버 접두사 캐시에 영향이 없다.
"""

from __future__ import annotations

import base64
import hashlib
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import cv2
import httpx
import numpy as np
from numpy.typing import NDArray

# 문맥 길이 초과로 보는 400·413 응답 본문의 단서 (소문자 비교)
_CONTEXT_HINTS = ("context", "too long")
# 오류 메시지에 남길 응답 본문 앞부분 길이 (요청 본문·키는 담지 않는다)
_SNIPPET = 200


class VlmUnavailableError(RuntimeError):
    """VLM 서버 장애 (연결 오류·시간 초과·5xx·408·429·응답 형식 오류). 백오프하며 다시 묻는다.

    Attributes:
        retry_after_s: 서버가 알려 준 대기 초 (429의 Retry-After). 없으면 None.
    """

    def __init__(self, message: str, *, retry_after_s: float | None = None) -> None:
        """메시지와 서버가 알려 준 대기 초."""
        super().__init__(message)
        self.retry_after_s = retry_after_s


class VlmRequestError(RuntimeError):
    """설정 오류 (재시도하지 않는 4xx, 응답 모델 불일치). 다시 물어도 같으므로 즉시 멈춘다.

    Attributes:
        status: HTTP 상태 코드 (모델 불일치처럼 HTTP 오류가 아니면 None).
    """

    def __init__(self, message: str, *, status: int | None = None) -> None:
        """메시지와 HTTP 상태 코드."""
        super().__init__(message)
        self.status = status


class VlmContextError(VlmRequestError):
    """요청이 서버 문맥 길이를 넘었다 (400·413). 호출자가 프레임·대상을 줄여 다시 물을 수 있다."""


@dataclass(frozen=True)
class ChatImage:
    """보낼 이미지 하나 (JPEG 바이트와 그 sha256). `of`로 만든다.

    Attributes:
        jpeg: JPEG 바이트.
        sha256: `jpeg`의 sha256 16진수 (감사 기록·프롬프트 다이제스트용).
    """

    jpeg: bytes = field(repr=False)
    sha256: str

    @classmethod
    def of(cls, jpeg: bytes) -> ChatImage:
        """JPEG 바이트에서 해시를 계산해 만든다."""
        return cls(jpeg, hashlib.sha256(jpeg).hexdigest())

    def data_uri(self) -> str:
        """`data:image/jpeg;base64,...` 형식 (요청 본문에 그대로 들어간다)."""
        return "data:image/jpeg;base64," + base64.b64encode(self.jpeg).decode()


@dataclass(frozen=True)
class ChatTurn:
    """대화 한 턴. user는 이미지·글을 섞어 순서대로, assistant는 글만 담는다.

    Attributes:
        role: "user" 또는 "assistant" (system은 `ChatRequest.system`).
        blocks: 이미지(`ChatImage`) 또는 글(str)의 순서 있는 목록.

    Raises:
        ValueError: assistant 턴에 이미지가 있을 때.
    """

    role: Literal["user", "assistant"]
    blocks: tuple[ChatImage | str, ...]

    def __post_init__(self) -> None:
        """assistant 턴은 글만 허용한다."""
        if self.role == "assistant" and any(isinstance(b, ChatImage) for b in self.blocks):
            raise ValueError("assistant 턴에는 이미지를 넣지 않습니다")


@dataclass(frozen=True)
class ChatRequest:
    """채팅 요청 하나.

    Attributes:
        system: system 메시지 (None이면 넣지 않는다).
        turns: 대화 턴 (보통 user 하나, 재시도면 assistant·user가 끝에 붙는다).
        schema: 응답 JSON Schema (None이면 구조화 출력을 요구하지 않는다).
        schema_name: `response_format.json_schema.name`.
        max_tokens: 생성 토큰 상한 (None이면 키를 넣지 않는다).
        schema_enforcement: "server"면 `response_format`으로 서버가 강제한다. "prompt"면 넣지 않는다
            (스키마는 호출자가 글에 싣는다). 어느 쪽이든 응답 재검사는 호출자가 한다.
        headers: 추가 HTTP 헤더 (추적용 `X-DLP-Unit` 등). 본문에 들어가지 않는다.
    """

    system: str | None
    turns: tuple[ChatTurn, ...]
    schema: Mapping[str, Any] | None = None
    schema_name: str = "answer"
    max_tokens: int | None = None
    schema_enforcement: Literal["server", "prompt"] = "server"
    headers: Mapping[str, str] = field(default_factory=dict[str, str])


@dataclass(frozen=True)
class ChatReply:
    """채팅 응답 하나.

    Attributes:
        content: 첫 선택지의 메시지 내용 (문자열로 바꾼 값. null이면 "None").
        model: 응답의 `model` (서버가 보고한 모델 이름, 없으면 None).
        prompt_tokens, cached_tokens, completion_tokens: `usage` 값 (서버가 주지 않으면 None).
        latency_ms: 요청 왕복 시간 ms.
    """

    content: str
    model: str | None
    prompt_tokens: int | None
    cached_tokens: int | None
    completion_tokens: int | None
    latency_ms: int


def _content_part(block: ChatImage | str) -> dict[str, Any]:
    """user 턴의 내용 조각 하나 (이미지는 data URI, 글은 text)."""
    if isinstance(block, ChatImage):
        return {"type": "image_url", "image_url": {"url": block.data_uri()}}
    return {"type": "text", "text": block}


def _message(turn: ChatTurn) -> dict[str, Any]:
    """턴 → 메시지. assistant는 글을 이어 붙인 문자열 내용이다."""
    if turn.role == "assistant":
        return {"role": "assistant", "content": "".join(str(b) for b in turn.blocks)}
    return {"role": "user", "content": [_content_part(b) for b in turn.blocks]}


def chat_body(model: str, req: ChatRequest) -> dict[str, Any]:
    """`/v1/chat/completions` 요청 본문 (순수 함수).

    키 순서: model, temperature(0), max_tokens(있으면), messages, response_format(서버 강제면).
    행동 단계의 요청 바이트가 이 순서에 고정돼 있다 (`test_vlm_body.py`).

    Args:
        model: 서버에 보낼 모델 이름.
        req: 요청.

    Returns:
        JSON으로 보낼 사전.
    """
    body: dict[str, Any] = {"model": model, "temperature": 0}
    if req.max_tokens is not None:
        body["max_tokens"] = req.max_tokens
    messages: list[dict[str, Any]] = []
    if req.system is not None:
        messages.append({"role": "system", "content": req.system})
    messages += [_message(t) for t in req.turns]
    body["messages"] = messages
    if req.schema is not None and req.schema_enforcement == "server":
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": req.schema_name, "schema": req.schema, "strict": True},
        }
    return body


def _field(data: Any, key: str) -> Any:
    """JSON 객체의 키 값 (객체가 아니거나 키가 없으면 None)."""
    if not isinstance(data, dict):
        return None
    obj: dict[str, Any] = data  # pyright: ignore[reportUnknownVariableType]
    return obj.get(key)


def _int_or_none(value: Any) -> int | None:
    """정수면 그 값, 아니면 None (usage 필드가 없거나 형식이 다른 서버)."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def parse_reply(data: Any, expected_model: str | None, *, latency_ms: int) -> ChatReply:
    """응답 JSON을 해석한다.

    Args:
        data: 응답 본문 JSON.
        expected_model: 응답 `model`이 이 값이어야 한다 (None이면 보지 않는다). 사내 서버가 정책
            모델을 띄웠는지 확인한다. 응답에 `model`이 없어도 불일치로 본다.
        latency_ms: 요청 왕복 시간 ms.

    Raises:
        VlmUnavailableError: 선택지·메시지가 없는 등 응답 형식 오류.
        VlmRequestError: 응답 모델이 `expected_model`과 다를 때.
    """
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise VlmUnavailableError(f"응답 형식 오류: {type(exc).__name__}: {exc}") from exc
    served = _field(data, "model")
    served = served if isinstance(served, str) else None
    if expected_model is not None and served != expected_model:
        raise VlmRequestError(f"서버 모델 {served}이(가) 정책 모델 {expected_model}과 다릅니다")
    usage: Any = _field(data, "usage")
    details: Any = _field(usage, "prompt_tokens_details")
    return ChatReply(
        content=str(content),
        model=served,
        prompt_tokens=_int_or_none(_field(usage, "prompt_tokens")),
        cached_tokens=_int_or_none(_field(details, "cached_tokens")),
        completion_tokens=_int_or_none(_field(usage, "completion_tokens")),
        latency_ms=latency_ms,
    )


def encode_jpeg(rgb: NDArray[np.uint8], quality: int) -> ChatImage:
    """RGB 영상 → JPEG (OpenCV libjpeg, 품질 `quality`).

    같은 입력이면 같은 바이트다 (서버 이미지·접두사 캐시와 프롬프트 다이제스트가 맞으려면 필요).

    Raises:
        ValueError: 인코딩 실패.
    """
    bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise ValueError("JPEG 인코딩 실패")
    return ChatImage.of(buf.tobytes())


def _retry_after(value: str | None) -> float | None:
    """Retry-After 헤더(초 정수) → 초. 날짜 형식이나 잘못된 값이면 None."""
    try:
        return max(0.0, float(value)) if value is not None else None
    except ValueError:
        return None


def http_error(status: int, text: str, retry_after: str | None = None) -> Exception:
    """HTTP 오류 상태 → 예외 (모듈 docstring의 분류).

    Args:
        status: 400 이상 상태 코드.
        text: 응답 본문 (앞부분만 메시지에 남긴다).
        retry_after: Retry-After 헤더 값.
    """
    snippet = text[:_SNIPPET].replace("\n", " ")
    if status >= 500 or status in (408, 429):
        wait = _retry_after(retry_after) if status == 429 else None
        return VlmUnavailableError(f"HTTP {status}: {snippet}", retry_after_s=wait)
    low = text.lower()
    if status in (400, 413) and (
        any(h in low for h in _CONTEXT_HINTS) or ("length" in low and "token" in low)
    ):
        return VlmContextError(f"HTTP {status} 문맥 길이 초과: {snippet}", status=status)
    return VlmRequestError(f"HTTP {status}: {snippet}", status=status)


def with_backoff[T](
    call: Callable[[], T],
    backoff_s: Sequence[float],
    sleep: Callable[[float], None] = time.sleep,
    on_error: Callable[[VlmUnavailableError], None] | None = None,
) -> T:
    """장애(`VlmUnavailableError`)면 `backoff_s`의 대기마다 다시 부른다. 끝내 실패하면 올린다.

    시도 횟수 = len(backoff_s) + 1. 대기는 max(정책 대기, 서버 Retry-After)이되 `backoff_s`의
    최댓값을 넘지 않는다 (프록시·레이트 리미터가 큰 Retry-After를 주어도 정책 대기 상한 안에서
    멈춘다). 설정 오류 (`VlmRequestError`)는 다시 부르지 않고 바로 올린다.

    Args:
        call: 부를 함수.
        backoff_s: 실패 뒤 대기 초 목록 (최댓값이 대기 상한).
        sleep: 대기 함수 (테스트에서 바꾼다).
        on_error: 실패할 때마다 부를 함수 (오류 기록용).
    """
    cap = max(backoff_s, default=0.0)
    for wait in backoff_s:
        try:
            return call()
        except VlmUnavailableError as exc:
            if on_error is not None:
                on_error(exc)
            sleep(min(max(wait, exc.retry_after_s or 0.0), cap))
    return call()


def ask_validated[T](
    ask: Callable[[Sequence[str]], str],
    validate: Callable[[str], tuple[T | None, list[str]]],
    max_retries: int,
    errors: list[str],
) -> tuple[T | None, int]:
    """응답을 검증하고, 위반이면 직전 위반 목록을 붙여 `max_retries`번까지 다시 묻는다.

    Args:
        ask: 위반 목록(첫 시도는 빈 목록)을 받아 원문 응답을 돌려준다. 재시도 프롬프트는 이 함수가
            만든다.
        validate: 원문 → (검증된 값 또는 None, 위반 목록).
        max_retries: 재시도 횟수 (총 시도 = max_retries + 1).
        errors: 위반을 이어 붙일 목록 (호출자의 오류 기록).

    Returns:
        (검증된 값 또는 None, 시도 횟수). 끝까지 위반이면 값이 None이다.
    """
    problems: list[str] = []
    for attempt in range(1, max_retries + 2):
        value, problems = validate(ask(problems))
        if value is not None:
            return value, attempt
        errors.extend(problems)
    return None, max_retries + 1


class OpenAICompatibleChat:
    """OpenAI 호환 채팅 API 클라이언트 (vLLM·SGLang·llama.cpp 등). 동기식이며 스레드마다 쓸 수 있다.

    `httpx.Client`는 스레드 안전하므로 한 인스턴스를 여러 스레드가 함께 써도 된다.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout_s: float,
        api_key: str | None = None,
        expected_model: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """HTTP 클라이언트를 만든다 (연결은 첫 요청 때).

        Args:
            base_url: 서버 주소 (예: `http://10.0.0.11:8000`).
            model: 요청 본문의 model.
            timeout_s: 요청 시간 초과 초.
            api_key: 있으면 `Authorization: Bearer` 헤더로만 보낸다.
            expected_model: 응답 model이 이 값이어야 한다 (None이면 보지 않는다).
            transport: httpx 전송 계층 (테스트에서 `MockTransport`).
        """
        self.model = model
        self.expected_model = expected_model
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.http = httpx.Client(
            base_url=base_url, timeout=timeout_s, transport=transport, headers=headers
        )

    def _send(self, method: str, path: str, **kw: Any) -> httpx.Response:
        """요청을 보내고 오류 상태·전송 오류를 분류해 올린다."""
        try:
            resp = self.http.request(method, path, **kw)
        except httpx.HTTPError as exc:
            raise VlmUnavailableError(f"{type(exc).__name__}: {exc}") from exc
        if resp.status_code >= 400:
            raise http_error(resp.status_code, resp.text, resp.headers.get("Retry-After"))
        return resp

    def chat(self, req: ChatRequest) -> ChatReply:
        """요청을 보내고 응답을 해석한다. 외부 서비스 호출이다.

        Raises:
            VlmUnavailableError: 장애 (다시 물을 수 있다).
            VlmRequestError: 설정 오류·응답 모델 불일치 (`VlmContextError` 포함).
        """
        start = time.monotonic()
        resp = self._send(
            "POST",
            "/v1/chat/completions",
            json=chat_body(self.model, req),
            headers=dict(req.headers),
        )
        latency = round((time.monotonic() - start) * 1000)
        try:
            data: Any = resp.json()
        except ValueError as exc:
            raise VlmUnavailableError(f"응답이 JSON이 아닙니다: {exc}") from exc
        return parse_reply(data, self.expected_model, latency_ms=latency)

    def list_models(self) -> list[str]:
        """`/v1/models`의 모델 id 목록.

        Raises:
            VlmUnavailableError: 장애·형식 오류. VlmRequestError: 4xx.
        """
        resp = self._send("GET", "/v1/models")
        try:
            data: Any = resp.json()
            return [str(m["id"]) for m in data["data"]]
        except (ValueError, KeyError, TypeError) as exc:
            raise VlmUnavailableError(f"/v1/models 형식 오류: {exc}") from exc

    def tokenize(self, text: str) -> int | None:
        """서버 `/tokenize`(vLLM)로 글의 토큰 수를 센다. 서버에 없으면(404 등 4xx) None.

        Raises:
            VlmUnavailableError: 장애.
        """
        try:
            resp = self._send("POST", "/tokenize", json={"model": self.model, "prompt": text})
        except VlmRequestError:
            return None
        try:
            data: Any = resp.json()
        except ValueError:
            return None
        return _int_or_none(_field(data, "count"))
