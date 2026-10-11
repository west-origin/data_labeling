"""가짜 OpenAI 호환 VLM 서버 (테스트·CI용, WP17.3). 실제 모델을 부르지 않는다.

`FakeOpenAIServer`는 127.0.0.1의 임의 포트에 `ThreadingHTTPServer`를 띄우고, vLLM이 내는 모양의 응답을
흉내 낸다. 벤치(`bench`)·점검(`probe`)·실행기(WP17.5) 테스트와 CLI 종단 테스트가 실제 서버 대신 쓴다.

흉내 내는 것
    - `GET /v1/models`: 모델 id와 `max_model_len`. `GET /version`: 엔진 버전.
    - `POST /v1/chat/completions`:
      - 답: 헤더 `X-DLP-Unit`에 등록한 답(`answer`) → 등록한 응답 함수(`respond_with`) → 기본 답.
        기본 답은 판정 요청이면(가변 글의 "대상 #0~#N") 모두 ok, 그 밖에 구조화 출력을 강제하면
        스키마를 만족하는 최소 값, 강제하지 않으면 지시를 따른 맨 글("모름")이다.
      - 지연(`latency_s`), 예약한 실패(`fail`: 503·429·400 등), 인증(`api_key`, 401), 요청당 이미지
        상한(`max_images`, 400), 문맥 길이(`context_tokens`, 400 "maximum context length"),
        구조화 출력의 범위·길이 키워드 거부(`full_keywords=False`, 400).
      - `usage`: 글 토큰 ≈ UTF-8 바이트 / 3, 이미지 토큰 = ⌈w/32⌉·⌈h/32⌉ (JPEG 크기를 읽는다).
        접두사 캐시: 앞서 본 요청과 같은 앞부분(system → 글 → 이미지 … 조각 단위)의 토큰을
        `prompt_tokens_details.cached_tokens`로 낸다 (`report_cached=False`면 빼고 낸다).
    - `POST /tokenize`: 글 토큰 수 (`tokenize=False`면 404).

기록: `calls`(경로·헤더·본문), 동시에 처리 중인 요청의 최대 수(`max_in_flight`).
네트워크 밖으로 나가지 않고 127.0.0.1에만 묶인다. 이미지·요청 본문을 파일로 남기지 않는다.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import TracebackType
from typing import Any

import cv2
import numpy as np

# 이미지 한 장의 토큰을 셀 때의 화소 단위 (Qwen3-VL patch 16 × merge 2)
_PATCH_PX = 32
# 기본 판정 요청 알아보기: 가변 글 끝 문장 "대상 #0~#N를 판정해 …" (dlp_aireview.prompt.unit_text)
_TARGETS = re.compile(r"대상 #0~#(\d+)를 판정해")
# 서버가 받지 않는 구조화 출력 키워드 (full_keywords=False일 때)
_FULL_KEYWORDS = ("minimum", "maximum", "minLength", "maxLength", "minItems")


@dataclass(frozen=True)
class FakeCall:
    """받은 요청 하나.

    Attributes:
        method, path: HTTP 메서드와 경로.
        headers: 요청 헤더 (소문자 키).
        body: JSON 본문 (없거나 JSON이 아니면 None).
    """

    method: str
    path: str
    headers: Mapping[str, str]
    body: Any

    @property
    def unit_id(self) -> str | None:
        """`X-DLP-Unit` 헤더 (판정 요청의 단위 키)."""
        return self.headers.get("x-dlp-unit")


@dataclass
class _Failure:
    """예약한 실패: 앞으로 `after`번 성공한 뒤 `times`번 이 상태로 답한다."""

    status: int
    times: int
    after: int
    body: str
    retry_after: str | None


@dataclass
class _Usage:
    """한 요청의 토큰 집계."""

    prompt: int = 0
    cached: int = 0
    images: int = 0
    keys: list[str] = field(default_factory=list[str])


def text_tokens(text: str) -> int:
    """글 토큰 근사 (UTF-8 바이트 / 3, 올림). 한글 한 글자 ≈ 1토큰."""
    return math.ceil(len(text.encode()) / 3)


def _image_tokens(url: str) -> int:
    """data URI JPEG의 토큰 (⌈w/32⌉·⌈h/32⌉). 읽지 못하면 1."""
    try:
        raw = base64.b64decode(url.split(",", 1)[1])
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    except (IndexError, ValueError):
        return 1
    if img is None:
        return 1
    h, w = img.shape[:2]
    return math.ceil(w / _PATCH_PX) * math.ceil(h / _PATCH_PX)


def _pieces(body: Mapping[str, Any]) -> list[tuple[str, int, bool]]:
    """요청 메시지 → (조각 해시 입력, 토큰 수, 이미지인가) 목록 (보낸 순서)."""
    out: list[tuple[str, int, bool]] = []
    for msg in body.get("messages", []):
        content = msg.get("content")
        if isinstance(content, str):
            out.append((f"{msg.get('role')}:{content}", text_tokens(content), False))
            continue
        for part in content or []:
            if part.get("type") == "image_url":
                url = str(part["image_url"]["url"])
                out.append((url, _image_tokens(url), True))
            else:
                text = str(part.get("text", ""))
                out.append((text, text_tokens(text), False))
    return out


def minimal_instance(schema: Mapping[str, Any]) -> Any:
    """JSON Schema(이 패키지가 쓰는 부분집합)를 만족하는 최소 값 (가짜 서버의 구조화 출력 흉내)."""
    if "enum" in schema:
        return schema["enum"][0]
    if "anyOf" in schema:
        return minimal_instance(schema["anyOf"][0])
    kind = schema.get("type")
    if isinstance(kind, list):
        kind = "null" if "null" in kind else kind[0]
    if kind == "object":
        props: Mapping[str, Any] = schema.get("properties", {})
        return {k: minimal_instance(props[k]) for k in schema.get("required", [])}
    defaults: dict[str, Any] = {
        "array": [], "string": "", "integer": 0, "number": 0, "boolean": False, "null": None,
    }  # fmt: skip
    return defaults.get(str(kind))


def _schema_keywords(schema: Any) -> set[str]:
    """스키마에 쓰인 키 이름 전부 (중첩 포함)."""
    if isinstance(schema, dict):
        out: set[str] = set()
        for k, v in schema.items():  # pyright: ignore[reportUnknownVariableType]
            out |= {str(k), *_schema_keywords(v)}  # pyright: ignore[reportUnknownArgumentType]
        return out
    if isinstance(schema, list):
        return {k for v in schema for k in _schema_keywords(v)}  # pyright: ignore[reportUnknownVariableType]
    return set()


class FakeOpenAIServer:
    """가짜 OpenAI 호환 VLM 서버 (모듈 docstring). `with FakeOpenAIServer(...) as server:`로 쓴다.

    Attributes:
        url: 기본 주소 (`http://127.0.0.1:<포트>`).
        calls: 받은 요청 (순서대로).
        max_in_flight: 동시에 처리 중이던 채팅 요청의 최대 수.
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen3-VL-8B-Instruct",
        *,
        served_model: str | None = None,
        latency_s: float = 0.0,
        api_key: str | None = None,
        enforce_schema: bool = True,
        full_keywords: bool = True,
        max_images: int = 64,
        context_tokens: int = 32_768,
        report_cached: bool = True,
        tokenize: bool = True,
        version: str | None = "0.0.0-fake",
        completion_chars_per_token: int = 3,
    ) -> None:
        """설정만 하고 띄우지는 않는다 (`start` 또는 `with`).

        Args:
            model: `/v1/models`에 올릴 모델 id.
            served_model: 응답 `model`에 쓸 이름 (기본 `model`. 다르게 주면 모델 불일치 흉내).
            latency_s: 채팅 요청마다 기다릴 초.
            api_key: 있으면 `Authorization: Bearer <키>`가 아닌 요청을 401로 거부한다.
            enforce_schema: 구조화 출력(`response_format`)을 강제하는가.
            full_keywords: 범위·길이 키워드가 든 스키마를 받는가 (False면 400).
            max_images: 요청당 이미지 상한 (넘으면 400).
            context_tokens: 문맥 길이 (프롬프트 + max_tokens가 넘으면 400).
            report_cached: usage에 `prompt_tokens_details.cached_tokens`를 넣는가.
            tokenize: `/tokenize`를 제공하는가.
            version: `/version`의 엔진 버전 (None이면 404).
            completion_chars_per_token: 생성 토큰 근사 (답 글자 수 / 이 값).
        """
        self.model = model
        self.served_model = served_model or model
        self.latency_s = latency_s
        self.api_key = api_key
        self.enforce_schema = enforce_schema
        self.full_keywords = full_keywords
        self.max_images = max_images
        self.context_tokens = context_tokens
        self.report_cached = report_cached
        self.tokenize = tokenize
        self.version = version
        self.completion_chars_per_token = completion_chars_per_token
        self.calls: list[FakeCall] = []
        self.max_in_flight = 0
        self._answers: dict[str, str] = {}
        self._responder: Callable[[FakeCall], str] | None = None
        self._failures: list[_Failure] = []
        self._seen: set[str] = set()
        self._in_flight = 0
        self._lock = threading.Lock()
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.url = ""

    # ------------------------------------------------------------ 수명

    def start(self) -> FakeOpenAIServer:
        """127.0.0.1 임의 포트에 띄운다."""
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 (http.server 규약)
                server.handle(self, "GET")

            def do_POST(self) -> None:  # noqa: N802
                server.handle(self, "POST")

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                """요청 로그를 남기지 않는다 (이미지 base64가 섞이지 않게)."""

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._httpd.daemon_threads = True
        self.url = f"http://127.0.0.1:{self._httpd.server_address[1]}"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def close(self) -> None:
        """서버를 멈춘다."""
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None

    def __enter__(self) -> FakeOpenAIServer:
        """`start`."""
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """`close`."""
        self.close()

    # ------------------------------------------------------------ 설정

    def answer(self, unit_id: str, content: str) -> None:
        """`X-DLP-Unit: unit_id` 요청에 낼 답(원문)을 등록한다."""
        with self._lock:
            self._answers[unit_id] = content

    def respond_with(self, fn: Callable[[FakeCall], str] | None) -> None:
        """등록한 답이 없는 채팅 요청의 답을 만드는 함수 (None이면 기본 답)."""
        self._responder = fn

    def fail(
        self,
        status: int,
        times: int = 1,
        *,
        after: int = 0,
        body: str = "fake failure",
        retry_after: str | None = None,
    ) -> None:
        """앞으로 채팅 요청이 `after`번 성공한 뒤 `times`번 `status`로 답한다 (예약은 차례로 쓴다)."""
        with self._lock:
            self._failures.append(_Failure(status, times, after, body, retry_after))

    @property
    def chat_calls(self) -> list[FakeCall]:
        """받은 채팅 요청."""
        return [c for c in self.calls if c.path == "/v1/chat/completions"]

    # ------------------------------------------------------------ 처리

    def handle(self, h: BaseHTTPRequestHandler, method: str) -> None:
        """요청 하나를 처리한다 (핸들러 스레드)."""
        length = int(h.headers.get("Content-Length") or 0)
        raw = h.rfile.read(length) if length else b""
        try:
            body: Any = json.loads(raw) if raw else None
        except ValueError:
            body = None
        call = FakeCall(method, h.path, {k.lower(): v for k, v in h.headers.items()}, body)
        with self._lock:
            self.calls.append(call)
        if self.api_key is not None and call.headers.get("authorization") != (
            f"Bearer {self.api_key}"
        ):
            self._send(h, 401, {"error": {"message": "invalid api key"}})
            return
        route = {
            ("GET", "/v1/models"): self._models,
            ("GET", "/version"): self._version,
            ("POST", "/tokenize"): self._tokenize,
            ("POST", "/v1/chat/completions"): self._chat_entry,
        }.get((method, h.path))
        if route is None:
            self._send(h, 404, {"error": {"message": "not found"}})
            return
        route(h, call)

    def _send(
        self,
        h: BaseHTTPRequestHandler,
        status: int,
        data: Any,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        """JSON 응답."""
        payload = json.dumps(data, ensure_ascii=False).encode()
        h.send_response(status)
        h.send_header("Content-Type", "application/json")
        h.send_header("Content-Length", str(len(payload)))
        for k, v in (headers or {}).items():
            h.send_header(k, v)
        h.end_headers()
        h.wfile.write(payload)

    def _models(self, h: BaseHTTPRequestHandler, call: FakeCall) -> None:
        """모델 목록."""
        entry = {"id": self.model, "object": "model", "max_model_len": self.context_tokens}
        self._send(h, 200, {"object": "list", "data": [entry]})

    def _version(self, h: BaseHTTPRequestHandler, call: FakeCall) -> None:
        """엔진 버전 (없으면 404)."""
        if self.version is None:
            self._send(h, 404, {"error": {"message": "not found"}})
        else:
            self._send(h, 200, {"version": self.version})

    def _tokenize(self, h: BaseHTTPRequestHandler, call: FakeCall) -> None:
        """글 토큰 수 (끄면 404)."""
        if not self.tokenize:
            self._send(h, 404, {"error": {"message": "not found"}})
            return
        prompt = str((call.body or {}).get("prompt", ""))
        self._send(h, 200, {"count": text_tokens(prompt), "max_model_len": self.context_tokens})

    def _chat_entry(self, h: BaseHTTPRequestHandler, call: FakeCall) -> None:
        """채팅 요청 (처리 중 요청 수를 센다)."""
        with self._lock:
            self._in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)
        try:
            self._chat(h, call)
        finally:
            with self._lock:
                self._in_flight -= 1

    def _scheduled_failure(self) -> _Failure | None:
        """이번 요청에 쓸 예약 실패 (없으면 None, 성공 횟수를 센다)."""
        with self._lock:
            if not self._failures:
                return None
            head = self._failures[0]
            if head.after > 0:
                head.after -= 1
                return None
            head.times -= 1
            if head.times <= 0:
                self._failures.pop(0)
            return head

    def _reject(self, body: Mapping[str, Any], usage: _Usage) -> tuple[int, str] | None:
        """요청 검사 (이미지 상한, 스키마 키워드, 문맥 길이). 거부면 (상태, 메시지)."""
        if usage.images > self.max_images:
            return 400, f"At most {self.max_images} image(s) may be provided in one request."
        fmt = body.get("response_format") or {}
        schema = (fmt.get("json_schema") or {}).get("schema")
        if not self.full_keywords and schema and _schema_keywords(schema) & set(_FULL_KEYWORDS):
            return 400, "Unsupported JSON schema keyword in response_format"
        need = usage.prompt + int(body.get("max_tokens") or 0)
        if need > self.context_tokens:
            return 400, (
                f"This model's maximum context length is {self.context_tokens} tokens. "
                f"However, you requested {need} tokens."
            )
        return None

    def _usage(self, body: Mapping[str, Any]) -> _Usage:
        """프롬프트 토큰·이미지 수·캐시 적중 토큰 (조각 단위 접두사 캐시)."""
        usage = _Usage()
        prefix = hashlib.sha256()
        hit = True
        for key, tokens, image in _pieces(body):
            prefix.update(key.encode())
            digest = prefix.hexdigest()
            usage.keys.append(digest)
            usage.prompt += tokens
            usage.images += int(image)
            hit = hit and digest in self._seen
            if hit:
                usage.cached += tokens
        return usage

    def _content(self, call: FakeCall, body: Mapping[str, Any]) -> str:
        """답 원문 (등록한 답 → 응답 함수 → 기본 답)."""
        unit = call.unit_id
        with self._lock:
            registered = self._answers.get(unit) if unit else None
        if registered is not None:
            return registered
        if self._responder is not None:
            return self._responder(call)
        texts = [
            str(p.get("text", ""))
            for m in body.get("messages", [])
            if isinstance(m.get("content"), list)
            for p in m["content"]
        ]
        match = _TARGETS.search(texts[-1]) if texts else None
        if match:
            n = int(match.group(1)) + 1
            return json.dumps({"ok": list(range(n)), "items": [], "missing": []})
        fmt = body.get("response_format") or {}
        schema = (fmt.get("json_schema") or {}).get("schema")
        if schema and self.enforce_schema:
            return json.dumps(minimal_instance(schema), ensure_ascii=False)
        return "모름"

    def _chat(self, h: BaseHTTPRequestHandler, call: FakeCall) -> None:
        """채팅 응답 (지연·실패·검사·usage)."""
        if self.latency_s:
            time.sleep(self.latency_s)
        failure = self._scheduled_failure()
        if failure is not None:
            extra = {"Retry-After": failure.retry_after} if failure.retry_after else None
            self._send(h, failure.status, {"error": {"message": failure.body}}, extra)
            return
        body: Mapping[str, Any] = call.body if isinstance(call.body, dict) else {}
        with self._lock:
            usage = self._usage(body)
        rejected = self._reject(body, usage)
        if rejected is not None:
            self._send(h, rejected[0], {"error": {"message": rejected[1]}})
            return
        with self._lock:
            self._seen.update(usage.keys)
        content = self._content(call, body)
        max_tokens = body.get("max_tokens")
        completion = max(1, math.ceil(len(content) / self.completion_chars_per_token))
        if isinstance(max_tokens, int) and completion > max_tokens:
            content = content[: max_tokens * self.completion_chars_per_token]
            completion = max_tokens
        details = {"prompt_tokens_details": {"cached_tokens": usage.cached}}
        data = {
            "id": "chatcmpl-fake", "object": "chat.completion", "model": self.served_model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": usage.prompt, "completion_tokens": completion,
                      "total_tokens": usage.prompt + completion,
                      **(details if self.report_cached else {})},
        }  # fmt: skip
        self._send(h, 200, data)
