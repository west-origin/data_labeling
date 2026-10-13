"""설명 번역기: CPU용 용어집 stub과 사내 OpenAI 호환 서버 어댑터 (ADR 0043).

- `Translator`: 번역기 규약 (`version`, `translate`). `translate.run_translate`가 쓴다.
- `GlossaryEntry`: 용어집 한 줄 (한국어 이름 → 영어 이름, 온톨로지 표시 이름에서 온다).
- `GlossaryStubTranslator`: CI·CPU용 결정적 stub. 알려 준 정답 번역이 있으면 그 문장, 없으면 용어집
  치환 + `[stub-en]` 표시. 실패를 흉내 낼 수 있다.
- `OpenAICompatibleTranslator`: 사내 OpenAI 호환 서버(vLLM)에 글만 보낸다 (이미지 없음). 전송·오류
  분류·백오프는 공통 모듈 `dlp_models.vlm`을 쓴다.
- 예외: `TranslationUnavailableError`(서버 장애, 그 세션의 남은 설명은 미룬다),
  `TranslationConfigError`(4xx 설정 오류, 미룬다), `TranslationInvalidError`(재시도 뒤에도
  응답 위반, 그 설명만 미룬다). 어느 경우도 라벨을 쓰지 않고 대기 목록에 남긴다.
- `PROMPT_VERSION`은 `translate_policy`에 있다 (정책 적재 시 대조).

외부 API는 쓰지 않는다. 사내 대역 검사는 CLI가 요청 전에 한다 (`dlp_cli.translate_cmds`).
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from dlp_actions.translate_policy import TranslatePolicy
from dlp_models.vlm import (
    ChatRequest,
    ChatTurn,
    OpenAICompatibleChat,
    VlmRequestError,
    VlmUnavailableError,
    ask_validated,
    with_backoff,
)

# 한글 음절·자모 (번역문에 남으면 위반)
HANGUL = re.compile("[\u1100-\u11ff\u3130-\u318f\uac00-\ud7a3]")

# 번역 프롬프트 틀 (PROMPT_VERSION 1). 바꾸면 translate_policy.PROMPT_VERSION과 정책의
# prompt_version을 함께 올린다
SYSTEM = (
    "You translate Korean descriptions of first-person work videos (cleaning, caregiving, nursing) "
    "into natural, concise English. Translate faithfully: do not add, guess or drop details. "
    "Where a glossary term applies, use its English term exactly. "
    'Reply with JSON only: {"text": "<English translation>"}.'
)


class TranslationUnavailableError(RuntimeError):
    """번역 서버 장애 (백오프 뒤에도 응답 없음). 그 세션의 남은 설명은 미룬다."""


class TranslationConfigError(RuntimeError):
    """번역 서버 설정 오류 (4xx: 인증·모델 이름 등). 다시 물어도 같아 남은 설명을 미룬다."""


class TranslationInvalidError(RuntimeError):
    """재시도 뒤에도 응답이 위반이다 (JSON 아님·빈 번역·한글 남음·길이 초과). 그 설명만 미룬다."""


@dataclass(frozen=True)
class GlossaryEntry:
    """용어집 한 줄: 한국어 표시 이름 → 영어 표시 이름 (온톨로지 `display_names`)."""

    ko: str
    en: str


class Translator(Protocol):
    """번역기 규약.

    Attributes:
        version: 번역기 버전 문자열 (번역 모델 버전에 들어간다. 예: `llm-<모델>`,
        `stub-glossary-1`).
    """

    version: str

    def translate(self, text: str, glossary: Sequence[GlossaryEntry]) -> str:
        """원문 한 문장을 번역한다.

        Raises:
            TranslationUnavailableError | TranslationConfigError | TranslationInvalidError.
        """
        ...


def apply_glossary(text: str, glossary: Sequence[GlossaryEntry]) -> str:
    """원문의 한국어 이름을 영어 이름으로 바꾼다 (긴 이름부터, 결정적). stub이 쓴다."""
    for entry in sorted(glossary, key=lambda e: (-len(e.ko), e.ko)):
        text = text.replace(entry.ko, entry.en)
    return text


class GlossaryStubTranslator:
    """CI·CPU용 결정적 번역기 stub.

    known에 원문이 있으면 그 번역을, 없으면 용어집 치환 결과 앞에 `[stub-en] `을 붙여 돌려준다
    (실제 번역이 아니라는 표시). fail_unavailable·fail_invalid에 든 원문은 해당 예외를 낸다.

    TODO(real-model): 실제 번역은 사내 VLM 서버(`OpenAICompatibleTranslator`)가 한다. 이 stub은 흐름
      검사용이다.
    """

    def __init__(
        self,
        known: Mapping[str, str] | None = None,
        *,
        fail_unavailable: Sequence[str] = (),
        fail_invalid: Sequence[str] = (),
        version: str = "stub-glossary-1",
    ) -> None:
        """정답 번역과 실패 흉내 목록으로 만든다. `calls`는 번역 요청 수다."""
        self.version = version
        self.known = dict(known or {})
        self.fail_unavailable = set(fail_unavailable)
        self.fail_invalid = set(fail_invalid)
        self.calls = 0

    def translate(self, text: str, glossary: Sequence[GlossaryEntry]) -> str:
        """정답 번역 → 실패 흉내 → 용어집 치환 순으로 답한다."""
        self.calls += 1
        if text in self.fail_unavailable:
            raise TranslationUnavailableError("stub: 서버 장애 흉내")
        if text in self.fail_invalid:
            raise TranslationInvalidError("stub: 응답 위반 흉내")
        if text in self.known:
            return self.known[text]
        return f"[stub-en] {apply_glossary(text, glossary)}"


def translation_schema(max_chars: int) -> dict[str, Any]:
    """번역 응답 JSON Schema ({"text": 번역문})."""
    return {
        "type": "object",
        "properties": {"text": {"type": "string", "minLength": 1, "maxLength": max_chars}},
        "required": ["text"],
        "additionalProperties": False,
    }


def user_prompt(text: str, glossary: Sequence[GlossaryEntry]) -> str:
    """사용자 메시지: 용어집(있으면) + 원문."""
    parts: list[str] = []
    if glossary:
        lines = "\n".join(f"- {e.ko} => {e.en}" for e in glossary)
        parts.append(f"Glossary (Korean => English):\n{lines}")
    parts.append(f"Korean:\n{text}")
    return "\n\n".join(parts)


def check_reply(raw: str, max_chars: int, reject_hangul: bool) -> tuple[str | None, list[str]]:
    """응답 원문 → (번역문 또는 None, 위반 목록). 순수 함수."""
    try:
        data: Any = json.loads(raw)
    except ValueError:
        return None, ["reply is not JSON"]
    value = data.get("text") if isinstance(data, dict) else None  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    if not isinstance(value, str) or not value.strip():
        return None, ['"text" must be a non-empty string']
    text = value.strip()
    problems: list[str] = []
    if len(text) > max_chars:
        problems.append(f'"text" is longer than {max_chars} characters')
    if reject_hangul and HANGUL.search(text):
        problems.append('"text" still contains Korean (Hangul); translate everything into English')
    return (None, problems) if problems else (text, [])


def retry_text(problems: Sequence[str]) -> str:
    """재시도 user 메시지: 직전 응답의 위반 목록."""
    return "Your previous reply was invalid:\n" + "\n".join(f"- {p}" for p in problems)


def server_translator_version(policy: TranslatePolicy) -> str:
    """사내 서버 번역기의 버전 문자열 `llm-<모델>` (서버에 묻지 않고 정책만으로 정해진다)."""
    return f"llm-{policy.backend.model}"


class OpenAICompatibleTranslator:
    """사내 OpenAI 호환 채팅 서버(vLLM 등)로 번역한다. 글만 보낸다 (이미지 없음).

    서버 장애는 `server.backoff_s`만큼 기다리며 다시 묻고, 응답 위반은 위반 목록을 붙여
    `backend.max_retries`번 다시 묻는다. temperature 0.

    TODO(real-model): 실제 사내 서버(services/vlm vlm-aireview, Qwen3-VL-8B)에서 아직 검증하지
      않았다. 요청 형식만 가짜 전송 계층으로 테스트한다.
    """

    def __init__(
        self,
        base_url: str,
        policy: TranslatePolicy,
        *,
        api_key: str | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """HTTP 클라이언트를 만든다 (연결은 첫 요청 때).

        Args:
            base_url: 사내 서버 주소 (`--translate-url` 또는 `DLP_TRANSLATE_URL`). 사내 대역 검사는
                호출자(CLI)가 먼저 한다.
            policy: 번역 정책 (모델·토큰 상한·재시도·시간 초과).
            api_key: 서버 API 키 (`DLP_VLM_API_KEY`). Authorization 헤더로만 보낸다.
            transport: httpx 전송 계층 (테스트에서 `MockTransport`).
            sleep: 백오프 대기 함수 (테스트에서 바꾼다).
        """
        backend = policy.backend
        name = backend.served_name or backend.model
        self.version = server_translator_version(policy)
        self.policy = policy
        self.sleep = sleep
        # 직전 응답 원문 (재시도 대화에 assistant 턴으로 붙인다)
        self._last = ""
        self.chat = OpenAICompatibleChat(
            base_url,
            name,
            timeout_s=policy.server.timeout_s,
            api_key=api_key,
            expected_model=name,
            transport=transport,
        )

    def _ask(self, text: str, glossary: Sequence[GlossaryEntry], problems: Sequence[str]) -> str:
        """한 번 묻는다 (위반이 있으면 직전 응답과 위반 목록을 대화에 붙인다)."""
        b = self.policy.backend
        turns = [ChatTurn("user", (user_prompt(text, glossary),))]
        if problems:
            turns += [
                ChatTurn("assistant", (self._last,)),
                ChatTurn("user", (retry_text(problems),)),
            ]
        req = ChatRequest(
            system=SYSTEM,
            turns=tuple(turns),
            schema=translation_schema(b.max_chars),
            schema_name="translation",
            max_tokens=b.max_tokens,
        )
        reply = with_backoff(lambda: self.chat.chat(req), self.policy.server.backoff_s, self.sleep)
        self._last = reply.content
        return reply.content

    def translate(self, text: str, glossary: Sequence[GlossaryEntry]) -> str:
        """번역문을 돌려준다 (응답 검사·재시도 포함). 외부(사내) 서비스 호출이다.

        Raises:
            TranslationUnavailableError: 백오프 뒤에도 서버 장애.
            TranslationConfigError: 4xx 설정 오류.
            TranslationInvalidError: 재시도 뒤에도 응답 위반.
        """
        b = self.policy.backend
        self._last = ""
        errors: list[str] = []
        try:
            value, _ = ask_validated(
                lambda problems: self._ask(text, glossary, problems),
                lambda raw: check_reply(raw, b.max_chars, b.reject_hangul),
                b.max_retries,
                errors,
            )
        except VlmUnavailableError as exc:
            raise TranslationUnavailableError(str(exc)) from exc
        except VlmRequestError as exc:
            raise TranslationConfigError(str(exc)) from exc
        if value is None:
            raise TranslationInvalidError("; ".join(errors[-3:]))
        return value
