"""판정 백엔드 경계 (WP17.2, ADR 0040). 백엔드를 바꿔 끼우는 곳은 여기 하나다.

- `JudgeRequest`: 요청 하나 (프롬프트 조각, 재시도 대화, 스키마, 추적 키). stub이 정답을 찾을 수
  있게 대상 라벨·개체표·프레임 시각도 담는다 (실제 서버에는 프롬프트 글로만 간다).
- `JudgeReply`: 응답 원문과 서버 모델·엔드포인트 이름·토큰·지연.
- `BackendInfo`: 실행 기록에 남길 백엔드 정보 (엔드포인트는 이름만, URL·키 아님).
- `JudgeBackend`: 프로토콜. `judge`는 장애면 `VlmUnavailableError`, 설정 오류면 `VlmRequestError`,
  문맥 초과면 `VlmContextError`를 낸다 (`dlp_models.vlm`).
- `OpenAICompatibleJudge`: 사내 OpenAI 호환 서버(vLLM 등) 하나. 지금 구현은 이것뿐이다.
  외부 백엔드(외부 API)는 쓰지 않는다 (`server.allow_external`는 false만 허용, 별도 ADR 필요).
- CPU·CI용 stub은 `dlp_aireview.oracle.OracleJudge`.

여러 엔드포인트 묶음·차단기(`EndpointPool`)와 사내 주소 검사(`check_hosts`)는 실행기와 함께
WP17.5에서 더한다.

판정은 `LabelRecord`를 내지 않으므로 `dlp_schema.predictor.Predictor`가 아니다. "어댑터 + CPU stub"
규칙은 `JudgeBackend` + `OracleJudge`로 지킨다.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import httpx

from dlp_aireview.policy import AiReviewPolicy
from dlp_aireview.prompt import PromptParts
from dlp_models.vlm import ChatRequest, ChatTurn, OpenAICompatibleChat
from dlp_schema.aireview import AiGroup
from dlp_schema.labels import LabelRecord


@dataclass(frozen=True)
class JudgeRequest:
    """판정 요청 하나.

    Attributes:
        unit_id: 단위 키 (HTTP 헤더 `X-DLP-Unit`, 본문에는 넣지 않는다).
        affinity: 같은 창의 분할 요청을 같은 엔드포인트로 보내는 키 (`PlannedWindow.window_key`).
        group: 묶음.
        parts: 프롬프트 조각.
        history: 재시도 대화 [(직전 응답 원문, 재시도 지시 글)] (끝에만 덧붙인다).
        schema: 응답 JSON Schema.
        schema_enforcement: server | prompt.
        max_tokens: 생성 토큰 상한.
        targets: 이 요청의 대상 라벨 (번호 순, stub용).
        entities: 창 개체표 (개체 ID, 클래스) (stub용).
        frame_ms: 이 요청의 프레임 시각 (F# 순, stub용).
    """

    unit_id: str
    affinity: str
    group: AiGroup
    parts: PromptParts
    history: tuple[tuple[str, str], ...]
    schema: Mapping[str, Any]
    schema_enforcement: Literal["server", "prompt"]
    max_tokens: int
    targets: tuple[LabelRecord, ...]
    entities: tuple[tuple[str, str | None], ...]
    frame_ms: tuple[int, ...]


@dataclass(frozen=True)
class JudgeReply:
    """판정 응답 하나.

    Attributes:
        content: 응답 원문 (DB에는 sha256만 남긴다).
        served_model: 서버가 보고한 모델.
        endpoint: 엔드포인트 이름 (URL·키 아님).
        prompt_tokens, cached_tokens, completion_tokens: 서버 usage (없으면 None).
        latency_ms: 왕복 시간.
    """

    content: str
    served_model: str | None
    endpoint: str
    prompt_tokens: int | None = None
    cached_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: int = 0


@dataclass(frozen=True)
class BackendInfo:
    """실행 기록용 백엔드 정보.

    Attributes:
        kind: 백엔드 종류. model: 정책 모델. fingerprint: 엔진·양자화·가중치 revision.
        endpoints: 엔드포인트 이름. external: 외부 서비스인가 (지금은 늘 False).
    """

    kind: str
    model: str
    fingerprint: str
    endpoints: tuple[str, ...]
    external: bool = False


class JudgeBackend(Protocol):
    """판정 백엔드 (모듈 docstring)."""

    info: BackendInfo

    def judge(self, request: JudgeRequest) -> JudgeReply:
        """요청 하나를 묻는다.

        Raises:
            VlmUnavailableError: 장애 (호출자가 백오프·다른 엔드포인트로 다시 묻는다).
            VlmContextError: 문맥 길이 초과 (호출자가 프레임·대상을 줄인다).
            VlmRequestError: 설정 오류·서버 모델 불일치 (실행 중단).
        """
        ...


def chat_request(request: JudgeRequest) -> ChatRequest:
    """판정 요청 → 채팅 요청: system, user(고정 블록 → 이미지 → 가변 글), 재시도 대화."""
    turns = [ChatTurn("user", request.parts.blocks())]
    for raw, retry in request.history:
        turns += [ChatTurn("assistant", (raw,)), ChatTurn("user", (retry,))]
    return ChatRequest(
        system=request.parts.system,
        turns=tuple(turns),
        schema=request.schema,
        schema_name=f"aireview_{request.group.value}",
        max_tokens=request.max_tokens,
        schema_enforcement=request.schema_enforcement,
        headers={"X-DLP-Unit": request.unit_id},
    )


class OpenAICompatibleJudge:
    """사내 OpenAI 호환 서버 하나로 판정한다 (`JudgeBackend` 구현).

    응답 `model`이 정책 모델(`served_name`, 없으면 `model`)과 다르면 `VlmRequestError`다 (I7).

    TODO(real-model): 실제 VLM 서버(DGX Spark의 vLLM, OpenAI 호환 구조화 출력)에서 아직 검증하지
      않았다. 요청 형식·오류 분류만 MockTransport와 가짜 서버로 테스트한다. WP17.3 probe·bench로
      서버 지원(이미지 수 한도, json_schema 강제, cached_tokens)을 확인한다.
    """

    def __init__(
        self,
        name: str,
        base_url: str,
        policy: AiReviewPolicy,
        *,
        api_key: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """클라이언트를 만든다 (연결은 첫 요청 때).

        Args:
            name: 엔드포인트 이름 (기록·진행률에 쓰는 짧은 이름, 예: spark-01).
            base_url: 서버 주소 (사내 주소 검사는 WP17.5 `check_hosts`).
            policy: 정책 (모델, 시간 초과).
            api_key: `DLP_VLM_API_KEY` (Authorization 헤더로만 보낸다, 기록하지 않는다).
            transport: httpx 전송 계층 (테스트).
        """
        self.name = name
        self.info = BackendInfo(
            kind=policy.backend.kind,
            model=policy.backend.model,
            fingerprint=policy.backend.fingerprint,
            endpoints=(name,),
        )
        self.chat = OpenAICompatibleChat(
            base_url,
            policy.backend.model,
            timeout_s=policy.server.timeout_s,
            api_key=api_key,
            expected_model=policy.served_model,
            transport=transport,
        )

    def judge(self, request: JudgeRequest) -> JudgeReply:
        """요청을 보내고 응답을 받는다 (외부 서비스 호출). 오류는 `JudgeBackend.judge`와 같다."""
        reply = self.chat.chat(chat_request(request))
        return JudgeReply(
            content=reply.content,
            served_model=reply.model,
            endpoint=self.name,
            prompt_tokens=reply.prompt_tokens,
            cached_tokens=reply.cached_tokens,
            completion_tokens=reply.completion_tokens,
            latency_ms=reply.latency_ms,
        )
