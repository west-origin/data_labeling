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

- `parse_endpoints`: `--vlm-url`·`DLP_VLM_URLS`(`이름=URL,…`) → 엔드포인트 이름 → URL.
- `check_hosts`: 주소가 사내 대역(`server.internal_hosts`) 안인지 (호스트 이름은 DNS로 풀어 모든
  주소를 본다, I7). 벤치·점검·실행 모두 보내기 전에 부른다.
- `EndpointPool`(WP17.5): 여러 엔드포인트(장비마다 서버 하나)를 하나의 `JudgeBackend`로
  묶는다. 같은 창의 요청은 같은 엔드포인트로(접두사·이미지 캐시), 그곳이 바쁘면 처리 중
  요청이 가장 적은 곳으로 보낸다. 엔드포인트마다 동시 요청 상한
  (`server.concurrency_per_endpoint`)을 지킨다. 장애는 백오프하며 다시 묻고, 연속 장애가
  `breaker_failures`번이면 그 엔드포인트의 차단기를 연다. `breaker_cooldown_s` 뒤 반개방
  확인(`ping`: `/v1/models`)을 통과하면 시험 요청 하나를 보내고, 그 요청이 성공하면 닫는다.
  모든 엔드포인트가 `abort_after_s`보다 오래 열려 있으면 `AllEndpointsDownError`로 실행을
  멈춘다 (종료 3). 실행기가 주는 멈춤 신호(`judge(request, cancel=…)`)가 켜지면 보낼 곳을
  기다리거나 백오프하던 요청은 보내지 않고 `JudgeCancelledError`로 끝난다 (실행기 멈춤·
  `--max-minutes`·Ctrl-C 뒤에 새 요청을 보내지 않게). 실행기는 `CancellableBackend`로 감싸 이
  신호를 넘긴다.

판정은 `LabelRecord`를 내지 않으므로 `dlp_schema.predictor.Predictor`가 아니다. "어댑터 + CPU stub"
규칙은 `JudgeBackend` + `OracleJudge`로 지킨다.
"""

from __future__ import annotations

import hashlib
import ipaddress
import re
import socket
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol
from urllib.parse import urlsplit

import httpx

from dlp_aireview.policy import AiReviewPolicy, ServerPolicy
from dlp_aireview.prompt import PromptParts
from dlp_models.vlm import (
    ChatRequest,
    ChatTurn,
    OpenAICompatibleChat,
    VlmRequestError,
    VlmUnavailableError,
)
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
        frame_nos: 이 요청에 보낸 프레임의 단위 프레임 번호 F# (보낸 순서, stub용).
        frame_ms: 그 프레임들의 시각 (frame_nos 순, stub용).
        extra_body: 서버 확장 키 (벤치 생성 용량 측정의 `min_tokens`·`ignore_eos`만 쓴다. 판정
            요청은 비운다).
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
    frame_nos: tuple[int, ...]
    frame_ms: tuple[int, ...]
    extra_body: Mapping[str, Any] = field(default_factory=dict[str, Any])


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
        extra_body=request.extra_body,
    )


class OpenAICompatibleJudge:
    """사내 OpenAI 호환 서버 하나로 판정한다 (`JudgeBackend` 구현).

    요청 본문의 `model`은 서버에 등록된 이름(`backend.served_name`, 없으면 `backend.model`)이다.
    vLLM은 `--served-model-name`에 준 이름만 받는다. 응답 `model`이 그 이름과 다르면
    `VlmRequestError`다 (I7). `backend.model`은 라이선스 검사(`make licenses`)와 판정 버전에 쓰는
    정책 모델 이름이다.

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
            policy.served_model,
            timeout_s=policy.server.timeout_s,
            api_key=api_key,
            expected_model=policy.served_model,
            transport=transport,
        )

    def ping(self) -> None:
        """서버가 정책 모델을 띄우고 있는지 `/v1/models`로 확인한다 (차단기 반개방 확인).

        Raises:
            VlmUnavailableError: 장애.
            VlmRequestError: 4xx, 또는 서버 모델 목록에 정책 모델(`served_name`)이 없을 때.
        """
        models = self.chat.list_models()
        if self.chat.model not in models:
            raise VlmRequestError(f"{self.name}: 서버에 정책 모델 {self.chat.model}이 없습니다")

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


# 엔드포인트 이름 (기록 열 endpoint VARCHAR(64), 진행률 표시용 짧은 이름)
_ENDPOINT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
# 주소 풀기 함수 (테스트에서 바꾼다): socket.getaddrinfo와 같은 모양
Resolver = Callable[[str, int | None], Sequence[tuple[Any, ...]]]


def parse_endpoints(specs: Iterable[str]) -> dict[str, str]:
    """엔드포인트 지정 → {이름: URL} (순서 유지).

    각 지정은 쉼표로 여러 개를 담을 수 있고, 항목은 `이름=URL` 또는 `URL`이다. 이름이 없으면
    `vlm-01`, `vlm-02`, … 순번을 붙인다 (`DLP_VLM_URLS="spark-01=http://10.0.0.11:8000,…"`,
    `--vlm-url URL`을 여러 번).

    Raises:
        ValueError: 빈 URL, http(s)가 아닌 주소, 잘못된 이름, 이름 중복.
    """
    out: dict[str, str] = {}
    items = [x.strip() for spec in specs for x in spec.split(",") if x.strip()]
    for k, item in enumerate(items, start=1):
        name, sep, url = item.partition("=")
        if not sep or "://" in name:
            name, url = f"vlm-{k:02d}", item
        name, url = name.strip(), url.strip().rstrip("/")
        if not _ENDPOINT_NAME.match(name):
            raise ValueError(f"엔드포인트 이름이 올바르지 않습니다: {name!r}")
        if urlsplit(url).scheme not in ("http", "https") or not urlsplit(url).hostname:
            raise ValueError(f"http(s) 주소가 아닙니다: {name}")
        if name in out:
            raise ValueError(f"엔드포인트 이름이 중복됩니다: {name}")
        out[name] = url
    return out


def _addresses(host: str, resolve: Resolver) -> list[str]:
    """호스트 → IP 주소 목록 (IP 글자면 그대로, 이름이면 DNS로 모두 푼다)."""
    try:
        return [str(ipaddress.ip_address(host))]
    except ValueError:
        pass
    return sorted({str(info[4][0]) for info in resolve(host, None)})


def check_hosts(
    endpoints: Mapping[str, str],
    internal_hosts: Sequence[str],
    *,
    resolve: Resolver = socket.getaddrinfo,
) -> list[str]:
    """사내 대역 밖 엔드포인트의 설명 목록 (비면 모두 사내). I7: 외부 서버로 보내지 않는다.

    호스트 이름이면 DNS로 풀어 나온 **모든** 주소가 사내 대역 안이어야 한다 (하나라도 밖이면
    위반). 풀리지 않는 이름도 위반이다. 설명에는 엔드포인트 이름과 주소만 넣는다 (URL의 키·경로는
    넣지 않는다).

    Args:
        endpoints: {이름: URL}.
        internal_hosts: 사내 대역 CIDR (`aireview.yaml server.internal_hosts`).
        resolve: 주소 풀기 함수 (테스트용).
    """
    networks = [ipaddress.ip_network(n, strict=False) for n in internal_hosts]
    problems: list[str] = []
    for name, url in endpoints.items():
        host = urlsplit(url).hostname or ""
        try:
            addresses = _addresses(host, resolve)
        except OSError:
            problems.append(f"{name}: 호스트 {host}의 주소를 풀 수 없습니다")
            continue
        outside = [a for a in addresses if not any(ipaddress.ip_address(a) in n for n in networks)]
        if not addresses or outside:
            problems.append(f"{name}: 사내 대역 밖 주소 {', '.join(outside) or host}")
    return problems


# ---------------------------------------------------------------- 여러 엔드포인트


class AllEndpointsDownError(VlmUnavailableError):
    """모든 엔드포인트의 차단기가 `server.abort_after_s`보다 오래 열려 있다 (실행 중단, 종료 3)."""


class JudgeCancelledError(RuntimeError):
    """실행이 멈춰(멈춤 신호) 이 요청을 보내지 않았다. 장애가 아니고, 그 단위는 기록하지 않는다
    (다음 실행이 묻는다)."""


@dataclass(frozen=True)
class PoolMember:
    """묶음에 넣을 엔드포인트 하나.

    Attributes:
        name: 엔드포인트 이름 (기록·진행률에 쓴다, URL·키 아님).
        backend: 그 엔드포인트의 판정 백엔드.
        ping: 반개방 때 시험 요청 전에 부르는 가벼운 확인 (예: `OpenAICompatibleJudge.ping`).
            장애면 `VlmUnavailableError`, 설정 오류면 `VlmRequestError`를 낸다. None이면 시험
            요청만으로 확인한다.
    """

    name: str
    backend: JudgeBackend
    ping: Callable[[], None] | None = None


@dataclass
class _Endpoint:
    """엔드포인트 상태 (묶음의 잠금 안에서만 바꾼다)."""

    member: PoolMember
    in_flight: int = 0
    failures: int = 0  # 연속 장애 수 (성공하면 0)
    opened_at: float | None = None  # 차단기를 연 시각 (None이면 닫힘)
    probing: bool = False  # 반개방 확인 중 (ping 또는 시험 요청)

    @property
    def state(self) -> str:
        """ok | open | half-open."""
        if self.opened_at is None:
            return "ok"
        return "half-open" if self.probing else "open"


class EndpointPool:
    """여러 엔드포인트를 묶은 `JudgeBackend` (모듈 docstring). 스레드 여럿이 함께 쓴다.

    요청 하나의 장애 시도는 `len(backoff_s) + 1`번까지다 (차단기가 열려 기다리는 시간은
    세지 않는다). 그래도 실패하면 마지막 `VlmUnavailableError`를 올린다 (실행기는 그 단위만
    기록하지 않고 넘어가며, 다음 실행이 다시 묻는다). 모든 엔드포인트가 오래 막히면
    `AllEndpointsDownError`를 올리고, 그 뒤의 요청도 모두 바로 같은 오류로 끝난다.

    Attributes:
        info: 백엔드 정보 (첫 엔드포인트의 종류·모델·fingerprint, 엔드포인트 이름 전부).
    """

    def __init__(
        self,
        members: Sequence[PoolMember],
        server: ServerPolicy,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """엔드포인트 묶음을 만든다.

        Args:
            members: 엔드포인트 (이름이 겹치면 안 된다).
            server: 정책 `server` 절 (동시성, 백오프, 차단기, 중단 시간).
            clock: 단조 시계 (테스트). sleep: 백오프 대기 함수 (테스트).

        Raises:
            ValueError: 엔드포인트가 없거나 이름이 겹칠 때.
        """
        names = [m.name for m in members]
        if not members or len(set(names)) != len(names):
            raise ValueError("엔드포인트가 없거나 이름이 겹칩니다")
        first = members[0].backend.info
        self.info = BackendInfo(
            kind=first.kind, model=first.model, fingerprint=first.fingerprint,
            endpoints=tuple(names), external=first.external,
        )  # fmt: skip
        self.server = server
        self.clock = clock
        self.sleep = sleep
        self._eps = [_Endpoint(m) for m in members]
        self._cond = threading.Condition()
        self._all_open_since: float | None = None
        self._dead: str | None = None

    # ------------------------------------------------------------ 상태

    def states(self) -> dict[str, str]:
        """엔드포인트 이름 → ok | open | half-open (진행률 줄)."""
        with self._cond:
            return {e.member.name: e.state for e in self._eps}

    @property
    def in_flight(self) -> int:
        """처리 중인 요청 수 (모든 엔드포인트 합)."""
        with self._cond:
            return sum(e.in_flight for e in self._eps)

    @property
    def dead(self) -> bool:
        """모든 엔드포인트가 `abort_after_s`보다 오래 막혀 묶음을 멈췄는가 (그 뒤 요청은 모두
        `AllEndpointsDownError`, CLI가 남은 세션을 시작하지 않는다)."""
        with self._cond:
            return self._dead is not None

    # ------------------------------------------------------------ 요청

    def judge(self, request: JudgeRequest, *, cancel: threading.Event | None = None) -> JudgeReply:
        """요청 하나를 엔드포인트에 보낸다 (장애면 백오프 뒤 다시, 클래스 docstring).

        Args:
            request: 판정 요청.
            cancel: 실행기 멈춤 신호. 켜지면 보낼 곳을 기다리거나 백오프하던 이 요청을 보내지 않는다
                (이미 보낸 요청은 끝까지 기다린다). None이면 보지 않는다.

        Raises:
            JudgeCancelledError: 보내기 전에 멈춤 신호가 켜졌다.
            AllEndpointsDownError: 모든 엔드포인트가 `abort_after_s`보다 오래 막힘.
            VlmUnavailableError: 이 요청이 재시도 상한까지 장애.
            VlmRequestError: 설정 오류·서버 모델 불일치 (`VlmContextError` 포함, 엔드포인트 장애로
                세지 않는다).
        """
        backoff = self.server.backoff_s
        attempts = len(backoff) + 1
        for attempt in range(attempts):
            ep = self._acquire(request.affinity, cancel)
            try:
                reply = ep.member.backend.judge(request)
            except VlmRequestError:
                self._release(ep, ok=True)  # 서버는 응답했다 (설정 오류·문맥 초과)
                raise
            except VlmUnavailableError as exc:
                self._release(ep, ok=False)
                if attempt + 1 == attempts:
                    raise
                cap = max(backoff, default=0.0)
                self._pause(min(max(backoff[attempt], exc.retry_after_s or 0.0), cap), cancel)
                continue
            except BaseException:
                self._release(ep, ok=None)
                raise
            self._release(ep, ok=True)
            return reply
        raise AssertionError("도달하지 않는다")  # pragma: no cover

    def _pick(self, affinity: str) -> _Endpoint | None:
        """닫힌 엔드포인트 중 보낼 곳 (잠금 안). 같은 창 키는 해시로 정한 곳이 여유가 있으면 그곳,
        아니면 처리 중 요청이 가장 적은 곳 (상한 미만). 없으면 None."""
        cap = self.server.concurrency_per_endpoint
        free = [e for e in self._eps if e.opened_at is None and e.in_flight < cap]
        if not free:
            return None
        home = self._eps[_affinity_slot(affinity, len(self._eps))]
        if home in free:
            return home
        return min(free, key=lambda e: e.in_flight)

    def _pause(self, seconds: float, cancel: threading.Event | None) -> None:
        """백오프 대기. 멈춤 신호가 있으면 그 신호로 기다리고 켜지면 `JudgeCancelledError`."""
        if cancel is None:
            self.sleep(seconds)
        elif cancel.wait(seconds):
            raise JudgeCancelledError("실행이 멈춰 다시 묻지 않습니다")

    def _acquire(self, affinity: str, cancel: threading.Event | None = None) -> _Endpoint:
        """보낼 엔드포인트를 정하고 처리 중 수를 올린다 (여유가 없거나 모두 막히면 기다린다).

        쉼(`breaker_cooldown_s`)이 끝난 열린 엔드포인트가 있으면 먼저 반개방 확인을 하고 이 요청을
        시험 요청으로 보낸다 (회복한 장비를 다시 돌린다). 그다음 닫힌 엔드포인트(`_pick`). 기다림은
        최대 0.5초씩이라 멈춤 신호(`cancel`)를 그 간격으로 본다.

        Raises:
            JudgeCancelledError: 멈춤 신호가 켜졌다 (보내지 않는다).
            AllEndpointsDownError: 모두 `abort_after_s`보다 오래 막힘.
            VlmRequestError: 반개방 확인(`ping`)이 설정 오류.
        """
        with self._cond:
            while True:
                if cancel is not None and cancel.is_set():
                    raise JudgeCancelledError("실행이 멈춰 새 요청을 보내지 않습니다")
                if self._dead is not None:
                    raise AllEndpointsDownError(self._dead)
                trial = self._half_open()
                if trial is not None:
                    return trial
                picked = self._pick(affinity)
                if picked is not None:
                    picked.in_flight += 1
                    return picked
                self._wait_or_abort()

    def _half_open(self) -> _Endpoint | None:
        """쉼이 끝난 열린 엔드포인트의 반개방 확인 (잠금 안, 없으면 None).

        ping이 있으면 잠금을 풀고 먼저 확인한다 (실패면 다시 쉬고 None). ping이 통과했거나 없으면 이
        요청을 시험 요청으로 그 엔드포인트에 보낸다. 시험 요청 결과가 차단기를 닫거나 다시 연다
        (`_release`). ping만 통과하고 채팅이 실패하는 서버(메모리 부족 등)도 닫히지 않는다.
        """
        now = self.clock()
        cool = self.server.breaker_cooldown_s
        due = [
            e
            for e in self._eps
            if not e.probing and e.opened_at is not None and now - e.opened_at >= cool
        ]
        if not due:
            return None
        ep = due[0]
        ep.probing = True
        if ep.member.ping is not None:
            self._cond.release()
            try:
                ping_ok = _try_ping(ep.member.ping)
            except BaseException:
                self._cond.acquire()
                ep.probing, ep.opened_at = False, self.clock()
                raise
            self._cond.acquire()
            if not ping_ok:
                ep.probing, ep.opened_at = False, self.clock()
                self._cond.notify_all()
                return None
        ep.in_flight += 1
        return ep

    def _wait_or_abort(self) -> None:
        """보낼 곳이 없을 때 (잠금 안): 기다린다. 모두 막힌 지 `abort_after_s`가 지났으면 멈춘다.

        Raises:
            AllEndpointsDownError: 모두 막힌 지 `abort_after_s`가 지났을 때.
        """
        now = self.clock()
        if any(e.opened_at is None for e in self._eps):
            self._cond.wait(timeout=0.5)  # 닫힌 곳이 모두 상한까지 바쁘다
            return
        since = self._all_open_since if self._all_open_since is not None else now
        self._all_open_since = since
        abort = self.server.abort_after_s
        if now - since > abort:
            names = ", ".join(e.member.name for e in self._eps)
            self._dead = f"모든 엔드포인트가 {abort:g}초 넘게 응답하지 않습니다 ({names})"
            self._cond.notify_all()
            raise AllEndpointsDownError(self._dead)
        cool = self.server.breaker_cooldown_s
        waits = [
            e.opened_at + cool - now for e in self._eps if e.opened_at is not None and not e.probing
        ]
        self._cond.wait(timeout=max(0.01, min([*waits, since + abort - now, 0.5])))

    def _release(self, ep: _Endpoint, *, ok: bool | None) -> None:
        """요청이 끝났다: 처리 중 수를 내리고 차단기 상태를 바꾼다 (ok None은 상태를 바꾸지 않는다).

        성공이면 연속 장애를 0으로 하고 (시험 요청이었으면) 차단기를 닫는다. 장애면 연속 장애를
        올리고, 시험 요청이었거나 `breaker_failures`에 닿으면 차단기를 (다시) 연다.
        """
        with self._cond:
            ep.in_flight -= 1
            trial, ep.probing = ep.probing, False
            if ok is True:
                ep.failures, ep.opened_at = 0, None
                self._all_open_since = None
            elif ok is False:
                ep.failures += 1
                if trial or (ep.opened_at is None and ep.failures >= self.server.breaker_failures):
                    ep.opened_at = self.clock()
                if all(e.opened_at is not None for e in self._eps) and self._all_open_since is None:
                    self._all_open_since = self.clock()
            elif trial and ep.opened_at is not None:
                ep.opened_at = self.clock()  # 시험 요청이 장애 아닌 예외로 끝남: 다시 쉰다
            self._cond.notify_all()


class CancellableBackend:
    """멈춤 신호를 보는 백엔드 감싸기 (실행기가 실행마다 하나 만든다, `JudgeBackend` 구현).

    신호가 이미 켜졌으면 보내지 않고 `JudgeCancelledError`를 낸다. 안쪽이 `EndpointPool`이면 보낼
    곳을 기다리거나 백오프하는 동안에도 신호를 본다 (`EndpointPool.judge(cancel=…)`). 이미 보낸
    요청은 끝까지 기다린다. 묶음은 여러 세션 실행이 함께 쓰므로 신호는 요청마다 넘긴다 (한 세션이
    멈춰도 다른 세션의 요청은 그대로다).
    """

    def __init__(self, inner: JudgeBackend, cancel: threading.Event) -> None:
        """inner: 실제 백엔드. cancel: 실행기 멈춤 신호."""
        self.inner = inner
        self.cancel = cancel
        self.info = inner.info

    def judge(self, request: JudgeRequest) -> JudgeReply:
        """신호를 보며 묻는다 (클래스 docstring). 오류는 `EndpointPool.judge`와 같다."""
        if self.cancel.is_set():
            raise JudgeCancelledError("실행이 멈춰 새 요청을 보내지 않습니다")
        if isinstance(self.inner, EndpointPool):
            return self.inner.judge(request, cancel=self.cancel)
        return self.inner.judge(request)


def _try_ping(ping: Callable[[], None]) -> bool:
    """반개방 확인: 장애면 False, 성공이면 True (설정 오류는 올린다)."""
    try:
        ping()
    except VlmRequestError:
        raise
    except VlmUnavailableError:
        return False
    return True


def _affinity_slot(affinity: str, n: int) -> int:
    """창 키 → 엔드포인트 번호 (프로세스와 무관한 해시)."""
    digest = hashlib.sha256(affinity.encode()).digest()
    return int.from_bytes(digest[:8], "big") % n
