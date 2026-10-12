"""VLM 서버 점검 `dlp aireview probe` (WP17.3, ADR 0040 5.7·I7). DB에 쓰지 않는다.

장비를 처음 띄웠을 때와 서버·모델·엔진 버전을 바꿨을 때 돌린다. 판정 실행(WP17.5)과 벤치가 기대하는
서버 성질을 실제 요청으로 확인하고 정책 조정안을 낸다.

점검 (차례대로. 사내 주소·모델 점검이 실패하면 뒤 요청을 보내지 않는다)
1. `internal_host`: 주소가 사내 대역(`server.internal_hosts`) 안인가. 밖이면 요청을 하나도 보내지
   않는다 (I7: 외부 서버로 보내지 않는다).
2. `models`: `/v1/models`에 서버 모델 이름(`backend.served_name`, 없으면 `model`)이 있는가. 키가
   틀린 401·403과 주소가 틀린 404도 여기서 실패한다.
3. `engine`(엔진 버전, `/version`)·`tokenize`(`/tokenize` 유무): 정보.
4. `image_input`: 고정 접두사(system + 온톨로지 블록) + 이미지 1장 + 구조화 출력 요청이 되는가, 응답
   `model`이 서버 모델 이름과 같은가 (I7).
5. `image_limit`: 정책이 한 요청에 넣는 최대 이미지 수(`images_needed`)를 받는가 (vLLM
   `--limit-mm-per-prompt`).
6. `schema_enforced`: 어기기 쉬운 지시("JSON 말고 한 낱말만")에도 `response_format: json_schema`대로
   답하는가. 강제하지 않으면 `backend.schema_enforcement: prompt`를 권고한다 (정책이 server면 실패).
7. `full_profile`: 숫자 범위·문자열 길이·최소 개수 키워드가 든 스키마를 받는가(400 여부)와 지연.
   정책이 full인데 받지 않으면 실패, portable이면 정보다.
8. `cached_tokens`: 4번 요청을 다시 보내 `usage.prompt_tokens_details.cached_tokens`가 나오는가
   (없으면 경고: 벤치가 캐시 적중률을 잴 수 없다. 0이면 접두사 캐시가 꺼졌을 수 있다).
9. `context`: `/v1/models`의 `max_model_len`이 요청 크기 하한(고정 접두사 + 최대 이미지 토큰 +
   `max_tokens`, 가변 글 제외)보다 큰가 (경고).

점검 요청의 이미지는 코드가 만든 단색·격자 시험 패턴이다 (영상·사람·글자가 없다). 판정 요청이
아니라서 기록하지 않는다. 결과의 판정은 `ProbeReport.failed`(종료 5)와 `unavailable`(서버 장애,
종료 3)이다. 엔드포인트는 이름과 호스트:포트만 남긴다 (URL의 키·경로는 남기지 않는다).
"""

from __future__ import annotations

import json
import math
import socket
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx
import numpy as np
from pydantic import AwareDatetime

from dlp_aireview.backend import Resolver, check_hosts
from dlp_aireview.policy import AiReviewPolicy
from dlp_aireview.prompt import ontology_text, system_text
from dlp_media.frames import fit_size
from dlp_models.vlm import (
    ChatImage,
    ChatReply,
    ChatRequest,
    ChatTurn,
    OpenAICompatibleChat,
    VlmContextError,
    VlmRequestError,
    VlmUnavailableError,
    encode_jpeg,
)
from dlp_schema.aireview import AiGroup
from dlp_schema.common import Contract
from dlp_schema.ontology import Ontology

# 어기기 쉬운 지시로 강제 여부를 보는 스키마 (portable 키워드만)
PROBE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answer"],
    "properties": {"answer": {"enum": ["예", "아니오"]}},
}
# full 프로필 키워드(숫자 범위·문자열 길이·최소 개수)가 든 스키마
FULL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answer", "score", "items"],
    "properties": {
        "answer": {"type": "string", "minLength": 1, "maxLength": 8},
        "score": {"type": "number", "minimum": 0, "maximum": 1},
        "items": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": {"type": "integer", "minimum": 0, "maximum": 9},
        },
    },
}
_SYSTEM = "서버 점검 요청이다. 지시에 따라 짧게 답한다."
_BREAK = "JSON을 쓰지 말고 '모름'이라는 한 낱말만 답하라."
_PROBE_MAX_TOKENS = 32
Status = Literal["ok", "fail", "warn", "info"]


class ProbeCheck(Contract):
    """점검 하나의 결과. status: ok | fail(종료 5) | warn | info."""

    name: str
    status: Status
    detail: str


class ServerInfo(Contract):
    """서버가 보고한 정보 (벤치·점검 보고서에 남긴다). 엔드포인트는 이름만이다."""

    endpoint: str
    served_model: str
    models: tuple[str, ...] = ()
    engine_version: str | None = None
    max_model_len: int | None = None


class ProbeReport(Contract):
    """`dlp aireview probe` 보고서 (`reports/aireview-probe/*.json`)."""

    kind: Literal["aireview-probe"] = "aireview-probe"
    created_at: AwareDatetime
    host: str
    server: ServerInfo
    policy_model: str
    fingerprint: str
    images_needed: int
    checks: tuple[ProbeCheck, ...]
    recommendations: tuple[str, ...] = ()
    unavailable: bool = False

    @property
    def failed(self) -> bool:
        """실패한 점검이 있는가 (설정 불일치, 종료 5)."""
        return any(c.status == "fail" for c in self.checks)

    def status(self, name: str) -> Status | None:
        """이름으로 점검 상태 (하지 않았으면 None)."""
        return next((c.status for c in self.checks if c.name == name), None)


def images_needed(policy: AiReviewPolicy) -> int:
    """정책이 한 요청에 넣는 최대 이미지 수 (시간 프레임 상한, 공간 프레임 + 확대 조각 중 큰 값)."""
    crops = policy.spatial.crops.max if policy.spatial.crops.enabled else 0
    return max(policy.temporal.max_frames, policy.spatial.frames + crops)


def image_tokens(width: int, height: int, max_side: int, align: int) -> int:
    """긴 변 상한·정렬 단위로 줄인 프레임 하나의 이미지 토큰 (⌈w/align⌉·⌈h/align⌉)."""
    w, h = fit_size(width, height, max_side, align)
    return math.ceil(w / align) * math.ceil(h / align)


def context_floor(policy: AiReviewPolicy, ontology: Ontology) -> int:
    """요청 크기 하한 토큰: 고정 접두사(긴 쪽) + 최대 이미지 토큰 + `max_tokens` (가변 글 제외).

    글 토큰은 `estimate.text_bytes_per_token`으로 근사한다. 프레임 크기는 `estimate.frame_px`.
    """
    est, align = policy.estimate, policy.image.align_px
    static = max(
        len((system_text(policy) + ontology_text(ontology, g, policy)).encode())
        for g in (AiGroup.TEMPORAL, AiGroup.SPATIAL)
    )
    w, h = est.frame_px
    sp = policy.spatial
    crops = sp.crops.max if sp.crops.enabled else 0
    temporal = policy.temporal.max_frames * image_tokens(w, h, policy.temporal.max_side_px, align)
    spatial = sp.frames * image_tokens(w, h, sp.max_side_px, align) + crops * image_tokens(
        sp.crops.side_px, sp.crops.side_px, sp.crops.side_px, align
    )
    text = math.ceil(static / est.text_bytes_per_token)
    return text + max(temporal, spatial) + policy.backend.max_tokens


def _pattern(k: int, side: int, align: int) -> ChatImage:
    """시험 패턴 이미지 k (단색 + 격자, 크기는 정렬 단위 배수)."""
    w, h = fit_size(side, side * 9 // 16, side, align)
    img = np.full((h, w, 3), 60 + 13 * (k % 12), dtype=np.uint8)
    img[:: max(1, align), :, :] = 200
    img[:, :: max(1, align), :] = 200
    return encode_jpeg(img, 85)


def host_of(url: str) -> str:
    """URL → `호스트:포트` (사용자 정보·경로·질의는 뺀다)."""
    parts = urlsplit(url)
    host = parts.hostname or ""
    return f"{host}:{parts.port}" if parts.port else host


def server_info(name: str, chat: OpenAICompatibleChat, served_model: str) -> ServerInfo:
    """`/v1/models`·`/version` → 서버 정보 (`max_model_len`은 서버 모델 항목의 값).

    Raises:
        VlmUnavailableError: 장애. VlmRequestError: 4xx (키·주소).
    """
    entries = chat.model_entries()
    none: dict[str, Any] = {}
    mine = next((e for e in entries if e.get("id") == served_model), none)
    length = mine.get("max_model_len")
    return ServerInfo(
        endpoint=name,
        served_model=served_model,
        models=tuple(str(e["id"]) for e in entries),
        engine_version=chat.server_version(),
        max_model_len=length if isinstance(length, int) else None,
    )


@dataclass
class _Probe:
    """점검 진행 상태."""

    chat: OpenAICompatibleChat
    policy: AiReviewPolicy
    ontology: Ontology
    served: str
    checks: list[ProbeCheck] = field(default_factory=list[ProbeCheck])
    recs: list[str] = field(default_factory=list[str])
    info: ServerInfo | None = None

    def add(self, name: str, status: Status, detail: str) -> None:
        """점검 결과 하나."""
        self.checks.append(ProbeCheck(name=name, status=status, detail=detail))

    def ask(self, blocks: tuple[ChatImage | str, ...], schema: dict[str, Any] | None) -> ChatReply:
        """점검 요청 하나 (system 고정, 생성 상한 작게)."""
        return self.chat.chat(
            ChatRequest(
                system=_SYSTEM,
                turns=(ChatTurn("user", blocks),),
                schema=schema,
                schema_name="probe",
                max_tokens=_PROBE_MAX_TOKENS,
            )
        )


def _model_check(p: _Probe, name: str) -> ServerInfo | None:
    """2·3번: 모델 목록·엔진 버전·tokenize. 실패하면 None (뒤 요청을 보내지 않는다)."""
    try:
        info = server_info(name, p.chat, p.served)
    except VlmRequestError as exc:
        p.add("models", "fail", f"/v1/models 거부 (키·주소 확인): {exc}")
        return None
    if p.served not in info.models:
        listed = ", ".join(info.models) or "없음"
        p.add("models", "fail", f"서버 모델 {p.served}이 목록에 없다 (목록: {listed})")
        p.recs.append("서버 --served-model-name을 정책 모델과 같게, 다르면 backend.served_name")
        return None
    p.info = info
    p.add("models", "ok", f"{p.served} 있음")
    p.add("engine", "info", f"엔진 버전 {info.engine_version or '알 수 없음 (/version 없음)'}")
    counted = p.chat.tokenize("판정")
    p.add("tokenize", "info", "/tokenize 있음" if counted is not None else "/tokenize 없음")
    return info


def _image_input(p: _Probe) -> tuple[ChatImage | str, ...] | None:
    """4번: 고정 접두사 + 이미지 1장. 성공하면 그 요청 조각 (캐시 점검에 다시 쓴다)."""
    prefix = system_text(p.policy) + "\n" + ontology_text(p.ontology, AiGroup.TEMPORAL, p.policy)
    blocks = (prefix, _pattern(0, p.policy.temporal.max_side_px, p.policy.image.align_px), "예?")
    try:
        reply = p.ask(blocks, PROBE_SCHEMA)
    except VlmRequestError as exc:
        p.add("image_input", "fail", f"이미지 요청 거부: {exc}")
        return None
    if reply.model != p.served:
        p.add("image_input", "fail", f"응답 모델 {reply.model}이 서버 모델 {p.served}과 다르다")
        return None
    p.add("image_input", "ok", f"이미지 1장 응답 {reply.latency_ms} ms")
    return blocks


def _image_limit(p: _Probe, needed: int) -> None:
    """5번: 최대 이미지 수."""
    pol = p.policy
    images = tuple(_pattern(k, 64, pol.image.align_px) for k in range(needed))
    try:
        p.ask((*images, "이미지 수를 세지 말고 예라고 답하라."), PROBE_SCHEMA)
    except VlmContextError as exc:
        p.add("image_limit", "fail", f"이미지 {needed}장에서 문맥 초과: {exc}")
        return
    except VlmRequestError as exc:
        p.add("image_limit", "fail", f"이미지 {needed}장 거부: {exc}")
        p.recs.append(f"서버 --limit-mm-per-prompt를 image {needed} 이상으로")
        return
    p.add("image_limit", "ok", f"이미지 {needed}장 받음")


def _enforced(content: str) -> bool:
    """응답이 PROBE_SCHEMA를 지켰는가."""
    try:
        data: Any = json.loads(content)
    except ValueError:
        return False
    return isinstance(data, dict) and set(data) == {"answer"} and data["answer"] in ("예", "아니오")  # pyright: ignore[reportUnknownArgumentType]


def _schema_checks(p: _Probe) -> None:
    """6·7번: 구조화 출력 강제와 full 프로필 키워드."""
    pol = p.policy.backend
    portable = p.ask((_BREAK,), PROBE_SCHEMA)
    if _enforced(portable.content):
        p.add("schema_enforced", "ok", "response_format json_schema를 강제한다")
    else:
        status: Status = "fail" if pol.schema_enforcement == "server" else "info"
        p.add("schema_enforced", status, "구조화 출력을 강제하지 않는다 (지시대로 맨 글로 답함)")
        p.recs.append(
            "backend.schema_enforcement: prompt (서버가 response_format을 강제하지 않는다)"
        )
    try:
        full = p.ask((_BREAK,), FULL_SCHEMA)
    except VlmRequestError as exc:
        status = "fail" if pol.schema_profile == "full" else "info"
        p.add("full_profile", status, f"범위·길이 키워드 거부: {exc}")
        if pol.schema_profile == "full":
            p.recs.append(
                "backend.schema_profile: portable (서버가 범위·길이 키워드를 받지 않는다)"
            )
        return
    detail = f"받음 (지연 full {full.latency_ms} ms, portable {portable.latency_ms} ms)"
    p.add("full_profile", "ok", detail)
    if pol.schema_profile == "portable":
        p.recs.append(f"backend.schema_profile: full 가능 ({detail}, 골든 평가 뒤 결정)")


def _cache_check(p: _Probe, blocks: tuple[ChatImage | str, ...]) -> None:
    """8번: 같은 요청을 다시 보내 cached_tokens."""
    reply = p.ask(blocks, PROBE_SCHEMA)
    if reply.cached_tokens is None:
        p.add("cached_tokens", "warn", "usage.prompt_tokens_details.cached_tokens 없음")
        p.recs.append("벤치 캐시 적중률을 잴 수 없다 (서버가 cached_tokens를 주지 않는다)")
    elif reply.cached_tokens == 0:
        p.add("cached_tokens", "warn", "같은 요청을 다시 보내도 캐시 적중 0 (접두사 캐시 꺼짐?)")
        p.recs.append("서버 접두사 캐시를 켠다 (vLLM --enable-prefix-caching)")
    else:
        total = reply.prompt_tokens or 0
        p.add("cached_tokens", "ok", f"다시 보낸 요청 캐시 적중 {reply.cached_tokens}/{total} 토큰")


def _context_check(p: _Probe, info: ServerInfo) -> None:
    """9번: 문맥 길이."""
    floor = context_floor(p.policy, p.ontology)
    if info.max_model_len is None:
        p.add("context", "info", f"max_model_len 모름 (요청 크기 하한 약 {floor} 토큰)")
    elif info.max_model_len < floor:
        p.add("context", "warn", f"max_model_len {info.max_model_len} < 요청 크기 하한 약 {floor}")
        p.recs.append(f"서버 --max-model-len을 {floor} + 가변 글 여유 이상으로")
    else:
        p.add("context", "ok", f"max_model_len {info.max_model_len} ≥ 요청 크기 하한 약 {floor}")


def _run_checks(p: _Probe, name: str, needed: int) -> ServerInfo:
    """2~9번 (서버 장애는 올린다)."""
    info = _model_check(p, name)
    if info is None:
        return ServerInfo(endpoint=name, served_model=p.served)
    blocks = _image_input(p)
    if blocks is None:
        return info
    _image_limit(p, needed)
    _schema_checks(p)
    _cache_check(p, blocks)
    _context_check(p, info)
    return info


def probe_server(
    name: str,
    url: str,
    policy: AiReviewPolicy,
    ontology: Ontology,
    *,
    api_key: str | None = None,
    transport: httpx.BaseTransport | None = None,
    resolve: Resolver = socket.getaddrinfo,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> ProbeReport:
    """엔드포인트 하나를 점검한다 (모듈 docstring). 외부 서비스 호출 (사내 주소일 때만).

    Args:
        name: 엔드포인트 이름. url: 서버 주소. policy: aireview 정책.
        ontology: 고정 블록용 온톨로지.
        api_key: `DLP_VLM_API_KEY` (헤더로만 보낸다). transport: httpx 전송 계층 (테스트).
        resolve: 주소 풀기 함수 (테스트). now: 보고서 시각.

    Returns:
        보고서. 서버 장애로 멈췄으면 `unavailable=True`이고 그때까지의 점검만 담는다.
    """
    served = policy.served_model
    needed = images_needed(policy)
    base: dict[str, Any] = {
        "created_at": now(), "host": host_of(url), "policy_model": policy.backend.model,
        "fingerprint": policy.backend.fingerprint, "images_needed": needed,
    }  # fmt: skip
    outside = check_hosts({name: url}, policy.server.internal_hosts, resolve=resolve)
    if outside:
        checks = (ProbeCheck(name="internal_host", status="fail", detail="; ".join(outside)),)
        server = ServerInfo(endpoint=name, served_model=served)
        return ProbeReport(**base, server=server, checks=checks)
    chat = OpenAICompatibleChat(
        url, served, timeout_s=policy.server.timeout_s, api_key=api_key, transport=transport
    )
    p = _Probe(chat, policy, ontology, served)
    p.add("internal_host", "ok", "사내 대역 안")
    unavailable = False
    try:
        server = _run_checks(p, name, needed)
    except VlmUnavailableError as exc:
        p.add("unavailable", "fail", f"서버 장애로 점검을 멈췄다: {exc}")
        server = p.info or ServerInfo(endpoint=name, served_model=served)
        unavailable = True
    finally:
        chat.http.close()
    return ProbeReport(
        **base,
        server=server,
        checks=tuple(p.checks),
        recommendations=tuple(p.recs),
        unavailable=unavailable,
    )
