"""여러 엔드포인트 묶음·차단기 테스트 (`backend.EndpointPool`, WP17.5, ADR 0040 5.3).

가짜 백엔드(정해 둔 순서로 성공·장애·설정 오류를 내는 stub)로 보낼 곳 고르기, 백오프 재시도, 차단기
열기·반개방·닫기, 모두 막혔을 때 중단, 엔드포인트별 동시 요청 상한을 본다. 시간 값은 작게 줄인
정책으로 실제 시계를 쓴다 (대기는 기록만 하는 sleep).

정답 근거: 정책 값(차단 연속 장애 수·쉼·중단 시간·백오프 목록)과 stub 호출 기록.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import pytest

from dlp_aireview.backend import (
    AllEndpointsDownError,
    BackendInfo,
    EndpointPool,
    JudgeReply,
    JudgeRequest,
    PoolMember,
)
from dlp_aireview.policy import AiReviewPolicy
from dlp_aireview.prompt import PromptParts
from dlp_models.vlm import VlmContextError, VlmRequestError, VlmUnavailableError
from dlp_schema.aireview import AiGroup

from .conftest import with_policy


class Stub:
    """정해 둔 결과를 차례로 내는 백엔드 (목록이 끝나면 `rest`를 계속 낸다).

    결과: "ok" | "down"(장애) | "bad"(설정 오류) | "ctx"(문맥 초과).
    """

    def __init__(
        self, name: str, script: tuple[str, ...] = (), rest: str = "ok", delay: float = 0.0
    ) -> None:
        self.name = name
        self.script = list(script)
        self.rest = rest
        self.delay = delay
        self.calls: list[str] = []
        self.info = BackendInfo(kind="stub", model="m", fingerprint="f", endpoints=(name,))
        self.in_flight = 0
        self.max_in_flight = 0
        self._lock = threading.Lock()

    def judge(self, request: JudgeRequest) -> JudgeReply:
        with self._lock:
            self.calls.append(request.affinity)
            outcome = self.script.pop(0) if self.script else self.rest
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if self.delay:
                time.sleep(self.delay)
            if outcome == "down":
                raise VlmUnavailableError(f"{self.name} 장애", retry_after_s=None)
            if outcome == "bad":
                raise VlmRequestError(f"{self.name} 설정 오류", status=401)
            if outcome == "ctx":
                raise VlmContextError("maximum context length", status=400)
            return JudgeReply(content="{}", served_model="m", endpoint=self.name)
        finally:
            with self._lock:
                self.in_flight -= 1


def request(affinity: str = "temporal|bodycam|0") -> JudgeRequest:
    """최소 판정 요청 (보낼 곳은 affinity로 정해진다)."""
    parts = PromptParts(system="s", ontology="o", images=(), text="t")
    return JudgeRequest(
        unit_id="air-" + "0" * 24, affinity=affinity, group=AiGroup.TEMPORAL, parts=parts,
        history=(), schema={}, schema_enforcement="server", max_tokens=64, targets=(),
        entities=(), frame_nos=(), frame_ms=(),
    )  # fmt: skip


def fast(policy: AiReviewPolicy, **server: object) -> AiReviewPolicy:
    """시간 값을 줄인 정책 (백오프 0.01·0.02초, 차단 2번, 쉼 0.05초, 중단 0.3초)."""
    values: dict[str, object] = {
        "backoff_s": (0.01, 0.02), "breaker_failures": 2, "breaker_cooldown_s": 0.05,
        "abort_after_s": 0.3, "concurrency_per_endpoint": 4,
    }  # fmt: skip
    return with_policy(policy, server=values | server)


def pool(
    policy: AiReviewPolicy, *stubs: Stub, ping: dict[str, Callable[[], None]] | None = None
) -> tuple[EndpointPool, list[float]]:
    """stub 묶음과 기록한 대기 시간."""
    sleeps: list[float] = []
    members = [PoolMember(s.name, s, (ping or {}).get(s.name)) for s in stubs]
    return EndpointPool(members, policy.server, sleep=sleeps.append), sleeps


def test_same_window_goes_to_same_endpoint(policy: AiReviewPolicy) -> None:
    """같은 창 키는 늘 같은 엔드포인트로, 여러 창 키는 두 엔드포인트에 나뉜다."""
    a, b = Stub("a"), Stub("b")
    p, _ = pool(fast(policy), a, b)
    for _ in range(5):
        p.judge(request("temporal|bodycam|10000"))
    assert (len(a.calls), len(b.calls)) in ((5, 0), (0, 5))
    for k in range(40):
        p.judge(request(f"temporal|bodycam|{k * 10000}"))
    assert a.calls and b.calls
    assert p.info.endpoints == ("a", "b") and p.states() == {"a": "ok", "b": "ok"}


def test_outage_retries_with_policy_backoff(policy: AiReviewPolicy) -> None:
    """장애 두 번 뒤 성공: 정책 백오프대로 기다리며 다시 묻고 결과를 돌려준다 (차단기는 닫힘)."""
    a = Stub("a", ("down", "down"))
    p, sleeps = pool(fast(policy, breaker_failures=5), a)
    assert p.judge(request()).endpoint == "a"
    assert len(a.calls) == 3 and sleeps == [0.01, 0.02]
    assert p.states() == {"a": "ok"}


def test_request_gives_up_after_backoff_list(policy: AiReviewPolicy) -> None:
    """요청 하나의 장애 시도는 len(backoff_s)+1번까지: 그 뒤 장애를 올린다 (전체 중단은 아님)."""
    a = Stub("a", rest="down")
    p, _ = pool(fast(policy, breaker_failures=50), a)
    with pytest.raises(VlmUnavailableError) as info:
        p.judge(request())
    assert not isinstance(info.value, AllEndpointsDownError)
    assert len(a.calls) == 3


def test_breaker_opens_and_traffic_moves(policy: AiReviewPolicy) -> None:
    """a가 연속 2번 장애면 차단기를 열고 요청은 b로 간다 (재시도도 b에서 성공)."""
    a, b = Stub("a", rest="down"), Stub("b")
    p, _ = pool(fast(policy, breaker_cooldown_s=60), a, b)
    for k in range(20):
        assert p.judge(request(f"k{k}")).endpoint == "b"
    assert len(a.calls) == 2 and p.states() == {"a": "open", "b": "ok"}


def test_half_open_ping_then_trial_closes(policy: AiReviewPolicy) -> None:
    """쉼이 끝나면 ping으로 확인하고 시험 요청 하나를 보낸다. 성공하면 차단기를 닫는다."""
    a = Stub("a", ("down", "down"))
    pings: list[str] = []
    p, _ = pool(fast(policy, abort_after_s=5), a, ping={"a": lambda: pings.append("a")})
    assert p.judge(request()).endpoint == "a"  # 장애 2번 → 열림 → 쉼 → ping → 시험 요청 성공
    assert pings == ["a"] and p.states() == {"a": "ok"}


def test_ping_ok_but_chat_down_stays_open_then_aborts(policy: AiReviewPolicy) -> None:
    """/v1/models는 되지만 채팅이 계속 장애인 서버: 시험 요청이 실패해 다시 열리고, 모두 막힌 지
    abort_after_s가 지나면 `AllEndpointsDownError`. 그 뒤 요청도 바로 같은 오류다."""
    a = Stub("a", rest="down")
    p, _ = pool(fast(policy, backoff_s=(0.0,) * 50), a, ping={"a": lambda: None})
    start = time.monotonic()
    with pytest.raises(AllEndpointsDownError):
        p.judge(request())
    assert 0.3 <= time.monotonic() - start < 3
    calls = len(a.calls)
    assert calls >= 3  # 열기 전 2번 + 시험 요청
    with pytest.raises(AllEndpointsDownError):
        p.judge(request())
    assert len(a.calls) == calls


def test_ping_failure_keeps_breaker_open(policy: AiReviewPolicy) -> None:
    """ping이 장애면 시험 요청을 보내지 않고 다시 쉰다 (채팅 호출 없음)."""
    a = Stub("a", ("down", "down"))

    def ping() -> None:
        raise VlmUnavailableError("ping 장애")

    p, _ = pool(fast(policy, backoff_s=(0.0,) * 50), a, ping={"a": ping})
    with pytest.raises(AllEndpointsDownError):
        p.judge(request())
    assert len(a.calls) == 2


def test_config_errors_pass_through_without_tripping(policy: AiReviewPolicy) -> None:
    """설정 오류·문맥 초과는 그대로 올라가고 차단기에 장애로 세지 않는다."""
    a = Stub("a", ("bad", "bad", "bad", "ctx", "ctx", "ctx"))
    p, sleeps = pool(fast(policy), a)
    for _ in range(3):
        with pytest.raises(VlmRequestError) as info:
            p.judge(request())
        assert not isinstance(info.value, VlmContextError)
    for _ in range(3):
        with pytest.raises(VlmContextError):
            p.judge(request())
    assert p.states() == {"a": "ok"} and sleeps == [] and len(a.calls) == 6


def test_per_endpoint_concurrency_cap(policy: AiReviewPolicy) -> None:
    """엔드포인트 하나에 동시에 보내는 요청은 `concurrency_per_endpoint`를 넘지 않는다."""
    a = Stub("a", delay=0.02)
    p, _ = pool(fast(policy, concurrency_per_endpoint=2), a)
    threads = [threading.Thread(target=p.judge, args=(request(f"k{k}"),)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert a.max_in_flight == 2 and len(a.calls) == 8 and p.in_flight == 0


def test_pool_rejects_duplicate_names(policy: AiReviewPolicy) -> None:
    """이름이 겹치거나 엔드포인트가 없으면 만들지 않는다."""
    with pytest.raises(ValueError):
        EndpointPool([PoolMember("a", Stub("a")), PoolMember("a", Stub("a"))], policy.server)
    with pytest.raises(ValueError):
        EndpointPool([], policy.server)
