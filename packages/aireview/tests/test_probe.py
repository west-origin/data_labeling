"""VLM 서버 점검(`dlp_aireview.probe`) 테스트 (WP17.3, ADR 0040 5.7·I7).

가짜 서버(`FakeOpenAIServer`)의 성질을 바꿔 가며 점검 결과·권고·종료 판정을 본다. 정답 근거: 가짜
서버에 준 설정(강제 여부, 이미지 상한, 키워드 거부, cached_tokens 제공, 모델 이름)과 정책 값.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from dlp_aireview.policy import AiReviewPolicy
from dlp_aireview.probe import (
    ProbeReport,
    context_floor,
    host_of,
    image_tokens,
    images_needed,
    probe_server,
)
from dlp_aireview.testing import FakeOpenAIServer
from dlp_schema.ontology import Ontology

from .conftest import with_policy

MODEL = "Qwen/Qwen3-VL-8B-Instruct"
NOW = datetime(2026, 10, 12, 9, 0, tzinfo=UTC)


def _probe(
    server: FakeOpenAIServer, policy: AiReviewPolicy, ontology: Ontology, **kw: Any
) -> ProbeReport:
    """엔드포인트 spark-01로 점검."""
    return probe_server("spark-01", server.url, policy, ontology, now=lambda: NOW, **kw)


def test_healthy_server_passes_all_checks(policy: AiReviewPolicy, ontology: Ontology) -> None:
    """기본 가짜 서버: 모든 점검 통과, 서버 정보·이미지 수·보고서 형식."""
    with FakeOpenAIServer(MODEL, context_tokens=32_768) as server:
        report = _probe(server, policy, ontology)
    assert not report.failed and not report.unavailable
    assert [c.name for c in report.checks] == [
        "internal_host", "models", "engine", "tokenize", "image_input", "image_limit",
        "schema_enforced", "full_profile", "cached_tokens", "context",
    ]  # fmt: skip
    assert {c.status for c in report.checks} <= {"ok", "info"}
    assert report.server.models == (MODEL,)
    assert report.server.engine_version == "0.0.0-fake"
    assert report.server.max_model_len == 32_768
    assert report.images_needed == images_needed(policy) == 12
    assert report.host == host_of(server.url)
    assert report.created_at == NOW
    # portable 정책 + full 지원 서버: full 가능 권고만 (실패 아님)
    assert any("schema_profile: full" in r for r in report.recommendations)
    data = report.model_dump(mode="json")
    assert data["kind"] == "aireview-probe" and ProbeReport.model_validate(data) == report


def test_outside_host_sends_nothing(policy: AiReviewPolicy, ontology: Ontology) -> None:
    """호스트 이름이 사내 대역 밖 주소로 풀리면 요청을 하나도 보내지 않고 실패 (I7)."""

    def public(host: str, port: int | None) -> list[tuple[Any, ...]]:
        return [(None, None, None, "", ("8.8.8.8", 0))]

    with FakeOpenAIServer(MODEL) as server:
        url = server.url.replace("127.0.0.1", "vlm.example")
        report = probe_server("ext", url, policy, ontology, resolve=public)
        assert server.calls == []
    assert report.failed and [c.name for c in report.checks] == ["internal_host"]
    assert "8.8.8.8" in report.checks[0].detail


def test_missing_model_or_wrong_key_stops_early(policy: AiReviewPolicy, ontology: Ontology) -> None:
    """서버 모델이 목록에 없거나 키가 틀리면 models 실패 뒤 채팅 요청을 보내지 않는다."""
    with FakeOpenAIServer("other/model") as server:
        report = _probe(server, policy, ontology)
        assert server.chat_calls == []
    assert report.status("models") == "fail" and report.failed
    assert any("served_name" in r for r in report.recommendations)
    with FakeOpenAIServer(MODEL, api_key="right") as server:
        denied = _probe(server, policy, ontology, api_key="wrong")
        ok = _probe(server, policy, ontology, api_key="right")
    assert denied.status("models") == "fail"
    assert not ok.failed


def test_reply_model_mismatch_fails(policy: AiReviewPolicy, ontology: Ontology) -> None:
    """목록에는 있지만 응답 model이 다르면 image_input 실패 (I7)."""
    with FakeOpenAIServer(MODEL, served_model="alias") as server:
        report = _probe(server, policy, ontology)
    assert report.status("image_input") == "fail" and report.failed
    assert report.status("schema_enforced") is None


def test_not_enforced_schema_and_image_limit(policy: AiReviewPolicy, ontology: Ontology) -> None:
    """구조화 출력 미강제는 정책 server에서 실패(prompt 권고), 이미지 상한이 작으면 실패."""
    with FakeOpenAIServer(MODEL, enforce_schema=False, max_images=4) as server:
        report = _probe(server, policy, ontology)
    assert report.status("schema_enforced") == "fail"
    assert report.status("image_limit") == "fail"
    assert any("schema_enforcement: prompt" in r for r in report.recommendations)
    assert any("limit-mm-per-prompt" in r for r in report.recommendations)
    prompt_policy = with_policy(policy, backend={"schema_enforcement": "prompt"})
    with FakeOpenAIServer(MODEL, enforce_schema=False) as server:
        relaxed = _probe(server, prompt_policy, ontology)
    assert relaxed.status("schema_enforced") == "info" and not relaxed.failed


def test_full_keywords_rejected(policy: AiReviewPolicy, ontology: Ontology) -> None:
    """범위·길이 키워드 거부: portable 정책이면 정보, full 정책이면 실패 + portable 권고."""
    with FakeOpenAIServer(MODEL, full_keywords=False) as server:
        portable = _probe(server, policy, ontology)
        full = _probe(server, with_policy(policy, backend={"schema_profile": "full"}), ontology)
    assert portable.status("full_profile") == "info" and not portable.failed
    assert full.status("full_profile") == "fail"
    assert any("schema_profile: portable" in r for r in full.recommendations)


def test_cache_and_context_warnings(policy: AiReviewPolicy, ontology: Ontology) -> None:
    """cached_tokens가 없으면 경고, 문맥 길이가 하한보다 작으면 경고 (둘 다 실패 아님)."""
    floor = context_floor(policy, ontology)
    with FakeOpenAIServer(MODEL, report_cached=False, context_tokens=floor - 1) as server:
        report = _probe(server, policy, ontology)
    assert report.status("cached_tokens") == "warn"
    assert report.status("context") == "warn"
    assert not report.failed
    assert any("--enable-prompt-tokens-details" in r for r in report.recommendations)


def test_unavailable_server_is_reported(policy: AiReviewPolicy, ontology: Ontology) -> None:
    """채팅 요청 중 503이면 그때까지의 점검과 서버 정보를 담아 unavailable로 돌려준다."""
    with FakeOpenAIServer(MODEL) as server:
        server.fail(503, times=10)
        report = _probe(server, policy, ontology)
    assert report.unavailable
    assert report.status("models") == "ok"
    assert report.server.models == (MODEL,)


def test_rejected_check_requests_are_recorded_not_raised(
    policy: AiReviewPolicy, ontology: Ontology
) -> None:
    """이미지 점검 뒤 구조화 출력 요청(3번째 채팅)이나 캐시 요청(5번째)이 4xx면 그 점검을 실패로
    남기고 보고서를 돌려준다 (예외로 끝나지 않는다, 앞선 점검 결과도 남는다)."""
    with FakeOpenAIServer(MODEL) as server:
        server.fail(400, times=1, after=2, body="response_format is not supported")
        schema = _probe(server, policy, ontology)
    assert schema.status("image_input") == "ok" and schema.status("image_limit") == "ok"
    assert schema.status("schema_enforced") == "fail" and schema.status("full_profile") is None
    assert schema.status("cached_tokens") == "ok"  # 다음 점검은 이어서 한다
    assert schema.failed and not schema.unavailable
    with FakeOpenAIServer(MODEL) as server:
        server.fail(400, times=1, after=4)
        cache = _probe(server, policy, ontology)
    assert cache.status("full_profile") == "ok"
    assert cache.status("cached_tokens") == "fail" and cache.status("context") == "ok"
    assert cache.failed and not cache.unavailable


def test_config_failure_before_outage_still_fails(
    policy: AiReviewPolicy, ontology: Ontology
) -> None:
    """설정 불일치(구조화 출력 미강제) 뒤 서버가 멈추면 unavailable이면서 failed다 (종료 5 > 3)."""
    with FakeOpenAIServer(MODEL, enforce_schema=False) as server:
        server.fail(503, times=10, after=3)
        report = _probe(server, policy, ontology)
    assert report.status("schema_enforced") == "fail" and report.status("unavailable") == "fail"
    assert report.unavailable and report.failed
    with FakeOpenAIServer(MODEL) as server:
        server.fail(503, times=10)
        down = _probe(server, policy, ontology)
    assert down.unavailable and not down.failed  # 장애만이면 설정 실패가 아니다


def test_size_helpers() -> None:
    """이미지 토큰 = 줄인 크기를 정렬 단위로 나눈 격자 수, 호스트에는 키·경로가 없다."""
    assert image_tokens(1920, 1080, 448, 32) == (448 // 32) * (252 // 32)
    assert image_tokens(100, 50, 448, 28) == 3 * 1  # 키우지 않는다 (84x28)
    assert host_of("http://user:key@10.0.0.11:8000/v1?x=1") == "10.0.0.11:8000"
    assert host_of("https://vlm.internal") == "vlm.internal"
