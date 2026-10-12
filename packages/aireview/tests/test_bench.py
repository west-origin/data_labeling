"""처리량 벤치(`dlp_aireview.bench`) 테스트 (WP17.3, ADR 0040 5.6).

합성 장면(`dlp_fixtures.aireview`)과 가짜 서버(`FakeOpenAIServer`, 요청마다 지연)로 실제 계획·
프롬프트·재검사 경로를 돈다. 정답 근거: 가짜 서버 지연으로 정해지는 동시성별 처리량 순서, 예약한
실패 수, 가짜 서버의 조각 단위 접두사 캐시(같은 요청 다시 = 전부 적중, 맨 앞 난수 = 0).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import replace
from fractions import Fraction
from typing import Any

import numpy as np
import pytest

from dlp_aireview.backend import OpenAICompatibleJudge
from dlp_aireview.bench import (
    BenchGrid,
    BenchInput,
    BenchPoint,
    BenchReport,
    bench_policy,
    bench_settings,
    group_settings,
    knee,
    run_bench,
)
from dlp_aireview.plan import VIDEO_KINDS, StreamMedia
from dlp_aireview.policy import AiReviewPolicy
from dlp_aireview.probe import server_info
from dlp_aireview.sensors import SensorContext
from dlp_aireview.testing import FakeCall, FakeOpenAIServer, text_tokens
from dlp_fixtures.aireview import AiReviewScene, aireview_scene, scene_frame
from dlp_media.pts import PtsIndex
from dlp_models.vlm import VlmRequestError
from dlp_schema.aireview import AiGroup
from dlp_schema.ontology import Ontology

MODEL = "Qwen/Qwen3-VL-8B-Instruct"
T, L, S = AiGroup.TEMPORAL, AiGroup.TEMPORAL_LONG, AiGroup.SPATIAL


def bench_input(scene: AiReviewScene, ontology: Ontology) -> BenchInput:
    """합성 장면 → 벤치 입력 (영상 스트림마다 같은 PTS, 프레임은 그때그때 그린다)."""
    times = np.asarray(scene.frame_times, dtype=np.int64)
    pts = PtsIndex(Fraction(1, 1000), times, np.ones(len(times), dtype=bool))
    media = {
        s.stream_id: StreamMedia(pts, "synthetic")
        for s in scene.session.streams
        if s.kind in VIDEO_KINDS
    }

    def frames(stream: str, idx: Sequence[int]) -> list[np.ndarray]:
        return [scene_frame(scene, stream, scene.frame_times[i]) for i in idx]

    return BenchInput(
        scene.session, scene.labels, media, ontology, SensorContext.absent(), frames, "synthetic"
    )


@pytest.fixture(scope="module")
def data(ontology: Ontology) -> BenchInput:
    """작은 합성 장면 (320x240)."""
    return bench_input(aireview_scene(0, n_units=4, width=320, height=240), ontology)


def _grid(**kw: object) -> BenchGrid:
    """작은 격자 (기본: temporal·spatial, 정책 값 설정만, 동시성 1·4, 단위 8개)."""
    values: dict[str, object] = {
        "concurrency": (1, 4), "frames": (), "sides": (), "groups": (T, S),
        "requests": 8, "warmup": 1, "ttft": 2, "cache_probe": True, "decode_tokens": 16,
    }  # fmt: skip
    values.update(kw)
    return BenchGrid(**values)  # pyright: ignore[reportArgumentType]


def _run(
    server: FakeOpenAIServer, data: BenchInput, policy: AiReviewPolicy, grid: BenchGrid
) -> BenchReport:
    """가짜 서버로 벤치."""
    judge = OpenAICompatibleJudge("fake-01", server.url, policy)
    info = server_info("fake-01", judge.chat, policy.served_model)
    return run_bench(judge, data, policy, grid, server=info, device="fake-01")


# 측정 표식 (`(측정 <토큰>-<설정 키>-<용도>)`)과 점 요청의 용도 (`c<동시성>-<번호>`)
_MARK = re.compile(r"\(측정 ([^)]+)\)$")
_POINT = re.compile(r"-c\d+-\d+$")


def _content(call: FakeCall) -> list[dict[str, Any]]:
    """첫 user 메시지의 조각 (고정 블록 → 이미지 → 가변 글)."""
    return call.body["messages"][1]["content"]


def _mark(call: FakeCall) -> str | None:
    """요청의 측정 표식 (고정 블록 끝, 없으면 None)."""
    match = _MARK.search(str(_content(call)[0].get("text", "")))
    return match.group(1) if match else None


def _images(call: FakeCall) -> tuple[str, ...]:
    """요청의 이미지 data URI 묶음."""
    return tuple(p["image_url"]["url"] for p in _content(call) if p["type"] == "image_url")


def test_bench_measures_points_knee_and_cache(data: BenchInput, policy: AiReviewPolicy) -> None:
    """동시성이 높을수록 처리량이 오르고(지연 30 ms), 무릎은 장애 0인 최고 처리량 점이다. 격자가
    비면 정책 값 설정만 잰다."""
    with FakeOpenAIServer(MODEL, latency_s=0.03) as server:
        report = _run(server, data, policy, _grid())
        assert 2 <= server.max_in_flight <= 4
    assert [(c.group, c.frames, c.side) for c in report.configs] == [(T, 8, 448), (S, 4, 672)]
    for config in report.configs:
        one, four = config.points
        assert config.planned_units > 0 and config.stopped is None
        assert (one.ok, one.unavailable, four.ok, four.unavailable) == (8, 0, 8, 0)
        assert four.units_per_s > 1.5 * one.units_per_s
        assert config.knee == four
        assert four.schema_ok_ratio == 1.0 and four.retry_ratio == 0 and four.fallback_ratio == 0
        assert four.prompt_tokens > 0 and four.prefill_tok_s > 0 and four.decode_tok_s > 0
        assert four.cache_hit_ratio is not None and 0 < four.cache_hit_ratio < 1
        assert config.ttft_p50_ms is not None and config.single_prefill_tok_s
        cache = config.cache
        assert cache is not None and cache.samples > 0
        assert cache.warm_cached_tokens == cache.prompt_tokens  # 같은 요청 다시 = 전부 적중
        assert cache.cold_cached_tokens == 0  # 맨 앞 난수 = 적중 없음
    assert report.server.served_model == MODEL and report.device == "fake-01"
    assert report.policy["verdict_version"] == policy.verdict_version
    assert report.policy["align_px"] == policy.image.align_px
    assert BenchReport.model_validate(report.model_dump(mode="json")) == report


def test_marker_ends_the_static_block_and_images_are_fresh(
    data: BenchInput, policy: AiReviewPolicy
) -> None:
    """측정 요청: 표식은 고정 블록(온톨로지) 끝에만 있고, system은 모두 같으며, 서버 접두사 캐시는
    system 조각에만 맞는다 (온톨로지 뒤의 이미지·가변 글은 매번 새로 계산). 계획 단위보다 요청이
    많아도 점 요청의 이미지 묶음은 모두 다르다 (서버 이미지 캐시가 인코딩을 건너뛰지 않는다)."""
    with FakeOpenAIServer(MODEL) as server:
        report = _run(server, data, policy, _grid(capacity=False))
        calls = server.chat_calls
    points = [c for c in calls if (m := _mark(c)) and _POINT.search(m)]
    assert len(points) == sum(p.units for c in report.configs for p in c.points)
    for call in points:
        content = _content(call)
        assert re.search(r"\n\(측정 [0-9a-f]{8}-[a-z_]+-f\d+-s\d+-c\d+-\d+\)$", content[0]["text"])
        assert "(측정" not in content[-1]["text"]  # 가변 글 끝에는 표식이 없다
    systems = {str(c.body["messages"][0]["content"]) for c in calls}
    assert (
        len({s for s in systems if not s.startswith("(측정")}) == 1
    )  # cold 요청만 system이 다르다
    sys_tokens = text_tokens(next(s for s in systems if not s.startswith("(측정")))
    for config in report.configs:
        assert config.planned_units < 8  # 단위를 돌려 쓴다
        for point in config.points:
            assert point.cached_tokens == point.requests * sys_tokens  # system 조각만 적중
            assert point.distinct_image_sets == point.units
    image_sets = [_images(c) for c in points]
    assert len(set(image_sets)) == len(image_sets)
    assert len({url for urls in image_sets for url in urls}) == sum(map(len, image_sets))


def test_capacity_runs_prefill_and_forced_decode(data: BenchInput, policy: AiReviewPolicy) -> None:
    """무릎 동시성에서 프리필(생성 1토큰, 새 이미지)과 생성(길이 고정, 구조화 출력 없음)을 따로
    잰다. 근사 토큰은 서버가 센 토큰과 표식 몇 토큰 차이다 (같은 이미지 셈법)."""
    grid = _grid(groups=(T,), concurrency=(2,), ttft=0, cache_probe=False, decode_tokens=16)
    with FakeOpenAIServer(MODEL) as server:
        report = _run(server, data, policy, grid)
        calls = server.chat_calls
    [config] = report.configs
    pre, dec, approx = config.prefill, config.decode, config.approx
    assert pre is not None and dec is not None and approx is not None
    assert (pre.concurrency, pre.requests, pre.ok, pre.unavailable) == (2, 8, 8, 0)
    assert pre.fresh_tokens == pytest.approx(8 * (approx.image + approx.text))
    assert pre.static_tokens == pytest.approx(8 * approx.static)
    assert abs(pre.prompt_tokens - (pre.fresh_tokens + pre.static_tokens)) < 8 * 20
    assert (dec.concurrency, dec.requests, dec.ok, dec.forced_tokens) == (2, 4, 4, 16)
    assert dec.honored and dec.completion_tokens == 4 * 16 and dec.error is None
    prefill_calls = [c for c in calls if (m := _mark(c)) and "-prefill-" in m]
    assert len(prefill_calls) == 8 and all(c.body["max_tokens"] == 1 for c in prefill_calls)
    assert len({_images(c) for c in prefill_calls}) == 8
    decode_calls = [c for c in calls if c.body.get("ignore_eos")]
    assert len(decode_calls) == 1 + 4  # 캐시에 올리는 첫 요청 + 측정 요청
    for call in decode_calls:
        assert call.body["min_tokens"] == call.body["max_tokens"] == 16
        assert "response_format" not in call.body
        assert re.search(r"\(측정 [^)]+-(prime|decode-\d+)\)$", _content(call)[-1]["text"])
    assert len({_images(c) for c in decode_calls}) == 1  # 같은 단위 (프롬프트가 캐시에 맞는다)


def test_decode_run_records_rejected_or_ignored_length(
    data: BenchInput, policy: AiReviewPolicy
) -> None:
    """서버가 길이 고정 확장을 거부하면 error, 무시하면 honored=False (벤치는 계속된다)."""
    grid = _grid(groups=(T,), concurrency=(1,), ttft=0, cache_probe=False, warmup=0,
                 decode_tokens=256)  # fmt: skip
    with FakeOpenAIServer(MODEL, extra_params=False) as server:
        rejected = _run(server, data, policy, grid).configs[0].decode
    assert rejected is not None and rejected.error and not rejected.honored

    with FakeOpenAIServer(MODEL, honor_ignore_eos=False) as server:
        ignored = _run(server, data, policy, grid).configs[0].decode
    assert ignored is not None and ignored.ok == ignored.requests and ignored.error is None
    assert not ignored.honored and ignored.completion_tokens < ignored.requests * 256
    with FakeOpenAIServer(MODEL) as server:
        off = _run(server, data, policy, replace(grid, capacity=False)).configs[0]
        assert not [c for c in server.chat_calls if c.body.get("ignore_eos")]
    assert off.decode is None and off.prefill is None


def _invalid_first(bad: set[int], always: set[int]) -> Callable[[FakeCall], str]:
    """점 요청 번호 k가 bad면 첫 시도만, always면 모든 시도에서 형식 틀린 JSON을 내는 응답 함수.

    그 밖에는 모든 대상 ok (대상 수는 첫 user 메시지 가변 글의 "대상 #0~#N"에서 읽는다).
    """

    def respond(call: FakeCall) -> str:
        mark = _mark(call) or ""
        k = int(m.group(1)) if (m := re.search(r"-c\d+-(\d+)$", mark)) else -1
        retry = any(msg["role"] == "assistant" for msg in call.body["messages"])
        if k in always or (k in bad and not retry):
            return "형식 틀린 JSON"
        found = re.search(r"대상 #0~#(\d+)를 판정해", _content(call)[-1]["text"])
        n = int(found.group(1)) + 1 if found else 0
        return json.dumps({"ok": list(range(n)), "items": [], "missing": []})

    return respond


def test_bench_counts_invalid_retries_fallback_and_context_errors(
    data: BenchInput, policy: AiReviewPolicy
) -> None:
    """스키마 적합률·재시도율·fallback율·문맥 초과 수는 판정 집계에서 나온다 (손 계산과 같다).

    동시성 1, 단위 6개: 0·3번은 첫 응답만 형식 틀림(재시도 1, 위반 1), 4번은 두 번 다 틀림(재시도 1,
    위반 2, fallback), 1번의 첫 요청은 문맥 초과(400)라 프레임을 줄여 다시 묻는다.
    시도 = 2+2+1+2+2+1 = 10, 응답 = 10 - 1 = 9, 위반 응답 = 4.
    """
    grid = _grid(groups=(T,), concurrency=(1,), requests=6, warmup=0, ttft=0,
                 cache_probe=False, capacity=False)  # fmt: skip
    with FakeOpenAIServer(MODEL) as server:
        server.respond_with(_invalid_first(bad={0, 3}, always={4}))
        server.fail(400, times=1, after=2, body="maximum context length is 10 tokens")
        report = _run(server, data, policy, grid)
    [point] = report.configs[0].points
    assert (point.units, point.ok, point.requests) == (6, 6, 10)
    assert point.context_errors == 1
    assert point.schema_ok_ratio == pytest.approx((9 - 4) / 9)
    assert point.retry_ratio == pytest.approx(3 / 6)
    assert point.fallback_ratio == pytest.approx(1 / 6)


def test_overload_errors_are_counted_and_down_server_stops(
    data: BenchInput, policy: AiReviewPolicy
) -> None:
    """과부하 503은 그 점의 장애로 세어 무릎에서 빠지고, 한 점이 모두 실패하면 벤치를 멈춘다."""
    grid = _grid(groups=(T,), warmup=0, ttft=0, cache_probe=False, requests=4, concurrency=(1, 2))
    with FakeOpenAIServer(MODEL) as server:
        server.fail(503, times=2)
        report = _run(server, data, policy, grid)
    one, two = report.configs[0].points
    assert (one.ok, one.unavailable) == (2, 2)
    assert (two.ok, two.unavailable) == (4, 0)
    assert report.configs[0].knee == two and not report.server_down
    with FakeOpenAIServer(MODEL) as server:
        server.fail(503, times=1000)
        down = _run(server, data, policy, _grid(warmup=0, ttft=0, cache_probe=False))
    assert down.server_down
    assert len(down.configs) == 1 and down.configs[0].stopped
    assert down.configs[0].knee is None


def test_config_error_raises(data: BenchInput, policy: AiReviewPolicy) -> None:
    """응답 모델이 정책 모델과 다르면 설정 오류를 그대로 올린다 (CLI 종료 5)."""
    with FakeOpenAIServer(MODEL, served_model="alias") as server, pytest.raises(VlmRequestError):
        _run(server, data, policy, _grid(groups=(T,)))


def test_group_without_units_is_recorded_empty(ontology: Ontology, policy: AiReviewPolicy) -> None:
    """대상 라벨이 없으면 그 설정은 단위 0으로 남고 서버에 묻지 않는다."""
    scene = aireview_scene(0, n_units=2)
    empty = bench_input(scene, ontology)
    empty = BenchInput(
        empty.session, (), empty.media, ontology, empty.sensor, empty.frames, "synthetic"
    )
    with FakeOpenAIServer(MODEL) as server:
        report = _run(server, empty, policy, _grid(groups=(L,), sides=(448,)))
        assert server.chat_calls == []
    assert [(c.side, c.planned_units, c.points) for c in report.configs] == [
        (336, 0, ()),  # 정책 값 (temporal_long 긴 변 336)
        (448, 0, ()),
    ]


def test_bench_policy_and_settings(policy: AiReviewPolicy) -> None:
    """프레임 수는 temporal에만 (상한을 올린다), 긴 변은 그 묶음의 max_side_px."""
    p = bench_policy(policy, T, 16, 336)
    assert (p.temporal.frames, p.temporal.max_frames, p.temporal.max_side_px) == (16, 16, 336)
    assert group_settings(p, T) == (16, 336)
    assert group_settings(bench_policy(policy, L, None, 512), L) == (
        policy.temporal.long_frames,
        512,
    )
    assert group_settings(bench_policy(policy, S, None, 960), S) == (policy.spatial.frames, 960)
    assert p.digest != policy.digest  # 벤치 정책은 판정 버전이 다르다 (DB에 쓰지 않는다)


def test_settings_always_include_policy_values(policy: AiReviewPolicy) -> None:
    """격자 후보에 지금 정책 값(temporal 8장·448, temporal_long 336, spatial 672)이 늘 더해진다.
    격자가 비면 정책 값만이다 (추정이 정책 값과 같은 설정의 측정만 쓴다)."""
    grid = _grid(groups=(T, L, S), frames=(12,), sides=(336,))
    assert [(g, f, side) for g, f, side, _ in bench_settings(policy, grid)] == [
        (T, 8, 336), (T, 8, 448), (T, 12, 336), (T, 12, 448),
        (L, 8, 336),
        (S, 4, 336), (S, 4, 672),
    ]  # fmt: skip
    empty = _grid(groups=(T, L, S))
    assert [(g, f, side) for g, f, side, _ in bench_settings(policy, empty)] == [
        (T, *group_settings(policy, T)), (L, *group_settings(policy, L)),
        (S, *group_settings(policy, S)),
    ]  # fmt: skip


def test_knee_rules() -> None:
    """장애가 있거나 p95가 시간 초과 이상인 점은 빼고, 처리량이 같으면 낮은 동시성."""

    def pt(c: int, ups: float, p95: int, bad: int = 0) -> BenchPoint:
        return BenchPoint(
            concurrency=c, units=10, ok=10 - bad, unavailable=bad, wall_s=1.0, units_per_s=ups,
            latency_p95_ms=p95,
        )  # fmt: skip

    points = [pt(1, 1.0, 900), pt(8, 5.0, 2000), pt(16, 5.0, 2500), pt(32, 9.0, 4000, bad=1)]
    assert knee(points, timeout_s=3) == points[1]
    assert knee([pt(64, 9.0, 3000)], timeout_s=3) is None
    assert knee([], timeout_s=3) is None
