"""처리량 벤치(`dlp_aireview.bench`) 테스트 (WP17.3, ADR 0040 5.6).

합성 장면(`dlp_fixtures.aireview`)과 가짜 서버(`FakeOpenAIServer`, 요청마다 지연)로 실제 계획·
프롬프트·재검사 경로를 돈다. 정답 근거: 가짜 서버 지연으로 정해지는 동시성별 처리량 순서, 예약한
실패 수, 가짜 서버의 조각 단위 접두사 캐시(같은 요청 다시 = 전부 적중, 맨 앞 난수 = 0).
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

import numpy as np
import pytest

from dlp_aireview.backend import OpenAICompatibleJudge
from dlp_aireview.bench import (
    BenchGrid,
    BenchInput,
    BenchPoint,
    BenchReport,
    bench_policy,
    group_settings,
    knee,
    run_bench,
)
from dlp_aireview.plan import VIDEO_KINDS, StreamMedia
from dlp_aireview.policy import AiReviewPolicy
from dlp_aireview.probe import server_info
from dlp_aireview.sensors import SensorContext
from dlp_aireview.testing import FakeOpenAIServer
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
    """작은 격자 (기본: temporal·spatial, 동시성 1·4, 단위 8개)."""
    values: dict[str, object] = {
        "concurrency": (1, 4), "frames": (8,), "sides": (448,), "groups": (T, S),
        "requests": 8, "warmup": 1, "ttft": 2, "cache_probe": True,
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


def test_bench_measures_points_knee_and_cache(data: BenchInput, policy: AiReviewPolicy) -> None:
    """동시성이 높을수록 처리량이 오르고(지연 30 ms), 무릎은 장애 0인 최고 처리량 점이다."""
    with FakeOpenAIServer(MODEL, latency_s=0.03) as server:
        report = _run(server, data, policy, _grid())
        assert 2 <= server.max_in_flight <= 4
        measured = [c for c in server.chat_calls if "(측정 " in str(c.body)]
    assert [(c.group, c.frames, c.side) for c in report.configs] == [(T, 8, 448), (S, 4, 448)]
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
    # 측정 표식은 고정 블록 끝에 있어 system과 온톨로지 블록 앞부분이 모든 요청에서 같다
    firsts = {
        str(c.body["messages"][1]["content"][0]["text"]).split("\n(측정 ")[0] for c in measured
    }
    assert len({str(c.body["messages"][0]["content"]) for c in measured}) >= 1
    assert len(firsts) == 2  # temporal·spatial 고정 블록 두 가지
    assert all(c.unit_id and c.unit_id.startswith("air-") for c in measured)
    assert report.server.served_model == MODEL and report.device == "fake-01"
    assert report.policy["verdict_version"] == policy.verdict_version
    assert BenchReport.model_validate(report.model_dump(mode="json")) == report


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
        report = _run(server, empty, policy, _grid(groups=(L,)))
        assert server.chat_calls == []
    assert report.configs[0].planned_units == 0 and report.configs[0].points == ()


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
