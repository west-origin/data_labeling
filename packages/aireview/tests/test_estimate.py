"""작업량·장비 대수 추정(`dlp_aireview.estimate`) 테스트 (WP17.3, ADR 0040 5.8·8.3).

정답 근거: 같은 창 논리(`plan_windows`)의 창 수·대상 수, 손 계산한 이미지 토큰(1920x1080을 긴 변
448·정렬 32로 줄이면 448x224 → 14·7 = 98), 정책 출력 가정(대상당 2 + 0.15·60 = 11토큰), 설계
예시(영상 1시간당 530단위, 장비 1대 1단위/s, H 164·여유 0.2·가동률 0.7 → 2대).
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime

import pytest

from dlp_aireview.bench import BenchConfig, BenchPoint, BenchReport, DecodeMeasure
from dlp_aireview.estimate import (
    GroupLoad,
    SessionLoad,
    Workload,
    capacity,
    group_throughput,
    session_load,
    to_json,
)
from dlp_aireview.plan import plan_windows
from dlp_aireview.policy import AiReviewPolicy
from dlp_aireview.probe import ServerInfo
from dlp_fixtures.aireview import aireview_scene
from dlp_schema.aireview import AiGroup
from dlp_schema.ontology import Ontology
from dlp_schema.review import FlaggedSpan, ReviewReason
from dlp_schema.session import CameraIntrinsics

from .conftest import with_policy

T, L, S = AiGroup.TEMPORAL, AiGroup.TEMPORAL_LONG, AiGroup.SPATIAL


def test_session_load_counts_like_plan(policy: AiReviewPolicy, ontology: Ontology) -> None:
    """단위·대상 수는 실행 경로의 창과 같고, 이미지·출력 토큰은 손 계산과 같다."""
    scene = aireview_scene(0, n_units=6)
    load = session_load(scene.session, list(scene.labels), policy, ontology)
    windows = plan_windows(scene.session, list(scene.labels), policy, ontology)
    assert set(load.groups) == {T, L, S}
    for group, g in load.groups.items():
        mine = [w for w in windows if w.group is group]
        assert g.units == len(mine)
        assert g.targets == sum(len(w.targets) for w in mine)
        assert g.frames == sum(len({round(t) for t in w.wanted_ms}) for w in mine)
        assert g.output_tokens == pytest.approx(g.targets * (2 + 0.15 * 60))
        assert g.text_tokens > 0 and g.static_tokens > 0
        assert g.static_tokens % g.units == 0  # 단위마다 같은 고정 접두사
    assert load.groups[T].image_tokens == load.groups[T].frames * 14 * 7
    assert load.video_hours == pytest.approx(scene.session.duration_ms / 3_600_000)
    assert load.video_streams == 2


def test_calibrated_frame_size_and_flagged_mode(policy: AiReviewPolicy, ontology: Ontology) -> None:
    """기준 스트림에 캘리브레이션 크기가 있으면 그 크기를 쓰고, flagged 모드는 고른 단위만 센다."""
    scene = aireview_scene(0, n_units=6)
    intr = CameraIntrinsics(width=640, height=480, fx=500, fy=500, cx=320, cy=240)
    calibrated = scene.session.model_copy(
        update={"calibration": scene.session.calibration.model_copy(update={"intrinsics": intr})}
    )
    load = session_load(calibrated, list(scene.labels), policy, ontology)
    assert load.groups[T].image_tokens == load.groups[T].frames * 14 * 10  # 448x320 → 14·10
    flagged = with_policy(policy, selection={"mode": "flagged", "audit_ratio": 0.0})
    first = next(x for x in scene.labels if x.kind == "action")
    span = FlaggedSpan(
        reason=ReviewReason.LOW_CONFIDENCE,
        t_start_ms=first.t_start_ms,
        t_end_ms=first.t_end_ms,
        label_ids=(first.label_id,),
    )
    picked = session_load(scene.session, list(scene.labels), flagged, ontology, flagged=[span])
    assert picked.groups[T].units == 1
    assert sum(g.units for g in picked.groups.values()) < sum(
        g.units
        for g in session_load(scene.session, list(scene.labels), policy, ontology).groups.values()
    )


def _report(
    policy: AiReviewPolicy, points: dict[AiGroup, BenchPoint], decode_fps: float | None = None
) -> BenchReport:
    """묶음별 무릎 지점만 든 벤치 보고서 (지금 정책 설정 값으로)."""
    settings = {
        T: (policy.temporal.frames, policy.temporal.max_side_px),
        L: (policy.temporal.long_frames, policy.temporal.long_max_side_px),
        S: (policy.spatial.frames, policy.spatial.max_side_px),
    }
    configs = tuple(
        BenchConfig(
            group=g,
            frames=settings[g][0],
            side=settings[g][1],
            planned_units=1,
            points=(p,),
            knee=p,
        )
        for g, p in points.items()
    )
    decode = (
        DecodeMeasure(fps=decode_fps, frames=100, width=1920, height=1080, max_side=448)
        if decode_fps
        else None
    )
    return BenchReport(
        created_at=datetime(2026, 10, 12, tzinfo=UTC), device="spark-01", source="synthetic",
        server=ServerInfo(endpoint="spark-01", served_model="m"), policy={"model": "m"},
        grid={}, configs=configs, decode=decode,
    )  # fmt: skip


def _point(prefill: float, decode: float, cached: int | None = 0) -> BenchPoint:
    """무릎 지점 (처리량 1 단위/s, 동시성 32)."""
    return BenchPoint(
        concurrency=32, units=100, ok=100, unavailable=0, wall_s=100.0, units_per_s=1.0,
        latency_p95_ms=1000, prefill_tok_s=prefill, decode_tok_s=decode, cached_tokens=cached,
    )  # fmt: skip


def _workload(actions: int | None = None) -> Workload:
    """1시간 세션 하나: temporal 530단위 (단위당 이미지 1000, 글 200, 고정 3000, 출력 100)."""
    g = GroupLoad(
        units=530, targets=530 * 9, frames=530 * 8, image_tokens=530_000, text_tokens=106_000,
        static_tokens=530 * 3000, output_tokens=53_000.0,
    )  # fmt: skip
    return Workload((SessionLoad("s1", 1.0, 2, {T: g}, actions),))


def test_capacity_matches_design_example(policy: AiReviewPolicy) -> None:
    """프리필 한계 1단위/s(1200 tok/s ÷ 1200), 디코드 한계 2단위/s → T=1, 대수 = ceil(1.72) = 2."""
    report = _report(policy, {T: _point(prefill=1200, decode=200)}, decode_fps=600)
    cap = capacity(
        _workload(), report, policy, hours_per_day=164, rerun_margin=0.2, utilization=0.7,
        decode_workers=4,
    )  # fmt: skip
    [share] = cap.groups
    assert (share.group, share.units_per_hour, share.limit) == ("temporal", 530, "prefill")
    assert share.units_per_s == pytest.approx(1.0)
    assert cap.device_s_per_hour == pytest.approx(530)
    assert cap.devices == math.ceil(164 * 1.2 * 530 / (86_400 * 0.7)) == 2
    assert cap.missing == ()
    assert cap.decode_hours_per_day == pytest.approx(164 * 30 * 2 / (600 * 4))
    json.dumps(to_json(cap))
    json.dumps(to_json(_workload()))


def test_capacity_limits_missing_and_actions(policy: AiReviewPolicy) -> None:
    """캐시 적중을 모르면 고정 접두사도 프리필에 더하고, 디코드가 더 좁으면 디코드 한계. 측정이 없는
    묶음·행동 VLM은 missing이라 대수를 내지 않는다."""
    load = _workload().total(T)
    no_cache = group_throughput(_report(policy, {T: _point(4200, 1000, None)}), policy, T, load)
    assert no_cache is not None and no_cache.limit == "prefill"
    assert no_cache.units_per_s == pytest.approx(4200 / 4200)
    decode_bound = group_throughput(_report(policy, {T: _point(12_000, 50)}), policy, T, load)
    assert decode_bound is not None and decode_bound.limit == "decode"
    assert decode_bound.units_per_s == pytest.approx(0.5)
    assert group_throughput(_report(policy, {S: _point(1, 1)}), policy, T, load) is None
    report = _report(policy, {T: _point(1200, 200)})
    kw = {"hours_per_day": 164.0, "rerun_margin": 0.2, "utilization": 0.7, "decode_workers": 4}
    missing = capacity(_workload(actions=7200), report, policy, **kw)  # pyright: ignore[reportArgumentType]
    assert missing.missing == ("actions",) and missing.devices is None
    assert missing.decode_hours_per_day is None
    with_actions = capacity(_workload(actions=7200), report, policy, actions_req_s=1.5, **kw)  # pyright: ignore[reportArgumentType]
    assert with_actions.device_s_per_hour == pytest.approx(530 + 7200 / 1.5)
    assert with_actions.devices == math.ceil(164 * 1.2 * (530 + 4800) / 60_480)


def test_workload_totals_and_per_hour() -> None:
    """세션 합·영상 시간 가중 스트림 수·행동 요청 합 (한 세션이라도 모르면 None)."""
    a = SessionLoad("a", 1.0, 1, {T: GroupLoad(units=10)}, 5)
    b = SessionLoad("b", 3.0, 2, {T: GroupLoad(units=30), S: GroupLoad(units=6)}, 7)
    w = Workload((a, b))
    assert w.video_hours == 4.0 and w.total(T).units == 40 and w.total(S).units == 6
    assert w.per_hour(w.total(T).units) == 10.0
    assert w.streams_per_hour == pytest.approx((1 * 1 + 3 * 2) / 4)
    assert w.actions_requests == 12
    assert Workload((a, SessionLoad("c", 1.0, 1, {}))).actions_requests is None
    assert Workload(()).per_hour(5) == 0.0
