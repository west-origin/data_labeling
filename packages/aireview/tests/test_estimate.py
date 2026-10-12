"""작업량·장비 대수 추정(`dlp_aireview.estimate`) 테스트 (WP17.3, ADR 0040 5.8·8.3).

정답 근거: 같은 창 논리(`plan_windows`)의 창 수·대상 수, 손 계산한 이미지 토큰(1920x1080을 긴 변
448·정렬 32로 줄이면 448x224 → 14·7 = 98, 확대 조각 224 → 7·7 = 49), 정책 출력 가정(대상당
2 + 0.15·60 = 11토큰), 시간 모형 T = 무릎 * t(벤치)/t(작업량)의 손 계산, 설계 예시(영상 1시간당
530단위, 장비 1대 1단위/s, H 164·여유 0.2·가동률 0.7 → 2대).
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from typing import Any

import pytest

from dlp_aireview.bench import (
    ApproxLoad,
    BenchConfig,
    BenchPoint,
    BenchReport,
    DecodeMeasure,
    DecodeRun,
    PrefillRun,
)
from dlp_aireview.estimate import (
    GroupLoad,
    SessionLoad,
    Workload,
    capacity,
    counting_policy,
    group_throughput,
    report_mismatch,
    session_load,
    to_json,
)
from dlp_aireview.overlay import crop_plan
from dlp_aireview.plan import plan_windows
from dlp_aireview.policy import AiReviewPolicy
from dlp_aireview.probe import ServerInfo, approx_text_tokens, image_tokens
from dlp_aireview.prompt import ontology_text, system_text
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


def test_spatial_crops_are_counted_like_rendering(
    policy: AiReviewPolicy, ontology: Ontology
) -> None:
    """공간 단위의 확대 조각을 실행과 같은 규칙(`crop_plan`)으로 세어 이미지 토큰에 더한다.

    합성 장면의 박스(기준 30 px)는 가정 프레임 1920x1080을 긴 변 672로 줄이면 짧은 변 < 48 px이라
    작은 박스다. 조각 하나 = 224 px → 정렬 32로 7·7 = 49토큰.
    """
    scene = aireview_scene(0, n_units=6)
    load = session_load(scene.session, list(scene.labels), policy, ontology).groups[S]
    windows = [
        w for w in plan_windows(scene.session, list(scene.labels), policy, ontology) if w.group is S
    ]
    crops = [
        crop_plan(w.targets, sorted({round(t) for t in w.wanted_ms}), (1920, 1080),
                  w.max_side_px, policy)
        for w in windows
    ]  # fmt: skip
    assert load.crops == sum(map(len, crops)) > 0
    assert all(len(c) <= policy.spatial.crops.max for c in crops)
    per_frame = image_tokens(1920, 1080, policy.spatial.max_side_px, 32)
    assert load.image_tokens == load.frames * per_frame + load.crops * 7 * 7
    assert load.static_tokens == load.units * approx_text_tokens(
        system_text(policy) + ontology_text(ontology, S, policy), policy
    )


# ---------------------------------------------------------------- 장비 대수

APPROX = ApproxLoad(image=1000.0, text=200.0, static=3000.0)


def _point(units_per_s: float = 1.0, *, cached: int | None = 0, completion: int = 30) -> BenchPoint:
    """무릎 지점 (단위 100개, 동시성 32). cached: 단위당 캐시 적중 토큰 (None이면 서버가 모름).

    서버 prompt 토큰은 근사와 같다고 둔다 (단위당 4200).
    """
    prompt = 100 * 4200
    return BenchPoint(
        concurrency=32, units=100, ok=100, unavailable=0, wall_s=100 / units_per_s,
        units_per_s=units_per_s, latency_p95_ms=1000, prompt_tokens=prompt,
        cached_tokens=None if cached is None else 100 * cached, completion_tokens=100 * completion,
        cache_hit_ratio=None if cached is None else cached / 4200,
    )  # fmt: skip


def _config(
    group: AiGroup,
    policy: AiReviewPolicy,
    point: BenchPoint,
    *,
    approx: ApproxLoad | None = APPROX,
    prefill_tok_s: float | None = 1200.0,
    decode_tok_s: float | None = 1000.0,
    honored: bool = True,
) -> BenchConfig:
    """정책 설정 값의 벤치 설정 (따로 잰 용량: 프리필은 새 토큰 1200/단위 * 10요청, 생성 10초)."""
    frames, side = {
        T: (policy.temporal.frames, policy.temporal.max_side_px),
        L: (policy.temporal.long_frames, policy.temporal.long_max_side_px),
        S: (policy.spatial.frames, policy.spatial.max_side_px),
    }[group]
    prefill = (
        PrefillRun(concurrency=32, requests=10, ok=10, unavailable=0,
                   wall_s=12_000 / prefill_tok_s, fresh_tokens=12_000.0, static_tokens=30_000.0)
        if prefill_tok_s
        else None
    )  # fmt: skip
    decode = (
        DecodeRun(concurrency=32, requests=64, forced_tokens=256, ok=64, wall_s=10.0,
                  completion_tokens=round(decode_tok_s * 10), decode_tok_s=decode_tok_s,
                  honored=honored)
        if decode_tok_s
        else None
    )  # fmt: skip
    return BenchConfig(
        group=group, frames=frames, side=side, planned_units=1, approx=approx, points=(point,),
        knee=point, prefill=prefill, decode=decode,
    )  # fmt: skip


def _report(
    configs: tuple[BenchConfig, ...], decode_fps: float | None = None, **policy_info: Any
) -> BenchReport:
    """벤치 보고서 (정책 정보는 모델 m만, 셈법 값은 policy_info로)."""
    decode = (
        DecodeMeasure(fps=decode_fps, frames=100, width=1920, height=1080, max_side=448)
        if decode_fps
        else None
    )
    return BenchReport(
        created_at=datetime(2026, 10, 12, tzinfo=UTC), device="spark-01", source="synthetic",
        server=ServerInfo(endpoint="spark-01", served_model="m"),
        policy={"model": "m", **policy_info}, grid={}, configs=configs, decode=decode,
    )  # fmt: skip


def _load(image: int = 1000, text: int = 200, static: int = 3000, output: float = 30) -> GroupLoad:
    """530단위 작업량 (단위당 이미지·글·고정 접두사·출력)."""
    n = 530
    return GroupLoad(
        units=n, targets=n * 9, frames=n * 8, image_tokens=n * image, text_tokens=n * text,
        static_tokens=n * static, output_tokens=n * output,
    )  # fmt: skip


def _workload(load: GroupLoad | None = None, actions: int | None = None) -> Workload:
    """1시간 세션 하나의 temporal 작업량."""
    return Workload((SessionLoad("s1", 1.0, 2, {T: load or _load()}, actions),))


def test_same_mix_as_bench_is_the_knee(policy: AiReviewPolicy) -> None:
    """작업량의 단위당 토큰이 벤치와 같으면 T = 무릎 단위/s (캐시 적중이 있든 없든)."""
    for cached in (None, 0, 3000):
        report = _report((_config(T, policy, _point(2.5, cached=cached)),))
        tp = group_throughput(report, policy, T, _load())
        assert tp is not None and tp.units_per_s == pytest.approx(2.5)
        assert tp.mix == pytest.approx(1.0)


def test_prefill_bound_knee_is_not_cut_by_unbound_decode(policy: AiReviewPolicy) -> None:
    """무릎이 프리필에 묶였을 때(단위당 생성 30토큰), 실제 출력 99토큰 작업량의 T는 묶이지 않은
    디코드의 관측 속도(105 tok/s ÷ 99 = 1.06)가 아니라 시간 모형으로 옮긴 값이다.

    캐시가 고정 접두사에 다 맞음(f = 1): C_p 4200, C_d 1000.
    t(벤치) = 1200/4200 + 30/1000 = 0.31571, t(작업량) = 1200/4200 + 99/1000 = 0.38471,
    T = 3.5 * 0.31571 / 0.38471 = 2.872 (프리필 몫이 크다).
    """
    point = _point(3.5, cached=3000, completion=30)
    report = _report((_config(T, policy, point, prefill_tok_s=4200.0, decode_tok_s=1000.0),))
    tp = group_throughput(report, policy, T, _load(output=99))
    assert tp is not None and tp.static_cached == pytest.approx(1.0)
    assert (tp.prefill_tok_s, tp.decode_tok_s) == (pytest.approx(4200), pytest.approx(1000))
    t_bench, t_work = 1200 / 4200 + 30 / 1000, 1200 / 4200 + 99 / 1000
    assert tp.units_per_s == pytest.approx(3.5 * t_bench / t_work)
    assert tp.limit == "prefill" and tp.units_per_s > 2.8
    decode_bound = group_throughput(report, policy, T, _load(image=100, text=20, output=900))
    assert decode_bound is not None and decode_bound.limit == "decode"
    assert decode_bound.units_per_s == pytest.approx(3.5 * t_bench / (120 / 4200 + 900 / 1000))


def test_uncached_static_prefix_counts_on_both_sides(policy: AiReviewPolicy) -> None:
    """서버가 cached_tokens를 주지 않거나 0이면 고정 접두사도 벤치·작업량 양쪽에서 매번 계산한다고
    본다 (한쪽에서만 빼면 T가 (고정 + 새)/새 배 부푼다).

    무릎 1 단위/s, 벤치·작업량 모두 새 1200 + 고정 3000이면 T = 1 (빼면 3.5가 된다).
    작업량 새 토큰이 600이면: f = 0 → C_p = (12000 + 30000)/10 = 4200, t(벤치) = 4200/4200 + 0.03,
    t(작업량) = 3600/4200 + 0.03. f = 1 → C_p = 1200, t(벤치) = 1 + 0.03, t(작업량) = 0.5 + 0.03.
    """
    for cached in (None, 0):
        report = _report((_config(T, policy, _point(1.0, cached=cached), prefill_tok_s=1200.0),))
        tp = group_throughput(report, policy, T, _load())
        assert tp is not None and tp.static_cached == 0.0
        assert tp.units_per_s == pytest.approx(1.0)
        assert tp.prefill_tok_s == pytest.approx(4200)
        small = group_throughput(report, policy, T, _load(image=500, text=100))
        assert small is not None
        assert small.units_per_s == pytest.approx(1.03 / (3600 / 4200 + 0.03))
    cached_report = _report((_config(T, policy, _point(1.0, cached=3000), prefill_tok_s=1200.0),))
    cached = group_throughput(cached_report, policy, T, _load(image=500, text=100))
    assert cached is not None and cached.units_per_s == pytest.approx(1.03 / 0.53)


def test_unknown_cache_ratio_takes_the_conservative_bound(policy: AiReviewPolicy) -> None:
    """서버가 cached_tokens를 보고하지 않으면 캐시 비율 0과 1로 각각 보정해 작은 T를 쓰고 경고한다.

    정답 근거(검수 지적 재현): 벤치 단위 새 토큰 1200, 작업량 2400, 고정 접두사 3000 중 2000이
    캐시에 맞는 서버. 보고하면 f = 2/3로 참값 T를 낸다. 예전에는 보고하지 않으면 f = 0으로 세어 T를
    약 19% 크게(장비 대수를 적게) 냈다. T는 f에 대해 단조라 f ∈ {0, 1} 중 작은 쪽은 참값 이하다.
    """
    heavy = _load(image=2000, text=400)
    known = _report((_config(T, policy, _point(1.0, cached=2000), prefill_tok_s=1200.0),))
    unknown = _report((_config(T, policy, _point(1.0, cached=None), prefill_tok_s=1200.0),))
    true = group_throughput(known, policy, T, heavy)
    guess = group_throughput(unknown, policy, T, heavy)
    assert true is not None and guess is not None
    assert true.cache_known and true.static_cached == pytest.approx(2 / 3)
    assert not guess.cache_known and guess.units_per_s <= true.units_per_s
    work = _workload(heavy)
    sure = capacity(
        work, known, policy, hours_per_day=8.0, rerun_margin=0.1, utilization=0.7, decode_workers=4
    )
    safe = capacity(
        work, unknown, policy, hours_per_day=8.0, rerun_margin=0.1, utilization=0.7,
        decode_workers=4,
    )  # fmt: skip
    assert sure.devices is not None and safe.devices is not None and safe.devices >= sure.devices
    assert any("cached_tokens" in w for w in safe.warnings)
    assert not any("cached_tokens" in w for w in sure.warnings)


def test_without_capacity_runs_the_knee_is_used_and_flagged(policy: AiReviewPolicy) -> None:
    """용량 측정이 없거나 생성 길이 고정이 지켜지지 않았으면 보정 없이 무릎 단위/s (경고)."""
    point = _point(2.0)
    for config in (
        _config(T, policy, point, prefill_tok_s=None),
        _config(T, policy, point, honored=False),
        _config(T, policy, point, approx=None),
    ):
        tp = group_throughput(_report((config,)), policy, T, _load(output=500))
        assert tp is not None and tp.limit == "knee" and tp.units_per_s == 2.0
        assert tp.mix is None
    kw = {"hours_per_day": 164.0, "rerun_margin": 0.2, "utilization": 0.7, "decode_workers": 4}
    cap = capacity(_workload(), _report((_config(T, policy, point, honored=False),)), policy, **kw)  # pyright: ignore[reportArgumentType]
    assert any("보정하지 않았다" in w for w in cap.warnings) and cap.devices is not None


def test_capacity_matches_design_example(policy: AiReviewPolicy) -> None:
    """영상 1시간당 530단위, 장비 1대 1단위/s(벤치와 같은 구성) → 대수 = ceil(1.72) = 2."""
    report = _report((_config(T, policy, _point(1.0)),), decode_fps=600, model=policy.backend.model)
    cap = capacity(
        _workload(), report, policy, hours_per_day=164, rerun_margin=0.2, utilization=0.7,
        decode_workers=4,
    )  # fmt: skip
    [share] = cap.groups
    assert (share.group, share.units_per_hour, share.limit) == ("temporal", 530, "prefill")
    assert share.units_per_s == pytest.approx(1.0)
    assert share.prefill_tok_s == pytest.approx(4200) and share.decode_tok_s == pytest.approx(1000)
    assert cap.device_s_per_hour == pytest.approx(530)
    assert cap.devices == math.ceil(164 * 1.2 * 530 / (86_400 * 0.7)) == 2
    assert cap.missing == () and cap.warnings == ()
    assert cap.decode_hours_per_day == pytest.approx(164 * 30 * 2 / (600 * 4))
    json.dumps(to_json(cap))
    json.dumps(to_json(_workload()))


def test_capacity_missing_groups_and_actions(policy: AiReviewPolicy) -> None:
    """측정이 없는 묶음·행동 VLM은 missing이라 대수를 내지 않는다. 행동 VLM 실측이 있으면 더한다."""
    load = _workload().total(T)
    assert group_throughput(_report((_config(S, policy, _point()),)), policy, T, load) is None
    report = _report((_config(T, policy, _point(1.0)),))
    kw = {"hours_per_day": 164.0, "rerun_margin": 0.2, "utilization": 0.7, "decode_workers": 4}
    missing = capacity(_workload(actions=7200), report, policy, **kw)  # pyright: ignore[reportArgumentType]
    assert missing.missing == ("actions",) and missing.devices is None
    assert missing.decode_hours_per_day is None
    with_actions = capacity(_workload(actions=7200), report, policy, actions_req_s=1.5, **kw)  # pyright: ignore[reportArgumentType]
    assert with_actions.device_s_per_hour == pytest.approx(530 + 7200 / 1.5)
    assert with_actions.devices == math.ceil(164 * 1.2 * (530 + 4800) / 60_480)


def test_report_policy_mismatch_and_counting_policy(policy: AiReviewPolicy) -> None:
    """보고서의 모델·fingerprint·셈법 값이 지금 정책과 다르면 경고하고, 셈법 값은 보고서 것으로
    다시 센다 (후보 모델마다 align_px가 다를 수 있다)."""
    same = _report((), align_px=policy.image.align_px, fingerprint=policy.backend.fingerprint)
    assert counting_policy(policy, same) is policy
    assert report_mismatch(same, policy) == [f"model: 보고서 m ≠ 지금 정책 {policy.backend.model}"]
    other = _report((), align_px=28, text_bytes_per_token=2.5, fingerprint="sglang-fp8")
    recount = counting_policy(policy, other)
    assert (recount.image.align_px, recount.estimate.text_bytes_per_token) == (28, 2.5)
    assert recount.temporal == policy.temporal and recount.backend == policy.backend
    assert [m.split(":")[0] for m in report_mismatch(other, policy)] == [
        "model", "fingerprint", "align_px", "text_bytes_per_token",
    ]  # fmt: skip
    assert [m.split(":")[0] for m in report_mismatch(other, recount)] == ["model", "fingerprint"]
    old = _report(())  # 이전 형식 보고서 (셈법 값 없음)
    assert counting_policy(policy, old) is policy


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
