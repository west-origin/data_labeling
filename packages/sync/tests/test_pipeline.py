"""`synchronize`의 방법 선택·대체 순서·정확도 테스트 (WP4 완료 기준, ADR 0004).

`conftest.build`로 만든 30초(일부 120초) 합성 시나리오를 쓴다. 판정 기준은 시나리오의 정답 시계
(`scenario.clocks`)와의 최대 시각 오차(0, 중간, 끝 세 지점)와 드리프트 오차(ppm)다.
완료 기준: 슬레이트·두드림 ≤ 1프레임(33 ms), 상호상관 ≤ 2프레임, 드리프트 ≤ 10 ppm.
옵션(`with_slates`, `audible_taps`)으로 앞 방법이 실패하게 만들어 다음 방법으로 넘어가는지 본다.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

import dlp_sync.pipeline as pipeline_mod
from dlp_fixtures.sync import SyncScenario
from dlp_schema.session import Session, SyncMethod
from dlp_sync.anchors import Anchor
from dlp_sync.pipeline import StreamMedia, SyncReport, apply_manual_adjustment, synchronize
from dlp_sync.policy import SyncPolicy
from dlp_sync.slate import detect_slates
from dlp_sync.xcorr import XcorrResult
from dlp_sync.xcorr import audio_anchors as xcorr_audio_anchors

FRAME_MS = 1000 / 30  # conftest의 영상 fps와 같다
Built = tuple[SyncScenario, Session, dict[str, StreamMedia]]


def _errors(built: Built, synced: Session, stream_id: str) -> tuple[float, float]:
    """(동기화 구간 전체에서의 최대 시각 오차 ms, 드리프트 오차 ppm)."""
    scenario = built[0]
    truth = scenario.clocks[stream_id]
    s = synced.stream(stream_id)
    worst = max(
        abs(s.to_master_ms(t) - float(truth.to_master(t)))
        for t in (0.0, scenario.duration_ms / 2, scenario.duration_ms)
    )
    return worst, abs(s.clock_scale - truth.clock_scale) * 1e6


def _chosen(report: SyncReport, stream_id: str) -> str | None:
    """보고서에서 그 스트림이 채택한 방법 이름 (없으면 None)."""
    return next(r.chosen for r in report.streams if r.stream_id == stream_id)


def test_slates_are_found_where_they_were_shown(
    build: Callable[..., Built], policy: SyncPolicy
) -> None:
    """바디캠 영상에서 슬레이트가 정답 순서대로, 정답 시각 이후 1프레임(+1 ms) 안에서 처음 보이는지
    검증한다.
    """
    scenario, session, _ = build()
    sightings = detect_slates(Path(session.reference_stream.uri), policy.slate)
    assert [s.payload for s in sightings] == [e.payload for e in scenario.slates]
    for seen, shown in zip(sightings, scenario.slates, strict=True):
        assert 0 <= seen.stream_ms - shown.master_ms < FRAME_MS + 1


def test_default_session_prefers_slate_then_tap(
    build: Callable[..., Built], policy: SyncPolicy
) -> None:
    """기본 정책에서 3인칭은 qr_slate(오디오 정밀화 포함, 오차 1 ms 미만), 장갑은 tap_event(1프레임
    안)를 채택하고, 기준·shared_clock 스트림은 바뀌지 않는지 검증한다.
    """
    built = build()
    synced, report = synchronize(built[1], built[2], policy)
    assert _chosen(report, "third_person") == "qr_slate"
    assert synced.stream("third_person").sync_method is SyncMethod.QR_SLATE
    # 슬레이트 맞춤을 출발점으로 오디오 상관이 다듬는다 (1프레임 기준보다 훨씬 작다)
    attempt = next(r for r in report.streams if r.stream_id == "third_person").attempts[0]
    assert "오디오 정밀화" in attempt.reason
    assert _errors(built, synced, "third_person")[0] < 1.0
    assert _chosen(report, "glove_right") == "tap_event"
    assert _errors(built, synced, "glove_right")[0] <= FRAME_MS
    # 기준·같은 시계 스트림은 그대로
    assert synced.stream("bodycam") == built[1].stream("bodycam")
    assert synced.stream("imu") == built[1].stream("imu")


def test_short_recording_slates_alone_fit_offset_only(
    build: Callable[..., Built], policy: SyncPolicy
) -> None:
    """오디오로 다듬지 않으면 짧은 녹화는 오프셋만 맞춘다.

    슬레이트 간격이 slate.min_drift_span_ms보다 짧다.
    """
    built = build()
    no_refine = policy.model_copy(
        update={"slate": policy.slate.model_copy(update={"refine_with_audio": False})}
    )
    synced, report = synchronize(built[1], built[2], no_refine)
    assert _chosen(report, "third_person") == "qr_slate"
    assert synced.stream("third_person").clock_scale == 1.0
    assert _errors(built, synced, "third_person")[0] <= FRAME_MS  # 완료 기준: 1프레임 이하


def test_audio_refinement_fit_error_falls_back_to_slate_fit(
    build: Callable[..., Built], policy: SyncPolicy, monkeypatch: pytest.MonkeyPatch
) -> None:
    """오디오 정밀화가 `FitError`(드리프트 상한 초과)로 실패해도 슬레이트 맞춤을 쓴다.

    정밀화 단계(prior가 있는 호출)의 오디오 상관만 드리프트 5%(50000 ppm)인 앵커를 돌려주게 바꾼다.
    감사 회귀: 예전에는 그 `FitError`가 `_try`까지 올라가 슬레이트 시도 전체가 신뢰도 0이 되고
    tap_event로 넘어갔다.
    """
    real = xcorr_audio_anchors

    def fake(*args: object, **kwargs: object) -> XcorrResult | None:
        """거친 탐색(prior 없음)은 진짜 함수로, 정밀화(prior 있음)는 드리프트 5% 앵커로."""
        if kwargs.get("prior") is None:
            return real(*args, **kwargs)  # type: ignore[arg-type]
        # 대상 시계 20초가 기준 21초에 해당: |scale - 1| = 5% → fit_clock이 거부한다
        return XcorrResult([Anchor(0.0, 0.0, "a"), Anchor(21_000.0, 20_000.0, "b")], psr=50.0)

    monkeypatch.setattr(pipeline_mod, "audio_anchors", fake)
    built = build()
    synced, report = synchronize(built[1], built[2], policy)
    assert _chosen(report, "third_person") == "qr_slate"
    attempt = next(r for r in report.streams if r.stream_id == "third_person").attempts[0]
    assert "오디오 정밀화 실패" in attempt.reason and attempt.confidence > 0
    # 슬레이트만으로 맞춘 결과 (오프셋만, 1프레임 이내)
    assert synced.stream("third_person").clock_scale == 1.0
    assert _errors(built, synced, "third_person")[0] <= FRAME_MS


def test_without_slates_third_person_uses_taps(
    build: Callable[..., Built], policy: SyncPolicy
) -> None:
    """슬레이트가 없으면 3인칭이 두드림으로 넘어가고 오차가 1 ms 미만인지 검증한다 (seed 2: 두 쌍
    모두 들림).
    """
    built = build(seed=2, with_slates=False)  # 3인칭이 두 번의 두 번 두드림을 모두 듣는 seed
    synced, report = synchronize(built[1], built[2], policy)
    assert _chosen(report, "third_person") == "tap_event"
    worst, _ = _errors(built, synced, "third_person")
    assert worst <= FRAME_MS
    assert worst < 1.0  # 오디오 두드림은 실제로 1 ms 안쪽


def test_single_tap_pair_is_not_trusted_alone(
    build: Callable[..., Built], policy: SyncPolicy
) -> None:
    """seed 4에서는 3인칭이 첫 두드림 뒤에 녹화를 시작해 한 쌍만 맞출 수 있다.

    한 쌍만으로는 믿지 않고 오디오 상관으로 넘어간다.
    """
    built = build(with_slates=False)
    synced, report = synchronize(built[1], built[2], policy)
    attempts = next(r for r in report.streams if r.stream_id == "third_person").attempts
    tap = next(a for a in attempts if a.method == "tap_event")
    assert tap.confidence == pytest.approx(0.5, abs=0.01)
    assert _chosen(report, "third_person") == "audio_xcorr"
    assert _errors(built, synced, "third_person")[0] < 1.0


def test_without_slates_or_audible_taps_falls_back_to_correlation(
    build: Callable[..., Built], policy: SyncPolicy
) -> None:
    """슬레이트도 들리는 두드림도 없으면 3인칭은 audio_xcorr, 장갑은 motion_xcorr로 넘어가 2프레임
    안에 맞추고, 시도 순서가 정책 순서(qr_slate → tap_event → audio_xcorr)와 같은지 검증한다.
    """
    built = build(with_slates=False, audible_taps=False)
    synced, report = synchronize(built[1], built[2], policy)
    assert _chosen(report, "third_person") == "audio_xcorr"
    assert _errors(built, synced, "third_person")[0] <= 2 * FRAME_MS  # 완료 기준: 2프레임 이하
    assert _chosen(report, "glove_right") == "motion_xcorr"
    assert _errors(built, synced, "glove_right")[0] <= 2 * FRAME_MS
    tried = next(r for r in report.streams if r.stream_id == "third_person").attempts
    assert [a.method for a in tried] == ["qr_slate", "tap_event", "audio_xcorr"]


def test_glove_without_any_signal_is_left_unsynced(
    build: Callable[..., Built], policy: SyncPolicy
) -> None:
    """장갑을 맞출 신호가 없으면 unsynced(신뢰도 None)가 되는지 검증한다.

    두드림이 기준 오디오에 들리지 않고(`audible_taps=False`) 기준 IMU도 빼서 두 방법 모두 실패.
    """
    _, session, media = build(with_slates=False, audible_taps=False)
    no_imu = {k: v for k, v in media.items() if k != "imu"}
    synced, report = synchronize(session, no_imu, policy)
    assert _chosen(report, "glove_right") is None
    glove = synced.stream("glove_right")
    assert glove.sync_method is SyncMethod.UNSYNCED and glove.sync_confidence is None


def test_drift_is_recovered_within_10_ppm_on_long_recording(
    build: Callable[..., Built], policy: SyncPolicy
) -> None:
    """120초 녹화(seed 9)에서 오디오 상관이 드리프트를 10 ppm 안, 시각을 1 ms 안으로 맞추는지
    검증한다.
    """
    built = build(seed=9, duration_ms=120_000.0, with_slates=False, audible_taps=False)
    synced, report = synchronize(built[1], built[2], policy)
    assert _chosen(report, "third_person") == "audio_xcorr"
    worst, drift_err = _errors(built, synced, "third_person")
    assert drift_err <= 10  # 완료 기준
    assert worst < 1.0


def test_manual_adjustment_survives_resync(build: Callable[..., Built], policy: SyncPolicy) -> None:
    """사람 조정값(-12.5 ms)이 재동기화 뒤에도 남아 마스터 시각에 더해지고, 기준 스트림 조정은
    거부되는지.
    """
    _, session, media = build()
    synced, _ = synchronize(session, media, policy)
    adjusted = apply_manual_adjustment(synced, "third_person", -12.5)
    again, _ = synchronize(adjusted, media, policy)
    tp = again.stream("third_person")
    assert tp.manual_adjustment_ms == -12.5
    assert tp.to_master_ms(0) == pytest.approx(synced.stream("third_person").to_master_ms(0) - 12.5)
    with pytest.raises(ValueError, match="기준"):
        apply_manual_adjustment(synced, "bodycam", 1.0)


def test_resync_is_deterministic(build: Callable[..., Built], policy: SyncPolicy) -> None:
    """동기화 결과를 다시 동기화해도 세션과 보고서가 같은지(멱등·결정적) 검증한다."""
    _, session, media = build()
    first, r1 = synchronize(session, media, policy)
    second, r2 = synchronize(first, media, policy)
    assert first == second
    assert r1.to_dict() == r2.to_dict()


def test_tap_drift_is_recovered_within_10_ppm(
    build: Callable[..., Built], policy: SyncPolicy
) -> None:
    """긴 녹화의 시작·끝 두 번 두드림(오디오)으로 드리프트를 10 ppm 안으로 추정한다 (완료 기준).

    seed 2는 3인칭이 두 번 두드림 두 쌍을 모두 듣는다. 장갑(100 Hz)은 샘플 간격 10 ms가
    앵커 오차라 드리프트 정밀도가 이보다 낮다 (프레임 오차 기준만 본다).
    """
    built = build(seed=2, duration_ms=120_000.0, with_slates=False)
    synced, report = synchronize(built[1], built[2], policy)
    assert _chosen(report, "third_person") == "tap_event"
    worst, drift_err = _errors(built, synced, "third_person")
    assert synced.stream("third_person").clock_scale != 1.0  # 드리프트를 실제로 추정했다
    assert drift_err <= 10
    assert worst < 1.0
    assert _chosen(report, "glove_right") == "tap_event"
    assert _errors(built, synced, "glove_right")[0] <= FRAME_MS


def test_adjusting_unsynced_stream_makes_it_manual(
    build: Callable[..., Built], policy: SyncPolicy
) -> None:
    """자동으로 맞추지 못한 스트림에 사람이 넣은 오프셋은 manual이 되고, 다시 동기화해도 남는다."""
    _, session, media = build(with_slates=False, audible_taps=False)
    no_imu = {k: v for k, v in media.items() if k != "imu"}
    synced, _ = synchronize(session, no_imu, policy)
    assert synced.stream("glove_right").sync_method is SyncMethod.UNSYNCED
    manual = apply_manual_adjustment(synced, "glove_right", 250.0)
    glove = manual.stream("glove_right")
    assert glove.sync_method is SyncMethod.MANUAL
    assert (glove.offset_ms, glove.clock_scale, glove.manual_adjustment_ms) == (0.0, 1.0, 250.0)
    assert glove.to_master_ms(1_000) == pytest.approx(1_250.0)
    # 이제 신호가 있어도 사람이 맞춘 결과를 덮어쓰지 않는다
    again, report = synchronize(manual, media, policy)
    assert again.stream("glove_right") == glove
    assert "glove_right" not in {r.stream_id for r in report.streams}


def test_failed_resync_keeps_previous_fit(build: Callable[..., Built], policy: SyncPolicy) -> None:
    """신호 없이 다시 돌려 모든 방법이 실패해도 이전 자동 결과와 사람 조정값을 그대로 두는지
    검증한다.
    """
    built = build()
    synced, _ = synchronize(built[1], built[2], policy)
    assert synced.stream("glove_right").sync_method is SyncMethod.TAP_EVENT
    adjusted = apply_manual_adjustment(synced, "glove_right", 3.0)
    # 신호를 잃은 재실행: unsynced로 내리지 않고 이전 결과와 사람 조정값을 둔다
    again, report = synchronize(adjusted, {}, policy)
    assert _chosen(report, "glove_right") is None
    assert again.stream("glove_right") == adjusted.stream("glove_right")
    assert again.stream("third_person") == synced.stream("third_person")
