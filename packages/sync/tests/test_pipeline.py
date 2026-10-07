from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from dlp_fixtures.sync import SyncScenario
from dlp_schema.session import Session, SyncMethod
from dlp_sync.pipeline import StreamMedia, SyncReport, apply_manual_adjustment, synchronize
from dlp_sync.policy import SyncPolicy
from dlp_sync.slate import detect_slates

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
    return next(r.chosen for r in report.streams if r.stream_id == stream_id)


def test_slates_are_found_where_they_were_shown(
    build: Callable[..., Built], policy: SyncPolicy
) -> None:
    scenario, session, _ = build()
    sightings = detect_slates(Path(session.reference_stream.uri), policy.slate)
    assert [s.payload for s in sightings] == [e.payload for e in scenario.slates]
    for seen, shown in zip(sightings, scenario.slates, strict=True):
        assert 0 <= seen.stream_ms - shown.master_ms < FRAME_MS + 1


def test_default_session_prefers_slate_then_tap(
    build: Callable[..., Built], policy: SyncPolicy
) -> None:
    built = build()
    synced, report = synchronize(built[1], built[2], policy)
    assert _chosen(report, "third_person") == "qr_slate"
    assert synced.stream("third_person").sync_method is SyncMethod.QR_SLATE
    assert _errors(built, synced, "third_person")[0] <= FRAME_MS  # 완료 기준: 1프레임 이하
    assert (
        synced.stream("third_person").clock_scale == 1.0
    )  # 슬레이트로는 드리프트를 추정하지 않는다
    assert _chosen(report, "glove_right") == "tap_event"
    assert _errors(built, synced, "glove_right")[0] <= FRAME_MS
    # 기준·같은 시계 스트림은 그대로
    assert synced.stream("bodycam") == built[1].stream("bodycam")
    assert synced.stream("imu") == built[1].stream("imu")


def test_without_slates_third_person_uses_taps(
    build: Callable[..., Built], policy: SyncPolicy
) -> None:
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
    _, session, media = build(with_slates=False, audible_taps=False)
    no_imu = {k: v for k, v in media.items() if k != "imu"}
    synced, report = synchronize(session, no_imu, policy)
    assert _chosen(report, "glove_right") is None
    glove = synced.stream("glove_right")
    assert glove.sync_method is SyncMethod.UNSYNCED and glove.sync_confidence is None


def test_drift_is_recovered_within_10_ppm_on_long_recording(
    build: Callable[..., Built], policy: SyncPolicy
) -> None:
    built = build(seed=9, duration_ms=120_000.0, with_slates=False, audible_taps=False)
    synced, report = synchronize(built[1], built[2], policy)
    assert _chosen(report, "third_person") == "audio_xcorr"
    worst, drift_err = _errors(built, synced, "third_person")
    assert drift_err <= 10  # 완료 기준
    assert worst < 1.0


def test_manual_adjustment_survives_resync(build: Callable[..., Built], policy: SyncPolicy) -> None:
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
    _, session, media = build()
    first, r1 = synchronize(session, media, policy)
    second, r2 = synchronize(first, media, policy)
    assert first == second
    assert r1.to_dict() == r2.to_dict()
