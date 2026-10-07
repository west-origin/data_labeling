"""긴 녹화(20분)·큰 드리프트(약 80 ppm)에서의 동기화 (ADR 0028).

20분이면 80 ppm 드리프트가 끝에서 약 96 ms(3프레임)로 쌓인다. 오프셋만 맞추면 1프레임 기준을
넘는다. 픽스처를 빨리 만들려고 영상은 가변 프레임레이트로 쓴다: 앞뒤 DENSE_MS 구간만 30 fps,
그 사이는 1 fps. 슬레이트 검색 구간(search_window_ms)을 그 안으로 줄인 정책을 쓴다
(슬레이트는 녹화 앞뒤 3초 안에 뜬다). 한 번 만든 픽스처로 방법 목록만 바꿔 각 방법을 따로 시험한다
(슬레이트 없음 = qr_slate를 빼고, 두드림이 안 들림 = tap_event를 뺀 정책).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dlp_fixtures.sync import SyncScenario, generate_sync_scenario
from dlp_schema.session import Session, StreamKind
from dlp_schema.testing import FIXED_TIME
from dlp_sync.loader import load_session_media
from dlp_sync.pipeline import StreamMedia, SyncReport, synchronize
from dlp_sync.policy import MethodName, SyncPolicy

from .conftest import FPS, session_for

FRAME_MS = 1000 / FPS
DURATION_MS = 1_200_000.0  # 20분
SEED = 8  # 3인칭 +78 ppm, 장갑 +46 ppm
# VFR 영상에서 30 fps로 쓰는 앞뒤 구간 ms (그 사이는 1 fps)
DENSE_MS = 6_000.0
# 테스트 정책의 slate.search_window_ms (DENSE_MS 안이어야 한다)
SEARCH_MS = 5_000.0

Built = tuple[SyncScenario, Session, dict[str, StreamMedia]]


@pytest.fixture(scope="module")
def long_policy(policy: SyncPolicy) -> SyncPolicy:
    """슬레이트 검색 구간을 `SEARCH_MS`로 줄인 정책 (VFR 영상의 촘촘한 구간 안)."""
    slate = policy.slate.model_copy(update={"search_window_ms": SEARCH_MS})
    return policy.model_copy(update={"slate": slate})


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory, long_policy: SyncPolicy) -> Built:
    """20분·seed 8 시나리오를 VFR 영상으로 한 번 쓰고 세션·입력을 만든다 (모듈 범위 캐시)."""
    scenario = generate_sync_scenario(SEED, recorded_at=FIXED_TIME, duration_ms=DURATION_MS)
    d = tmp_path_factory.mktemp("sync_long")
    scenario.write(d, videos=False, wavs=False)
    for name in ("bodycam", "third_person"):
        scenario.write_video(d / f"{name}.mp4", name, fps=FPS, dense_edges_ms=DENSE_MS)
    session = session_for(scenario, d)
    return scenario, session, load_session_media(session, Path, long_policy)


def _with(policy: SyncPolicy, **methods: tuple[MethodName, ...]) -> SyncPolicy:
    """스트림 종류별 방법 목록만 바꾼 정책 사본 (예: `_with(p, third_person=("tap_event",))`)."""
    kinds = {StreamKind(k): v for k, v in methods.items()}
    return policy.model_copy(update={"methods": {**policy.methods, **kinds}})


def _errors(built: Built, synced: Session, stream_id: str) -> tuple[float, float]:
    """(녹화 전체 11개 지점에서의 최대 시각 오차 ms, 드리프트 오차 ppm)."""
    truth = built[0].clocks[stream_id]
    s = synced.stream(stream_id)
    worst = max(
        abs(s.to_master_ms(t) - float(truth.to_master(t)))
        for t in (DURATION_MS * i / 10 for i in range(11))
    )
    return worst, abs(s.clock_scale - truth.clock_scale) * 1e6


def _chosen(report: SyncReport, stream_id: str) -> str | None:
    """보고서에서 그 스트림이 채택한 방법 이름 (없으면 None)."""
    return next(r.chosen for r in report.streams if r.stream_id == stream_id)


def test_fixture_has_realistic_drift(built: Built) -> None:
    """픽스처 전제 확인: 3인칭 > 70 ppm, 장갑 > 40 ppm이고, 오프셋만 맞추면 녹화 끝 오차가
    1프레임을 넘는다.
    """
    clocks = built[0].clocks
    assert (clocks["third_person"].clock_scale - 1) * 1e6 > 70
    assert (clocks["glove_right"].clock_scale - 1) * 1e6 > 40
    # 오프셋만 맞추면 녹화 끝에서 1프레임을 넘는다
    assert (clocks["third_person"].clock_scale - 1) * DURATION_MS > FRAME_MS


def test_slates_with_audio_refinement(built: Built, long_policy: SyncPolicy) -> None:
    """기본 정책(슬레이트 + 오디오 정밀화)으로 3인칭이 1 ms·10 ppm 안, 장갑은 드리프트만큼 넓힌
    두드림 짝짓기로 두 쌍을 맞춰 1프레임 안(드리프트도 추정)인지 검증한다.
    """
    synced, report = synchronize(built[1], built[2], long_policy)
    assert _chosen(report, "third_person") == "qr_slate"
    worst, drift_err = _errors(built, synced, "third_person")
    assert worst < 1.0 and drift_err <= 10  # 완료 기준: 1프레임, 10 ppm
    # 장갑: 두드림 짝짓기 허용 오차가 드리프트만큼 넓어져 두 쌍을 모두 맞춘다
    assert _chosen(report, "glove_right") == "tap_event"
    assert _errors(built, synced, "glove_right")[0] <= FRAME_MS
    assert synced.stream("glove_right").clock_scale != 1.0


def test_two_slates_fit_drift_without_audio(built: Built, long_policy: SyncPolicy) -> None:
    """오디오로 다듬지 않아도 10분 넘게 떨어진 두 슬레이트로 드리프트를 맞춰 1프레임 안이다."""
    policy = long_policy.model_copy(
        update={"slate": long_policy.slate.model_copy(update={"refine_with_audio": False})}
    )
    synced, report = synchronize(built[1], built[2], policy)
    assert _chosen(report, "third_person") == "qr_slate"
    assert synced.stream("third_person").clock_scale != 1.0
    assert _errors(built, synced, "third_person")[0] <= FRAME_MS


def test_offset_only_slates_lose_confidence_and_fall_through(
    built: Built, long_policy: SyncPolicy
) -> None:
    """드리프트를 맞추지 않는 슬레이트 맞춤은 잔차·드리프트 상한이 커서 믿지 않는다.

    다음 방법(두드림)으로 넘어간다.
    """
    slate = long_policy.slate.model_copy(
        update={"refine_with_audio": False, "estimate_drift": False}
    )
    policy = long_policy.model_copy(update={"slate": slate})
    synced, report = synchronize(built[1], built[2], policy)
    attempts = next(r for r in report.streams if r.stream_id == "third_person").attempts
    assert attempts[0].method == "qr_slate"
    assert attempts[0].confidence < policy.min_confidence
    assert _chosen(report, "third_person") == "tap_event"
    worst, drift_err = _errors(built, synced, "third_person")
    assert worst <= FRAME_MS and drift_err <= 10


def test_taps_scale_tolerance_with_drift(built: Built, long_policy: SyncPolicy) -> None:
    """3인칭을 두드림만으로 맞춰도 1 ms·10 ppm 안인지 검증한다.

    허용 오차가 드리프트만큼 넓어져 녹화 끝 두드림까지 짝지어진다 (ADR 0028 결정 4).
    """
    policy = _with(long_policy, third_person=("tap_event",))
    synced, report = synchronize(built[1], built[2], policy)
    assert _chosen(report, "third_person") == "tap_event"
    worst, drift_err = _errors(built, synced, "third_person")
    assert worst < 1.0 and drift_err <= 10


def test_correlation_widens_search_with_drift(built: Built, long_policy: SyncPolicy) -> None:
    """오디오 상관(3인칭)과 장갑-IMU 운동 상관 모두 2프레임 안 (완료 기준)."""
    policy = _with(long_policy, third_person=("audio_xcorr",), glove_right=("motion_xcorr",))
    synced, report = synchronize(built[1], built[2], policy)
    assert _chosen(report, "third_person") == "audio_xcorr"
    worst, drift_err = _errors(built, synced, "third_person")
    assert worst <= 2 * FRAME_MS and drift_err <= 10
    assert worst < 1.0
    assert _chosen(report, "glove_right") == "motion_xcorr"
    assert _errors(built, synced, "glove_right")[0] <= 2 * FRAME_MS
