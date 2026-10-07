from __future__ import annotations

from pathlib import Path

import av
import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from dlp_fixtures.io import read_parquet, read_wav
from dlp_fixtures.sync import AUDIO_RATE, SLATE_DURATION_MS, SyncScenario, generate_sync_scenario
from dlp_schema.testing import FIXED_TIME


@pytest.fixture(scope="module")
def scenario() -> SyncScenario:
    return generate_sync_scenario(11, recorded_at=FIXED_TIME)


def _peaks(signal: NDArray[np.float32], rate: float, n: int, min_gap_ms: float) -> list[float]:
    """|신호|가 큰 순서로 서로 min_gap_ms 이상 떨어진 n개 위치(ms)."""
    order = np.argsort(-np.abs(signal))
    found: list[float] = []
    for idx in order:
        t = float(idx) / rate * 1000
        if all(abs(t - f) > min_gap_ms for f in found):
            found.append(t)
        if len(found) == n:
            break
    return sorted(found)


def test_clock_mapping_roundtrip(scenario: SyncScenario) -> None:
    clock = scenario.clocks["third_person"]
    assert float(clock.to_master(clock.to_stream(12_345.0))) == pytest.approx(12_345.0)
    assert scenario.clocks["imu"].offset_ms == 0.0


@pytest.mark.parametrize("stream", ["bodycam", "third_person"])
def test_taps_are_audible_at_truth_times(scenario: SyncScenario, stream: str) -> None:
    peaks = _peaks(scenario.audio[stream], AUDIO_RATE, 4, min_gap_ms=100)
    master = scenario.clocks[stream].to_master(np.array(peaks))
    # 버스트 첫 최대값은 시작 후 약 0.1 ms
    assert np.allclose(master, scenario.tap_master_ms, atol=1.0)


def test_glove_spikes_follow_its_own_clock(scenario: SyncScenario) -> None:
    g = scenario.glove_right
    peaks = _peaks(g["pressure_0"].astype(np.float32), 100.0, 4, min_gap_ms=100)
    master = scenario.clocks["glove_right"].to_master(np.array(peaks))
    assert np.allclose(master, scenario.tap_master_ms, atol=10.0)  # 100 Hz 샘플 간격


def test_ambient_noise_allows_cross_correlation(scenario: SyncScenario) -> None:
    """두드림이 없는 구간만으로도 오프셋을 찾을 수 있어야 한다."""
    body = scenario.audio["bodycam"]
    third = scenario.audio["third_person"]
    clock = scenario.clocks["third_person"]
    m0 = 10_000.0
    seg = body[int(m0 * AUDIO_RATE / 1000) : int((m0 + 2_000) * AUDIO_RATE / 1000)]
    expected = float(clock.to_stream(m0))
    lo = int((expected - 500) * AUDIO_RATE / 1000)
    window = third[lo : lo + seg.size + AUDIO_RATE]
    corr = np.correlate(window, seg, mode="valid")
    found = (lo + int(np.argmax(corr))) / AUDIO_RATE * 1000
    assert found == pytest.approx(expected, abs=1.0)


def test_imu_and_glove_files_roundtrip(scenario: SyncScenario, tmp_path: Path) -> None:
    scenario.write(tmp_path, videos=False)
    imu = read_parquet(tmp_path / "imu.parquet")
    assert np.allclose(imu["az"], scenario.imu["az"])
    samples, rate = read_wav(tmp_path / "bodycam.wav")
    assert rate == AUDIO_RATE
    assert np.allclose(samples, scenario.audio["bodycam"], atol=1e-4)


def test_videos_show_decodable_slates_at_truth_times(
    scenario: SyncScenario, tmp_path: Path
) -> None:
    detector = cv2.QRCodeDetector()
    for stream in ("bodycam", "third_person"):
        path = tmp_path / f"{stream}.mp4"
        scenario.write_video(path, stream)
        clock = scenario.clocks[stream]
        seen: dict[str, float] = {}
        with av.open(str(path)) as container:
            assert container.streams.audio, "오디오 트랙이 있어야 한다"
            for frame in container.decode(video=0):
                text, _, _ = detector.detectAndDecode(frame.to_ndarray(format="bgr24"))
                if text and text not in seen:
                    seen[text] = float(clock.to_master(float(frame.time or 0) * 1000))
        assert set(seen) == {s.payload for s in scenario.slates}
        for slate in scenario.slates:
            # 10 fps이므로 슬레이트 시작 후 첫 프레임은 100 ms 안에 온다
            assert 0 <= seen[slate.payload] - slate.master_ms < min(100.0, SLATE_DURATION_MS)
