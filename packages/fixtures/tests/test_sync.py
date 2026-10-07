"""동기화 시나리오 생성기(`generate_sync_scenario`) 자체 테스트 (WP2).

신호가 정답 시계(`clocks`)대로 만들어졌는지 생성기 출력만으로 확인한다: 두드림·장갑 스파이크·
슬레이트가 정답 마스터 시각에 있고, 주변 소음으로 상관이 가능하며, 파일 왕복이 무손실에 가깝다.
동기화 알고리즘(`dlp_sync`)은 쓰지 않는다.
"""

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
    """seed 11의 30초 동기화 시나리오 (모듈 범위 캐시)."""
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
    """정답 시계의 to_stream → to_master 왕복이 항등이고, IMU 오프셋이 0인지."""
    clock = scenario.clocks["third_person"]
    assert float(clock.to_master(clock.to_stream(12_345.0))) == pytest.approx(12_345.0)
    assert scenario.clocks["imu"].offset_ms == 0.0


@pytest.mark.parametrize("stream", ["bodycam", "third_person"])
def test_taps_are_audible_at_truth_times(scenario: SyncScenario, stream: str) -> None:
    """각 마이크 오디오의 큰 봉우리 4개가 정답 두드림 시각에 있는지 검증한다.

    봉우리 시각을 그 스트림의 정답 시계로 마스터로 바꿔 1 ms 안에서 맞아야 한다.
    """
    peaks = _peaks(scenario.audio[stream], AUDIO_RATE, 4, min_gap_ms=100)
    master = scenario.clocks[stream].to_master(np.array(peaks))
    # 버스트 첫 최대값은 시작 후 약 0.1 ms
    assert np.allclose(master, scenario.tap_master_ms, atol=1.0)


def test_glove_spikes_follow_its_own_clock(scenario: SyncScenario) -> None:
    """장갑 압력 스파이크를 장갑 시계로 마스터로 바꾸면 정답 두드림 시각과 맞는지 검증한다.

    허용 오차는 100 Hz 샘플 간격인 10 ms.
    """
    g = scenario.glove_right
    peaks = _peaks(g["pressure_0"].astype(np.float32), 100.0, 4, min_gap_ms=100)
    master = scenario.clocks["glove_right"].to_master(np.array(peaks))
    assert np.allclose(master, scenario.tap_master_ms, atol=10.0)  # 100 Hz 샘플 간격


def test_ambient_noise_allows_cross_correlation(scenario: SyncScenario) -> None:
    """두드림이 없는 구간만으로도 오프셋을 찾을 수 있어야 한다.

    바디캠 10~12 s 구간을 3인칭 오디오의 예상 위치 ±0.5 s에서 상관해, 찾은 위치가 정답 스트림
    시각과 1 ms 안인지 본다.
    """
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
    """`write`로 쓴 IMU Parquet과 WAV를 다시 읽으면 원래 값과 같은지 검증한다.

    WAV는 16비트 양자화 오차(1e-4) 안에서 같으면 된다.
    """
    scenario.write(tmp_path, videos=False)
    imu = read_parquet(tmp_path / "imu.parquet")
    assert np.allclose(imu["az"], scenario.imu["az"])
    samples, rate = read_wav(tmp_path / "bodycam.wav")
    assert rate == AUDIO_RATE
    assert np.allclose(samples, scenario.audio["bodycam"], atol=1e-4)


def test_videos_show_decodable_slates_at_truth_times(
    scenario: SyncScenario, tmp_path: Path
) -> None:
    """10 fps로 쓴 두 영상에서 슬레이트 QR이 모두 디코딩되는지 검증한다.

    처음 보인 프레임은 정답 시각 뒤 100 ms(1프레임) 안이어야 하고, 영상에 오디오 트랙이 있어야 한다.
    """
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
