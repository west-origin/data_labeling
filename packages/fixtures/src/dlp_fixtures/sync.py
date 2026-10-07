"""오프셋·드리프트를 아는 다중 스트림 동기화 시나리오 (WP2 → WP4 동기화 테스트).

기준 시각(마스터) m과 스트림 시각 t의 관계는 계약과 같다: m = offset_ms + t * clock_scale.
바디캠과 IMU는 같은 시계(offset 0, scale 1)다.
3인칭 영상과 장갑은 각자의 오프셋과 드리프트를 가진다.

신호 구성
- 오디오: 두 마이크가 공유하는 대역 제한 주변 소음 + 각자의 잡음
  + 두드림(감쇠 사인 버스트).
  주변 소음 덕분에 두드림이 없어도 상호상관으로 오프셋을 찾을 수 있다.
- 장갑 압력: 두드림 순간 압력 스파이크.
- IMU: 두드림 순간 가속도 스파이크.
- QR 슬레이트: 정해진 기준 시각에 시각 정보를 담은 QR을 화면에 띄운다 (영상은 video.py가 그린다).

정답 (`SyncScenario`)
- `clocks`: 스트림별 정답 오프셋·배율 (`ClockTruth`). `dlp_sync` 테스트는 동기화 결과를 이것과
  비교한다 (완료 기준: 슬레이트·두드림 1프레임, 상관 2프레임, 드리프트 10 ppm).
- `tap_master_ms`: 두 번 두드림 2회(녹화 앞·끝)의 마스터 시각 4개.
- `slates`: 슬레이트가 뜬 마스터 시각과 QR 내용 (녹화 앞·끝 2개, 각 `SLATE_DURATION_MS` 동안).
`write`가 `truth.json`으로도 쓴다.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from dlp_fixtures.io import write_json, write_parquet, write_wav
from dlp_fixtures.video import slate_frame, write_video

# 오디오 샘플레이트 Hz (dlp_sync.signals.AUDIO_RATE와 같다)
AUDIO_RATE = 16_000
# 장갑 압력 샘플레이트 Hz
GLOVE_RATE = 100.0
# 바디캠 내장 IMU 샘플레이트 Hz
IMU_RATE = 200.0
TAP_GAP_MS = 200.0  # 두 번 두드림 사이 간격
# 슬레이트 QR이 화면에 떠 있는 시간 ms
SLATE_DURATION_MS = 1_000.0


@dataclass(frozen=True)
class ClockTruth:
    """스트림 하나의 정답 시계: master = offset_ms + stream * clock_scale.

    드리프트(ppm) = (clock_scale - 1) * 1e6. 기본 시나리오는 ±80 ppm 안에서 고른다.
    """

    offset_ms: float
    clock_scale: float

    def to_master(self, stream_ms: NDArray[np.float64] | float) -> NDArray[np.float64]:
        """스트림 시각 ms(스칼라·배열) → 마스터 시각 ms (배열로 돌려준다)."""
        return np.asarray(self.offset_ms + np.asarray(stream_ms) * self.clock_scale)

    def to_stream(self, master_ms: NDArray[np.float64] | float) -> NDArray[np.float64]:
        """마스터 시각 ms(스칼라·배열) → 스트림 시각 ms (배열로 돌려준다)."""
        return np.asarray((np.asarray(master_ms) - self.offset_ms) / self.clock_scale)


@dataclass(frozen=True)
class SlateEvent:
    """슬레이트 하나: 화면에 뜬 마스터 시각 ms와 QR 내용(`slate_payload`)."""

    master_ms: float
    payload: str


@dataclass
class SyncScenario:
    """동기화 시나리오와 정답.

    스트림 이름: `bodycam`(기준), `imu`(바디캠 시계), `third_person`, `glove_right`.
    신호 dict(`audio`, `glove_right`, `imu`)의 시각은 모두 그 스트림 자기 시계 기준이다.

    Attributes:
        session_id: 세션 ID (슬레이트 QR에 들어간다).
        recorded_at: 녹화 시작 시각 (마스터 0 ms의 절대 시각).
        duration_ms: 마스터 구간 길이 ms.
        clocks: 스트림 이름 → 정답 시계.
        tap_master_ms: 두드림 4개의 마스터 시각 ms (앞 두 번, 끝 두 번).
        slates: 슬레이트 이벤트 (with_slates=False면 빈 목록).
        audio: `bodycam`, `third_person` → 16 kHz float32 샘플 (각 스트림 시각 0부터).
        glove_right: 열 dict (`t_ms`, `pressure_0`~`pressure_4`). 100 Hz.
        imu: 열 dict (`t_ms`, `ax`~`gz`). 200 Hz, 바디캠 시계. `az`에 중력 9.81과 두드림 스파이크.
    """

    session_id: str
    recorded_at: datetime
    duration_ms: float
    clocks: dict[str, ClockTruth]
    tap_master_ms: list[float]
    slates: list[SlateEvent]
    audio: dict[str, NDArray[np.float32]] = field(repr=False)
    glove_right: dict[str, NDArray[np.float64]] = field(repr=False)
    imu: dict[str, NDArray[np.float64]] = field(repr=False)

    def write(self, out_dir: Path, *, videos: bool = True, wavs: bool = True) -> None:
        """WAV·Parquet·정답 JSON, 그리고 (videos=True면) QR 슬레이트와 오디오가 든 MP4.

        wavs=False면 WAV를 쓰지 않는다 (영상에 오디오가 들어 있으므로 긴 녹화 픽스처에서
        시간을 아낀다).
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, samples in self.audio.items():
            if wavs:
                write_wav(out_dir / f"{name}.wav", samples, AUDIO_RATE)
            if videos:
                self.write_video(out_dir / f"{name}.mp4", name)
        write_parquet(out_dir / "glove_right.parquet", self.glove_right, {"clock": "glove_right"})
        write_parquet(out_dir / "imu.parquet", self.imu, {"clock": "bodycam"})
        write_json(out_dir / "truth.json", self.truth())

    def write_video(
        self,
        path: Path,
        stream: str,
        *,
        width: int = 240,
        height: int = 240,
        fps: float = 10.0,
        dense_edges_ms: float | None = None,
        sparse_every: int = 30,
    ) -> None:
        """스트림 시계 기준으로 프레임을 찍는다. 슬레이트가 떠 있는 동안은 QR을 보여준다.

        dense_edges_ms를 주면 가변 프레임레이트(VFR)로 쓴다: 스트림 앞뒤 dense_edges_ms 구간은
        fps로, 그 사이는 sparse_every 프레임마다 하나만 쓴다. 긴 녹화(20~40분) 픽스처를 빨리
        만들기 위한 옵션이다. 슬레이트 검출 구간(sync.yaml slate.search_window_ms)을 덮도록 정한다.

        프레임 i의 PTS = round(i * 1000 / fps) ms (스트림 시계). 그 PTS를 마스터로 바꿔
        [슬레이트 시각, + SLATE_DURATION_MS) 안이면 QR 화면, 아니면 잡음 섞인 회색 화면이다.
        영상에는 그 스트림의 오디오가 AAC로 함께 들어간다.
        """
        clock = self.clocks[stream]
        audio = self.audio[stream]
        stream_len_ms = audio.size / AUDIO_RATE * 1000
        # 배경 잡음은 스트림 이름 길이로 seed를 정해 결정적으로 만든다
        rng = np.random.default_rng(len(stream))
        blank = rng.integers(90, 110, size=(height, width, 3), dtype=np.uint8)
        cache: dict[str, NDArray[np.uint8]] = {}

        def frames() -> Iterator[tuple[int, NDArray[np.uint8]]]:
            """(PTS ms, 프레임) 생성기. QR 프레임은 payload별로 한 번만 그린다."""
            for i in range(int(stream_len_ms / 1000 * fps)):
                pts = round(i * 1000 / fps)
                # VFR: 앞뒤 dense 구간 밖에서는 sparse_every 프레임마다 하나만 남긴다
                if (
                    dense_edges_ms is not None
                    and dense_edges_ms < pts < stream_len_ms - dense_edges_ms
                    and i % sparse_every
                ):
                    continue
                master = float(clock.to_master(pts))
                shown = [s for s in self.slates if 0 <= master - s.master_ms < SLATE_DURATION_MS]
                if shown:
                    payload = shown[0].payload
                    if payload not in cache:
                        cache[payload] = slate_frame(payload, width, height)
                    yield pts, cache[payload]
                else:
                    yield pts, blank

        write_video(path, frames(), width=width, height=height, audio=audio, audio_rate=AUDIO_RATE)

    def truth(self) -> dict[str, object]:
        """정답을 JSON으로 쓸 수 있는 dict로 (`truth.json` 내용)."""
        return {
            "session_id": self.session_id,
            "recorded_at": self.recorded_at.isoformat(),
            "duration_ms": self.duration_ms,
            "clocks": {
                k: {"offset_ms": c.offset_ms, "clock_scale": c.clock_scale}
                for k, c in self.clocks.items()
            },
            "tap_master_ms": self.tap_master_ms,
            "slates": [{"master_ms": s.master_ms, "payload": s.payload} for s in self.slates],
        }


def slate_payload(session_id: str, recorded_at: datetime, master_ms: float) -> str:
    """슬레이트 QR 내용: 세션 ID와 그 순간의 절대 시각(Unix ms)."""
    epoch_ms = round(recorded_at.timestamp() * 1000 + master_ms)
    return f"DLP-SLATE|{session_id}|{epoch_ms}"


def generate_sync_scenario(
    seed: int = 0,
    *,
    session_id: str = "syn-sync",
    recorded_at: datetime,
    duration_ms: float = 30_000.0,
    max_offset_ms: float = 4_000.0,
    max_drift_ppm: float = 80.0,
    with_slates: bool = True,
    audible_taps: bool = True,
) -> SyncScenario:
    """동기화 시나리오를 만든다.

    with_slates=False면 슬레이트를 띄우지 않는다. audible_taps=False면 두드림이 마이크에 들리지
    않는다 (장갑·IMU에는 남는다). 둘 다 동기화 방법의 대체 경로를 시험하기 위한 옵션이다.
    난수 사용 순서는 바꾸지 않으므로 같은 seed의 나머지 값은 그대로다.

    Args:
        seed: 난수 seed. 오프셋·드리프트·두드림·슬레이트 시각·잡음이 모두 여기서 정해진다.
        session_id: 세션 ID.
        recorded_at: 녹화 시작 시각 (시간대 필수).
        duration_ms: 마스터 구간 길이 ms.
        max_offset_ms: 3인칭 오프셋을 ±이 값에서, 장갑 오프셋을 ±이 값/2에서 고른다.
        max_drift_ppm: 드리프트를 ±이 값(ppm)에서 고른다.
        with_slates: 슬레이트를 띄울지.
        audible_taps: 두드림이 오디오에 들릴지.

    Returns:
        `SyncScenario`. 두드림은 녹화 2.5~3.5 s와 끝 3~4 s 전에 두 번씩(간격 `TAP_GAP_MS`),
        슬레이트는 0.5~1 s와 끝 1.8 s 전에 뜬다.
    """
    rng = np.random.default_rng(seed)

    def random_clock(max_offset: float) -> ClockTruth:
        """오프셋 ±max_offset, 드리프트 ±max_drift_ppm인 정답 시계."""
        return ClockTruth(
            offset_ms=float(rng.uniform(-max_offset, max_offset)),
            clock_scale=1.0 + float(rng.uniform(-max_drift_ppm, max_drift_ppm)) * 1e-6,
        )

    clocks = {
        "bodycam": ClockTruth(0.0, 1.0),
        "imu": ClockTruth(0.0, 1.0),
        "third_person": random_clock(max_offset_ms),
        "glove_right": random_clock(max_offset_ms / 2),
    }
    # 두 번 두드림 2회: 녹화 앞과 끝
    start_tap = float(rng.uniform(2_500, 3_500))
    end_tap = duration_ms - float(rng.uniform(3_000, 4_000))
    taps = [start_tap, start_tap + TAP_GAP_MS, end_tap, end_tap + TAP_GAP_MS]
    slates = [
        SlateEvent(t, slate_payload(session_id, recorded_at, t))
        for t in (float(rng.uniform(500, 1_000)), duration_ms - 1_800.0)
    ]
    if not with_slates:
        slates = []

    # 주변 소음: 마스터 시간축 위의 대역 제한 잡음. 모든 스트림 구간을 덮도록 여유를 둔다.
    margin = max_offset_ms * 2 + 1_000
    amb_t = np.arange(-margin, duration_ms + margin, 1000 / AUDIO_RATE)
    white = rng.standard_normal(amb_t.size)
    # 8샘플 이동 평균 = 약 2 kHz 이하 저역 통과 (대역 제한 잡음)
    kernel = np.ones(8) / 8
    ambient = np.convolve(white, kernel, mode="same") * 0.15

    def mic(clock: ClockTruth, stream_len_ms: float, gain: float) -> NDArray[np.float32]:
        """한 스트림의 마이크 신호 (스트림 시각 0 ~ stream_len_ms, 16 kHz).

        공유 주변 소음(마스터 시간축)을 스트림 시각 → 마스터 시각으로 옮겨 읽고 `gain`을 곱한 뒤
        마이크별 잡음과 (들리면) 두드림을 더한다.
        """
        t = np.arange(0, stream_len_ms, 1000 / AUDIO_RATE)
        m = clock.to_master(t)
        sig = np.interp(m, amb_t, ambient) * gain
        sig += rng.standard_normal(t.size) * 0.01
        if audible_taps:
            sig += _taps(m, taps)
        return sig.astype(np.float32)

    # 각 스트림은 자기 시각 0에서 녹화를 시작해 마스터 구간 끝까지 녹화한다.
    def stream_len(clock: ClockTruth) -> float:
        """스트림이 마스터 끝 + 500 ms까지 녹화하도록 하는 스트림 길이 ms."""
        return float(clock.to_stream(duration_ms + 500))

    # 바디캠은 마스터 구간 길이, 3인칭은 자기 시계로 마스터 끝 + 500 ms까지
    audio = {
        "bodycam": mic(clocks["bodycam"], duration_ms, 1.0),
        "third_person": mic(clocks["third_person"], stream_len(clocks["third_person"]), 0.7),
    }

    # 장갑: 압력 채널 5개 = 두드림 펄스(채널마다 다른 세기) + 잡음, 0 이상
    glove_clock = clocks["glove_right"]
    gt = np.arange(0, stream_len(glove_clock), 1000 / GLOVE_RATE)
    gm = glove_clock.to_master(gt)
    spikes = _pulses(gm, taps, width_ms=40.0)
    glove = {"t_ms": gt}
    for ch in range(5):
        glove[f"pressure_{ch}"] = np.clip(
            spikes * rng.uniform(0.7, 1.0) + rng.standard_normal(gt.size) * 0.01, 0, None
        )

    # IMU: 바디캠 시계. 잡음 + az에 중력과 두드림 스파이크(3 m/s²)
    it = np.arange(0, duration_ms, 1000 / IMU_RATE)
    imu = {"t_ms": it}
    for axis in ("ax", "ay", "az", "gx", "gy", "gz"):
        imu[axis] = rng.standard_normal(it.size) * 0.05
    imu["az"] = imu["az"] + 9.81 + _pulses(it, taps, width_ms=20.0) * 3.0

    return SyncScenario(
        session_id=session_id,
        recorded_at=recorded_at,
        duration_ms=duration_ms,
        clocks=clocks,
        tap_master_ms=taps,
        slates=slates,
        audio=audio,
        glove_right=glove,
        imu=imu,
    )


def _taps(master_ms: NDArray[np.float64], taps: list[float]) -> NDArray[np.float64]:
    """감쇠 사인 버스트 (2 kHz, 시정수 8 ms, 30 ms 길이)."""
    out = np.zeros(master_ms.size)
    for tap in taps:
        dt = master_ms - tap
        on = (dt >= 0) & (dt < 30)
        out[on] += 0.8 * np.exp(-dt[on] / 8.0) * np.sin(2 * np.pi * 2.0 * dt[on])
    return out


def _pulses(
    master_ms: NDArray[np.float64], taps: list[float], width_ms: float
) -> NDArray[np.float64]:
    """두드림마다 폭 `width_ms`의 톱니 펄스 (시작에서 1, 끝으로 갈수록 0). 겹치면 큰 값.

    Args:
        master_ms: 샘플의 마스터 시각 ms.
        taps: 두드림 마스터 시각 ms.
        width_ms: 펄스 폭 ms.
    """
    out = np.zeros(master_ms.size)
    for tap in taps:
        dt = master_ms - tap
        on = (dt >= 0) & (dt < width_ms)
        out[on] = np.maximum(out[on], 1.0 - dt[on] / width_ms)
    return out
