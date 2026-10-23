"""시간 일관성(`dlp_prelabel.temporal`)과 손 3D 제약(`dlp_prelabel.constraints3d`) 시험 (ADR 0054).

정답을 아는 합성 신호에 깜박임·떨림·끊김·튐을 넣고, 결과를 그 정답과 비교한다:
- 박스: 진짜 물체 트랙(점수 0.6, 3초)과 깜박이는 오탐(점수 0.15 한 번, 시작 문턱 미만 트랙).
- 손: 정답 궤적(원 운동) + 프레임마다 가우스 떨림 → 평활 뒤 정답과의 오차가 준다.
- 끊김: 정답 직선 운동에서 3프레임을 빼면 채운 키프레임이 정답 위치에 온다.
- 3D: 정답 손목(0.45 m 거리, 천천히 움직임)에 한 표본 튐(1.2 m)과 손끝 길이 어긋남.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from dlp_prelabel.common import BoxTrack
from dlp_prelabel.constraints3d import bone_flags, constrain_hands, wrist_flags
from dlp_prelabel.policy import (
    Constraints3DPolicy,
    GapFillPolicy,
    SmoothingPolicy,
    TrackFilterPolicy,
    load_policy,
)
from dlp_prelabel.temporal import fill_gaps, filter_tracks, smooth_hand_frames
from dlp_schema.labels import Constraint3DFlag, Keypoint, KeypointFrame, Trajectory3DSample

ROOT = Path(__file__).resolve().parents[3]
W, H = 1920, 1080


def _track(key: str, times: list[int], score: float) -> BoxTrack:
    """같은 박스·같은 점수의 트랙."""
    return BoxTrack(key, {t: ((10.0, 10.0, 50.0, 50.0), score) for t in times})


def test_filter_tracks_hysteresis_and_min_length() -> None:
    """정답: 3초 진짜 트랙만 남고, 시작 문턱 아래로만 이어진 트랙과 한 키프레임 트랙은 빠진다.

    시작 문턱 앞 키프레임(점수 0.15)은 잘린다: 진짜 트랙 앞에 0.15 키프레임 둘을 붙이면 결과는
    시작 문턱을 넘은 시각부터다.
    """
    policy = TrackFilterPolicy(start_score=0.3, min_keyframes=2, min_duration_ms=400)
    real = _track("mop", list(range(1000, 4001, 500)), 0.6)
    real.frames[0] = ((0.0, 0.0, 5.0, 5.0), 0.15)
    real.frames[500] = ((0.0, 0.0, 5.0, 5.0), 0.15)
    weak = _track("bathtub", [0, 500, 1000], 0.15)
    blip = _track("toilet_brush", [2000], 0.8)
    kept = filter_tracks([real, weak, blip], policy)
    assert [t.key for t in kept] == ["mop"]
    assert sorted(kept[0].frames) == list(range(1000, 4001, 500))


def _hand_frame(t: int, x: float, y: float) -> KeypointFrame:
    """21관절이 모두 (x, y) 픽셀에 있는 손 키프레임."""
    return KeypointFrame(t_ms=t, points=tuple(Keypoint(x=x, y=y, visibility=2) for _ in range(21)))


def _jitter_ratio(period_ms: float, radius: float, noise_px: float = 4.0) -> float:
    """정답 원 운동 + 떨림을 정책 기본값으로 평활했을 때 (평활 오차 RMS / 입력 오차 RMS).

    30 fps 10초, 처음 0.5초(필터 적응)는 빼고 잰다. period_ms가 아주 크면 정지에 가깝다.
    """
    policy = load_policy(ROOT).hands.smoothing
    rng = np.random.default_rng(0)
    times = [round(i * 1000 / 30) for i in range(300)]
    truth = [
        (960 + radius * math.cos(2 * math.pi * t / period_ms),
         540 + radius * math.sin(2 * math.pi * t / period_ms))
        for t in times
    ]  # fmt: skip
    noisy = [
        _hand_frame(t, x + rng.normal(0, noise_px), y + rng.normal(0, noise_px))
        for t, (x, y) in zip(times, truth, strict=True)
    ]
    smooth = smooth_hand_frames(noisy, W, H, policy)
    assert [f.t_ms for f in smooth] == times

    def rms(frames: list[KeypointFrame]) -> float:
        err = [
            math.dist((f.points[0].x, f.points[0].y), truth[i])
            for i, f in enumerate(frames)
            if f.t_ms >= 500
        ]
        return float(np.sqrt(np.mean(np.square(err))))

    return rms(smooth) / rms(noisy)


def test_one_euro_against_truth() -> None:
    """정답 궤적 + 떨림(표준편차 4 px)을 정책 기본값으로 평활한 오차.

    정답: 거의 정지(주기 무한대)는 오차가 x0.85 미만으로 줄고, 느린 원 운동(반지름 100 px, 주기 2초,
    약 300 px/s)은 늘지 않으며, 매우 빠른 원 운동(150 px, 0.7초, 약 1300 px/s)의 지연 오차는
    x1.3 안이다 (정책 주석의 상충 관계).
    """
    assert _jitter_ratio(1e12, 100) < 0.85
    assert _jitter_ratio(2000, 100) < 1.0
    assert _jitter_ratio(700, 150) < 1.3


def test_smoothing_restarts_after_gap_and_can_be_disabled() -> None:
    """끊김(reset_gap_ms 초과) 뒤 첫 키프레임은 필터 없이 그대로다. 꺼져 있으면 입력 그대로."""
    policy = SmoothingPolicy(
        enabled=True, min_cutoff_hz=1.0, beta=0.0, d_cutoff_hz=1.0, reset_gap_ms=100
    )
    frames = [_hand_frame(0, 100, 100), _hand_frame(33, 110, 100), _hand_frame(1000, 500, 500)]
    out = smooth_hand_frames(frames, W, H, policy)
    assert out[1].points[0].x < 110  # 평활됨 (앞 값 쪽으로 끌림)
    assert (out[2].points[0].x, out[2].points[0].y) == pytest.approx((500, 500))
    off = policy.model_copy(update={"enabled": False})
    assert smooth_hand_frames(frames, W, H, off) == frames


def test_fill_gaps_interpolates_short_gaps_only() -> None:
    """정답: x = 10 px/프레임 직선 운동. 3프레임(100 ms) 끊김은 정답 위치로 채우고, 500 ms 끊김은
    둔다.

    채운 키프레임은 interpolated, 관절 visibility 1·신뢰도 0.3, world 없음.
    """
    policy = GapFillPolicy(max_gap_ms=150, confidence=0.3)
    all_times = [round(i * 1000 / 30) for i in range(40)]
    observed = [i for i in range(40) if i not in (5, 6, 7) and not 20 <= i < 35]
    frames = [_hand_frame(all_times[i], 100 + 10 * i, 200) for i in observed]
    out = fill_gaps(frames, all_times, policy)
    filled = [f for f in out if f.interpolated]
    assert [f.t_ms for f in filled] == [all_times[i] for i in (5, 6, 7)]
    for f, i in zip(filled, (5, 6, 7), strict=True):
        # 시각이 정수 ms로 반올림돼 있어 정답과 1 px 안
        assert f.points[0].x == pytest.approx(100 + 10 * i, abs=1.0)
        assert f.points[0].visibility == 1 and f.points[0].confidence == 0.3
        assert f.world is None
    assert [f.t_ms for f in out] == sorted(f.t_ms for f in out)
    assert fill_gaps(frames, all_times, policy.model_copy(update={"max_gap_ms": 0})) == frames


def _s(t: int, x: float, y: float, z: float, conf: float = 0.8) -> Trajectory3DSample:
    """3D 표본."""
    return Trajectory3DSample(t_ms=t, x=x, y=y, z=z, confidence=conf)


@pytest.fixture
def c3d() -> Constraints3DPolicy:
    """저장소 정책의 depth.constraints."""
    return load_policy(ROOT).depth.constraints


def test_wrist_spike_and_reach_are_flagged(c3d: Constraints3DPolicy) -> None:
    """정답 손목: z 0.45 m에서 x가 200 ms마다 1 cm 움직임. 1000 ms 표본만 z 1.2 m로 튐.

    → 그 표본은 reach(0.9 m 밖)·speed(들어오고 나가는 속도 약 3.75 m/s)·accel. 나머지는 표시 없음.
    """
    wrist = [_s(t, 0.01 * i, 0.1, 0.45) for i, t in enumerate(range(0, 2001, 200))]
    wrist[5] = _s(1000, 0.05, 0.1, 1.2)
    flags = wrist_flags(wrist, c3d)
    assert set(flags) == {1000}
    assert flags[1000] == {Constraint3DFlag.REACH, Constraint3DFlag.SPEED, Constraint3DFlag.ACCEL}


def test_bone_length_outlier_is_flagged(c3d: Constraints3DPolicy) -> None:
    """정답 중지 끝 거리 0.19 m (표본 20개). 한 시각만 0.30 m → 그 시각만 bone_length."""
    times = list(range(0, 4000, 200))
    wrist = [_s(t, 0.0, 0.1, 0.45) for t in times]
    tip = [_s(t, 0.0, 0.1 - 0.19, 0.45) for t in times]
    tip[7] = _s(times[7], 0.0, 0.1 - 0.30, 0.45)
    assert set(bone_flags(wrist, tip, c3d)) == {times[7]}
    # 표본이 min_bone_samples보다 적으면 중앙값을 내지 않는다
    assert bone_flags(wrist[:3], tip[:3], c3d) == {}


def test_constrain_hands_applies_actions_and_keeps_flags(c3d: Constraints3DPolicy) -> None:
    """정책 동작: speed는 drop, reach·accel·bone_length는 downweight(신뢰도 x downweight^개수).

    튄 표본(1000 ms)은 speed라 손목·손끝 모두 빠진다. 박스 중심(part None)은 그대로다.
    """
    times = list(range(0, 4001, 200))
    wrist = [_s(t, 0.002 * i, 0.1, 0.45) for i, t in enumerate(times)]
    wrist[5] = _s(1000, 0.01, 0.1, 1.2)
    tip = [_s(t, s.x, s.y - 0.19, s.z) for t, s in zip(times, wrist, strict=True)]
    tip[12] = _s(times[12], tip[12].x, tip[12].y - 0.11, tip[12].z)  # 0.30 m
    box = [_s(0, 0.0, 0.0, 3.0)]
    out = constrain_hands(
        {("right_hand", "wrist"): wrist, ("right_hand", "middle_tip"): tip, ("mop_01", None): box},
        c3d,
    )
    assert 1000 not in {s.t_ms for s in out[("right_hand", "wrist")]}
    assert 1000 not in {s.t_ms for s in out[("right_hand", "middle_tip")]}
    bone = next(s for s in out[("right_hand", "middle_tip")] if s.t_ms == times[12])
    assert bone.flags == (Constraint3DFlag.BONE_LENGTH,)
    assert bone.confidence == pytest.approx(0.8 * c3d.downweight)
    assert out[("mop_01", None)] == box
    off = c3d.model_copy(update={"enabled": False})
    assert constrain_hands({("right_hand", "wrist"): wrist}, off)[("right_hand", "wrist")] == wrist


def test_lift_skips_interpolated_keyframes() -> None:
    """3D 단계는 보간해 채운 키프레임을 올리지 않는다 (지어낸 2D로 3D를 만들지 않음, ADR 0054)."""
    from dlp_prelabel.lift3d import hand_frames_by_time
    from dlp_schema.testing import hand_track_payload, make_label

    payload = hand_track_payload("right_hand", (0, 33, 66))
    payload["keyframes"][1]["interpolated"] = True
    label = make_label(payload, stream_id="bodycam")
    assert sorted(hand_frames_by_time([label])) == [0, 66]
