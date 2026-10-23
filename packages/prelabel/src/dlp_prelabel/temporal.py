"""시간 일관성: 박스 트랙 거르기(히스테리시스·최소 길이), 손 키포인트 평활(One Euro)·짧은 끊김
채우기.

ADR 0054. 2026-10-22 첫 실영상 파일럿에서 OWLv2 도구 트랙의 약 69%가 키프레임 하나짜리였고, 손
트랙은 분당 70여 번 끊기고 프레임마다 떨렸다. 이 모듈은 어댑터가 라벨을 만들기 직전에 적용하는 순수
함수다 (모델·DB 없음).

- `filter_tracks`: 객체·도구 박스 트랙. 시작 문턱(start_score) 이상인 탐지가 나온 뒤부터 트랙을 열고
  (그 앞 키프레임은 버림), 탐지 문턱(정책 min_score) 이상이면 잇는다(히스테리시스). 남은 키프레임
  수·길이가 정책보다 짧으면 버린다. 프라이버시 탐지에는 쓰지 않는다 (재현 우선, 지우지 않고 표시만:
  `dlp_privacy.hints`).
- `OneEuro`·`smooth_hand_frames`: 손 2D를 정규화 좌표(0~1)에서 One Euro 필터로 평활한다 (느릴 때
  강하게, 빠를 때 약하게 → 정지 떨림은 줄고 빠른 동작의 지연은 작다). 끊김(reset_gap_ms 초과) 뒤에는
  새로 시작한다. 손 중심 3D 점(world)은 손 모양이라 그대로 둔다.
- `fill_gaps`: 같은 손이 max_gap_ms 이하로 끊긴 사이의 영상 프레임 시각에 앞뒤 관측을 선형 보간한
  키프레임을 넣는다. 채운 키프레임은 `interpolated=True`, 관절 visibility 1(관측 아님)·신뢰도
  `confidence`, world 없음 (3D 단계는 채운 키프레임을 올리지 않는다).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray

from dlp_prelabel.common import BoxTrack
from dlp_prelabel.policy import GapFillPolicy, SmoothingPolicy, TrackFilterPolicy
from dlp_schema.labels import Keypoint, KeypointFrame


def filter_tracks(tracks: Sequence[BoxTrack], policy: TrackFilterPolicy) -> list[BoxTrack]:
    """시작 문턱·최소 키프레임 수·최소 길이로 박스 트랙을 거른다 (순서 유지).

    트랙마다: 점수가 start_score 이상인 첫 키프레임 앞은 버린다. 그런 키프레임이 없으면 트랙을
    버린다. 남은 키프레임 수가 min_keyframes 미만이거나 길이(끝 - 시작 ms)가 min_duration_ms
    미만이면
    버린다.
    """
    out: list[BoxTrack] = []
    for tr in tracks:
        times = sorted(tr.frames)
        start = next((t for t in times if tr.frames[t][1] >= policy.start_score), None)
        if start is None:
            continue
        kept = [t for t in times if t >= start]
        if len(kept) < policy.min_keyframes or kept[-1] - kept[0] < policy.min_duration_ms:
            continue
        out.append(BoxTrack(tr.key, {t: tr.frames[t] for t in kept}))
    return out


def _alpha(cutoff_hz: float, dt_s: float) -> float:
    """1차 저역 통과 필터 계수 (차단 주파수, 시간 간격)."""
    tau = 1.0 / (2 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt_s)


@dataclass
class OneEuro:
    """One Euro 필터 (Casiez 2012) — 값 배열을 한꺼번에 평활한다.

    차단 주파수 = min_cutoff + beta x |변화 속도(평활)|. 속도 단위는 값 단위/초.
    """

    min_cutoff_hz: float
    beta: float
    d_cutoff_hz: float
    _x: NDArray[np.float64] | None = None
    _dx: NDArray[np.float64] | None = None
    _t: float | None = None

    def __call__(self, t_ms: float, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """시각 t_ms의 값 x를 평활한 값. 첫 호출이나 시각이 앞서지 않으면 그대로 돌려준다."""
        if self._x is None or self._t is None or self._dx is None or t_ms <= self._t:
            self._x, self._dx, self._t = x.copy(), np.zeros_like(x), t_ms
            return x.copy()
        dt = (t_ms - self._t) / 1000.0
        dx = (x - self._x) / dt
        a_d = _alpha(self.d_cutoff_hz, dt)
        self._dx = a_d * dx + (1 - a_d) * self._dx
        cutoff = self.min_cutoff_hz + self.beta * np.abs(self._dx)
        tau = 1.0 / (2 * math.pi * cutoff)
        a = 1.0 / (1.0 + tau / dt)
        self._x = a * x + (1 - a) * self._x
        self._t = t_ms
        return self._x.copy()


def _xy(frame: KeypointFrame, width: int, height: int) -> NDArray[np.float64]:
    """키프레임 2D 점 → 정규화 좌표 (N, 2)."""
    return np.array([[p.x / width, p.y / height] for p in frame.points], dtype=np.float64)


def _with_xy(
    frame: KeypointFrame, xy: NDArray[np.float64], width: int, height: int
) -> KeypointFrame:
    """정규화 좌표를 픽셀로 되돌려 넣은 키프레임 사본 (보임 정도·신뢰도·world는 그대로)."""
    points = tuple(
        p.model_copy(update={"x": float(xy[i, 0] * width), "y": float(xy[i, 1] * height)})
        for i, p in enumerate(frame.points)
    )
    return frame.model_copy(update={"points": points})


def smooth_hand_frames(
    frames: Sequence[KeypointFrame], width: int, height: int, policy: SmoothingPolicy
) -> list[KeypointFrame]:
    """한 손의 키프레임(시각순)을 One Euro로 평활한다. 꺼져 있으면 그대로.

    reset_gap_ms보다 오래 끊긴 뒤의 첫 키프레임에서 필터를 새로 시작한다 (끊김 너머로 끌려가지
    않게).
    """
    if not policy.enabled:
        return list(frames)
    out: list[KeypointFrame] = []
    filt: OneEuro | None = None
    prev: int | None = None
    for f in frames:
        if filt is None or prev is None or f.t_ms - prev > policy.reset_gap_ms:
            filt = OneEuro(policy.min_cutoff_hz, policy.beta, policy.d_cutoff_hz)
        out.append(_with_xy(f, filt(f.t_ms, _xy(f, width, height)), width, height))
        prev = f.t_ms
    return out


def _interpolate(a: KeypointFrame, b: KeypointFrame, t: int, confidence: float) -> KeypointFrame:
    """a·b 사이 시각 t의 선형 보간 키프레임 (관측 아님 표시)."""
    w = (t - a.t_ms) / (b.t_ms - a.t_ms)
    points = tuple(
        Keypoint(
            x=pa.x + (pb.x - pa.x) * w,
            y=pa.y + (pb.y - pa.y) * w,
            visibility=1,
            confidence=confidence,
        )
        for pa, pb in zip(a.points, b.points, strict=True)
    )
    return KeypointFrame(t_ms=t, points=points, interpolated=True)


def fill_gaps(
    frames: Sequence[KeypointFrame], frame_times: Sequence[int], policy: GapFillPolicy
) -> list[KeypointFrame]:
    """짧은 끊김을 보간 키프레임으로 채운다 (시각순 결과).

    Args:
        frames: 한 손의 관측 키프레임 (시각순).
        frame_times: 처리한 영상 프레임 시각 전체 (오름차순, 스트림 PTS ms). 채울 시각은 이 중
            끊김 안에 있는 것만이다 (프레임이 없는 시각은 만들지 않는다).
        policy: max_gap_ms(이 이하 끊김만), confidence(채운 관절 신뢰도). max_gap_ms 0이면 채우지
            않는다.
    """
    if policy.max_gap_ms <= 0 or len(frames) < 2:
        return list(frames)
    times = np.asarray(frame_times, dtype=np.int64)
    out: list[KeypointFrame] = [frames[0]]
    for a, b in pairwise(frames):
        if b.t_ms - a.t_ms <= policy.max_gap_ms:
            lo, hi = np.searchsorted(times, [a.t_ms, b.t_ms], side="right")
            out.extend(
                _interpolate(a, b, int(t), policy.confidence) for t in times[lo:hi] if t < b.t_ms
            )
        out.append(b)
    return out
