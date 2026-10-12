"""공간 트랙의 시각별 모양 (키프레임 선형 보간, 순수 함수, WP17.2, ADR 0040 2.2).

계획(`plan`)과 그리기(`overlay`)가 같은 규칙을 쓰도록 한곳에 둔다. 계획은 대상 트랙의 표식이
그려지는 시각으로 프레임을 고르고(대상마다 적어도 한 프레임에 표식), 그리기는 그 시각의 상자·점을
그린다.

- `box_at`: 시각 t의 박스 (앞뒤 키프레임 선형 보간). 첫 키프레임 전, 마지막 키프레임 뒤, outside
  구간이면 None (CVAT 트랙과 같은 뜻: outside 키프레임부터 다음 키프레임 전까지 화면 밖).
- `points_at`: 시각 t의 보이는 키포인트 (선형 보간, 앞 키프레임에서 visibility > 0인 점만). 구간
  밖이면 None.
- `Timeline`: 트랙의 키프레임 시각과 "그 키프레임부터 다음 키프레임 전까지 표식이 그려지는가"를 미리
  계산해 둔 것 (`drawn_at`, `drawn_times`). 계획이 후보 시각마다 묻는다 (이분 탐색).

좌표는 그 스트림 영상 픽셀, 시각은 그 스트림 PTS ms다 (공간 라벨 키프레임은 그 스트림 영상의 PTS
시각, ADR 0019). 키프레임은 시각 오름차순이다 (계약 검사).
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass

from dlp_schema.labels import BoxKeyframe, BoxTrackPayload, KeypointTrackPayload, LabelRecord

Box = tuple[float, float, float, float]


def _around(times: Sequence[int], t_ms: float) -> int | None:
    """t 이하의 마지막 키프레임 번호 (첫 키프레임 전이나 마지막 뒤면 None)."""
    if not times or t_ms < times[0] or t_ms > times[-1]:
        return None
    return bisect_right(times, t_ms) - 1


def box_at(keyframes: Sequence[BoxKeyframe], t_ms: float) -> Box | None:
    """시각 t의 상자 (앞뒤 키프레임 선형 보간). 첫 키프레임 전·마지막 뒤·outside 구간이면 None."""
    k = _around([f.t_ms for f in keyframes], t_ms)
    if k is None or keyframes[k].outside:
        return None
    prev = keyframes[k]
    nxt = keyframes[k + 1] if k + 1 < len(keyframes) else None
    if nxt is None or nxt.outside or nxt.t_ms == prev.t_ms:
        return prev.x, prev.y, prev.w, prev.h
    a = (t_ms - prev.t_ms) / (nxt.t_ms - prev.t_ms)
    return (
        prev.x + a * (nxt.x - prev.x), prev.y + a * (nxt.y - prev.y),
        prev.w + a * (nxt.w - prev.w), prev.h + a * (nxt.h - prev.h),
    )  # fmt: skip


def points_at(p: KeypointTrackPayload, t_ms: float) -> list[tuple[float, float]] | None:
    """시각 t의 보이는 키포인트 (앞뒤 키프레임 선형 보간). 구간 밖이면 None."""
    kfs = p.keyframes
    k = _around([f.t_ms for f in kfs], t_ms)
    if k is None:
        return None
    prev = kfs[k]
    nxt = kfs[k + 1] if k + 1 < len(kfs) else prev
    a = 0.0 if nxt.t_ms == prev.t_ms else (t_ms - prev.t_ms) / (nxt.t_ms - prev.t_ms)
    return [
        (q.x + a * (r.x - q.x), q.y + a * (r.y - q.y))
        for q, r in zip(prev.points, nxt.points, strict=True)
        if q.visibility > 0
    ]


@dataclass(frozen=True)
class Timeline:
    """트랙의 표식 시간표.

    Attributes:
        times: 키프레임 시각 (오름차순).
        shown: 키프레임마다 그 키프레임부터 다음 키프레임 전까지 표식이 그려지는가 (박스: outside가
            아님, 키포인트: 보이는 점이 있음). `box_at`·`points_at`과 같은 규칙이다.
    """

    times: tuple[int, ...]
    shown: tuple[bool, ...]

    @classmethod
    def of(cls, label: LabelRecord) -> Timeline:
        """라벨 → 시간표 (공간 트랙이 아니면 빈 시간표)."""
        p = label.payload
        if isinstance(p, BoxTrackPayload):
            return cls(
                tuple(k.t_ms for k in p.keyframes), tuple(not k.outside for k in p.keyframes)
            )
        if isinstance(p, KeypointTrackPayload):
            return cls(
                tuple(k.t_ms for k in p.keyframes),
                tuple(any(q.visibility > 0 for q in k.points) for k in p.keyframes),
            )
        return cls((), ())

    def drawn_at(self, t_ms: float) -> bool:
        """시각 t에 표식이 그려지는가."""
        k = _around(self.times, t_ms)
        return k is not None and self.shown[k]

    def drawn_times(self) -> list[int]:
        """표식이 그려지는 키프레임 시각 (오름차순, 중복 없음)."""
        return sorted({t for t in self.times if self.drawn_at(t)})


def drawn_at(label: LabelRecord, t_ms: float) -> bool:
    """시각 t에 이 공간 트랙의 표식이 그려지는가 (`Timeline.drawn_at`)."""
    return Timeline.of(label).drawn_at(t_ms)
