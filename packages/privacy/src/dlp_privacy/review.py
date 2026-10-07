"""검수 우선 구간. 블러 검수 화면은 누락 찾기에 맞춰 이 구간을 먼저 보여준다."""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import Field

from dlp_privacy.policy import ReviewReason
from dlp_privacy.tracker import TrackFrame
from dlp_schema.common import Contract, Ms


class ReviewSegment(Contract):
    stream_id: str
    target: str
    reason: ReviewReason
    t_start_ms: Ms
    t_end_ms: Ms
    priority: int = Field(ge=0, description="작을수록 먼저")
    detail: str = ""


def spans(times: Iterable[int], frame_times: list[int]) -> list[tuple[int, int]]:
    """프레임 시각 집합을 연속 구간으로 묶는다 (사이에 빠진 프레임이 없으면 같은 구간)."""
    index = {t: i for i, t in enumerate(frame_times)}
    out: list[tuple[int, int]] = []
    prev_i: int | None = None
    for t in sorted(set(times)):
        i = index[t]
        if prev_i is not None and i == prev_i + 1:
            out[-1] = (out[-1][0], t)
        else:
            out.append((t, t))
        prev_i = i
    return out


def track_segments(
    stream_id: str,
    target: str,
    frames: list[TrackFrame],
    frame_times: list[int],
    *,
    review_score: float,
    available_detectors: set[str],
    priority: dict[ReviewReason, int],
) -> list[ReviewSegment]:
    picks: dict[ReviewReason, list[int]] = {
        "track_gap": [f.t_ms for f in frames if f.kind in ("interpolated", "held")],
        "low_confidence": [
            f.t_ms for f in frames if f.score is not None and f.score < review_score
        ],
        "disagreement": [
            f.t_ms
            for f in frames
            if f.kind == "observed"
            and len(available_detectors) > 1
            and f.detectors < available_detectors
        ],
        "reflection": [f.t_ms for f in frames if f.box is not None]
        if target == "reflection"
        else [],
    }
    out: list[ReviewSegment] = []
    for reason, times in picks.items():
        for start, end in spans(times, frame_times):
            out.append(
                ReviewSegment(
                    stream_id=stream_id,
                    target=target,
                    reason=reason,
                    t_start_ms=start,
                    t_end_ms=end,
                    priority=priority[reason],
                )
            )
    return out


def split_gap_segments(
    stream_id: str,
    target: str,
    covered: Iterable[int],
    frame_times: list[int],
    *,
    max_gap_ms: float,
    priority: int,
) -> list[ReviewSegment]:
    """같은 대상의 트랙이 나뉜 틈(블러가 없는 프레임)을 검수 우선 구간(track_gap)으로 낸다.

    트래커는 max_gap_ms보다 오래 끊기면 트랙을 나누고, 그 사이 프레임은 어떤 트랙도 덮지 않아
    블러가 빠진다. covered: 이 대상 블러가 보이는 프레임 시각. 앞뒤로 블러가 있는 틈 중
    (다음 블러 시각 - 이전 블러 시각)이 max_gap_ms 이하인 것만 낸다 (멀리 떨어진 다른 등장은 빼고).
    """
    shown = set(covered)
    out: list[ReviewSegment] = []
    prev: int | None = None  # 직전에 블러가 보인 프레임 시각
    gap: list[int] = []
    for t in frame_times:
        if t not in shown:
            if prev is not None:
                gap.append(t)
            continue
        if gap and prev is not None and t - prev <= max_gap_ms:
            out.append(
                ReviewSegment(
                    stream_id=stream_id, target=target, reason="track_gap",
                    t_start_ms=gap[0], t_end_ms=gap[-1], priority=priority,
                    detail="split",
                )
            )  # fmt: skip
        prev, gap = t, []
    return out


def sort_segments(segments: list[ReviewSegment]) -> list[ReviewSegment]:
    return sorted(segments, key=lambda s: (s.priority, s.t_start_ms, s.target))
