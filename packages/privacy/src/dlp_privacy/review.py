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


def sort_segments(segments: list[ReviewSegment]) -> list[ReviewSegment]:
    return sorted(segments, key=lambda s: (s.priority, s.t_start_ms, s.target))
