"""검수 우선 구간. 블러 검수 화면은 누락 찾기에 맞춰 이 구간을 먼저 보여준다.

WP5, ADR 0005·0023. `pipeline.detect_video`가 트랙별 프레임(`tracker.TrackFrame`)에서
구간을 계산하고, `runner.detect_session`이 원본 버킷
`sessions/<세션>/derived/privacy_review/<스트림>.json`에 쓴다.
검수 작업 생성(dlp_review)이 이 파일을 읽어 CVAT 블러 검수 화면에 먼저 보여 줄 구간으로 쓴다.

- `ReviewSegment`: 구간 하나 (스트림, 대상, 이유, 시작·끝 ms, 우선순위, 상세).
- `spans`: 프레임 시각 집합 → 연속 구간.
- `track_segments`: 트랙 하나의 구간 (track_gap·low_confidence·disagreement·reflection).
- `split_gap_segments`: 같은 대상의 트랙이 나뉜 틈 (track_gap, detail="split").
- `sort_segments`: 우선순위 → 시작 시각 → 대상 순 정렬.

시간: 구간 시각은 그 스트림의 프레임 PTS 시각(정수 ms)이며, 구간은 양 끝 프레임을 포함한다.
"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import Field

from dlp_privacy.policy import ReviewReason
from dlp_privacy.tracker import TrackFrame
from dlp_schema.common import Contract, Ms


class ReviewSegment(Contract):
    """검수 우선 구간 하나. JSON으로 원본 버킷에 저장된다 (`runner.review_key`)."""

    # 영상 스트림 ID
    stream_id: str
    # 블러 대상 ID (face, document …)
    target: str
    # 구간 종류 (policy.ReviewReason 주석 참고)
    reason: ReviewReason
    # 구간 첫 프레임·마지막 프레임 시각 (스트림 PTS ms, 양 끝 포함)
    t_start_ms: Ms
    t_end_ms: Ms
    # review_priority 목록에서의 위치 (작을수록 먼저 보여 준다)
    priority: int = Field(ge=0, description="작을수록 먼저")
    # 부가 정보: no_detector는 쓸 수 없는 이유, trained_model은 모델 버전, 나뉜 트랙 틈은 "split".
    # runner.merge_segments가 trained_model 구간을 모델 버전별로 고를 때 이 값을 쓴다.
    detail: str = ""


def spans(times: Iterable[int], frame_times: list[int]) -> list[tuple[int, int]]:
    """프레임 시각 집합을 연속 구간으로 묶는다 (사이에 빠진 프레임이 없으면 같은 구간).

    "연속"은 시각 차이가 아니라 프레임 순서로 본다 (VFR에서도 이웃 프레임이면 이어진다).

    Args:
        times: 묶을 프레임 시각들 (중복 허용). 모두 frame_times에 있어야 한다.
        frame_times: 영상의 전체 프레임 시각 (증가 순).

    Returns:
        (첫 프레임 시각, 마지막 프레임 시각) 목록, 시각 순.

    Raises:
        KeyError: times에 frame_times 밖 시각이 있을 때.

    예: frame_times=[0, 33, 66, 100], times=[0, 33, 100] → [(0, 33), (100, 100)].
    """
    index = {t: i for i, t in enumerate(frame_times)}
    out: list[tuple[int, int]] = []
    prev_i: int | None = None
    for t in sorted(set(times)):
        i = index[t]
        if prev_i is not None and i == prev_i + 1:
            out[-1] = (out[-1][0], t)  # 바로 다음 프레임이면 직전 구간을 늘린다
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
    """트랙 하나에서 검수 우선 구간을 고른다.

    - track_gap: 보간(interpolated)·유지(held) 프레임. 탐지기가 직접 보지 못한 프레임이다.
    - low_confidence: 관측 점수가 review_score 미만인 프레임.
    - disagreement: 이 대상에 쓸 수 있는 탐지기가 둘 이상인데 관측에 참여한 탐지기가 그 일부뿐인
      프레임 (예: QR 탐지기만 찾고 OWLv2는 못 찾은 송장).
    - reflection: 대상이 reflection(반사면 속 얼굴)이면 블러가 보이는 모든 프레임.

    Args:
        stream_id, target: 구간에 넣을 스트림·대상.
        frames: `tracker.track_frames`의 결과.
        frame_times: 영상 전체 프레임 시각 (연속 판정용).
        review_score: privacy.yaml `review_score`.
        available_detectors: 이 대상에 쓸 수 있는 탐지기 이름.
        priority: 이유 → 우선순위 (review_priority 위치). 구간이 생긴 이유가 없으면 KeyError.

    Returns:
        이유별로 연속 프레임을 묶은 구간 목록 (정렬하지 않음).
    """
    picks: dict[ReviewReason, list[int]] = {
        "track_gap": [f.t_ms for f in frames if f.kind in ("interpolated", "held")],
        "low_confidence": [
            f.t_ms for f in frames if f.score is not None and f.score < review_score
        ],
        # f.detectors < available_detectors: 진부분집합 (일부 탐지기만 찾았다)
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

    Args:
        max_gap_ms: privacy.yaml `tracker.split_review_ms` (트래커의 max_gap_ms와 다른 값이다).
        priority: track_gap의 우선순위.

    Returns:
        detail="split"인 track_gap 구간 목록 (시각 순). 첫 블러 앞, 마지막 블러 뒤의 틈은 내지
        않는다.
    """
    shown = set(covered)
    out: list[ReviewSegment] = []
    prev: int | None = None  # 직전에 블러가 보인 프레임 시각
    gap: list[int] = []  # prev 뒤로 블러가 없는 프레임 시각들
    for t in frame_times:
        if t not in shown:
            if prev is not None:
                gap.append(t)
            continue
        # 블러가 다시 보인다: 틈의 양쪽 블러 간격이 짧으면 놓친 프레임일 가능성이 높다
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
    """(우선순위, 시작 시각, 대상) 순으로 정렬한 새 목록. 검수 화면이 이 순서로 보여 준다."""
    return sorted(segments, key=lambda s: (s.priority, s.t_start_ms, s.target))
