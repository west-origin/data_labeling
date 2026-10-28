"""블러 검수 힌트: 오탐 의심 표시 (ADR 0055). 블러 트랙을 지우거나 바꾸지 않는다.

프라이버시는 재현이 우선이다. 이 모듈은 사람의 블러 검수 화면에 "오탐 의심" 구간을 덧붙여, 검수자가
누락 위험 구간(`review.ReviewSegment`의 기존 이유)을 먼저 보고 그 다음에 오탐을 빨리 걷어내게 한다.
힌트는 검수 우선 구간 목록 끝(우선순위 = review_priority 길이)에 붙는 `suspected_false_positive`
구간이다. 블러 라벨·렌더·승인 판정에는 쓰이지 않는다.

규칙 (트랙마다, 정책 `privacy.yaml hints`):
- `no_head` (얼굴만): 키프레임 중 박스(여유 margin) 안에 보이는 RTMPose 머리 관절(코·눈·귀)이
  없는 비율이 min_share 이상. 1인칭 영상의 싱크·손에 붙은 얼굴 오탐이 여기에 걸린다. RTMPose
  트랙이 하나도 없으면 판단하지 않는다.
- `on_hand` (얼굴만): 키프레임 중 박스 중심이 MediaPipe 손 관절 외접 박스(여유 margin) 안인 비율이
  min_share 이상. 손 외접 박스는 보이는(visibility > 0) 관절만으로 만든다 (F-PMS-47).
- `short_flicker` (모든 대상): 관측 길이가 short_ms 미만 (한두 프레임 깜박임, ADR 0054 2절).
  관측 길이는 화면 안 키프레임의 처음~끝 길이다. 모델 출처 트랙은 탐지 유지(첫 관측 앞·마지막
  관측 뒤 blur_hold_ms, `tracker.track_frames`)를 앞뒤로 뺀다 (F-PMS-46). 영상 처음·끝에 닿아
  유지가 잘린 모델 트랙은 실제보다 짧게 보여 힌트가 더 붙을 수 있다 (표시만이라 재현에는 무해).
- `ai_crop_no` (AI 조각 확인을 돌린 트랙, ADR 0056): 사내 VLM이 블러본 조각의 주변 맥락으로 그
  대상이 아니라고 답함 (`dlp_aireview.crops`).
- `tta_unstable` (TTA를 돌린 트랙): 좌우 뒤집기·배율 변환 화면에서 같은 자리 탐지가 다시 나온
  비율(`dlp_media.geometry.tta_support`)의 평균이 tta_min_support 미만.

저장: 원본 버킷 `sessions/<세션>/derived/privacy_review/<스트림>.hints.json` (원본 화소에서 나온
판단이라 원본 버킷). 탐지 단계의 우선 구간 파일과 따로 두어 다시 탐지해도 지워지지 않고, 다시
계산하면 통째로 바꾼다. 블러 검수 작업 생성이 두 파일을 합쳐 CVAT 이슈로 단다.
"""

from __future__ import annotations

import bisect
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from dlp_media.geometry import Box, Transform, tta_support, warp_image
from dlp_media.storage import ObjectStore, sha256_file
from dlp_privacy.policy import PrivacyHintsPolicy
from dlp_privacy.review import ReviewSegment
from dlp_schema.labels import (
    BlurTrackPayload,
    BoxKeyframe,
    KeypointTrackPayload,
    LabelRecord,
    Source,
)

COCO_HEAD = (0, 1, 2, 3, 4)
HINT_REASON = "suspected_false_positive"


@dataclass(frozen=True)
class Hint:
    """오탐 의심 하나 (트랙 단위)."""

    label_id: str
    stream_id: str
    target: str
    reason: str  # no_head | on_hand | short_flicker | tta_unstable | ai_crop_no
    # 규칙에 걸린 키프레임 비율 (short_flicker는 1, tta_unstable은 1 - 지지 비율, ai_crop_no는 AI
    # 조각 확인의 의심 정도)
    share: float
    t_start_ms: int
    t_end_ms: int


def _expand(x: float, y: float, w: float, h: float, m: float) -> tuple[float, float, float, float]:
    """각 변을 m 비율 넓힌 (x0, y0, x1, y1)."""
    return (x - w * m, y - h * m, x + w * (1 + m), y + h * (1 + m))


class _ByTime[T]:
    """시각 → 항목 목록 (허용 오차 안 찾기)."""

    def __init__(self, items: Iterable[tuple[int, T]]) -> None:
        """(시각, 항목)들."""
        self.data: dict[int, list[T]] = {}
        for t, x in items:
            self.data.setdefault(t, []).append(x)
        self.keys = sorted(self.data)

    def near(self, t: int, tol: int) -> list[T]:
        """|시각 - t| <= tol인 항목."""
        lo = bisect.bisect_left(self.keys, t - tol)
        hi = bisect.bisect_right(self.keys, t + tol)
        return [x for k in self.keys[lo:hi] for x in self.data[k]]


Point = tuple[float, float]


def _heads(labels: Iterable[LabelRecord]) -> _ByTime[Point] | None:
    """보이는 COCO 머리 관절 점 (시각별). COCO 트랙이 없으면 None (판단하지 않음)."""
    items: list[tuple[int, Point]] = []
    found = False
    for x in labels:
        p = x.payload
        if isinstance(p, KeypointTrackPayload) and p.skeleton == "coco17":
            found = True
            for k in p.keyframes:
                for i in COCO_HEAD:
                    if i < len(k.points) and k.points[i].visibility == 2:
                        items.append((k.t_ms, (k.points[i].x, k.points[i].y)))
    return _ByTime(items) if found else None


def _hands(labels: Iterable[LabelRecord]) -> _ByTime[Box]:
    """MediaPipe 손 관절 외접 박스 (시각별, 보간 키프레임 제외)."""
    items: list[tuple[int, Box]] = []
    for x in labels:
        p = x.payload
        if isinstance(p, KeypointTrackPayload) and p.skeleton == "hand21":
            for k in p.keyframes:
                if k.interpolated:
                    continue
                # visibility 0 관절의 좌표는 의미가 없다 (F-PMS-47)
                seen = [q for q in k.points if q.visibility > 0]
                if not seen:
                    continue
                xs, ys = [q.x for q in seen], [q.y for q in seen]
                items.append((k.t_ms, (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))))
    return _ByTime(items)


def _share(frames: Sequence[BoxKeyframe], test: Callable[[BoxKeyframe], bool]) -> float:
    """키프레임 중 test가 참인 비율."""
    return sum(test(k) for k in frames) / len(frames) if frames else 0.0


def face_rules(
    face: LabelRecord, heads: _ByTime[Point] | None, hands: _ByTime[Box], policy: PrivacyHintsPolicy
) -> list[tuple[str, float]]:
    """얼굴 트랙 하나의 (규칙, 비율) 중 min_share 이상인 것."""
    p = face.payload
    assert isinstance(p, BlurTrackPayload)
    frames = [k for k in p.keyframes if not k.outside]
    tol, m = policy.time_tolerance_ms, policy.margin
    out: list[tuple[str, float]] = []
    if heads is not None:

        def no_head(k: BoxKeyframe) -> bool:
            x0, y0, x1, y1 = _expand(k.x, k.y, k.w, k.h, m)
            return not any(x0 <= hx <= x1 and y0 <= hy <= y1 for hx, hy in heads.near(k.t_ms, tol))

        out.append(("no_head", _share(frames, no_head)))

    def on_hand(k: BoxKeyframe) -> bool:
        cx, cy = k.x + k.w / 2, k.y + k.h / 2
        for b in hands.near(k.t_ms, tol):
            x0, y0, x1, y1 = _expand(*b, m)
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                return True
        return False

    out.append(("on_hand", _share(frames, on_hand)))
    return [(r, round(s, 4)) for r, s in out if s >= policy.min_share]


def observed_ms(track: LabelRecord, hold_ms: float) -> float:
    """블러 트랙의 관측 길이 (ms): 화면 안 키프레임의 처음~끝. 모델 출처면 앞뒤 유지 hold_ms를 뺀다.

    화면 안 키프레임이 없으면 0. 사람이 그리거나 고친 트랙에는 유지가 없어 빼지 않는다.
    """
    p = track.payload
    assert isinstance(p, BlurTrackPayload)
    shown = [k.t_ms for k in p.keyframes if not k.outside]
    if not shown:
        return 0.0
    span = float(shown[-1] - shown[0])
    if track.provenance.source is Source.MODEL:
        span -= 2 * hold_ms
    return max(span, 0.0)


def privacy_hints(
    labels: Sequence[LabelRecord],
    stream_id: str,
    policy: PrivacyHintsPolicy,
    tta: Mapping[str, float] | None = None,
    ai: Mapping[str, float] | None = None,
) -> list[Hint]:
    """한 스트림 운영 라벨에서 블러 트랙의 오탐 의심 힌트.

    Args:
        labels: 그 세션의 운영 라벨 (블러·손·COCO 트랙). 다른 스트림 라벨은 무시한다.
        tta: 블러 라벨 ID → TTA 지지 비율 평균 (`geometry.tta_tracks`). 없으면 그 규칙을 보지
            않는다.
        ai: 블러 라벨 ID → AI 조각 확인의 의심 정도 (`dlp_aireview.crops.to_doubts`, ADR 0056).
            있으면 `ai_crop_no` 힌트.
    """
    own = [x for x in labels if x.stream_id == stream_id]
    heads, hands = _heads(own), _hands(own)
    out: list[Hint] = []
    for x in own:
        p = x.payload
        if not isinstance(p, BlurTrackPayload):
            continue
        found: list[tuple[str, float]] = []
        if p.target == "face":
            found += face_rules(x, heads, hands, policy)
        if observed_ms(x, policy.blur_hold_ms) < policy.short_ms:
            found.append(("short_flicker", 1.0))
        if tta is not None and x.label_id in tta and tta[x.label_id] < policy.tta_min_support:
            found.append(("tta_unstable", round(1 - tta[x.label_id], 4)))
        if ai is not None and x.label_id in ai:
            found.append(("ai_crop_no", ai[x.label_id]))
        out += [
            Hint(x.label_id, stream_id, p.target, r, s, x.t_start_ms, x.t_end_ms) for r, s in found
        ]
    return out


def hint_segments(hints: Sequence[Hint], priority: int) -> list[ReviewSegment]:
    """힌트 → 검수 우선 구간 (이유 suspected_false_positive, 상세 "<규칙> <비율> <라벨 ID>")."""
    return [
        ReviewSegment(
            stream_id=h.stream_id,
            target=h.target,
            reason=HINT_REASON,
            t_start_ms=h.t_start_ms,
            t_end_ms=h.t_end_ms,
            priority=priority,
            detail=f"{h.reason} {h.share:.2f} {h.label_id}",
        )
        for h in sorted(hints, key=lambda h: (h.t_start_ms, h.label_id, h.reason))
    ]


def hints_key(session_id: str, stream_id: str) -> str:
    """원본 버킷의 블러 오탐 의심 구간 파일."""
    return f"sessions/{session_id}/derived/privacy_review/{stream_id}.hints.json"


def write_hint_segments(
    raw: ObjectStore, session_id: str, stream_id: str, segments: Sequence[ReviewSegment], work: Path
) -> str:
    """힌트 구간을 통째로 덮어쓴다 (파생물). 키를 돌려준다.

    부작용: 원본 버킷 쓰기 (감사 저장소면 write 기록).
    """
    out = work / f"{stream_id}-hints.json"
    out.write_text(json.dumps([s.model_dump(mode="json") for s in segments]), encoding="utf-8")
    key = hints_key(session_id, stream_id)
    raw.put_file(key, out, sha256_file(out))
    return key


def read_hint_segments(
    raw: ObjectStore, session_id: str, stream_id: str, work: Path
) -> list[ReviewSegment]:
    """저장한 힌트 구간. 없으면 빈 목록. 부작용: 원본 버킷 읽기."""
    key = hints_key(session_id, stream_id)
    if raw.head(key) is None:
        return []
    dest = work / f"{stream_id}-hints.in.json"
    raw.get_file(key, dest)
    data: list[object] = json.loads(dest.read_text(encoding="utf-8"))
    return [ReviewSegment.model_validate(x) for x in data]


Detect = Callable[[NDArray[np.uint8]], list[Box]]


def tta_box_support(
    image: NDArray[np.uint8],
    boxes: list[Box],
    detect: Detect,
    transforms: Sequence[Transform],
    iou_match: float,
) -> list[float]:
    """한 프레임 박스들의 TTA 지지 비율 (변환 화면마다 탐지를 다시 돌린다,
    `geometry.tta_support`).

    현재 쓰는 곳 없음. 실제 TTA 경로는 `dlp_cli.hints_cmds`(`geometry.tta_tracks`)다.
    """
    h, w = image.shape[:2]
    views = [(t, detect(warp_image(image, t))) for t in transforms]
    return tta_support(boxes, views, w, h, iou_match)


def pick_tta_frames(track: LabelRecord, n: int) -> list[BoxKeyframe]:
    """트랙 안에서 고르게 n개 키프레임 (화면 안인 것만).

    현재 쓰는 곳 없음. 실제 TTA 경로는 `dlp_cli.hints_cmds`(`geometry.tta_tracks`)다.
    """
    p = track.payload
    assert isinstance(p, BlurTrackPayload)
    frames = [k for k in p.keyframes if not k.outside]
    if len(frames) <= n:
        return frames
    idx = np.linspace(0, len(frames) - 1, num=n, dtype=int)
    return [frames[i] for i in idx]
