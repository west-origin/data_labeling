"""1인칭 영상에서 착용자 자신의 몸(팔·몸통)을 알아보는 순수 규칙 (ADR 0055·0058).

가슴 장착 바디캠에는 착용자의 아래팔·손·몸통이 화면 아래 가장자리에서 들어온다. COCO 사람 박스
(MediaPipe 객체·YOLOX)와 RTMPose 전신 자세는 이것을 "사람"으로 잡는다 (2026-10-22 파일럿: COCO
person 박스 대부분이 착용자 자신). 이 모듈은 착용자 증거(같은 스트림 hand21 손 트랙의 손목과 손
크기, 다른 사람 자세의 보이는 머리 관절)로 박스·자세 한 프레임이 착용자 몸인지 판정한다. 라벨을
바꾸지 않는다. 쓰는 곳:

- `dlp_prelabel`(ADR 0058): 객체·도구·전신 단계가 착용자 몸 탐지를 라벨로 만들지 않는다 (억제).
- `dlp_review.object_hints`(ADR 0055): 남은 사람 박스에 `wearer_body` 오탐 의심 힌트.

판정 (정책 값은 호출자가 넘긴다: `prelabel.yaml wearer_suppression`, `review.yaml object_hints`):
- 박스 프레임 (`wearer_box_frame`): 박스(각 변 margin 비율만큼 넓힘) 안에 착용자 손목이 있고, 다른
  사람의 보이는 머리 관절은 없고, (프레임 높이를 알면) 박스 아래 변이 화면 아래 bottom_band 띠
  안이다.
- 자세 프레임 (`wearer_pose_frame`): COCO 손목(9·10번) 하나라도 착용자 손목에서 손 크기 x
  wrist_match_palms 안이고, 보이는 관절을 감싸는 박스 아래 변이 화면 아래 bottom_band 띠 안이다.
  착용자 자세에는 머리 관절을 보지 않는다: 1인칭 화면에서 RTMPose가 착용자 팔에 머리를 지어내
  "보임"으로 내는 일이 많다 (파일럿 일치 자세 7,083개 중 5,203개).

증거의 시각은 스트림 PTS ms이고, 다른 트랙 키프레임은 허용 오차 안이면 같은 시각으로 본다. 허용
오차는 호출자가 정한다 (박스는 정책 time_tolerance_ms, 전신 어댑터 자세는 0: 같은 시각만).
보간한 손 키프레임(ADR 0054)은 관측이 아니라 증거로 쓰지 않는다.

공개 이름: `HAND_WRIST`, `HAND_MIDDLE_MCP`, `COCO_HEAD`, `COCO_WRISTS`, `Box`, `HandAt`,
`WearerEvidence`, `hand_at`, `inside`, `wearer_box_frame`, `wearer_pose_frame`.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from dlp_schema.labels import Hand, Keypoint, KeypointFrame, KeypointTrackPayload, LabelRecord

# hand21 손목·중지 뿌리 번호 (손 크기 = 둘 사이 픽셀 거리)
HAND_WRIST, HAND_MIDDLE_MCP = 0, 9
# COCO17 머리 다섯 점(코·눈·귀)과 왼손목·오른손목 번호
COCO_HEAD = (0, 1, 2, 3, 4)
COCO_WRISTS = (9, 10)

# (x, y, w, h) 픽셀, 왼쪽 위 원점
Box = tuple[float, float, float, float]
Point = tuple[float, float]


@dataclass(frozen=True)
class HandAt:
    """한 시각의 착용자 손 하나 (hand21 키프레임에서).

    wrist: 손목 픽셀. palm_px: 손목→중지 뿌리 픽셀 거리 (1 이상, 손 크기 단위). 중지 뿌리가
    보이지 않으면 1.0(손 크기 모름)이라 손 크기 배수 문턱은 사실상 통과하지 않는다. box: 보이는 손
    관절을 감싸는 박스. frame: 원래 키프레임 (팔 단계가 손 방향을 읽는다).
    """

    hand: Hand
    wrist: Point
    palm_px: float
    box: Box
    frame: KeypointFrame


def hand_at(hand: Hand, frame: KeypointFrame) -> HandAt | None:
    """hand21 키프레임 → `HandAt`. 손목이 보이지 않거나(visibility 0) 보간 키프레임이면 None."""
    if frame.interpolated or frame.points[HAND_WRIST].visibility == 0:
        return None
    w, m = frame.points[HAND_WRIST], frame.points[HAND_MIDDLE_MCP]
    seen = [p for p in frame.points if p.visibility > 0]
    xs, ys = [p.x for p in seen], [p.y for p in seen]
    box = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
    palm = math.dist((w.x, w.y), (m.x, m.y)) if m.visibility > 0 else 0.0
    return HandAt(hand, (w.x, w.y), max(palm, 1.0), box, frame)


@dataclass
class _Index[T]:
    """시각 → 항목 목록 (허용 오차 안 시각 찾기)."""

    data: dict[int, list[T]] = field(default_factory=dict[int, list[T]])
    keys: list[int] = field(default_factory=list[int])

    def add(self, t: int, item: T) -> None:
        """시각 t에 항목 하나를 더한다 (`freeze` 전에)."""
        self.data.setdefault(t, []).append(item)

    def freeze(self) -> None:
        """시각 키를 정렬한다 (다 더한 뒤 한 번, `near` 전에)."""
        self.keys = sorted(self.data)

    def near(self, t: int, tol: int) -> list[T]:
        """시각 t ± tol 안 항목 전부 (시각 순)."""
        lo, hi = bisect.bisect_left(self.keys, t - tol), bisect.bisect_right(self.keys, t + tol)
        return [x for k in self.keys[lo:hi] for x in self.data[k]]


class WearerEvidence:
    """한 스트림의 착용자 증거: 시각별 착용자 손(hand21)과 다른 사람의 보이는 머리 관절(coco17).

    같은 시각 허용 오차 안에서 같은 손이 둘 이상 잡히면 모두 돌려준다. 판정 함수
    (`wearer_box_frame`·`wearer_pose_frame`)는 그중 하나라도 맞으면 참이다.
    """

    def __init__(
        self, hands: Iterable[tuple[int, HandAt]], heads: Iterable[tuple[int, Point]] = ()
    ) -> None:
        self._hands: _Index[HandAt] = _Index()
        self._heads: _Index[Point] = _Index()
        for t, h in hands:
            self._hands.add(t, h)
        for t, p in heads:
            self._heads.add(t, p)
        self._hands.freeze()
        self._heads.freeze()

    @property
    def empty(self) -> bool:
        """손 증거가 하나도 없는가 (억제·팔 단계가 아무것도 하지 않는다)."""
        return not self._hands.keys

    @property
    def hand_times(self) -> list[int]:
        """손 증거가 있는 시각 (정렬)."""
        return self._hands.keys

    def hands_near(self, t: int, tol: int) -> list[HandAt]:
        """시각 t ± tol의 착용자 손."""
        return self._hands.near(t, tol)

    def heads_near(self, t: int, tol: int) -> list[Point]:
        """시각 t ± tol의 다른 사람 머리 관절."""
        return self._heads.near(t, tol)

    @classmethod
    def from_labels(cls, labels: Iterable[LabelRecord], stream_id: str) -> WearerEvidence:
        """한 스트림 운영 라벨의 hand21 트랙(착용자 손)과 coco17 트랙(보이는 머리 관절)으로 만든다.

        hand21은 손이 정해진(hand 필드) 관측 키프레임만, coco17은 visibility 2인 머리 관절만 쓴다.
        다른 스트림·다른 골격은 무시한다.
        """
        hands: list[tuple[int, HandAt]] = []
        heads: list[tuple[int, Point]] = []
        for x in labels:
            p = x.payload
            if x.stream_id != stream_id or not isinstance(p, KeypointTrackPayload):
                continue
            if p.skeleton == "hand21" and p.hand is not None:
                hands += [(f.t_ms, h) for f in p.keyframes if (h := hand_at(p.hand, f))]
            elif p.skeleton == "coco17":
                heads += [
                    (f.t_ms, (f.points[i].x, f.points[i].y))
                    for f in p.keyframes
                    if not f.interpolated
                    for i in COCO_HEAD
                    if f.points[i].visibility == 2
                ]
        return cls(hands, heads)


def inside(box: Box, points: Sequence[Point], margin: float) -> bool:
    """점 하나라도 (각 변을 너비·높이 x margin만큼 넓힌) 박스 안인가."""
    x, y, w, h = box
    x0, y0, x1, y1 = x - w * margin, y - h * margin, x + w * (1 + margin), y + h * (1 + margin)
    return any(x0 <= px <= x1 and y0 <= py <= y1 for px, py in points)


def _at_bottom(bottom_y: float, frame_h: int | None, bottom_band: float) -> bool:
    """아래 변이 화면 아래 띠 안인가 (프레임 높이를 모르면 참: 이 조건을 보지 않는다)."""
    return frame_h is None or bottom_y >= frame_h * (1 - bottom_band)


def wearer_box_frame(
    box: Box,
    hands: Sequence[HandAt],
    heads: Sequence[Point],
    *,
    margin: float,
    frame_h: int | None,
    bottom_band: float,
) -> bool:
    """박스 한 프레임이 착용자 몸으로 보이는가 (모듈 docstring의 박스 규칙).

    Args:
        box: 박스 (x, y, w, h) 픽셀.
        hands / heads: 그 시각의 착용자 손과 다른 사람 머리 관절 (`WearerEvidence`).
        margin: 박스를 넓히는 비율 (손목이 박스 경계에 걸쳐도 안으로 본다).
        frame_h: 프레임 높이(px). None이면 아래 가장자리 조건을 보지 않는다 (검수 힌트).
        bottom_band: 화면 높이에 대한 아래 띠 비율 (0~1).
    """
    wrists = [h.wrist for h in hands]
    return (
        inside(box, wrists, margin)
        and not inside(box, heads, margin)
        and _at_bottom(box[1] + box[3], frame_h, bottom_band)
    )


def wearer_pose_frame(
    points: Sequence[Keypoint],
    hands: Sequence[HandAt],
    *,
    frame_h: int | None,
    bottom_band: float,
    wrist_match_palms: float,
) -> bool:
    """COCO17 자세 한 프레임이 착용자 몸으로 보이는가 (모듈 docstring의 자세 규칙).

    Args:
        points: COCO17 17점 (스트림 픽셀).
        hands: 그 시각의 착용자 손.
        frame_h / bottom_band: `wearer_box_frame`과 같다 (아래 변 = 보이는 관절의 최대 y).
        wrist_match_palms: COCO 손목과 착용자 손목이 손 크기의 이 배 안이면 같은 손목으로 본다.
    """
    seen = [p for p in points if p.visibility > 0]
    if not seen or not hands:
        return False
    matched = any(
        math.dist((points[i].x, points[i].y), h.wrist) <= wrist_match_palms * h.palm_px
        for i in COCO_WRISTS
        if points[i].visibility > 0
        for h in hands
    )
    return matched and _at_bottom(max(p.y for p in seen), frame_h, bottom_band)
