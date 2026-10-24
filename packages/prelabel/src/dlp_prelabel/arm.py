"""착용자 팔 골격 arm3 (손목 → 아래팔 → 팔꿈치 → 어깨, ADR 0058).

가슴 장착 바디캠에서 착용자의 팔은 화면 아래 가장자리에서 들어오고 팔꿈치는 자주 화면 밖이다.
위에서 아래로 보는(top-down) 자세 모델(RTMPose)은 사람 박스가 있어야 도는데 착용자 몸은 화면에서
잘려 있다. 그래서 전신 어댑터(`adapters.rtmpose.RtmPose`)가 두 가지 자세를 후보로 넘긴다:

- 착용자로 판정한 YOLOX 박스 자세 (`dlp_schema.wearer_body.wearer_pose_frame`, coco17 트랙에서는
  빠진다).
- 합성 착용자 박스 자세: 그 시각 보이는 손 박스를 옆·위로 손 크기만큼 넓히고 아래는 화면 끝까지
  늘린 박스 (`synthetic_box`).

이 모듈은 후보 자세의 팔 사슬(COCO 왼팔 9-7-5, 오른팔 10-8-6)에서 손마다 하나를 고른다
(`estimate_arms`):

1. 손목 기준: 손목은 늘 같은 손 hand21 손목이다. 사슬 손목이 그 손목에서 손 크기 x
   wrist_match_palms 안인 사슬만 그 손의 팔이다. 좌우는 손 판정(hand21)을 따른다: 사슬의 COCO
   좌우는 보지 않는다 (RTMPose 좌우는 1인칭에서 자주 바뀐다). 가장 가까운 (손, 사슬) 쌍부터
   탐욕으로 짝짓고 한 사슬은 한 손에만 준다. 일치 정도 1 - 거리/허용 거리를 손목 점 신뢰도로
   싣는다 (사슬이 없으면 신뢰도 없음): 팔 모델 손목과 hand21 손목의 일치 대리 지표다.
2. 팔꿈치 판정 (`elbow_reason`): 점수 elbow_min_score 이상, 화면 안(가장자리 edge_margin_px 여유),
   화면 아래팔 길이 / 손 크기가 forearm_palms 안, 손가락 반대 방향(손목→팔꿈치와 손목→중지
   뿌리의 코사인 max_hand_cos 이하). 하나라도 어기면 보이지 않음(visibility 0)이고 좌표는 손목
   자리다 (화면 밖 팔꿈치를 지어내지 않는다).
3. 좌우 일관: 두 팔꿈치가 모두 보이는데 왼팔 팔꿈치가 오른팔 팔꿈치보다 손 크기 x
   cross_margin_palms 넘게 오른쪽이면 둘 다 보이지 않음으로 둔다 (side_conflict).
4. 어깨는 emit_shoulder일 때만 같은 방식(점수 shoulder_min_score, 화면 안)으로 내고, 아니면 늘
   보이지 않음이다 (가슴 장착 화면에는 어깨가 거의 없다).

결과 트랙(`arm_tracks`)은 손마다 하나, entity_id `<left|right>_arm`, hand = 그 손, 골격 arm3
(`dlp_schema.labels.ARM3_JOINTS`). 키프레임 시각은 그 스트림 PTS ms로 손 키프레임과 같은 시각이다.
보이는 팔꿈치가 하나도 없는 손은 트랙을 내지 않는다 (요약에는 센다).

정책: `prelabel.yaml arm`. 좌표: 스트림 픽셀, 왼쪽 위 원점.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

from dlp_prelabel.common import model_label
from dlp_prelabel.policy import ArmBoxPolicy, ArmPolicy
from dlp_schema.episode import version_tag
from dlp_schema.labels import (
    Hand,
    Keypoint,
    KeypointFrame,
    KeypointTrackPayload,
    LabelRecord,
)
from dlp_schema.predictor import Clip
from dlp_schema.wearer_body import HAND_MIDDLE_MCP, HandAt

# COCO17 팔 사슬 (손목, 팔꿈치, 어깨): 왼팔, 오른팔. 사슬의 COCO 좌우는 쓰지 않는다 (손 판정이 기준)
ARM_CHAINS = ((9, 7, 5), (10, 8, 6))
# 팔꿈치를 보이지 않음으로 둔 이유 (요약). ok는 보임
REASONS = ("ok", "no_chain", "low_score", "off_frame", "length", "direction", "side_conflict")


@dataclass(frozen=True)
class PoseCandidate:
    """후보 자세 하나: COCO17 (17, 2) 픽셀 좌표와 (17,) 점수."""

    xy: NDArray[np.float64]
    scores: NDArray[np.float64]


@dataclass(frozen=True)
class ArmFrame:
    """한 시각 한 손의 팔 추정.

    points: arm3 세 점 (손목, 팔꿈치, 어깨). elbow_score: 보이는 팔꿈치의 모델 점수 (아니면 None).
    reason: 팔꿈치 판정 (`REASONS`). wrist_agreement: 사슬 손목과 hand21 손목 일치 정도 (0~1,
    사슬이 없으면 None).
    """

    t_ms: int
    hand: Hand
    points: tuple[Keypoint, Keypoint, Keypoint]
    elbow_score: float | None
    reason: str
    wrist_agreement: float | None


@dataclass
class ArmReport:
    """팔 단계 요약 (라벨 아님): 손별 키프레임·보이는 팔꿈치 수, 이유별 수, 손목 일치 값."""

    keyframes: Counter[str] = field(default_factory=Counter[str])
    elbows: Counter[str] = field(default_factory=Counter[str])
    reasons: Counter[str] = field(default_factory=Counter[str])
    agreement: list[float] = field(default_factory=list[float])
    tracks: int = 0

    def add(self, f: ArmFrame) -> None:
        """한 손 한 시각 결과를 센다."""
        self.keyframes[f.hand.value] += 1
        self.elbows[f.hand.value] += f.reason == "ok"
        self.reasons[f.reason] += 1
        if f.wrist_agreement is not None:
            self.agreement.append(f.wrist_agreement)

    def text(self) -> str:
        """한 줄 요약 (CLI 출력)."""
        hands = ", ".join(
            f"{h} 팔꿈치 보임 {self.elbows[h]}/{n}" for h, n in sorted(self.keyframes.items())
        )
        matched = len(self.agreement) / max(sum(self.keyframes.values()), 1)
        reasons = ", ".join(f"{k} {v}" for k, v in sorted(self.reasons.items()))
        head = f"팔 트랙 {self.tracks}개, {hands or '손 없음'}"
        return f"{head}, 사슬 일치 {matched:.3f}, 이유 {reasons}"


def synthetic_box(
    hands: Sequence[HandAt], size: tuple[int, int], policy: ArmBoxPolicy
) -> tuple[float, float, float, float]:
    """합성 착용자 박스 (x1, y1, x2, y2) px: 손 박스 합집합을 옆·위로 넓히고 아래는 화면 끝까지.

    Args:
        hands: 그 시각 보이는 손 (1개 이상).
        size: 프레임 (너비, 높이) px.
    """
    w, h = size
    palm = max(x.palm_px for x in hands)
    x0 = min(x.box[0] for x in hands) - policy.side_palms * palm
    x1 = max(x.box[0] + x.box[2] for x in hands) + policy.side_palms * palm
    y0 = min(x.box[1] for x in hands) - policy.up_palms * palm
    return (max(0.0, x0), max(0.0, y0), min(float(w), x1), float(h))


def _in_frame(x: float, y: float, size: tuple[int, int], margin: float) -> bool:
    """화면 안(가장자리 margin 여유)인가."""
    w, h = size
    return margin <= x <= w - margin and margin <= y <= h - margin


def elbow_reason(
    hand: HandAt,
    elbow: tuple[float, float],
    score: float,
    size: tuple[int, int],
    policy: ArmPolicy,
) -> str:
    """팔꿈치 후보 판정 (모듈 docstring 2). "ok"면 보임, 아니면 이유."""
    if score < policy.elbow_min_score:
        return "low_score"
    if not _in_frame(*elbow, size, policy.edge_margin_px):
        return "off_frame"
    wx, wy = hand.wrist
    v = (elbow[0] - wx, elbow[1] - wy)
    length = math.hypot(*v)
    low, high = policy.forearm_palms
    if not low <= length / hand.palm_px <= high:
        return "length"
    mcp = hand.frame.points[HAND_MIDDLE_MCP]
    u = (mcp.x - wx, mcp.y - wy)
    norm = math.hypot(*u) * length
    if mcp.visibility > 0 and norm > 0 and (u[0] * v[0] + u[1] * v[1]) / norm > policy.max_hand_cos:
        return "direction"
    return "ok"


def _pairs(
    hands: dict[Hand, HandAt], poses: Sequence[PoseCandidate], policy: ArmPolicy
) -> dict[Hand, tuple[PoseCandidate, tuple[int, int, int], float]]:
    """손 → (자세, 사슬, 손 크기로 나눈 손목 거리). 가까운 쌍부터 탐욕, 한 사슬은 한 손에만."""
    cands: list[tuple[float, Hand, int, int]] = []
    for hand, h in hands.items():
        for pi, pose in enumerate(poses):
            for ci, chain in enumerate(ARM_CHAINS):
                d = math.dist(tuple(pose.xy[chain[0]]), h.wrist) / h.palm_px
                if d <= policy.wrist_match_palms:
                    cands.append((d, hand, pi, ci))
    out: dict[Hand, tuple[PoseCandidate, tuple[int, int, int], float]] = {}
    used: set[tuple[int, int]] = set()
    for d, hand, pi, ci in sorted(cands, key=lambda c: (c[0], c[1].value, c[2], c[3])):
        if hand in out or (pi, ci) in used:
            continue
        out[hand] = (poses[pi], ARM_CHAINS[ci], d)
        used.add((pi, ci))
    return out


def _hidden(h: HandAt) -> Keypoint:
    """보이지 않는 점 (visibility 0, 좌표는 손목 자리: 의미 없음)."""
    return Keypoint(x=h.wrist[0], y=h.wrist[1], visibility=0)


def _point(xy: NDArray[np.float64], score: float) -> Keypoint:
    """보이는 점 (소수 셋째 자리, 신뢰도 = 모델 점수)."""
    return Keypoint(
        x=round(float(xy[0]), 3),
        y=round(float(xy[1]), 3),
        visibility=2,
        confidence=round(min(max(score, 0.0), 1.0), 4),
    )


def _one_arm(
    t_ms: int,
    h: HandAt,
    pair: tuple[PoseCandidate, tuple[int, int, int], float] | None,
    size: tuple[int, int],
    policy: ArmPolicy,
) -> ArmFrame:
    """한 손의 팔 (좌우 충돌 검사 전)."""
    if pair is None:
        wrist = Keypoint(x=h.wrist[0], y=h.wrist[1], visibility=2)
        return ArmFrame(t_ms, h.hand, (wrist, _hidden(h), _hidden(h)), None, "no_chain", None)
    pose, (_, ei, si), d = pair
    agreement = round(1 - d / policy.wrist_match_palms, 4)
    wrist = Keypoint(x=h.wrist[0], y=h.wrist[1], visibility=2, confidence=agreement)
    e_score = float(pose.scores[ei])
    reason = elbow_reason(h, (float(pose.xy[ei][0]), float(pose.xy[ei][1])), e_score, size, policy)
    elbow = _point(pose.xy[ei], e_score) if reason == "ok" else _hidden(h)
    s_score = float(pose.scores[si])
    show_shoulder = (
        policy.emit_shoulder
        and reason == "ok"
        and s_score >= policy.shoulder_min_score
        and _in_frame(float(pose.xy[si][0]), float(pose.xy[si][1]), size, policy.edge_margin_px)
    )
    shoulder = _point(pose.xy[si], s_score) if show_shoulder else _hidden(h)
    score = e_score if reason == "ok" else None
    return ArmFrame(t_ms, h.hand, (wrist, elbow, shoulder), score, reason, agreement)


def _conflict(arms: dict[Hand, ArmFrame], hands: dict[Hand, HandAt], policy: ArmPolicy) -> bool:
    """두 팔꿈치가 모두 보이는데 왼팔 팔꿈치가 오른팔 팔꿈치보다 많이 오른쪽인가."""
    left, right = arms.get(Hand.LEFT), arms.get(Hand.RIGHT)
    if left is None or right is None or left.reason != "ok" or right.reason != "ok":
        return False
    palm = (hands[Hand.LEFT].palm_px + hands[Hand.RIGHT].palm_px) / 2
    return left.points[1].x - right.points[1].x > policy.cross_margin_palms * palm


def estimate_arms(
    t_ms: int,
    hands: dict[Hand, HandAt],
    poses: Sequence[PoseCandidate],
    size: tuple[int, int],
    policy: ArmPolicy,
) -> dict[Hand, ArmFrame]:
    """한 시각의 손마다 팔 추정 (모듈 docstring 1~4).

    Args:
        t_ms: 스트림 PTS ms (손 키프레임 시각).
        hands: 그 시각 보이는 착용자 손 (손 → `HandAt`).
        poses: 후보 자세 (착용자 YOLOX 자세, 합성 박스 자세).
        size: 프레임 (너비, 높이) px.
        policy: `prelabel.yaml arm`.
    """
    pairs = _pairs(hands, poses, policy)
    arms = {hand: _one_arm(t_ms, h, pairs.get(hand), size, policy) for hand, h in hands.items()}
    if _conflict(arms, hands, policy):
        for hand, f in list(arms.items()):
            h = hands[hand]
            pts = (f.points[0], _hidden(h), _hidden(h))
            arms[hand] = ArmFrame(t_ms, hand, pts, None, "side_conflict", f.wrist_agreement)
    return arms


def arm_tracks(
    frames: dict[Hand, list[ArmFrame]],
    clip: Clip,
    *,
    prefix: str,
    version: str,
    ontology_version: str,
    now: datetime,
) -> list[LabelRecord]:
    """손마다 arm3 키포인트 트랙 하나 (보이는 팔꿈치가 없으면 내지 않는다).

    라벨 ID `<세션>-<스트림>-<prefix>-<version_tag>-arm-<손>`, entity_id `<손>_arm`. 신뢰도는 보이는
    팔꿈치 점수의 평균.
    """
    out: list[LabelRecord] = []
    for hand in sorted(frames, key=lambda h: h.value):
        fs = sorted(frames[hand], key=lambda f: f.t_ms)
        scores = [f.elbow_score for f in fs if f.elbow_score is not None]
        if not scores:
            continue
        payload = KeypointTrackPayload(
            entity_id=f"{hand.value}_arm",
            skeleton="arm3",
            hand=hand,
            keyframes=tuple(KeypointFrame(t_ms=f.t_ms, points=f.points) for f in fs),
        )
        out.append(
            model_label(
                label_id=f"{clip.session_id}-{clip.stream_id}-{prefix}-{version_tag(version)}-arm-{hand.value}",
                session_id=clip.session_id,
                stream_id=clip.stream_id,
                t_start_ms=fs[0].t_ms,
                t_end_ms=fs[-1].t_ms,
                ontology_version=ontology_version,
                model_version=version,
                confidence=sum(scores) / len(scores),
                payload=payload,
                now=now,
            )
        )
    return out
