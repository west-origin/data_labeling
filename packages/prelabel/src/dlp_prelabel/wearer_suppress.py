"""착용자 자신의 몸 탐지를 라벨로 만들지 않는다 (ADR 0058).

가슴 장착 바디캠의 COCO 사람 박스(`MediaPipeObjects`), OWLv2 박스(`OwlObjects`), YOLOX→RTMPose
자세(`RtmPose`)가 착용자 자신의 팔·몸통을 잡으면 그것은 객체·사람 라벨이 아니다. 판정 규칙은 검수
힌트와 같은 `dlp_schema.wearer_body`이고, 정책은 `prelabel.yaml wearer_suppression`이다.

- 박스 트랙: 키프레임마다 `wearer_box_frame`(사람 클래스) 또는 그것 + 면적 문턱(사람 아닌 클래스,
  `other_min_area`)으로 착용자 프레임인지 보고, 그 비율이 track_share 이상이면 트랙째 버린다.
  트랙의 일부만 착용자면 남긴다 (사람이 바뀐 트랙을 반쯤 자르지 않는다. 검수 힌트가 남는다).
- 자세: 프레임마다 `wearer_pose_frame`이면 그 자세를 coco17 트랙에 넣지 않는다 (팔 단계 후보로는
  쓴다, `dlp_prelabel.arm`).

러너(`runner.run_prelabel`)가 바디캠 스트림에서만 어댑터에 착용자 증거(`WearerEvidence`, 같은
스트림의 지금 손·전신 트랙)를 넘긴다 (`WearerAware.set_wearer`). 3인칭 스트림이나 손 트랙이 없으면
None이라 억제하지 않는다. 버린 것은 라벨이 아니라 `WearerReport`로 단계 요약에만 남는다.
프라이버시 탐지(블러)는 이 모듈과 무관하다 (블러를 약하게 하지 않는다).

공개 이름: `WearerReport`, `WearerAware`, `wearer_tag`, `box_frame_is_wearer`,
`suppress_box_tracks`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from dlp_prelabel.common import BoxTrack
from dlp_prelabel.policy import PrelabelPolicy, WearerSuppressionPolicy
from dlp_schema.wearer_body import Box, WearerEvidence, wearer_box_frame


@dataclass
class WearerReport:
    """한 어댑터 실행에서 착용자 몸이라 라벨로 만들지 않은 것 (요약·대리 지표용, 라벨 아님).

    tracks / keyframes: 버린 박스 트랙 수와 그 키프레임 수. by_class: 클래스별 버린 트랙 수.
    poses: 버린 자세 프레임 수 (전신 단계). kept_person_tracks: 남긴 사람 클래스 박스 트랙 수.
    """

    tracks: int = 0
    keyframes: int = 0
    poses: int = 0
    kept_person_tracks: int = 0
    by_class: Counter[str] = field(default_factory=Counter[str])

    def text(self) -> str:
        """한 줄 요약 (CLI 출력)."""
        classes = ", ".join(f"{k} {v}" for k, v in sorted(self.by_class.items()))
        parts = [f"박스 트랙 {self.tracks}개(키프레임 {self.keyframes})"]
        if classes:
            parts.append(f"클래스별 {classes}")
        if self.poses:
            parts.append(f"자세 프레임 {self.poses}개")
        parts.append(f"남긴 사람 박스 트랙 {self.kept_person_tracks}개")
        return ", ".join(parts)


class WearerAware:
    """착용자 증거를 받는 어댑터의 공통 부분 (러너가 `set_wearer`로 넘기고 `last_wearer`를 읽는다).

    wearer: 이번 실행의 착용자 증거 (None이면 억제·팔 단계를 하지 않는다).
    last_wearer: 마지막 실행의 억제 요약 (억제하지 않았으면 None).
    """

    wearer: WearerEvidence | None = None
    last_wearer: WearerReport | None = None

    def set_wearer(self, evidence: WearerEvidence | None) -> None:
        """다음 `run`에 쓸 착용자 증거를 정한다 (빈 증거는 None과 같다)."""
        self.wearer = None if evidence is None or evidence.empty else evidence


def wearer_tag(policy: PrelabelPolicy, stage: str) -> str:
    """어댑터 버전 접미사: 억제를 적용하는 단계면 `+w<wearer_suppression 절 해시>`, 아니면 "".

    억제 정책이 바뀌면 그 단계가 다시 돌고 검수 전인 이전 결과를 지운다 (ADR 0015).
    """
    if stage not in policy.wearer_suppression.stages:
        return ""
    return f"+w{policy.digest('wearer_suppression')}"


def box_frame_is_wearer(
    box: Box,
    class_id: str,
    t_ms: int,
    evidence: WearerEvidence,
    policy: WearerSuppressionPolicy,
    size: tuple[int, int],
) -> bool:
    """박스 한 키프레임이 착용자 몸인가 (사람 클래스는 박스 규칙, 아니면 + 면적 문턱).

    size: 프레임 (너비, 높이) px. 사람 아닌 클래스이고 other_min_area가 None이면 늘 거짓이다.
    """
    person = class_id in policy.person_classes
    if not person:
        if policy.other_min_area is None:
            return False
        w, h = size
        if w <= 0 or h <= 0 or box[2] * box[3] < policy.other_min_area * w * h:
            return False
    tol = policy.time_tolerance_ms
    return wearer_box_frame(
        box,
        evidence.hands_near(t_ms, tol),
        evidence.heads_near(t_ms, tol),
        margin=policy.margin,
        frame_h=size[1] or None,
        bottom_band=policy.bottom_band,
    )


def suppress_box_tracks(
    tracks: list[BoxTrack],
    evidence: WearerEvidence | None,
    policy: WearerSuppressionPolicy,
    size: tuple[int, int],
) -> tuple[list[BoxTrack], WearerReport | None]:
    """착용자 프레임 비율이 track_share 이상인 박스 트랙을 뺀다.

    Args:
        tracks: 박스 트랙 (key = 온톨로지 클래스 ID).
        evidence: 착용자 증거. None이거나 정책이 꺼져 있으면 그대로 돌려주고 요약은 None.
        policy: `prelabel.yaml wearer_suppression`. size: 프레임 (너비, 높이) px.

    Returns:
        (남긴 트랙, 요약).
    """
    if evidence is None or not policy.enabled:
        return tracks, None
    report = WearerReport()
    kept: list[BoxTrack] = []
    for tr in tracks:
        hits = sum(
            box_frame_is_wearer(box, tr.key, t, evidence, policy, size)
            for t, (box, _) in tr.frames.items()
        )
        if tr.frames and hits / len(tr.frames) >= policy.track_share:
            report.tracks += 1
            report.keyframes += len(tr.frames)
            report.by_class[tr.key] += 1
            continue
        kept.append(tr)
        report.kept_person_tracks += tr.key in policy.person_classes
    return kept, report
