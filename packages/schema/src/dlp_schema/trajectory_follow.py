"""3D 손 궤적이 그것을 계산한 2D 손 트랙의 검수 결과를 따른다 (F-COE-19, ADR 0048).

3D 궤적(`Trajectory3DPayload`)은 검수 도구(CVAT·Label Studio)에 경로가 없다. 그래서 기본
내보내기(검수 라벨만)에서 로봇 목표·손 3D 점이 비었다. 사용자가 고른 규칙은 "3D 궤적은 그것을
계산한 2D 손 키포인트 트랙의 검수 상태를 물려받는다"다. 이 모듈은 세션 이력과 물려받기 기록만
보고 결과를 정하는 순수 계산이다. DB 적용은 `dlp_schema.db.trajectory_repository`가 맡는다.
설명 따르기(`dlp_schema.description_follow`, ADR 0043 3절)와 같은 틀이다.

출처 (`trajectory_basis`, 모두 지금 운영 현재 라벨이어야 한다)
    - 카메라 좌표 손 궤적: 모델 출처, frame=camera, `derived_from` = (2D 트랙 하나). 그 2D 트랙이
      hand21 키포인트 트랙이고 개체(entity_id)가 같아야 한다. 라벨은 덮어쓰지 않으므로(고치면 새
      ID) derived_from의 ID가 지금 현재 트랙이면 3D는 정확히 그 2D 내용으로 계산된 것이다.
    - 월드 좌표 손 궤적: 모델 출처, frame=world, `derived_from` = (카메라 좌표 궤적, 카메라 자세).
      카메라 좌표 궤적이 위 조건을 만족하고(같은 개체·부위) 자세가 현재 자세 궤적이어야 한다.
      카메라 자세는 자동 추정이고 물려받지 않는다 (검수 근거 표시는 내보내기 manifest).
    - derived_from이 없는 예전 3D(`depth3d-…+hf2` 이전), 박스 중심 3D, 카메라 자세, 사람 출처
      궤적은 출처가 없다 (물려받지 않는다).

규칙 (`plan_trajectory_follow`)
    1. 검수 전 모델 궤적의 출처 2D 트랙이 사람이 확인한 상태(`human_confirmed`: 승인·표본 검증·
       사람 수정, 또는 검수자가 있는 사람 출처)면 물려받는다.
       - 상태: 2D가 표본 검증이면 sample_verified, 아니면 human_approved.
       - 검수자: 2D의 검수자. 시각: max(2D 검수 시각, 3D 생성 시각).
    2. 2D가 아직 검수 전이면 기다린다. 2D가 고쳐져(새 레코드) 3D가 낡으면 출처가 현재가 아니므로
       물려받지 않는다. 다음 `dlp prelabel run`이 고친 2D로 새 3D를 만들고(입력 해시가 바뀐다) 그
       새 3D가 물려받는다.
    3. 물려받은 검수만 있는 궤적(마지막 기록과 지금 검수 상태가 같다)의 출처가 더는 성립하지 않으면
       (2D가 고쳐지거나 지워짐, 카메라 좌표 궤적·자세가 바뀜) "끊김"(`lapsed`)이다. 끊긴 궤적은
       지우지 않는다: CLAUDE.md의 좁은 예외는 물려받은 설명만 다루므로 3D를 지우려면 규칙을 넓혀야
       한다 (ADR 0048 "남은 결정"). 대신 프리라벨이 그 옆에 새 3D를 만들 수 있게 보호 판정에서
       빼고(`dlp_prelabel.runner`), 내보내기는 끊긴 궤적을 넣지 않는다(`dlp_export.source`).
    4. 직접 검수(기록 없음, 기록 뒤 상태가 바뀜)와 사람 출처 궤적은 건드리지 않고 끊김으로도 보지
       않는다.

결과는 결정적이고 멱등이다 (적용한 뒤 다시 계산하면 approve가 빈다).

공개 이름: `TRAJECTORY_FOLLOW_VERSION`, `TrajectoryBasis`, `InheritedTrajectoryReview`,
`TrajectoryFollow`, `trajectory_basis`, `plan_trajectory_follow`, `active_trajectory_inheritance`,
`lapsed_trajectories`, `review_basis`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

from dlp_schema.description_follow import human_confirmed
from dlp_schema.episode import current_labels
from dlp_schema.labels import (
    CoordinateFrame,
    KeypointTrackPayload,
    LabelRecord,
    Source,
    Trajectory3DPayload,
    VerificationState,
)
from dlp_schema.review import TrajectoryInheritance, TrajectoryInheritanceMechanism

# 이 규칙의 버전 (물려받기 기록 `rule_version`, 정책 값이 없다)
TRAJECTORY_FOLLOW_VERSION = "trajectory-follow-1"

# 카메라 자세 궤적의 개체 ID (`dlp_prelabel.campose`, ADR 0046)
CAMERA_ENTITY = "camera"


@dataclass(frozen=True)
class TrajectoryBasis:
    """3D 궤적의 검수 출처: 2D 손 트랙과 그 사이 입력 (월드 좌표: 카메라 좌표 궤적, 자세)."""

    source: LabelRecord
    via: tuple[str, ...] = ()


@dataclass(frozen=True)
class InheritedTrajectoryReview:
    """궤적에 기록할 검수 상태 (`record_review` 인자)와 출처."""

    label_id: str
    state: VerificationState
    reviewer_id: str
    reviewed_at: datetime
    source_label_id: str
    via_label_ids: tuple[str, ...] = ()

    def record(
        self, session_id: str, mechanism: TrajectoryInheritanceMechanism, recorded_at: datetime
    ) -> TrajectoryInheritance:
        """이 물려받음의 명시 기록 (`trajectory_inheritance` 한 행, event_id는 DB가 매긴다)."""
        return TrajectoryInheritance(
            event_id=1,
            session_id=session_id,
            trajectory_label_id=self.label_id,
            source_label_id=self.source_label_id,
            via_label_ids=self.via_label_ids,
            state=self.state,
            reviewer_id=self.reviewer_id,
            reviewed_at=self.reviewed_at,
            mechanism=mechanism,
            rule_version=TRAJECTORY_FOLLOW_VERSION,
            recorded_at=recorded_at,
        )


@dataclass
class TrajectoryFollow:
    """궤적이 따를 결과. approve는 검수 상태 기록, lapsed는 출처가 끊긴 물려받은 궤적 (안 지움)."""

    approve: list[InheritedTrajectoryReview] = field(
        default_factory=list[InheritedTrajectoryReview]
    )
    lapsed: list[LabelRecord] = field(default_factory=list[LabelRecord])


def _hand_track_of(
    x: LabelRecord, p: Trajectory3DPayload, current: dict[str, LabelRecord]
) -> LabelRecord | None:
    """카메라 좌표 궤적 x의 출처 2D 손 트랙 (현재, hand21, 같은 개체·스트림). 아니면 None."""
    if p.frame is not CoordinateFrame.CAMERA or p.derived_from is None or len(p.derived_from) != 1:
        return None
    g = current.get(p.derived_from[0])
    if g is None or g.stream_id != x.stream_id:
        return None
    gp = g.payload
    if not isinstance(gp, KeypointTrackPayload) or gp.skeleton != "hand21":
        return None
    return g if gp.entity_id == p.entity_id else None


def _world_basis(p: Trajectory3DPayload, current: dict[str, LabelRecord]) -> TrajectoryBasis | None:
    """월드 좌표 궤적의 출처 (카메라 좌표 궤적 → 2D 트랙, 자세는 경유)."""
    if p.derived_from is None or len(p.derived_from) != 2 or p.entity_id == CAMERA_ENTITY:
        return None
    cam, pose = (current.get(i) for i in p.derived_from)
    if cam is None or pose is None:
        return None
    cp, pp = cam.payload, pose.payload
    if (
        not isinstance(cp, Trajectory3DPayload)
        or not isinstance(pp, Trajectory3DPayload)
        or pp.entity_id != CAMERA_ENTITY
        or (cp.entity_id, cp.part) != (p.entity_id, p.part)
        or cam.provenance.source is not Source.MODEL
    ):
        return None
    g = _hand_track_of(cam, cp, current)
    return None if g is None else TrajectoryBasis(g, (cam.label_id, pose.label_id))


def trajectory_basis(x: LabelRecord, current: dict[str, LabelRecord]) -> TrajectoryBasis | None:
    """모델 3D 궤적 x의 검수 출처 (모듈 docstring "출처"). 없으면 None.

    Args:
        x: 라벨.
        current: 지금 운영 현재 라벨 (라벨 ID → 레코드).
    """
    p = x.payload
    if not isinstance(p, Trajectory3DPayload) or x.provenance.source is not Source.MODEL:
        return None
    if p.frame is CoordinateFrame.WORLD:
        return _world_basis(p, current)
    g = _hand_track_of(x, p, current)
    return None if g is None else TrajectoryBasis(g)


def _inherit(x: LabelRecord, basis: TrajectoryBasis) -> InheritedTrajectoryReview | None:
    """출처 2D가 확인됐으면 x가 물려받을 검수 (규칙 1), 아니면 None."""
    g = basis.source
    if not human_confirmed(g):
        return None
    v = g.verification
    assert v.reviewer_id is not None and v.reviewed_at is not None  # human_confirmed가 보장한다
    sampled = v.state is VerificationState.SAMPLE_VERIFIED
    state = VerificationState.SAMPLE_VERIFIED if sampled else VerificationState.HUMAN_APPROVED
    at = max(v.reviewed_at, x.created_at)
    return InheritedTrajectoryReview(x.label_id, state, v.reviewer_id, at, g.label_id, basis.via)


def _latest(inheritance: Sequence[TrajectoryInheritance]) -> dict[str, TrajectoryInheritance]:
    """궤적 라벨 ID → 마지막 물려받기 기록 (일련번호 순)."""
    out: dict[str, TrajectoryInheritance] = {}
    for r in sorted(inheritance, key=lambda r: r.event_id):
        out[r.trajectory_label_id] = r
    return out


def _active(
    x: LabelRecord, latest: dict[str, TrajectoryInheritance]
) -> TrajectoryInheritance | None:
    """`active_trajectory_inheritance`의 계산 (마지막 기록을 미리 모은 판)."""
    if not isinstance(x.payload, Trajectory3DPayload) or x.provenance.source is Source.HUMAN:
        return None
    r = latest.get(x.label_id)
    v = x.verification
    if r is None or (r.state, r.reviewer_id, r.reviewed_at) != (
        v.state,
        v.reviewer_id,
        v.reviewed_at,
    ):
        return None
    return r


def active_trajectory_inheritance(
    x: LabelRecord, inheritance: Sequence[TrajectoryInheritance]
) -> TrajectoryInheritance | None:
    """x의 지금 검수 상태를 준 물려받기 기록. 물려받은 것이 아니면 None (기록 없는 검수는 직접)."""
    return _active(x, _latest(inheritance))


def _holds(x: LabelRecord, r: TrajectoryInheritance, current: dict[str, LabelRecord]) -> bool:
    """물려받은 기록의 출처가 지금도 성립하는가 (같은 2D·경유, 2D가 아직 확인 상태)."""
    basis = trajectory_basis(x, current)
    return (
        basis is not None
        and basis.source.label_id == r.source_label_id
        and basis.via == r.via_label_ids
        and human_confirmed(basis.source)
    )


def plan_trajectory_follow(
    history: Sequence[LabelRecord], inheritance: Sequence[TrajectoryInheritance]
) -> TrajectoryFollow:
    """세션 이력에서 3D 궤적이 따를 결과를 정한다 (모듈 docstring 규칙). 순수 함수.

    Args:
        history: 세션의 라벨 이력 전체 (`get_labels`, 종류를 거르지 않는다).
        inheritance: 세션의 궤적 물려받기 기록 (`get_trajectory_inheritance`). 기본값이 없다:
            빠뜨리면 물려받은 궤적을 직접 검수로 보아 끊김을 놓친다 (F-PAR-24와 같은 이유).

    Returns:
        `TrajectoryFollow` (현재 라벨 순서).
    """
    current = current_labels(list(history))
    by_id = {x.label_id: x for x in current}
    latest = _latest(inheritance)
    out = TrajectoryFollow()
    for x in current:
        if not isinstance(x.payload, Trajectory3DPayload) or x.provenance.source is Source.HUMAN:
            continue
        if x.verification.state is not VerificationState.UNREVIEWED:
            r = _active(x, latest)
            if r is not None and not _holds(x, r, by_id):
                out.lapsed.append(x)  # 규칙 3 (지우지 않는다)
            continue  # 직접 검수는 그대로 (규칙 4)
        basis = trajectory_basis(x, by_id)
        review = _inherit(x, basis) if basis is not None else None
        if review is not None:
            out.approve.append(review)
    return out


def lapsed_trajectories(
    history: Sequence[LabelRecord], inheritance: Sequence[TrajectoryInheritance]
) -> frozenset[str]:
    """출처가 끊긴 물려받은 궤적의 라벨 ID (규칙 3). 프리라벨 보호 판정·내보내기가 뺀다."""
    return frozenset(x.label_id for x in plan_trajectory_follow(history, inheritance).lapsed)


def review_basis(x: LabelRecord, inheritance: Sequence[TrajectoryInheritance]) -> str:
    """내보내기 manifest의 검수 근거 이름.

    human(사람 출처), inherited_from_2d(2D 손 트랙에서 물려받음), direct(기록 없는 검수),
    unreviewed(검수 전).
    """
    if x.provenance.source is Source.HUMAN:
        return "human"
    if x.verification.state is VerificationState.UNREVIEWED:
        return "unreviewed"
    return "inherited_from_2d" if active_trajectory_inheritance(x, inheritance) else "direct"
