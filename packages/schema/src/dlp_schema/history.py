"""라벨 이력에서 검수가 모델 라벨을 어떻게 바꿨는지 읽는다.

재학습 예제(dlp_train)와 액티브 러닝 수정률(dlp_active)이 같이 쓴다.

- accepted: 모델 라벨이 그대로 인정됨 (states에 든 검증 상태, origin = 그 모델 레코드). 모델이 낸
  자식 레코드(부모가 있는 모델 레코드)도 자기 검증 상태로 판단한다 (수정이 아니다).
- corrected: 사람이 고침 (현재 레코드 출처가 사람, origin = 처음 모델 레코드)
- added: 모델이 놓친 것을 사람이 추가함 (origin 없음)
- deleted: 사람이 모델 라벨을 지움 (오탐, label = 지운 레코드)
운영 라벨만 본다 (오류 삽입·측정 레코드와 그 후손 제외). 모델 버전이 바뀌어 지운 삭제 레코드
(출처 model)는 사람의 삭제가 아니다.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from dlp_schema.episode import current_labels, non_operational_ids
from dlp_schema.labels import (
    ActionPayload,
    BlurTrackPayload,
    BoxTrackPayload,
    HandStatePayload,
    KeypointTrackPayload,
    LabelRecord,
    MaskTrackPayload,
    ObjectStatePayload,
    Source,
    VerificationState,
)

Change = Literal["accepted", "corrected", "added", "deleted"]


@dataclass(frozen=True)
class ReviewChange:
    change: Change
    label: LabelRecord  # 최종 레코드 (deleted면 지운 레코드)
    origin: LabelRecord | None  # 처음 모델이 낸 레코드
    at: datetime  # 검수한 시각 (승인·수정 시각, 지웠으면 삭제 레코드 시각)
    reviewer: str | None = None


def label_class(label: LabelRecord) -> str:
    """라벨의 클래스 (종류 안에서). 학습기가 배우고 수정률을 세는 단위."""
    p = label.payload
    match p:
        case BoxTrackPayload() | MaskTrackPayload():
            return p.class_id
        case BlurTrackPayload():
            return p.target
        case KeypointTrackPayload():
            return f"{p.skeleton}/{p.hand.value if p.hand else 'body'}"
        case HandStatePayload():
            return p.grasp_type or p.contact_target_kind
        case ActionPayload():
            return p.verb
        case ObjectStatePayload():
            return f"{p.attribute}={p.value}"
        case _:
            return label.kind


def _root(label: LabelRecord, by_id: dict[str, LabelRecord]) -> LabelRecord:
    seen: set[str] = set()
    x = label
    while x.parent_label_id and x.parent_label_id in by_id and x.label_id not in seen:
        seen.add(x.label_id)
        x = by_id[x.parent_label_id]
    return x


def review_changes(
    history: list[LabelRecord], states: Collection[VerificationState]
) -> list[ReviewChange]:
    """세션 하나의 라벨 이력에서 검수 결과.

    사람이 만든 현재 라벨과 states 상태의 모델 라벨만 센다.
    """
    by_id = {x.label_id: x for x in history}
    excluded = non_operational_ids(history)
    out: list[ReviewChange] = []
    for x in current_labels(history):
        if x.provenance.source is not Source.HUMAN and x.verification.state not in states:
            continue
        root = _root(x, by_id)
        at = x.verification.reviewed_at or x.created_at
        who = x.verification.reviewer_id
        if x.provenance.source is Source.MODEL:
            # 모델이 낸 현재 레코드는 부모가 있어도 (예: 3인칭 착용자 사본, parent=원래 트랙) 사람의
            # 수정이 아니다. 자기 검증 상태로 판단한다: states 안이면 그대로 인정된 것이다.
            out.append(ReviewChange("accepted", x, x, at, who))
        elif root.provenance.source is Source.MODEL:
            out.append(ReviewChange("corrected", x, root, at, who))
        else:
            out.append(ReviewChange("added", x, None, at, who))
    for r in history:
        if (
            not r.retracted
            or r.provenance.source is not Source.HUMAN
            or r.label_id in excluded
            or r.parent_label_id not in by_id
        ):
            continue
        gone = by_id[r.parent_label_id]
        root = _root(gone, by_id)
        if root.provenance.source is Source.MODEL:
            at = r.verification.reviewed_at or r.created_at
            out.append(ReviewChange("deleted", gone, root, at, r.verification.reviewer_id))
    return out
