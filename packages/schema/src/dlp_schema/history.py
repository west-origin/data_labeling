"""라벨 이력에서 검수가 모델 라벨을 어떻게 바꿨는지 읽는다.

재학습 예제(dlp_train)와 액티브 러닝 수정률(dlp_active)이 같이 쓴다.

- accepted: 모델 라벨이 그대로 인정됨 (states에 든 검증 상태, origin = 그 모델 레코드). 모델이 낸
  자식 레코드(부모가 있는 모델 레코드)도 자기 검증 상태로 판단한다 (수정이 아니다).
- corrected: 사람이 고침 (현재 레코드 출처가 사람, origin = 처음 모델 레코드)
- added: 모델이 놓친 것을 사람이 추가함 (origin 없음)
- deleted: 사람이 모델 라벨을 지움 (오탐, label = 지운 레코드)
운영 라벨만 본다 (오류 삽입·측정 레코드와 그 후손 제외). 모델 버전이 바뀌어 지운 삭제 레코드
(출처 model)는 사람의 삭제가 아니다.

위치
    WP13(재학습: 자동 원본과 수정본 차이) · WP14(액티브 러닝: 클래스별 수정률) · WP16(운영 지표:
    수정률·자동 승인율). DB를 직접 읽지 않는다. 호출자가 `db.repository.get_labels`로 세션의 전체
    이력을 읽어 넘긴다.

주요 이름
    - `Change`: 변화 종류 리터럴.
    - `ReviewChange`: 라벨 하나의 검수 결과.
    - `label_class`: 수정률을 세는 "클래스" 문자열.
    - `review_changes`: 세션 이력 → 검수 결과 목록.

주의
    - 이력은 불변 레코드의 사슬이다: 수정은 `parent_label_id`로 이전 레코드를 가리키는 새 레코드,
      삭제는 `retracted=True`인 새 레코드 (CLAUDE.md, ADR 0002).
    - 시각(`at`)은 datetime(시간대 있음)이다. 라벨 구간 시각(ms)과 혼동하지 않는다.
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

# 검수가 라벨에 한 일. 모듈 docstring의 정의를 따른다.
Change = Literal["accepted", "corrected", "added", "deleted"]


@dataclass(frozen=True)
class ReviewChange:
    """라벨 하나에 대한 검수 결과.

    Attributes:
        change: 변화 종류 (accepted / corrected / added / deleted).
        label: 최종 레코드. deleted면 사람이 지운 (모델 출처 사슬의) 레코드.
        origin: 사슬의 맨 처음 모델 레코드. added면 None. accepted면 label 자신.
        at: 검수 시각 (승인·수정 시각, 지웠으면 삭제 레코드의 검수 시각 또는 생성 시각).
        reviewer: 검수자 ID. 기록이 없으면 None.
    """

    change: Change
    label: LabelRecord  # 최종 레코드 (deleted면 지운 레코드)
    origin: LabelRecord | None  # 처음 모델이 낸 레코드
    at: datetime  # 검수한 시각 (승인·수정 시각, 지웠으면 삭제 레코드 시각)
    reviewer: str | None = None


def label_class(label: LabelRecord) -> str:
    """라벨의 클래스 (종류 안에서). 학습기가 배우고 수정률을 세는 단위.

    종류별 규칙:
        - box_track / mask_track: 객체 클래스 ID (`class_id`).
        - blur_track: 프라이버시 대상 ID (`target`).
        - keypoint_track: `<골격>/<left|right|body>` (예: `hand21/right`, `coco17/body`).
        - hand_state: 파지 유형, 없으면 접촉 대상 종류 (예: `tool_grip`, `none`).
        - action: 동사 ID.
        - object_state: `<속성>=<값>` (예: `cleanliness=clean`).
        - 그 밖(segment, gap, event, relation, coverage, trajectory3d, description): 라벨 종류 이름.

    Returns:
        클래스 문자열. 종류가 다르면 같은 문자열이어도 다른 클래스로 보는 것은 호출자 몫이다
        (보통 `(label.kind, label_class(label))` 쌍으로 묶는다).
    """
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
    """`parent_label_id` 사슬을 거슬러 올라가 맨 처음 레코드를 찾는다.

    부모가 이력(`by_id`)에 없으면 거기서 멈춘다. 순환이 있으면(데이터 오류) 한 바퀴 돌고 멈춘다.
    """
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

    Args:
        history: 세션 하나의 전체 라벨 이력 (`get_labels` 결과, 수정·삭제 레코드 포함).
        states: "검수됨"으로 볼 검증 상태 (예: human_approved, human_corrected, sample_verified).
            모델 출처 현재 라벨은 이 상태일 때만 accepted로 센다.

    Returns:
        현재 라벨에 대한 accepted / corrected / added 결과, 그다음 사람의 삭제(deleted) 결과.
        순서는 입력 이력 순서를 따른다.

    알고리즘:
        1) 운영 현재 라벨(`current_labels`)마다: 모델 출처면 accepted(states 안일 때만),
           사람 출처이고 사슬 맨 앞이 모델이면 corrected, 맨 앞도 사람이면 added.
        2) 삭제 레코드 중 사람 출처이고 운영 라벨이며 부모가 이력에 있는 것마다: 지운 레코드의
           사슬 맨 앞이 모델이면 deleted. 모델 출처 삭제(버전 교체 `retractions`)는 세지 않는다.
    """
    by_id = {x.label_id: x for x in history}
    excluded = non_operational_ids(history)
    out: list[ReviewChange] = []
    for x in current_labels(history):
        # 사람 라벨은 검증 상태와 무관하게 센다. 모델·센서 라벨은 states 안일 때만 센다.
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
            # 사람이 만든 사슬. 주의: 센서 출처 현재 라벨(states 안)도 사슬 맨 앞이 모델이 아니면
            # 여기로 와 added로 센다 (사람이 추가한 것이 아닌데도).
            out.append(ReviewChange("added", x, None, at, who))
    # 사람이 지운 모델 라벨 (오탐 제거)
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
