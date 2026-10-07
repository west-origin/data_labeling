"""검수 작업: 외부 검수 도구(CVAT, Label Studio)의 작업과 세션·스트림의 대응."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import AwareDatetime, Field

from dlp_schema.common import Contract, Identifier, Ms


class ReviewTool(StrEnum):
    CVAT = "cvat"
    LABEL_STUDIO = "label_studio"


class ReviewStage(StrEnum):
    """프라이버시 검수는 원본 접근 권한자만, 작업 라벨 검수는 블러본만 본다."""

    PRIVACY = "privacy"
    LABELING = "labeling"
    QA = "qa"


class ReviewTaskStatus(StrEnum):
    OPEN = "open"
    COLLECTED = "collected"


class ReviewMode(StrEnum):
    """검수 방식. 블라인드·이중의 결과는 측정용 레코드, 오류 삽입 과제의 결과는 학습에서 빠진다."""

    STANDARD = "standard"  # 프리라벨을 고친다 (운영 라벨)
    BLIND = "blind"  # 프리라벨 없이 처음부터 (프리라벨 편향 측정)
    SEEDED_ERROR = "seeded_error"  # 정답에 오류를 넣은 사본을 고친다 (검수자 발견율 측정)
    DOUBLE = "double"  # 같은 구간을 다른 라벨러가 한 번 더 (일치도 측정)
    QA = "qa"  # 선임 재검수 (운영 라벨)


class ReviewReason(StrEnum):
    LOW_CONFIDENCE = "low_confidence"
    MODEL_DISAGREEMENT = "model_disagreement"
    NEW_OBJECT = "new_object"
    CONTACT_MISMATCH = "contact_mismatch"
    SAMPLE = "sample"  # 높은 신뢰도 묶음의 표본 검수
    ROUTINE = "routine"  # 사유 없는 일반 검수


class FlaggedSpan(Contract):
    """검수 단위 안에서 먼저 볼 구간."""

    reason: ReviewReason
    t_start_ms: Ms
    t_end_ms: Ms
    label_ids: tuple[Identifier, ...] = ()


class InjectedError(Contract):
    error_type: Literal["boundary_shift", "class_swap", "blur_deletion"]
    original_label_id: Identifier
    seeded_label_id: Identifier | None = Field(
        default=None, description="오류를 넣은 사본. 블러 삭제는 사본이 없다"
    )
    detail: dict[str, str | int | float] = Field(default_factory=dict[str, str | int | float])


class AssignmentStatus(StrEnum):
    OPEN = "open"
    DONE = "done"


class ReviewAssignment(Contract):
    """검수 단위(세션, 스트림, 라벨 종류 묶음)의 배정.

    우선순위·방식·담당자와 오류 삽입 기록을 남긴다.
    """

    assignment_id: Identifier
    session_id: Identifier
    stream_id: Identifier | None = None
    label_kinds: tuple[str, ...] = Field(min_length=1)
    mode: ReviewMode
    priority: float = Field(description="클수록 먼저")
    flagged: tuple[FlaggedSpan, ...] = ()
    assignee: Identifier | None = None
    pair_id: Identifier | None = Field(
        default=None, description="블라인드·이중·QA 배정이 짝을 이루는 표준 배정"
    )
    injected: tuple[InjectedError, ...] = ()
    sample_label_ids: tuple[Identifier, ...] = Field(
        default=(), description="높은 신뢰도 묶음의 표본. 이 배정의 결과로 묶음 합격 여부를 정한다"
    )
    withheld_label_ids: tuple[Identifier, ...] = Field(
        default=(), description="작업에 넣지 않은 묶음의 나머지. 표본이 합격하면 표본 검증이 된다"
    )
    only_label_ids: tuple[Identifier, ...] = Field(
        default=(), description="비어 있지 않으면 작업에 이 라벨만 넣는다 (불합격 묶음 재검수)"
    )
    task_key: Identifier | None = None
    status: AssignmentStatus = AssignmentStatus.OPEN
    created_at: AwareDatetime
    completed_at: AwareDatetime | None = None


class ReviewTask(Contract):
    task_key: Identifier  # 도구:작업 ID (예: cvat:42)
    tool: ReviewTool
    external_id: str
    session_id: Identifier
    stream_id: Identifier
    stage: ReviewStage
    assignee: Identifier | None = None
    media_uri: str
    label_kinds: tuple[str, ...]
    mode: ReviewMode = ReviewMode.STANDARD
    assignment_id: Identifier | None = None
    status: ReviewTaskStatus = ReviewTaskStatus.OPEN
    created_at: AwareDatetime
    collected_at: AwareDatetime | None = None
