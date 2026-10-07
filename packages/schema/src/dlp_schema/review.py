"""검수 작업: 외부 검수 도구(CVAT, Label Studio)의 작업과 세션·스트림의 대응."""

from __future__ import annotations

from enum import StrEnum

from pydantic import AwareDatetime

from dlp_schema.common import Contract, Identifier


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
    status: ReviewTaskStatus = ReviewTaskStatus.OPEN
    created_at: AwareDatetime
    collected_at: AwareDatetime | None = None
