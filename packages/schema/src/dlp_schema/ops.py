"""운영 기록: 원본 접근 감사, 검수 작업 시간, 잔여 블러 감사, 원본 보관 결정 (WP16).

모두 추가만 하는 기록이다 (고치거나 지우지 않는다). 운영 지표와 감사 리포트는 이 기록에서 계산한다.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import AwareDatetime, Field, model_validator

from dlp_schema.common import Contract, Identifier, Ms

RawAction = Literal["read", "presign", "write", "grant"]


class RawAccessEvent(Contract):
    """원본(블러 전) 버킷 접근 하나. read=내려받기, presign=서명 URL 발급, write=올리기,
    grant=검수 도구에 원본을 올려 사람(actor)에게 보여 줌."""

    event_id: Identifier
    at: AwareDatetime
    actor: Identifier = Field(description="사람 또는 서비스 계정")
    purpose: str = Field(min_length=1, description="단계·명령 (예: privacy.detect, review.create)")
    action: RawAction
    bucket: str
    key: str
    session_id: Identifier | None = None


class ReviewWork(Contract):
    """검수 작업 시간. 검수 도구가 재거나(Label Studio lead_time) 사람이 기록한다."""

    work_id: Identifier
    task_key: Identifier | None = None
    session_id: Identifier
    reviewer: Identifier
    stage: Literal["privacy", "labeling", "qa"]
    seconds: float = Field(ge=0)
    video_ms: Ms = Field(description="검수한 영상 길이")
    source: Literal["label_studio", "cvat", "manual"]
    recorded_at: AwareDatetime


class PrivacyAuditRecord(Contract):
    """승인된 블러본의 잔여 누락 감사 결과 (원 검수자가 아닌 감사자)."""

    audit_id: Identifier
    session_id: Identifier
    stream_id: Identifier
    duration_ms: Ms
    misses: int = Field(ge=0)
    auditor: Identifier
    blur_reviewer: Identifier
    audited_at: AwareDatetime

    @model_validator(mode="after")
    def _independent(self) -> PrivacyAuditRecord:
        if self.auditor == self.blur_reviewer:
            raise ValueError("감사자는 원 블러 검수자와 달라야 합니다")
        return self


class RetentionDecision(Contract):
    """원본 보관 기간이 다가오거나 지난 세션에 대한 결정 (연장은 기한과 사유가 필요하다)."""

    decision_id: Identifier
    session_id: Identifier
    decision: Literal["extend", "delete"]
    until: date | None = None
    reason: str = Field(min_length=1)
    decided_by: Identifier
    decided_at: AwareDatetime

    @model_validator(mode="after")
    def _until(self) -> RetentionDecision:
        if (self.decision == "extend") != (self.until is not None):
            raise ValueError("연장(extend)에만 기한(until)이 있어야 합니다")
        return self
