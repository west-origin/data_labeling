"""운영 기록: 원본 접근 감사, 검수 작업 시간, 잔여 블러 감사, 원본 보관 결정 (WP16).

모두 추가만 하는 기록이다 (고치거나 지우지 않는다). 운영 지표와 감사 리포트는 이 기록에서 계산한다.

위치·저장
    - `RawAccessEvent` → DB `raw_access_log`. `dlp_cli.raw_access.raw_store`(감사 저장소)가 원본
      버킷 접근마다 남긴다 (ADR 0020, 0021). DB 없는 로컬
      개발은 로컬 JSON Lines에 같은 형태로 남긴다.
    - `ReviewWork` → `review_work`. `dlp ops log-work` 또는 검수 도구 웹훅 수집이 남긴다.
    - `PrivacyAuditRecord` → `privacy_audits`. `dlp ops privacy-audit`가 남긴다.
    - `RetentionDecision` → `retention_decisions`. `dlp ops retention-decide`가 남긴다.
    DB에서는 Alembic 0009·0010의 트리거가 UPDATE·DELETE·TRUNCATE를 막는다.

주의
    - 모든 시각은 시간대가 있어야 한다 (AwareDatetime).
    - 지표 계산(`dlp ops weekly`, `dlp ops audit-report`)은 `config/policies/ops.yaml`을 쓴다.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import AwareDatetime, Field, model_validator

from dlp_schema.common import Contract, Identifier, Ms

# 원본 버킷 접근 종류. read=내려받기, presign=서명 URL 발급, write=올리기,
# grant=검수 도구에 원본을 올려 사람에게 보여 줌 (RawAccessEvent docstring과 같다).
RawAction = Literal["read", "presign", "write", "grant"]


class RawAccessEvent(Contract):
    """원본(블러 전) 버킷 접근 하나. read=내려받기, presign=서명 URL 발급, write=올리기,
    grant=검수 도구에 원본을 올려 사람(actor)에게 보여 줌."""

    # event_id: 기록 ID (고유).
    # at: 접근 시각.
    # actor: 접근한 사람 또는 서비스 계정 (grant면 원본을 보게 되는 사람).
    # purpose: 접근한 단계·명령 (예: privacy.detect, review.create). 월간 감사 리포트의 분류 기준.
    # bucket / key: 접근한 객체 위치 (원본 버킷과 키).
    # session_id: 관련 세션. 세션과 무관한 접근이면 None.
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

    # work_id: 기록 ID.
    # task_key: 검수 작업 키 (`도구:작업 ID`). 수기 기록이면 None일 수 있다.
    # session_id: 검수한 세션.
    # reviewer: 검수자 ID.
    # stage: 검수 단계 (privacy=블러, labeling=작업 라벨, qa=선임 재검수).
    # seconds: 들인 시간(초, 0 이상).
    # video_ms: 검수한 영상 길이(ms). "영상 1시간당 검수 분" 지표의 분모.
    # source: 시간을 잰 곳 (label_studio / cvat / manual).
    # recorded_at: 기록 시각 (주간 지표의 기간 판정 기준).
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

    # audit_id: 기록 ID.
    # session_id / stream_id: 감사한 블러본의 세션·스트림.
    # duration_ms: 감사한 영상 길이(ms). "시간당 잔여 누락" 지표의 분모.
    # misses: 감사자가 찾은 블러 누락 수 (0 이상).
    # auditor: 감사자 ID. blur_reviewer와 같으면 검증 오류 (독립 감사).
    # blur_reviewer: 원래 블러를 승인한 검수자 ID.
    # audited_at: 감사 시각.
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
        """감사자가 원 블러 검수자와 다른지 확인한다 (자기 작업 감사 금지).

        Raises:
            ValueError: `auditor == blur_reviewer`일 때.
        """
        if self.auditor == self.blur_reviewer:
            raise ValueError("감사자는 원 블러 검수자와 달라야 합니다")
        return self


class RetentionDecision(Contract):
    """원본 보관 기간이 다가오거나 지난 세션에 대한 결정 (연장은 기한과 사유가 필요하다)."""

    # decision_id: 기록 ID.
    # session_id: 대상 세션.
    # decision: extend(보관 연장) / delete(원본 삭제 결정).
    # until: 연장 기한(날짜). extend일 때만 있고 delete면 None이어야 한다.
    # reason: 결정 사유 (비어 있으면 안 된다).
    # decided_by: 결정한 사람.
    # decided_at: 결정 시각. 같은 세션에 여러 결정이 있으면 가장 최근 것이 유효하다고 보는 쪽은
    #   소비자(`dlp_ops.retention`)다.
    decision_id: Identifier
    session_id: Identifier
    decision: Literal["extend", "delete"]
    until: date | None = None
    reason: str = Field(min_length=1)
    decided_by: Identifier
    decided_at: AwareDatetime

    @model_validator(mode="after")
    def _until(self) -> RetentionDecision:
        """`until`이 extend일 때만 있는지 확인한다.

        Raises:
            ValueError: extend인데 until이 없거나, delete인데 until이 있을 때.
        """
        if (self.decision == "extend") != (self.until is not None):
            raise ValueError("연장(extend)에만 기한(until)이 있어야 합니다")
        return self
