"""계보: 골든셋 버전, 학습 실행, 내보내기, 사용 중지 기록."""

from __future__ import annotations

from pydantic import AwareDatetime, Field

from dlp_schema.common import Contract, Identifier
from dlp_schema.session import Domain


class GoldenSet(Contract):
    """도메인별 골든셋 버전. 학습에 절대 쓰지 않는 평가 기준 세션 묶음."""

    version: Identifier
    domain: Domain
    session_ids: tuple[Identifier, ...] = Field(min_length=1)
    created_at: AwareDatetime
    note: str = ""


class TrainingRun(Contract):
    run_id: Identifier
    dataset_version_id: Identifier
    model_name: str
    model_version: str
    mlflow_run_id: str | None = None
    created_at: AwareDatetime


class ExportRecord(Contract):
    export_id: Identifier
    dataset_version_id: Identifier
    target: str = Field(description="내보낸 곳 (구매자, 내부 학습 등)")
    format: str = Field(description="coco, interval_json, lerobot 등")
    uri: str
    session_ids: tuple[Identifier, ...] = Field(
        description="실제로 내보낸 세션 (사용 중지 제외 후)"
    )
    created_at: AwareDatetime


class Withdrawal(Contract):
    session_id: Identifier
    reason: str = Field(min_length=1)
    withdrawn_at: AwareDatetime
