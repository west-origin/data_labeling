"""계보: 골든셋 버전, 학습 실행, 내보내기, 사용 중지 기록."""

from __future__ import annotations

from enum import StrEnum

from pydantic import AwareDatetime, Field, model_validator

from dlp_schema.common import Contract, Identifier
from dlp_schema.labels import VerificationState
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


class ModelStatus(StrEnum):
    CANDIDATE = "candidate"  # 학습만 끝남 (평가 전)
    PASSED = "passed"  # 게이트 통과, 사람의 배포 승인 대기 (정책 deploy: approve)
    DEPLOYED = "deployed"  # 게이트 통과, 과제마다 하나
    REJECTED = "rejected"  # 게이트 실패
    RETIRED = "retired"  # 배포됐다가 새 모델로 바뀜


class ModelVersion(Contract):
    """재학습한 모델 버전 (레지스트리). 학습 실행 하나가 모델 버전 하나를 만든다."""

    model_version: Identifier
    task: Identifier = Field(
        description="평가 과제 (objects, hands, body, contact, actions, privacy 등)"
    )
    run_id: Identifier = Field(description="학습 실행 (training_runs)")
    trainer: Identifier = Field(description="학습기 이름. 배포할 때 같은 이름의 로더로 읽는다")
    artifact_uri: str = Field(description="가중치·설정 파일 위치 (학습 산출물 버킷)")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    train_examples: int = Field(ge=0, description="학습에 쓴 예제 수 (누적 판단에 쓴다)")
    status: ModelStatus
    report_uri: str | None = Field(default=None, description="골든셋 평가·게이트 리포트")
    created_at: AwareDatetime
    decided_at: AwareDatetime | None = Field(default=None, description="게이트 판정 또는 은퇴 시각")

    @model_validator(mode="after")
    def _check(self) -> ModelVersion:
        if (self.status is ModelStatus.CANDIDATE) != (self.decided_at is None):
            raise ValueError("판정 전(candidate)에만 decided_at이 비어 있어야 합니다")
        return self


class ExportRecord(Contract):
    export_id: Identifier
    dataset_version_id: Identifier
    target: str = Field(description="내보낸 곳 (구매자, 내부 학습 등)")
    format: str = Field(description="coco, interval_json, lerobot 등")
    uri: str
    session_ids: tuple[Identifier, ...] = Field(
        description="실제로 내보낸 세션 (사용 중지 제외 후)"
    )
    label_states: tuple[VerificationState, ...] = Field(
        default=(),
        description="검증 정책: 내보낸 라벨의 검증 상태 (사람이 만든 라벨은 늘 포함). "
        "unreviewed가 있으면 미검수 포함 옵션으로 내보낸 것",
    )
    created_at: AwareDatetime


class Withdrawal(Contract):
    session_id: Identifier
    reason: str = Field(min_length=1)
    withdrawn_at: AwareDatetime
