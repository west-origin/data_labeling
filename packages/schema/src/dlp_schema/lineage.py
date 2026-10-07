"""계보: 골든셋 버전, 학습 실행, 내보내기, 사용 중지 기록.

역할
    세션 → 데이터셋 버전 → 학습 실행·모델 버전·내보내기로 이어지는 계보를 추적하는 계약
    (WP7, WP13, WP15, ADR 0007·0016·0018). `dlp lineage <세션>`이 이 기록들을 모아 보여 준다.

저장 (DB 테이블, `db.repository`)
    - `GoldenSet` → `golden_sets` (`dlp dataset golden`)
    - `TrainingRun` → `training_runs`, `ModelVersion` → `model_versions` (`dlp train run|approve`)
    - `ExportRecord` → `exports` (`dlp export ...`, 올리기 전에 따로 커밋한다)
    - `Withdrawal` → `withdrawals` (`dlp dataset withdraw`)

주의
    - 골든셋 세션은 학습에 절대 쓰지 않는다. 후보 모델의 골든셋 예측은 DB에 쓰지 않는다.
    - 모델 배포는 게이트를 통과한 모델만 한다 (`ModelStatus` 전이는 `dlp_train`이 관리).
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import AwareDatetime, Field, model_validator

from dlp_schema.common import Contract, Identifier
from dlp_schema.labels import VerificationState
from dlp_schema.session import Domain


class GoldenSet(Contract):
    """도메인별 골든셋 버전. 학습에 절대 쓰지 않는 평가 기준 세션 묶음."""

    # version: 골든셋 버전 ID (최대 128자, dataset_versions.golden_set_version이 참조).
    # domain: 도메인 (골든셋은 도메인마다 따로 둔다).
    # session_ids: 골든셋에 든 세션 (1개 이상).
    # created_at: 만든 시각.
    # note: 사람이 남기는 설명 (선택).
    version: Identifier
    domain: Domain
    session_ids: tuple[Identifier, ...] = Field(min_length=1)
    created_at: AwareDatetime
    note: str = ""


# 학습 실행 하나 (데이터셋 버전 하나에서 모델 하나를 학습).
#   run_id: 실행 ID.
#   dataset_version_id: 학습 데이터를 뽑은 데이터셋 버전 (DB FK → dataset_versions).
#   model_name / model_version: 학습기 이름과 결과 모델 버전 문자열.
#   mlflow_run_id: MLflow 실행 ID. MLflow에 기록하지 않았으면 None.
#   created_at: 실행 시각.
class TrainingRun(Contract):
    run_id: Identifier
    dataset_version_id: Identifier
    model_name: str
    model_version: str
    mlflow_run_id: str | None = None
    created_at: AwareDatetime


# 재학습 모델 버전의 상태. 전이: candidate → passed(사람 승인 대기) → deployed → retired,
# 또는 candidate → rejected. `config/policies/training.yaml`의 과제별 `deploy`가 auto면
# passed를 건너뛰고 바로 deployed가 된다 (approve면 `dlp train approve`가 passed → deployed).
class ModelStatus(StrEnum):
    CANDIDATE = "candidate"  # 학습만 끝남 (평가 전)
    PASSED = "passed"  # 게이트 통과, 사람의 배포 승인 대기 (정책 deploy: approve)
    DEPLOYED = "deployed"  # 게이트 통과, 과제마다 하나
    REJECTED = "rejected"  # 게이트 실패
    RETIRED = "retired"  # 배포됐다가 새 모델로 바뀜


class ModelVersion(Contract):
    """재학습한 모델 버전 (레지스트리). 학습 실행 하나가 모델 버전 하나를 만든다."""

    # model_version: 모델 버전 ID (DB 기본 키).
    # task / run_id / trainer / artifact_uri: Field description 참고.
    # sha256: 산출물(가중치) 파일의 SHA-256 (소문자 16진 64자). 배포 때 무결성 확인에 쓴다.
    # train_examples: 학습 예제 수.
    # status: 현재 상태 (ModelStatus).
    # report_uri: 골든셋 평가·게이트 리포트 위치. 평가 전이면 None.
    # created_at: 등록 시각.
    # decided_at: 게이트 판정·은퇴 시각. candidate일 때만 None이어야 한다 (검증기).
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
        """상태와 판정 시각이 맞는지 확인한다.

        Raises:
            ValueError: candidate인데 decided_at이 있거나, candidate가 아닌데 decided_at이 없을 때.
        """
        if (self.status is ModelStatus.CANDIDATE) != (self.decided_at is None):
            raise ValueError("판정 전(candidate)에만 decided_at이 비어 있어야 합니다")
        return self


# 내보내기 이력 하나 (DB exports). 내보낸 파일을 올리기 전에 따로 커밋한다 (CLAUDE.md).
#   export_id: 내보내기 ID. 가명 HMAC 키 유도의 재료.
#   dataset_version_id: 출처 데이터셋 버전.
#   target / format / session_ids / label_states: Field description 참고.
#   uri: 내보낸 결과 위치 (datasets 버킷 등. 원본 버킷 URI는 넣지 않는다).
#   created_at: 내보낸 시각.
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


# 세션 사용 중지 기록 (동의 철회 등). DB에서 session_id가 기본 키라 세션당 한 번만 기록된다.
#   session_id: 중지한 세션.
#   reason: 사유 (비어 있으면 안 된다).
#   withdrawn_at: 중지 시각. 이후 데이터셋 버전·내보내기에서 이 세션은 빠진다 (dlp_datasets 전파).
class Withdrawal(Contract):
    session_id: Identifier
    reason: str = Field(min_length=1)
    withdrawn_at: AwareDatetime
