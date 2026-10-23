"""재학습 정책 (config/policies/training.yaml).

WP13, ADR 0016. `load_policy(root)`가 YAML을 읽어 `TrainingPolicy`로 검증한다. 검증기가 CLAUDE.md의
규칙 세 가지를 코드로 막는다:
- 학습 분할에 골든·holdout을 넣을 수 없다 (`_no_golden`).
- 블러(privacy) 과제는 `deploy: approve`여야 한다 (`_privacy_needs_approval`, 사람 승인 후 배포).
- 미검수(unreviewed) 상태는 학습 예제가 될 수 없다 (`_reviewed`).

과제 키는 평가 과제(`dlp_eval.policy.Task`)와 같다. 이 정책은 모델 버전 해시에 들어가지 않는다
(모델 버전 태그는 `loop._short`가 과제·데이터셋 버전·학습기·파라미터·시각으로 만든다).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, field_validator

from dlp_eval.policy import Task
from dlp_schema.common import Contract
from dlp_schema.dataset import Split
from dlp_schema.labels import VerificationState


class MlflowPolicy(Contract):
    """MLflow 이름 규칙 (`mlflow` 절)."""

    experiment_prefix: str  # 실험 이름 = 접두 + 과제
    registered_model_prefix: str  # MLflow 등록 모델 이름 = 접두 + 과제
    deployed_alias: str  # 배포 버전에 붙이는 별칭


class TaskTraining(Contract):
    """과제 하나의 학습 작업 템플릿 (`tasks.<과제>`)."""

    # 배포 모델 연결 단계: prelabel(프리라벨 Predictor), privacy(블러 탐지기).
    # actions는 예약 값이며 현재 행동 실행 경로에서는 배포 모델을 읽지 않는다.
    stage: Literal["prelabel", "privacy", "actions"]
    # 배포되면 빼는 기본 어댑터 이름 (Predictor.name). 비어 있으면 기본 어댑터에 더한다
    replaces: tuple[str, ...] = ()
    min_examples: int = Field(ge=1)  # 학습을 시작할 최소 예제 수
    min_new_examples: int = Field(ge=0)  # 직전 학습(모든 상태) 대비 새로 늘어야 할 예제 수
    deploy: Literal["auto", "approve"]  # auto: 게이트 통과 시 바로 배포, approve: 사람 승인 대기
    trainer: str  # 기본 학습기 이름 (`trainers.TRAINERS`·`LOADERS`의 키)
    params: dict[str, Any] = Field(default_factory=dict[str, Any])  # 기본 하이퍼파라미터


class GeometricAugment(Contract):
    """기하 증강 (`augmentation.geometric`). hflip_p: 좌우 뒤집기 확률. rotate_deg: 회전 최대
    각(도).
    scale_range: 확대 배율 범위 (1보다 크면 자르기). perspective_frac: 모서리 흔들기 최대 비율."""

    hflip_p: float = Field(ge=0, le=1)
    rotate_deg: float = Field(ge=0, le=45)
    scale_range: tuple[float, float]
    perspective_frac: float = Field(ge=0, le=0.2)


class PhotometricAugment(Contract):
    """광학 증강 (`augmentation.photometric`). brightness·contrast: ±최대 변화 (0~1).
    blur_sigma_max: 가우스 흐림 최대 sigma(px). noise_std_max: 잡음 최대 표준편차(0~255).
    jpeg_p / jpeg_quality_range: JPEG 손상 확률과 품질 범위."""

    brightness: float = Field(ge=0, le=1)
    contrast: float = Field(ge=0, le=1)
    blur_sigma_max: float = Field(ge=0)
    noise_std_max: float = Field(ge=0)
    jpeg_p: float = Field(ge=0, le=1)
    jpeg_quality_range: tuple[int, int]


class EgocentricAugment(Contract):
    """1인칭 증강 (`augmentation.egocentric`). motion_blur_px_max: 움직임 흐림 최대 길이(px).
    highlight_p / highlights_max: 반사광(젖은 손·스테인리스) 확률과 최대 개수. distortion_k1_max:
    렌즈 방사 왜곡 k1 최대 크기 (±)."""

    motion_blur_px_max: int = Field(ge=0)
    highlight_p: float = Field(ge=0, le=1)
    highlights_max: int = Field(ge=0)
    distortion_k1_max: float = Field(ge=0, le=0.5)


class AugmentationPolicy(Contract):
    """학습 증강 (`augmentation` 절, ADR 0057, `dlp_train.augment`).

    enabled: 끄면 학습 파라미터에 넣지 않는다. tasks: 증강을 쓰는 과제. seed: 표본 씨앗의 바탕.
    copies: 표본마다 증강본 수. 해시(`augment.policy_digest`)가 모델 버전·MLflow에 들어간다.
    """

    enabled: bool
    tasks: tuple[Task, ...] = ()
    seed: int = Field(ge=0)
    copies: int = Field(ge=1)
    geometric: GeometricAugment
    photometric: PhotometricAugment
    egocentric: EgocentricAugment


class TrainingPolicy(Contract):
    """`config/policies/training.yaml` 전체. 각 키의 의미는 YAML 주석을 본다."""

    version: int  # 정책 형식 버전
    splits: tuple[Split, ...] = Field(min_length=1)  # 학습 예제를 뽑는 분할 (train·val)
    # 학습 예제가 되는 모델 라벨의 검증 상태 (사람 라벨은 상태와 무관하게 예제가 된다)
    trainable_states: tuple[VerificationState, ...] = Field(min_length=1)
    artifact_prefix: str  # 학습 산출물 버킷(buckets.mlflow) 안의 키 접두
    mlflow: MlflowPolicy
    tasks: dict[Task, TaskTraining]
    augmentation: AugmentationPolicy  # 학습 증강 (ADR 0057)

    @field_validator("splits")
    @classmethod
    def _no_golden(cls, v: tuple[Split, ...]) -> tuple[Split, ...]:
        """골든·holdout 분할이 있으면 ValueError (골든셋 평가의 독립성을 지킨다)."""
        bad = {Split.GOLDEN, Split.HOLDOUT} & set(v)
        if bad:
            raise ValueError(f"골든·holdout 분할은 학습에 쓸 수 없습니다: {sorted(bad)}")
        return v

    @field_validator("tasks")
    @classmethod
    def _privacy_needs_approval(cls, v: dict[Task, TaskTraining]) -> dict[Task, TaskTraining]:
        """privacy 과제가 `deploy: approve`가 아니면 ValueError (블러는 사람 승인 후에만 배포)."""
        if "privacy" in v and v["privacy"].deploy != "approve":
            raise ValueError("블러(privacy) 모델은 사람 승인 후에만 배포한다 (deploy: approve)")
        return v

    @field_validator("trainable_states")
    @classmethod
    def _reviewed(cls, v: tuple[VerificationState, ...]) -> tuple[VerificationState, ...]:
        """unreviewed가 있으면 ValueError (미검수 모델 라벨로 학습하면 모델이 자기 오류를
        배운다)."""
        if VerificationState.UNREVIEWED in v:
            raise ValueError("미검수 라벨은 학습 예제가 될 수 없습니다")
        return v


def load_policy(root: Path) -> TrainingPolicy:
    """저장소 루트 `root`의 `config/policies/training.yaml`을 읽어 검증한다.

    Raises:
        FileNotFoundError: 파일이 없을 때.
        pydantic.ValidationError: 형식이 틀리거나 위 검증기 규칙을 어길 때.
    """
    data: Any = yaml.safe_load((root / "config" / "policies" / "training.yaml").read_text("utf-8"))
    return TrainingPolicy.model_validate(data)
