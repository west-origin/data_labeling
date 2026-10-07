"""재학습 정책 (config/policies/training.yaml)."""

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
    experiment_prefix: str
    registered_model_prefix: str
    deployed_alias: str


class TaskTraining(Contract):
    stage: Literal["prelabel", "privacy", "actions"]
    replaces: tuple[str, ...] = ()
    min_examples: int = Field(ge=1)
    min_new_examples: int = Field(ge=0)
    deploy: Literal["auto", "approve"]
    trainer: str
    params: dict[str, Any] = Field(default_factory=dict[str, Any])


class TrainingPolicy(Contract):
    version: int
    splits: tuple[Split, ...] = Field(min_length=1)
    trainable_states: tuple[VerificationState, ...] = Field(min_length=1)
    artifact_prefix: str
    mlflow: MlflowPolicy
    tasks: dict[Task, TaskTraining]

    @field_validator("splits")
    @classmethod
    def _no_golden(cls, v: tuple[Split, ...]) -> tuple[Split, ...]:
        bad = {Split.GOLDEN, Split.HOLDOUT} & set(v)
        if bad:
            raise ValueError(f"골든·holdout 분할은 학습에 쓸 수 없습니다: {sorted(bad)}")
        return v

    @field_validator("tasks")
    @classmethod
    def _privacy_needs_approval(cls, v: dict[Task, TaskTraining]) -> dict[Task, TaskTraining]:
        if "privacy" in v and v["privacy"].deploy != "approve":
            raise ValueError("블러(privacy) 모델은 사람 승인 후에만 배포한다 (deploy: approve)")
        return v

    @field_validator("trainable_states")
    @classmethod
    def _reviewed(cls, v: tuple[VerificationState, ...]) -> tuple[VerificationState, ...]:
        if VerificationState.UNREVIEWED in v:
            raise ValueError("미검수 라벨은 학습 예제가 될 수 없습니다")
        return v


def load_policy(root: Path) -> TrainingPolicy:
    data: Any = yaml.safe_load((root / "config" / "policies" / "training.yaml").read_text("utf-8"))
    return TrainingPolicy.model_validate(data)
