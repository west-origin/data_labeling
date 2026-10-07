"""검수 운영 정책 (config/policies/review.yaml + config/defaults.yaml review 비율)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, model_validator

from dlp_schema.common import Contract
from dlp_schema.config import ReviewConfig, load_config

ErrorType = Literal["boundary_shift", "class_swap", "blur_deletion"]


CVAT_KINDS = ("box_track", "keypoint_track")


class UnitsPolicy(Contract):
    spatial: tuple[str, ...]
    temporal: tuple[str, ...]

    @model_validator(mode="after")
    def _shown_by_tools(self) -> UnitsPolicy:
        from dlp_review.labelstudio import LS_KINDS

        extra = (set(self.spatial) - set(CVAT_KINDS)) | (set(self.temporal) - set(LS_KINDS))
        if extra:
            raise ValueError(f"검수 도구가 보여 주지 않는 라벨 종류: {sorted(extra)}")
        return self


class PriorityPolicy(Contract):
    low_confidence: float = Field(ge=0, le=1)
    disagreement_iou: float = Field(gt=0, le=1)
    disagreement_overlap: float = Field(gt=0, le=1)
    contact_mismatch_max_confidence: float = Field(ge=0, le=1)
    weights: dict[str, float]
    routine_priority: float = Field(ge=0)


class SamplingPolicy(Contract):
    high_confidence: float = Field(ge=0, le=1)
    ratio: float = Field(gt=0, le=1)
    min_sample: int = Field(ge=1)
    max_defect_ratio: float = Field(ge=0, le=1)


class SeedingPolicy(Contract):
    errors_per_task: int = Field(ge=1)
    types: tuple[ErrorType, ...] = Field(min_length=1)
    boundary_shift_ms: tuple[int, int]
    detect_tolerance_ms: int = Field(ge=0)
    blur_overlap: float = Field(gt=0, le=1)


class MeasurementPolicy(Contract):
    tolerance_ms: int = Field(ge=0)
    match_iou: float = Field(gt=0, le=1)


class ReviewersPolicy(Contract):
    senior: tuple[str, ...] = ()
    privacy: tuple[str, ...] = Field(default=(), description="원본 접근 권한자 (블러 검수)")


class ReviewOpsPolicy(Contract):
    version: int
    units: UnitsPolicy
    priority: PriorityPolicy
    sampling: SamplingPolicy
    measurement: MeasurementPolicy
    seeding: SeedingPolicy
    reviewers: ReviewersPolicy
    ratios: ReviewConfig  # config/defaults.yaml review


def load_policy(root: Path) -> ReviewOpsPolicy:
    data: Any = yaml.safe_load((root / "config" / "policies" / "review.yaml").read_text("utf-8"))
    data["ratios"] = load_config(root / "config" / "defaults.yaml").review.model_dump()
    return ReviewOpsPolicy.model_validate(data)
