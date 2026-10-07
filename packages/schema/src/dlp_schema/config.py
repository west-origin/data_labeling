"""config/defaults.yaml 로더. 코드는 정책 값을 하드코딩하지 않고 이 설정을 읽는다."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field

from dlp_schema.common import Contract, OntologyId


class InfrastructureConfig(Contract):
    object_store: Literal["seaweedfs", "minio"]
    data_versioning: Literal["lakefs", "dvc"]
    annotation_tools: Literal["hybrid", "cvat_only"]
    orchestrator: Literal["prefect", "airflow"]
    experiment_tracking: Literal["mlflow"]


class BucketsConfig(Contract):
    raw: str
    labeling: str
    datasets: str
    mlflow: str


class ProxyConfig(Contract):
    """검수 화면용 프록시 영상 인코딩 설정."""

    max_height: int = Field(gt=0)
    crf: int = Field(ge=0, le=51)
    keyframe_ms: int = Field(gt=0)


class MediaConfig(Contract):
    proxy: ProxyConfig


class FullReviewExit(Contract):
    weeks_below_target: int = Field(gt=0)
    audit_sample_ratio: float = Field(gt=0, le=1)


class PrivacyConfig(Contract):
    blur_hold_ms: int = Field(ge=0)
    render_mode: Literal["mosaic", "solid"]
    strip_audio_in_release: bool
    extra_targets_v1: tuple[OntologyId, ...]
    full_review_exit: FullReviewExit


class ReviewConfig(Contract):
    qa_sample_ratio: float = Field(ge=0, le=1)
    double_annotation_ratio: float = Field(ge=0, le=1)
    blind_task_ratio: float = Field(ge=0, le=1)
    seeded_error_task_ratio: float = Field(ge=0, le=1)


class GoldenSetConfig(Contract):
    min_instances_per_class: int = Field(gt=0)
    split_unit: tuple[Literal["worker_id", "site_id"], ...] = Field(min_length=1)


class ExportConfig(Contract):
    include_unreviewed: bool


class RetentionConfig(Contract):
    raw_retention_days: int | None = Field(default=None, gt=0)


class SuccessCriteria(Contract):
    review_minutes_per_video_hour: float | None = None
    residual_blur_miss_per_hour_max: float | None = None
    boundary_agreement_min: float | None = None


class PlatformConfig(Contract):
    version: int
    infrastructure: InfrastructureConfig
    buckets: BucketsConfig
    media: MediaConfig
    privacy: PrivacyConfig
    review: ReviewConfig
    golden_set: GoldenSetConfig
    export: ExportConfig
    retention: RetentionConfig
    success_criteria: SuccessCriteria


def load_config(path: Path) -> PlatformConfig:
    data: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    return PlatformConfig.model_validate(data)


def repo_root(start: Path | None = None) -> Path:
    """config/defaults.yaml이 있는 가장 가까운 상위 디렉터리."""
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "config" / "defaults.yaml").is_file():
            return candidate
    raise FileNotFoundError("config/defaults.yaml을 찾을 수 없습니다")
