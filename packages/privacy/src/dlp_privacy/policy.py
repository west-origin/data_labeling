"""프라이버시 정책 (config/policies/privacy.yaml + config/defaults.yaml privacy)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field

from dlp_schema.common import Contract, OntologyId
from dlp_schema.config import PrivacyConfig, load_config

ReviewReason = Literal["no_detector", "track_gap", "disagreement", "low_confidence", "reflection"]


class TargetPolicy(Contract):
    margin: float = Field(ge=0)
    detectors: tuple[str, ...]


class TrackerPolicy(Contract):
    iou_match: float = Field(gt=0, le=1)
    max_gap_ms: float = Field(ge=0)


class RenderPolicy(Contract):
    min_block_px: int = Field(ge=1)
    blocks_per_box: int = Field(ge=1)


class DetectorSpec(Contract):
    kind: Literal["yunet", "opencv_codes", "reflection", "open_vocab", "unavailable", "oracle"]
    model: str | None = Field(default=None, description="config/models.yaml 이름")
    tokenizer: str | None = None
    region_detector: str | None = None
    face_detector: str | None = None
    threshold_scale: float | None = None
    # open_vocab 전용
    frame_stride_ms: int = Field(default=0, ge=0)
    score_threshold: float | None = Field(default=None, ge=0, le=1)
    score_full: float | None = Field(default=None, gt=0, le=1)
    queries: dict[str, str] = Field(default_factory=dict[str, str])


class PrivacyPolicy(Contract):
    version: int
    targets: dict[OntologyId, TargetPolicy]
    detection_threshold: float = Field(ge=0, le=1)
    review_score: float = Field(ge=0, le=1)
    tracker: TrackerPolicy
    render: RenderPolicy
    review_priority: tuple[ReviewReason, ...]
    detectors: dict[str, DetectorSpec]
    # config/defaults.yaml에서 채운다
    platform: PrivacyConfig


def load_policy(root: Path) -> PrivacyPolicy:
    """저장소 루트에서 privacy.yaml과 defaults.yaml을 함께 읽는다."""
    data: Any = yaml.safe_load((root / "config" / "policies" / "privacy.yaml").read_text("utf-8"))
    data["platform"] = load_config(root / "config" / "defaults.yaml").privacy.model_dump()
    return PrivacyPolicy.model_validate(data)
