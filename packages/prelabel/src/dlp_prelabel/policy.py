"""프리라벨 정책 (config/policies/prelabel.yaml)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field

from dlp_schema.common import Contract, OntologyId


class ModelRoles(Contract):
    """역할 → config/models.yaml 이름."""

    hands: str
    body_detector: str
    body: str
    coco_objects: str
    open_vocab: str
    open_vocab_tokenizer: str
    depth: str


class HandsPolicy(Contract):
    min_score: float = Field(ge=0, le=1)
    input_is_mirrored: bool


class BodyPolicy(Contract):
    min_score: float = Field(ge=0, le=1)
    keypoint_score: float = Field(ge=0, le=1)
    max_people: int = Field(ge=1)
    track_iou: float = Field(gt=0, le=1)


class ObjectsPolicy(Contract):
    min_score: float = Field(ge=0, le=1)
    track_iou: float = Field(gt=0, le=1)
    max_gap_ms: float = Field(ge=0)
    coco_to_ontology: dict[str, OntologyId]


class OpenVocabObjectsPolicy(Contract):
    frame_stride_ms: int = Field(ge=0)
    min_score: float = Field(ge=0, le=1)
    track_iou: float = Field(gt=0, le=1)
    max_gap_ms: float = Field(ge=0)
    queries: dict[str, OntologyId] = Field(min_length=1)


class DepthPolicy(Contract):
    frame_stride_ms: int = Field(ge=0)
    default_hfov_deg: float = Field(gt=0, lt=180)
    patch_radius_px: int = Field(ge=0)
    min_depth_m: float = Field(gt=0)
    max_depth_m: float = Field(gt=0)
    hand_points: tuple[int, ...] = Field(min_length=1)


class GloveContactPolicy(Contract):
    on_threshold: float
    off_threshold: float
    min_duration_ms: float


class VideoContactPolicy(Contract):
    max_distance_px: float
    min_duration_ms: float
    merge_gap_ms: float


class ContactPolicy(Contract):
    glove: GloveContactPolicy
    video: VideoContactPolicy


class WearerPolicy(Contract):
    rate_hz: float = Field(gt=0)
    min_correlation: float = Field(ge=-1, le=1)


class PrelabelPolicy(Contract):
    version: int
    models: ModelRoles
    hands: HandsPolicy
    body: BodyPolicy
    objects: ObjectsPolicy
    open_vocab_objects: OpenVocabObjectsPolicy
    contact: ContactPolicy
    wearer_matching: WearerPolicy
    depth: DepthPolicy


def load_policy(root: Path) -> PrelabelPolicy:
    data: Any = yaml.safe_load((root / "config" / "policies" / "prelabel.yaml").read_text("utf-8"))
    return PrelabelPolicy.model_validate(data)
