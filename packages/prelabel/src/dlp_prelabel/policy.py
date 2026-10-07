"""프리라벨 정책 (config/policies/prelabel.yaml)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field

from dlp_schema.common import Contract, OntologyId


class ModelSpec(Contract):
    path: str
    url: str
    sha256: str


class HandsPolicy(Contract):
    min_score: float = Field(ge=0, le=1)
    input_is_mirrored: bool


class BodyPolicy(Contract):
    min_score: float = Field(ge=0, le=1)
    max_people: int = Field(ge=1)
    track_iou: float = Field(gt=0, le=1)


class ObjectsPolicy(Contract):
    min_score: float = Field(ge=0, le=1)
    track_iou: float = Field(gt=0, le=1)
    max_gap_ms: float = Field(ge=0)
    coco_to_ontology: dict[str, OntologyId]


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
    models: dict[str, ModelSpec]
    hands: HandsPolicy
    body: BodyPolicy
    objects: ObjectsPolicy
    contact: ContactPolicy
    wearer_matching: WearerPolicy


def load_policy(root: Path) -> PrelabelPolicy:
    data: Any = yaml.safe_load((root / "config" / "policies" / "prelabel.yaml").read_text("utf-8"))
    return PrelabelPolicy.model_validate(data)
