"""프리라벨 정책 (config/policies/prelabel.yaml)."""

from __future__ import annotations

import hashlib
import json
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
    detector_score: float = Field(ge=0, le=1)
    min_score: float = Field(ge=0, le=1)
    keypoint_score: float = Field(ge=0, le=1)
    max_people: int = Field(ge=1)
    track_iou: float = Field(gt=0, le=1)
    max_gap_ms: float = Field(ge=0)


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
    nms_iou: float = Field(gt=0, le=1)
    queries: dict[str, OntologyId] = Field(min_length=1)


class DepthPolicy(Contract):
    frame_stride_ms: int = Field(ge=0)
    default_hfov_deg: float = Field(gt=0, lt=180)
    patch_radius_px: int = Field(ge=0)
    min_depth_m: float = Field(gt=0)
    max_depth_m: float = Field(gt=0)
    hand_points: tuple[int, ...] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class GloveContactPolicy(Contract):
    on_threshold: float
    off_threshold: float
    min_duration_ms: float


class VideoContactPolicy(Contract):
    max_distance_px: float
    min_duration_ms: float
    merge_gap_ms: float
    box_max_gap_ms: float = Field(ge=0)


class ContactConfidence(Contract):
    """접촉 구간 출처별 신뢰도. 장갑과 영상이 모두 접촉이면 fused."""

    fused: float = Field(ge=0, le=1)
    glove: float = Field(ge=0, le=1)
    video: float = Field(ge=0, le=1)


class ContactPolicy(Contract):
    glove: GloveContactPolicy
    video: VideoContactPolicy
    confidence: ContactConfidence
    unresolved_target_id: str = Field(min_length=1)


class WearerPolicy(Contract):
    rate_hz: float = Field(gt=0)
    min_correlation: float = Field(ge=-1, le=1)
    min_overlap_samples: int = Field(ge=2)


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

    def digest(self, *sections: str) -> str:
        """정책 절들의 짧은 해시. 모델 버전에 붙여 정책 값이 바뀌면 다시 돌게 한다."""
        data = {name: getattr(self, name).model_dump(mode="json") for name in sections}
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:8]


def load_policy(root: Path) -> PrelabelPolicy:
    data: Any = yaml.safe_load((root / "config" / "policies" / "prelabel.yaml").read_text("utf-8"))
    return PrelabelPolicy.model_validate(data)
