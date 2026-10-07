"""행동 구간 정책 (config/policies/actions.yaml)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field

from dlp_schema.common import Contract


class BoundaryPolicy(Contract):
    smooth_ms: int = Field(ge=0)
    still_speed: float = Field(gt=0)
    min_still_ms: int = Field(ge=0)
    valley_speed: float = Field(gt=0)
    valley_ratio: float = Field(gt=0, le=1)
    valley_window_ms: int = Field(ge=0)
    min_valley_separation_ms: int = Field(ge=0)
    merge_ms: int = Field(ge=0)
    min_segment_ms: int = Field(ge=0)


class ToleranceMs(Contract):
    approach: int
    contact_glove: int
    contact_video: int
    end: int


class VlmPolicy(Contract):
    max_retries: int = Field(ge=0)
    frames_per_segment: int = Field(ge=1)
    max_side_px: int = Field(ge=64)
    default_confidence: float = Field(ge=0, le=1)
    model: str
    timeout_s: float = Field(gt=0)


class ActionsPolicy(Contract):
    version: int
    # 대상을 모르는 접촉의 target_id (prelabel.yaml contact.unresolved_target_id).
    # 대상 후보에서 뺀다
    unresolved_entity_ids: tuple[str, ...] = ()
    boundaries: BoundaryPolicy
    tolerance_ms: ToleranceMs
    vlm: VlmPolicy

    @property
    def digest(self) -> str:
        content = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:12]


def load_policy(root: Path) -> ActionsPolicy:
    data: Any = yaml.safe_load((root / "config" / "policies" / "actions.yaml").read_text("utf-8"))
    return ActionsPolicy.model_validate(data)
