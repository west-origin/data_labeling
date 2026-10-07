"""액티브 러닝 정책 (config/policies/active.yaml)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field

from dlp_schema.common import Contract
from dlp_schema.session import LifecycleState


class CorrectionPolicy(Contract):
    count_added: bool
    prior_weight: float = Field(ge=0)


class CandidatePolicy(Contract):
    lifecycle: tuple[LifecycleState, ...] = Field(min_length=1)
    exclude_golden: bool


class ScorePolicy(Contract):
    normalize: Literal["total", "per_minute"]
    terms: dict[str, float] = Field(min_length=1)


class FiftyOnePolicy(Contract):
    dataset_prefix: str
    frame_tolerance_ms: int = Field(ge=0)


class ActivePolicy(Contract):
    version: int
    correction: CorrectionPolicy
    candidates: CandidatePolicy
    excluded_kinds: tuple[str, ...]
    score: ScorePolicy
    select: int = Field(ge=1)
    fiftyone: FiftyOnePolicy


def load_policy(root: Path) -> ActivePolicy:
    data: Any = yaml.safe_load((root / "config" / "policies" / "active.yaml").read_text("utf-8"))
    return ActivePolicy.model_validate(data)
