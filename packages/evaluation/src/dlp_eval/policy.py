"""평가 정책 (config/policies/evaluation.yaml)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field

from dlp_schema.common import Contract

Task = Literal["objects", "hands", "body", "contact", "actions", "relations", "states", "coverage"]
TASKS: tuple[Task, ...] = (
    "objects",
    "hands",
    "body",
    "contact",
    "actions",
    "relations",
    "states",
    "coverage",
)


class Tolerances(Contract):
    contact: int
    contact_glove: int
    boundary: int
    state: int


class GateRule(Contract):
    primary: str
    min_gain: float
    max_drop: float = Field(ge=0)
    max_drop_by: dict[str, float] = Field(
        default_factory=dict[str, float],
        description="지표별 허용 하락 (단위가 다른 지표용, 예: 오차 ms). 없으면 max_drop",
    )
    guards: tuple[str, ...]

    def allowed_drop(self, metric: str) -> float:
        return self.max_drop_by.get(metric, self.max_drop)

    first_deploy: float


class EvaluationPolicy(Contract):
    version: int
    tolerance_ms: Tolerances
    max_interp_ms: int = Field(ge=0)
    track_iou: float = Field(gt=0, le=1)
    segment_iou: tuple[float, ...]
    relation_iou: float = Field(gt=0, le=1)
    match_iou: float = Field(gt=0, le=1)
    pck_alpha: float = Field(gt=0)
    ece_bins: int = Field(ge=1)
    min_samples_per_class: int = Field(ge=1)
    subgroups: tuple[Literal["glove", "site"], ...]
    gate: dict[Task, GateRule]


def load_policy(root: Path) -> EvaluationPolicy:
    data: Any = yaml.safe_load(
        (root / "config" / "policies" / "evaluation.yaml").read_text("utf-8")
    )
    return EvaluationPolicy.model_validate(data)
