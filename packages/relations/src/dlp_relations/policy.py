"""관계·커버리지 정책 (config/policies/relations.yaml)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, model_validator

from dlp_schema.common import Contract, OntologyId
from dlp_schema.labels import RelationPredicate

RuleSource = Literal["hand_state", "tool_surface"]


class RelationTemplate(Contract):
    """관계 필드 틀. {필드} 자리에 입력 값을 넣는다."""

    subject: str
    subject_part: str | None = None
    predicate: RelationPredicate
    object: str
    object_part: str | None = None


class Rule(Contract):
    id: OntologyId
    source: RuleSource
    when: dict[str, tuple[str | None, ...]] = Field(
        default_factory=dict[str, tuple[str | None, ...]]
    )
    emit: RelationTemplate


class ToolSurfacePolicy(Contract):
    on_distance_m: float = Field(gt=0)
    off_distance_m: float = Field(gt=0)
    inside_margin: float = Field(ge=0)
    min_duration_ms: int = Field(ge=0)
    merge_gap_ms: int = Field(ge=0)
    max_time_gap_ms: int = Field(ge=0)
    require_grasp: bool
    confidence: float = Field(ge=0, le=1)
    corner_parts: tuple[OntologyId, OntologyId, OntologyId, OntologyId]

    @model_validator(mode="after")
    def _check(self) -> ToolSurfacePolicy:
        if self.off_distance_m < self.on_distance_m:
            raise ValueError("off_distance_m은 on_distance_m 이상이어야 합니다")
        return self


class CoveragePolicy(Contract):
    grid_m: float = Field(gt=0)
    default_footprint_m: float = Field(gt=0)
    footprint_m: dict[OntologyId, float]

    def footprint(self, tool_class: str | None) -> float:
        return self.footprint_m.get(tool_class or "", self.default_footprint_m)


class RelationsPolicy(Contract):
    version: int
    rules: tuple[Rule, ...] = Field(min_length=1)
    merge_gap_ms: int = Field(ge=0)
    tool_surface: ToolSurfacePolicy
    coverage: CoveragePolicy

    @model_validator(mode="after")
    def _unique(self) -> RelationsPolicy:
        ids = [r.id for r in self.rules]
        if len(ids) != len(set(ids)):
            raise ValueError("규칙 ID가 겹칩니다")
        return self

    @property
    def digest(self) -> str:
        """정책 내용의 해시. 관계 레코드의 model_version에 넣어 규칙 변경을 추적한다."""
        content = json.dumps(self.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(content.encode()).hexdigest()[:12]


def load_policy(root: Path) -> RelationsPolicy:
    data: Any = yaml.safe_load((root / "config" / "policies" / "relations.yaml").read_text("utf-8"))
    return RelationsPolicy.model_validate(data)
