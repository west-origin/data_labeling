"""관계·커버리지 정책 (config/policies/relations.yaml).

`RelationsPolicy.digest`는 파싱·검증된 값(`model_dump`)의 해시라 YAML 주석·키 순서는 영향이 없다.
관계·커버리지 레코드의 model_version이 `relations-<digest>`다.
"""

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
    """관계 필드 틀. {필드} 자리에 입력 값을 넣는다.

    subject / object: 주어·목적어 개체 ID 틀 (예: "{hand}_hand", "{target_id}"). 필수.
    subject_part / object_part: 부위 틀 (없으면 None). predicate: 관계 술어 (계약
    `RelationPredicate`).
    """

    subject: str
    subject_part: str | None = None
    predicate: RelationPredicate
    object: str
    object_part: str | None = None


class Rule(Contract):
    """규칙 하나.

    id: 규칙 ID (관계의 derived_by로 남는다, 정책 안에서 유일).
    source: 입력 종류 "hand_state"(손 상태 구간) | "tool_surface"(작용부-표면 접촉 구간).
    when: 입력 필드 → 허용 값 (None은 값 없음). 모든 조건이 맞아야 한다. 비면 항상 맞다.
    emit: 만들 관계 틀.
    """

    id: OntologyId
    source: RuleSource
    when: dict[str, tuple[str | None, ...]] = Field(
        default_factory=dict[str, tuple[str | None, ...]]
    )
    emit: RelationTemplate


class ToolSurfacePolicy(Contract):
    """도구-표면 접촉 정책 (`tool_surface` 절).

    on_distance_m / off_distance_m: 평면까지 거리 히스테리시스 (m, off ≥ on 검증).
    inside_margin: 표면 사각형 밖으로 변 길이 비율만큼 여유. min_duration_ms / merge_gap_ms: 최소
    길이, 병합 간격. max_time_gap_ms: 작용부 시각과 표면 꼭짓점 샘플의 최대 시각 차 (보간 허용).
    require_grasp: 손이 도구를 쥔 구간(hand_state, tool, target_id=도구)에서만 접촉으로 본다.
    confidence: 접촉 관계·커버리지 신뢰도. corner_parts: 표면 꼭짓점 부위 이름 4개 (0 원점, 1
    가로 끝, 2 대각, 3 세로 끝).
    """

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
        """off_distance_m < on_distance_m이면 ValueError (히스테리시스가 뒤집힌다)."""
        if self.off_distance_m < self.on_distance_m:
            raise ValueError("off_distance_m은 on_distance_m 이상이어야 합니다")
        return self


class CoveragePolicy(Contract):
    """커버리지 정책 (`coverage` 절).

    grid_m: 격자 칸 크기(m). default_footprint_m: 작용부 원 반지름 기본값(m).
    footprint_m: 도구 클래스 → 반지름(m).
    """

    grid_m: float = Field(gt=0)
    default_footprint_m: float = Field(gt=0)
    footprint_m: dict[OntologyId, float]

    def footprint(self, tool_class: str | None) -> float:
        """도구 클래스의 작용부 반지름(m). 목록에 없거나 None이면 기본값."""
        return self.footprint_m.get(tool_class or "", self.default_footprint_m)


class RelationsPolicy(Contract):
    """`config/policies/relations.yaml` 전체.

    version: 형식 버전. rules: 규칙 목록 (최소 1개, ID 유일). merge_gap_ms: 같은 관계 병합 간격.
    unresolved_target_ids: 대상을 모르는 접촉 표시 ID (규칙 입력에서 None으로 본다).
    """

    version: int
    rules: tuple[Rule, ...] = Field(min_length=1)
    merge_gap_ms: int = Field(ge=0)
    # 대상을 모르는 접촉의 target_id (prelabel.yaml contact.unresolved_target_id).
    # 규칙 입력에서는 값이 없는 것으로 본다
    unresolved_target_ids: tuple[str, ...] = ()
    tool_surface: ToolSurfacePolicy
    coverage: CoveragePolicy

    @model_validator(mode="after")
    def _unique(self) -> RelationsPolicy:
        """규칙 ID가 겹치면 ValueError."""
        ids = [r.id for r in self.rules]
        if len(ids) != len(set(ids)):
            raise ValueError("규칙 ID가 겹칩니다")
        return self

    @property
    def digest(self) -> str:
        """정책 내용의 해시. 관계 레코드의 model_version에 넣어 규칙 변경을 추적한다.

        정책 전체(모든 절)의 model_dump를 키 정렬 JSON으로 만든 sha256 앞 12자. 파싱된 값만
        보므로 YAML 주석은 해시를 바꾸지 않는다. 내용이 같은 관계는 정책이 바뀌어도 다시 넣지
        않는다 (`runner`).
        """
        content = json.dumps(self.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(content.encode()).hexdigest()[:12]


def load_policy(root: Path) -> RelationsPolicy:
    """`<root>/config/policies/relations.yaml`을 읽어 검증한다 (형식이 틀리면 ValidationError)."""
    data: Any = yaml.safe_load((root / "config" / "policies" / "relations.yaml").read_text("utf-8"))
    return RelationsPolicy.model_validate(data)
