"""온톨로지 사전과 로더. 온톨로지는 config/ontology/<버전>/*.yaml에 둔다."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, model_validator

from dlp_schema.common import Contract, OntologyId, SemVer


class Term(Contract):
    ko: str
    description: str | None = None


class DomainTerm(Term):
    phase: str


class TaskTerm(Term):
    domain: OntologyId
    substeps: dict[OntologyId, Term] = Field(default_factory=dict)


class VerbLevel(StrEnum):
    PRIMITIVE = "primitive"
    SKILL = "skill"
    HUMAN_SKILL = "human_skill"


class VerbTerm(Term):
    level: VerbLevel
    contactless: bool = False


class StateAttribute(Term):
    values: tuple[OntologyId, ...] = Field(min_length=2)


class ToolSpec(Contract):
    working_parts: tuple[OntologyId, ...] = Field(min_length=1)
    grip_parts: tuple[OntologyId, ...] = Field(min_length=1)


class ObjectClass(Term):
    physical_type: OntologyId
    states: tuple[OntologyId, ...] = ()
    tool: ToolSpec | None = None
    surface: bool = False


class EventTerm(Term):
    category: Literal["failure_retry", "worker_safety", "subject_safety", "infection_control"]
    form: Literal["point", "interval"]
    severity: bool = False
    requires_action: bool = False
    requires_object: bool = False
    raw_access: bool = Field(default=False, description="원본 접근 권한자만 기록 (표정 등)")


class PrivacyTarget(Term):
    required: bool


class Ontology(Contract):
    version: SemVer
    status: Literal["draft", "frozen"]
    provisional: tuple[str, ...] = ()
    domains: dict[OntologyId, DomainTerm]
    tasks: dict[OntologyId, TaskTerm]
    verbs: dict[OntologyId, VerbTerm]
    gap_types: dict[OntologyId, Term]
    contact_target_kinds: dict[OntologyId, Term]
    grasp_types: dict[OntologyId, Term]
    hand_roles: dict[OntologyId, Term]
    body_parts: dict[OntologyId, Term]
    postures: dict[OntologyId, Term]
    physical_types: dict[OntologyId, Term]
    state_attributes: dict[OntologyId, StateAttribute]
    objects: dict[OntologyId, ObjectClass]
    events: dict[OntologyId, EventTerm]
    privacy_targets: dict[OntologyId, PrivacyTarget]
    surface_parts: dict[OntologyId, Term] = Field(
        default_factory=dict, description="표면 평면 꼭짓점 등 표면 3D 궤적의 부분 ID"
    )
    hand_joints: dict[OntologyId, Term] = Field(
        default_factory=dict, description="손 3D 궤적의 부분 ID (hand21 골격의 이름 붙은 점)"
    )

    @model_validator(mode="after")
    def _check_references(self) -> Ontology:
        errors: list[str] = []
        for tid, task in self.tasks.items():
            if task.domain not in self.domains:
                errors.append(f"작업 {tid}: 알 수 없는 도메인 {task.domain}")
        for oid, obj in self.objects.items():
            if obj.physical_type not in self.physical_types:
                errors.append(f"객체 {oid}: 알 수 없는 물리 유형 {obj.physical_type}")
            errors.extend(
                f"객체 {oid}: 알 수 없는 상태 속성 {s}"
                for s in obj.states
                if s not in self.state_attributes
            )
        substep_ids = [s for t in self.tasks.values() for s in t.substeps]
        if len(substep_ids) != len(set(substep_ids)):
            errors.append("하위 단계 ID는 작업 사이에서도 고유해야 합니다")
        if errors:
            raise ValueError("; ".join(errors))
        return self

    # ------------------------------------------------------------ 조회 도우미

    def substep_task(self, substep_id: str) -> str | None:
        for tid, task in self.tasks.items():
            if substep_id in task.substeps:
                return tid
        return None

    def tool_parts(self, class_id: str) -> frozenset[str]:
        obj = self.objects.get(class_id)
        if obj is None or obj.tool is None:
            return frozenset()
        return frozenset(obj.tool.working_parts) | frozenset(obj.tool.grip_parts)

    def known_parts(self) -> frozenset[str]:
        """부분 ID 전체: 모든 도구의 작용부·파지부, 표면 부분, 손 관절.

        3D 궤적과 관계의 part는 객체 클래스를 함께 싣지 않으므로 이 합집합으로 검사한다.
        """
        tool = {p for oid in self.objects for p in self.tool_parts(oid)}
        return frozenset(tool) | frozenset(self.surface_parts) | frozenset(self.hand_joints)


def load_ontology(directory: Path) -> Ontology:
    """디렉터리의 YAML 파일을 합쳐 온톨로지를 만든다. 같은 최상위 키가 두 파일에 있으면 오류."""
    files = sorted(directory.glob("*.yaml"))
    if not files:
        raise FileNotFoundError(f"온톨로지 YAML이 없습니다: {directory}")
    merged: dict[str, Any] = {}
    for path in files:
        data: Any = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{path.name}: 최상위는 매핑이어야 합니다")
        for key, value in data.items():  # pyright: ignore[reportUnknownVariableType]
            if not isinstance(key, str):
                raise ValueError(f"{path.name}: 최상위 키는 문자열이어야 합니다")
            if key in merged:
                raise ValueError(f"{path.name}: 키 {key}가 다른 파일과 중복됩니다")
            merged[key] = value
    return Ontology.model_validate(merged)


def find_ontology_dir(root: Path, version: str) -> Path:
    """config/ontology 아래에서 manifest의 version이 일치하는 디렉터리를 찾는다."""
    for manifest in sorted(root.glob("*/manifest.yaml")):
        data: Any = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
        if isinstance(data, dict) and data.get("version") == version:  # pyright: ignore[reportUnknownMemberType]
            return manifest.parent
    raise FileNotFoundError(f"온톨로지 버전 {version}을 {root}에서 찾을 수 없습니다")
