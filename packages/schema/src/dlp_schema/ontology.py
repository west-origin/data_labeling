"""온톨로지 사전과 로더. 온톨로지는 config/ontology/<버전>/*.yaml에 둔다.

역할
    라벨이 쓸 수 있는 ID(도메인·작업·동사·객체·이벤트·프라이버시 대상·신체 부위·부분 등)의 사전을
    타입으로 정의하고 YAML에서 읽는다 (WP1, ADR 0002·0028).

파일 구성
    한 버전 디렉터리(예: `config/ontology/v1/`)의 모든 `*.yaml`을 합쳐 하나의 `Ontology`가 된다.
    파일마다 서로 다른 최상위 키를 둔다 (같은 키가 두 파일에 있으면 오류). `manifest.yaml`이
    version·status·provisional을 가진다.

주요 이름
    - 사전 항목 타입: `Term`과 하위 타입(`DomainTerm`, `TaskTerm`, `VerbTerm`, `StateAttribute`,
      `ObjectClass`, `EventTerm`, `PrivacyTarget`), `ToolSpec`, `VerbLevel`.
    - `Ontology`: 사전 전체와 교차 참조 검증, 조회 도우미.
    - `load_ontology(dir)`: 디렉터리의 YAML을 합쳐 검증한다.
    - `find_ontology_dir(root, version)`: manifest의 version으로 디렉터리를 찾는다.

DB 등록
    `dlp ontology register <버전>`(`make db-upgrade`)이 `db.repository.register_ontology`로
    `ontology_versions`에 넣는다. 저장·비교 대상은 파싱된 값(`model_dump(mode="json")`)이라 YAML
    주석이나 서식을 바꿔도 등록 내용은 같다. 값을 바꾸면 초안(draft)의 덧붙이기만 허용되고, 그 밖은
    새 버전과 이관(`migration`)이 필요하다.

주의
    - 사전 키는 `OntologyId`(소문자 snake_case) 형식이어야 한다.
    - 클래스 docstring·Field description은 `schemas/ontology.schema.json`에 들어간다.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, model_validator

from dlp_schema.common import Contract, OntologyId, SemVer


# 사전 항목의 공통 필드. ko: 사람이 읽는 한국어 이름 (필수). description: 보충 설명 (선택).
class Term(Contract):
    ko: str
    description: str | None = None


# 도메인 (청소·요양돌봄·간호). phase: 기준 문서 로드맵의 도입 단계 (A/B/C).
class DomainTerm(Term):
    phase: str


# 작업. domain: 속한 도메인 ID (domains의 키여야 한다).
# substeps: 하위 단계 ID → 항목. 하위 단계 ID는 작업 사이에서도 고유해야 한다 (Ontology 검증기).
class TaskTerm(Term):
    domain: OntologyId
    substeps: dict[OntologyId, Term] = Field(default_factory=dict)


# 동사 수준. 행동(action) 라벨에는 primitive만 쓴다. skill·human_skill은 상위 구간(segment
# level=skill)의 ref_id로 쓴다 (`validation.check_label`).
class VerbLevel(StrEnum):
    PRIMITIVE = "primitive"  # 원시 동작 (라벨 단위: 잡다, 놓다, 밀다 ...)
    SKILL = "skill"  # 원시 동작을 묶은 기술 (닦다, 열다 ...)
    HUMAN_SKILL = "human_skill"  # 사람 대상 기술 (굴리다, 일으키다 ..., Phase B부터)


# 동사. level: 수준. contactless: 접촉 없이 하는 동사(예: inspect). 이런 동사의 행동 라벨에는
# 접촉 시작 시각이 있으면 안 된다.
class VerbTerm(Term):
    level: VerbLevel
    contactless: bool = False


# 객체 상태 속성 (예: cleanliness). values: 허용 값 ID (2개 이상, 예: dirty, clean).
class StateAttribute(Term):
    values: tuple[OntologyId, ...] = Field(min_length=2)


# 도구의 부분. working_parts: 작용부 (표면에 닿아 일하는 부분, 예: mop_head).
# grip_parts: 파지부 (손으로 쥐는 부분, 예: handle).
# 둘 다 1개 이상. 부분 마스크·관계의 part ID가 된다.
class ToolSpec(Contract):
    working_parts: tuple[OntologyId, ...] = Field(min_length=1)
    grip_parts: tuple[OntologyId, ...] = Field(min_length=1)


# 객체 클래스.
#   physical_type: 물리 유형 ID (physical_types의 키).
#   states: 이 클래스가 가질 수 있는 상태 속성 ID (state_attributes의 키).
#   tool: 도구면 부분 정의, 아니면 None.
#   surface: 처리 대상 표면인지 (청결 상태·커버리지 기록 대상).
class ObjectClass(Term):
    physical_type: OntologyId
    states: tuple[OntologyId, ...] = ()
    tool: ToolSpec | None = None
    surface: bool = False


# 이벤트 유형.
#   category: 분류 (실패·재시도 / 작업자 안전 / 대상자 안전 / 감염 관리).
#   form: point(시점: t_start == t_end) | interval(구간).
#   severity: 라벨에 심각도(1~3)가 필수인지. requires_action: related_action_id 필수인지.
#   requires_object: related_entity_id 필수인지. raw_access: Field description 참고.
class EventTerm(Term):
    category: Literal["failure_retry", "worker_safety", "subject_safety", "infection_control"]
    form: Literal["point", "interval"]
    severity: bool = False
    requires_action: bool = False
    requires_object: bool = False
    raw_access: bool = Field(default=False, description="원본 접근 권한자만 기록 (표정 등)")


# 프라이버시(블러) 대상. required: v1 필수 대상인지. false는 확장 후보.
class PrivacyTarget(Term):
    required: bool


# 온톨로지 한 버전의 사전 전체. 필드 이름 = YAML 최상위 키.
#   version / status: manifest.yaml의 버전과 상태 (draft=초안, 덧붙이기 허용 / frozen=확정).
#   provisional: 파일럿 후 확정할 범주 이름 (설명용, 동작에 영향 없음).
#   domains·tasks·verbs·gap_types·contact_target_kinds·grasp_types·hand_roles·body_parts·postures·
#   physical_types·state_attributes·objects·events·privacy_targets: 범주별 사전 (ID → 항목).
#   surface_parts·hand_joints: 부분 ID 사전 (3차 검수에서 추가, 기본값 빈 사전).
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
        """사전 사이의 교차 참조를 검사한다.

        - 작업의 domain이 domains에 있는지
        - 객체의 physical_type·states가 각 사전에 있는지
        - 하위 단계 ID가 작업 사이에서도 고유한지

        Raises:
            ValueError: 위반을 모두 모아 `; `로 이은 메시지 (Pydantic이 ValidationError로 감싼다).
        """
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
        """하위 단계 ID가 속한 작업 ID. 어느 작업에도 없으면 None."""
        for tid, task in self.tasks.items():
            if substep_id in task.substeps:
                return tid
        return None

    def tool_parts(self, class_id: str) -> frozenset[str]:
        """객체 클래스의 작용부와 파지부 ID 합집합. 클래스가 없거나 도구가 아니면 빈 집합."""
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
    """디렉터리의 YAML 파일을 합쳐 온톨로지를 만든다. 같은 최상위 키가 두 파일에 있으면 오류.

    Args:
        directory: 버전 디렉터리 (예: `config/ontology/v1`). 하위 디렉터리는 읽지 않는다.

    Returns:
        검증된 `Ontology`.

    Raises:
        FileNotFoundError: `*.yaml`이 하나도 없을 때.
        ValueError: 최상위가 매핑이 아니거나, 키가 문자열이 아니거나, 키가 파일 사이에 중복될 때.
        pydantic.ValidationError: 사전 형식·교차 참조 위반.
    """
    # 파일 이름 순으로 읽어 결과(오류 메시지 포함)가 결정적이게 한다. 빈 파일은 빈 매핑으로 본다.
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
    """config/ontology 아래에서 manifest의 version이 일치하는 디렉터리를 찾는다.

    Args:
        root: 온톨로지 루트 (보통 `config/ontology`). 바로 아래 디렉터리의 `manifest.yaml`만 본다.
        version: 찾을 SemVer 문자열 (예: "1.0.0"). manifest에서 문자열로 적혀 있어야 일치한다.

    Raises:
        FileNotFoundError: 일치하는 manifest가 없을 때.
    """
    for manifest in sorted(root.glob("*/manifest.yaml")):
        data: Any = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
        if isinstance(data, dict) and data.get("version") == version:  # pyright: ignore[reportUnknownMemberType]
            return manifest.parent
    raise FileNotFoundError(f"온톨로지 버전 {version}을 {root}에서 찾을 수 없습니다")
