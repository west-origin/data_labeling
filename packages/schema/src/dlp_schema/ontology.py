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
    - 객체 물리 사전값(ADR 0049): `PhysicalPriors`, `PriorRange`, `StatePriorRange`, `PriorSource`.
      재질(`materials`)·촉감(`tactile_properties`) 어휘와 함께 쓴다. 측정값이 아니라 사전값이다.
    - `TermName`, `TERM_CATEGORIES`: 한·영 표시 이름과 이름이 있는 범주 (ADR 0043).
    - `Ontology`: 사전 전체와 교차 참조 검증, 조회 도우미 (`display_names`, `name` 포함).
    - `load_ontology(dir)`: 디렉터리의 YAML을 합쳐 검증한다.
    - `find_ontology_dir(root, version)`: manifest의 version으로 디렉터리를 찾는다.
    - `ontology_digest(ontology)`: 내용 해시 (같은 버전의 draft 덧붙이기도 구별한다).
    - `PHYSICS_CATEGORIES`: 물리 사전값 어휘 범주 (`Ontology.physics_names`, 용어집에 넣지 않는다).

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

import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import Field, StringConstraints, model_validator

from dlp_schema.common import Contract, OntologyId, SemVer

# 사람이 읽는 이름의 글자 제약: 앞뒤 공백을 지우고 1자 이상 (빈 이름 금지, ADR 0043).
DisplayName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


# 사전 항목의 공통 필드. ko: 사람이 읽는 한국어 이름 (필수). en: 영어 이름 (필수, ADR 0043).
# description: 보충 설명 (선택, 한국어). ID는 언어와 무관한 영어 snake_case이고 라벨에는 ID만
# 저장한다. ko·en은 화면·내보내기·번역 용어집에 쓰는 표시 이름이다.
class Term(Contract):
    ko: DisplayName
    en: DisplayName
    description: str | None = None


# 사전 항목 하나의 한·영 표시 이름 (`Ontology.display_names`). 내보내기 구간 JSON의 `term_names`
# 값이기도 하다 (`dlp_schema.export.IntervalFile`).
class TermName(Contract):
    ko: DisplayName
    en: DisplayName


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


# ---------------------------------------------------------------- 물리 사전값 (ADR 0049)


# 물리 사전값의 출처. 둘 다 측정값이 아니다 (이 세션·이 물체를 잰 값이 아니라 클래스의 일반 범위).
class PriorSource(StrEnum):
    PRIOR_LITERATURE = "prior_literature"  # 문헌·제품 사양·표준 표에서 옮긴 범위 (출처를 note에)
    PRIOR_ESTIMATE = "prior_estimate"  # 일반 지식으로 잡은 넓은 추정 범위 (전문가 검토 전)


class PriorRange(Contract):
    """사전값 범위 [min, max]. 둘 다 0보다 큰 유한한 수이고 min ≤ max다. 측정값이 아니다."""

    min: float = Field(gt=0, allow_inf_nan=False)
    max: float = Field(gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _ordered(self) -> PriorRange:
        """min ≤ max인지 본다.

        Raises:
            ValueError: min > max일 때.
        """
        if self.min > self.max:
            raise ValueError(f"사전값 범위의 min({self.min})이 max({self.max})보다 큽니다")
        return self


# 기본 범위(min·max)가 특정 상태 값의 범위인 상태 속성 → 그 값 (ADR 0049·0053). 젖음은 기본이
# 건조(dry) 조건이고 젖은 경우만 by_state.wetness.wet로 따로 적는다 (dry를 따로 적으면 거부). 이
# 표에 없는 상태 속성(fill_state·cleanliness 등)의 기본 범위는 그 속성의 모든 상태를 덮는 범위이고,
# by_state 범위는 기본 범위 안이어야 한다. 구매자 규약의 base_range와 같은 뜻이다
# (`dlp_schema.export.PHYSICAL_PRIOR_CONVENTIONS`).
BASE_CONDITION_STATES: dict[str, str] = {"wetness": "dry"}
# 물리 사전값을 적지 않는 물리 유형 (ADR 0049 1절: 사람·액체·분말은 클래스 수준 값이 뜻이 없다)
NO_PRIOR_PHYSICAL_TYPES = frozenset({"person", "fluid_powder"})


class StatePriorRange(PriorRange):
    """기본 사전값 범위와, 객체 상태 값별로 바꿔 쓰는 범위 (예: 양동이 fill_state=full의 질량).

    min·max는 기본 범위다. 젖음(wetness)이 있는 객체에서는 건조 조건의 범위이고(젖은 경우는
    by_state.wetness.wet), 그 밖의 상태 속성에서는 그 속성의 모든 상태를 덮는 범위다
    (`BASE_CONDITION_STATES`). 상태를 모르면 기본 범위와 by_state 범위 전체를 덮는 범위를 쓴다.
    by_state: 상태 속성 ID → 상태 값 ID → 범위 (상태 속성은 하나만, 조합 규칙은 두지 않는다). 상태
    속성은 그 객체의 states에, 값은 그 속성의 values에 있어야 한다 (`Ontology` 검증기).
    """

    by_state: dict[OntologyId, dict[OntologyId, PriorRange]] = Field(
        default_factory=dict,
        description="상태 속성 ID → 상태 값 ID → 사전값 범위 (상태 속성 하나만)",
    )

    @model_validator(mode="after")
    def _one_attribute(self) -> StatePriorRange:
        """by_state는 상태 속성 하나만, 그 아래 값이 하나 이상.

        두 속성(젖음 + 청결)을 함께 적으면 둘 다 해당하는 물체에 어느 범위를 쓸지 정해지지 않는다.

        Raises:
            ValueError: 상태 속성이 둘 이상이거나 값 사전이 비었을 때.
        """
        if len(self.by_state) > 1:
            raise ValueError(f"by_state에는 상태 속성을 하나만 적습니다: {sorted(self.by_state)}")
        for attr, values in self.by_state.items():
            if not values:
                raise ValueError(f"by_state.{attr}에 상태 값이 없습니다")
        return self


def _base_meaning_errors(
    oid: str, name: str, attr: str, rng: StatePriorRange, by_value: dict[str, PriorRange]
) -> list[str]:
    """기본 범위의 뜻(`BASE_CONDITION_STATES`)과 by_state가 맞는지.

    기본이 특정 상태 값(젖음 → 건조)인 속성은 그 값을 by_state에 따로 적지 않는다. 그 밖의 속성은
    기본 범위가 모든 상태를 덮으므로 by_state 범위가 기본 범위 안이어야 한다.
    """
    base_value = BASE_CONDITION_STATES.get(attr)
    if base_value is not None:
        if base_value in by_value:
            return [
                f"객체 {oid}: {name}.by_state.{attr}.{base_value}는 기본 범위의 조건이라 따로 "
                "적지 않습니다"
            ]
        return []
    return [
        f"객체 {oid}: {name}.by_state.{attr}.{v} 범위가 기본 범위(모든 상태를 덮음) 밖입니다"
        for v, r in by_value.items()
        if r.min < rng.min or r.max > rng.max
    ]


class PhysicalPriors(Contract):
    """객체 클래스의 물리 사전값 (재질·촉감·질량·정지 마찰 계수). 측정값이 아니라 사전값(prior)이다.

    클래스의 흔한 실물 범위를 넓게 잡은 값이며, 영상 속 특정 물체를 잰 값이 아니다. 내보내기는
    `measured: false`와 출처(source)를 함께 싣는다 (ADR 0049).
    """

    # 필드:
    #   source: 수치 사전값의 출처 (필수). note: 근거·가정 메모 (선택, 한국어. prior_literature면
    #     출처 필수).
    #   material: 대표 재질 ID (materials의 키). part_materials: 도구 부분 ID → 재질 ID (그 객체의
    #     작용부·파지부만, 대표 재질과 다른 부분만 적는다).
    #   tactile: 촉감 속성 ID 목록 (tactile_properties의 키, 중복 금지).
    #   mass_kg: 질량 범위 (kg). by_state로 상태별 범위 (예: fill_state).
    #   static_friction: 정지 마찰 계수 범위 (무차원). 상대면은 건조한 맨손 피부·니트릴 장갑·고무
    #     그리퍼 패드를 함께 보는 넓은 범위이고, 젖은 경우는 by_state(wetness)로 따로 적는다. 물체
    #     쪽 면은 도구면 손으로 쥐는 파지부(grip_parts), 아니면 주 접촉면이다 (촉감도 같다).
    source: PriorSource = Field(description="수치 사전값의 출처 (prior_*: 측정값이 아님)")
    note: str | None = None
    material: OntologyId | None = None
    part_materials: dict[OntologyId, OntologyId] = Field(
        default_factory=dict, description="도구 부분 ID → 재질 ID (대표 재질과 다른 부분만)"
    )
    tactile: tuple[OntologyId, ...] = ()
    mass_kg: StatePriorRange | None = Field(default=None, description="질량 사전값 범위 (kg)")
    static_friction: StatePriorRange | None = Field(
        default=None,
        description=(
            "정지 마찰 계수 사전값 범위 (무차원). 상대면: 건조한 맨손 피부·니트릴 장갑·고무 그리퍼 "
            "패드. 물체 쪽 면: 도구는 파지부(grip_parts), 그 밖은 주 접촉면. "
            "젖음은 by_state.wetness"
        ),
    )

    @model_validator(mode="after")
    def _not_empty(self) -> PhysicalPriors:
        """재질·촉감·질량·마찰 중 하나는 있어야 하고 촉감 ID는 겹치지 않아야 한다.
        prior_literature면 note에 출처가 있어야 하고(구매자에게 '문헌 출처'로 판다),
        part_materials는 대표 재질과 다른 부분만 적는다.

        Raises:
            ValueError: 모두 비었거나 촉감 ID가 중복될 때, 문헌 출처에 note가 없을 때, 대표 재질과
                같은 부분 재질이 있을 때.
        """
        if self.source is PriorSource.PRIOR_LITERATURE and not (self.note or "").strip():
            raise ValueError("source가 prior_literature면 note에 출처를 적습니다")
        same = sorted(k for k, v in self.part_materials.items() if v == self.material)
        if same:
            raise ValueError(f"part_materials에 대표 재질과 같은 부분이 있습니다: {same}")
        if (
            self.material is None
            and not self.part_materials
            and not self.tactile
            and self.mass_kg is None
            and self.static_friction is None
        ):
            raise ValueError("물리 사전값이 비었습니다 (재질·촉감·질량·마찰 중 하나는 필요)")
        if len(set(self.tactile)) != len(self.tactile):
            raise ValueError("촉감 속성 ID가 중복됩니다")
        return self


# 객체 클래스.
#   physical_type: 물리 유형 ID (physical_types의 키).
#   states: 이 클래스가 가질 수 있는 상태 속성 ID (state_attributes의 키).
#   tool: 도구면 부분 정의, 아니면 None.
#   surface: 처리 대상 표면인지 (청결 상태·커버리지 기록 대상).
#   physical_priors: 물리 사전값 (선택, ADR 0049). 측정값이 아니다.
class ObjectClass(Term):
    physical_type: OntologyId
    states: tuple[OntologyId, ...] = ()
    tool: ToolSpec | None = None
    surface: bool = False
    physical_priors: PhysicalPriors | None = Field(
        default=None, description="물리 사전값 (재질·촉감·질량·마찰, 측정값 아님, ADR 0049)"
    )


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
#   materials·tactile_properties: 객체 물리 사전값(`ObjectClass.physical_priors`)의 재질·촉감 어휘
#     (ADR 0049, 기본값 빈 사전). 라벨에는 쓰지 않는다.
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
    materials: dict[OntologyId, Term] = Field(
        default_factory=dict, description="객체 물리 사전값의 재질 ID (ADR 0049)"
    )
    tactile_properties: dict[OntologyId, Term] = Field(
        default_factory=dict, description="객체 물리 사전값의 촉감 속성 ID (ADR 0049)"
    )

    @model_validator(mode="after")
    def _check_references(self) -> Ontology:
        """사전 사이의 교차 참조를 검사한다.

        - 작업의 domain이 domains에 있는지
        - 객체의 physical_type·states가 각 사전에 있는지
        - 객체 물리 사전값의 재질·촉감·부분·상태 참조 (`_physics_errors`)
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
            errors.extend(self._physics_errors(oid, obj))
        substep_ids = [s for t in self.tasks.values() for s in t.substeps]
        if len(substep_ids) != len(set(substep_ids)):
            errors.append("하위 단계 ID는 작업 사이에서도 고유해야 합니다")
        if errors:
            raise ValueError("; ".join(errors))
        return self

    def _physics_errors(self, oid: str, obj: ObjectClass) -> list[str]:
        """객체 물리 사전값의 교차 참조 위반 목록 (ADR 0049).

        재질·촉감 ID가 어휘에 있는지, part_materials의 부분이 그 객체의 도구 부분인지, by_state의
        상태 속성이 그 객체의 states에 있고 값이 그 속성의 values에 있는지 본다.
        """
        p = obj.physical_priors
        if p is None:
            return []
        errors: list[str] = []
        if obj.physical_type in NO_PRIOR_PHYSICAL_TYPES:
            errors.append(f"객체 {oid}: 물리 유형 {obj.physical_type}에는 사전값을 적지 않습니다")
        materials = [p.material] if p.material is not None else []
        for m in [*materials, *p.part_materials.values()]:
            if m not in self.materials:
                errors.append(f"객체 {oid}: 알 수 없는 재질 {m}")
        parts = set(obj.tool.working_parts) | set(obj.tool.grip_parts) if obj.tool else set[str]()
        errors.extend(
            f"객체 {oid}: 도구 부분이 아닌 재질 부분 {part}"
            for part in p.part_materials
            if part not in parts
        )
        errors.extend(
            f"객체 {oid}: 알 수 없는 촉감 속성 {t}"
            for t in p.tactile
            if t not in self.tactile_properties
        )
        for name, rng in (("mass_kg", p.mass_kg), ("static_friction", p.static_friction)):
            if rng is not None:
                errors.extend(self._state_range_errors(oid, obj, name, rng))
        return errors

    def _state_range_errors(
        self, oid: str, obj: ObjectClass, name: str, rng: StatePriorRange
    ) -> list[str]:
        """by_state의 상태 속성·값이 그 객체에 허용된 것인지 본다."""
        errors: list[str] = []
        for attr, by_value in rng.by_state.items():
            if attr not in obj.states:
                errors.append(
                    f"객체 {oid}: {name}.by_state의 상태 속성 {attr}가 객체 states에 없음"
                )
                continue
            values = self.state_attributes[attr].values if attr in self.state_attributes else ()
            errors.extend(
                f"객체 {oid}: {name}.by_state.{attr}의 알 수 없는 값 {v}"
                for v in by_value
                if v not in values
            )
            errors.extend(_base_meaning_errors(oid, name, attr, rng, by_value))
        return errors

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

    def display_names(self) -> dict[str, TermName]:
        """모든 사전 항목의 한·영 표시 이름. 키는 `"<범주>:<ID>"`다 (ADR 0043).

        범주는 `Ontology`의 사전 필드 이름(예: `verbs`, `objects`)이고, 작업의 하위 단계는
        `substeps`다. 같은 ID가 여러 범주에 있을 수 있어(예: `person`) 범주를 붙인다.
        상태 속성의 값(`values`, 예: dirty)은 사전 항목이 아니라 이름이 없다. 순서는 필드 순 →
        범주 안 ID 순으로 결정적이다.
        """
        out: dict[str, TermName] = {}
        for category in TERM_CATEGORIES:
            table: dict[str, Term] = getattr(self, category)
            for term_id, term in sorted(table.items()):
                out[f"{category}:{term_id}"] = TermName(ko=term.ko, en=term.en)
        substeps = sorted((k, v) for t in self.tasks.values() for k, v in t.substeps.items())
        for term_id, term in substeps:
            out[f"substeps:{term_id}"] = TermName(ko=term.ko, en=term.en)
        return out

    def physics_names(self) -> dict[str, TermName]:
        """물리 사전값 어휘(재질·촉감)의 한·영 표시 이름. 키는 `"<범주>:<ID>"`다 (ADR 0049).

        `display_names`(라벨 어휘, 번역 용어집)와 따로 둔다. 라벨에는 이 ID가 들어가지 않고,
        용어집에 넣으면 번역 버전(용어집 해시)이 바뀌어 검수 전 번역을 모두 다시 만들기 때문이다.
        """
        out: dict[str, TermName] = {}
        for category in PHYSICS_CATEGORIES:
            table: dict[str, Term] = getattr(self, category)
            for term_id, term in sorted(table.items()):
                out[f"{category}:{term_id}"] = TermName(ko=term.ko, en=term.en)
        return out

    def name(self, category: str, term_id: str, language: Literal["ko", "en"]) -> str | None:
        """사전 항목의 표시 이름 (`display_names`의 키 규칙). 없는 범주·ID면 None."""
        found = self.display_names().get(f"{category}:{term_id}")
        if found is None:
            return None
        return found.ko if language == "ko" else found.en

    def known_parts(self) -> frozenset[str]:
        """부분 ID 전체: 모든 도구의 작용부·파지부, 표면 부분, 손 관절.

        3D 궤적과 관계의 part는 객체 클래스를 함께 싣지 않으므로 이 합집합으로 검사한다.
        """
        tool = {p for oid in self.objects for p in self.tool_parts(oid)}
        return frozenset(tool) | frozenset(self.surface_parts) | frozenset(self.hand_joints)


# 이름(ko·en)이 있는 사전 범주 = `Ontology`의 사전 필드 (하위 단계는 `display_names`가 따로 넣는다)
TERM_CATEGORIES = (
    "domains", "tasks", "verbs", "gap_types", "contact_target_kinds", "grasp_types", "hand_roles",
    "body_parts", "postures", "physical_types", "state_attributes", "objects", "events",
    "privacy_targets", "surface_parts", "hand_joints",
)  # fmt: skip

# 물리 사전값 어휘 범주 (ADR 0049). `display_names`·번역 용어집에는 넣지 않는다 (`physics_names`).
PHYSICS_CATEGORIES = ("materials", "tactile_properties")


def ontology_digest(ontology: Ontology) -> str:
    """온톨로지 내용 해시 (sha256 64자). 파싱된 값(`model_dump(mode="json")`)의 키 정렬 JSON.

    같은 버전이라도 draft 덧붙이기(`register_ontology`)로 내용이 바뀌면 달라진다. YAML 주석·서식은
    영향을 주지 않는다. 검수 화면 프로젝트 이름(`dlp_review.ontology_projects`)과 AI 1차 검수 단위
    키(`dlp_aireview.plan.unit_id`, F-AIR-32)가 쓴다.
    """
    content = json.dumps(ontology.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(content.encode()).hexdigest()


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
