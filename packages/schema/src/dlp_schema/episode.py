"""에피소드 그래프: 개체·관계·이벤트·상태 전이 네 층으로 라벨을 묶는다."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from datetime import datetime
from enum import StrEnum
from itertools import pairwise

from pydantic import Field, model_validator

from dlp_schema.common import Contract, Identifier, Ms, OntologyId, SemVer
from dlp_schema.labels import (
    ActionPayload,
    BoxTrackPayload,
    CoveragePayload,
    EventPayload,
    GapPayload,
    HandStatePayload,
    KeypointTrackPayload,
    LabelRecord,
    MaskTrackPayload,
    ObjectStatePayload,
    Provenance,
    RelationPayload,
    SegmentPayload,
    Source,
    Trajectory3DPayload,
    Verification,
)


class EntityKind(StrEnum):
    HAND = "hand"
    OBJECT = "object"
    TOOL = "tool"
    PERSON = "person"
    SURFACE = "surface"
    CAMERA = "camera"


class Entity(Contract):
    entity_id: Identifier
    kind: EntityKind
    class_id: OntologyId | None = None


class StateTransition(Contract):
    entity_id: Identifier
    attribute: OntologyId
    from_value: OntologyId
    to_value: OntologyId
    t_ms: Ms


class EpisodeGraph(Contract):
    episode_id: Identifier
    session_id: Identifier
    ontology_version: SemVer
    t_start_ms: Ms
    t_end_ms: Ms
    entities: tuple[Entity, ...]
    labels: tuple[LabelRecord, ...] = Field(description="현재 유효한(대체·삭제되지 않은) 라벨")

    @model_validator(mode="after")
    def _check(self) -> EpisodeGraph:
        ids = {e.entity_id for e in self.entities}
        if len(ids) != len(self.entities):
            raise ValueError("entity_id가 중복되었습니다")
        problems: list[str] = []
        for label in self.labels:
            if label.session_id != self.session_id:
                problems.append(f"{label.label_id}: 다른 세션의 라벨")
            if label.ontology_version != self.ontology_version:
                problems.append(f"{label.label_id}: 온톨로지 버전 불일치")
            if label.retracted:
                problems.append(f"{label.label_id}: 삭제 레코드는 그래프에 넣지 않습니다")
            if label.t_end_ms < self.t_start_ms or label.t_start_ms > self.t_end_ms:
                problems.append(f"{label.label_id}: 에피소드 구간 밖의 라벨")
            problems.extend(
                f"{label.label_id}: 알 수 없는 개체 {ref}"
                for ref in entity_refs(label)
                if ref not in ids
            )
        if problems:
            raise ValueError("; ".join(problems))
        return self

    # ------------------------------------------------------------ 층별 보기

    @property
    def relations(self) -> list[LabelRecord]:
        return [x for x in self.labels if isinstance(x.payload, RelationPayload)]

    @property
    def events(self) -> list[LabelRecord]:
        kinds = (ActionPayload, SegmentPayload, GapPayload, EventPayload)
        return sorted(
            (x for x in self.labels if isinstance(x.payload, kinds)), key=lambda x: x.t_start_ms
        )

    @property
    def coverage(self) -> list[LabelRecord]:
        return [x for x in self.labels if isinstance(x.payload, CoveragePayload)]

    def state_transitions(self) -> list[StateTransition]:
        """같은 개체·속성의 상태 구간을 시간순으로 이어 전이를 만든다."""
        series: dict[tuple[str, str], list[tuple[int, str]]] = {}
        for x in self.labels:
            p = x.payload
            if isinstance(p, ObjectStatePayload):
                series.setdefault((p.entity_id, p.attribute), []).append((x.t_start_ms, p.value))
        out: list[StateTransition] = []
        for (entity_id, attribute), points in series.items():
            points.sort()
            for (_, prev), (t, cur) in pairwise(points):
                if prev != cur:
                    out.append(
                        StateTransition(
                            entity_id=entity_id,
                            attribute=attribute,
                            from_value=prev,
                            to_value=cur,
                            t_ms=t,
                        )
                    )
        return sorted(out, key=lambda s: s.t_ms)


def entity_refs(label: LabelRecord) -> list[str]:
    """라벨이 참조하는 개체 ID 목록."""
    p = label.payload
    match p:
        case BoxTrackPayload() | MaskTrackPayload() | KeypointTrackPayload():
            return [p.entity_id]
        case Trajectory3DPayload():
            return [p.entity_id]
        case HandStatePayload():
            return [p.target_id] if p.target_id else []
        case ActionPayload():
            return [r for r in (p.target_id, p.tool_id) if r]
        case ObjectStatePayload():
            return [p.entity_id]
        case CoveragePayload():
            return [r for r in (p.surface_id, p.tool_id) if r]
        case EventPayload():
            return [p.related_entity_id] if p.related_entity_id else []
        case RelationPayload():
            return [p.subject_id, p.object_id]
        case _:
            return []


def non_operational_ids(labels: list[LabelRecord]) -> set[str]:
    """운영 라벨이 아닌 레코드: 오류 삽입 레코드와 그 후손(검수자가 고친 것 포함), 측정용 레코드."""
    out = {x.label_id for x in labels if x.seeded_error or x.measurement is not None}
    children: dict[str, list[str]] = {}
    for x in labels:
        if x.parent_label_id:
            children.setdefault(x.parent_label_id, []).append(x.label_id)
    stack = list(out)
    while stack:
        for child in children.get(stack.pop(), []):
            if child not in out:
                out.add(child)
                stack.append(child)
    return out


def current_labels(labels: list[LabelRecord], *, operational: bool = True) -> list[LabelRecord]:
    """수정 이력에서 최신 레코드만 남긴다. 다른 레코드의 parent인 레코드와 삭제 레코드를 뺀다.

    operational이면 오류 삽입·측정용 레코드와 그 후손도 뺀다 (학습·내보내기·다른 단계의 입력).
    """
    excluded = non_operational_ids(labels) if operational else set[str]()
    superseded = {x.parent_label_id for x in labels if x.parent_label_id}
    retracted = {x.label_id for x in labels if x.retracted}
    return [
        x
        for x in labels
        if x.label_id not in superseded
        and x.label_id not in retracted
        and x.label_id not in excluded
    ]


def version_tag(model_version: str) -> str:
    """모델 버전마다 다른 짧은 ID 조각.

    모델 출처 라벨 ID에 넣어 버전이 바뀌어도 ID가 겹치지 않게 한다.
    """
    return hashlib.sha256(model_version.encode()).hexdigest()[:8]


def retractions(
    stale: Iterable[LabelRecord], model_version: str, now: datetime
) -> list[LabelRecord]:
    """이전 버전의 모델 라벨을 지우는 삭제 레코드 (parent=원래, retracted).

    ID는 `<원래>:retracted`이고 출처는 지운 쪽(새 버전)이다. 한 레코드는 한 번만 지울 수 있다
    (지운 뒤에는 현재 라벨이 아니므로 다시 대상이 되지 않는다).
    """
    return [
        x.model_copy(
            update={
                "label_id": f"{x.label_id}:retracted",
                "parent_label_id": x.label_id,
                "retracted": True,
                "verification": Verification(),
                "provenance": Provenance(source=Source.MODEL, model_version=model_version),
                "confidence": x.confidence if x.confidence is not None else 1.0,
                "created_at": now,
            }
        )
        for x in stale
    ]
