"""세션 라벨에서 관계·커버리지를 도출한다 (DB와 무관한 순수 계산).

입력은 수정 이력을 반영한 현재 라벨이다. 이 모듈이 이전에 만든 관계·커버리지는 입력에서 뺀다.
- 개체 클래스: 박스·마스크 트랙의 class_id. 없으면 개체 ID에서 추정한다 (rag_01 → rag).
- 도구 작용부 궤적: 도구 클래스이고 part가 온톨로지의 작용부인 trajectory3d.
- 표면: surface 클래스이고 네 꼭짓점 궤적이 모두 있는 개체. 작용부와 좌표계가 같아야 한다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from dlp_relations.contact import SurfaceContact, Track3D, tool_surface_contacts
from dlp_relations.coverage import coverage_ratio
from dlp_relations.policy import RelationsPolicy
from dlp_relations.rules import RelationDraft, apply_rules
from dlp_schema.episode import current_labels
from dlp_schema.labels import (
    BoxTrackPayload,
    CoveragePayload,
    HandStatePayload,
    LabelRecord,
    MaskTrackPayload,
    Source,
    Trajectory3DPayload,
)
from dlp_schema.ontology import Ontology

VERSION_PREFIX = "relations-"


@dataclass(frozen=True)
class CoverageDraft:
    payload: CoveragePayload
    start_ms: int
    end_ms: int
    confidence: float


@dataclass
class Derived:
    relations: list[RelationDraft]
    coverage: list[CoverageDraft]
    contacts: list[SurfaceContact]


def is_derived(x: LabelRecord) -> bool:
    return x.provenance.source is Source.MODEL and (x.provenance.model_version or "").startswith(
        VERSION_PREFIX
    )


def entity_classes(labels: list[LabelRecord]) -> dict[str, str]:
    out: dict[str, str] = {}
    for x in labels:
        if isinstance(x.payload, BoxTrackPayload | MaskTrackPayload):
            out[x.payload.entity_id] = x.payload.class_id
    return out


def class_of(entity_id: str, classes: dict[str, str]) -> str:
    if entity_id in classes:
        return classes[entity_id]
    return re.sub(r"_\d+$", "", entity_id.removeprefix("ov_"))


def derive(labels: list[LabelRecord], ontology: Ontology, policy: RelationsPolicy) -> Derived:
    current = [x for x in current_labels(labels) if not is_derived(x)]
    classes = entity_classes(current)
    hand_states = [x for x in current if isinstance(x.payload, HandStatePayload)]

    tools: list[tuple[Track3D, str]] = []  # (작용부 궤적, 좌표계)
    corners: dict[tuple[str, str], dict[str, Track3D]] = {}  # (표면, 좌표계) → part → 궤적
    for x in current:
        p = x.payload
        if not isinstance(p, Trajectory3DPayload) or p.part is None:
            continue
        obj = ontology.objects.get(class_of(p.entity_id, classes))
        if obj is None:
            continue
        if obj.tool is not None and p.part in obj.tool.working_parts:
            tools.append((Track3D.from_payload(p), p.frame.value))
        elif obj.surface and p.part in policy.tool_surface.corner_parts:
            corners.setdefault((p.entity_id, p.frame.value), {})[p.part] = Track3D.from_payload(p)

    contacts: list[SurfaceContact] = []
    coverage: list[CoverageDraft] = []
    ts = policy.tool_surface
    for tool, frame in tools:
        grasped = [
            (x.t_start_ms, x.t_end_ms)
            for x in hand_states
            if isinstance(x.payload, HandStatePayload)
            and x.payload.contact_target_kind == "tool"
            and x.payload.target_id == tool.entity_id
        ]
        for (surface_id, surface_frame), parts in sorted(corners.items()):
            if surface_frame != frame or set(parts) != set(ts.corner_parts):
                continue
            found = tool_surface_contacts(
                tool, surface_id, [parts[c] for c in ts.corner_parts], grasped, ts
            )
            if not found:
                continue
            contacts += found
            footprint = policy.coverage.footprint(class_of(tool.entity_id, classes))
            ratio = coverage_ratio(found, footprint, policy.coverage.grid_m)
            coverage.append(
                CoverageDraft(
                    CoveragePayload(
                        surface_id=surface_id, tool_id=tool.entity_id, ratio=round(ratio, 4)
                    ),
                    found[0].start_ms,
                    found[-1].end_ms,
                    ts.confidence,
                )
            )
    return Derived(apply_rules(policy, hand_states, contacts), coverage, contacts)
