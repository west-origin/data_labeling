"""세션 라벨에서 관계·커버리지를 도출한다 (DB와 무관한 순수 계산).

입력은 수정 이력을 반영한 현재 라벨이다. 이 모듈이 이전에 만든 관계·커버리지는 입력에서 뺀다.
- 개체 클래스: 박스·마스크 트랙의 class_id. 없으면 개체 ID에서 추정한다 (rag_01 → rag).
- 도구 작용부 궤적: 도구 클래스이고 part가 온톨로지의 작용부인 trajectory3d.
- 표면: surface 클래스이고 네 꼭짓점 궤적이 모두 있는 개체. 작용부와 좌표계가 같아야 한다.
- 같은 (개체, 부위, 좌표계) 궤적이 여럿이면 하나만 쓰고, 커버리지는 (도구, 표면)마다 하나다.
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
    VerificationState,
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


def _preference(x: LabelRecord) -> tuple[bool, float, str]:
    """정렬 키: 검수된·사람 라벨 먼저, 그다음 최신, 같으면 ID 순 (결정적)."""
    reviewed = (
        x.provenance.source is Source.HUMAN
        or x.verification.state is not VerificationState.UNREVIEWED
    )
    return (not reviewed, -x.created_at.timestamp(), x.label_id)


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

    # (도구, 부위, 좌표계)마다 궤적 하나. 같은 작용부 궤적이 여럿이면(예: 입력이 바뀌어 3D 단계가
    # 다시 돌았는데 이전 것은 검수돼 남음) 검수된·사람 궤적을 먼저, 그다음 최신을 쓴다.
    # 같은 접촉·커버리지가 두 번 나와 같은 ID로 겹치지 않게 한다
    chosen: dict[tuple[str, str, str], LabelRecord] = {}
    corners: dict[tuple[str, str], dict[str, Track3D]] = {}  # (표면, 좌표계) → part → 궤적
    corner_src: dict[tuple[str, str, str], LabelRecord] = {}
    for x in sorted(current, key=_preference):
        p = x.payload
        if not isinstance(p, Trajectory3DPayload) or p.part is None:
            continue
        obj = ontology.objects.get(class_of(p.entity_id, classes))
        if obj is None:
            continue
        key = (p.entity_id, p.part, p.frame.value)
        if obj.tool is not None and p.part in obj.tool.working_parts:
            chosen.setdefault(key, x)
        elif (
            obj.surface
            and p.part in policy.tool_surface.corner_parts
            and corner_src.setdefault(key, x) is x
        ):
            corners.setdefault((p.entity_id, p.frame.value), {})[p.part] = Track3D.from_payload(p)
    # (도구, 좌표계) → 작용부 궤적들. 커버리지는 (도구, 표면)마다 하나다 (작용부를 합친다)
    tools: dict[tuple[str, str], list[Track3D]] = {}
    for (entity_id, _, frame), x in sorted(chosen.items()):
        assert isinstance(x.payload, Trajectory3DPayload)
        tools.setdefault((entity_id, frame), []).append(Track3D.from_payload(x.payload))

    contacts: list[SurfaceContact] = []
    coverage: list[CoverageDraft] = []
    ts = policy.tool_surface
    for (tool_id, frame), parts_tracks in sorted(tools.items()):
        grasped = [
            (x.t_start_ms, x.t_end_ms)
            for x in hand_states
            if isinstance(x.payload, HandStatePayload)
            and x.payload.contact_target_kind == "tool"
            and x.payload.target_id == tool_id
        ]
        for (surface_id, surface_frame), parts in sorted(corners.items()):
            if surface_frame != frame or set(parts) != set(ts.corner_parts):
                continue
            found = sorted(
                (
                    c
                    for tool in parts_tracks
                    for c in tool_surface_contacts(
                        tool, surface_id, [parts[k] for k in ts.corner_parts], grasped, ts
                    )
                ),
                key=lambda c: (c.start_ms, c.end_ms),
            )
            if not found:
                continue
            contacts += found
            footprint = policy.coverage.footprint(class_of(tool_id, classes))
            ratio = coverage_ratio(found, footprint, policy.coverage.grid_m)
            coverage.append(
                CoverageDraft(
                    CoveragePayload(surface_id=surface_id, tool_id=tool_id, ratio=round(ratio, 4)),
                    min(c.start_ms for c in found),
                    max(c.end_ms for c in found),
                    ts.confidence,
                )
            )
    return Derived(apply_rules(policy, hand_states, contacts), coverage, contacts)
