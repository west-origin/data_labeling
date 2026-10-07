"""세션 관계·커버리지 도출 실행 (`dlp relations run`). 멱등이다.

관계·커버리지 레코드의 model_version은 "relations-<정책 해시>"다. 다시 실행하면:
- 내용(구간·필드)이 같은 현재 레코드는 그대로 둔다 (정책이 바뀌어도).
- 더 이상 나오지 않는 이 모듈의 현재 레코드는 삭제 레코드(retracted, parent=원래)로 표시한다.
- 새로 나온 것만 넣는다. 사람이 고치거나 지운 적이 있는 내용은 다시 넣지 않는다.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime

import sqlalchemy as sa

from dlp_relations.derive import VERSION_PREFIX, derive, is_derived
from dlp_relations.policy import RelationsPolicy
from dlp_schema.db.repository import get_labels, insert_labels
from dlp_schema.episode import current_labels
from dlp_schema.labels import (
    CoveragePayload,
    Evidence,
    LabelRecord,
    Provenance,
    RelationPayload,
    Source,
    Verification,
)
from dlp_schema.ontology import Ontology


@dataclass
class RelationsSummary:
    version: str
    relations: int = 0
    coverage: dict[str, float] = field(default_factory=dict[str, float])  # "도구→표면" → 비율
    inserted: int = 0
    retracted: int = 0
    kept: int = 0
    skipped_by_review: int = 0


def _base_id(
    session_id: str, payload: RelationPayload | CoveragePayload, start: int, end: int
) -> str:
    content = json.dumps([payload.model_dump(mode="json"), start, end], sort_keys=True)
    tag = "rel" if isinstance(payload, RelationPayload) else "cov"
    return f"{session_id}-{tag}-{hashlib.sha256(content.encode()).hexdigest()[:16]}"


def run_relations(
    conn: sa.Connection,
    session_id: str,
    ontology: Ontology,
    policy: RelationsPolicy,
    now: datetime,
) -> RelationsSummary:
    version = VERSION_PREFIX + policy.digest
    summary = RelationsSummary(version)
    labels = get_labels(conn, session_id)
    derived = derive(labels, ontology, policy)
    summary.relations = len(derived.relations)
    summary.coverage = {
        f"{c.payload.tool_id}→{c.payload.surface_id}": c.payload.ratio for c in derived.coverage
    }

    by_id = {x.label_id: x for x in labels}
    children: dict[str, list[LabelRecord]] = {}
    for x in labels:
        if x.parent_label_id:
            children.setdefault(x.parent_label_id, []).append(x)
    current_ids = {x.label_id for x in current_labels(labels)}
    ontology_version = next((x.ontology_version for x in labels), ontology.version)

    desired: set[str] = set()
    new: list[LabelRecord] = []
    drafts = [(d.payload, d.start_ms, d.end_ms, d.confidence) for d in derived.relations] + [
        (c.payload, c.start_ms, c.end_ms, c.confidence) for c in derived.coverage
    ]
    for payload, start, end, confidence in drafts:
        base = _base_id(session_id, payload, start, end)
        history = sorted(i for i in by_id if i == base or i.startswith(base + "-"))
        live = [i for i in history if i in current_ids]
        if live:
            desired.update(live)
            summary.kept += 1
            continue
        reviewed = any(
            c.provenance.source is Source.HUMAN for i in history for c in children.get(i, [])
        )
        if reviewed:  # 검수자가 고치거나 지웠다
            summary.skipped_by_review += 1
            continue
        label_id = base if not history else f"{base}-{len(history)}"
        desired.add(label_id)
        new.append(
            LabelRecord(
                label_id=label_id,
                session_id=session_id,
                t_start_ms=start,
                t_end_ms=end,
                ontology_version=ontology_version,
                provenance=Provenance(source=Source.MODEL, model_version=version),
                evidence=Evidence.INFERRED,
                confidence=round(confidence, 4),
                created_at=now,
                payload=payload,
            )
        )
    stale = [
        by_id[i]
        for i in sorted(current_ids)
        if i not in desired and is_derived(by_id[i]) and by_id[i].kind in ("relation", "coverage")
    ]
    retractions = [
        x.model_copy(
            update={
                "label_id": f"{x.label_id}:retracted",
                "parent_label_id": x.label_id,
                "retracted": True,
                "verification": Verification(),
                "provenance": Provenance(source=Source.MODEL, model_version=version),
                "created_at": now,
            }
        )
        for x in stale
    ]
    insert_labels(conn, [*new, *retractions])
    summary.inserted, summary.retracted = len(new), len(retractions)
    return summary
