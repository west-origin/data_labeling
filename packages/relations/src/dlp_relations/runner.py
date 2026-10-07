"""세션 관계·커버리지 도출 실행 (`dlp relations run`). 멱등이다.

관계·커버리지 레코드의 model_version은 "relations-<정책 해시>"다. 다시 실행하면:
- 내용(구간·필드)이 같은 현재 레코드는 그대로 둔다 (정책이 바뀌어도).
- 더 이상 나오지 않는 이 모듈의 현재 레코드는 삭제 레코드(retracted, parent=원래)로 표시한다.
- 새로 나온 것만 넣는다. 사람이 고치거나 지운 적이 있는 내용은 다시 넣지 않는다.

라벨 ID: `<세션>-<rel|cov>-<내용 해시 16자>`. 같은 내용이 지워진 뒤 다시 나오면 `-<n>` 접미사를
붙인다. 검수자가 승인·표본 검증한 레코드는 규칙이 바뀌어도 지우지 않는다 (ADR 0015).
부작용: DB `labels`에 새 레코드와 삭제 레코드만 추가한다. 호출자가 트랜잭션을 연다.
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
    VerificationState,
)
from dlp_schema.ontology import Ontology


@dataclass
class RelationsSummary:
    """`run_relations` 결과 요약.

    version: 이번 실행의 model_version. relations: 도출한 관계 초안 수. coverage: "도구→표면" →
    비율. inserted / retracted / kept: 새로 넣은 / 삭제 표시한 / 그대로 둔 레코드 수.
    skipped_by_review: 검수자가 고치거나 지워서 다시 넣지 않은 초안 수.
    """

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
    """내용(페이로드 + 구간) 해시로 만든 기본 라벨 ID `<세션>-<rel|cov>-<sha256 16자>`.

    같은 내용이면 정책이 바뀌어도 같은 ID라 다시 넣지 않는다.
    """
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
    """세션의 관계·커버리지를 다시 도출해 DB와 맞춘다.

    Args:
        conn: DB 연결 (호출자가 트랜잭션을 연다). session_id: 세션.
        ontology, policy: 도출 입력. now: 새 레코드 created_at (시간대 필수).

    Returns:
        `RelationsSummary`.
    """
    version = VERSION_PREFIX + policy.digest
    summary = RelationsSummary(version)
    labels = get_labels(conn, session_id)
    derived = derive(labels, ontology, policy)
    summary.relations = len(derived.relations)
    summary.coverage = {
        f"{c.payload.tool_id}→{c.payload.surface_id}": c.payload.ratio for c in derived.coverage
    }

    by_id = {x.label_id: x for x in labels}
    # parent → 자식 레코드 (검수자가 고치거나 지웠는지 판단용)
    children: dict[str, list[LabelRecord]] = {}
    for x in labels:
        if x.parent_label_id:
            children.setdefault(x.parent_label_id, []).append(x)
    current_ids = {x.label_id for x in current_labels(labels)}
    # 세션 라벨의 온톨로지 버전 (첫 라벨 기준, 라벨이 없으면 넘겨받은 온톨로지)
    ontology_version = next((x.ontology_version for x in labels), ontology.version)

    desired: set[str] = set()
    new: list[LabelRecord] = []
    drafts = [(d.payload, d.start_ms, d.end_ms, d.confidence) for d in derived.relations] + [
        (c.payload, c.start_ms, c.end_ms, c.confidence) for c in derived.coverage
    ]
    seen: set[str] = set()
    for payload, start, end, confidence in drafts:
        base = _base_id(session_id, payload, start, end)
        # 내용이 같은 초안은 하나만 (같은 ID로 두 번 넣지 않는다)
        if base in seen:
            continue
        seen.add(base)
        # 같은 내용의 과거 레코드 ID들 (기본 ID와 `-n` 재삽입본, 그 자식 일부)
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
        # 같은 내용이 지워진 적이 있으면 새 ID로 다시 넣는다 (기존 ID는 이력에 남아 재사용 불가)
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
        if i not in desired
        and is_derived(by_id[i])
        and by_id[i].kind in ("relation", "coverage")
        # 검수자가 승인·표본 검증한 레코드는 규칙이 바뀌어도 지우지 않는다 (ADR 0015)
        and by_id[i].verification.state is VerificationState.UNREVIEWED
    ]
    # 삭제 레코드: ID `<원래>:retracted`, parent=원래, 출처는 이번 실행 버전.
    # (dlp_schema.episode.retractions와 같은 모양이지만 128자 초과 ID 줄이기는 하지 않는다)
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
