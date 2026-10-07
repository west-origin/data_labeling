"""검수 결과를 라벨 이력으로 바꾼다.

검수 도구에서 돌아온 항목(ReviewedItem)을 검수에 보낸 원래 라벨과 비교한다.
- 바뀌지 않음: 원래 라벨의 검수 상태만 human_approved로 갱신한다.
- 바뀜: 원래 라벨을 parent로 하는 새 human 레코드 (human_corrected).
- 없어짐: 원래 라벨을 지우는 retracted 레코드 (human_corrected).
- 새로 생김: parent 없는 새 human 레코드 (human_corrected).
새 레코드 ID는 (원래 ID, 검수자, 시각, 내용)의 해시라 같은 검수 결과를 다시 수집해도 같다.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from dlp_schema.labels import (
    LabelPayload,
    LabelRecord,
    Provenance,
    Source,
    Verification,
    VerificationState,
)


@dataclass(frozen=True)
class ReviewedItem:
    """검수 도구에서 돌아온 라벨 하나. origin_label_id가 없으면 검수자가 새로 만든 것이다."""

    origin_label_id: str | None
    stream_id: str | None
    t_start_ms: int
    t_end_ms: int
    payload: LabelPayload


@dataclass
class ReviewOutcome:
    approved: list[str] = field(default_factory=list[str])
    new_records: list[LabelRecord] = field(default_factory=list[LabelRecord])

    @property
    def corrected(self) -> int:
        return sum(1 for r in self.new_records if r.parent_label_id and not r.retracted)

    @property
    def retracted(self) -> int:
        return sum(1 for r in self.new_records if r.retracted)

    @property
    def added(self) -> int:
        return sum(1 for r in self.new_records if r.parent_label_id is None)


def _same(a: LabelRecord, b: ReviewedItem) -> bool:
    """도구로 보낸 형태(normalize 적용 후)와 돌아온 항목이 같은가."""
    return (
        a.payload == b.payload
        and a.t_start_ms == b.t_start_ms
        and a.t_end_ms == b.t_end_ms
        and a.stream_id == b.stream_id
    )


def _new_id(base: str, reviewer: str, now: datetime, content: str) -> str:
    digest = hashlib.sha256(f"{base}|{reviewer}|{now.isoformat()}|{content}".encode()).hexdigest()
    return f"{base}:r{digest[:12]}"[-128:]


def reconcile(
    originals: list[LabelRecord],
    reviewed: list[ReviewedItem],
    *,
    session_id: str,
    ontology_version: str,
    reviewer_id: str,
    now: datetime,
    normalize: Callable[[LabelRecord], LabelRecord] | None = None,
) -> ReviewOutcome:
    """normalize: 도구로 보낼 때 적용한 변환 (예: CVAT 좌표 반올림). 비교는 그 결과와 한다."""
    by_id = {x.label_id: x for x in originals}
    sent = {x.label_id: (normalize(x) if normalize else x) for x in originals}
    verification = Verification(
        state=VerificationState.HUMAN_CORRECTED, reviewer_id=reviewer_id, reviewed_at=now
    )
    outcome = ReviewOutcome()
    seen: set[str] = set()

    def record(
        item: ReviewedItem, parent: LabelRecord | None, *, retracted: bool = False
    ) -> LabelRecord:
        content = json.dumps(
            [
                item.t_start_ms,
                item.t_end_ms,
                item.stream_id,
                item.payload.model_dump(mode="json"),
                retracted,
            ],
            sort_keys=True,
        )
        base = parent.label_id if parent else f"{session_id}-h"
        return LabelRecord(
            label_id=_new_id(base, reviewer_id, now, content),
            session_id=session_id,
            stream_id=item.stream_id,
            t_start_ms=item.t_start_ms,
            t_end_ms=item.t_end_ms,
            ontology_version=parent.ontology_version if parent else ontology_version,
            provenance=Provenance(source=Source.HUMAN),
            verification=verification,
            parent_label_id=parent.label_id if parent else None,
            retracted=retracted,
            created_at=now,
            payload=item.payload,
        )

    for item in reviewed:
        origin = by_id.get(item.origin_label_id or "")
        if origin is None:
            outcome.new_records.append(record(item, None))
            continue
        if origin.label_id in seen:
            raise ValueError(f"검수 결과에 같은 원래 라벨이 두 번 나옵니다: {origin.label_id}")
        seen.add(origin.label_id)
        if _same(sent[origin.label_id], item):
            outcome.approved.append(origin.label_id)
        else:
            outcome.new_records.append(record(item, origin))
    for origin in originals:
        if origin.label_id not in seen:
            gone = ReviewedItem(
                origin.label_id,
                origin.stream_id,
                origin.t_start_ms,
                origin.t_end_ms,
                origin.payload,
            )
            outcome.new_records.append(record(gone, origin, retracted=True))
    return outcome
