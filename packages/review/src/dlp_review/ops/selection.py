"""배정 방식에 따라 검수 작업에 넣을 라벨을 고른다.

작업을 만들 때와 결과를 수집할 때 같은 함수를 쓴다. 수집 때 다르게 고르면 보내지 않은 라벨을
검수자가 지운 것으로 잘못 본다.
"""

from __future__ import annotations

import sqlalchemy as sa

from dlp_review.ops.seeding import seed_prefix
from dlp_review.tasks import Selector, current_for_review
from dlp_schema.db.repository import get_labels
from dlp_schema.episode import current_labels
from dlp_schema.labels import LabelRecord
from dlp_schema.review import ReviewAssignment, ReviewMode


def assignment_selector(conn: sa.Connection, a: ReviewAssignment) -> Selector:
    def standard(stream_id: str | None, kinds: tuple[str, ...]) -> list[LabelRecord]:
        labels = current_for_review(conn, a.session_id, stream_id, kinds)
        if a.only_label_ids:
            keep = set(a.only_label_ids)
            return [x for x in labels if x.label_id in keep]
        withheld = set(a.withheld_label_ids)
        return [x for x in labels if x.label_id not in withheld]

    def blind(stream_id: str | None, kinds: tuple[str, ...]) -> list[LabelRecord]:
        return []

    def seeded(stream_id: str | None, kinds: tuple[str, ...]) -> list[LabelRecord]:
        prefix = seed_prefix(a.assignment_id)
        labels = current_labels(
            get_labels(conn, a.session_id, kinds=list(kinds)), operational=False
        )
        return [
            x
            for x in labels
            if x.label_id.startswith(prefix)
            and (stream_id is None or x.stream_id in (stream_id, None))
        ]

    if a.mode is ReviewMode.BLIND:
        return blind
    if a.mode is ReviewMode.SEEDED_ERROR:
        return seeded
    return standard
