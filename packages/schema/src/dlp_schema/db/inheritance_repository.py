"""설명의 검수 물려받기 기록 저장·조회 (`description_inheritance`, 추가만, ADR 0043 3절).

쓰기는 설명 따르기 적용(`dlp_schema.db.description_repository.apply_description_follow`)이 물려받는
`record_review`와 같은 트랜잭션에서만 한다. 읽기는 설명 따르기 계획과 저장 검증의 좁은 예외
(`label_validation._FollowExemption`)가 쓴다.
"""

from __future__ import annotations

from datetime import UTC
from typing import Any

import sqlalchemy as sa

from dlp_schema.db.tables import description_inheritance
from dlp_schema.review import DescriptionInheritance


def _utc(row: Any, name: str) -> Any:
    """SQLite는 시간대를 저장하지 않는다 (UTC로 복원)."""
    t = row[name]
    return t.replace(tzinfo=UTC) if t.utcoffset() is None else t


def record_description_inheritance(
    conn: sa.Connection, record: DescriptionInheritance
) -> DescriptionInheritance:
    """기록 하나를 추가한다 (event_id는 DB가 매긴다). 호출자가 세션을 잠근 트랜잭션에서 부른다.

    Returns:
        DB 일련번호를 채운 기록.
    """
    values = record.model_dump(mode="python", exclude={"event_id"})
    values["owner_label_ids"] = list(record.owner_label_ids)
    values["state"] = record.state.value
    values["mechanism"] = record.mechanism.value
    values["reviewed_at"] = record.reviewed_at.astimezone(UTC)
    values["recorded_at"] = record.recorded_at.astimezone(UTC)
    event_id = conn.execute(
        description_inheritance.insert()
        .values(**values)
        .returning(description_inheritance.c.event_id)
    ).scalar_one()
    return record.model_copy(update={"event_id": int(event_id)})


def get_description_inheritance(
    conn: sa.Connection, session_id: str
) -> list[DescriptionInheritance]:
    """세션의 물려받기 기록 (일련번호 순)."""
    rows = conn.execute(
        sa.select(description_inheritance)
        .where(description_inheritance.c.session_id == session_id)
        .order_by(description_inheritance.c.event_id)
    ).mappings()
    return [
        DescriptionInheritance.model_validate(
            {
                **r,
                "owner_label_ids": tuple(r["owner_label_ids"]),
                "reviewed_at": _utc(r, "reviewed_at"),
                "recorded_at": _utc(r, "recorded_at"),
            }
        )
        for r in rows
    ]
