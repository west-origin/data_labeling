"""청소 전후 3D 비교 산출물 기록(추가 전용 `scene3d_records`) 저장·조회 (ADR 0066).

`dlp scene3d …`가 버킷에 산출물을 쓴 뒤 같은 명령에서 한 줄씩 남긴다. 같은 (종류, 태그, 키)가 이미
있으면 다시 넣지 않는다 (멱등 재실행이 기록을 늘리지 않게).
"""

from __future__ import annotations

from datetime import UTC

import sqlalchemy as sa

from dlp_schema.db.tables import scene3d_records
from dlp_schema.scene3d import Scene3dRecord


def record_scene3d(conn: sa.Connection, record: Scene3dRecord) -> bool:
    """기록을 하나 추가한다. 같은 (세션, 종류, 태그, 키)가 이미 있으면 넣지 않고 False."""
    t = scene3d_records
    exists = conn.execute(
        sa.select(t.c.record_id).where(
            t.c.session_id == record.session_id,
            t.c.kind == record.kind,
            t.c.tag == record.tag,
            t.c.storage_key == record.storage_key,
        )
    ).first()
    if exists is not None:
        return False
    conn.execute(t.insert().values(**record.model_dump()))
    return True


def scene3d_records_for(conn: sa.Connection, session_id: str) -> list[Scene3dRecord]:
    """세션의 기록 (기록 순)."""
    t = scene3d_records
    rows = conn.execute(
        sa.select(t).where(t.c.session_id == session_id).order_by(t.c.record_id)
    ).mappings()
    out: list[Scene3dRecord] = []
    for r in rows:
        data = {k: v for k, v in r.items() if k != "record_id"}
        if data["created_at"].tzinfo is None:  # SQLite는 시간대를 잃는다 (UTC로 저장)
            data["created_at"] = data["created_at"].replace(tzinfo=UTC)
        out.append(Scene3dRecord.model_validate(data))
    return out
