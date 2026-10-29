"""프리라벨 단계 실행 기록(결과 0개 포함, 추가 전용) 저장·조회 (ADR 0015 개정, F-PAR-75).

프리라벨 단계는 다시 돌릴지를 라벨 이력과 이 기록으로 함께 정한다 (`dlp_prelabel.runner`). 결과가
0개인 단계는 라벨 이력에 버전이 남지 않으므로, 이 기록이 없으면 매 실행 원본을 다시 받고 다시 돈다.
"""

from __future__ import annotations

from datetime import UTC, datetime

import sqlalchemy as sa

from dlp_schema.db.locks import lock_session
from dlp_schema.db.tables import prelabel_stage_runs
from dlp_schema.session import PrelabelStageRun


def record_prelabel_stage_run(
    conn: sa.Connection,
    session_id: str,
    stream_id: str,
    stage: str,
    model_version: str,
    produced: int,
    at: datetime,
) -> PrelabelStageRun:
    """단계 실행 기록을 하나 추가한다 (추가만, 라벨과 같은 DB 트랜잭션). 세션 행을 잠근다.

    계약으로 시간대·식별자·결과 수(0 이상)를 검사한 뒤 쓴다. 건너뛴 실행에는 호출하지 않는다.

    Raises:
        pydantic.ValidationError: 시각에 시간대가 없거나 식별자·버전이 잘못됐을 때.
    """
    record = PrelabelStageRun(
        event_id=1,
        session_id=session_id,
        stream_id=stream_id,
        stage=stage,
        model_version=model_version,
        produced=produced,
        at=at,
    )
    lock_session(conn, session_id)
    # SQLite는 시간대 정보를 저장하지 않으므로 UTC 시각을 써야 읽을 때 같은 순간이 된다
    values = record.model_dump(mode="python", exclude={"event_id"})
    values["at"] = record.at.astimezone(UTC)
    event_id = conn.execute(
        prelabel_stage_runs.insert().values(**values).returning(prelabel_stage_runs.c.event_id)
    ).scalar_one()
    return record.model_copy(update={"event_id": int(event_id)})


def list_prelabel_stage_runs(
    conn: sa.Connection,
    session_id: str,
    *,
    stream_id: str | None = None,
    stage: str | None = None,
) -> list[PrelabelStageRun]:
    """세션의 단계 실행 기록 (일련번호 순). stream_id·stage를 주면 그것만. SQLite 시각은 UTC로
    복원한다."""
    t = prelabel_stage_runs.c
    query = sa.select(prelabel_stage_runs).where(t.session_id == session_id)
    if stream_id is not None:
        query = query.where(t.stream_id == stream_id)
    if stage is not None:
        query = query.where(t.stage == stage)
    rows = conn.execute(query.order_by(t.event_id)).mappings()
    return [
        PrelabelStageRun.model_validate(
            {
                **row,
                "at": row["at"].replace(tzinfo=UTC) if row["at"].utcoffset() is None else row["at"],
            }
        )
        for row in rows
    ]
