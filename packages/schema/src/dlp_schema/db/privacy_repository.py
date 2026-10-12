"""추가 전용 탐지 완료 이력(ADR 0032·0033)과 승인 보류 기록(ADR 0042) 저장·조회."""

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import sqlalchemy as sa

from dlp_schema.db.locks import lock_session
from dlp_schema.db.tables import privacy_detection_runs, privacy_holds
from dlp_schema.session import HoldAction, PrivacyDetectionRun, PrivacyHoldEvent


def record_privacy_detection(
    conn: sa.Connection, session_id: str, stream_id: str, versions: Sequence[str], at: datetime
) -> int:
    """완료한 탐지 버전들을 추가한다 (0건 포함, 라벨과 같은 DB 트랜잭션).

    시간대와 버전 목록을 계약으로 검사한 뒤 쓴다. 멱등 실행을 건너뛴 경우에는 호출하지 않는다.
    """
    record = PrivacyDetectionRun(
        event_id=1,
        session_id=session_id,
        stream_id=stream_id,
        model_versions=tuple(sorted(set(versions))),
        at=at,
    )
    lock_session(conn, session_id)
    # SQLite는 시간대 정보를 저장하지 않으므로 UTC 시각을 써야 읽을 때 같은 순간이 된다.
    values = record.model_dump(mode="python", exclude={"event_id"})
    values["at"] = record.at.astimezone(UTC)
    return conn.execute(
        privacy_detection_runs.insert()
        .values(**values)
        .returning(privacy_detection_runs.c.event_id)
    ).scalar_one()


def list_privacy_detections(conn: sa.Connection, session_id: str) -> list[PrivacyDetectionRun]:
    """세션 탐지 완료 기록을 일련번호 순으로 읽는다. SQLite 시각은 UTC로 복원한다."""
    rows = conn.execute(
        sa.select(privacy_detection_runs)
        .where(privacy_detection_runs.c.session_id == session_id)
        .order_by(privacy_detection_runs.c.event_id)
    ).mappings()
    return [
        PrivacyDetectionRun.model_validate(
            {
                **row,
                "at": row["at"].replace(tzinfo=UTC) if row["at"].utcoffset() is None else row["at"],
            }
        )
        for row in rows
    ]


def latest_privacy_detection(
    conn: sa.Connection, session_id: str, stream_id: str
) -> PrivacyDetectionRun | None:
    """스트림의 마지막 완료 세대. 잠금 아래 추가한 일련번호 순으로 고른다 (시계 역행 무관)."""
    row = (
        conn.execute(
            sa.select(privacy_detection_runs)
            .where(
                privacy_detection_runs.c.session_id == session_id,
                privacy_detection_runs.c.stream_id == stream_id,
            )
            .order_by(privacy_detection_runs.c.event_id.desc())
            .limit(1)
        )
        .mappings()
        .one_or_none()
    )
    if row is None:
        return None
    return PrivacyDetectionRun.model_validate(
        {
            **row,
            "at": row["at"].replace(tzinfo=UTC) if row["at"].utcoffset() is None else row["at"],
        }
    )


def _hold(row: Any) -> PrivacyHoldEvent:
    """privacy_holds 행 → 계약. SQLite 시각은 UTC로 복원한다."""
    at = row["at"]
    return PrivacyHoldEvent.model_validate(
        {**row, "at": at.replace(tzinfo=UTC) if at.utcoffset() is None else at}
    )


def record_privacy_hold(
    conn: sa.Connection,
    session_id: str,
    action: HoldAction,
    *,
    reason: str,
    actor: str,
    at: datetime,
) -> PrivacyHoldEvent:
    """승인 보류·해제 기록을 하나 추가한다 (추가만, ADR 0042). 세션 행을 잠근다.

    상태 전이(보류 중에만 해제 등)는 호출자(`dlp_privacy.hold`)가 확인한다. 계약 검증(시간대·
    사유·행위자 비지 않음)을 거친 뒤 쓴다.

    Raises:
        pydantic.ValidationError: 시각에 시간대가 없거나 사유·행위자가 비었을 때.
    """
    record = PrivacyHoldEvent(
        event_id=1, session_id=session_id, action=action, reason=reason, actor=actor, at=at
    )
    lock_session(conn, session_id)
    values = record.model_dump(mode="python", exclude={"event_id"})
    values["at"] = record.at.astimezone(UTC)
    event_id = conn.execute(
        privacy_holds.insert().values(**values).returning(privacy_holds.c.event_id)
    ).scalar_one()
    return record.model_copy(update={"event_id": int(event_id)})


def list_privacy_holds(conn: sa.Connection, session_id: str) -> list[PrivacyHoldEvent]:
    """세션의 보류·해제 기록 (일련번호 순)."""
    rows = conn.execute(
        sa.select(privacy_holds)
        .where(privacy_holds.c.session_id == session_id)
        .order_by(privacy_holds.c.event_id)
    ).mappings()
    return [_hold(r) for r in rows]


def active_privacy_hold(conn: sa.Connection, session_id: str) -> PrivacyHoldEvent | None:
    """지금 유효한 보류 기록: 마지막 기록(일련번호 순)이 hold면 그것, 아니면 None."""
    row = (
        conn.execute(
            sa.select(privacy_holds)
            .where(privacy_holds.c.session_id == session_id)
            .order_by(privacy_holds.c.event_id.desc())
            .limit(1)
        )
        .mappings()
        .one_or_none()
    )
    if row is None or row["action"] != "hold":
        return None
    return _hold(row)
