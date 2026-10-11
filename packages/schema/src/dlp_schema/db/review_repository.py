"""검수 도구 작업·운영 배정 저장소. 커밋과 잠금 순서는 호출자가 관리한다.

매체 회수 outbox(ADR 0039): `request_task_revocation`(요청), `mark_task_media_revoked`·
`mark_task_tool_revoked`(완료 표시), `pending_revocations`(대기 조회). 요청은 블러를 바꾼
트랜잭션에서 세션 잠금 아래에, 완료 표시는 실행기(`dlp_review.revoke`)가 확인한 뒤 짧은
트랜잭션에서 쓴다.
"""

from datetime import datetime

import sqlalchemy as sa

from dlp_schema.db.tables import review_assignments, review_tasks
from dlp_schema.review import (
    AssignmentStatus,
    ReviewAssignment,
    ReviewStage,
    ReviewTask,
    ReviewTaskStatus,
)

# review_tasks의 시각 열 (JSON 직렬화하지 않고 datetime 객체로 넣는다)
_TASK_TIMES = (
    "created_at",
    "collected_at",
    "revoke_requested_at",
    "media_revoked_at",
    "tool_revoked_at",
)


def _aware(at: datetime) -> datetime:
    """시간대 없는 시각이면 ValueError (DB 열은 TIMESTAMP WITH TIME ZONE)."""
    if at.tzinfo is None or at.utcoffset() is None:
        raise ValueError(f"시간대 없는 시각은 저장하지 않습니다: {at!r}")
    return at


def insert_review_task(conn: sa.Connection, task: ReviewTask) -> None:
    """검수 작업 1행을 추가한다 (`review_tasks`). 같은 task_key가 있으면 IntegrityError."""
    data = task.model_dump(mode="json")
    # JSON 직렬화는 시각을 문자열로 바꾸므로 datetime 열은 원래 객체로 되돌린다
    for name in _TASK_TIMES:
        data[name] = getattr(task, name)
    conn.execute(review_tasks.insert().values(**data))


def get_review_task(conn: sa.Connection, task_key: str) -> ReviewTask:
    """작업 키로 검수 작업을 읽는다. Raises: sqlalchemy.exc.NoResultFound (없을 때)."""
    row = (
        conn.execute(sa.select(review_tasks).where(review_tasks.c.task_key == task_key))
        .mappings()
        .one()
    )
    return ReviewTask.model_validate(dict(row))


def list_review_tasks(
    conn: sa.Connection, session_id: str, stage: ReviewStage | None = None
) -> list[ReviewTask]:
    """세션의 검수 작업 목록 (만든 순). stage를 주면 그 단계만."""
    query = sa.select(review_tasks).where(review_tasks.c.session_id == session_id)
    if stage is not None:
        query = query.where(review_tasks.c.stage == stage.value)
    rows = conn.execute(query.order_by(review_tasks.c.created_at)).mappings().all()
    return [ReviewTask.model_validate(dict(r)) for r in rows]


def mark_review_task_collected(conn: sa.Connection, task_key: str, at: datetime) -> None:
    """작업을 수집 완료(collected)로 표시하고 collected_at을 기록한다.

    이미 collected여도 다시 덮어쓴다 (시각이 갱신된다). 회수한(revoked) 작업은 수집 표시하지 않는다.
    Raises: KeyError (작업이 없거나 회수한 작업일 때).
    """
    result = conn.execute(
        review_tasks.update()
        .where(
            review_tasks.c.task_key == task_key,
            review_tasks.c.status != ReviewTaskStatus.REVOKED.value,
        )
        .values(status=ReviewTaskStatus.COLLECTED.value, collected_at=at)
    )
    if result.rowcount != 1:
        raise KeyError(task_key)


def request_task_revocation(
    conn: sa.Connection, task_key: str, *, reason: str, at: datetime
) -> bool:
    """작업 매체 회수를 요청한다 (outbox). 열린 작업은 revoked로, 수집한 작업은 상태를 둔다.

    아직 요청이 없는 비프라이버시 작업만 바꾼다 (블러 검수 작업은 원본 권한자 화면이라 회수하지
    않는다. 계약 검증기와 이 SQL 양쪽에서 막는다).

    Returns:
        이번에 요청을 남겼으면 True (이미 요청했거나 블러 검수 작업·없는 작업이면 False).

    Raises:
        ValueError: at에 시간대가 없거나 사유가 비었을 때.
    """
    if not reason:
        raise ValueError("회수 요청에는 사유가 있어야 합니다")
    result = conn.execute(
        review_tasks.update()
        .where(
            review_tasks.c.task_key == task_key,
            review_tasks.c.revoke_requested_at.is_(None),
            review_tasks.c.stage != ReviewStage.PRIVACY.value,
        )
        .values(
            revoke_requested_at=_aware(at),
            revoke_reason=reason,
            status=sa.case(
                (
                    review_tasks.c.status == ReviewTaskStatus.OPEN.value,
                    ReviewTaskStatus.REVOKED.value,
                ),
                else_=review_tasks.c.status,
            ),
        )
    )
    return result.rowcount == 1


def mark_task_media_revoked(
    conn: sa.Connection, session_id: str, media_uri: str, at: datetime
) -> int:
    """같은 세션·매체 URI를 쓰는 회수 요청 작업에 매체 삭제 확인 시각을 남긴다. 표시한 행 수.

    회수 단위는 매체 객체다 (같은 키를 쓰는 작업은 함께 지워진다).
    요청이 없는 작업은 표시하지 않는다.
    """
    result = conn.execute(
        review_tasks.update()
        .where(
            review_tasks.c.session_id == session_id,
            review_tasks.c.media_uri == media_uri,
            review_tasks.c.revoke_requested_at.is_not(None),
            review_tasks.c.media_revoked_at.is_(None),
        )
        .values(media_revoked_at=_aware(at))
    )
    return result.rowcount


def mark_task_tool_revoked(
    conn: sa.Connection, task_key: str, at: datetime, note: str | None = None
) -> bool:
    """회수 요청 작업에 도구 작업 없음 확인 시각(과 수동 완료 사유)을 남긴다. 표시했으면 True."""
    values: dict[str, object] = {"tool_revoked_at": _aware(at)}
    if note is not None:
        values["revoke_note"] = note
    result = conn.execute(
        review_tasks.update()
        .where(
            review_tasks.c.task_key == task_key,
            review_tasks.c.revoke_requested_at.is_not(None),
            review_tasks.c.tool_revoked_at.is_(None),
        )
        .values(**values)
    )
    return result.rowcount == 1


def pending_revocations(conn: sa.Connection, session_id: str | None = None) -> list[ReviewTask]:
    """회수 대기 작업: 요청이 있고 매체 또는 도구 완료 표시가 비었다 (요청 순, 같으면 키 순).

    session_id가 None이면 모든 세션 (`dlp review revoke --pending`).
    """
    query = sa.select(review_tasks).where(
        review_tasks.c.revoke_requested_at.is_not(None),
        sa.or_(review_tasks.c.media_revoked_at.is_(None), review_tasks.c.tool_revoked_at.is_(None)),
    )
    if session_id is not None:
        query = query.where(review_tasks.c.session_id == session_id)
    query = query.order_by(review_tasks.c.revoke_requested_at, review_tasks.c.task_key)
    return [ReviewTask.model_validate(dict(r)) for r in conn.execute(query).mappings()]


def insert_assignment(conn: sa.Connection, a: ReviewAssignment) -> None:
    """검수 배정 1행을 추가한다 (`review_assignments`). 같은 ID가 있으면 IntegrityError."""
    data = a.model_dump(mode="json")
    # datetime 열은 JSON 문자열이 아니라 원래 객체로 넣는다
    data["created_at"], data["completed_at"] = a.created_at, a.completed_at
    conn.execute(review_assignments.insert().values(**data))


def get_assignment(conn: sa.Connection, assignment_id: str) -> ReviewAssignment:
    """배정 ID로 배정을 읽는다. Raises: sqlalchemy.exc.NoResultFound (없을 때)."""
    row = (
        conn.execute(
            sa.select(review_assignments).where(review_assignments.c.assignment_id == assignment_id)
        )
        .mappings()
        .one()
    )
    return ReviewAssignment.model_validate(dict(row))


def list_assignments(
    conn: sa.Connection, session_id: str | None = None, status: AssignmentStatus | None = None
) -> list[ReviewAssignment]:
    """우선순위 높은 순 (같으면 만든 순).

    Args:
        session_id: 주면 그 세션만.
        status: 주면 그 상태만 (예: open = 검수 대기열).

    Returns:
        (priority 내림차순, created_at, assignment_id) 순 목록. 마지막 키는 결정적 순서를 위한 것.
    """
    query = sa.select(review_assignments)
    if session_id is not None:
        query = query.where(review_assignments.c.session_id == session_id)
    if status is not None:
        query = query.where(review_assignments.c.status == status.value)
    query = query.order_by(
        review_assignments.c.priority.desc(),
        review_assignments.c.created_at,
        review_assignments.c.assignment_id,
    )
    return [ReviewAssignment.model_validate(dict(r)) for r in conn.execute(query).mappings()]


def update_assignment(conn: sa.Connection, assignment_id: str, **values: object) -> None:
    """배정 상태·담당자·작업 키만 바꾼다.

    Args:
        assignment_id: 배정 ID.
        **values: assignee, task_key, status(`AssignmentStatus` 또는 문자열), completed_at 중 일부.

    Raises:
        ValueError: 허용되지 않은 필드를 바꾸려 할 때.
        KeyError: 배정이 없을 때.
    """
    allowed = {"assignee", "task_key", "status", "completed_at"}
    if not set(values) <= allowed:
        raise ValueError(f"바꿀 수 없는 필드: {set(values) - allowed}")
    data = {k: (v.value if isinstance(v, AssignmentStatus) else v) for k, v in values.items()}
    result = conn.execute(
        review_assignments.update()
        .where(review_assignments.c.assignment_id == assignment_id)
        .values(**data)
    )
    if result.rowcount != 1:
        raise KeyError(assignment_id)
