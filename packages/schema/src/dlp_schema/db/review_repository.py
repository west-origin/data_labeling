"""검수 도구 작업·운영 배정 저장소. 커밋과 잠금 순서는 호출자가 관리한다."""

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


def insert_review_task(conn: sa.Connection, task: ReviewTask) -> None:
    """검수 작업 1행을 추가한다 (`review_tasks`). 같은 task_key가 있으면 IntegrityError."""
    data = task.model_dump(mode="json")
    # JSON 직렬화는 시각을 문자열로 바꾸므로 datetime 열은 원래 객체로 되돌린다
    data["created_at"], data["collected_at"] = task.created_at, task.collected_at
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

    이미 collected여도 다시 덮어쓴다 (시각이 갱신된다). Raises: KeyError (작업이 없을 때).
    """
    result = conn.execute(
        review_tasks.update()
        .where(review_tasks.c.task_key == task_key)
        .values(status=ReviewTaskStatus.COLLECTED.value, collected_at=at)
    )
    if result.rowcount != 1:
        raise KeyError(task_key)


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
