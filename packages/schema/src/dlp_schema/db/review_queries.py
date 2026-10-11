"""검수 계획에 필요한 집계. 라벨 이력·기하 페이로드 전체를 파이썬으로 가져오지 않는다.

현재 라벨 판정은 episode.current_labels와 동일하다. 오류 삽입·측정 후손을 재귀 제외하고,
수정·철회된 부모는 제외한다. 연결·트랜잭션은 호출자가 관리한다.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from dlp_schema.db.tables import label_records, review_assignments
from dlp_schema.labels import Source, VerificationState
from dlp_schema.review import AssignmentStatus


def verified_object_classes(conn: sa.Connection, exclude_session: str) -> set[str]:
    """다른 세션의 현재 운영 박스·마스크 중 사람이 확인한 클래스 (쿼리 1회).

    종류를 먼저 고르는 기존 current_labels 경로와 같은 범위로 부모·자식 이력을 판단한다.
    표본 검증은 직접 확인이 아니므로 넣지 않는다.
    """
    labels = (
        sa.select(label_records)
        .where(
            label_records.c.session_id != exclude_session,
            label_records.c.kind.in_(["box_track", "mask_track"]),
        )
        .cte("object_history")
    )
    excluded = (
        sa.select(labels.c.label_id, labels.c.session_id)
        .where(sa.or_(labels.c.seeded_error.is_(True), labels.c.measurement.is_not(None)))
        .cte("non_operational", recursive=True)
    )
    # UNION은 중복을 제거해 잘못된 순환 참조에서도 종료한다.
    excluded = excluded.union(
        sa.select(labels.c.label_id, labels.c.session_id).join(
            excluded,
            sa.and_(
                labels.c.parent_label_id == excluded.c.label_id,
                labels.c.session_id == excluded.c.session_id,
            ),
        )
    )
    child = labels.alias("child")
    query = (
        sa.select(labels.c.payload["class_id"].as_string())
        .distinct()
        .where(
            labels.c.retracted.is_(False),
            labels.c.label_id.not_in(sa.select(excluded.c.label_id)),
            ~sa.exists().where(
                child.c.parent_label_id == labels.c.label_id,
                child.c.session_id == labels.c.session_id,
            ),
            sa.or_(
                labels.c.source == Source.HUMAN.value,
                labels.c.verification_state.in_(
                    [
                        VerificationState.HUMAN_APPROVED.value,
                        VerificationState.HUMAN_CORRECTED.value,
                    ]
                ),
            ),
        )
    )
    return {value for value in conn.execute(query).scalars() if value is not None}


def open_assignment_counts(conn: sa.Connection, reviewers: Sequence[str]) -> dict[str, int]:
    """후보 담당자의 열린 배정 수만 집계한다 (쿼리 1회, 전체 배정 계약 역직렬화 없음)."""
    query = (
        sa.select(review_assignments.c.assignee, sa.func.count())
        .where(
            review_assignments.c.status == AssignmentStatus.OPEN.value,
            review_assignments.c.assignee.in_(reviewers),
        )
        .group_by(review_assignments.c.assignee)
    )
    return {assignee: count for assignee, count in conn.execute(query)}


def existing_assignment_ids(conn: sa.Connection, ids: Sequence[str]) -> set[str]:
    """생성할 배정 ID의 존재만 읽는다. 오류 삽입 원천이 다른 세션인 경우도 포함한다."""
    if not ids:
        return set()
    return set(
        conn.execute(
            sa.select(review_assignments.c.assignment_id).where(
                review_assignments.c.assignment_id.in_(ids)
            )
        ).scalars()
    )
