"""세션·모델 배포의 트랜잭션 잠금. 연결과 커밋은 호출자가 관리한다."""

import hashlib

import sqlalchemy as sa

from dlp_schema.db.tables import sessions


def lock_session(conn: sa.Connection, session_id: str) -> None:
    """세션 행을 FOR UPDATE로 잠근다 (트랜잭션 끝까지, PostgreSQL).

    탐지·승인·검수 수집은 이력 읽기 전에 호출한다. 라벨 추가도 같은 행을 잠가 내보내기의
    최종 FOR SHARE 검사·게시와 충돌하게 한다. 여러 세션을 잠글 때는 ID 순서를 지킨다.
    """
    conn.execute(
        sa.select(sessions.c.session_id)
        .where(sessions.c.session_id == session_id)
        .with_for_update()
    ).scalar_one()


def try_lock_session(conn: sa.Connection, session_id: str) -> bool:
    """세션 행을 FOR UPDATE SKIP LOCKED로 잠근다. 잠갔으면 True (트랜잭션 끝까지).

    다른 트랜잭션이 이미 잡고 있거나(예: 검수 작업 생성 중) 세션이 없으면 False이고 기다리지 않는다.
    고아 정리(`dlp_review.orphans`)가 작업 생성과 겹치지 않게 쓴다. SQLite는 행 잠금이 없어 세션이
    있으면 늘 True다.
    """
    row = conn.execute(
        sa.select(sessions.c.session_id)
        .where(sessions.c.session_id == session_id)
        .with_for_update(skip_locked=True)
    ).first()
    return row is not None


def lock_model_task(conn: sa.Connection, task: str) -> None:
    """과제별 배포를 PostgreSQL 트랜잭션 advisory lock으로 직렬화한다 (첫 배포 포함).

    SQLite는 배포 유일성 색인과 쓰기 잠금으로 중복을 거부한다. 키는 모든 프로세스에서
    같은 SHA-256 앞 64비트이며 다른 용도의 잠금과 구분하는 접두사를 포함한다.
    """
    if conn.dialect.name == "postgresql":
        key = int.from_bytes(
            hashlib.sha256(f"dlp:deploy:{task}".encode()).digest()[:8], signed=True
        )
        conn.execute(sa.select(sa.func.pg_advisory_xact_lock(key)))
