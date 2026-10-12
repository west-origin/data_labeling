"""세션·모델 배포·AI 판정 실행 잠금. 연결과 커밋은 호출자가 관리한다.

- `lock_session`: 세션 행 FOR UPDATE (트랜잭션 잠금).
- `lock_model_task`: 과제별 배포 직렬화 (트랜잭션 advisory lock).
- `try_lock_aireview`·`unlock_aireview`: 세션별 AI 판정 실행기 하나 (세션 수준 advisory lock,
  WP17.4).
"""

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


def _aireview_key(session_id: str) -> int:
    """AI 판정 실행 잠금 키 (다른 잠금과 구분하는 접두사 `dlp:aireview:` + SHA-256 앞 64비트)."""
    return int.from_bytes(
        hashlib.sha256(f"dlp:aireview:{session_id}".encode()).digest()[:8], signed=True
    )


def try_lock_aireview(conn: sa.Connection, session_id: str) -> bool:
    """세션의 AI 판정 실행 잠금을 시도한다 (PostgreSQL 세션 수준 advisory lock, 기다리지 않음).

    같은 세션에는 실행기 하나만 돌게 한다 (WP17.5, ADR 0040). 세션 수준 잠금이라 트랜잭션을
    커밋해도 풀리지 않고, 이 연결이 `unlock_aireview`를 부르거나 DB 연결이 끊길 때까지 유지된다.
    주의: SQLAlchemy `Connection.close()`는 DB 연결을 풀(pool)로 돌려보낼 뿐 끊지 않으므로 잠금이
    남는다. 그래서 실행 내내 잠금 전용 연결을 쥐고 `finally`에서 반드시 `unlock_aireview`를 부른다
    (프로세스가 죽으면 DB 연결이 끊겨 풀린다). 세션 행 잠금(`lock_session`)은 쓰지 않는다: 몇
    시간짜리 판정 실행이 수집·라벨 추가를 막지 않게 하려는 것이다. 잠금이 뚫려도 단위 기록은
    `ON CONFLICT DO NOTHING`이라 중복 행이 생기지 않는다.

    SQLite(테스트)는 잠금이 없어 늘 True다.

    Returns:
        잠금을 얻었으면 True, 다른 연결이 쥐고 있으면 False.
    """
    if conn.dialect.name != "postgresql":
        return True
    got = conn.execute(sa.select(sa.func.pg_try_advisory_lock(_aireview_key(session_id))))
    return bool(got.scalar_one())


def unlock_aireview(conn: sa.Connection, session_id: str) -> None:
    """`try_lock_aireview`로 얻은 잠금을 푼다 (같은 연결에서). 쥐고 있지 않으면 아무 일도 없다."""
    if conn.dialect.name == "postgresql":
        conn.execute(sa.select(sa.func.pg_advisory_unlock(_aireview_key(session_id))))
