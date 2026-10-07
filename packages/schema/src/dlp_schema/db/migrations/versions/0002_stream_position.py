"""keep stream order within a session

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-07
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | None = None
depends_on: str | None = None

# 0002 streams.position 추가: 세션 안 스트림 순서를 보존한다 (get_session이 이 순서로 복원).
# 1) nullable로 열 추가 → 2) 기존 행을 stream_id 알파벳 순으로 0부터 번호 매김 → 3) NOT NULL로 바꿈.
# UPDATE ... FROM 구문은 PostgreSQL과 SQLite(3.33+) 모두 지원한다.


def upgrade() -> None:
    """streams.position 열을 추가하고 기존 행을 채운 뒤 NOT NULL로 바꾼다."""
    # 기존 행은 stream_id 순서로 위치를 매긴다
    op.add_column(
        "streams",
        sa.Column("position", sa.Integer(), nullable=True, comment="세션 안 스트림 순서"),
    )
    op.execute(
        "UPDATE streams SET position = ranked.pos FROM ("
        " SELECT session_id, stream_id,"
        " ROW_NUMBER() OVER (PARTITION BY session_id ORDER BY stream_id) - 1 AS pos"
        " FROM streams) AS ranked"
        " WHERE streams.session_id = ranked.session_id AND streams.stream_id = ranked.stream_id"
    )
    # SQLite는 ALTER COLUMN이 없어 batch(표 다시 만들기)로 NOT NULL을 건다
    with op.batch_alter_table("streams") as batch:
        batch.alter_column("position", existing_type=sa.Integer(), nullable=False)


def downgrade() -> None:
    """streams.position 열을 지운다 (순서 정보가 사라진다)."""
    with op.batch_alter_table("streams") as batch:
        batch.drop_column("position")
