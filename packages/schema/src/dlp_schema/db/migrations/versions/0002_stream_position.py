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


def upgrade() -> None:
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
    with op.batch_alter_table("streams") as batch:
        batch.alter_column("position", existing_type=sa.Integer(), nullable=False)


def downgrade() -> None:
    with op.batch_alter_table("streams") as batch:
        batch.drop_column("position")
