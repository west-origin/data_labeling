"""audit 4: append-only session lifecycle events (backfilled with each session's current state)

Revision ID: 0011
Revises: 0010
Create Date: 2026-10-07
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0011"
down_revision: str | None = "0010"
branch_labels: str | None = None
depends_on: str | None = None

TABLE = "session_lifecycle_events"


def upgrade() -> None:
    op.create_table(
        TABLE,
        sa.Column(
            "event_id",
            sa.BigInteger().with_variant(sa.Integer(), "sqlite"),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column("session_id", sa.String(length=128), nullable=False),
        sa.Column("from_state", sa.String(length=32), nullable=True),
        sa.Column("to_state", sa.String(length=32), nullable=False),
        sa.Column("at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("actor", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["sessions.session_id"],
            name=op.f("fk_session_lifecycle_events_session_id_sessions"),
        ),
        sa.PrimaryKeyConstraint("event_id", name=op.f("pk_session_lifecycle_events")),
    )
    op.create_index(
        op.f("ix_session_lifecycle_events_session_id"), TABLE, ["session_id"], unique=False
    )
    # 기존 세션: 지금 상태를 한 번 기록한다 (이전 전이 이력은 없으므로 from_state 없음, 시각은 등록 시각)
    op.execute(
        f"INSERT INTO {TABLE} (session_id, from_state, to_state, at, actor) "
        "SELECT session_id, NULL, lifecycle_state, created_at, 'migration:0011' FROM sessions "
        "ORDER BY created_at, session_id"
    )
    # 추가만 한다: 0009의 dlp_append_only(), 0010의 dlp_no_truncate()를 그대로 쓴다
    if op.get_bind().dialect.name == "postgresql":
        op.execute(
            f"CREATE TRIGGER {TABLE}_append_only BEFORE UPDATE OR DELETE ON {TABLE} "
            "FOR EACH ROW EXECUTE FUNCTION dlp_append_only()"
        )
        op.execute(
            f"CREATE TRIGGER {TABLE}_no_truncate BEFORE TRUNCATE ON {TABLE} "
            "FOR EACH STATEMENT EXECUTE FUNCTION dlp_no_truncate()"
        )


def downgrade() -> None:
    if op.get_bind().dialect.name == "postgresql":
        op.execute(f"DROP TRIGGER IF EXISTS {TABLE}_no_truncate ON {TABLE}")
        op.execute(f"DROP TRIGGER IF EXISTS {TABLE}_append_only ON {TABLE}")
    op.drop_index(op.f("ix_session_lifecycle_events_session_id"), table_name=TABLE)
    op.drop_table(TABLE)
