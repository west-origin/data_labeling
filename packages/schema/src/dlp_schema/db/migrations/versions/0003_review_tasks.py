"""review tasks in external annotation tools

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-07
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | None = None
depends_on: str | None = None

# 0003 review_tasks: 외부 검수 도구(CVAT·Label Studio) 작업과 세션·스트림의 대응 (WP6, ADR 0006).
# mode·assignment_id는 0005, sent_label_ids는 0006에서 더한다.


def upgrade() -> None:
    """review_tasks 테이블과 (session_id, stage) 색인을 만든다."""
    op.create_table(
        "review_tasks",
        sa.Column("task_key", sa.String(length=128), nullable=False),
        sa.Column("tool", sa.String(length=32), nullable=False),
        sa.Column("external_id", sa.String(length=64), nullable=False),
        sa.Column("session_id", sa.String(length=128), nullable=False),
        sa.Column("stream_id", sa.String(length=128), nullable=False),
        sa.Column("stage", sa.String(length=16), nullable=False),
        sa.Column("assignee", sa.String(length=128), nullable=True),
        sa.Column("media_uri", sa.Text(), nullable=False),
        sa.Column(
            "label_kinds",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=False,
        ),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("collected_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["sessions.session_id"],
            name=op.f("fk_review_tasks_session_id_sessions"),
        ),
        sa.PrimaryKeyConstraint("task_key", name=op.f("pk_review_tasks")),
    )
    op.create_index(
        "ix_review_tasks_session", "review_tasks", ["session_id", "stage"], unique=False
    )


def downgrade() -> None:
    """review_tasks 테이블을 지운다."""
    op.drop_index("ix_review_tasks_session", table_name="review_tasks")
    op.drop_table("review_tasks")
