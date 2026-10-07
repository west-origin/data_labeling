"""review tasks remember the labels they sent

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-07
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.add_column(
        "review_tasks",
        sa.Column(
            "sent_label_ids",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=True,
        ),
    )


def downgrade() -> None:
    with op.batch_alter_table("review_tasks") as batch:
        batch.drop_column("sent_label_ids")
