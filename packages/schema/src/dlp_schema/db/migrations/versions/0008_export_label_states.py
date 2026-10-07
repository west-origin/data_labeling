"""exports record their verification policy

Revision ID: 0008
Revises: 0007
Create Date: 2026-10-07
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.add_column(
        "exports",
        sa.Column(
            "label_states",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            server_default="[]",
            nullable=False,
        ),
    )


def downgrade() -> None:
    with op.batch_alter_table("exports") as batch:
        batch.drop_column("label_states")
