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

# 0008 exports.label_states: 내보낸 라벨의 검증 상태 목록 (검증 정책 기록, WP15, ADR 0018).
# 서버 기본값 '[]'로 기존 행을 채운다 (예전 내보내기는 정책 기록이 없음을 뜻한다).


def upgrade() -> None:
    """exports.label_states 열(JSON, 기본 빈 목록)을 더한다."""
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
    """exports.label_states 열을 지운다."""
    with op.batch_alter_table("exports") as batch:
        batch.drop_column("label_states")
