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

# 0006 review_tasks.sent_label_ids: 작업에 보낸 라벨 ID 목록 (JSON, nullable).
# 수집이 "보낸 라벨"과 결과를 비교하게 해, 그 사이 다른 단계가 라벨을 바꿔도 검수 결과를 바르게 해석한다.
# 기존 행은 NULL로 남는다 (수집 코드가 배정 정보로 대신 판단한다).


def upgrade() -> None:
    """review_tasks.sent_label_ids 열을 더한다."""
    op.add_column(
        "review_tasks",
        sa.Column(
            "sent_label_ids",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=True,
        ),
    )


def downgrade() -> None:
    """review_tasks.sent_label_ids 열을 지운다."""
    with op.batch_alter_table("review_tasks") as batch:
        batch.drop_column("sent_label_ids")
