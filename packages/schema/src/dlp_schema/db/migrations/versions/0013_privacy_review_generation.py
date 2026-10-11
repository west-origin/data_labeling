"""프라이버시 검수 작업을 완료 탐지 세대에 연결 (ADR 0033).

Revision ID: 0013
Revises: 0012
Create Date: 2026-10-09
"""

import sqlalchemy as sa
from alembic import op

revision: str = "0013"
down_revision: str | None = "0012"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """기존 작업은 세대를 추측해 채우지 않는다. 다음 탐지 뒤 새 작업을 만들어야 한다."""
    with op.batch_alter_table("privacy_detection_runs") as batch:
        batch.create_unique_constraint(
            "uq_privacy_detection_runs_event_id", ["event_id", "session_id", "stream_id"]
        )
    with op.batch_alter_table("review_tasks") as batch:
        batch.add_column(
            sa.Column("privacy_detection_id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"))
        )
        batch.create_foreign_key(
            "fk_review_tasks_privacy_detection_id_privacy_detection_runs",
            "privacy_detection_runs",
            ["privacy_detection_id", "session_id", "stream_id"],
            ["event_id", "session_id", "stream_id"],
        )


def downgrade() -> None:
    """검수 세대 참조를 제거한다. 탐지 완료 이력은 보존한다."""
    with op.batch_alter_table("review_tasks") as batch:
        batch.drop_constraint(
            "fk_review_tasks_privacy_detection_id_privacy_detection_runs", type_="foreignkey"
        )
        batch.drop_column("privacy_detection_id")
    with op.batch_alter_table("privacy_detection_runs") as batch:
        batch.drop_constraint("uq_privacy_detection_runs_event_id", type_="unique")
