"""retrained model registry

Revision ID: 0007
Revises: 0006
Create Date: 2026-10-07
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | None = None
depends_on: str | None = None

# 0007 model_versions: 재학습 모델 레지스트리 (WP13, ADR 0016). training_runs에 FK.
# task·run_id·status 색인: 과제별 배포 모델 조회와 상태별 목록에 쓴다.


def upgrade() -> None:
    """model_versions 테이블과 색인 세 개를 만든다."""
    op.create_table(
        "model_versions",
        sa.Column("model_version", sa.String(length=128), nullable=False),
        sa.Column("task", sa.String(length=64), nullable=False),
        sa.Column("run_id", sa.String(length=128), nullable=False),
        sa.Column("trainer", sa.String(length=128), nullable=False),
        sa.Column("artifact_uri", sa.Text(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("train_examples", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("report_uri", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["training_runs.run_id"],
            name=op.f("fk_model_versions_run_id_training_runs"),
        ),
        sa.PrimaryKeyConstraint("model_version", name=op.f("pk_model_versions")),
    )
    op.create_index(op.f("ix_model_versions_task"), "model_versions", ["task"], unique=False)
    op.create_index(op.f("ix_model_versions_run_id"), "model_versions", ["run_id"], unique=False)
    op.create_index(op.f("ix_model_versions_status"), "model_versions", ["status"], unique=False)


def downgrade() -> None:
    """model_versions 테이블을 지운다."""
    op.drop_index(op.f("ix_model_versions_status"), table_name="model_versions")
    op.drop_index(op.f("ix_model_versions_run_id"), table_name="model_versions")
    op.drop_index(op.f("ix_model_versions_task"), table_name="model_versions")
    op.drop_table("model_versions")
