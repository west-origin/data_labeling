"""lineage tables

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-07
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | None = None
depends_on: str | None = None

# 0004 계보 테이블 (WP7, ADR 0007): golden_sets, exports, training_runs, withdrawals.
# exports.label_states는 0008, model_versions는 0007에서 더한다.


def upgrade() -> None:
    """계보 테이블 네 개와 데이터셋 버전 FK 색인을 만든다."""
    op.create_table(
        "golden_sets",
        sa.Column("version", sa.String(length=128), nullable=False),
        sa.Column("domain", sa.String(length=32), nullable=False),
        sa.Column(
            "session_ids",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("note", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("version", name=op.f("pk_golden_sets")),
    )
    op.create_table(
        "exports",
        sa.Column("export_id", sa.String(length=128), nullable=False),
        sa.Column("dataset_version_id", sa.String(length=128), nullable=False),
        sa.Column("target", sa.String(length=128), nullable=False),
        sa.Column("format", sa.String(length=64), nullable=False),
        sa.Column("uri", sa.Text(), nullable=False),
        sa.Column(
            "session_ids",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["dataset_version_id"],
            ["dataset_versions.version_id"],
            name=op.f("fk_exports_dataset_version_id_dataset_versions"),
        ),
        sa.PrimaryKeyConstraint("export_id", name=op.f("pk_exports")),
    )
    op.create_index(
        op.f("ix_exports_dataset_version_id"), "exports", ["dataset_version_id"], unique=False
    )
    op.create_table(
        "training_runs",
        sa.Column("run_id", sa.String(length=128), nullable=False),
        sa.Column("dataset_version_id", sa.String(length=128), nullable=False),
        sa.Column("model_name", sa.String(length=128), nullable=False),
        sa.Column("model_version", sa.String(length=128), nullable=False),
        sa.Column("mlflow_run_id", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["dataset_version_id"],
            ["dataset_versions.version_id"],
            name=op.f("fk_training_runs_dataset_version_id_dataset_versions"),
        ),
        sa.PrimaryKeyConstraint("run_id", name=op.f("pk_training_runs")),
    )
    op.create_index(
        op.f("ix_training_runs_dataset_version_id"),
        "training_runs",
        ["dataset_version_id"],
        unique=False,
    )
    op.create_table(
        "withdrawals",
        sa.Column("session_id", sa.String(length=128), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("withdrawn_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["session_id"], ["sessions.session_id"], name=op.f("fk_withdrawals_session_id_sessions")
        ),
        sa.PrimaryKeyConstraint("session_id", name=op.f("pk_withdrawals")),
    )


def downgrade() -> None:
    """계보 테이블 네 개를 지운다."""
    op.drop_table("withdrawals")
    op.drop_index(op.f("ix_training_runs_dataset_version_id"), table_name="training_runs")
    op.drop_table("training_runs")
    op.drop_index(op.f("ix_exports_dataset_version_id"), table_name="exports")
    op.drop_table("exports")
    op.drop_table("golden_sets")
