"""review operations: measurement labels, review modes, assignments

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-07
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | None = None
depends_on: str | None = None

# 불변성 트리거에 measurement 열을 더한다 (검수 상태 외에는 바꿀 수 없다)
IMMUTABLE_LABEL_FUNCTION = """
CREATE OR REPLACE FUNCTION dlp_label_records_immutable() RETURNS trigger AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'label_records는 삭제할 수 없습니다 (label_id=%)', OLD.label_id;
    END IF;
    IF (NEW.label_id, NEW.session_id, NEW.stream_id, NEW.kind, NEW.t_start_ms, NEW.t_end_ms,
        NEW.ontology_version, NEW.source, NEW.model_version, NEW.sensor_id, NEW.evidence,
        NEW.confidence, NEW.parent_label_id, NEW.retracted, NEW.seeded_error, NEW.measurement,
        NEW.created_at, NEW.payload)
       IS DISTINCT FROM
       (OLD.label_id, OLD.session_id, OLD.stream_id, OLD.kind, OLD.t_start_ms, OLD.t_end_ms,
        OLD.ontology_version, OLD.source, OLD.model_version, OLD.sensor_id, OLD.evidence,
        OLD.confidence, OLD.parent_label_id, OLD.retracted, OLD.seeded_error, OLD.measurement,
        OLD.created_at, OLD.payload) THEN
        RAISE EXCEPTION 'label_records는 검수 상태 외에 수정할 수 없습니다 (label_id=%)', OLD.label_id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
"""

PREVIOUS_FUNCTION = IMMUTABLE_LABEL_FUNCTION.replace(" NEW.measurement,", "").replace(
    " OLD.measurement,", ""
)

JSON = sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")


def upgrade() -> None:
    op.add_column("label_records", sa.Column("measurement", sa.String(length=16), nullable=True))
    op.add_column(
        "review_tasks",
        sa.Column("mode", sa.String(length=16), nullable=False, server_default="standard"),
    )
    with op.batch_alter_table("review_tasks") as batch:  # SQLite는 ALTER COLUMN이 없어 일괄 변경
        batch.alter_column("mode", server_default=None)
    op.add_column("review_tasks", sa.Column("assignment_id", sa.String(length=128), nullable=True))
    op.create_table(
        "review_assignments",
        sa.Column("assignment_id", sa.String(length=128), nullable=False),
        sa.Column("session_id", sa.String(length=128), nullable=False),
        sa.Column("stream_id", sa.String(length=128), nullable=True),
        sa.Column("label_kinds", JSON, nullable=False),
        sa.Column("mode", sa.String(length=16), nullable=False),
        sa.Column("priority", sa.Float(), nullable=False),
        sa.Column("flagged", JSON, nullable=False),
        sa.Column("assignee", sa.String(length=128), nullable=True),
        sa.Column("pair_id", sa.String(length=128), nullable=True),
        sa.Column("injected", JSON, nullable=False),
        sa.Column("sample_label_ids", JSON, nullable=False),
        sa.Column("withheld_label_ids", JSON, nullable=False),
        sa.Column("only_label_ids", JSON, nullable=False),
        sa.Column("task_key", sa.String(length=128), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["sessions.session_id"],
            name=op.f("fk_review_assignments_session_id_sessions"),
        ),
        sa.PrimaryKeyConstraint("assignment_id", name=op.f("pk_review_assignments")),
    )
    op.create_index(
        "ix_review_assignments_session", "review_assignments", ["session_id"], unique=False
    )
    op.create_index(
        "ix_review_assignments_queue", "review_assignments", ["status", "priority"], unique=False
    )
    if op.get_bind().dialect.name == "postgresql":
        op.execute(IMMUTABLE_LABEL_FUNCTION)


def downgrade() -> None:
    if op.get_bind().dialect.name == "postgresql":
        op.execute(PREVIOUS_FUNCTION)
    op.drop_index("ix_review_assignments_queue", table_name="review_assignments")
    op.drop_index("ix_review_assignments_session", table_name="review_assignments")
    op.drop_table("review_assignments")
    with op.batch_alter_table("review_tasks") as batch:
        batch.drop_column("assignment_id")
        batch.drop_column("mode")
    with op.batch_alter_table("label_records") as batch:
        batch.drop_column("measurement")
