"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-10-07
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | None = None
depends_on: str | None = None

IMMUTABLE_LABEL_FUNCTION = """
CREATE OR REPLACE FUNCTION dlp_label_records_immutable() RETURNS trigger AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'label_records는 삭제할 수 없습니다 (label_id=%)', OLD.label_id;
    END IF;
    IF (NEW.label_id, NEW.session_id, NEW.stream_id, NEW.kind, NEW.t_start_ms, NEW.t_end_ms,
        NEW.ontology_version, NEW.source, NEW.model_version, NEW.sensor_id, NEW.evidence,
        NEW.confidence, NEW.parent_label_id, NEW.retracted, NEW.seeded_error, NEW.created_at,
        NEW.payload)
       IS DISTINCT FROM
       (OLD.label_id, OLD.session_id, OLD.stream_id, OLD.kind, OLD.t_start_ms, OLD.t_end_ms,
        OLD.ontology_version, OLD.source, OLD.model_version, OLD.sensor_id, OLD.evidence,
        OLD.confidence, OLD.parent_label_id, OLD.retracted, OLD.seeded_error, OLD.created_at,
        OLD.payload) THEN
        RAISE EXCEPTION 'label_records는 검수 상태 외에 수정할 수 없습니다 (label_id=%)', OLD.label_id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
"""


def upgrade() -> None:
    op.create_table(
        "ontology_versions",
        sa.Column("version", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column(
            "content",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=False,
        ),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("version", name=op.f("pk_ontology_versions")),
    )
    op.create_table(
        "dataset_versions",
        sa.Column("version_id", sa.String(length=128), nullable=False),
        sa.Column("parent_version_id", sa.String(length=128), nullable=True),
        sa.Column("ontology_version", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("snapshot_uri", sa.Text(), nullable=False),
        sa.Column("golden_set_version", sa.String(length=64), nullable=True),
        sa.Column(
            "excluded_sessions",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["ontology_version"],
            ["ontology_versions.version"],
            name=op.f("fk_dataset_versions_ontology_version_ontology_versions"),
        ),
        sa.ForeignKeyConstraint(
            ["parent_version_id"],
            ["dataset_versions.version_id"],
            name=op.f("fk_dataset_versions_parent_version_id_dataset_versions"),
        ),
        sa.PrimaryKeyConstraint("version_id", name=op.f("pk_dataset_versions")),
    )
    op.create_table(
        "sessions",
        sa.Column("session_id", sa.String(length=128), nullable=False),
        sa.Column("domain", sa.String(length=32), nullable=False),
        sa.Column("worker_id", sa.String(length=128), nullable=False),
        sa.Column("site_id", sa.String(length=128), nullable=False),
        sa.Column("consent_version", sa.String(length=64), nullable=False),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("duration_ms", sa.BigInteger(), nullable=False),
        sa.Column(
            "calibration",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=False,
        ),
        sa.Column("privacy_state", sa.String(length=32), nullable=False),
        sa.Column("lifecycle_state", sa.String(length=32), nullable=False),
        sa.Column("ontology_version", sa.String(length=32), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(
            ["ontology_version"],
            ["ontology_versions.version"],
            name=op.f("fk_sessions_ontology_version_ontology_versions"),
        ),
        sa.PrimaryKeyConstraint("session_id", name=op.f("pk_sessions")),
    )
    op.create_index(op.f("ix_sessions_site_id"), "sessions", ["site_id"], unique=False)
    op.create_index(op.f("ix_sessions_worker_id"), "sessions", ["worker_id"], unique=False)
    op.create_table(
        "dataset_split_assignments",
        sa.Column("version_id", sa.String(length=128), nullable=False),
        sa.Column("session_id", sa.String(length=128), nullable=False),
        sa.Column("split", sa.String(length=16), nullable=False),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["sessions.session_id"],
            name=op.f("fk_dataset_split_assignments_session_id_sessions"),
        ),
        sa.ForeignKeyConstraint(
            ["version_id"],
            ["dataset_versions.version_id"],
            name=op.f("fk_dataset_split_assignments_version_id_dataset_versions"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "version_id", "session_id", name=op.f("pk_dataset_split_assignments")
        ),
    )
    op.create_table(
        "label_records",
        sa.Column("label_id", sa.String(length=128), nullable=False),
        sa.Column("session_id", sa.String(length=128), nullable=False),
        sa.Column("stream_id", sa.String(length=128), nullable=True),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("t_start_ms", sa.BigInteger(), nullable=False),
        sa.Column("t_end_ms", sa.BigInteger(), nullable=False),
        sa.Column("ontology_version", sa.String(length=32), nullable=False),
        sa.Column("source", sa.String(length=16), nullable=False),
        sa.Column("model_version", sa.String(length=128), nullable=True),
        sa.Column("sensor_id", sa.String(length=128), nullable=True),
        sa.Column("evidence", sa.String(length=16), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("verification_state", sa.String(length=32), nullable=False),
        sa.Column("reviewer_id", sa.String(length=128), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("parent_label_id", sa.String(length=128), nullable=True),
        sa.Column("retracted", sa.Boolean(), nullable=False),
        sa.Column("seeded_error", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "payload",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=False,
        ),
        sa.CheckConstraint("t_start_ms <= t_end_ms", name=op.f("ck_label_records_time_order")),
        sa.ForeignKeyConstraint(
            ["ontology_version"],
            ["ontology_versions.version"],
            name=op.f("fk_label_records_ontology_version_ontology_versions"),
        ),
        sa.ForeignKeyConstraint(
            ["parent_label_id"],
            ["label_records.label_id"],
            name=op.f("fk_label_records_parent_label_id_label_records"),
        ),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["sessions.session_id"],
            name=op.f("fk_label_records_session_id_sessions"),
        ),
        sa.PrimaryKeyConstraint("label_id", name=op.f("pk_label_records")),
    )
    op.create_index(
        op.f("ix_label_records_parent_label_id"), "label_records", ["parent_label_id"], unique=False
    )
    op.create_index(
        "ix_label_records_session_kind", "label_records", ["session_id", "kind"], unique=False
    )
    op.create_index(
        "ix_label_records_session_start",
        "label_records",
        ["session_id", "t_start_ms"],
        unique=False,
    )
    op.create_table(
        "streams",
        sa.Column("session_id", sa.String(length=128), nullable=False),
        sa.Column("stream_id", sa.String(length=128), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("uri", sa.Text(), nullable=False),
        sa.Column("sample_rate_hz", sa.Float(), nullable=True),
        sa.Column("pts_index_uri", sa.Text(), nullable=True),
        sa.Column("offset_ms", sa.Float(), nullable=False),
        sa.Column("clock_scale", sa.Float(), nullable=False),
        sa.Column("sync_method", sa.String(length=32), nullable=False),
        sa.Column("sync_confidence", sa.Float(), nullable=True),
        sa.Column("manual_adjustment_ms", sa.Float(), nullable=False),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["sessions.session_id"],
            name=op.f("fk_streams_session_id_sessions"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("session_id", "stream_id", name=op.f("pk_streams")),
    )

    # 라벨 불변성: 검수 상태(verification_*, reviewer_id, reviewed_at) 외의 열은 수정할 수 없다.
    # 수정은 새 레코드 + parent_label_id로 한다. 삭제도 막는다 (삭제는 retracted 레코드로 표현).
    if op.get_bind().dialect.name == "postgresql":
        op.execute(IMMUTABLE_LABEL_FUNCTION)
        op.execute(
            "CREATE TRIGGER label_records_immutable BEFORE UPDATE OR DELETE ON label_records "
            "FOR EACH ROW EXECUTE FUNCTION dlp_label_records_immutable()"
        )


def downgrade() -> None:
    if op.get_bind().dialect.name == "postgresql":
        op.execute("DROP TRIGGER IF EXISTS label_records_immutable ON label_records")
        op.execute("DROP FUNCTION IF EXISTS dlp_label_records_immutable()")
    op.drop_table("streams")
    op.drop_index("ix_label_records_session_start", table_name="label_records")
    op.drop_index("ix_label_records_session_kind", table_name="label_records")
    op.drop_index(op.f("ix_label_records_parent_label_id"), table_name="label_records")
    op.drop_table("label_records")
    op.drop_table("dataset_split_assignments")
    op.drop_index(op.f("ix_sessions_worker_id"), table_name="sessions")
    op.drop_index(op.f("ix_sessions_site_id"), table_name="sessions")
    op.drop_table("sessions")
    op.drop_table("dataset_versions")
    op.drop_table("ontology_versions")
