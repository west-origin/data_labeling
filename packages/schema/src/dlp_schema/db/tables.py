"""PostgreSQL 테이블 정의 (SQLAlchemy Core). 스키마 변경은 반드시 Alembic 마이그레이션으로 한다."""

from __future__ import annotations

from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

metadata = sa.MetaData(
    naming_convention={
        "ix": "ix_%(column_0_label)s",
        "uq": "uq_%(table_name)s_%(column_0_name)s",
        "ck": "ck_%(table_name)s_%(constraint_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "pk": "pk_%(table_name)s",
    }
)

Json: sa.types.TypeEngine[Any] = sa.JSON().with_variant(JSONB(), "postgresql")
Ts = sa.DateTime(timezone=True)

ontology_versions = sa.Table(
    "ontology_versions",
    metadata,
    sa.Column("version", sa.String(32), primary_key=True),
    sa.Column("status", sa.String(16), nullable=False),
    sa.Column("content", Json, nullable=False),
    sa.Column("created_at", Ts, nullable=False, server_default=sa.func.now()),
)

sessions = sa.Table(
    "sessions",
    metadata,
    sa.Column("session_id", sa.String(128), primary_key=True),
    sa.Column("domain", sa.String(32), nullable=False),
    sa.Column("worker_id", sa.String(128), nullable=False, index=True),
    sa.Column("site_id", sa.String(128), nullable=False, index=True),
    sa.Column("consent_version", sa.String(64), nullable=False),
    sa.Column("recorded_at", Ts, nullable=False),
    sa.Column("duration_ms", sa.BigInteger, nullable=False),
    sa.Column("calibration", Json, nullable=False),
    sa.Column("privacy_state", sa.String(32), nullable=False),
    sa.Column("lifecycle_state", sa.String(32), nullable=False),
    sa.Column(
        "ontology_version",
        sa.String(32),
        sa.ForeignKey("ontology_versions.version"),
        nullable=True,
    ),
    sa.Column("created_at", Ts, nullable=False, server_default=sa.func.now()),
)

streams = sa.Table(
    "streams",
    metadata,
    sa.Column(
        "session_id",
        sa.String(128),
        sa.ForeignKey("sessions.session_id", ondelete="CASCADE"),
        primary_key=True,
    ),
    sa.Column("stream_id", sa.String(128), primary_key=True),
    sa.Column("kind", sa.String(32), nullable=False),
    sa.Column("uri", sa.Text(), nullable=False),
    sa.Column("sample_rate_hz", sa.Float(), nullable=True),
    sa.Column("pts_index_uri", sa.Text(), nullable=True),
    sa.Column("offset_ms", sa.Float(), nullable=False),
    sa.Column("clock_scale", sa.Float(), nullable=False),
    sa.Column("sync_method", sa.String(32), nullable=False),
    sa.Column("sync_confidence", sa.Float(), nullable=True),
    sa.Column("manual_adjustment_ms", sa.Float(), nullable=False),
)

label_records = sa.Table(
    "label_records",
    metadata,
    sa.Column("label_id", sa.String(128), primary_key=True),
    sa.Column("session_id", sa.String(128), sa.ForeignKey("sessions.session_id"), nullable=False),
    sa.Column("stream_id", sa.String(128), nullable=True),
    sa.Column("kind", sa.String(32), nullable=False),
    sa.Column("t_start_ms", sa.BigInteger, nullable=False),
    sa.Column("t_end_ms", sa.BigInteger, nullable=False),
    sa.Column(
        "ontology_version",
        sa.String(32),
        sa.ForeignKey("ontology_versions.version"),
        nullable=False,
    ),
    sa.Column("source", sa.String(16), nullable=False),
    sa.Column("model_version", sa.String(128), nullable=True),
    sa.Column("sensor_id", sa.String(128), nullable=True),
    sa.Column("evidence", sa.String(16), nullable=False),
    sa.Column("confidence", sa.Float(), nullable=True),
    sa.Column("verification_state", sa.String(32), nullable=False),
    sa.Column("reviewer_id", sa.String(128), nullable=True),
    sa.Column("reviewed_at", Ts, nullable=True),
    sa.Column(
        "parent_label_id",
        sa.String(128),
        sa.ForeignKey("label_records.label_id"),
        nullable=True,
        index=True,
    ),
    sa.Column("retracted", sa.Boolean, nullable=False),
    sa.Column("seeded_error", sa.Boolean, nullable=False),
    sa.Column("created_at", Ts, nullable=False),
    sa.Column("payload", Json, nullable=False),
    sa.CheckConstraint("t_start_ms <= t_end_ms", name="time_order"),
    sa.Index("ix_label_records_session_kind", "session_id", "kind"),
    sa.Index("ix_label_records_session_start", "session_id", "t_start_ms"),
)

dataset_versions = sa.Table(
    "dataset_versions",
    metadata,
    sa.Column("version_id", sa.String(128), primary_key=True),
    sa.Column(
        "parent_version_id",
        sa.String(128),
        sa.ForeignKey("dataset_versions.version_id"),
        nullable=True,
    ),
    sa.Column(
        "ontology_version",
        sa.String(32),
        sa.ForeignKey("ontology_versions.version"),
        nullable=False,
    ),
    sa.Column("created_at", Ts, nullable=False),
    sa.Column("snapshot_uri", sa.Text(), nullable=False),
    sa.Column("golden_set_version", sa.String(64), nullable=True),
    sa.Column("excluded_sessions", Json, nullable=False),
)

dataset_split_assignments = sa.Table(
    "dataset_split_assignments",
    metadata,
    sa.Column(
        "version_id",
        sa.String(128),
        sa.ForeignKey("dataset_versions.version_id", ondelete="CASCADE"),
        primary_key=True,
    ),
    sa.Column("session_id", sa.String(128), sa.ForeignKey("sessions.session_id"), primary_key=True),
    sa.Column("split", sa.String(16), nullable=False),
)
