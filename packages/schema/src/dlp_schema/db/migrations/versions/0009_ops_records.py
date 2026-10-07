"""operations records: raw access audit log, review work, privacy audits, retention decisions

Revision ID: 0009
Revises: 0008
Create Date: 2026-10-07
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: str | None = None
depends_on: str | None = None

# 0009 운영 기록 (WP16, ADR 0020): raw_access_log, review_work, privacy_audits, retention_decisions.
# 네 테이블 모두 추가 전용이다: PostgreSQL에서 행 단위 BEFORE UPDATE OR DELETE 트리거가 dlp_append_only()를
# 불러 예외를 낸다. TRUNCATE는 0010에서 막는다.

# 추가 전용 트리거를 거는 테이블
TABLES = ("raw_access_log", "review_work", "privacy_audits", "retention_decisions")

# 어떤 수정·삭제든 거부하는 트리거 함수 (TG_TABLE_NAME으로 테이블 이름을 메시지에 넣는다).
# 0011이 session_lifecycle_events에도 같은 함수를 쓴다.
APPEND_ONLY_FUNCTION = """
CREATE OR REPLACE FUNCTION dlp_append_only() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION '% 기록은 추가만 할 수 있습니다 (수정·삭제 금지)', TG_TABLE_NAME;
END;
$$ LANGUAGE plpgsql;
"""


def upgrade() -> None:
    """운영 기록 테이블 네 개와 색인, (PostgreSQL이면) 추가 전용 트리거를 만든다."""
    ts = sa.DateTime(timezone=True)
    op.create_table(
        "raw_access_log",
        sa.Column("event_id", sa.String(length=128), nullable=False),
        sa.Column("at", ts, nullable=False),
        sa.Column("actor", sa.String(length=128), nullable=False),
        sa.Column("purpose", sa.String(length=128), nullable=False),
        sa.Column("action", sa.String(length=16), nullable=False),
        sa.Column("bucket", sa.String(length=128), nullable=False),
        sa.Column("key", sa.Text(), nullable=False),
        sa.Column("session_id", sa.String(length=128), nullable=True),
        sa.PrimaryKeyConstraint("event_id", name=op.f("pk_raw_access_log")),
    )
    op.create_index(op.f("ix_raw_access_log_at"), "raw_access_log", ["at"], unique=False)
    op.create_index(op.f("ix_raw_access_log_actor"), "raw_access_log", ["actor"], unique=False)
    op.create_index(
        op.f("ix_raw_access_log_session_id"), "raw_access_log", ["session_id"], unique=False
    )
    op.create_table(
        "review_work",
        sa.Column("work_id", sa.String(length=128), nullable=False),
        sa.Column("task_key", sa.String(length=128), nullable=True),
        sa.Column("session_id", sa.String(length=128), nullable=False),
        sa.Column("reviewer", sa.String(length=128), nullable=False),
        sa.Column("stage", sa.String(length=16), nullable=False),
        sa.Column("seconds", sa.Float(), nullable=False),
        sa.Column("video_ms", sa.BigInteger(), nullable=False),
        sa.Column("source", sa.String(length=16), nullable=False),
        sa.Column("recorded_at", ts, nullable=False),
        sa.PrimaryKeyConstraint("work_id", name=op.f("pk_review_work")),
    )
    op.create_index(op.f("ix_review_work_session_id"), "review_work", ["session_id"], unique=False)
    op.create_index(
        op.f("ix_review_work_recorded_at"), "review_work", ["recorded_at"], unique=False
    )
    op.create_table(
        "privacy_audits",
        sa.Column("audit_id", sa.String(length=128), nullable=False),
        sa.Column("session_id", sa.String(length=128), nullable=False),
        sa.Column("stream_id", sa.String(length=128), nullable=False),
        sa.Column("duration_ms", sa.BigInteger(), nullable=False),
        sa.Column("misses", sa.Integer(), nullable=False),
        sa.Column("auditor", sa.String(length=128), nullable=False),
        sa.Column("blur_reviewer", sa.String(length=128), nullable=False),
        sa.Column("audited_at", ts, nullable=False),
        sa.PrimaryKeyConstraint("audit_id", name=op.f("pk_privacy_audits")),
    )
    op.create_index(
        op.f("ix_privacy_audits_session_id"), "privacy_audits", ["session_id"], unique=False
    )
    op.create_index(
        op.f("ix_privacy_audits_audited_at"), "privacy_audits", ["audited_at"], unique=False
    )
    op.create_table(
        "retention_decisions",
        sa.Column("decision_id", sa.String(length=128), nullable=False),
        sa.Column("session_id", sa.String(length=128), nullable=False),
        sa.Column("decision", sa.String(length=16), nullable=False),
        sa.Column("until", sa.Date(), nullable=True),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("decided_by", sa.String(length=128), nullable=False),
        sa.Column("decided_at", ts, nullable=False),
        sa.PrimaryKeyConstraint("decision_id", name=op.f("pk_retention_decisions")),
    )
    op.create_index(
        op.f("ix_retention_decisions_session_id"),
        "retention_decisions",
        ["session_id"],
        unique=False,
    )
    # 감사·운영 기록은 추가만 한다 (수정·삭제를 DB에서 막는다)
    if op.get_bind().dialect.name == "postgresql":
        op.execute(APPEND_ONLY_FUNCTION)
        for table in TABLES:
            op.execute(
                f"CREATE TRIGGER {table}_append_only BEFORE UPDATE OR DELETE ON {table} "
                "FOR EACH ROW EXECUTE FUNCTION dlp_append_only()"
            )


def downgrade() -> None:
    """트리거·함수를 지우고 운영 기록 테이블을 지운다 (감사 기록이 사라진다)."""
    if op.get_bind().dialect.name == "postgresql":
        for table in TABLES:
            op.execute(f"DROP TRIGGER IF EXISTS {table}_append_only ON {table}")
        op.execute("DROP FUNCTION IF EXISTS dlp_append_only()")
    op.drop_index(op.f("ix_retention_decisions_session_id"), table_name="retention_decisions")
    op.drop_table("retention_decisions")
    op.drop_index(op.f("ix_privacy_audits_audited_at"), table_name="privacy_audits")
    op.drop_index(op.f("ix_privacy_audits_session_id"), table_name="privacy_audits")
    op.drop_table("privacy_audits")
    op.drop_index(op.f("ix_review_work_recorded_at"), table_name="review_work")
    op.drop_index(op.f("ix_review_work_session_id"), table_name="review_work")
    op.drop_table("review_work")
    op.drop_index(op.f("ix_raw_access_log_session_id"), table_name="raw_access_log")
    op.drop_index(op.f("ix_raw_access_log_actor"), table_name="raw_access_log")
    op.drop_index(op.f("ix_raw_access_log_at"), table_name="raw_access_log")
    op.drop_table("raw_access_log")
