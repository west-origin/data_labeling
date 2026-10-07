"""audit 3: wider model_version / golden_set_version, block TRUNCATE on append-only tables

Revision ID: 0010
Revises: 0009
Create Date: 2026-10-07
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0010"
down_revision: str | None = "0009"
branch_labels: str | None = None
depends_on: str | None = None

# 수정·삭제를 행 단위 트리거로 막는 표. TRUNCATE는 행 트리거를 거치지 않으므로 문장 단위로 막는다.
APPEND_ONLY_TABLES = (
    "label_records",
    "raw_access_log",
    "review_work",
    "privacy_audits",
    "retention_decisions",
)

NO_TRUNCATE_FUNCTION = """
CREATE OR REPLACE FUNCTION dlp_no_truncate() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION '% 기록은 비울 수 없습니다 (TRUNCATE 금지)', TG_TABLE_NAME;
END;
$$ LANGUAGE plpgsql;
"""


def upgrade() -> None:
    # 모델 버전에는 정책 해시 등이 붙어 128자를 넘을 수 있다 (프라이버시 파이프라인 등)
    with op.batch_alter_table("label_records") as batch:
        batch.alter_column(
            "model_version",
            existing_type=sa.String(length=128),
            type_=sa.Text(),
            existing_nullable=True,
        )
    # 골든셋 버전 ID는 golden_sets.version(128자)와 같은 길이여야 한다
    with op.batch_alter_table("dataset_versions") as batch:
        batch.alter_column(
            "golden_set_version",
            existing_type=sa.String(length=64),
            type_=sa.String(length=128),
            existing_nullable=True,
        )
    if op.get_bind().dialect.name == "postgresql":
        op.execute(NO_TRUNCATE_FUNCTION)
        for table in APPEND_ONLY_TABLES:
            op.execute(
                f"CREATE TRIGGER {table}_no_truncate BEFORE TRUNCATE ON {table} "
                "FOR EACH STATEMENT EXECUTE FUNCTION dlp_no_truncate()"
            )


def downgrade() -> None:
    if op.get_bind().dialect.name == "postgresql":
        for table in APPEND_ONLY_TABLES:
            op.execute(f"DROP TRIGGER IF EXISTS {table}_no_truncate ON {table}")
        op.execute("DROP FUNCTION IF EXISTS dlp_no_truncate()")
    with op.batch_alter_table("dataset_versions") as batch:
        batch.alter_column(
            "golden_set_version",
            existing_type=sa.String(length=128),
            type_=sa.String(length=64),
            existing_nullable=True,
        )
    with op.batch_alter_table("label_records") as batch:
        batch.alter_column(
            "model_version",
            existing_type=sa.Text(),
            type_=sa.String(length=128),
            existing_nullable=True,
        )
