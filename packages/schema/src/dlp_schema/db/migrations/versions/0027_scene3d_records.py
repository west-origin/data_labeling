"""청소 전후 3D 비교 산출물 기록 (ADR 0066)

Revision ID: 0027
Revises: 0026
Create Date: 2026-11-04

`scene3d_records`(추가 전용)를 만든다. `dlp scene3d …`가 수집·블러 탐지·사람 승인·렌더·복원·정렬·비교
산출물을 버킷에 쓸 때마다 한 줄씩 남긴다 (본문은 버킷의 JSON). 수정·삭제는 0009의
`dlp_append_only()`, TRUNCATE는 0010의 `dlp_no_truncate()` 트리거로 막는다 (PostgreSQL). 새 표라
백필이 없다.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0027"
down_revision: str | None = "0026"
branch_labels: str | None = None
depends_on: str | None = None

TABLE = "scene3d_records"


def upgrade() -> None:
    """기록 표와 추가 전용·TRUNCATE 금지 트리거를 만든다."""
    op.create_table(
        TABLE,
        sa.Column(
            "record_id",
            sa.BigInteger().with_variant(sa.Integer(), "sqlite"),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column("session_id", sa.String(length=128), nullable=False),
        sa.Column("phase", sa.String(length=16), nullable=True),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("model_version", sa.Text(), nullable=True),
        sa.Column("tag", sa.String(length=64), nullable=False),
        sa.Column("bucket", sa.String(length=16), nullable=False),
        sa.Column("storage_key", sa.Text(), nullable=False),
        sa.Column("actor", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("record_id", name=op.f("pk_scene3d_records")),
    )
    op.create_index(op.f("ix_scene3d_records_session_id"), TABLE, ["session_id"], unique=False)
    if op.get_bind().dialect.name == "postgresql":
        op.execute(
            f"CREATE TRIGGER {TABLE}_append_only BEFORE UPDATE OR DELETE ON {TABLE} "
            "FOR EACH ROW EXECUTE FUNCTION dlp_append_only()"
        )
        op.execute(
            f"CREATE TRIGGER {TABLE}_no_truncate BEFORE TRUNCATE ON {TABLE} "
            "FOR EACH STATEMENT EXECUTE FUNCTION dlp_no_truncate()"
        )


def downgrade() -> None:
    """기록 표를 지운다. 기록이 하나라도 있으면 거부한다 (블러 승인 이력을 잃는다)."""
    count = op.get_bind().execute(sa.text(f"SELECT count(*) FROM {TABLE}")).scalar_one()
    if count:
        raise RuntimeError(
            f"scene3d 기록 {count}개가 있어 0027을 되돌릴 수 없습니다 (블러 승인·복원 이력을 잃는다)"
        )
    op.drop_index(op.f("ix_scene3d_records_session_id"), table_name=TABLE)
    op.drop_table(TABLE)
