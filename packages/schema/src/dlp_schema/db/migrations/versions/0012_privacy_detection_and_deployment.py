"""탐지 완료 이력(0건 포함)과 과제별 배포 유일성 (ADR 0032).

Revision ID: 0012
Revises: 0011
Create Date: 2026-10-08
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0012"
down_revision: str | None = "0011"
branch_labels: str | None = None
depends_on: str | None = None

TABLE = "privacy_detection_runs"
INDEX = "uq_model_versions_deployed_task"


def upgrade() -> None:
    """추가 전용 탐지 기록과 배포 유일성 색인을 만든다.

    기존의 중복 배포를 임의로 퇴역시키지 않는다. 그런 데이터가 있으면 유일성 색인 생성이
    실패하므로 운영자가 올바른 배포를 확인해 정리한 뒤 다시 적용한다. 구 탐지 시각은 블러
    라벨 이력으로 대체하며, 외부 0건 표시는 다음 detect 호출에서 확인 시각으로 복구한다.
    """
    op.create_table(
        TABLE,
        sa.Column(
            "event_id",
            sa.BigInteger().with_variant(sa.Integer(), "sqlite"),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column("session_id", sa.String(128), nullable=False),
        sa.Column("stream_id", sa.String(128), nullable=False),
        sa.Column(
            "model_versions",
            sa.JSON().with_variant(postgresql.JSONB(), "postgresql"),
            nullable=False,
        ),
        sa.Column("at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["session_id", "stream_id"],
            ["streams.session_id", "streams.stream_id"],
            name=op.f("fk_privacy_detection_runs_session_id_streams"),
        ),
        sa.PrimaryKeyConstraint("event_id", name=op.f("pk_privacy_detection_runs")),
    )
    op.create_index("ix_privacy_detection_runs_session_stream", TABLE, ["session_id", "stream_id"])
    if op.get_bind().dialect.name == "postgresql":
        op.execute(
            f"CREATE TRIGGER {TABLE}_append_only BEFORE UPDATE OR DELETE ON {TABLE} "
            "FOR EACH ROW EXECUTE FUNCTION dlp_append_only()"
        )
        op.execute(
            f"CREATE TRIGGER {TABLE}_no_truncate BEFORE TRUNCATE ON {TABLE} "
            "FOR EACH STATEMENT EXECUTE FUNCTION dlp_no_truncate()"
        )
    op.create_index(
        INDEX,
        "model_versions",
        ["task"],
        unique=True,
        postgresql_where=sa.text("status = 'deployed'"),
        sqlite_where=sa.text("status = 'deployed'"),
    )


def downgrade() -> None:
    """배포 색인과 탐지 기록을 제거한다. 공용 트리거 함수는 이전 리비전 소유다."""
    op.drop_index(INDEX, table_name="model_versions")
    op.drop_index("ix_privacy_detection_runs_session_stream", table_name=TABLE)
    op.drop_table(TABLE)
