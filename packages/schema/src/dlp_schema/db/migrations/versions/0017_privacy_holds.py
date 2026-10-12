"""사고 대응 프라이버시 승인 보류·해제 기록 (ADR 0042, F-REV-18)

Revision ID: 0017
Revises: 0015
Create Date: 2026-10-11

`privacy_holds`(추가 전용)를 만든다. 세션의 마지막 기록이 hold면 보류 중이고 그동안 승인이
거부된다. 수정·삭제는 0009의 `dlp_append_only()`, TRUNCATE는 0010의 `dlp_no_truncate()` 트리거로
막는다 (PostgreSQL). 기존 세션은 보류 기록이 없다 (백필 없음).
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0017"
down_revision: str | None = "0015"
branch_labels: str | None = None
depends_on: str | None = None

# 대상 테이블 이름
TABLE = "privacy_holds"


def upgrade() -> None:
    """보류 기록 테이블과 추가 전용·TRUNCATE 금지 트리거를 만든다."""
    op.create_table(
        TABLE,
        sa.Column(
            "event_id",
            sa.BigInteger().with_variant(sa.Integer(), "sqlite"),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column("session_id", sa.String(length=128), nullable=False),
        sa.Column("action", sa.String(length=16), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("actor", sa.Text(), nullable=False),
        sa.Column("at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["sessions.session_id"],
            name=op.f("fk_privacy_holds_session_id_sessions"),
        ),
        sa.PrimaryKeyConstraint("event_id", name=op.f("pk_privacy_holds")),
    )
    op.create_index(op.f("ix_privacy_holds_session_id"), TABLE, ["session_id"], unique=False)
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
    """보류 기록 테이블을 지운다. 기록이 하나라도 있으면 거부한다.

    기록을 지우면 보류 중인 세션의 승인 거부가 사라지고 사고 대응 감사 기록도 잃는다. 공용 트리거
    함수는 이전 리비전 소유라 지우지 않는다.
    """
    count = op.get_bind().execute(sa.text(f"SELECT count(*) FROM {TABLE}")).scalar_one()
    if count:
        raise RuntimeError(
            f"승인 보류 기록 {count}개가 있어 0017을 되돌릴 수 없습니다 "
            "(보류 중 승인 거부와 사고 대응 감사 기록을 잃는다)"
        )
    op.drop_index(op.f("ix_privacy_holds_session_id"), table_name=TABLE)
    op.drop_table(TABLE)
