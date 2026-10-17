"""3D 궤적의 검수 물려받기 기록 (F-COE-19, ADR 0048)

Revision ID: 0023
Revises: 0022
Create Date: 2026-10-17

`trajectory_inheritance`(추가 전용)를 만든다. 3D 손 궤적이 그것을 계산한 2D 손 트랙의 검수를
물려받을 때(`apply_trajectory_follow`) 그 `record_review`와 같은 트랜잭션에 한 행을 쓴다. 백필은
없다: 이 리비전 전에는 3D가 검수 상태를 가질 경로가 없었다 (검수 도구에 3D가 없다).

수정·삭제는 0009의 `dlp_append_only()`, TRUNCATE는 0010의 `dlp_no_truncate()` 트리거로 막는다
(PostgreSQL).
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0023"
down_revision: str | None = "0022"
branch_labels: str | None = None
depends_on: str | None = None

# 대상 테이블 이름
TABLE = "trajectory_inheritance"


def upgrade() -> None:
    """기록 테이블과 추가 전용·TRUNCATE 금지 트리거를 만든다."""
    op.create_table(
        TABLE,
        sa.Column(
            "event_id",
            sa.BigInteger().with_variant(sa.Integer(), "sqlite"),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column("session_id", sa.String(length=128), nullable=False),
        sa.Column("trajectory_label_id", sa.String(length=128), nullable=False),
        sa.Column("source_label_id", sa.String(length=128), nullable=False),
        sa.Column(
            "via_label_ids",
            sa.JSON().with_variant(postgresql.JSONB(), "postgresql"),
            nullable=False,
        ),
        sa.Column("state", sa.String(length=32), nullable=False),
        sa.Column("reviewer_id", sa.String(length=128), nullable=False),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("mechanism", sa.String(length=32), nullable=False),
        sa.Column("rule_version", sa.String(length=64), nullable=False),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["sessions.session_id"],
            name=op.f("fk_trajectory_inheritance_session_id_sessions"),
        ),
        sa.ForeignKeyConstraint(
            ["trajectory_label_id"],
            ["label_records.label_id"],
            name=op.f("fk_trajectory_inheritance_trajectory_label_id_label_records"),
        ),
        sa.PrimaryKeyConstraint("event_id", name=op.f("pk_trajectory_inheritance")),
    )
    op.create_index(
        op.f("ix_trajectory_inheritance_session_id"), TABLE, ["session_id"], unique=False
    )
    op.create_index(
        op.f("ix_trajectory_inheritance_trajectory_label_id"),
        TABLE,
        ["trajectory_label_id"],
        unique=False,
    )
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
    """기록 테이블을 지운다. 기록이 하나라도 있으면 거부한다.

    그 기록을 지우면 물려받은 3D가 직접 검수한 것처럼 보여 출처가 끊겨도(2D가 고쳐져도) 내보내기가
    낡은 3D를 검수된 것으로 넣고, 물려받음 감사 기록도 잃는다. 공용 트리거 함수는 이전 리비전
    소유라 지우지 않는다.
    """
    count = op.get_bind().execute(sa.text(f"SELECT count(*) FROM {TABLE}")).scalar_one()
    if count:
        raise RuntimeError(
            f"3D 궤적 물려받기 기록 {count}개가 있어 0023을 되돌릴 수 없습니다 "
            "(끊긴 물려받음 판정과 감사 기록을 잃는다)"
        )
    op.drop_index(op.f("ix_trajectory_inheritance_trajectory_label_id"), table_name=TABLE)
    op.drop_index(op.f("ix_trajectory_inheritance_session_id"), table_name=TABLE)
    op.drop_table(TABLE)
