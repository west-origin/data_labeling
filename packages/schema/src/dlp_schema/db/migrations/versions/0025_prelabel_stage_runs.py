"""프리라벨 단계 실행 기록 (결과 0개 포함, ADR 0015 개정, F-PAR-75)

Revision ID: 0025
Revises: 0024
Create Date: 2026-10-28

- `prelabel_stage_runs`: 프리라벨 단계(예측기·마스크·3D·자세·월드 좌표·접촉)가 그 버전으로 계산을
  마친 기록. 결과가 0개면 라벨 이력에 그 버전이 남지 않아 매 실행 원본을 다시 받고 다시 돌던 것을
  막는다. 라벨 계약(`label_records`)은 바꾸지 않는다. 이전 실행은 기록이 없으므로 백필하지 않는다
  (결과가 있던 단계는 라벨 이력으로 이미 건너뛰고, 결과 0개였던 단계는 다음 실행 한 번 더 돈 뒤
  기록이 남는다).

추가 전용이다. 수정·삭제는 0009의 `dlp_append_only()`, TRUNCATE는 0010의 `dlp_no_truncate()`
트리거로 막는다 (PostgreSQL).
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0025"
down_revision: str | None = "0024"
branch_labels: str | None = None
depends_on: str | None = None

TABLE = "prelabel_stage_runs"
INDEX = "ix_prelabel_stage_runs_session_stage"


def upgrade() -> None:
    """실행 기록 표·색인과 추가 전용·TRUNCATE 금지 트리거를 만든다."""
    op.create_table(
        TABLE,
        sa.Column(
            "event_id",
            sa.BigInteger().with_variant(sa.Integer(), "sqlite"),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column("session_id", sa.String(length=128), nullable=False),
        sa.Column("stream_id", sa.String(length=128), nullable=False),
        sa.Column("stage", sa.String(length=128), nullable=False),
        sa.Column("model_version", sa.Text(), nullable=False),
        sa.Column("produced", sa.Integer(), nullable=False),
        sa.Column("at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["session_id", "stream_id"],
            ["streams.session_id", "streams.stream_id"],
            name=op.f("fk_prelabel_stage_runs_session_id_streams"),
        ),
        sa.PrimaryKeyConstraint("event_id", name=op.f("pk_prelabel_stage_runs")),
    )
    op.create_index(INDEX, TABLE, ["session_id", "stream_id", "stage"])
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
    """표를 지운다. 기록이 하나라도 있으면 거부한다.

    추가 전용 기록이고, 지우면 결과 0개로 끝난 단계가 다시 원본을 받고 돈다. 공용 트리거 함수는
    이전 리비전 소유라 지우지 않는다.
    """
    count = op.get_bind().execute(sa.text(f"SELECT count(*) FROM {TABLE}")).scalar_one()
    if count:
        raise RuntimeError(
            f"{TABLE}에 기록 {count}개가 있어 0025를 되돌릴 수 없습니다 "
            "(추가 전용 프리라벨 단계 실행 기록을 잃는다)"
        )
    op.drop_index(INDEX, table_name=TABLE)
    op.drop_table(TABLE)
