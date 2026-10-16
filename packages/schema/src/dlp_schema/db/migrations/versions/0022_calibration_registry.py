"""카메라 내부 파라미터·작업자 손 크기 등록 (ADR 0045)

Revision ID: 0022
Revises: 0021
Create Date: 2026-10-16

`camera_calibrations`(기기별 내부 파라미터, `dlp calib intrinsics|import`)와 `worker_hand_sizes`
(작업자 손바닥 길이, `dlp calib palm|palm-estimate`)를 만든다. 둘 다 추가 전용이다. 늦게 등록한 값이
앞서고, 바꾸려면 새 행을 더한다. 백필은 없다 (예전 세션은 등록이 없으면 지금처럼 세션 값 또는 화각
근사·정책 손 크기를 쓴다).

수정·삭제는 0009의 `dlp_append_only()`, TRUNCATE는 0010의 `dlp_no_truncate()` 트리거로 막는다
(PostgreSQL).
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0022"
down_revision: str | None = "0021"
branch_labels: str | None = None
depends_on: str | None = None

TABLES = ("camera_calibrations", "worker_hand_sizes")


def upgrade() -> None:
    """두 등록 테이블과 추가 전용·TRUNCATE 금지 트리거를 만든다."""
    op.create_table(
        "camera_calibrations",
        sa.Column("calibration_id", sa.String(length=128), nullable=False),
        sa.Column("camera_id", sa.String(length=128), nullable=False),
        sa.Column(
            "intrinsics",
            sa.JSON().with_variant(postgresql.JSONB(), "postgresql"),
            nullable=False,
        ),
        sa.Column("method", sa.String(length=32), nullable=False),
        sa.Column("rms_px", sa.Float(), nullable=True),
        sa.Column("n_views", sa.Integer(), nullable=True),
        sa.Column("source_sha256", sa.String(length=64), nullable=True),
        sa.Column("valid_from", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("calibration_id", name=op.f("pk_camera_calibrations")),
    )
    op.create_index(
        op.f("ix_camera_calibrations_camera_id"), "camera_calibrations", ["camera_id"], unique=False
    )
    op.create_table(
        "worker_hand_sizes",
        sa.Column("record_id", sa.String(length=128), nullable=False),
        sa.Column("worker_id", sa.String(length=128), nullable=False),
        sa.Column("palm_m", sa.Float(), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("n_samples", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("record_id", name=op.f("pk_worker_hand_sizes")),
    )
    op.create_index(
        op.f("ix_worker_hand_sizes_worker_id"), "worker_hand_sizes", ["worker_id"], unique=False
    )
    if op.get_bind().dialect.name == "postgresql":
        for table in TABLES:
            op.execute(
                f"CREATE TRIGGER {table}_append_only BEFORE UPDATE OR DELETE ON {table} "
                "FOR EACH ROW EXECUTE FUNCTION dlp_append_only()"
            )
            op.execute(
                f"CREATE TRIGGER {table}_no_truncate BEFORE TRUNCATE ON {table} "
                "FOR EACH STATEMENT EXECUTE FUNCTION dlp_no_truncate()"
            )


def downgrade() -> None:
    """두 테이블을 지운다. 등록이 하나라도 있으면 거부한다 (3D 단계 입력과 감사 기록을 잃는다)."""
    bind = op.get_bind()
    for table in TABLES:
        count = bind.execute(sa.text(f"SELECT count(*) FROM {table}")).scalar_one()
        if count:
            raise RuntimeError(
                f"{table}에 등록 {count}개가 있어 0022를 되돌릴 수 없습니다 "
                "(3D 단계가 쓴 내부 파라미터·손 크기 기록을 잃는다)"
            )
    op.drop_index(op.f("ix_worker_hand_sizes_worker_id"), table_name="worker_hand_sizes")
    op.drop_table("worker_hand_sizes")
    op.drop_index(op.f("ix_camera_calibrations_camera_id"), table_name="camera_calibrations")
    op.drop_table("camera_calibrations")
