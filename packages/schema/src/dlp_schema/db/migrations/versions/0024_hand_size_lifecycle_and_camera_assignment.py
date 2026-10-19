"""작업자 손 크기 출처·무효 기록과 수집 뒤 세션 기기 연결 (ADR 0052)

Revision ID: 0024
Revises: 0023
Create Date: 2026-10-19

- `worker_hand_sizes.estimator`(NULL 허용): 추정기 판. 잰 값과 이 리비전 이전 추정은 NULL이다.
- `worker_hand_size_sources`: 추정 표본을 낸 세션 (등록 ID, 세션 ID). 출처 세션이 사용 중지되면 그
  추정은 손 크기 결정에서 빠진다. 이전 추정은 출처를 모르므로 백필하지 않는다 (ADR 0052 4절).
- `worker_hand_size_invalidations`: 손 크기 등록 무효 기록 (등록 하나 또는 작업자 전체). 동의 철회·
  삭제 요구를 행 수정·삭제 없이 처리한다 (`dlp calib palm-revoke`).
- `session_camera_assignments`: 매니페스트에 camera_id가 없던 세션을 기기에 잇는 기록
  (`dlp calib assign`).

셋 다 추가 전용이다. 수정·삭제는 0009의 `dlp_append_only()`, TRUNCATE는 0010의 `dlp_no_truncate()`
트리거로 막는다 (PostgreSQL). 열 추가(DDL)는 0022의 행 트리거와 무관하다.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0024"
down_revision: str | None = "0023"
branch_labels: str | None = None
depends_on: str | None = None

TABLES = (
    "worker_hand_size_sources",
    "worker_hand_size_invalidations",
    "session_camera_assignments",
)


def _create_tables() -> None:
    """세 기록 테이블과 색인을 만든다."""
    op.create_table(
        "worker_hand_size_sources",
        sa.Column("record_id", sa.String(length=128), nullable=False),
        sa.Column("session_id", sa.String(length=128), nullable=False),
        sa.ForeignKeyConstraint(
            ["record_id"],
            ["worker_hand_sizes.record_id"],
            name=op.f("fk_worker_hand_size_sources_record_id_worker_hand_sizes"),
        ),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["sessions.session_id"],
            name=op.f("fk_worker_hand_size_sources_session_id_sessions"),
        ),
        sa.PrimaryKeyConstraint(
            "record_id", "session_id", name=op.f("pk_worker_hand_size_sources")
        ),
    )
    op.create_index(
        op.f("ix_worker_hand_size_sources_session_id"),
        "worker_hand_size_sources",
        ["session_id"],
        unique=False,
    )
    op.create_table(
        "worker_hand_size_invalidations",
        sa.Column("invalidation_id", sa.String(length=128), nullable=False),
        sa.Column("worker_id", sa.String(length=128), nullable=False),
        sa.Column("record_id", sa.String(length=128), nullable=True),
        sa.Column("reason", sa.String(length=500), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["record_id"],
            ["worker_hand_sizes.record_id"],
            name=op.f("fk_worker_hand_size_invalidations_record_id_worker_hand_sizes"),
        ),
        sa.PrimaryKeyConstraint("invalidation_id", name=op.f("pk_worker_hand_size_invalidations")),
    )
    op.create_index(
        op.f("ix_worker_hand_size_invalidations_worker_id"),
        "worker_hand_size_invalidations",
        ["worker_id"],
        unique=False,
    )
    op.create_table(
        "session_camera_assignments",
        sa.Column("assignment_id", sa.String(length=128), nullable=False),
        sa.Column("session_id", sa.String(length=128), nullable=False),
        sa.Column("camera_id", sa.String(length=128), nullable=False),
        sa.Column("reason", sa.String(length=500), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["sessions.session_id"],
            name=op.f("fk_session_camera_assignments_session_id_sessions"),
        ),
        sa.PrimaryKeyConstraint("assignment_id", name=op.f("pk_session_camera_assignments")),
    )
    op.create_index(
        op.f("ix_session_camera_assignments_session_id"),
        "session_camera_assignments",
        ["session_id"],
        unique=False,
    )


def upgrade() -> None:
    """추정기 판 열, 세 기록 테이블, 추가 전용·TRUNCATE 금지 트리거를 만든다."""
    with op.batch_alter_table("worker_hand_sizes") as batch:
        batch.add_column(sa.Column("estimator", sa.String(length=256), nullable=True))
    _create_tables()
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
    """세 테이블과 열을 지운다. 기록이 하나라도 있거나 추정기 판이 적힌 등록이 있으면 거부한다.

    무효 기록을 지우면 동의를 철회한 작업자의 손 크기가 다시 3D에 쓰이고, 출처 세션을 지우면 사용
    중지된 세션에서 나온 추정이 다시 쓰이며, 기기 연결을 지우면 연결한 세션의 3D 입력이 바뀐다.
    공용 트리거 함수는 이전 리비전 소유라 지우지 않는다.
    """
    bind = op.get_bind()
    for table in TABLES:
        count = bind.execute(sa.text(f"SELECT count(*) FROM {table}")).scalar_one()
        if count:
            raise RuntimeError(
                f"{table}에 기록 {count}개가 있어 0024를 되돌릴 수 없습니다 "
                "(손 크기 무효·출처, 세션 기기 연결 기록을 잃는다)"
            )
    stamped = bind.execute(
        sa.text("SELECT count(*) FROM worker_hand_sizes WHERE estimator IS NOT NULL")
    ).scalar_one()
    if stamped:
        raise RuntimeError(
            f"worker_hand_sizes에 추정기 판이 적힌 등록 {stamped}개가 있어 0024를 되돌릴 수 없습니다"
        )
    op.drop_index(
        op.f("ix_session_camera_assignments_session_id"), table_name="session_camera_assignments"
    )
    op.drop_table("session_camera_assignments")
    op.drop_index(
        op.f("ix_worker_hand_size_invalidations_worker_id"),
        table_name="worker_hand_size_invalidations",
    )
    op.drop_table("worker_hand_size_invalidations")
    op.drop_index(
        op.f("ix_worker_hand_size_sources_session_id"), table_name="worker_hand_size_sources"
    )
    op.drop_table("worker_hand_size_sources")
    with op.batch_alter_table("worker_hand_sizes") as batch:
        batch.drop_column("estimator")
