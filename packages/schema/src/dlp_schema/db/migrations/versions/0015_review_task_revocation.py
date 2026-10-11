"""검수 작업 매체 회수 outbox 열 (ADR 0039).

Revision ID: 0015
Revises: 0014
Create Date: 2026-10-11

`review_tasks`에 작업 매체의 렌더 해시와 회수 요청·완료 표시 열을 더한다. 기존 작업은 렌더 해시를
추측해 채우지 않는다 (ADR 0033과 같은 원칙). 구 작업은 `dlp review revoke --include-legacy`로
운영자가 정리한다. status 값 `revoked`는 CHECK 제약이 없어 DDL이 필요 없다.
"""

from typing import Any

import sqlalchemy as sa
from alembic import op

revision: str = "0015"
down_revision: str | None = "0014"
branch_labels: str | None = None
depends_on: str | None = None

# 이 리비전이 더하는 열 (이름, 타입)
COLUMNS: tuple[tuple[str, sa.types.TypeEngine[Any]], ...] = (
    ("render_hash", sa.String(64)),
    ("revoke_requested_at", sa.DateTime(timezone=True)),
    ("revoke_reason", sa.Text()),
    ("media_revoked_at", sa.DateTime(timezone=True)),
    ("tool_revoked_at", sa.DateTime(timezone=True)),
    ("revoke_note", sa.Text()),
)


def upgrade() -> None:
    """회수 열을 더한다 (모두 NULL 허용, 백필 없음)."""
    with op.batch_alter_table("review_tasks") as batch:
        for name, kind in COLUMNS:
            batch.add_column(sa.Column(name, kind, nullable=True))


def downgrade() -> None:
    """회수 열을 지운다. 회수한 작업이나 끝나지 않은 회수가 있으면 거부한다.

    회수 대기 기록을 지우면 이미 배포한 매체의 노출이 잊힌다. 구 코드는 status='revoked' 행을 읽지
    못하므로 회수를 끝낸 행이 하나라도 있어도 되돌리지 않는다 (운영자가 직접 정리해야 한다).
    """
    pending = (
        op.get_bind()
        .execute(
            sa.text(
                "SELECT count(*) FROM review_tasks WHERE status = 'revoked' OR "
                "(revoke_requested_at IS NOT NULL AND "
                "(media_revoked_at IS NULL OR tool_revoked_at IS NULL))"
            )
        )
        .scalar_one()
    )
    if pending:
        raise RuntimeError(
            f"회수한 작업 또는 회수 대기 작업 {pending}개가 있어 0015를 되돌릴 수 없습니다 "
            "(구 코드는 revoked 상태를 읽지 못하고, 대기 기록을 지우면 노출이 잊힌다)"
        )
    with op.batch_alter_table("review_tasks") as batch:
        for name, _ in reversed(COLUMNS):
            batch.drop_column(name)
