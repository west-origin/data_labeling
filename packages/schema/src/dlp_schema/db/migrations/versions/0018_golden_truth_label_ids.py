"""골든셋 버전의 정답 라벨 ID 고정 열 (ADR 0041).

Revision ID: 0018
Revises: 0017
Create Date: 2026-10-11

`golden_sets`에 만들 때 고정한 정답 라벨 ID 목록(`truth_label_ids`, JSON)을 더한다. 기존 행은
정답을 고정한 시점을 알 수 없으므로 채우지 않는다 (NULL = 고정하지 않은 이전 형식). 그런 골든셋은
`dlp dataset golden --from <버전> --create <새 버전>`으로 정답을 고정한 새 버전을 만들어 쓴다.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0018"
down_revision: str | None = "0017"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """정답 라벨 ID 열을 더한다 (NULL 허용, 백필 없음)."""
    with op.batch_alter_table("golden_sets") as batch:
        batch.add_column(
            sa.Column(
                "truth_label_ids",
                sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
                nullable=True,
            )
        )


def downgrade() -> None:
    """정답 라벨 ID 열을 지운다. 정답을 고정한 골든셋이 있으면 거부한다.

    열을 지우면 그 골든셋 버전이 현재 라벨로 정답을 읽는 이전 동작으로 돌아가 같은 버전의 정답이
    바뀔 수 있다 (재현성 상실). 운영자가 직접 판단해 정리해야 한다.
    """
    frozen = (
        op.get_bind()
        .execute(sa.text("SELECT count(*) FROM golden_sets WHERE truth_label_ids IS NOT NULL"))
        .scalar_one()
    )
    if frozen:
        raise RuntimeError(
            f"정답을 고정한 골든셋 {frozen}개가 있어 0018을 되돌릴 수 없습니다 "
            "(되돌리면 같은 골든 버전의 정답이 현재 라벨로 바뀐다)"
        )
    with op.batch_alter_table("golden_sets") as batch:
        batch.drop_column("truth_label_ids")
