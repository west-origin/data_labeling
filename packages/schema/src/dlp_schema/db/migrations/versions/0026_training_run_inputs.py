"""학습 실행의 입력 지문·해시와 학습 세션 (ADR 0016 2026-11-05 보강, T2·T3)

Revision ID: 0026
Revises: 0025
Create Date: 2026-11-05

`training_runs`에 세 열을 더한다 (모두 NULL 허용).

- `example_fingerprints` (JSON): 학습기에 넘긴 예제마다의 내용 지문 (정렬). 다음 재학습이 직전
  예제가 바뀌었는지·새 예제가 몇 개인지 정한다 (T2).
- `examples_digest` (VARCHAR(64)): 지문 목록의 sha256 (감사용 한 값).
- `train_session_ids` (JSON): 예제가 나온 학습·검증 세션. 배포 직전에 철회를 다시 확인한다 (T3).

기존 행은 학습 당시 예제를 알 수 없으므로 백필하지 않는다 (NULL = 0026 이전 실행). 재학습 판단은 그런
직전 실행에 예전 개수 규칙을 쓰고, 배포는 그 데이터셋 버전의 학습·검증 분할 세션 전체로 철회를
확인한다.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0026"
down_revision: str | None = "0025"
branch_labels: str | None = None
depends_on: str | None = None

TABLE = "training_runs"
# PostgreSQL은 JSONB, 그 밖은 JSON (`tables.Json`과 같다)
JSON = sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")


def upgrade() -> None:
    """세 열을 더한다 (NULL 허용, 백필 없음)."""
    with op.batch_alter_table(TABLE) as batch:
        batch.add_column(sa.Column("example_fingerprints", JSON, nullable=True))
        batch.add_column(sa.Column("examples_digest", sa.String(length=64), nullable=True))
        batch.add_column(sa.Column("train_session_ids", JSON, nullable=True))


def downgrade() -> None:
    """세 열을 지운다. 값을 남긴 학습 실행이 있으면 거부한다.

    지우면 같은 입력의 중복 재학습 방지와 학습 세션 단위 철회 확인 근거가 사라진다 (배포 확인은
    데이터셋 버전 분할 전체로 넓어지지만 계보의 학습 입력 기록은 되살릴 수 없다). 운영자가 직접
    판단해 정리해야 한다.
    """
    recorded = (
        op.get_bind()
        .execute(
            sa.text(
                f"SELECT count(*) FROM {TABLE} "
                "WHERE examples_digest IS NOT NULL OR train_session_ids IS NOT NULL"
            )
        )
        .scalar_one()
    )
    if recorded:
        raise RuntimeError(
            f"학습 입력을 기록한 학습 실행 {recorded}개가 있어 0026을 되돌릴 수 없습니다 "
            "(되돌리면 학습 입력 지문과 학습 세션 기록이 사라진다)"
        )
    with op.batch_alter_table(TABLE) as batch:
        batch.drop_column("train_session_ids")
        batch.drop_column("examples_digest")
        batch.drop_column("example_fingerprints")
