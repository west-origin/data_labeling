"""AI 1차 검수 판정 단위의 온톨로지 내용 해시 열 (F-AIR-32, ADR 0040).

Revision ID: 0020
Revises: 0019
Create Date: 2026-10-15

`ai_review_units`에 판정에 쓴 온톨로지 내용 해시(`ontology_digest`, sha256 64자)를 더한다. 같은
버전의 draft 온톨로지에 항목을 덧붙이면(`register_ontology`) 프롬프트 개체표·응답 스키마가 바뀌는데
단위 키가 같아 다시 판정하지 않았다. 이제 단위 키와 표시 판정의 유효 조건에 내용 해시가 든다.
기존 행은 어떤 내용으로 판정했는지 알 수 없어 채우지 않는다 (NULL = 이 열 전에 기록한 단위). 그런
단위는 표시 프로필에서 빠지고 다음 `dlp aireview run`이 새 키로 다시 판정한다.
표는 추가 전용이다 (0019 트리거). 열 추가는 행을 바꾸지 않는다.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0020"
down_revision: str | None = "0019"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """내용 해시 열을 더한다 (NULL 허용, 백필 없음)."""
    with op.batch_alter_table("ai_review_units") as batch:
        batch.add_column(sa.Column("ontology_digest", sa.String(64), nullable=True))


def downgrade() -> None:
    """내용 해시 열을 지운다 (판정 기록 자체는 남는다. 단위 키는 그대로라 재판정이 생기지 않는다)."""
    with op.batch_alter_table("ai_review_units") as batch:
        batch.drop_column("ontology_digest")
