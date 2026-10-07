## Alembic 새 리비전 템플릿 (`alembic revision`이 versions/NNNN_*.py를 만들 때 쓴다).
## `##`로 시작하는 줄은 Mako 주석이라 생성 파일에 들어가지 않는다.
## 규칙: 리비전 ID는 4자리 일련번호, db/tables.py 변경과 함께 커밋, PostgreSQL 전용 DDL(트리거 등)은
## `op.get_bind().dialect.name == "postgresql"`로 감싸 SQLite 비교 테스트가 돌게 한다.
"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

${imports if imports else ""}
revision: str = ${repr(up_revision)}
down_revision: str | None = ${repr(down_revision)}
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    ${downgrades if downgrades else "pass"}
