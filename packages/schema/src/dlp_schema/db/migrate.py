"""Alembic 마이그레이션 실행. alembic.ini 없이 코드에서 설정한다.

위치
    `dlp db upgrade`(`make db-upgrade`)와 DB 테스트(`tests/test_db.py`)가 부른다. 마이그레이션
    스크립트는 `db/migrations/versions/NNNN_*.py`에 있고, 대상 메타데이터는 `db.tables.metadata`다.

주요 이름
    - `MIGRATIONS_DIR`: 마이그레이션 스크립트 디렉터리 (env.py, versions/).
    - `alembic_config(url)`: 코드로 만든 Alembic 설정.
    - `upgrade(url, revision)`, `downgrade(url, revision)`.

주의
    - 새 리비전은 `db.tables`를 바꾼 것과 같은 커밋에 넣는다 (테스트가 둘의 일치를 검사한다).
    - PostgreSQL 전용 트리거(라벨 불변, 운영 기록 추가 전용)는 SQLite에서는 만들어지지 않는다.
"""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config

# 이 파일 옆의 migrations/ (패키지에 포함되어 설치 위치와 무관하게 찾는다)
MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def alembic_config(url: str) -> Config:
    """DB URL로 Alembic 설정 객체를 만든다 (alembic.ini 없음).

    Args:
        url: SQLAlchemy DB URL (예: `postgresql+psycopg://user:pw@host/db`, `sqlite:///path`).

    Returns:
        script_location과 sqlalchemy.url을 채운 `Config`.
    """
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    # Alembic 설정은 ConfigParser 보간을 쓰므로 URL의 `%`(비밀번호의 URL 인코딩 등)를 `%%`로
    # 이스케이프한다
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    return cfg


def upgrade(url: str, revision: str = "head") -> None:
    """DB를 `revision`(기본 최신 head)까지 올린다. 이미 그 리비전이면 아무것도 하지 않는다."""
    command.upgrade(alembic_config(url), revision)


def downgrade(url: str, revision: str) -> None:
    """DB를 `revision`까지 내린다 (`"base"`면 모든 테이블 삭제). 데이터가 사라질 수 있다."""
    command.downgrade(alembic_config(url), revision)
