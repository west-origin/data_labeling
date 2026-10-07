"""Alembic 실행 환경 (Alembic이 마이그레이션마다 이 파일을 실행한다).

`db.migrate.alembic_config`가 script_location을 이 디렉터리로 지정하면 Alembic이 이 모듈을 불러
오프라인(SQL 출력) 또는 온라인(DB 연결) 모드로 `versions/`의 리비전을 실행한다.

- `target_metadata`는 `db.tables.metadata`다 (autogenerate·비교 기준).
- `compare_type=True`: 열 타입 차이도 비교한다 (예: VARCHAR(128) → TEXT).
- 호출자가 `config.attributes["connection"]`에 연결을 넣어 주면 그 연결을 쓰고, 아니면 URL로 엔진을
  만든다 (NullPool: 마이그레이션 뒤 연결을 남기지 않는다).
"""

from __future__ import annotations

from alembic import context
from sqlalchemy import engine_from_config, pool

from dlp_schema.db.tables import metadata

# Alembic이 넘겨주는 설정 (alembic_config에서 만든 Config)
config = context.config


def run_migrations_offline() -> None:
    """DB 연결 없이 SQL 문을 출력하는 모드 (`alembic upgrade --sql`). 값은 리터럴로 박는다."""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=metadata,
        literal_binds=True,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """DB에 연결해 마이그레이션을 실행한다. 주어진 연결이 있으면 재사용한다."""
    connectable = config.attributes.get("connection")
    if connectable is None:
        connectable = engine_from_config(
            config.get_section(config.config_ini_section, {}),
            prefix="sqlalchemy.",
            poolclass=pool.NullPool,
        )
        with connectable.connect() as connection:
            _run(connection)
    else:
        _run(connectable)


def _run(connection: object) -> None:
    """연결 하나에서 한 트랜잭션으로 마이그레이션을 실행한다 (PostgreSQL은 DDL도 트랜잭션 안)."""
    context.configure(connection=connection, target_metadata=metadata, compare_type=True)  # type: ignore[arg-type]
    with context.begin_transaction():
        context.run_migrations()


# 모듈 최상위: Alembic이 이 파일을 실행할 때 모드에 따라 바로 마이그레이션을 돈다
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
