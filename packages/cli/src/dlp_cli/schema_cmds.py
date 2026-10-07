"""스키마·온톨로지·DB 하위 명령 (WP1, ADR 0002).

등록하는 명령:
- `dlp schema export [--out] [--check]` — 계약 타입(`dlp_schema`)에서 JSON
  Schema(`schemas/*.schema.json`)를
  만든다. `--check`는 파일이 코드와 일치하는지만 본다 (`make schemas` / `make contracts`).
- `dlp ontology validate [--root]` — `config/ontology/*/manifest.yaml`의 모든 온톨로지 버전을
  검증한다.
- `dlp ontology register <버전>` — 온톨로지 버전을 DB(`ontology_versions` 테이블)에 등록한다.
- `dlp db upgrade [revision]` — Alembic 마이그레이션 적용 (`make db-upgrade`).

공개 함수:
- `database_url(explicit)` — 모든 `*_cmds.py`가 공유하는 DB URL 결정 규칙
  (`--url` > 환경 변수 `DLP_DATABASE_URL` > 개발용 기본값).
- `cmd_*` — 각 하위 명령의 실행 함수 (반환값은 종료 코드).

주의: 계약 타입을 바꾼 뒤에는 `make schemas`로 재생성해야 `make contracts`(= `schema export
--check`)가 통과한다.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import sqlalchemy as sa
from pydantic import ValidationError

from dlp_schema import find_ontology_dir, load_ontology, repo_root
from dlp_schema.db.migrate import upgrade
from dlp_schema.db.repository import register_ontology
from dlp_schema.jsonschema import stale_schemas, write_schemas

# 개발용 docker compose PostgreSQL(services/docker-compose.yml, .env.example 기본값)의 접속 URL.
# 운영에서는 반드시 `DLP_DATABASE_URL` 또는 `--url`로 덮어쓴다 (비밀번호가 개발용이다).
DEFAULT_DATABASE_URL = "postgresql+psycopg://dlp:dlp-dev-password@localhost:5432/dlp"


def database_url(explicit: str | None) -> str:
    """DB 접속 URL을 정한다.

    인자:
        explicit: 명령줄 `--url` 값. None이거나 빈 문자열이면 무시한다.

    반환: `explicit` → 환경 변수 `DLP_DATABASE_URL` → `DEFAULT_DATABASE_URL` 순으로 처음 있는 값
    (SQLAlchemy URL, 드라이버는 psycopg 3).
    """
    return explicit or os.environ.get("DLP_DATABASE_URL", DEFAULT_DATABASE_URL)


def cmd_schema_export(args: argparse.Namespace) -> int:
    """`dlp schema export`: 계약 타입의 JSON Schema 파일을 쓰거나(기본) 최신인지
        검사한다(`--check`).

    인자:
        args.out: 출력 디렉터리. None이면 `<저장소>/schemas`.
        args.check: 참이면 쓰지 않고, 코드에서 만든 스키마와 다른(낡은) 파일만 `[STALE]`로 출력한다.

    반환: 검사 모드에서 낡은 파일이 있으면 1, 그 밖에는 0.
    부작용: 쓰기 모드에서 `<out>/*.schema.json`을 덮어쓴다.
    """
    out = Path(args.out) if args.out else repo_root() / "schemas"
    if args.check:
        stale = stale_schemas(out)
        for name in stale:
            print(f"[STALE] {out / name}")
        if stale:
            print("`dlp schema export`로 다시 생성하세요")
        return 1 if stale else 0
    for path in write_schemas(out):
        print(f"wrote {path}")
    return 0


def cmd_ontology_validate(args: argparse.Namespace) -> int:
    """`dlp ontology validate`: 루트 아래 모든 온톨로지 버전 디렉터리를 불러 검증한다.

    `<root>/*/manifest.yaml`이 있는 디렉터리 하나가 온톨로지 한 버전이다. 하나가 실패해도
    나머지를 계속 검사해 결과를 모두 출력한다.

    인자:
        args.root: 온톨로지 루트. None이면 `<저장소>/config/ontology`.

    반환: 온톨로지가 하나도 없거나 하나라도 검증에 실패하면 1, 모두 통과하면 0.
    """
    root = Path(args.root) if args.root else repo_root() / "config" / "ontology"
    dirs = sorted(p.parent for p in root.glob("*/manifest.yaml"))
    if not dirs:
        print(f"온톨로지가 없습니다: {root}")
        return 1
    failed = 0
    for d in dirs:
        try:
            o = load_ontology(d)
        except (ValidationError, ValueError, FileNotFoundError) as exc:
            # 스키마 위반(ValidationError), 교차 참조 오류(ValueError), 빠진 파일을 실패로 센다
            failed += 1
            print(f"[FAIL] {d}: {exc}")
            continue
        print(
            f"[OK  ] {d.name}: v{o.version} ({o.status}) 작업 {len(o.tasks)}, "
            f"동사 {len(o.verbs)}, 객체 {len(o.objects)}, 이벤트 {len(o.events)}"
        )
    return 1 if failed else 0


def cmd_db_upgrade(args: argparse.Namespace) -> int:
    """`dlp db upgrade [revision]`: Alembic 마이그레이션을 `revision`(기본 `head`)까지 적용한다.

    부작용: DB 스키마 변경. 이미 적용된 리비전은 건너뛰므로 여러 번 돌려도 안전하다(멱등).
    """
    url = database_url(args.url)
    upgrade(url, args.revision)
    print(f"마이그레이션 완료: {args.revision}")
    return 0


def cmd_ontology_register(args: argparse.Namespace) -> int:
    """`dlp ontology register <버전>`: 온톨로지 한 버전을 불러 DB에 등록한다.

    인자:
        args.version: 등록할 온톨로지 버전 문자열 (`find_ontology_dir`가 디렉터리를 찾는다).
        args.root: 온톨로지 루트. None이면 `<저장소>/config/ontology`.
        args.url: DB URL (`database_url` 규칙).

    부작용: 한 트랜잭션 안에서 `register_ontology`로 `ontology_versions`에 쓴다.
    같은 버전을 다시 등록하면 내용이 같을 때 아무것도 하지 않고(멱등), 초안(draft)에 키를 덧붙이기만
    했으면 내용을 바꾸며(ADR 0028), 그 밖의 변경은 `register_ontology`가 오류를 낸다.
    """
    root = Path(args.root) if args.root else repo_root() / "config" / "ontology"
    ontology = load_ontology(find_ontology_dir(root, args.version))
    engine = sa.create_engine(database_url(args.url))
    with engine.begin() as conn:
        register_ontology(conn, ontology)
    engine.dispose()
    print(f"온톨로지 {ontology.version} 등록됨")
    return 0


def add_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # pyright: ignore[reportPrivateUsage]
    """`schema`, `ontology`, `db` 하위 명령 묶음을 최상위 파서에 등록한다."""
    schema = sub.add_parser("schema", help="계약 타입 JSON Schema")
    schema_sub = schema.add_subparsers(dest="schema_command", required=True)
    export = schema_sub.add_parser("export", help="JSON Schema 파일 생성")
    export.add_argument("--out", help="출력 디렉터리 (기본: <저장소>/schemas)")
    export.add_argument("--check", action="store_true", help="파일이 최신인지 검사만 한다")
    export.set_defaults(func=cmd_schema_export)

    onto = sub.add_parser("ontology", help="온톨로지")
    onto_sub = onto.add_subparsers(dest="ontology_command", required=True)
    validate = onto_sub.add_parser("validate", help="모든 온톨로지 버전 검증")
    validate.add_argument("--root", help="온톨로지 루트 (기본: <저장소>/config/ontology)")
    validate.set_defaults(func=cmd_ontology_validate)
    register = onto_sub.add_parser("register", help="온톨로지 버전을 DB에 등록")
    register.add_argument("version")
    register.add_argument("--root")
    register.add_argument("--url", help="DB URL (기본: DLP_DATABASE_URL)")
    register.set_defaults(func=cmd_ontology_register)

    db = sub.add_parser("db", help="메타데이터 DB")
    db_sub = db.add_subparsers(dest="db_command", required=True)
    up = db_sub.add_parser("upgrade", help="Alembic 마이그레이션 적용")
    up.add_argument("revision", nargs="?", default="head")
    up.add_argument("--url", help="DB URL (기본: DLP_DATABASE_URL)")
    up.set_defaults(func=cmd_db_upgrade)
