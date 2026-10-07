"""스키마·온톨로지·DB 하위 명령."""

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

DEFAULT_DATABASE_URL = "postgresql+psycopg://dlp:dlp-dev-password@localhost:5432/dlp"


def database_url(explicit: str | None) -> str:
    return explicit or os.environ.get("DLP_DATABASE_URL", DEFAULT_DATABASE_URL)


def cmd_schema_export(args: argparse.Namespace) -> int:
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
            failed += 1
            print(f"[FAIL] {d}: {exc}")
            continue
        print(
            f"[OK  ] {d.name}: v{o.version} ({o.status}) 작업 {len(o.tasks)}, "
            f"동사 {len(o.verbs)}, 객체 {len(o.objects)}, 이벤트 {len(o.events)}"
        )
    return 1 if failed else 0


def cmd_db_upgrade(args: argparse.Namespace) -> int:
    url = database_url(args.url)
    upgrade(url, args.revision)
    print(f"마이그레이션 완료: {args.revision}")
    return 0


def cmd_ontology_register(args: argparse.Namespace) -> int:
    root = Path(args.root) if args.root else repo_root() / "config" / "ontology"
    ontology = load_ontology(find_ontology_dir(root, args.version))
    engine = sa.create_engine(database_url(args.url))
    with engine.begin() as conn:
        register_ontology(conn, ontology)
    engine.dispose()
    print(f"온톨로지 {ontology.version} 등록됨")
    return 0


def add_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # pyright: ignore[reportPrivateUsage]
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
