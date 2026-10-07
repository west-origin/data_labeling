"""관계 도출·커버리지 하위 명령."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime

import sqlalchemy as sa

from dlp_cli.schema_cmds import database_url
from dlp_relations.policy import load_policy
from dlp_relations.runner import run_relations
from dlp_schema import load_ontology, repo_root


def cmd_run(args: argparse.Namespace) -> int:
    root = repo_root()
    policy = load_policy(root)
    ontology = load_ontology(root / "config/ontology/v1")
    engine = sa.create_engine(database_url(args.url))
    with engine.begin() as conn:
        s = run_relations(conn, args.session_id, ontology, policy, datetime.now(UTC))
    engine.dispose()
    print(f"정책 {s.version}: 관계 {s.relations}개")
    print(f"레코드: 새로 {s.inserted}, 유지 {s.kept}, 삭제 표시 {s.retracted}")
    if s.skipped_by_review:
        print(f"검수자가 고치거나 지운 관계 {s.skipped_by_review}개는 다시 넣지 않음")
    for pair, ratio in s.coverage.items():
        print(f"커버리지 {pair}: {ratio:.1%}")
    return 0


def add_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # pyright: ignore[reportPrivateUsage]
    rel = sub.add_parser("relations", help="관계 도출과 표면 커버리지")
    rsub = rel.add_subparsers(dest="relations_command", required=True)
    run = rsub.add_parser("run", help="규칙으로 관계를 만들고 커버리지를 계산 (멱등)")
    run.add_argument("session_id")
    run.add_argument("--url", help="DB URL (기본: DLP_DATABASE_URL)")
    run.set_defaults(func=cmd_run)
