"""데이터셋·계보 하위 명령."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime

import sqlalchemy as sa

from dlp_cli.schema_cmds import database_url
from dlp_datasets.build import build_dataset_version
from dlp_datasets.lineage import session_lineage, withdraw_session
from dlp_datasets.policy import load_policy
from dlp_datasets.snapshot import LakeFSSnapshotStore
from dlp_datasets.splitter import propose_golden
from dlp_schema import repo_root
from dlp_schema.db.repository import get_session, insert_golden_set, list_session_ids
from dlp_schema.lineage import GoldenSet
from dlp_schema.session import Domain


def _engine(args: argparse.Namespace) -> sa.Engine:
    return sa.create_engine(database_url(args.url))


def cmd_golden(args: argparse.Namespace) -> int:
    policy = load_policy(repo_root())
    engine = _engine(args)
    with engine.begin() as conn:
        sessions = [get_session(conn, sid) for sid in list_session_ids(conn, args.ontology_version)]
        proposed = propose_golden(
            sessions, args.domain, args.count or policy.golden_sessions_per_domain, seed=args.seed
        )
        if args.create:
            insert_golden_set(
                conn,
                GoldenSet(
                    version=args.create,
                    domain=Domain(args.domain),
                    session_ids=tuple(proposed),
                    created_at=datetime.now(UTC),
                    note=args.note,
                ),
            )
    engine.dispose()
    print(
        f"{args.domain} 골든셋 {'생성 ' + args.create if args.create else '제안'}: "
        f"세션 {len(proposed)}개"
    )
    for sid in proposed:
        print(f"  {sid}")
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    root = repo_root()
    policy = load_policy(root)
    lp = policy.lakefs
    snapshots = LakeFSSnapshotStore.from_env(
        repository=lp.repository, branch=lp.branch, storage_namespace=lp.storage_namespace
    )
    engine = _engine(args)
    with engine.begin() as conn:
        result = build_dataset_version(
            conn,
            snapshots,
            policy,
            version_id=args.version_id,
            ontology_version=args.ontology_version,
            golden_set_version=args.golden,
            domain=args.domain,
            parent_version_id=args.parent,
            seed=args.seed,
            now=datetime.now(UTC),
        )
    engine.dispose()
    r = result.report
    print(f"{result.version.version_id}: {result.version.snapshot_uri}")
    print(f"  분할 {r.counts}, 검증 비율 {r.val_ratio:.3f}, holdout 비율 {r.holdout_ratio:.3f}")
    print(
        f"  사용 중지로 제외 {len(result.version.excluded_sessions)}개, "
        f"라벨 {sum(result.label_counts.values())}개"
    )
    return 0


def cmd_withdraw(args: argparse.Namespace) -> int:
    engine = _engine(args)
    with engine.begin() as conn:
        lineage = withdraw_session(conn, args.session_id, args.reason, datetime.now(UTC))
    engine.dispose()
    print(f"{args.session_id}: 사용 중지. 이후 버전·내보내기에서 자동 제외")
    _print_lineage(
        lineage.dataset_versions,
        [r.run_id for r in lineage.training_runs],
        [e.export_id for e in lineage.exports],
        "이미 들어간 곳",
    )
    return 0


def cmd_lineage(args: argparse.Namespace) -> int:
    engine = _engine(args)
    with engine.connect() as conn:
        lineage = session_lineage(conn, args.session_id)
    engine.dispose()
    print(f"{args.session_id}: {lineage.lifecycle.value}")
    _print_lineage(
        lineage.dataset_versions,
        [r.run_id for r in lineage.training_runs],
        [e.export_id for e in lineage.exports],
        "계보",
    )
    return 0


def _print_lineage(versions: list[str], runs: list[str], exports: list[str], title: str) -> None:
    print(
        f"  {title}: 데이터셋 버전 {versions or '-'}, 학습 실행 {runs or '-'}, "
        f"내보내기 {exports or '-'}"
    )


def add_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # pyright: ignore[reportPrivateUsage]
    ds = sub.add_parser("dataset", help="데이터셋 버전·골든셋·사용 중지")
    dsub = ds.add_subparsers(dest="dataset_command", required=True)

    g = dsub.add_parser("golden", help="도메인 골든셋 제안 (--create로 확정)")
    g.add_argument("--domain", required=True, choices=["cleaning", "caregiving", "nursing"])
    g.add_argument("--count", type=int, help="목표 세션 수 (기본: 정책 값)")
    g.add_argument("--create", metavar="VERSION", help="제안을 이 버전 이름으로 확정")
    g.add_argument("--note", default="")
    g.add_argument("--ontology-version", default="1.0.0")
    g.add_argument("--seed", type=int, default=0)
    g.set_defaults(func=cmd_golden)

    b = dsub.add_parser("build", help="데이터셋 버전 빌드 (lakeFS 커밋)")
    b.add_argument("version_id")
    b.add_argument("--ontology-version", default="1.0.0")
    b.add_argument("--golden", help="골든셋 버전")
    b.add_argument("--domain", choices=["cleaning", "caregiving", "nursing"])
    b.add_argument("--parent")
    b.add_argument("--seed", type=int, default=0)
    b.set_defaults(func=cmd_build)

    w = dsub.add_parser("withdraw", help="세션 사용 중지 (동의 철회·삭제 요청)")
    w.add_argument("session_id")
    w.add_argument("--reason", required=True)
    w.set_defaults(func=cmd_withdraw)

    lin = sub.add_parser("lineage", help="세션 계보: 데이터셋 버전 → 학습 실행 → 내보내기")
    lin.add_argument("session_id")
    lin.set_defaults(func=cmd_lineage)

    for p in (g, b, w, lin):
        p.add_argument("--url", help="DB URL (기본: DLP_DATABASE_URL)")
