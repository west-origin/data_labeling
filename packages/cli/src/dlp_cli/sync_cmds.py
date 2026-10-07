"""동기화 하위 명령."""

from __future__ import annotations

import argparse

import sqlalchemy as sa

from dlp_cli.schema_cmds import database_url
from dlp_media.storage import store_from_spec
from dlp_schema import load_config, repo_root
from dlp_sync.policy import load_policy
from dlp_sync.runner import adjust, run_sync


def cmd_sync_run(args: argparse.Namespace) -> int:
    root = repo_root()
    policy = load_policy(root / "config" / "policies" / "sync.yaml")
    raw = store_from_spec(args.store, load_config(root / "config" / "defaults.yaml").buckets.raw)
    engine = sa.create_engine(database_url(args.url))
    with engine.begin() as conn:
        synced, report = run_sync(conn, args.session_id, raw, policy)
    engine.dispose()
    for r in report.streams:
        s = synced.stream(r.stream_id)
        tried = ", ".join(f"{a.method} {a.confidence:.2f}" for a in r.attempts)
        print(
            f"{r.stream_id:<14} {s.sync_method.value:<13} offset {s.offset_ms:9.2f} ms "
            f"drift {(s.clock_scale - 1) * 1e6:7.2f} ppm  [{tried}]"
        )
    return 0


def cmd_sync_adjust(args: argparse.Namespace) -> int:
    engine = sa.create_engine(database_url(args.url))
    with engine.begin() as conn:
        adjust(conn, args.session_id, args.stream_id, args.adjustment_ms)
    engine.dispose()
    print(f"{args.session_id}/{args.stream_id}: 수동 조정 {args.adjustment_ms} ms")
    return 0


def add_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # pyright: ignore[reportPrivateUsage]
    sync = sub.add_parser("sync", help="스트림 동기화")
    sync_sub = sync.add_subparsers(dest="sync_command", required=True)
    run = sync_sub.add_parser("run", help="세션 동기화 (DB 갱신 + 보고서 저장)")
    run.add_argument("session_id")
    run.add_argument("--store", default="s3", help="'s3' 또는 'local:<디렉터리>'")
    run.add_argument("--url", help="DB URL (기본: DLP_DATABASE_URL)")
    run.set_defaults(func=cmd_sync_run)
    adj = sync_sub.add_parser("adjust", help="사람이 정한 미세 조정값 기록")
    adj.add_argument("session_id")
    adj.add_argument("stream_id")
    adj.add_argument("adjustment_ms", type=float)
    adj.add_argument("--url", help="DB URL (기본: DLP_DATABASE_URL)")
    adj.set_defaults(func=cmd_sync_adjust)
