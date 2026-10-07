"""프라이버시 게이트 하위 명령."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime

import sqlalchemy as sa

from dlp_cli.schema_cmds import database_url
from dlp_media.storage import store_from_spec
from dlp_privacy.detectors import build_detectors
from dlp_privacy.policy import load_policy
from dlp_privacy.runner import approve_session, detect_session, render_session
from dlp_schema import load_config, repo_root


def _engine(args: argparse.Namespace) -> sa.Engine:
    return sa.create_engine(database_url(args.url))


def cmd_detect(args: argparse.Namespace) -> int:
    root = repo_root()
    policy = load_policy(root)
    raw = store_from_spec(args.store, load_config(root / "config" / "defaults.yaml").buckets.raw)
    detectors, missing = build_detectors(policy, root)
    for name, reason in missing.items():
        print(f"[탐지기 없음] {name}: {reason}")
    engine = _engine(args)
    with engine.begin() as conn:
        s = detect_session(
            conn, args.session_id, raw, detectors, missing, policy, datetime.now(UTC)
        )
    engine.dispose()
    for stream, n in s.detected.items():
        print(f"{stream}: 블러 트랙 {n}개")
    for stream in s.skipped:
        print(f"{stream}: 같은 모델 버전 결과가 있어 건너뜀")
    for target, reason in s.missing.items():
        print(f"[전수 검수 필요] {target}: 쓸 수 있는 탐지기가 없음 ({reason})")
    return 0


def cmd_approve(args: argparse.Namespace) -> int:
    engine = _engine(args)
    with engine.begin() as conn:
        session = approve_session(conn, args.session_id)
    engine.dispose()
    print(f"{session.session_id}: 프라이버시 승인 ({session.lifecycle_state.value})")
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    root = repo_root()
    buckets = load_config(root / "config" / "defaults.yaml").buckets
    raw = store_from_spec(args.store, buckets.raw)
    labeling = store_from_spec(args.store, buckets.labeling)
    engine = _engine(args)
    with engine.begin() as conn:
        uris = render_session(conn, args.session_id, raw, labeling, load_policy(root))
    engine.dispose()
    for stream, uri in uris.items():
        print(f"{stream}: {uri}")
    return 0


def add_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # pyright: ignore[reportPrivateUsage]
    privacy = sub.add_parser("privacy", help="프라이버시 게이트")
    psub = privacy.add_subparsers(dest="privacy_command", required=True)
    for name, func, help_text in (
        ("detect", cmd_detect, "블러 대상 자동 탐지 → 블러 트랙 라벨 + 검수 우선 구간"),
        ("approve", cmd_approve, "모든 블러 라벨이 사람 검수를 거쳤으면 프라이버시 승인"),
        ("render", cmd_render, "승인된 세션의 블러본을 라벨링 버킷에 렌더"),
    ):
        p = psub.add_parser(name, help=help_text)
        p.add_argument("session_id")
        p.add_argument("--url", help="DB URL (기본: DLP_DATABASE_URL)")
        if name != "approve":
            p.add_argument("--store", default="s3", help="'s3' 또는 'local:<디렉터리>'")
        p.set_defaults(func=func)
