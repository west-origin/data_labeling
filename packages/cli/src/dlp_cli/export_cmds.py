"""내보내기 하위 명령."""

from __future__ import annotations

import argparse
import os
from datetime import UTC, datetime

import sqlalchemy as sa

from dlp_cli.schema_cmds import database_url
from dlp_datasets.policy import load_policy as load_dataset_policy
from dlp_datasets.snapshot import LakeFSSnapshotStore
from dlp_export.policy import load_policy
from dlp_export.pseudonym import check_secret
from dlp_export.runner import run_export
from dlp_media.storage import store_from_spec
from dlp_schema import load_config, load_ontology, repo_root
from dlp_schema.dataset import Split


def cmd_export(args: argparse.Namespace) -> int:
    root = repo_root()
    config = load_config(root / "config/defaults.yaml")
    lp = load_dataset_policy(root).lakefs
    snapshots = LakeFSSnapshotStore.from_env(
        repository=lp.repository, branch=lp.branch, storage_namespace=lp.storage_namespace
    )
    policy = load_policy(root)
    secret = os.environ.get(policy.ids.secret_env)
    try:
        check_secret(secret, policy.ids, os.environ)
    except ValueError as e:
        print(f"오류: {e}")
        return 2
    engine = sa.create_engine(database_url(args.url))
    # 트랜잭션은 run_export가 연다: 이력을 올리기 전에 따로 커밋한다
    r = run_export(
        engine,
        root=root,
        version_id=args.dataset_version,
        fmt=args.format,
        target=args.target,
        snapshots=snapshots,
        labeling=store_from_spec(args.store, config.buckets.labeling),
        datasets=store_from_spec(args.store, config.buckets.datasets),
        raw_bucket=config.buckets.raw,
        policy=policy,
        ontology=load_ontology(root / "config/ontology/v1"),
        include_unreviewed=args.include_unreviewed or config.export.include_unreviewed,
        splits=tuple(Split(s) for s in args.split) or None,
        now=datetime.now(UTC),
        id_secret=secret.encode() if secret else None,
    )
    engine.dispose()
    n = len(r.record.session_ids)
    print(f"{r.record.export_id}: {r.record.uri} (파일 {r.files}개, 세션 {n}개)")
    print(f"검증 정책: {', '.join(s.value for s in r.record.label_states)}")
    if policy.ids.pseudonymize and not secret:
        env = policy.ids.secret_env
        print(f"작업자·장소·세션·라벨 가명: {env}가 없어 임의 비밀값을 썼습니다 (되짚을 수 없음)")
    for k, n in r.label_counts.items():
        print(f"  {k}: {n}")
    return 0


def add_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # pyright: ignore[reportPrivateUsage]
    exp = sub.add_parser("export", help="데이터셋 버전 내보내기 (COCO, 구간 JSON, LeRobot)")
    exp.add_argument("format", choices=["coco", "intervals", "lerobot"])
    exp.add_argument("dataset_version")
    exp.add_argument("--target", required=True, help="내보내는 곳 (구매자, 내부 학습 등)")
    exp.add_argument(
        "--include-unreviewed", action="store_true", help="미검수 모델 라벨도 넣는다 (명시적 옵션)"
    )
    exp.add_argument(
        "--split", action="append", default=[], choices=[s.value for s in Split],
        help="내보낼 분할 (기본: config/policies/export.yaml splits)",
    )  # fmt: skip
    exp.add_argument("--store", default="s3", help="'s3' 또는 'local:<디렉터리>'")
    exp.add_argument("--url", help="DB URL (기본: DLP_DATABASE_URL)")
    exp.set_defaults(func=cmd_export)
