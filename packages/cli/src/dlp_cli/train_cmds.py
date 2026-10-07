"""재학습 하위 명령."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from typing import Any

import sqlalchemy as sa

from dlp_cli.schema_cmds import database_url
from dlp_datasets.policy import load_policy as load_dataset_policy
from dlp_datasets.snapshot import LakeFSSnapshotStore
from dlp_eval.policy import load_policy as load_eval_policy
from dlp_media.storage import store_from_spec
from dlp_schema import load_config, repo_root
from dlp_schema.db.repository import list_model_versions
from dlp_schema.lineage import ModelStatus
from dlp_train.loop import TrainingJob, deploy, raw_clips, run_training_job
from dlp_train.policy import load_policy
from dlp_train.tracking import MlflowTracker


def _param(item: str) -> tuple[str, Any]:
    key, _, value = item.partition("=")
    try:
        return key, json.loads(value)
    except json.JSONDecodeError:
        return key, value


def cmd_run(args: argparse.Namespace) -> int:
    root = repo_root()
    policy = load_policy(root)
    buckets = load_config(root / "config/defaults.yaml").buckets
    lp = load_dataset_policy(root).lakefs
    snapshots = LakeFSSnapshotStore.from_env(
        repository=lp.repository, branch=lp.branch, storage_namespace=lp.storage_namespace
    )
    job = TrainingJob(
        task=args.task,
        dataset_version_id=args.dataset_version,
        trainer=args.trainer,
        params=dict(_param(p) for p in args.param),
        baseline_version=args.baseline_version,
        force=args.force,
    )
    engine = sa.create_engine(database_url(args.url))
    with engine.begin() as conn:
        r = run_training_job(
            conn,
            job,
            snapshots=snapshots,
            artifacts=store_from_spec(args.store, buckets.mlflow),
            tracker=MlflowTracker.from_env(),
            clips=raw_clips(store_from_spec(args.store, buckets.raw)),
            policy=policy,
            eval_policy=load_eval_policy(root),
            now=datetime.now(UTC),
        )
    engine.dispose()
    print(f"학습 예제: {r.examples}")
    print(f"{args.task}: {r.status} — {r.reason}")
    if r.model_version:
        print(f"모델 버전 {r.model_version} (MLflow 실행 {r.mlflow_run_id})")
    return 0 if r.status != "rejected" else 1


def cmd_models(args: argparse.Namespace) -> int:
    engine = sa.create_engine(database_url(args.url))
    with engine.connect() as conn:
        status = ModelStatus(args.status) if args.status else None
        rows = list_model_versions(conn, args.task, status)
    engine.dispose()
    for m in rows:
        print(
            f"{m.task:9} {m.status.value:9} {m.model_version}  예제 {m.train_examples}  "
            f"{m.created_at:%Y-%m-%d %H:%M}  {m.report_uri or '-'}"
        )
    return 0


def cmd_approve(args: argparse.Namespace) -> int:
    root = repo_root()
    engine = sa.create_engine(database_url(args.url))
    with engine.begin() as conn:
        mv = deploy(
            conn,
            args.model_version,
            tracker=MlflowTracker.from_env(),
            policy=load_policy(root),
            now=datetime.now(UTC),
        )
    engine.dispose()
    print(f"{mv.task}: {mv.model_version} 배포")
    return 0


def add_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # pyright: ignore[reportPrivateUsage]
    train = sub.add_parser("train", help="재학습 루프 (학습 → 골든셋 평가 → 게이트 → 배포)")
    tsub = train.add_subparsers(dest="train_command", required=True)
    run = tsub.add_parser("run", help="데이터셋 버전으로 과제 하나를 재학습")
    run.add_argument("task", choices=["objects", "hands", "body", "contact", "actions", "privacy"])
    run.add_argument("dataset_version")
    run.add_argument("--trainer", help="학습기 (기본: config/policies/training.yaml 템플릿)")
    run.add_argument("--param", action="append", default=[], help="key=value (JSON 값), 여러 번")
    run.add_argument("--baseline-version", help="배포 모델이 없을 때 비교할 DB 예측의 모델 버전")
    run.add_argument("--force", action="store_true", help="누적 조건을 무시")
    run.add_argument("--store", default="s3", help="'s3' 또는 'local:<디렉터리>'")
    run.add_argument("--url", help="DB URL (기본: DLP_DATABASE_URL)")
    run.set_defaults(func=cmd_run)
    models = tsub.add_parser("models", help="재학습 모델 레지스트리")
    models.add_argument("--task")
    models.add_argument("--status", choices=[s.value for s in ModelStatus])
    models.add_argument("--url")
    models.set_defaults(func=cmd_models)
    approve = tsub.add_parser("approve", help="게이트를 통과한(passed) 모델을 사람이 배포 승인")
    approve.add_argument("model_version")
    approve.add_argument("--url")
    approve.set_defaults(func=cmd_approve)
