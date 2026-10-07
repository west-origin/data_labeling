"""재학습 하위 명령 (WP13, ADR 0016·0025).

등록하는 명령:
- `dlp train run <과제> <데이터셋 버전>` — 재학습 루프 한 바퀴. 데이터셋 버전의 학습·검증 분할에서
  학습 예제를 뽑고(`dlp_train.extract`, 자동 원본과 수정본 차이 포함), 누적 조건(`min_examples`,
  `min_new_examples`)을 만족하면 학습한다. 후보를 골든셋으로 평가해 게이트를 판정한 뒤
  정책의 `deploy` 방식에 따라 자동 배포하거나 `passed`(사람 승인 대기)로 둔다.
- `dlp train models [--task] [--status]` — DB `model_versions` 레지스트리 목록 (읽기 전용).
- `dlp train approve <모델 버전>` — 게이트를 통과한(`passed`) 모델을 사람이 배포 승인한다.

순서: `dlp dataset build` 뒤. 배포된 모델은 다음 `dlp prelabel run` / `dlp privacy detect`부터
쓰인다 (`dlp_train.deployed`).

주의:
- 후보 모델의 골든셋 예측은 DB에 쓰지 않는다. 배포는 게이트를 통과한 모델만, 프라이버시 과제는
  사람 승인 후에만 한다 (`training.yaml`의 `deploy: approve`).
- MLflow 레지스트리 별칭 이동(`tracker.register`)은 DB 트랜잭션 커밋 **뒤**에 한다.
  DB가 롤백되었는데 MLflow만 배포 별칭을 가리키는 일을 막기 위해서다.
- 학습·평가는 원본 영상을 읽으므로 `raw_store(..., "train.run")` 감사 저장소를 쓴다.

정책 출처: `config/policies/training.yaml` (과제별 학습기·파라미터·누적 조건·배포 방식·MLflow),
`evaluation.yaml`(게이트), `dataset.yaml`의 `lakefs` 절.
CI·CPU에서는 `oracle-stub` 학습기를 쓴다.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from typing import Any

import sqlalchemy as sa

from dlp_cli.raw_access import raw_store
from dlp_cli.schema_cmds import database_url
from dlp_datasets.policy import load_policy as load_dataset_policy
from dlp_datasets.snapshot import LakeFSSnapshotStore
from dlp_eval.policy import load_policy as load_eval_policy
from dlp_media.storage import store_from_spec
from dlp_schema import load_config, repo_root
from dlp_schema.db.repository import list_model_versions
from dlp_schema.lineage import ModelStatus
from dlp_train.loop import TrainingJob, deploy, raw_clips, registration, run_training_job
from dlp_train.policy import load_policy
from dlp_train.tracking import MlflowTracker


def _param(item: str) -> tuple[str, Any]:
    """`--param key=value` 한 개를 (키, 값)으로 바꾼다.

    값은 JSON으로 해석하고(`3` → int, `true` → bool, `[1,2]` → list), 실패하면 문자열 그대로
    둔다. `=`가 없으면 값은 빈 문자열이다.
    """
    key, _, value = item.partition("=")
    try:
        return key, json.loads(value)
    except json.JSONDecodeError:
        return key, value


def cmd_run(args: argparse.Namespace) -> int:
    """`dlp train run <과제> <데이터셋 버전>`: 재학습 루프를 한 번 돌린다.

    인자:
        args.task: 과제 (`objects`, `hands`, `body`, `contact`, `actions`, `privacy`).
        args.dataset_version: 학습 예제를 뽑을 데이터셋 버전 ID (lakeFS 스냅샷).
        args.trainer: 학습기 이름. 없으면 `training.yaml`의 과제 템플릿.
        args.param: `key=value` 목록. 정책 파라미터 위에 덮어쓴다.
        args.baseline_version: 배포 모델이 아직 없을 때 비교할 DB 예측의 모델 버전들.
        args.force: 참이면 누적 조건을 무시한다.
        args.store: 원본·MLflow 산출물 저장소 지정.

    결과 상태(`r.status`): `skipped`(누적 부족 등), `rejected`(게이트 실패), `passed`(승인 대기),
    `deployed`(자동 배포).
    반환: `rejected`면 1, 그 밖에는 0.
    부작용: 한 트랜잭션에서 `training_runs`·`model_versions` 쓰기, MLflow 실행·산출물 기록(외부),
    MLflow 산출물 버킷 쓰기, 원본 읽기(감사 기록), 커밋 뒤 MLflow 레지스트리 별칭 이동.
    """
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
        baseline_versions=tuple(args.baseline_version),
        force=args.force,
    )
    tracker = MlflowTracker.from_env()
    engine = sa.create_engine(database_url(args.url))
    with engine.begin() as conn:
        r = run_training_job(
            conn,
            job,
            snapshots=snapshots,
            artifacts=store_from_spec(args.store, buckets.mlflow),
            tracker=tracker,
            clips=raw_clips(raw_store(args.store, args.url, "train.run")),
            policy=policy,
            eval_policy=load_eval_policy(root),
            now=datetime.now(UTC),
        )
    engine.dispose()
    if r.registration is not None:  # DB 커밋 뒤에 MLflow 레지스트리 별칭을 옮긴다
        tracker.register(*r.registration)
    print(f"학습 예제: {r.examples}")
    print(f"{args.task}: {r.status} — {r.reason}")
    if r.model_version:
        print(f"모델 버전 {r.model_version} (MLflow 실행 {r.mlflow_run_id})")
    return 0 if r.status != "rejected" else 1


def cmd_models(args: argparse.Namespace) -> int:
    """`dlp train models`: 재학습 모델 레지스트리를 출력한다 (과제·상태 필터). 읽기 전용."""
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
    """`dlp train approve <모델 버전>`: 게이트를 통과한 모델을 사람이 배포 승인한다.

    `deploy`는 평가 리포트(MLflow 산출물 저장소의 `report_uri`)에 적힌 비교 대상 배포 모델이
    지금 배포 모델과 같을 때만 배포한다. 그 사이 다른 모델이 배포되었으면 `TrainingError`이며
    다시 평가해야 한다. 과제마다 배포 모델은 하나다(이전 배포는 내려간다).
    부작용: 한 트랜잭션에서 `model_versions` 상태 갱신, 커밋 뒤 MLflow 레지스트리 별칭 이동.
    """
    root = repo_root()
    buckets = load_config(root / "config/defaults.yaml").buckets
    # 평가 리포트에 적힌 비교 대상 배포 모델이 지금 배포 모델과 같아야 배포한다
    artifacts = store_from_spec(args.store, buckets.mlflow)
    engine = sa.create_engine(database_url(args.url))
    with engine.begin() as conn:
        mv = deploy(conn, args.model_version, now=datetime.now(UTC), artifacts=artifacts)
        reg = registration(conn, mv, load_policy(root))
    engine.dispose()
    if reg is not None:  # DB 커밋 뒤에
        MlflowTracker.from_env().register(*reg)
    print(f"{mv.task}: {mv.model_version} 배포")
    return 0


def add_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # pyright: ignore[reportPrivateUsage]
    """`train run|models|approve` 하위 명령을 등록한다. 과제 선택지는 하드코딩되어 있다
    (`training.yaml` tasks와 맞춰야 함, 리팩토링 후보).
    """
    train = sub.add_parser("train", help="재학습 루프 (학습 → 골든셋 평가 → 게이트 → 배포)")
    tsub = train.add_subparsers(dest="train_command", required=True)
    run = tsub.add_parser("run", help="데이터셋 버전으로 과제 하나를 재학습")
    run.add_argument("task", choices=["objects", "hands", "body", "contact", "actions", "privacy"])
    run.add_argument("dataset_version")
    run.add_argument("--trainer", help="학습기 (기본: config/policies/training.yaml 템플릿)")
    run.add_argument("--param", action="append", default=[], help="key=value (JSON 값), 여러 번")
    run.add_argument(
        "--baseline-version",
        action="append",
        default=[],
        help=(
            "배포 모델이 없을 때 비교할 DB 예측의 모델 버전 (대신할 기본 어댑터마다, 여러 번). "
            "골든셋에 예측이 없는 버전이면 실패한다"
        ),
    )
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
    approve.add_argument(
        "--store", default="s3", help="평가 리포트 저장소 's3' 또는 'local:<디렉터리>'"
    )
    approve.add_argument("--url")
    approve.set_defaults(func=cmd_approve)
