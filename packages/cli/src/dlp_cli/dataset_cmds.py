"""데이터셋·계보 하위 명령 (WP7, ADR 0007).

등록하는 명령:
- `dlp dataset golden --domain <도메인> [--create 버전]` — 도메인 골든셋 후보를 제안한다.
  `--create`를 주면 그 이름으로 골든셋을 확정해 DB `golden_sets`에 쓴다 (사람이 확정).
- `dlp dataset build <버전 ID> [--golden] [--domain]` — 데이터셋 버전을 만든다.
  작업자·장소 단위 분할(골든·학습·검증·holdout) → 라벨·세션·매니페스트를 lakeFS에 커밋 →
  DB `dataset_versions`에 기록.
- `dlp dataset withdraw <세션> --reason` — 세션 사용 중지(동의 철회·삭제 요청).
  이후 버전·내보내기에서 자동 제외되고, 이미 들어간 곳을 출력한다.
- `dlp lineage <세션>` — 세션 계보: 골든셋 → 데이터셋 버전 → 학습 실행 → 내보내기 (읽기 전용).

순서: `dataset golden`(도메인마다 처음 한 번, 프라이버시 승인 세션에서 제안한 뒤 사람이 처음부터
라벨링) → 검수 완료(`dlp review verify`) → `dataset build` → `dlp train run` / `dlp eval golden` /
`dlp export …`. 빌드 후보는 프라이버시 승인 세션 전체이고, `split_assigned`로 옮기는 것은 검수
완료 세션뿐이다 (ADR 0031).

정책 출처: `config/policies/dataset.yaml` (`golden_sessions_per_domain`, `eligible_privacy_state`,
`include_label_history`, `val_ratio`, `lakefs` 절).

주의:
- 골든·학습·검증 사이에 작업자나 장소가 겹치면 안 된다. 분할은 `dlp_datasets.splitter`로만 만든다
  (`build_dataset_version`이 겹침을 검사해 `DatasetBuildError`).
- 오류 삽입·측정 레코드는 스냅샷에서 뺀다 (`non_operational_ids`).
- 도메인 선택지와 기본 온톨로지 버전(`1.0.0`)이 명령줄 정의에 하드코딩되어 있다 (리팩토링 후보).
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime

import sqlalchemy as sa

from dlp_cli.schema_cmds import database_url
from dlp_datasets.build import build_dataset_version, propose_golden_set
from dlp_datasets.lineage import session_lineage, withdraw_session
from dlp_datasets.policy import load_policy
from dlp_datasets.snapshot import LakeFSSnapshotStore
from dlp_media.audit import current_actor
from dlp_schema import repo_root
from dlp_schema.db.repository import insert_golden_set
from dlp_schema.lineage import GoldenSet
from dlp_schema.session import Domain


def _engine(args: argparse.Namespace) -> sa.Engine:
    """`--url`(없으면 `DLP_DATABASE_URL`/기본값)로 엔진을 만든다. 호출자가 `dispose`한다."""
    return sa.create_engine(database_url(args.url))


def cmd_golden(args: argparse.Namespace) -> int:
    """`dlp dataset golden`: 도메인 골든셋 후보를 제안하고, `--create`면 확정해 저장한다.

    인자:
        args.domain: `cleaning` / `caregiving` / `nursing` (`Domain` 열거형 값).
        args.count: 목표 세션 수. 없거나 0이면 정책 `golden_sessions_per_domain`.
        args.create: 확정할 골든셋 버전 이름. None이면 제안만 출력한다(DB 쓰기 없음).
        args.note: 확정 시 메모.
        args.ontology_version: 후보 세션을 고를 온톨로지 버전.
        args.seed: 제안 난수 시드 (같은 DB 상태·시드면 같은 제안).

    후보에서 빼는 세션: 프라이버시 미승인(`eligible_privacy_state`가 아님), 사용 중지.
    부작용: `--create`면 한 트랜잭션에서 `golden_sets` INSERT (같은 버전이 있으면 DB 오류).
    """
    policy = load_policy(repo_root())
    engine = _engine(args)
    with engine.begin() as conn:
        # 프라이버시 승인되지 않았거나 사용 중지된 세션은 제안하지 않는다
        proposed = propose_golden_set(
            conn,
            policy,
            ontology_version=args.ontology_version,
            domain=args.domain,
            target=args.count or policy.golden_sessions_per_domain,
            seed=args.seed,
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
    """`dlp dataset build <버전 ID>`: 데이터셋 버전을 만든다.

    흐름 (`build_dataset_version`):
    1. 온톨로지 버전·도메인이 맞고 프라이버시 승인된 세션을 후보로 모은다.
       사용 중지 세션은 `excluded_sessions`로 기록한다.
    2. 골든셋이 있으면 골든셋 전체 세션의 작업자·장소를 골든 쪽에 묶고,
       나머지를 학습·검증·holdout으로 나눈다. 겹치면 `DatasetBuildError`.
    3. `labels.jsonl`(운영 라벨, 정책에 따라 수정 이력 포함), `sessions.jsonl`, `manifest.json`을
       lakeFS `datasets/<버전>`에 커밋한다.
    4. `dataset_versions` INSERT. holdout이 아닌 `human_verified` 세션은 `split_assigned`로
       전이한다.

    인자: `version_id`, `--ontology-version`, `--golden`(골든셋 버전), `--domain`, `--parent`(이전
    버전), `--seed`(분할 시드).
    환경 변수: lakeFS 접속(`LakeFSSnapshotStore.from_env`, `.env.example`의 `DLP_LAKEFS_*`).
    부작용: lakeFS 커밋(외부 서비스), DB 쓰기. lakeFS 커밋 뒤 DB 트랜잭션이 실패하면 커밋은 남는다.
    """
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
    """`dlp dataset withdraw <세션> --reason`: 세션을 사용 중지한다.

    부작용: 한 트랜잭션에서 세션 생애주기를 `withdrawn`으로 바꾸고, 아직 기록이 없으면 `withdrawals`
    INSERT (다시 실행해도 기록은 하나, 멱등). 이미 만든 스냅샷·내보내기는 바꾸지 않으며(불변),
    그 목록을 "이미 들어간 곳"으로 출력해 사람이 후속 조치(재학습·회수)를 판단하게 한다.
    생애주기 기록의 실행자는 `--actor`, 없으면 `DLP_ACTOR`, 그것도 없으면 OS 사용자다.
    """
    actor = args.actor or current_actor()
    engine = _engine(args)
    with engine.begin() as conn:
        lineage = withdraw_session(
            conn, args.session_id, args.reason, datetime.now(UTC), actor=actor
        )
    engine.dispose()
    print(f"{args.session_id}: 사용 중지. 이후 버전·내보내기에서 자동 제외")
    _print_lineage(
        lineage.dataset_versions,
        [r.run_id for r in lineage.training_runs],
        [e.export_id for e in lineage.exports],
        "이미 들어간 곳",
        lineage.golden_sets,
    )
    return 0


def cmd_lineage(args: argparse.Namespace) -> int:
    """`dlp lineage <세션>`: 세션의 생애주기 상태와 계보를 출력한다. 읽기 전용.

    학습 실행은 그 버전에서 세션이 학습·검증 분할에 있었던 것만 보여 준다.
    """
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
        lineage.golden_sets,
    )
    return 0


def _print_lineage(
    versions: list[str], runs: list[str], exports: list[str], title: str, golden: list[str]
) -> None:
    """계보 한 줄 출력 도우미. 빈 목록은 `-`로 표시한다.

    인자: 데이터셋 버전 ID들, 학습 실행 ID들, 내보내기 ID들, 제목, 골든셋 버전들.
    """
    print(
        f"  {title}: 골든셋 {golden or '-'}, 데이터셋 버전 {versions or '-'}, "
        f"학습 실행 {runs or '-'}, 내보내기 {exports or '-'}"
    )


def add_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # pyright: ignore[reportPrivateUsage]
    """`dataset golden|build|withdraw`와 최상위 `lineage`를 등록하고, 넷 모두에 `--url`을 붙인다."""
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
    w.add_argument("--actor", help="사용 중지를 실행한 사람 (기본: DLP_ACTOR 또는 OS 사용자)")
    w.set_defaults(func=cmd_withdraw)

    lin = sub.add_parser("lineage", help="세션 계보: 데이터셋 버전 → 학습 실행 → 내보내기")
    lin.add_argument("session_id")
    lin.set_defaults(func=cmd_lineage)

    for p in (g, b, w, lin):
        p.add_argument("--url", help="DB URL (기본: DLP_DATABASE_URL)")
