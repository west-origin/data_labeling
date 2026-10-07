"""내보내기 하위 명령 (WP15, ADR 0018·0021·0027·0029).

등록하는 명령:
- `dlp export coco|intervals|lerobot <데이터셋 버전> --target <받는 곳>` — 데이터셋 버전 스냅샷에서
  COCO(블러본 프레임 + 박스·마스크·키포인트), 구간 JSON(`dlp_schema.export`), LeRobot v3.0
  에피소드를 만들어 데이터셋 버킷에 올린다. LeRobot은 격리된 일회용 환경에서 공식 API로 쓰고
  읽어 검증한다 (`scripts/lerobot_write.py`, `scripts/lerobot_check.py`).

순서: `dlp dataset build` 뒤 (보통 `dlp review verify`로 검수 완료된 세션).
내보낸 세션 중 `split_assigned`인 것만 생애주기가 `exported`로 전이된다 (검수 완료 전 세션은
그대로, ADR 0029).

규칙 (CLAUDE.md, ADR 0018·0021):
- 기본은 사람이 만들거나 승인·수정·표본 검증한 라벨만 넣는다. 미검수는 `--include-unreviewed` 또는
  `config/defaults.yaml export.include_unreviewed`로만 넣는다.
- 블러 라벨·원본 위치·검수자 ID는 어떤 형식에도 넣지 않는다. 블러본만 쓰고 원본 버킷을
  읽거나 쓰지 않는다 (`run_export`가 버킷 이름을 비교해 거부).
- 작업자·장소·세션·라벨 ID는 내보내기마다 다른 가명(HMAC, 키 = `DLP_EXPORT_ID_SECRET` +
  내보내기 ID)으로 바꾼다. 개발용 비밀값은 `DLP_ENV=dev`에서만 받는다.
- 내보내기 이력(`exports`)은 올리기 전에 따로 커밋한다 (`run_export`가 트랜잭션을 연다).

정책 출처: `config/policies/export.yaml` (`ids`, `splits`, `excluded_kinds`, 형식별 설정).
"""

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
    """`dlp export <형식> <데이터셋 버전> --target <받는 곳>`: 내보내기를 하나 만든다.

    인자:
        args.format: `coco` / `intervals` / `lerobot`.
        args.dataset_version: 데이터셋 버전 ID (lakeFS 스냅샷).
        args.target: 받는 곳 이름 (구매자, 내부 학습 등). 내보내기 ID와 이력에 들어간다.
        args.include_unreviewed: 미검수 모델 라벨 포함 (명시적 옵션).
        args.split: 내보낼 분할 목록. 비우면 정책 `splits`(기본 train, val).
        args.store: 라벨링·데이터셋 버킷 저장소 지정.

    반환: 0 성공, 2 가명 비밀값이 운영 환경에서 개발용 값일 때.
    `ExportError`(내보낼 세션 없음, 원본 버킷 사용 등)는 예외로 끝난다.
    부작용: `exports` 이력 INSERT(별도 커밋), 세션 생애주기 전이, 라벨링 버킷 읽기(블러본),
    데이터셋 버킷 쓰기. 비밀값이 없으면 임의 비밀값으로 가명을 만들어 되짚을 수 없음을 알린다.
    온톨로지는 `config/ontology/v1`을 고정으로 읽는다 (리팩토링 후보).
    """
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
    """`export` 명령(형식은 위치 인자)을 등록한다."""
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
