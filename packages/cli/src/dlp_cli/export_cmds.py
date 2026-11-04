"""내보내기 하위 명령 (WP15, ADR 0018·0021·0027·0029).

순서: `dlp dataset build` 뒤 (보통 `dlp review verify`로 검수 완료된 세션). 내보낸 세션 중
`split_assigned`인 것만 생애주기가 `exported`로 전이된다 (검수 완료 전 세션은 그대로, ADR 0029).

규칙 (CLAUDE.md, ADR 0018·0021):
- 기본은 사람이 만들거나 승인·수정·표본 검증한 라벨만 넣는다. 미검수는 `--include-unreviewed` 또는
  `config/defaults.yaml export.include_unreviewed`로만 넣는다.
- 블러 라벨·원본 위치·검수자 ID는 어떤 형식에도 넣지 않는다. 블러본만 쓰고 원본 버킷을 읽거나 쓰지
  않는다 (`run_export`가 버킷 이름을 비교해 거부).
- 작업자·장소·세션·라벨 ID는 내보내기마다 다른 가명(HMAC, 키 = `DLP_EXPORT_ID_SECRET` + 내보내기
  ID)으로 바꾼다. 개발용 비밀값은 `DLP_ENV=dev`에서만 받는다.
- 내보내기 이력(`exports`)은 올리기 전에 따로 커밋한다 (`run_export`가 트랜잭션을 연다).

정책 출처: `config/policies/export.yaml`(`ids`, `splits`, `excluded_kinds`, 형식별 설정),
`dataset.yaml lakefs`, `config/defaults.yaml`(`buckets`, `export.include_unreviewed`), `--robot`이면
`retarget.yaml`과 로봇 설정.

하위 명령 (인자 오류는 argparse가 종료 코드 2로 끝낸다):

`dlp export coco|intervals|lerobot|gr00t <dataset_version> --target NAME [--include-unreviewed]
[--split golden|train|val|holdout ...] [--store SPEC] [--url URL] [--robot ID]`
    데이터셋 버전 스냅샷에서 COCO(블러본 프레임 + 박스·마스크·키포인트), 구간 JSON
    (`dlp_schema.export`), LeRobot v3.0 에피소드, GR00T LeRobot v2.1(NVIDIA Isaac GR00T N1.7
    미세조정 형식, `dlp_export.gr00t`, ADR 0065)을 만들어 데이터셋 버킷에 올린다
    (`dlp_export.runner.run_export`). LeRobot은 격리된 일회용 환경에서 공식 API로 쓰고 읽어 검증한다
    (`scripts/lerobot_write.py`, `scripts/lerobot_check.py`). GR00T는 pyarrow·ffmpeg로 쓰고
    `dlp_export.gr00t_check`로 다시 확인한다. `--target`은 받는 곳 이름(내보내기
    ID와 이력에 들어간다), `--split`(반복) 기본은 정책 `splits`, `--store`(기본 `s3`)는
    라벨링·데이터셋 버킷 지정이다. `--robot`은 lerobot에서만 쓰고 사람 손을 그 로봇 관절 궤적으로
    옮긴다 (`config/robots/<ID>.yaml` 또는 `DLP_ROBOTS_DIR`, ADR 0047). 원본 버킷: 접근하지 않는다.
    종료 코드: 0 성공, 2 가명 비밀값이 운영 환경에서 개발용 값(`check_secret`). `ExportError`(내보낼
    세션 없음, 원본 버킷 사용, lerobot이 아닌 형식에 `--robot` 등)는 예외로 1이다.

`dlp export gr00t-check <디렉터리>`
    GR00T LeRobot 데이터셋 디렉터리를 NVIDIA 공개 요구(배치·열·modality 구간·주석 번호·H.264 영상
    프레임 수)로 확인한다 (`dlp_export.gr00t_check.check_dataset`). 종료 코드: 0 통과, 1 문제 있음.
"""

from __future__ import annotations

import argparse
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import sqlalchemy as sa

from dlp_cli.ontology_access import dataset_ontology
from dlp_cli.schema_cmds import database_url
from dlp_datasets.policy import load_policy as load_dataset_policy
from dlp_datasets.snapshot import LakeFSSnapshotStore
from dlp_export.gr00t_check import check_dataset
from dlp_export.policy import load_policy
from dlp_export.pseudonym import check_secret
from dlp_export.runner import run_export
from dlp_media.storage import store_from_spec
from dlp_schema import load_config, repo_root
from dlp_schema.dataset import Split


def cmd_export(args: argparse.Namespace) -> int:
    """`dlp export <형식> <데이터셋 버전> --target <받는 곳>`: 내보내기를 하나 만든다.

    인자:
        args.format: `coco` / `intervals` / `lerobot` / `gr00t` (하위 명령 이름).
        args.dataset_version: 데이터셋 버전 ID (lakeFS 스냅샷).
        args.target: 받는 곳 이름 (구매자, 내부 학습 등). 내보내기 ID와 이력에 들어간다.
        args.include_unreviewed: 미검수 모델 라벨 포함 (명시적 옵션).
        args.split: 내보낼 분할 목록. 비우면 정책 `splits`(기본 train, val).
        args.store: 라벨링·데이터셋 버킷 저장소 지정.
        args.robot: lerobot에서만 받는 옵션. 사람 손을 옮길 로봇 ID (`config/robots/<ID>.yaml` 또는
            DLP_ROBOTS_DIR, ADR 0047). 주면 observation.state가 로봇 관절이고 도달 가능성 보고가
            meta/dlp_retarget.json·manifest에 들어간다.

    반환: 0 성공, 2 가명 비밀값이 운영 환경에서 개발용 값일 때.
    `ExportError`(내보낼 세션 없음, 원본 버킷 사용 등)는 예외로 끝난다.
    부작용: `exports` 이력 INSERT(별도 커밋), 세션 생애주기 전이, 라벨링 버킷 읽기(블러본),
    데이터셋 버킷 쓰기. 비밀값이 없으면 임의 비밀값으로 가명을 만들어 되짚을 수 없음을 알린다.
    온톨로지는 데이터셋 버전의 DB 등록 사전을 읽는다.
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
        ontology=dataset_ontology(args.url, args.dataset_version),
        include_unreviewed=args.include_unreviewed or config.export.include_unreviewed,
        splits=tuple(Split(s) for s in args.split) or None,
        now=datetime.now(UTC),
        id_secret=secret.encode() if secret else None,
        robot=args.robot,
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
    _print_retarget(r.details.get("retarget"))
    for w in r.details.get("warnings", []):
        print(f"경고: {w}")
    return 0


def _print_retarget(meta: dict[str, Any] | None) -> None:
    """로봇 리타기팅 요약 (에피소드별 도달 비율·최대 오차·결정)."""
    if meta is None:
        return
    sha = meta["model_sha256"][:12]
    print(f"로봇 {meta['robot_id']} (모델 sha256 {sha}, {meta['retarget_version']})")
    for e in meta["episodes"]:
        err = e["position_error_max_m"]
        worst = "-" if err is None else f"{err * 1000:.0f} mm"
        reason = f" ({e['reason']})" if e["reason"] else ""
        print(
            f"  {e['session_id']}: 목표 {e['target_fraction']:.0%}, "
            f"도달 {e['reachable_fraction']:.0%}, 최대 위치 오차 {worst}, {e['decision']}{reason}"
        )


def cmd_gr00t_check(args: argparse.Namespace) -> int:
    """`dlp export gr00t-check <디렉터리>`: GR00T 데이터셋이 NVIDIA 요구를 따르는지 확인한다.

    인자:
        args.directory: 데이터셋 루트 (meta/·data/·videos/가 있는 디렉터리, 받은 내보내기의 gr00t/).

    반환: 0 통과, 1 문제 있음 (문제마다 한 줄 출력). DB·저장소를 쓰지 않는다.
    """
    policy = load_policy(repo_root())
    problems = check_dataset(Path(args.directory), policy.gr00t.video.ffprobe)
    for p in problems:
        print(f"문제: {p}")
    print("GR00T 형식 확인: " + ("통과" if not problems else f"문제 {len(problems)}개"))
    return 0 if not problems else 1


def _format_parser(
    esub: argparse._SubParsersAction[argparse.ArgumentParser],  # pyright: ignore[reportPrivateUsage]
    fmt: str,
    help_text: str,
) -> argparse.ArgumentParser:
    """형식 하나의 하위 명령 (`dlp export <형식> <데이터셋 버전> --target ...`)."""
    p = esub.add_parser(fmt, help=help_text)
    p.add_argument("dataset_version")
    p.add_argument("--target", required=True, help="내보내는 곳 (구매자, 내부 학습 등)")
    p.add_argument(
        "--include-unreviewed", action="store_true", help="미검수 모델 라벨도 넣는다 (명시적 옵션)"
    )
    p.add_argument(
        "--split", action="append", default=[], choices=[s.value for s in Split],
        help="내보낼 분할 (기본: config/policies/export.yaml splits)",
    )  # fmt: skip
    p.add_argument("--store", default="s3", help="'s3' 또는 'local:<디렉터리>'")
    p.add_argument("--url", help="DB URL (기본: DLP_DATABASE_URL)")
    p.set_defaults(func=cmd_export, format=fmt, robot=None)
    return p


def add_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # pyright: ignore[reportPrivateUsage]
    """`export` 명령(형식마다 하위 명령)과 `export gr00t-check`를 등록한다."""
    exp = sub.add_parser("export", help="데이터셋 버전 내보내기 (COCO, 구간 JSON, LeRobot, GR00T)")
    esub = exp.add_subparsers(dest="export_command", required=True)
    _format_parser(esub, "coco", "COCO (블러본 프레임 + 박스·마스크·키포인트)")
    _format_parser(esub, "intervals", "구간 JSON (시간 구간 라벨)")
    lerobot = _format_parser(esub, "lerobot", "LeRobot v3.0 에피소드 (격리 환경 공식 API)")
    lerobot.add_argument(
        "--robot",
        help="lerobot: 사람 손을 이 로봇 관절 궤적으로 옮긴다 (config/robots/<ID>.yaml 또는 "
        "DLP_ROBOTS_DIR, ADR 0047)",
    )
    _format_parser(esub, "gr00t", "NVIDIA Isaac GR00T N1.7 미세조정 형식 (LeRobot v2.1, ADR 0065)")
    check = esub.add_parser(
        "gr00t-check", help="GR00T LeRobot 데이터셋이 NVIDIA 요구를 따르는지 확인"
    )
    check.add_argument("directory", help="데이터셋 루트 (meta/·data/·videos/)")
    check.set_defaults(func=cmd_gr00t_check)
