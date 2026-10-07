"""동기화 하위 명령 (WP4, ADR 0004·0022·0028).

등록하는 명령:
- `dlp sync run <세션> [--store] [--url]` — 세션의 모든 스트림을 바디캠(기준 스트림) 시계에 맞춘다.
  QR 슬레이트 → 두드림 → 오디오·운동 상호상관 순의 방법 우선순위와 드리프트 보정은
  `config/policies/sync.yaml`을 따른다. 결과(오프셋 ms, `clock_scale`, 방법, 신뢰도)를 DB
  `sessions`의 스트림에 쓰고, 보고서를 원본 버킷 `sessions/<세션>/derived/sync_report.json`에
  덮어쓴다.
- `dlp sync adjust <세션> <스트림> <ms>` — 사람이 정한 미세 조정값을 기록한다 (자동 결과 위에
  더해짐).

순서: `dlp ingest` 다음, `dlp privacy detect` 전에 돈다 (마스터 타임라인이 정해져야 시간 구간 라벨을
다룰 수 있다, ADR 0019).

주의:
- 원본 스트림 파일을 내려받으므로 원본 접근이다. 저장소는 `raw_store(..., "sync.run")` 감사 저장소로
  만든다.
- 다시 돌려도 바뀐 스트림만 DB에 쓴다 (멱등). 수동 조정값(`manual_adjustment_ms`)은 재동기화해도
  유지된다 (`dlp_sync.pipeline` 규약).
"""

from __future__ import annotations

import argparse

import sqlalchemy as sa

from dlp_cli.raw_access import raw_store
from dlp_cli.schema_cmds import database_url
from dlp_schema import repo_root
from dlp_sync.policy import load_policy
from dlp_sync.runner import adjust, run_sync


def cmd_sync_run(args: argparse.Namespace) -> int:
    """`dlp sync run <세션>`: 세션 동기화를 실행하고 스트림별 결과를 표로 출력한다.

    인자:
        args.session_id: 대상 세션 ID (DB `sessions`에 있어야 한다).
        args.store: 원본 저장소 지정 (`s3` 또는 `local:<디렉터리>`).
        args.url: DB URL.

    출력 열: 스트림 ID, 채택된 방법, 오프셋(ms), 드리프트(ppm = (`clock_scale` - 1) * 10^6),
    시도한 방법별 신뢰도 목록.

    부작용: 한 트랜잭션에서 `sessions` 스트림 동기화 필드 갱신, 원본 버킷 읽기(감사 기록)와
    `derived/sync_report.json` 쓰기.
    """
    root = repo_root()
    policy = load_policy(root / "config" / "policies" / "sync.yaml")
    raw = raw_store(args.store, args.url, "sync.run")
    engine = sa.create_engine(database_url(args.url))
    with engine.begin() as conn:
        synced, report = run_sync(conn, args.session_id, raw, policy)
    engine.dispose()
    for r in report.streams:
        s = synced.stream(r.stream_id)
        # 시도 순서대로 "방법 신뢰도"를 나열한다 (어떤 방법이 왜 대체되었는지 보는 용도)
        tried = ", ".join(f"{a.method} {a.confidence:.2f}" for a in r.attempts)
        print(
            f"{r.stream_id:<14} {s.sync_method.value:<13} offset {s.offset_ms:9.2f} ms "
            f"drift {(s.clock_scale - 1) * 1e6:7.2f} ppm  [{tried}]"
        )
    return 0


def cmd_sync_adjust(args: argparse.Namespace) -> int:
    """`dlp sync adjust <세션> <스트림> <ms>`: 사람이 검수 화면에서 정한 오프셋 미세 조정값을
    기록한다.

    인자:
        args.adjustment_ms: 조정값(ms). 명령줄에서는 실수로 받는다 (`dlp_sync.runner.adjust`의
            시그니처가 float). 자동 오프셋에 더해지는 값이다. 자동으로 맞추지 못한(unsynced)
            스트림이면 사람이 오프셋 전체를 정한 것으로 보고 방법을 `manual`로 바꾼다.

    예외: 기준 스트림(바디캠, `reference`)을 조정하려 하면 `ValueError`.
    부작용: 한 트랜잭션에서 해당 스트림의 동기화 필드(`manual_adjustment_ms` 등)를 갱신한다.
    원본 파일은 읽지 않으므로 감사 기록이 없다.
    """
    engine = sa.create_engine(database_url(args.url))
    with engine.begin() as conn:
        adjust(conn, args.session_id, args.stream_id, args.adjustment_ms)
    engine.dispose()
    print(f"{args.session_id}/{args.stream_id}: 수동 조정 {args.adjustment_ms} ms")
    return 0


def add_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # pyright: ignore[reportPrivateUsage]
    """`sync run|adjust` 하위 명령을 등록한다."""
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
