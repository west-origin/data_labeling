"""`dlp` 진입점. 파이프라인 단계는 이후 작업 패키지에서 하위 명령으로 추가한다."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from dlp_cli import (
    __version__,
    fixtures_cmds,
    media_cmds,
    models_cmds,
    privacy_cmds,
    review_cmds,
    schema_cmds,
    sync_cmds,
)
from dlp_cli.health import default_checks, run_checks


def _cmd_services_check(args: argparse.Namespace) -> int:
    results = run_checks(default_checks(include_cvat=args.include_cvat), timeout=args.timeout)
    for r in results:
        mark = "OK  " if r.ok else "FAIL"
        print(f"[{mark}] {r.name:<14} {r.detail}")
    return 0 if all(r.ok for r in results) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dlp", description="돌봄 영상 라벨링 플랫폼 CLI")
    parser.add_argument("--version", action="version", version=f"dlp {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    services = sub.add_parser("services", help="개발 서비스 관리")
    services_sub = services.add_subparsers(dest="services_command", required=True)
    check = services_sub.add_parser("check", help="docker compose 서비스 헬스체크")
    check.add_argument("--include-cvat", action="store_true", help="CVAT도 점검")
    check.add_argument("--timeout", type=float, default=3.0, help="서비스별 제한 시간(초)")
    check.set_defaults(func=_cmd_services_check)

    schema_cmds.add_commands(sub)
    fixtures_cmds.add_commands(sub)
    media_cmds.add_commands(sub)
    sync_cmds.add_commands(sub)
    privacy_cmds.add_commands(sub)
    models_cmds.add_commands(sub)
    review_cmds.add_commands(sub)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
