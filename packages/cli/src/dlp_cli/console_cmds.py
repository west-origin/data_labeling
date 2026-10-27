"""운영 콘솔 하위 명령 (ADR 0061).

등록하는 명령:
- `dlp console serve [--host 127.0.0.1] [--port 8090] [--allow-remote]` — 브라우저 운영 콘솔을
  띄운다. 콘솔은 기존 `dlp <단계>` 명령을 작업으로 돌리고 상태·결과를 보이는 화면이다
  (`dlp_console`). 기본은 이 컴퓨터에서만 연다. 다른 주소는 `--allow-remote`가 있어야 하고 경고를
  찍는다.

반환: Ctrl-C로 끝나면 0, 허용 없이 외부 주소에 묶으려 하면 2. 작업 상태·로그·업로드 사본은
`DLP_CONSOLE_DIR`(기본 ~/.dlp-console)에 남는다.
"""

from __future__ import annotations

import argparse

from dlp_console.server import RemoteBindError, serve
from dlp_schema import repo_root


def cmd_serve(args: argparse.Namespace) -> int:
    """`dlp console serve`."""
    try:
        serve(repo_root(), args.host, args.port, allow_remote=args.allow_remote)
    except RemoteBindError as exc:
        print(exc)
        return 2
    return 0


def add_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # pyright: ignore[reportPrivateUsage]
    """`console serve` 하위 명령을 등록한다."""
    console = sub.add_parser("console", help="브라우저 운영 콘솔")
    csub = console.add_subparsers(dest="console_command", required=True)
    s = csub.add_parser("serve", help="운영 콘솔 띄우기 (기본 127.0.0.1:8090)")
    s.add_argument("--host", help="묶을 주소 (기본: config/console.yaml server.host)")
    s.add_argument("--port", type=int, help="포트 (기본: config/console.yaml server.port)")
    s.add_argument(
        "--allow-remote",
        action="store_true",
        help="127.0.0.1·localhost 밖의 주소에 묶는 것을 허용 (토큰 하나뿐, 암호화 없음)",
    )
    s.set_defaults(func=cmd_serve)
