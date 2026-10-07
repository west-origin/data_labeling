"""합성 픽스처 하위 명령."""

from __future__ import annotations

import argparse
from pathlib import Path

from dlp_fixtures import generate_all


def cmd_fixtures_generate(args: argparse.Namespace) -> int:
    for name, path in generate_all(Path(args.out), args.seed, args.sessions).items():
        print(f"{name:<9} {path}")
    return 0


def add_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # pyright: ignore[reportPrivateUsage]
    fixtures = sub.add_parser("fixtures", help="정답을 아는 합성 데이터")
    fixtures_sub = fixtures.add_subparsers(dest="fixtures_command", required=True)
    gen = fixtures_sub.add_parser("generate", help="모든 합성 픽스처 생성")
    gen.add_argument("--out", required=True, help="출력 디렉터리")
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--sessions", type=int, default=300, help="가짜 세션 수")
    gen.set_defaults(func=cmd_fixtures_generate)
