"""합성 픽스처 하위 명령 (WP2, `make fixtures`).

- `dlp fixtures generate --out <디렉터리> [--seed] [--sessions]` — 정답을 아는 합성 데이터(가짜
  세션 메타데이터, 오프셋·드리프트를 아는 동기화 신호와 QR 슬레이트 영상, 위치를 아는 블러 대상 VFR
  영상, 경계를 아는 행동 시퀀스)를 한 번에 만든다 (`dlp_fixtures.generate_all`).

같은 시드면 같은 출력이 나온다 (결정적). 생성물은 저장소에 넣지 않는다 (`data/fixtures/`).
DB·저장소에는 접근하지 않는다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dlp_fixtures import generate_all


def cmd_fixtures_generate(args: argparse.Namespace) -> int:
    """`dlp fixtures generate`: 모든 합성 픽스처를 만들고 종류별 출력 경로를 출력한다.

    인자:
        args.out: 출력 디렉터리 (없으면 만든다).
        args.seed: 난수 시드 (기본 0).
        args.sessions: 가짜 세션 수 (기본 300, 분할·골든셋 통계 테스트용).
    반환: 0.
    """
    for name, path in generate_all(Path(args.out), args.seed, args.sessions).items():
        print(f"{name:<9} {path}")
    return 0


def add_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # pyright: ignore[reportPrivateUsage]
    """`fixtures generate` 하위 명령을 등록한다."""
    fixtures = sub.add_parser("fixtures", help="정답을 아는 합성 데이터")
    fixtures_sub = fixtures.add_subparsers(dest="fixtures_command", required=True)
    gen = fixtures_sub.add_parser("generate", help="모든 합성 픽스처 생성")
    gen.add_argument("--out", required=True, help="출력 디렉터리")
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--sessions", type=int, default=300, help="가짜 세션 수")
    gen.set_defaults(func=cmd_fixtures_generate)
