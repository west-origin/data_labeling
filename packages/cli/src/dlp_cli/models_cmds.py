"""모델 가중치 내려받기·변환 (config/models.yaml). 가중치는 저장소에 넣지 않는다."""

from __future__ import annotations

import argparse
import subprocess

from dlp_models.registry import fetch, load_registry
from dlp_schema import repo_root

# 변환 스크립트는 무거운 PyTorch가 필요해 프로젝트 환경과 따로 일회용 환경에서 돌린다
EXPORT_ENV = [
    "uv", "run", "--no-project", "--python", "3.12",
    "--with", "torch==2.9.1", "--with", "transformers==4.57.1", "--with", "onnx",
    "--index", "https://download.pytorch.org/whl/cpu", "--index-strategy", "unsafe-best-match",
    "python",
]  # fmt: skip


def cmd_fetch(args: argparse.Namespace) -> int:
    root = repo_root()
    failed = 0
    for name, spec in load_registry(root).models.items():
        if args.names and name not in args.names:
            continue
        message = fetch(root, name, spec)
        failed += message.startswith("[실패]")
        print(message)
    return 1 if failed else 0


def cmd_export(args: argparse.Namespace) -> int:
    root = repo_root()
    for name, spec in load_registry(root).models.items():
        if spec.export is None or (root / spec.path).is_file():
            continue
        print(f"[변환] {name}: {spec.export}")
        subprocess.run([*EXPORT_ENV, str(root / spec.export)], cwd=root, check=True)
    return 0


def add_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # pyright: ignore[reportPrivateUsage]
    models = sub.add_parser("models", help="모델 가중치")
    msub = models.add_subparsers(dest="models_command", required=True)
    fetch_p = msub.add_parser("fetch", help="config/models.yaml의 가중치를 받고 해시를 확인")
    fetch_p.add_argument("names", nargs="*", help="받을 모델 이름 (기본: 전부)")
    fetch_p.set_defaults(func=cmd_fetch)
    export = msub.add_parser("export", help="공개 ONNX가 없는 모델을 공식 가중치에서 변환")
    export.set_defaults(func=cmd_export)
