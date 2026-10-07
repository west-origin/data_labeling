"""모델 가중치 내려받기·변환 (config/models.yaml). 가중치는 저장소에 넣지 않는다."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from dlp_models.registry import fetch, load_registry
from dlp_prelabel.policy import load_policy as load_prelabel_policy
from dlp_privacy.policy import load_policy as load_privacy_policy
from dlp_schema import repo_root

# 변환 스크립트는 무거운 PyTorch가 필요해 프로젝트 환경과 따로 일회용 환경에서 돌린다
EXPORT_ENV = [
    "uv", "run", "--no-project", "--python", "3.12",
    "--with", "torch==2.9.1", "--with", "transformers==4.57.1", "--with", "onnx",
    "--index", "https://download.pytorch.org/whl/cpu", "--index-strategy", "unsafe-best-match",
    "python",
]  # fmt: skip


def used_models(root: Path) -> dict[str, str]:
    """정책이 쓰는 모델: 쓰는 곳 → config/models.yaml 이름."""
    used: dict[str, str] = {}
    privacy = load_privacy_policy(root)
    names = {n for tp in privacy.targets.values() for n in tp.detectors}
    while names:  # reflection처럼 다른 탐지기를 쓰는 탐지기를 따라간다
        name = names.pop()
        spec = privacy.detectors.get(name)
        if spec is None:
            continue
        for field in ("model", "tokenizer"):
            if (model := getattr(spec, field)) is not None:
                used[f"privacy.detectors.{name}.{field}"] = model
        names |= {n for n in (spec.region_detector, spec.face_detector) if n is not None}
    for role, model in load_prelabel_policy(root).models.model_dump().items():
        used[f"prelabel.models.{role}"] = model
    return used


def cmd_licenses(args: argparse.Namespace) -> int:
    """정책이 쓰는 모델의 라이선스·학습 데이터·상업 사용 분류 (판매 실사 자료). 위반이 있으면 1."""
    root = repo_root()
    registry = load_registry(root)
    used = used_models(root)
    print("| 모델 | 쓰는 곳 | 가중치 라이선스 | 학습 데이터 | 상업 사용 |")
    print("| --- | --- | --- | --- | --- |")
    for name, spec in registry.models.items():
        where = ", ".join(sorted(w for w, n in used.items() if n == name)) or "-"
        data = "; ".join(spec.training_data) or "-"
        print(f"| {name} | {where} | {spec.license} | {data} | {spec.commercial} |")
    problems = registry.violations(used)
    for p in problems:
        print(f"[위반] {p}")
    print(f"허용 분류: {', '.join(registry.accept)}")
    return 1 if problems else 0


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
    lic = msub.add_parser("licenses", help="모델 라이선스 표와 상업 사용 검사 (위반 시 실패)")
    lic.set_defaults(func=cmd_licenses)
    export = msub.add_parser("export", help="공개 ONNX가 없는 모델을 공식 가중치에서 변환")
    export.set_defaults(func=cmd_export)
