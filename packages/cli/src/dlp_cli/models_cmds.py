"""모델 가중치 내려받기.

정책 파일들(privacy.yaml의 detectors, prelabel.yaml의 models)에 적힌 URL·sha256을 따른다.
가중치는 저장소에 넣지 않는다.
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path
from typing import Any

import yaml

from dlp_schema import repo_root


def model_specs(root: Path) -> dict[str, tuple[str, str, str | None]]:
    """이름 → (상대 경로, URL, sha256)."""
    specs: dict[str, tuple[str, str, str | None]] = {}
    privacy: Any = yaml.safe_load((root / "config/policies/privacy.yaml").read_text("utf-8"))
    for name, spec in privacy.get("detectors", {}).items():
        if spec.get("model_url") and spec.get("model_path"):
            specs[name] = (spec["model_path"], spec["model_url"], spec.get("model_sha256"))
    prelabel: Any = yaml.safe_load((root / "config/policies/prelabel.yaml").read_text("utf-8"))
    for name, spec in prelabel.get("models", {}).items():
        specs[name] = (spec["path"], spec["url"], spec.get("sha256"))
    return specs


def cmd_fetch(args: argparse.Namespace) -> int:
    root = repo_root()
    failed = 0
    for name, (rel, url, sha) in model_specs(root).items():
        path = root / rel
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == sha:
            print(f"[있음] {name}: {path}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=120) as resp:
            data = resp.read()
        digest = hashlib.sha256(data).hexdigest()
        if sha and digest != sha:
            print(f"[실패] {name}: 해시 불일치 {digest}")
            failed += 1
            continue
        path.write_bytes(data)
        print(f"[받음] {name}: {path} ({len(data)} bytes)")
    return 1 if failed else 0


def add_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # pyright: ignore[reportPrivateUsage]
    models = sub.add_parser("models", help="모델 가중치")
    msub = models.add_subparsers(dest="models_command", required=True)
    fetch = msub.add_parser("fetch", help="정책에 적힌 가중치를 내려받고 해시를 확인")
    fetch.set_defaults(func=cmd_fetch)
