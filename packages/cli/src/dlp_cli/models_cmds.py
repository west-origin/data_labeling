"""모델 가중치 내려받기.

정책 파일의 model_url·model_sha256을 따른다. 가중치는 저장소에 넣지 않는다.
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request

from dlp_privacy.policy import load_policy
from dlp_schema import repo_root


def cmd_fetch(args: argparse.Namespace) -> int:
    root = repo_root()
    failed = 0
    for name, spec in load_policy(root).detectors.items():
        if not (spec.model_url and spec.model_path):
            continue
        path = root / spec.model_path
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == spec.model_sha256:
            print(f"[있음] {name}: {path}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(spec.model_url, timeout=60) as resp:
            data = resp.read()
        digest = hashlib.sha256(data).hexdigest()
        if spec.model_sha256 and digest != spec.model_sha256:
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
