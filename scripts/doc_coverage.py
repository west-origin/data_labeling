"""패키지별 설명(docstring)·주석 양을 센다 (`make doc-coverage`).

처음 맡는 개발자가 기대는 설명이 어디에 비어 있는지 숫자로 보이려고 둔다.
문장이 맞는지는 이 스크립트가 판단하지 못한다. 그것은 사람이나 검수 에이전트가
코드와 대조해서 본다.

세는 것 (packages/*/src 아래 .py만, tests는 뺀다):
- 모듈 docstring이 있는 파일 수
- 공개 함수(이름이 `_`로 시작하지 않음)·내부 함수·클래스 중 docstring이 있는 수
- `#`로 시작하는 주석 줄 수

`--fail-under 100`을 주면 공개 함수·클래스·모듈 중 하나라도 docstring이 없을 때
종료 코드 1로 끝나고, 빠진 위치(파일:줄 이름)를 출력한다.
계약 클래스 일부는 docstring이 JSON Schema 설명으로 들어가서 `#` 주석을 쓰는
관례가 있었다(dlp_schema.labels 머리말). 이 기준에서는 그런 클래스도 빠진 것으로 센다.
"""

from __future__ import annotations

import argparse
import ast
import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _is_public(name: str) -> bool:
    """`_`로 시작하지 않거나 `__init__` 같은 특수 이름이면 공개로 본다."""
    return not name.startswith("_") or (name.startswith("__") and name.endswith("__"))


def scan(root: Path) -> tuple[dict[str, collections.Counter[str]], list[str]]:
    """패키지 이름 → 셈 표, 그리고 docstring이 빠진 공개 대상 목록(`파일:줄 이름`)을 돌려준다."""
    stats: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    missing: list[str] = []
    for path in sorted(root.glob("packages/*/src/**/*.py")):
        package = path.relative_to(root / "packages").parts[0]
        source = path.read_text("utf-8")
        tree = ast.parse(source)
        counter = stats[package]
        rel = path.relative_to(root)
        counter["files"] += 1
        counter["lines"] += source.count("\n")
        counter["comment_lines"] += sum(
            1 for line in source.splitlines() if line.lstrip().startswith("#")
        )
        if ast.get_docstring(tree):
            counter["module_doc"] += 1
        else:
            missing.append(f"{rel}:1 (모듈)")
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            kind = "class" if isinstance(node, ast.ClassDef) else "func"
            scope = "public" if _is_public(node.name) else "private"
            key = f"{kind}_{scope}"
            counter[key] += 1
            if ast.get_docstring(node):
                counter[f"{key}_doc"] += 1
            elif scope == "public":
                missing.append(f"{rel}:{node.lineno} {node.name}")
    return stats, missing


def _ratio(counter: collections.Counter[str], key: str) -> str:
    return f"{counter[key + '_doc']}/{counter[key]}"


def main() -> int:
    """표를 출력하고, `--fail-under`가 있으면 공개 대상 docstring 비율을 검사한다."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--fail-under", type=float, default=None, help="공개 대상 docstring 비율 하한 (%%)"
    )
    parser.add_argument(
        "--list", action="store_true", help="docstring이 빠진 공개 대상을 모두 출력"
    )
    args = parser.parse_args()

    stats, missing = scan(ROOT)
    total: collections.Counter[str] = collections.Counter()
    print("패키지 / 파일 / 줄 / 모듈 / 공개 함수 / 내부 함수 / 클래스 / 주석 줄")
    for package, counter in sorted(stats.items()):
        total.update(counter)
        class_doc = counter["class_public_doc"] + counter["class_private_doc"]
        class_all = counter["class_public"] + counter["class_private"]
        print(
            f"{package:<11}{counter['files']:>5}{counter['lines']:>8}  "
            f"{counter['module_doc']:>3}/{counter['files']:<3}  "
            f"{_ratio(counter, 'func_public'):>9}  {_ratio(counter, 'func_private'):>9}  "
            f"{class_doc:>4}/{class_all:<4}  {counter['comment_lines']:>6}"
        )
    public = total["func_public"] + total["class_public"] + total["files"]
    documented = total["func_public_doc"] + total["class_public_doc"] + total["module_doc"]
    percent = 100.0 * documented / public if public else 100.0
    print(
        f"공개 대상(모듈·공개 함수·공개 클래스) docstring: {documented}/{public} = {percent:.1f}%"
    )
    if args.list or (args.fail_under is not None and percent < args.fail_under):
        for item in missing:
            print(f"  빠짐: {item}")
    if args.fail_under is not None and percent < args.fail_under:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
