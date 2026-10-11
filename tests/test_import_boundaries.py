"""패키지 import 경계 검사 (외주 리팩토링의 경계 고정).

docs/handover/architecture.md "실제 의존 관계" 표를 허용 목록으로 읽어, 소스(packages/*/src)의 실제
패키지 간 import가 표와 **같은지** 본다. 리팩토링이 새 의존을 만들면 실패하고, 의존을 줄였으면 표도
함께 고치도록 실패한다 (문서와 코드가 따로 놀지 않게). 지연 import(함수 안 import)도 센다.

- 표의 의존은 순환이 없어야 한다.
- 다른 모듈의 비공개 이름(`_` 시작) import는 허용 목록에 있는 것만 둔다 (A-19).
정답은 저장소의 소스와 architecture.md다. 서비스 없이 돈다.
"""

from __future__ import annotations

import ast
import re
from collections import defaultdict
from pathlib import Path

from dlp_schema import repo_root

ROOT = repo_root()
ARCHITECTURE = ROOT / "docs" / "handover" / "architecture.md"
# 다른 모듈의 비공개 이름 import 허용 목록: (가져오는 모듈, 원래 모듈, 이름).
# 새로 늘리지 않는다. 정리하면 여기서 지운다 (docs/handover/issues.yaml A-19).
PRIVATE_IMPORTS_ALLOWED = {("dlp_train.deployed", "dlp_train.loop", "_download")}


def _packages() -> dict[str, Path]:
    """모듈 이름(dlp_*) → src 아래 디렉터리. dlp_media처럼 __init__.py가 없는 패키지도 포함."""
    out: dict[str, Path] = {}
    for src in sorted(ROOT.glob("packages/*/src/dlp_*")):
        if src.is_dir() and any(src.rglob("*.py")):
            out[src.name] = src
    return out


def _module_name(path: Path, src: Path) -> str:
    """파일 경로 → 점 이름 (패키지 __init__은 패키지 이름)."""
    parts = path.relative_to(src.parent).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _scan() -> tuple[dict[str, set[str]], set[tuple[str, str, str]]]:
    """(패키지 → 다른 내부 패키지 의존, 다른 모듈의 비공개 이름 import 목록)."""
    pkgs = _packages()
    deps: dict[str, set[str]] = defaultdict(set)
    private: set[tuple[str, str, str]] = set()
    for name, src in pkgs.items():
        for path in src.rglob("*.py"):
            module = _module_name(path, src)
            for node in ast.walk(ast.parse(path.read_text("utf-8"))):
                targets: list[tuple[str, list[str]]] = []
                if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    targets.append((node.module, [a.name for a in node.names]))
                elif isinstance(node, ast.Import):
                    targets += [(a.name, []) for a in node.names]
                for target, names in targets:
                    top = target.split(".")[0]
                    if top not in pkgs:
                        continue
                    if top != name:
                        deps[name].add(top)
                    if target != module:
                        private |= {
                            (module, target, n)
                            for n in names
                            if n.startswith("_") and not n.startswith("__")
                        }
    return {k: deps.get(k, set()) for k in pkgs}, private


def _documented() -> dict[str, set[str]]:
    """architecture.md 의존 표 → 패키지(dlp_*) → 의존 패키지. "나머지 N개 패키지"는 다른 전부."""
    text = ARCHITECTURE.read_text("utf-8")
    section = text.split("## 실제 의존 관계", 1)[1].split("\n## ", 1)[0]
    names = set(_packages())
    out: dict[str, set[str]] = {}
    for left, right in re.findall(r"^\| ([^|]+?) \| ([^|]+?) \|$", section, re.M):
        if left in ("패키지", "---"):
            continue
        right_set: set[str]
        if right == "없음":
            right_set = set()
        elif right.startswith("나머지"):
            right_set = {"*"}
        else:
            right_set = {f"dlp_{x.strip()}" for x in right.split(",")}
        for pkg in left.split(","):
            out[f"dlp_{pkg.strip()}"] = right_set
    for pkg, right in out.items():
        if right == {"*"}:
            out[pkg] = names - {pkg}
    return out


def test_package_dependencies_match_architecture_table() -> None:
    """실제 패키지 간 import = architecture.md 표. 새 의존·없어진 의존 모두 표 갱신이 필요하다."""
    actual, _ = _scan()
    documented = _documented()
    assert set(documented) == set(actual), (
        f"표에 없는/없어진 패키지 {set(actual) ^ set(documented)}"
    )
    diff = {
        pkg: {
            "표에 없음": sorted(actual[pkg] - documented[pkg]),
            "코드에 없음": sorted(documented[pkg] - actual[pkg]),
        }
        for pkg in actual
        if actual[pkg] != documented[pkg]
    }
    assert not diff, f"architecture.md 의존 표와 실제 import가 다르다: {diff}"


def test_documented_dependencies_are_acyclic() -> None:
    """의존 표에 순환이 없다 (위상 정렬이 끝난다)."""
    graph = _documented()
    done: set[str] = set()
    while len(done) < len(graph):
        ready = [p for p, d in graph.items() if p not in done and d <= done]
        assert ready, f"순환 의존: {sorted(set(graph) - done)}"
        done |= set(ready)


def test_private_names_not_imported_across_modules() -> None:
    """다른 모듈의 `_` 이름 import는 허용 목록만. 줄었으면 허용 목록도 줄인다."""
    _, private = _scan()
    assert private == PRIVATE_IMPORTS_ALLOWED, (
        f"새 비공개 import {sorted(private - PRIVATE_IMPORTS_ALLOWED)}, "
        f"정리된 항목 {sorted(PRIVATE_IMPORTS_ALLOWED - private)} (허용 목록에서 지운다)"
    )
