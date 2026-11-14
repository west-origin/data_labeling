"""파일 지도 (`make file-map`): 파일마다 역할 한 문단을 모아 docs/handover/file-map.md를 만든다.

처음 맡는 개발자가 "이 일은 어느 파일인가"를 검색 없이 찾게 하려는 표다. 역할 문장을 따로 적지 않고
각 파일이 스스로 적은 설명의 첫 문단을 옮긴다. 그래서 지도를 고치려면 그 파일의 첫 문단을 고치고
`make file-map`을 다시 돌린다 (생성 파일을 직접 고치지 않는다).

- 파이썬(.py): 모듈 docstring의 첫 문단
- 설정·compose(.yaml·.yml), 셸 스크립트(.sh), 환경 변수 예시(.example), Dockerfile, Makefile: 맨 위
  주석의 첫 문단 (셔뱅 줄과 Dockerfile `# syntax=` 줄은 건너뛴다)

대상은 Git이 추적하거나 아직 추가하지 않은(무시 규칙에 걸리지 않는) 파일 중 위 종류다. 문서(docs/),
생성물(schemas/), 잠금 파일, 데이터 파일은 넣지 않는다.

`--check`는 저장소의 지도가 지금 파일들과 같은지 본다 (`make contracts`에 포함, 다르면 종료 코드 1).
역할 설명이 없는 파일이 있으면 생성·검사 모두 종료 코드 1로 끝나고 그 파일을 알려 준다. 새 파일을
더할 때 첫 문단에 역할을 적어야 검사를 통과한다.

사용: `uv run python scripts/file_map.py [--check]`. 시험: `tests/test_file_map.py`.
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/handover/file-map.md"

# 지도에 넣는 파일 종류 (확장자 또는 파일 이름)
COMMENT_SUFFIXES = (".yaml", ".yml", ".sh", ".example")
COMMENT_NAMES = ("Dockerfile", "Makefile")
# 넣지 않는 경로 (앞부분 일치): 문서, 생성물, 다른 도구의 설정
SKIP = ("docs/", "schemas/", ".github/", ".claude/")

HEADER = """\
# 파일 지도

> 생성 파일이다. 직접 고치지 않는다. 각 파일 맨 위 설명의 첫 문단(파이썬은 모듈 docstring, 설정과
> 스크립트는 맨 위 주석)을 고친 뒤 `make file-map`으로 다시 만든다. `make contracts`가 이 파일이
> 코드와 같은지 검사한다.

"이 일은 어느 파일인가"를 찾는 표다. 파일마다 그 파일이 스스로 적은 역할을 한 칸에 옮겼다. 자세한
설명(공개 함수, 지키는 규칙, 호출 위치)은 각 파일 맨 위에 이어서 적혀 있다. 패키지 사이의 큰 그림과
의존 표는 [구조 지도](architecture.md), 바꿀 때 함께 볼 파일과 시험은
[변경 및 복구 안내](change-guide.md)에 있다.
"""


@dataclass(frozen=True)
class Entry:
    """지도 한 줄: 저장소 루트 기준 경로와 역할 문단."""

    path: str
    role: str


def tracked_files(root: Path) -> list[str]:
    """Git이 추적하거나 아직 추가하지 않은(무시되지 않는) 파일 중 지금 있는 것.

    저장소 루트 기준 경로를 정렬해 돌려준다.
    """
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=root, capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    return sorted({p for p in out.splitlines() if (root / p).is_file()})


def wanted(path: str) -> bool:
    """지도에 넣는 파일인가 (종류와 제외 경로)."""
    if path.startswith(SKIP):
        return False
    name = path.rsplit("/", 1)[-1]
    return path.endswith((".py", *COMMENT_SUFFIXES)) or name in COMMENT_NAMES


def first_paragraph(text: str) -> str:
    """빈 줄 전까지의 첫 문단을 한 줄로 잇는다 (앞뒤 빈 줄은 건너뛴다)."""
    lines: list[str] = []
    for line in text.strip().splitlines():
        if not line.strip():
            break
        lines.append(line.strip())
    return " ".join(lines)


def comment_header(source: str) -> str:
    """파일 맨 위 `#` 주석 묶음 (셔뱅, Dockerfile `# syntax=` 줄, 그 앞 빈 줄은 건너뛴다).

    주석 기호를 뗀 본문을 돌려준다. `#`만 있는 줄은 빈 줄(문단 경계)이 된다.
    """
    body: list[str] = []
    for line in source.splitlines():
        if not body and line.startswith(("#!", "# syntax=")):
            continue
        if line.startswith("#"):
            body.append(line[1:].removeprefix(" "))
        elif line.strip() or body:
            break
    return "\n".join(body)


def role_of(path: str, source: str) -> str:
    """파일의 역할 문단. 없으면 빈 문자열."""
    if path.endswith(".py"):
        return first_paragraph(ast.get_docstring(ast.parse(source)) or "")
    return first_paragraph(comment_header(source))


def collect(root: Path) -> tuple[list[Entry], list[str]]:
    """(지도 항목, 역할 설명이 없는 파일 경로)."""
    entries: list[Entry] = []
    missing: list[str] = []
    for path in tracked_files(root):
        if not wanted(path):
            continue
        role = role_of(path, (root / path).read_text("utf-8"))
        if role:
            entries.append(Entry(path, role))
        else:
            missing.append(path)
    return entries, missing


def section_of(path: str) -> tuple[str, str]:
    """(절 제목, 절 안 묶음). 패키지는 소스·시험·마이그레이션으로 나눈다."""
    parts = path.split("/")
    if parts[0] == "packages" and len(parts) > 2:
        if parts[2] == "tests":
            return f"packages/{parts[1]}", "시험"
        if "migrations" in parts:
            return f"packages/{parts[1]}", "DB 마이그레이션"
        return f"packages/{parts[1]}", "소스"
    if len(parts) == 1:
        return "저장소 루트", ""
    return parts[0], ""


def shown_path(path: str, section: str) -> str:
    """표에 보이는 경로: 패키지 소스는 모듈 디렉터리 아래부터, 그 밖은 절 디렉터리 아래부터."""
    parts = path.split("/")
    if section.startswith("packages/"):
        skip = 4 if parts[2] == "src" else 3
        return "/".join(parts[skip:]) or parts[-1]
    return "/".join(parts[1:]) if len(parts) > 1 else path


def cell(text: str) -> str:
    """표 칸에 넣을 글 (세로줄은 칸을 나누므로 피한다)."""
    return text.replace("|", "\\|")


def render(entries: list[Entry]) -> str:
    """지도 Markdown 전체."""
    grouped: dict[str, dict[str, list[Entry]]] = {}
    for e in entries:
        section, group = section_of(e.path)
        grouped.setdefault(section, {}).setdefault(group, []).append(e)
    order = sorted(grouped, key=lambda s: (not s.startswith("packages/"), s))
    out = [HEADER, f"파일 {len(entries)}개.", "", "## 목차", ""]
    out += [f"- [{s}](#{anchor(s)})" for s in order]
    for section in order:
        out += ["", f"## {section}"]
        for group in sorted(grouped[section], key=("소스", "DB 마이그레이션", "시험", "").index):
            out += [
                "",
                *([f"### {group}", ""] if group else []),
                "| 파일 | 역할 |",
                "| --- | --- |",
            ]
            for e in grouped[section][group]:
                link = f"[`{shown_path(e.path, section)}`](../../{e.path})"
                out.append(f"| {link} | {cell(e.role)} |")
    return "\n".join(out) + "\n"


def anchor(title: str) -> str:
    """GitHub 제목 링크 규칙: 소문자, 빈칸은 `-`, 글자·숫자·`-`·`_` 밖은 뺀다."""
    kept = [c for c in title.lower().replace(" ", "-") if c.isalnum() or c in "-_"]
    return "".join(kept)


def main(argv: list[str] | None = None) -> int:
    """지도를 만들거나(`--check` 없음) 저장소의 지도와 비교한다. 0 성공, 1 역할 없음·불일치."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="쓰지 않고 지금 파일과 같은지만 본다")
    args = parser.parse_args(argv)
    entries, missing = collect(ROOT)
    if missing:
        print("역할 설명(파일 맨 위 첫 문단)이 없는 파일:", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        return 1
    text = render(entries)
    if args.check:
        if not OUT.is_file() or OUT.read_text("utf-8") != text:
            print(
                f"{OUT.relative_to(ROOT)}가 지금 파일들과 다릅니다. make file-map", file=sys.stderr
            )
            return 1
        return 0
    OUT.write_text(text, "utf-8")
    print(f"{OUT.relative_to(ROOT)}: 파일 {len(entries)}개")
    return 0


if __name__ == "__main__":
    sys.exit(main())
