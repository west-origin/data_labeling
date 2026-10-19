"""인수인계 문서가 인용하는 시험 이름이 실제로 있는지 (F-COE-35).

`docs/handover/invariants.md`의 불변 조건 표는 근거 시험을 `<경로>::<시험 이름>`으로 든다. 같은
칸에서는 파일 이름만(`test_db.py::<이름>`) 또는 줄임(`::<이름>`)으로도 쓴다. 외주 검수자가 그
노드 ID로 시험을 돌리므로, 시험 이름이 바뀌거나 오타가 나면 문서와 함께 잡는다.

판정: 저장소 경로(packages/·tests/로 시작)를 함께 적은 참조는 그 파일에 `def <이름>(`이 있어야
하고, 파일 이름만·줄임 참조는 저장소 시험 파일 어딘가에 그 이름이 있어야 한다. 정답은 저장소 파일
자체다 (서비스 없이 돈다).
"""

from __future__ import annotations

import re

from dlp_schema import repo_root

ROOT = repo_root()
DOCS = ("docs/handover/invariants.md",)
# (저장소 경로 또는 파일 이름)::시험 이름, 또는 줄임 ::시험 이름
REF = re.compile(r"(?P<path>(?:packages|tests)/[\w/.-]+\.py|[\w-]+\.py)?::(?P<name>test_\w+)")


def _test_sources() -> str:
    """저장소의 모든 시험 파일 내용 (파일 이름만·줄임 참조를 찾는다)."""
    files = [*ROOT.glob("packages/*/tests/**/*.py"), *ROOT.glob("tests/**/*.py")]
    return "\n".join(f.read_text("utf-8") for f in files)


def test_cited_tests_exist() -> None:
    """문서가 인용한 시험 함수가 있다 (판정은 모듈 설명)."""
    everywhere = _test_sources()
    missing: list[str] = []
    for doc in DOCS:
        refs = list(REF.finditer((ROOT / doc).read_text("utf-8")))
        assert refs, f"{doc}: 시험 참조를 찾지 못함"
        for m in refs:
            path, name = m.group("path") or "", m.group("name")
            if "/" in path:
                file = ROOT / path
                ok = file.is_file() and f"def {name}(" in file.read_text("utf-8")
            else:
                ok = f"def {name}(" in everywhere
            if not ok:
                missing.append(f"{doc}: {path}::{name}")
    assert not missing, missing
