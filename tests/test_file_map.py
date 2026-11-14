"""파일 지도 생성기(scripts/file_map.py) 검사.

- 역할 문단 뽑기: 파이썬은 모듈 docstring의 첫 문단, 설정·스크립트는 맨 위 주석의 첫 문단이다
  (셔뱅과 Dockerfile `# syntax=` 줄은 건너뛴다).
- 지도에 넣는 파일 종류와 표 모양.
- 저장소의 모든 대상 파일이 맨 위에 역할을 적었다 (새 파일을 역할 없이 더하면 여기서 걸린다).
정답은 손으로 적은 작은 본문과 저장소의 파일 목록이다. 저장소의 지도가 지금 파일들과 같은지는
`make contracts`(`scripts/file_map.py --check`)가 본다. Git이 없으면 저장소 검사는 건너뛴다.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from types import ModuleType

import pytest

from dlp_schema import repo_root

ROOT = repo_root()


def _load() -> ModuleType:
    """scripts/file_map.py를 모듈로 읽는다 (scripts는 패키지가 아니다)."""
    spec = importlib.util.spec_from_file_location("file_map", ROOT / "scripts/file_map.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclass가 모듈을 sys.modules에서 찾으므로 실행 전에 등록한다
    sys.modules["file_map"] = module
    spec.loader.exec_module(module)
    return module


def test_python_role_is_the_first_docstring_paragraph() -> None:
    """여러 줄에 걸친 첫 문단은 한 줄로 잇고, 빈 줄 뒤 문단은 넣지 않는다."""
    fm = _load()
    source = '"""세션 수집: 원본을 올리고\nPTS 인덱스를 만든다.\n\n자세한 설명.\n"""\nX = 1\n'
    assert fm.role_of("packages/media/src/dlp_media/ingest.py", source) == (
        "세션 수집: 원본을 올리고 PTS 인덱스를 만든다."
    )
    assert fm.role_of("a.py", "X = 1\n") == ""


# (경로, 파일 본문, 역할 문단)
HEADERS = [
    (
        "config/policies/a.yaml",
        "# 정책 한 줄\n# 이어지는 줄.\n#\n# 둘째 문단.\nkey: 1\n",
        "정책 한 줄 이어지는 줄.",
    ),
    ("services/x/run.sh", "#!/bin/sh\n# 서비스를 띄운다.\n\necho hi\n", "서비스를 띄운다."),
    (
        "services/m/Dockerfile",
        "# syntax=docker/dockerfile:1\n# 이미지를 만든다.\nFROM x\n",
        "이미지를 만든다.",
    ),
    ("config/a.yaml", "key: 1\n# 본문 안 주석은 역할이 아니다\n", ""),
]


@pytest.mark.parametrize(("path", "source", "role"), HEADERS)
def test_comment_role_is_the_first_header_paragraph(path: str, source: str, role: str) -> None:
    """맨 위 주석 묶음의 첫 문단만 쓴다 (`#`만 있는 줄이 문단 경계다)."""
    assert _load().role_of(path, source) == role


def test_map_covers_code_and_config_but_not_documents_or_generated_files() -> None:
    fm = _load()
    for path in (
        "packages/x/src/m.py",
        "tests/test_a.py",
        "config/a.yaml",
        "Makefile",
        ".env.example",
    ):
        assert fm.wanted(path), path
    for path in ("docs/handover/issues.yaml", "schemas/a.schema.json", "README.md", "uv.lock"):
        assert not fm.wanted(path), path


def test_render_groups_package_files_and_escapes_table_cells() -> None:
    """패키지 절이 먼저 오고, 그 안은 소스 → DB 마이그레이션 → 시험 순이다.

    역할 문단의 세로줄은 표 칸을 깨지 않게 바꿔 쓴다.
    """
    fm = _load()
    text = fm.render(
        [
            fm.Entry("tests/test_a.py", "통합 시험"),
            fm.Entry("packages/schema/tests/test_b.py", "계약 시험"),
            fm.Entry(
                "packages/schema/src/dlp_schema/db/migrations/versions/0001_x.py", "첫 스키마"
            ),
            fm.Entry("packages/schema/src/dlp_schema/labels.py", "라벨 `a|b` 계약"),
        ]
    )
    assert text.index("## packages/schema") < text.index("## tests")
    assert text.index("### 소스") < text.index("### DB 마이그레이션") < text.index("### 시험")
    assert (
        "| [`labels.py`](../../packages/schema/src/dlp_schema/labels.py) | 라벨 `a\\|b` 계약 |"
        in text
    )
    assert "- [packages/schema](#packagesschema)" in text


@pytest.mark.skipif(shutil.which("git") is None, reason="git 없음")
def test_every_mapped_file_states_its_role() -> None:
    """저장소의 대상 파일은 모두 맨 위 첫 문단에 역할을 적었다."""
    entries, missing = _load().collect(ROOT)
    assert not missing, f"역할 설명이 없는 파일: {missing}"
    assert len(entries) > 600  # 훑기가 저장소에 닿았다는 확인 (2026-11 기준 657개)
