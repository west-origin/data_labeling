"""검수 범위표 생성기(scripts/review_scope.py) 검사.

- 위험 등급 규칙의 경로가 실제로 있다 (경로를 옮기면 규칙도 고쳐야 한다).
- Git이 추적하는 모든 파일이 표에 한 번씩 들어가고, 개인정보·원본 접근 경로는 high다.
정답은 저장소의 파일 목록이다. Git이 없으면 건너뛴다.
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
    """scripts/review_scope.py를 모듈로 읽는다 (scripts는 패키지가 아니다)."""
    spec = importlib.util.spec_from_file_location("review_scope", ROOT / "scripts/review_scope.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclass가 모듈을 sys.modules에서 찾으므로 실행 전에 등록한다
    sys.modules["review_scope"] = module
    spec.loader.exec_module(module)
    return module


def test_risk_rule_paths_exist() -> None:
    """HIGH 규칙의 경로는 모두 실제 파일·디렉터리다."""
    scope = _load()
    missing = [p for p in scope.HIGH if not (ROOT / p.rstrip("/")).exists()]
    assert not missing, f"없는 경로 규칙: {missing}"


@pytest.mark.skipif(shutil.which("git") is None, reason="git 없음")
def test_scope_covers_every_tracked_file() -> None:
    """추적 파일이 빠짐없이 한 번씩 들어가고, 핵심 경계 파일은 high다."""
    scope = _load()
    rows = scope.build_rows(ROOT)
    paths = [r.path for r in rows]
    assert paths == scope.tracked_files(ROOT)
    risk = {r.path: r.risk for r in rows}
    for path in (
        "packages/cli/src/dlp_cli/raw_access.py",
        "packages/privacy/src/dlp_privacy/runner.py",
        "packages/review/src/dlp_review/webhook.py",
        "packages/schema/src/dlp_schema/db/repository.py",
    ):
        assert risk[path] == "high", path
    assert all(r.risk == "low" for r in rows if r.kind == "test")
