"""물려받기 계획기는 물려받기 기록을 빠뜨릴 수 없다 (F-PAR-24, ADR 0043 3절, ADR 0048).

3D 궤적 따르기(`dlp_schema.trajectory_follow`)와 그 기록으로 끊긴 궤적을 빼는 내보내기 선택
(`dlp_export.source.select_labels`)도 같은 규칙이다 (빠뜨리면 끊긴 3D를 검수된 것으로 넣는다).

정답 근거: 물려받기 기록(`description_inheritance`)이 비면 검수된 설명은 모두 직접 검수로 보여
보호된다. 호출자가 기록을 빠뜨리면 조용히 다른 결과를 낸다 (`dlp actions describe --pending`이
재생성할 세션을 건너뛴 결함). 그래서 기록을 받는 계획 함수는 그 인자에 기본값을 두지 않고
(pyright가 빠뜨린 호출을 잡는다), 저장소 코드의 모든 호출이 그 인자를 넘기는지도 구문 트리로
대조한다 (`**kwargs`·위치 인자 수 부족이 없어야 한다).
"""

from __future__ import annotations

import ast
import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dlp_actions.describe import plan_describe
from dlp_export.source import select_labels
from dlp_schema.description_follow import (
    active_inheritance,
    inheritance_only,
    inherited_owner,
    plan_description_follow,
)
from dlp_schema.trajectory_follow import (
    active_trajectory_inheritance,
    inheritance_only_trajectory,
    lapsed_trajectories,
    plan_trajectory_follow,
    review_basis,
)

ROOT = Path(__file__).resolve().parents[1]

# 함수 → 물려받기 기록 인자의 위치 (0부터)
PLANNERS: dict[str, tuple[Callable[..., Any], int]] = {
    "plan_description_follow": (plan_description_follow, 1),
    "plan_describe": (plan_describe, 2),
    "inherited_owner": (inherited_owner, 2),
    "active_inheritance": (active_inheritance, 1),
    "inheritance_only": (inheritance_only, 1),
    "plan_trajectory_follow": (plan_trajectory_follow, 1),
    "lapsed_trajectories": (lapsed_trajectories, 1),
    "active_trajectory_inheritance": (active_trajectory_inheritance, 1),
    "inheritance_only_trajectory": (inheritance_only_trajectory, 1),
    "review_basis": (review_basis, 1),
    "select_labels": (select_labels, 3),
}


def test_inheritance_parameter_has_no_default() -> None:
    """기록 인자는 필수다 (기본값 없음)."""
    for name, (fn, index) in PLANNERS.items():
        param = list(inspect.signature(fn).parameters.values())[index]
        assert param.name == "inheritance", name
        assert param.default is inspect.Parameter.empty, name


def _calls(path: Path) -> list[tuple[str, ast.Call]]:
    """파일 안의 계획 함수 호출 (이름 또는 속성 호출)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[tuple[str, ast.Call]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""
        if name in PLANNERS:
            found.append((name, node))
    return found


def test_every_caller_passes_the_inheritance_records() -> None:
    """패키지 소스·테스트의 모든 호출이 기록 인자를 위치 또는 이름으로 넘긴다."""
    files = [*ROOT.glob("packages/*/src/**/*.py"), *ROOT.glob("packages/*/tests/**/*.py")]
    files += [*ROOT.glob("tests/**/*.py"), *ROOT.glob("scripts/**/*.py")]
    missing: list[str] = []
    seen = 0
    for path in files:
        for name, call in _calls(path):
            seen += 1
            index = PLANNERS[name][1]
            starred = any(isinstance(a, ast.Starred) for a in call.args)
            by_name = any(k.arg == "inheritance" for k in call.keywords)
            if not (len(call.args) > index or by_name or starred):
                missing.append(f"{path.relative_to(ROOT)}:{call.lineno} {name}")
    assert seen > 10  # 대조할 호출을 실제로 찾았다
    assert missing == []
