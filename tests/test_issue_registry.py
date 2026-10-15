"""알려진 문제 레지스트리(docs/handover/issues.yaml)의 일관성 검사.

외주 검수·리팩토링이 새 발견을 기존 문제와 기계적으로 대조할 수 있게,
레지스트리를 단일 출처로 고정한다.

- ID 형식·중복, 심각도·상태·종류 값, 관련 파일과 출처 문서가 실제로 있는지.
- 문서(known-issues·training-loop-review·ontology-structure-review·audit-notes·
  전수 검수 보고)에 나오는
  ID가 모두 레지스트리에 있는지.
- 확정·미수정(open) 결함의 재현 테스트(`repro`)가 실제로 있고, `xfail(strict=True)`이며
  reason이 그 ID로
  시작하는지. 반대로 tests/test_known_defects.py의 모든 xfail이 레지스트리의 open 항목을 가리키는지.
정답은 저장소의 문서와 테스트 파일 자체다. 서비스 없이 돈다.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from dlp_schema import repo_root

ROOT = repo_root()
REGISTRY = ROOT / "docs" / "handover" / "issues.yaml"
DEFECTS = ROOT / "tests" / "test_known_defects.py"
# F-<영역>: 2026-10-11 전수 검수 영역 6개 + AIR(AI 1차 검수 WP17 후속, 출처 ADR 0040)
ID_RE = re.compile(r"^(S\d+|R\d+|T\d+|O\d+|A-\d{2}|F-(SCH|REV|PMS|PAR|ETA|COE|AIR)-\d{2})$")
# 전수 검수 보고(full-review-2026-10-11.md)의 건수 표가 세는 영역
REPORT_AREAS = ("F-SCH-", "F-REV-", "F-PMS-", "F-PAR-", "F-ETA-", "F-COE-")
SEVERITY = {"high", "medium", "low"}
STATUS = {"open", "candidate", "fixed", "superseded"}
KIND = {"code", "comment", "test", "ops", "product"}
# 표(| ID ...) 첫 칸에 ID를 쓰는 문서와 그 ID 모양
TABLE_DOCS = {
    "docs/handover/known-issues.md": r"^\| ((?:S|R)\d+)\b",
    "docs/handover/training-loop-review.md": r"^\| (T\d+)\b",
    "docs/handover/ontology-structure-review.md": r"^\| (O\d+)\b",
}


@pytest.fixture(scope="module")
def issues() -> list[dict[str, Any]]:
    """레지스트리 항목 목록."""
    data: Any = yaml.safe_load(REGISTRY.read_text("utf-8"))
    return list(data["issues"])


def _xfail_reasons() -> dict[str, str]:
    """test_known_defects.py의 테스트 함수 이름 → xfail reason (strict=True·raises 지정 필수)."""
    tree = ast.parse(DEFECTS.read_text("utf-8"))
    out: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        for deco in node.decorator_list:
            if not (isinstance(deco, ast.Call) and ast.unparse(deco.func) == "pytest.mark.xfail"):
                continue
            kw = {k.arg: k.value for k in deco.keywords}
            strict = kw.get("strict")
            reason = kw.get("reason")
            assert isinstance(strict, ast.Constant) and strict.value is True, node.name
            assert isinstance(reason, ast.Constant) and isinstance(reason.value, str), node.name
            # 결함이 내는 예외만 xfail로 인정한다. 없으면 무관한 오류(패치 대상이 사라진
            # AttributeError 등)도 xfail로 묻혀, 결함을 고쳐도 XPASS가 나지 않는다 (F-REV-01).
            assert "raises" in kw, f"{node.name}: xfail에 raises=<결함이 내는 예외>를 지정한다"
            out[node.name] = reason.value
    return out


def test_registry_fields_are_valid(issues: list[dict[str, Any]]) -> None:
    """ID 형식·중복, 값 범위, 파일·출처 문서 존재, 관련 ID(see) 존재."""
    ids = [i["id"] for i in issues]
    assert len(ids) == len(set(ids)), "중복 ID"
    known = set(ids)
    for item in issues:
        iid = item["id"]
        assert ID_RE.match(iid), iid
        assert item["title"], iid
        assert item["severity"] in SEVERITY, iid
        assert item["status"] in STATUS, iid
        assert item["kind"] in KIND, iid
        assert item["files"], iid
        for path in item["files"]:
            assert (ROOT / path).exists(), f"{iid}: 없는 경로 {path}"
        assert (ROOT / item["source"]).is_file(), f"{iid}: 없는 출처 {item['source']}"
        for other in item.get("see", []):
            assert other in known, f"{iid}: 없는 관련 ID {other}"
        if item["status"] == "superseded":
            assert item.get("see"), f"{iid}: superseded는 옮긴 ID(see)가 필요"


def test_document_ids_are_registered(issues: list[dict[str, Any]]) -> None:
    """문서에 나오는 문제 ID는 모두 레지스트리에 있다 (문서와 레지스트리가 따로 놀지 않게)."""
    known = {i["id"] for i in issues}
    for doc, pattern in TABLE_DOCS.items():
        found = set(re.findall(pattern, (ROOT / doc).read_text("utf-8"), re.M))
        assert found, f"{doc}: 표에서 ID를 찾지 못함"
        assert found <= known, f"{doc}: 레지스트리에 없는 ID {sorted(found - known)}"
    audit = set(
        re.findall(r"\[(A-\d{2})\]", (ROOT / "docs/handover/audit-notes.md").read_text("utf-8"))
    )
    assert audit == {i for i in known if i.startswith("A-")}
    report = ROOT / "docs/handover/full-review-2026-10-11.md"
    found = set(re.findall(r"\bF-(?:SCH|REV|PMS|PAR|ETA|COE)-\d{2}\b", report.read_text("utf-8")))
    registered = {i for i in known if i.startswith(REPORT_AREAS)}
    # 보고서의 범위 표기(F-SCH-07~10)는 첫 ID만 잡히므로 "보고서 ⊆ 레지스트리"만 요구한다
    assert found <= registered, f"보고서에만 있는 ID {sorted(found - registered)}"


def test_open_defect_repros_are_strict_xfail(issues: list[dict[str, Any]]) -> None:
    """repro가 가리키는 테스트는 strict xfail이고 reason이 그 ID로 시작한다. 그 반대 방향도 맞다."""
    reasons = _xfail_reasons()
    by_id = {i["id"]: i for i in issues}
    pointed: set[str] = set()
    for item in issues:
        repro = item.get("repro")
        if repro is None:
            continue
        path, _, name = repro.partition("::")
        assert path == "tests/test_known_defects.py", (
            f"{item['id']}: 재현은 test_known_defects.py에 둔다"
        )
        assert name in reasons, f"{item['id']}: {name}이 없거나 strict xfail이 아님"
        assert reasons[name].startswith(f"{item['id']}:"), (
            f"{name}: reason이 {item['id']}로 시작해야 함"
        )
        assert item["status"] == "open", (
            f"{item['id']}: 재현 중인 결함은 open이어야 함 (고쳤으면 xfail 제거)"
        )
        pointed.add(name)
    for name, reason in reasons.items():
        iid = reason.split(":", 1)[0]
        assert iid in by_id, f"{name}: 레지스트리에 없는 ID {iid}"
        assert by_id[iid].get("repro", "").endswith(f"::{name}"), (
            f"{iid}: repro가 {name}을 가리키지 않음"
        )
    assert pointed == set(reasons)


# 심각도별 건수 표가 있는 전수 검수 보고와 그 보고에서 센 ID 접두사
REPORTS = {
    "docs/handover/full-review-2026-10-11.md": REPORT_AREAS,
    "docs/handover/full-review-2026-10-12.md": (*REPORT_AREAS, "F-AIR-"),
    "docs/handover/external-review-2026-10-15.md": (*REPORT_AREAS, "F-AIR-"),
}


@pytest.mark.parametrize("report", sorted(REPORTS))
def test_registry_counts_in_report_match(issues: list[dict[str, Any]], report: str) -> None:
    """전수 검수 보고의 심각도별 건수 표가 레지스트리(그 보고가 출처인 F-*)와 같다.

    2026-10-11 보고는 F-AIR(WP17 후속 과제)를 세지 않는다. 보고에 나온 F-* ID는 모두 레지스트리에
    있다.
    """
    text = (ROOT / report).read_text("utf-8")
    mine = [i for i in issues if i["id"].startswith(REPORTS[report]) and i["source"] == report]
    label = {"high": "상", "medium": "중", "low": "하"}
    for sev, ko in label.items():
        n = sum(1 for i in mine if i["severity"] == sev)
        assert re.search(rf"^\| {ko} \| {n} \|", text, re.M), f"{report}의 {ko} 건수가 {n}이 아님"
    known = {i["id"] for i in issues}
    found = set(re.findall(r"\bF-[A-Z]{3}-\d{2}\b", text))
    assert found <= known, f"보고서에만 있는 ID {sorted(found - known)}"


def test_registry_paths_use_repo_layout() -> None:
    """레지스트리의 경로는 저장소 루트 기준 상대 경로다 (절대 경로·저장소 밖 경로 금지)."""
    text = REGISTRY.read_text("utf-8")
    assert "/home/" not in text and ".cloud-onboarding" not in text
    assert not re.search(r"files: \[/", text)
    assert Path(REGISTRY).suffix == ".yaml"
