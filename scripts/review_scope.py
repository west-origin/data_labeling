"""전수 검수 범위표 (`make review-scope`).

Git이 추적하는 파일마다 영역·종류·위험 등급·줄 수·관련 미해결 문제 수를 CSV로 낸다. 외주 검수 계약의
범위를 파일 목록으로 확정하고, 위험이 높은 경로부터 읽게 하려는 것이다. 표는 매번 새로 만들므로
저장소에 넣지 않는다 (기본 출력: reports/review-scope.csv, Git 제외).

위험 등급 규칙(`HIGH`·`MEDIUM`)은 이 파일에 있다. 개인정보·원본 접근·내보내기·DB·검수 수집·
권한·분할·배포 경로는 high다. 규칙의 경로는 `tests/test_review_scope.py`가 실제로 있는지 검사한다.
관련 문제 수는 docs/handover/issues.yaml에서 status가 open·candidate인 항목의 files가 그 파일이거나
그 파일을 담은 디렉터리인 경우를 센다.

사용: `uv run python scripts/review_scope.py [--out 경로]` (경로가 `-`면 표준 출력).
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]

# high: 잘못되면 개인정보 노출·데이터 손상·잘못된 배포로 이어지는 경로 (앞부분 일치)
HIGH = (
    "packages/privacy/src/",
    "packages/export/src/",
    "packages/cli/src/dlp_cli/raw_access.py",
    "packages/media/src/dlp_media/audit.py",
    "packages/media/src/dlp_media/storage.py",
    "packages/media/src/dlp_media/ingest.py",
    "packages/aireview/src/dlp_aireview/media.py",  # AI 판정 픽셀은 승인된 블러본만 (I1)
    "packages/schema/src/dlp_schema/db/",
    "packages/schema/src/dlp_schema/label_validation.py",
    "packages/schema/src/dlp_schema/label_graph.py",
    "packages/schema/src/dlp_schema/episode.py",
    "packages/schema/src/dlp_schema/labels.py",
    "packages/review/src/dlp_review/webhook.py",
    "packages/review/src/dlp_review/roles.py",
    "packages/review/src/dlp_review/collect.py",
    "packages/review/src/dlp_review/tasks.py",
    "packages/review/src/dlp_review/reconcile.py",
    "packages/datasets/src/dlp_datasets/splitter.py",
    "packages/datasets/src/dlp_datasets/lineage.py",
    "packages/training/src/dlp_train/loop.py",
    "packages/evaluation/src/dlp_eval/gate.py",
    "config/policies/privacy.yaml",
    "config/policies/export.yaml",
    "config/policies/review.yaml",
    "config/models.yaml",
    "config/ontology/",
)
# medium: 그 밖의 소스·설정·실행 환경
MEDIUM = ("packages/", "config/", "services/", "scripts/", "Makefile", ".github/", "pyproject.toml")


@dataclass(frozen=True)
class Row:
    """범위표 한 줄."""

    path: str
    area: str
    kind: str
    risk: str
    lines: int
    open_issues: int


def tracked_files(root: Path = ROOT) -> list[str]:
    """Git이 추적하는 파일 (저장소 루트 기준 경로, 정렬)."""
    out = subprocess.run(
        ["git", "ls-files"], cwd=root, check=True, capture_output=True, text=True
    ).stdout
    return sorted(line for line in out.splitlines() if line)


def area_of(path: str) -> str:
    """영역: packages/<이름>이면 그 이름, 아니면 첫 디렉터리(루트 파일은 root)."""
    parts = path.split("/")
    if parts[0] == "packages" and len(parts) > 1:
        return parts[1]
    return parts[0] if len(parts) > 1 else "root"


def kind_of(path: str) -> str:
    """종류: src / test / config / doc / schema / infra / other."""
    if "/tests/" in path or path.startswith("tests/"):
        return "test"
    if "/src/" in path:
        return "src"
    if path.startswith("config/"):
        return "config"
    if path.startswith("schemas/"):
        return "schema"
    if path.startswith("docs/") or path.endswith(".md"):
        return "doc"
    if path.startswith(("services/", ".github/", "scripts/")) or path in ("Makefile",):
        return "infra"
    return "other"


def risk_of(path: str) -> str:
    """위험 등급: HIGH 규칙 → high, 테스트·문서 → low, MEDIUM 규칙 → medium, 나머지 low."""
    kind = kind_of(path)
    if kind in ("test", "doc"):
        return "low"
    if path.startswith(HIGH):
        return "high"
    if path.startswith(MEDIUM):
        return "medium"
    return "low"


def _open_issue_files(root: Path) -> list[str]:
    """레지스트리에서 open·candidate 항목의 files (항목마다 한 번씩)."""
    data: Any = yaml.safe_load((root / "docs/handover/issues.yaml").read_text("utf-8"))
    out: list[str] = []
    for item in data["issues"]:
        if item["status"] in ("open", "candidate"):
            out += [str(f).rstrip("/") for f in item["files"]]
    return out


def build_rows(root: Path = ROOT) -> list[Row]:
    """추적 파일마다 범위표 줄을 만든다."""
    issue_files = _open_issue_files(root)
    rows: list[Row] = []
    for path in tracked_files(root):
        full = root / path
        try:
            lines = len(full.read_text("utf-8").splitlines())
        except (UnicodeDecodeError, OSError):
            lines = 0  # 바이너리(로고 등)·지워진 파일
        hits = sum(1 for f in issue_files if path == f or path.startswith(f + "/"))
        rows.append(Row(path, area_of(path), kind_of(path), risk_of(path), lines, hits))
    return rows


def main(argv: list[str] | None = None) -> int:
    """범위표를 CSV로 쓴다. 끝에 등급별 파일 수를 표준 오류로 알린다."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out", default="reports/review-scope.csv", help="출력 경로 (- = 표준 출력)"
    )
    args = parser.parse_args(argv)
    rows = build_rows()
    header = ["path", "area", "kind", "risk", "lines", "open_issues"]
    if args.out == "-":
        writer = csv.writer(sys.stdout)
        writer.writerow(header)
        writer.writerows([r.path, r.area, r.kind, r.risk, r.lines, r.open_issues] for r in rows)
    else:
        out = ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows([r.path, r.area, r.kind, r.risk, r.lines, r.open_issues] for r in rows)
    counts = {k: sum(1 for r in rows if r.risk == k) for k in ("high", "medium", "low")}
    print(f"파일 {len(rows)}개: {counts} → {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
