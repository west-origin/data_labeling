"""AI 1차 검수 패키지의 경계 정적 검사 (WP17, ADR 0040 I1·I3·I4).

`packages/aireview/src`를 줄 단위 텍스트로 훑는다 (주석·docstring 포함, 기존 원본 접근 검사와 같은
방식). 판정은 라벨도 검수 상태도 아니므로 이 패키지는 라벨·검수·생애주기 기록 함수와 라벨 이력 표
이름을 쓰지 않는다. 원본 버킷·프록시·감사 원본 저장소·CLI·검수 패키지도 쓰지 않는다. 픽셀은 블러본
경계 모듈(`media.py`, WP17.5)에서만 얻는다. 반대로 내보내기·학습·데이터셋·액티브 러닝 패키지는 AI
판정 계약과 저장소를 import하지 않는다 (판정이 학습·판매 데이터에 섞이지 않게).

정답은 저장소 소스 자체다. 서비스 없이 돈다.
"""

from __future__ import annotations

import re
from pathlib import Path

from dlp_schema import repo_root

ROOT = repo_root()
SRC = ROOT / "packages" / "aireview" / "src" / "dlp_aireview"
# 판정 패키지에 나오면 안 되는 이름 (쓰기 함수, 라벨 이력 표, 원본·프록시 경로)
FORBIDDEN = (
    "insert_labels", "record_review", "set_lifecycle", "update_assignment", "label_records",
    "proxy_key", "raw_store", '"dlp-raw"',
)  # fmt: skip
FORBIDDEN_IMPORTS = re.compile(r"^\s*(from|import)\s+(dlp_cli|dlp_review|dlp_media\.audit)\b")
# 블러본 경계 모듈 밖에서 쓰면 안 되는 것 (저장소에서 파일 받기, 픽셀 객체 만들기)
MEDIA_ONLY = ("get_file(", "BlurredFrame(")
# AI 판정을 import하면 안 되는 패키지 (학습·판매 데이터 경로)
CLEAN_PACKAGES = ("export", "training", "datasets", "active")


def _lines(path: Path) -> list[tuple[int, str]]:
    """(줄 번호, 내용)."""
    return list(enumerate(path.read_text("utf-8").splitlines(), start=1))


def test_aireview_does_not_write_labels_or_touch_raw() -> None:
    """판정 패키지 소스에 쓰기 함수·라벨 이력 표·원본·프록시 이름과 금지 import가 없다."""
    offenders: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        for n, line in _lines(path):
            hits = [name for name in FORBIDDEN if name in line]
            if hits or FORBIDDEN_IMPORTS.match(line):
                offenders.append(f"{path.relative_to(ROOT)}:{n}: {line.strip()}")
    assert offenders == [], "\n".join(offenders)


def test_pixels_only_through_blurred_media_module() -> None:
    """저장소 파일 받기와 블러본 프레임 생성은 `media.py`(블러본 경계)에서만 한다."""
    offenders = [
        f"{path.relative_to(ROOT)}:{n}"
        for path in sorted(SRC.rglob("*.py"))
        if path.name != "media.py"
        for n, line in _lines(path)
        if any(name in line for name in MEDIA_ONLY)
    ]
    assert offenders == []


def test_training_and_export_never_import_ai_verdicts() -> None:
    """내보내기·학습·데이터셋·액티브 러닝은 AI 판정 계약·저장소·패키지를 import하지 않는다."""
    pattern = re.compile(r"^\s*(from|import)\s+\S*aireview")
    offenders = [
        f"{path.relative_to(ROOT)}:{n}"
        for pkg in CLEAN_PACKAGES
        for path in sorted((ROOT / "packages" / pkg / "src").rglob("*.py"))
        for n, line in _lines(path)
        if pattern.match(line)
    ]
    assert offenders == []


def test_scan_patterns_catch_examples() -> None:
    """검사 정규식의 회귀 테스트."""
    assert FORBIDDEN_IMPORTS.match("from dlp_review.tasks import x")
    assert FORBIDDEN_IMPORTS.match("    import dlp_cli.main")
    assert FORBIDDEN_IMPORTS.match("from dlp_media.audit import AuditedStore")
    assert not FORBIDDEN_IMPORTS.match("from dlp_media.frames import pick_frames")
