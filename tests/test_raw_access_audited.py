"""원본 버킷 저장소는 감사 저장소로만 만든다 (WP16).

CLI·라이브러리 코드를 훑어 감사를 거치지 않는 원본 저장소 생성을 찾는다.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {
    Path("packages/cli/src/dlp_cli/raw_access.py"),  # 감사 저장소를 만드는 유일한 곳
}
# 원본 버킷 이름을 저장소를 만들지 않고 비교·전달만 하는 줄
NAME_ONLY = re.compile(r"raw_bucket=|buckets\.labeling == buckets\.raw|원본 버킷을 읽지 않습니다")


def test_cli_builds_raw_stores_only_through_audit() -> None:
    offenders: list[str] = []
    for path in sorted((ROOT / "packages").glob("*/src/**/*.py")):
        rel = path.relative_to(ROOT)
        if rel in ALLOWED:
            continue
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            uses_raw = "buckets.raw" in line or '"dlp-raw"' in line
            if uses_raw and not NAME_ONLY.search(line):
                offenders.append(f"{rel}:{n}: {line.strip()}")
    assert offenders == [], "원본 저장소는 dlp_cli.raw_access.raw_store로 만든다:\n" + "\n".join(
        offenders
    )
