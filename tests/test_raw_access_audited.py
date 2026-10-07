"""원본 버킷 저장소는 감사 저장소로만 만든다 (WP16).

CLI·라이브러리 코드를 훑어 감사를 거치지 않는 원본 저장소 생성을 찾는다.
원본 버킷 이름을 얻는 길(config buckets.raw, 리터럴 "dlp-raw",
dlp_cli.raw_access.raw_bucket())을 raw_access.py 밖에서 쓰면 실패한다.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {
    Path("packages/cli/src/dlp_cli/raw_access.py"),  # 감사 저장소를 만드는 유일한 곳
}
# 원본 버킷 이름을 저장소를 만들지 않고 비교·전달만 하는 줄
# raw_access.raw_bucket() 호출과 가져오기 (같은 이름의 매개변수·변수는 아니다)
RAW_BUCKET_CALL = re.compile(r"\braw_bucket\(\)|\bimport\b.*\braw_bucket\b")
NAME_ONLY = re.compile(r"raw_bucket=|buckets\.labeling == buckets\.raw|원본 버킷을 읽지 않습니다")


def test_cli_builds_raw_stores_only_through_audit() -> None:
    offenders: list[str] = []
    for path in sorted((ROOT / "packages").glob("*/src/**/*.py")):
        rel = path.relative_to(ROOT)
        if rel in ALLOWED:
            continue
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            uses_raw = "buckets.raw" in line or '"dlp-raw"' in line or RAW_BUCKET_CALL.search(line)
            if uses_raw and not NAME_ONLY.search(line):
                offenders.append(f"{rel}:{n}: {line.strip()}")
    assert offenders == [], "원본 저장소는 dlp_cli.raw_access.raw_store로 만든다:\n" + "\n".join(
        offenders
    )


def test_scan_catches_raw_bucket_helper() -> None:
    """raw_bucket()으로 이름을 얻어 저장소를 만드는 줄도 잡는다 (예전 media_cmds --no-db 경로)."""
    assert RAW_BUCKET_CALL.search("store_from_spec(args.store, raw_bucket())")
    assert RAW_BUCKET_CALL.search("from dlp_cli.raw_access import raw_bucket, raw_store")
    assert not RAW_BUCKET_CALL.search("raw_bucket=config.buckets.raw,")
    assert not RAW_BUCKET_CALL.search("    raw_bucket: str,")
    assert not RAW_BUCKET_CALL.search("if labeling.bucket == raw_bucket:")
