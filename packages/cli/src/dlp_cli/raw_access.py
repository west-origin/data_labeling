"""원본 버킷 저장소는 이 모듈로만 만든다 (모든 접근이 감사 기록으로 남는다, WP16).

정적 검사(tests/test_raw_access_audited.py)가 다른 곳에서 원본 버킷 이름(buckets.raw,
raw_bucket())으로 저장소를 만드는 것을 막는다.
"""

from __future__ import annotations

from pathlib import Path

import sqlalchemy as sa

from dlp_cli.schema_cmds import database_url
from dlp_media.audit import AuditedStore, DbAccessSink, FileAccessSink, current_actor
from dlp_media.storage import S3Store, store_from_spec
from dlp_schema import load_config, repo_root

# DB 없이 돌 때 감사 기록 파일 (로컬 저장소 루트 아래)
OFFLINE_LOG = "raw_access.jsonl"


def raw_bucket() -> str:
    return load_config(repo_root() / "config" / "defaults.yaml").buckets.raw


def raw_store(spec: str, url: str | None, purpose: str) -> AuditedStore:
    """감사되는 원본 저장소. purpose는 명령 이름 (예: privacy.detect)."""
    sink = DbAccessSink(sa.create_engine(database_url(url)))
    inner = S3Store.from_env(raw_bucket()) if spec == "s3" else store_from_spec(spec, raw_bucket())
    return AuditedStore(inner, sink, actor=current_actor(), purpose=purpose)


def raw_store_offline(spec: str, purpose: str) -> AuditedStore:
    """DB 없이 쓰는 감사 원본 저장소 (로컬 저장소 'local:<디렉터리>'만).

    접근 기록은 <디렉터리>/raw_access.jsonl에 덧붙인다. 실제 원본 버킷(S3)은 DB 감사 기록이
    있어야 하므로 여기서 만들지 않는다.
    """
    if not spec.startswith("local:"):
        raise SystemExit(
            "원본 버킷에 올릴 때는 DB가 필요합니다 (접근 감사 기록). --no-db는 local: 저장소에만"
        )
    root = Path(spec.removeprefix("local:"))
    inner = store_from_spec(spec, raw_bucket())
    sink = FileAccessSink(root / OFFLINE_LOG)
    return AuditedStore(inner, sink, actor=current_actor(), purpose=purpose)
