"""원본 버킷 저장소는 이 함수로만 만든다 (모든 접근이 감사 기록으로 남는다, WP16)."""

from __future__ import annotations

import sqlalchemy as sa

from dlp_cli.schema_cmds import database_url
from dlp_media.audit import AuditedStore, DbAccessSink, current_actor
from dlp_media.storage import S3Store, store_from_spec
from dlp_schema import load_config, repo_root


def raw_bucket() -> str:
    return load_config(repo_root() / "config" / "defaults.yaml").buckets.raw


def raw_store(spec: str, url: str | None, purpose: str) -> AuditedStore:
    """감사되는 원본 저장소. purpose는 명령 이름 (예: privacy.detect)."""
    sink = DbAccessSink(sa.create_engine(database_url(url)))
    inner = S3Store.from_env(raw_bucket()) if spec == "s3" else store_from_spec(spec, raw_bucket())
    return AuditedStore(inner, sink, actor=current_actor(), purpose=purpose)
