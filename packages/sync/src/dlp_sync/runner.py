"""DB에 등록된 세션을 동기화하고 결과를 저장한다."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import sqlalchemy as sa

from dlp_media.storage import ObjectStore, sha256_file
from dlp_schema.db.repository import get_session, update_stream_sync
from dlp_schema.session import Session
from dlp_sync.loader import load_session_media
from dlp_sync.pipeline import SyncReport, apply_manual_adjustment, synchronize
from dlp_sync.policy import SyncPolicy


def run_sync(
    conn: sa.Connection, session_id: str, store: ObjectStore, policy: SyncPolicy
) -> tuple[Session, SyncReport]:
    """원본 저장소에서 스트림 파일을 내려받아 동기화하고, DB와 동기화 보고서를 갱신한다.

    보고서는 파생 산출물이라 `sessions/<세션>/derived/sync_report.json`을 덮어쓴다.
    """
    session = get_session(conn, session_id)
    prefix = store.uri("")
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)

        def fetch(uri: str) -> Path:
            if not uri.startswith(prefix):
                raise ValueError(f"{uri}는 저장소 {store.bucket}에 있지 않습니다")
            key = uri.removeprefix(prefix)
            dest = work / key.replace("/", "__")
            if not dest.exists():
                store.get_file(key, dest)
            return dest

        synced, report = synchronize(session, load_session_media(session, fetch, policy), policy)
        for stream in synced.streams:
            if stream != session.stream(stream.stream_id):
                update_stream_sync(conn, session_id, stream)
        out = work / "sync_report.json"
        out.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        store.put_file(f"sessions/{session_id}/derived/sync_report.json", out, sha256_file(out))
    return synced, report


def adjust(conn: sa.Connection, session_id: str, stream_id: str, adjustment_ms: float) -> Session:
    session = apply_manual_adjustment(get_session(conn, session_id), stream_id, adjustment_ms)
    update_stream_sync(conn, session_id, session.stream(stream_id))
    return session
