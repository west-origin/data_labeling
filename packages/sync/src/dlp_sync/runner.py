"""DB에 등록된 세션을 동기화하고 결과를 저장한다 (`dlp sync run`, `dlp sync adjust`, WP4).

입력: DB `sessions`·`streams`의 세션, 원본 저장소의 스트림 파일.
출력
- DB 스트림 행의 `offset_ms`·`clock_scale`·`sync_method`·`sync_confidence` 갱신
  (`adjust`는 `manual_adjustment_ms`)
- 저장소의 `sessions/<세션>/derived/sync_report.json` (덮어씀, 파생 산출물)

- `run_sync`: 자동 동기화 (멱등: 같은 입력·정책이면 같은 결과이고, 바뀐 스트림만 DB에 쓴다)
- `adjust`: 사람이 정한 미세 조정값 기록

주의
- 원본 저장소는 CLI가 감사 저장소(`dlp_cli.raw_access.raw_store`, 사유 `"sync.run"`)로 만들어
  넘긴다. 여기서 하는 `get_file`/`put_file`은 모두 `raw_access_log`에 남는다 (ADR 0020).
- DB 트랜잭션은 호출자(CLI)가 연다 (`engine.begin()`). 예외가 나면 DB 갱신이 되돌려진다
  (단, 이미 올린 보고서 파일은 남을 수 있다).
"""

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

    Args:
        conn: 열린 트랜잭션의 DB 연결 (호출자가 커밋한다).
        session_id: 동기화할 세션 ID.
        store: 스트림 파일이 있는 저장소 (원본 버킷, 감사 저장소여야 한다).
        policy: `config/policies/sync.yaml`.

    Returns:
        (동기화된 세션, 보고서).

    Raises:
        ValueError: 스트림 URI가 이 저장소에 없을 때, 장갑 압력 채널이 없을 때.
        sqlalchemy.exc.NoResultFound: 세션이 DB에 없을 때 (`get_session`).

    부작용: 바뀐 스트림만 `update_stream_sync`로 DB에 쓴다. 저장소에 보고서를 올린다.
    임시 디렉터리에 스트림 파일을 받았다가 끝나면 지운다.
    """
    session = get_session(conn, session_id)
    # 저장소 URI 접두사 (예: "s3://dlp-raw/"). 스트림 URI에서 이걸 떼면 저장소 키가 된다
    prefix = store.uri("")
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)

        def fetch(uri: str) -> Path:
            """URI → 임시 디렉터리의 로컬 파일. 같은 URI는 한 번만 받는다."""
            if not uri.startswith(prefix):
                raise ValueError(f"{uri}는 저장소 {store.bucket}에 있지 않습니다")
            key = uri.removeprefix(prefix)
            # 키의 "/"를 바꿔 한 디렉터리에 평평하게 둔다
            dest = work / key.replace("/", "__")
            if not dest.exists():
                store.get_file(key, dest)
            return dest

        synced, report = synchronize(session, load_session_media(session, fetch, policy), policy)
        # 바뀐 스트림만 쓴다 (기준·shared_clock·manual·결과가 같은 스트림은 건너뛴다)
        for stream in synced.streams:
            if stream != session.stream(stream.stream_id):
                update_stream_sync(conn, session_id, stream)
        out = work / "sync_report.json"
        out.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        store.put_file(f"sessions/{session_id}/derived/sync_report.json", out, sha256_file(out))
    return synced, report


def adjust(conn: sa.Connection, session_id: str, stream_id: str, adjustment_ms: float) -> Session:
    """사람이 정한 미세 조정값(ms)을 스트림에 기록한다 (`dlp sync adjust`).

    Args:
        conn: 열린 트랜잭션의 DB 연결.
        session_id: 세션 ID.
        stream_id: 조정할 스트림 ID.
        adjustment_ms: 마스터 시각에 더할 값 ms. `manual_adjustment_ms`를 이 값으로 바꾼다
            (기존 값에 누적하지 않는다).

    Returns:
        갱신된 세션.

    Raises:
        ValueError: 기준 스트림을 조정하려 할 때 (`apply_manual_adjustment`).

    부작용: DB 스트림 행 하나를 갱신한다.
    """
    session = apply_manual_adjustment(get_session(conn, session_id), stream_id, adjustment_ms)
    update_stream_sync(conn, session_id, session.stream(stream_id))
    return session
