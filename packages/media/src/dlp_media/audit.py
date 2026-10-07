"""원본 버킷 접근 감사 (WP16).

원본(블러 전) 저장소는 늘 AuditedStore로 감싸서 쓴다. 내려받기(read)·서명 URL(presign)·
올리기(write)를 하기 **전에** 감사 이벤트를 남긴다 (시도도 접근으로 본다). 검수 도구에 원본을 올려
사람에게 보여 주면 grant로 남긴다. DB 기록은 이벤트마다 따로 커밋한다:
그 뒤 작업이 실패해 트랜잭션이 되돌아가도 접근 기록은 남아야 한다. DB 없이 로컬 저장소로 돌 때는
FileAccessSink(JSON Lines)에 남긴다.

ADR 0020·0021. 만드는 곳은 `dlp_cli.raw_access`뿐이다:
- `raw_store(spec, url, purpose)`: DB 감사(`DbAccessSink` → 테이블 `raw_access_log`).
- `raw_store_offline(spec, purpose)`: 로컬 저장소 전용, `<디렉터리>/raw_access.jsonl`.
`purpose`는 명령 이름(예: "privacy.detect", "media.ingest")이고 `actor`는 `current_actor()`다.
감사 기록은 추가만 한다 (수정·삭제 없음). `dlp ops audit-report`가 월간 리포트를 만든다.
`tests/test_raw_access_audited.py` 정적 검사가 다른 경로로 원본 저장소를 만드는 것을 막는다.

- `AccessSink` / `MemorySink` / `DbAccessSink` / `FileAccessSink`: 기록 대상.
- `AuditedStore`: `ObjectStore`를 감싸 접근을 기록하는 저장소.
- `grant`: 감사 저장소면 원본을 사람에게 보여 준 사실을 남긴다.
- `current_actor`, `session_of`: 기록 필드 도우미.
"""

from __future__ import annotations

import getpass
import os
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import sqlalchemy as sa

from dlp_media.storage import ObjectStore, StoredObject
from dlp_schema.db.repository import insert_raw_access
from dlp_schema.ops import RawAccessEvent, RawAction


class AccessSink(Protocol):
    """감사 이벤트를 받는 곳."""

    def record(self, event: RawAccessEvent) -> None:
        """이벤트 하나를 영구히 남긴다. 실패하면 예외를 던져 접근 자체를 막아야 한다."""
        ...


@dataclass
class MemorySink:
    """테스트용."""

    events: list[RawAccessEvent] = field(default_factory=list[RawAccessEvent])

    def record(self, event: RawAccessEvent) -> None:
        """메모리 목록에 덧붙인다."""
        self.events.append(event)


class DbAccessSink:
    """감사 기록을 DB에 이벤트마다 따로 커밋한다 (호출자의 트랜잭션과 무관하게 남는다)."""

    def __init__(self, engine: sa.Engine) -> None:
        """engine: 감사 기록 전용으로 쓸 엔진 (호출자 연결과 다른 연결을 연다)."""
        self.engine = engine

    def record(self, event: RawAccessEvent) -> None:
        """새 연결·트랜잭션으로 `raw_access_log`에 한 행을 넣고 바로 커밋한다."""
        with self.engine.begin() as conn:
            insert_raw_access(conn, event)


class FileAccessSink:
    """DB 없이 돌 때(로컬 저장소 개발·시험) 감사 기록을 JSON Lines 파일에 덧붙인다.

    기록은 접근 전에 한 줄씩 쓰고 바로 flush한다 (DbAccessSink와 같은 규약, 추가만).
    """

    def __init__(self, path: Path) -> None:
        """path: JSON Lines 파일 (없으면 만든다)."""
        self.path = path

    def record(self, event: RawAccessEvent) -> None:
        """이벤트를 JSON 한 줄로 덧붙인다 (상위 디렉터리를 만든다)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(event.model_dump_json() + "\n")
            f.flush()


def current_actor() -> str:
    """실행한 사람·서비스 계정 (DLP_ACTOR, 없으면 OS 사용자)."""
    return os.environ.get("DLP_ACTOR") or getpass.getuser()


def session_of(key: str) -> str | None:
    """원본 키 sessions/<세션>/... 에서 세션 ID.

    그 형식이 아니면 None (예: "other/abc"). 감사 리포트가 세션별로 묶을 때 쓴다.
    """
    parts = key.split("/")
    return parts[1] if len(parts) > 2 and parts[0] == "sessions" else None


class AuditedStore:
    """ObjectStore를 감싸 접근을 감사 기록으로 남긴다.

    head(존재·해시 확인)는 내용을 보지 않으므로 남기지 않는다.

    기록 순서: 항상 기록을 먼저 남기고 실제 접근을 한다. 기록이 실패하면 접근하지 않는다.
    접근이 실패해도(없는 키 등) 기록은 남는다.
    """

    def __init__(
        self,
        inner: ObjectStore,
        sink: AccessSink,
        *,
        actor: str,
        purpose: str,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        """
        Args:
            inner: 실제 원본 저장소 (S3Store·LocalStore).
            sink: 기록 대상.
            actor: 기본 행위자 (실행한 사람·서비스 계정).
            purpose: 접근 목적 (명령 이름).
            clock: 기록 시각 함수 (테스트에서 고정한다, 시간대 필수).
        """
        self.inner, self.sink = inner, sink
        self.actor, self.purpose, self.clock = actor, purpose, clock
        self.bucket = inner.bucket

    def _log(self, action: RawAction, key: str, actor: str | None = None) -> None:
        """감사 이벤트 하나를 남긴다 (이벤트 ID는 임의 UUID)."""
        self.sink.record(
            RawAccessEvent(
                event_id=uuid.uuid4().hex,
                at=self.clock(),
                actor=actor or self.actor,
                purpose=self.purpose,
                action=action,
                bucket=self.bucket,
                key=key,
                session_id=session_of(key),
            )
        )

    def uri(self, key: str) -> str:
        """감싼 저장소의 URI (기록하지 않는다: 내용 접근이 아니다)."""
        return self.inner.uri(key)

    def head(self, key: str) -> StoredObject | None:
        """메타데이터 확인 (기록하지 않는다)."""
        return self.inner.head(key)

    def put_file(self, key: str, path: Path, sha256: str) -> None:
        """write 기록 후 올린다."""
        self._log("write", key)
        self.inner.put_file(key, path, sha256)

    def get_file(self, key: str, dest: Path) -> None:
        """read 기록 후 내려받는다."""
        self._log("read", key)
        self.inner.get_file(key, dest)

    def presign(self, key: str, expires_s: int = 7 * 24 * 3600) -> str:
        """presign 기록 후 서명 URL을 만든다.

        Raises:
            TypeError: 감싼 저장소가 서명 URL을 만들 수 없을 때 (LocalStore). 이때는 기록하지
            않는다.
        """
        presign: Callable[[str, int], str] | None = getattr(self.inner, "presign", None)
        if presign is None:
            raise TypeError(f"{type(self.inner).__name__}는 서명 URL을 만들 수 없습니다")
        self._log("presign", key)
        return presign(key, expires_s)

    def grant(self, key: str, to_actor: str) -> None:
        """원본을 사람(검수 도구 사용자)에게 보여 줬다.

        행위자는 보여 준 상대(to_actor)로 남긴다 (예: 블러 검수자).
        """
        self._log("grant", key, actor=to_actor)


def grant(store: ObjectStore, key: str, to_actor: str | None) -> None:
    """감사 저장소면 grant를 남긴다 (담당자가 비어 있으면 'unassigned').

    감사 저장소가 아니면(라벨링 버킷 등) 아무것도 하지 않는다.
    """
    if isinstance(store, AuditedStore):
        store.grant(key, to_actor or "unassigned")
