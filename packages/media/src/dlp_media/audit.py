"""원본 버킷 접근 감사 (WP16).

원본(블러 전) 저장소는 늘 AuditedStore로 감싸서 쓴다. 내려받기(read)·서명 URL(presign)·
올리기(write)를 하기 **전에** 감사 이벤트를 남긴다 (시도도 접근으로 본다). 검수 도구에 원본을 올려
사람에게 보여 주면 grant로 남긴다. DB 기록은 이벤트마다 따로 커밋한다:
그 뒤 작업이 실패해 트랜잭션이 되돌아가도 접근 기록은 남아야 한다. DB 없이 로컬 저장소로 돌 때는
FileAccessSink(JSON Lines)에 남긴다.
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
    def record(self, event: RawAccessEvent) -> None: ...


@dataclass
class MemorySink:
    """테스트용."""

    events: list[RawAccessEvent] = field(default_factory=list[RawAccessEvent])

    def record(self, event: RawAccessEvent) -> None:
        self.events.append(event)


class DbAccessSink:
    """감사 기록을 DB에 이벤트마다 따로 커밋한다 (호출자의 트랜잭션과 무관하게 남는다)."""

    def __init__(self, engine: sa.Engine) -> None:
        self.engine = engine

    def record(self, event: RawAccessEvent) -> None:
        with self.engine.begin() as conn:
            insert_raw_access(conn, event)


class FileAccessSink:
    """DB 없이 돌 때(로컬 저장소 개발·시험) 감사 기록을 JSON Lines 파일에 덧붙인다.

    기록은 접근 전에 한 줄씩 쓰고 바로 flush한다 (DbAccessSink와 같은 규약, 추가만).
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def record(self, event: RawAccessEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(event.model_dump_json() + "\n")
            f.flush()


def current_actor() -> str:
    """실행한 사람·서비스 계정 (DLP_ACTOR, 없으면 OS 사용자)."""
    return os.environ.get("DLP_ACTOR") or getpass.getuser()


def session_of(key: str) -> str | None:
    """원본 키 sessions/<세션>/... 에서 세션 ID."""
    parts = key.split("/")
    return parts[1] if len(parts) > 2 and parts[0] == "sessions" else None


class AuditedStore:
    """ObjectStore를 감싸 접근을 감사 기록으로 남긴다.

    head(존재·해시 확인)는 내용을 보지 않으므로 남기지 않는다.
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
        self.inner, self.sink = inner, sink
        self.actor, self.purpose, self.clock = actor, purpose, clock
        self.bucket = inner.bucket

    def _log(self, action: RawAction, key: str, actor: str | None = None) -> None:
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
        return self.inner.uri(key)

    def head(self, key: str) -> StoredObject | None:
        return self.inner.head(key)

    def put_file(self, key: str, path: Path, sha256: str) -> None:
        self._log("write", key)
        self.inner.put_file(key, path, sha256)

    def get_file(self, key: str, dest: Path) -> None:
        self._log("read", key)
        self.inner.get_file(key, dest)

    def presign(self, key: str, expires_s: int = 7 * 24 * 3600) -> str:
        presign: Callable[[str, int], str] | None = getattr(self.inner, "presign", None)
        if presign is None:
            raise TypeError(f"{type(self.inner).__name__}는 서명 URL을 만들 수 없습니다")
        self._log("presign", key)
        return presign(key, expires_s)

    def grant(self, key: str, to_actor: str) -> None:
        """원본을 사람(검수 도구 사용자)에게 보여 줬다."""
        self._log("grant", key, actor=to_actor)


def grant(store: ObjectStore, key: str, to_actor: str | None) -> None:
    """감사 저장소면 grant를 남긴다 (담당자가 비어 있으면 'unassigned')."""
    if isinstance(store, AuditedStore):
        store.grant(key, to_actor or "unassigned")
