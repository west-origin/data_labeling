from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from dlp_media.audit import AuditedStore, MemorySink, grant, session_of
from dlp_media.storage import LocalStore, sha256_file

T0 = datetime(2026, 10, 7, 9, 0, tzinfo=UTC)


def test_every_raw_access_is_logged_before_it_happens(tmp_path: Path) -> None:
    sink = MemorySink()
    store = AuditedStore(
        LocalStore(tmp_path / "s", "dlp-raw"), sink, actor="worker-1", purpose="privacy.detect",
        clock=lambda: T0,
    )  # fmt: skip
    src = tmp_path / "a.mp4"
    src.write_bytes(b"video")
    store.put_file("sessions/s1/bodycam.mp4", src, sha256_file(src))
    assert store.head("sessions/s1/bodycam.mp4") is not None  # 내용을 보지 않으므로 기록 안 함
    store.get_file("sessions/s1/bodycam.mp4", tmp_path / "b.mp4")
    with pytest.raises(FileNotFoundError):  # 실패한 시도도 남는다
        store.get_file("sessions/s2/missing.mp4", tmp_path / "c.mp4")
    with pytest.raises(TypeError):  # 로컬 저장소는 서명 URL이 없다
        store.presign("sessions/s1/bodycam.mp4")
    grant(store, "sessions/s1/derived/bodycam.proxy.mp4", "privacy-reviewer-1")
    grant(store, "sessions/s1/derived/bodycam.proxy.mp4", None)
    got = [(e.action, e.actor, e.session_id, e.key) for e in sink.events]
    assert got == [
        ("write", "worker-1", "s1", "sessions/s1/bodycam.mp4"),
        ("read", "worker-1", "s1", "sessions/s1/bodycam.mp4"),
        ("read", "worker-1", "s2", "sessions/s2/missing.mp4"),
        ("grant", "privacy-reviewer-1", "s1", "sessions/s1/derived/bodycam.proxy.mp4"),
        ("grant", "unassigned", "s1", "sessions/s1/derived/bodycam.proxy.mp4"),
    ]
    assert {e.purpose for e in sink.events} == {"privacy.detect"}
    assert {e.bucket for e in sink.events} == {"dlp-raw"} and all(e.at == T0 for e in sink.events)
    assert len({e.event_id for e in sink.events}) == len(sink.events)


def test_session_of() -> None:
    assert session_of("sessions/abc/derived/x.json") == "abc"
    assert session_of("other/abc") is None
