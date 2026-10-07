"""원본 버킷 접근 감사 저장소 단위 테스트 (dlp_media.audit, WP16, ADR 0020).

메모리 기록(MemorySink)과 로컬 저장소로, 어떤 접근이 어떤 순서·필드로 기록되는지 본다.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from dlp_media.audit import AuditedStore, MemorySink, grant, session_of
from dlp_media.storage import LocalStore, sha256_file

T0 = datetime(2026, 10, 7, 9, 0, tzinfo=UTC)


def test_every_raw_access_is_logged_before_it_happens(tmp_path: Path) -> None:
    """모든 원본 접근이 접근 전에 기록된다.

    시나리오: 올리기 → head → 내려받기 → 없는 키 내려받기(실패) → 서명 URL(로컬은 불가) → grant 둘.
    정답: write, read, read(실패한 시도도 남는다), grant(검수자), grant(unassigned) 순서로 남고,
    head와 만들지 못한 서명 URL은 남지 않는다. 목적·버킷·시각이 모두 같고 이벤트 ID는 고유하다.
    """
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
    """원본 키 sessions/<세션>/… 에서 세션 ID를 뽑고, 다른 형식이면 None."""
    assert session_of("sessions/abc/derived/x.json") == "abc"
    assert session_of("other/abc") is None
