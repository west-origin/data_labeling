"""블러본 경계 테스트 (`media.open_blurred`·`frames_for`, WP17.5, ADR 0040 2.4·I1).

판정 프레임은 라벨링 버킷의 **지금 승인된** 블러본에서만 나온다. 원본 버킷 이름의 저장소(감사 원본
저장소 포함), 승인 전 세션, 낡은 렌더 기록, 렌더 기록과 다른 파일, 받는 사이 바뀐 블러를 모두
거부한다.

정답 근거: 렌더 기록을 쓴 합성 블러본(`blurred.db_scene`), 같은 파일을 직접 디코드한 결과
(`dlp_media.frames.decode_frames`)와 PTS 인덱스.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dlp_aireview.media import (
    BlurredMedia,
    MediaBoundaryError,
    frames_for,
    open_blurred,
    render_hashes_now,
)
from dlp_media.audit import AuditedStore, FileAccessSink
from dlp_media.frames import decode_frames, frame_ms
from dlp_media.storage import LocalStore, ObjectStore, StoredObject, blurred_key
from dlp_privacy.errors import RenderNotCurrentError
from dlp_privacy.rendering import RenderMeta, write_render_meta
from dlp_schema.db.repository import insert_labels, set_privacy_state
from dlp_schema.labels import BlurTrackPayload, BoxKeyframe, LabelRecord, Provenance, Source
from dlp_schema.session import PrivacyState
from dlp_schema.testing import FIXED_TIME

from .blurred import RAW, DbScene, db_scene


def _open(
    db: DbScene, work: Path, stream: str = "bodycam", store: ObjectStore | None = None
) -> BlurredMedia:
    """open_blurred (라벨링 저장소 기본)."""
    return open_blurred(
        db.engine, store or db.labeling, db.session_id, stream, db.privacy, work, raw_bucket=RAW
    )


def _blur_label(sid: str) -> LabelRecord:
    """사람이 그린 얼굴 블러 라벨 하나 (렌더 해시를 바꾼다)."""
    return LabelRecord(
        label_id=f"{sid}-blur-1", session_id=sid, stream_id="bodycam", t_start_ms=0,
        t_end_ms=100, ontology_version="1.0.0", provenance=Provenance(source=Source.HUMAN),
        created_at=FIXED_TIME,
        payload=BlurTrackPayload(
            target="face", keyframes=(BoxKeyframe(t_ms=0, x=10, y=10, w=20, h=20),)
        ),
    )  # fmt: skip


def test_open_blurred_returns_current_render_and_frames(tmp_path: Path) -> None:
    """승인된 세션의 블러본: 렌더 해시·파일 sha·PTS 인덱스가 맞고, 프레임은 같은 파일을 직접
    디코드한 원 해상도 픽셀과 같다 (프레임 번호 순, 시각은 PTS 인덱스)."""
    db = db_scene(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    media = _open(db, work)
    now = render_hashes_now(db.engine, db.session_id, ["bodycam"], db.privacy)
    assert media.render_hash == now["bodycam"]
    head = db.labeling.head(blurred_key(db.session_id, "bodycam"))
    assert head is not None and media.blurred_sha256 == head.sha256
    assert media.path.parent == work and len(media.pts) == len(db.scene.frame_times)
    assert media.stream_media.render_hash == media.render_hash
    want = [5, 0, 40, 5]
    frames = list(frames_for(media, want))
    assert [f.index for f in frames] == [0, 5, 40]
    assert [f.t_ms for f in frames] == [frame_ms(media.pts, i) for i in (0, 5, 40)]
    assert all(f.stream_id == "bodycam" for f in frames)
    direct = list(decode_frames(media.path, media.pts, [0, 5, 40], max_side=10_000))
    for f, (_, _, img) in zip(frames, direct, strict=True):
        assert f.rgb.shape == (db.scene.height, db.scene.width, 3)
        assert np.array_equal(f.rgb, img)


def test_raw_bucket_store_is_refused(tmp_path: Path) -> None:
    """원본 버킷 이름의 저장소와 그것을 감싼 감사 원본 저장소는 받기 전에 거부한다."""
    db = db_scene(tmp_path)
    raw = LocalStore(tmp_path / "store", RAW)
    audited = AuditedStore(raw, FileAccessSink(tmp_path / "raw.jsonl"), actor="t", purpose="x.y")
    for store in (raw, audited):
        with pytest.raises(MediaBoundaryError):
            _open(db, tmp_path, store=store)
    assert not (tmp_path / "raw.jsonl").exists()  # 원본 접근 기록도 생기지 않았다 (받지 않음)


def test_not_approved_session_is_refused(tmp_path: Path) -> None:
    """승인 상태가 아니면 (블러본이 남아 있어도) 거부한다."""
    db = db_scene(tmp_path)
    with db.engine.begin() as conn:
        set_privacy_state(conn, db.session_id, PrivacyState.AUTO_BLURRED)
    with pytest.raises(RenderNotCurrentError):
        _open(db, tmp_path)


def test_stale_render_record_is_refused(tmp_path: Path) -> None:
    """렌더 기록의 해시가 지금 블러 라벨의 해시와 다르면 거부한다 (블러 라벨이 바뀐 뒤)."""
    db = db_scene(tmp_path)
    with db.engine.begin() as conn:
        insert_labels(conn, [_blur_label(db.session_id)])
    with pytest.raises(RenderNotCurrentError, match="dlp privacy render"):
        _open(db, tmp_path)


def test_fetched_file_must_match_render_record(tmp_path: Path) -> None:
    """버킷 파일이 렌더 기록 뒤 바뀌었으면 (해시 옆 파일은 그대로) 받은 파일 해시로 거부한다."""
    db = db_scene(tmp_path)
    path = db.root / "dlp-labeling" / blurred_key(db.session_id, "bodycam")
    path.write_bytes(path.read_bytes()[:-10] + b"0" * 10)
    with pytest.raises(RenderNotCurrentError, match="받은 블러본"):
        _open(db, tmp_path)


def test_missing_render_record_is_refused(tmp_path: Path) -> None:
    """렌더 기록이 무효(render_hash 없음)면 거부한다."""
    db = db_scene(tmp_path)
    write_render_meta(db.labeling, db.session_id, "bodycam", RenderMeta(None), tmp_path)
    with pytest.raises(RenderNotCurrentError):
        _open(db, tmp_path)


class _ChangesBlurWhileFetching:
    """받는 동안 블러 라벨이 바뀌는 라벨링 저장소 (렌더 재확인 시험)."""

    def __init__(self, db: DbScene) -> None:
        self.db = db
        self.bucket = db.labeling.bucket

    def uri(self, key: str) -> str:
        return self.db.labeling.uri(key)

    def head(self, key: str) -> StoredObject | None:
        return self.db.labeling.head(key)

    def put_file(self, key: str, path: Path, sha256: str) -> None:
        raise AssertionError("쓰지 않는다")

    def get_file(self, key: str, dest: Path) -> None:
        self.db.labeling.get_file(key, dest)
        if key == blurred_key(self.db.session_id, "bodycam"):  # 렌더 기록 읽기는 그대로
            with self.db.engine.begin() as conn:
                insert_labels(conn, [_blur_label(self.db.session_id)])


def test_blur_change_during_fetch_is_refused(tmp_path: Path) -> None:
    """확인과 받기 사이에 블러가 바뀌면 받은 뒤 렌더 해시 재확인에서 거부한다."""
    db = db_scene(tmp_path)
    with pytest.raises(RenderNotCurrentError, match="받는 사이"):
        _open(db, tmp_path, store=_ChangesBlurWhileFetching(db))


def test_render_hashes_now_requires_approval(tmp_path: Path) -> None:
    """지금 렌더 해시: 승인 상태면 스트림마다, 아니면 RenderNotCurrentError."""
    db = db_scene(tmp_path, publish=False)
    hashes = render_hashes_now(db.engine, db.session_id, ["tp1", "bodycam"], db.privacy)
    assert sorted(hashes) == ["bodycam", "tp1"] and hashes["bodycam"] == hashes["tp1"]
    with db.engine.begin() as conn:
        set_privacy_state(conn, db.session_id, PrivacyState.AUTO_BLURRED)
    with pytest.raises(RenderNotCurrentError):
        render_hashes_now(db.engine, db.session_id, ["bodycam"], db.privacy)
