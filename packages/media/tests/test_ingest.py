from __future__ import annotations

import os
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pytest
import sqlalchemy as sa

from dlp_fixtures.sync import SyncScenario
from dlp_media import imu as imu_module
from dlp_media.imu import ImuData
from dlp_media.ingest import SessionConflictError, ingest_session, load_manifest
from dlp_media.probe import MediaInfo
from dlp_media.pts import PtsIndex
from dlp_media.storage import ImmutableObjectError, LocalStore, S3Store
from dlp_schema.db.migrate import upgrade
from dlp_schema.session import StreamKind, SyncMethod
from dlp_schema.testing import FIXED_TIME

ManifestWriter = Callable[..., Path]


def _local(uri: str, store: LocalStore) -> Path:
    return store.root / uri.removeprefix(f"local://{store.bucket}/")


def test_ingest_builds_session_and_is_idempotent(
    sync: tuple[SyncScenario, Path],
    tmp_path: Path,
    write_manifest: ManifestWriter,
) -> None:
    store = LocalStore(tmp_path / "store", "dlp-raw")
    manifest, base = load_manifest(write_manifest(tmp_path, sync[1]))

    first = ingest_session(manifest, base, store)
    s = first.session
    assert s.recorded_at == FIXED_TIME
    assert s.reference_stream.sync_method is SyncMethod.REFERENCE
    assert {st.stream_id for st in s.streams} == {"bodycam", "third_person", "glove_right"}
    assert s.stream("glove_right").sample_rate_hz == pytest.approx(100.0)
    assert s.duration_ms == pytest.approx(12_000, abs=150)
    assert len(first.uploaded) == 8  # 원본 3 + PTS 2 + 프록시 2 + 장갑 정규화 1
    assert first.skipped == []

    body = s.reference_stream
    assert body.pts_index_uri is not None
    index = PtsIndex.read(_local(body.pts_index_uri, store))
    assert len(index) == 120  # 10 fps, 12초
    assert _local(body.uri.replace("raw/bodycam.mp4", "derived/bodycam.proxy.mp4"), store).is_file()

    second = ingest_session(manifest, base, store)
    assert second.uploaded == []
    assert sorted(second.skipped) == sorted(first.uploaded)
    assert second.session == s


def test_changed_raw_file_is_rejected(
    sync: tuple[SyncScenario, Path], tmp_path: Path, write_manifest: ManifestWriter
) -> None:
    store = LocalStore(tmp_path / "store", "dlp-raw")
    glove = tmp_path / "glove.parquet"
    glove.write_bytes((sync[1] / "glove_right.parquet").read_bytes())
    streams = [
        {"stream_id": "bodycam", "kind": "bodycam", "path": str(sync[1] / "bodycam.mp4")},
        {"stream_id": "glove_right", "kind": "glove_right", "path": str(glove)},
    ]
    manifest, base = load_manifest(write_manifest(tmp_path, sync[1], streams=streams))
    ingest_session(manifest, base, store)
    glove.write_bytes(glove.read_bytes() + b"\x00")
    with pytest.raises(ImmutableObjectError):
        ingest_session(manifest, base, store)


def test_recorded_at_required_when_container_has_none(
    sync: tuple[SyncScenario, Path],
    tmp_path: Path,
    write_manifest: ManifestWriter,
) -> None:
    manifest, base = load_manifest(write_manifest(tmp_path, sync[1], recorded_at=None))
    with pytest.raises(ValueError, match="recorded_at"):
        ingest_session(manifest, base, LocalStore(tmp_path / "store", "dlp-raw"))


def test_embedded_imu_becomes_shared_clock_stream(
    sync: tuple[SyncScenario, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    write_manifest: ManifestWriter,
) -> None:
    class FakeExtractor:
        name = "fake"

        def can_handle(self, info: MediaInfo) -> bool:
            return info.video is not None

        def extract(self, path: Path) -> ImuData:
            t = np.arange(0, 1_000, 5.0)
            return ImuData(t, np.zeros((t.size, 3)), np.zeros((t.size, 3)), source="fake")

    monkeypatch.setattr(imu_module, "EXTRACTORS", [FakeExtractor()])
    streams = [{"stream_id": "bodycam", "kind": "bodycam", "path": str(sync[1] / "bodycam.mp4")}]
    manifest, base = load_manifest(write_manifest(tmp_path, sync[1], streams=streams))
    session = ingest_session(manifest, base, LocalStore(tmp_path / "store", "dlp-raw")).session
    imu = session.stream("imu")
    assert imu.kind is StreamKind.IMU
    assert imu.sync_method is SyncMethod.SHARED_CLOCK
    assert imu.sample_rate_hz == pytest.approx(200.0)


# ---------------------------------------------------------------- 실제 서비스


@pytest.fixture
def pg() -> Iterator[sa.Engine]:
    url = sa.make_url(
        os.environ.get(
            "DLP_DATABASE_URL", "postgresql+psycopg://dlp:dlp-dev-password@localhost:5432/dlp"
        )
    )
    name = f"dlp_test_{uuid.uuid4().hex[:8]}"
    admin = sa.create_engine(url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(sa.text(f'CREATE DATABASE "{name}"'))
    test_url = url.set(database=name).render_as_string(hide_password=False)
    upgrade(test_url)
    engine = sa.create_engine(test_url)
    try:
        yield engine
    finally:
        engine.dispose()
        with admin.connect() as conn:
            conn.execute(sa.text(f'DROP DATABASE "{name}" WITH (FORCE)'))
        admin.dispose()


@pytest.mark.services
def test_ingest_into_seaweedfs_and_postgres(
    sync: tuple[SyncScenario, Path],
    tmp_path: Path,
    pg: sa.Engine,
    write_manifest: ManifestWriter,
) -> None:
    store = S3Store.from_env("dlp-raw")
    sid = f"ing-{uuid.uuid4().hex[:8]}"
    manifest, base = load_manifest(write_manifest(tmp_path, sync[1], session_id=sid))

    with pg.begin() as conn:
        first = ingest_session(manifest, base, store, conn)
    assert first.db == "inserted" and len(first.uploaded) == 8
    assert first.session.reference_stream.uri == f"s3://dlp-raw/sessions/{sid}/raw/bodycam.mp4"

    with pg.begin() as conn:
        second = ingest_session(manifest, base, store, conn)
    assert second.db == "unchanged" and second.uploaded == []

    changed, _ = load_manifest(write_manifest(tmp_path, sync[1], session_id=sid, worker_id="w99"))
    with pytest.raises(SessionConflictError), pg.begin() as conn:
        ingest_session(changed, base, store, conn)
