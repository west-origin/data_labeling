"""세션 수집 테스트 (dlp_media.ingest, WP3, ADR 0003).

로컬 저장소 테스트는 CI에서 돈다. 마지막 테스트는 SeaweedFS·PostgreSQL이 필요하다
(`@pytest.mark.services`). 정답 근거: 동기화 픽스처는 12초, 10 fps 바디캠(프레임 120개),
100 Hz 장갑, 200 Hz IMU를 만든다.
"""

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
from dlp_schema.db.repository import (
    get_session,
    set_lifecycle,
    set_privacy_state,
    update_stream_sync,
)
from dlp_schema.session import LifecycleState, PrivacyState, StreamKind, SyncMethod
from dlp_schema.testing import FIXED_TIME

ManifestWriter = Callable[..., Path]


def _local(uri: str, store: LocalStore) -> Path:
    """로컬 저장소 URI(local://버킷/키) → 실제 파일 경로."""
    return store.root / uri.removeprefix(f"local://{store.bucket}/")


def test_ingest_builds_session_and_is_idempotent(
    sync: tuple[SyncScenario, Path],
    tmp_path: Path,
    write_manifest: ManifestWriter,
) -> None:
    """수집이 세션을 만들고, 같은 매니페스트로 다시 돌리면 아무것도 바꾸지 않는다.

    정답: 기록 시각·기준 스트림(REFERENCE)·장갑 100 Hz·길이 약 12초, 업로드 8개(원본 3 + PTS 2 +
    프록시 2 + 장갑 정규화 1), PTS 인덱스 프레임 120개, 프록시 파일이 있다. 두 번째 실행은 모두
    건너뛰고 같은 세션을 낸다.
    """
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
    """같은 스트림 원본 파일 내용이 바뀌면 다시 수집할 때 ImmutableObjectError."""
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
    """매니페스트에도 컨테이너에도 촬영 시각이 없으면 ValueError.

    픽스처 영상에는 creation_time 태그가 없다.
    """
    manifest, base = load_manifest(write_manifest(tmp_path, sync[1], recorded_at=None))
    with pytest.raises(ValueError, match="recorded_at"):
        ingest_session(manifest, base, LocalStore(tmp_path / "store", "dlp-raw"))


def test_embedded_imu_becomes_shared_clock_stream(
    sync: tuple[SyncScenario, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    write_manifest: ManifestWriter,
) -> None:
    """바디캠 내장 IMU가 있으면 "imu" 스트림(SHARED_CLOCK)을 추가한다.

    가짜 추출기가 5 ms 간격 샘플을 내므로 샘플레이트 200 Hz가 정답이다.
    """

    class FakeExtractor:
        """비디오가 있으면 항상 처리하는 가짜 IMU 추출기 (0~1초, 5 ms 간격, 값 0)."""

        name = "fake"

        def can_handle(self, info: MediaInfo) -> bool:
            """비디오 트랙이 있으면 참."""
            return info.video is not None

        def extract(self, path: Path) -> ImuData:
            """200 Hz, 1초 길이의 0 값 IMU."""
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


def test_embedded_extractor_without_samples_adds_no_imu_stream(
    sync: tuple[SyncScenario, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    write_manifest: ManifestWriter,
) -> None:
    """GPMF에 ACCL이 없으면 샘플레이트 0인 IMU 스트림 대신 IMU 없이 수집한다."""

    class EmptyExtractor:
        """처리는 하지만 샘플이 없어 None을 내는 가짜 추출기."""

        name = "empty"

        def can_handle(self, info: MediaInfo) -> bool:
            """비디오 트랙이 있으면 참."""
            return info.video is not None

        def extract(self, path: Path) -> ImuData | None:
            """샘플 없음 (None)."""
            return None

    monkeypatch.setattr(imu_module, "EXTRACTORS", [EmptyExtractor()])
    streams = [{"stream_id": "bodycam", "kind": "bodycam", "path": str(sync[1] / "bodycam.mp4")}]
    manifest, base = load_manifest(write_manifest(tmp_path, sync[1], streams=streams))
    session = ingest_session(manifest, base, LocalStore(tmp_path / "store", "dlp-raw")).session
    assert [s.kind for s in session.streams] == [StreamKind.BODYCAM]


# ---------------------------------------------------------------- 실제 서비스


@pytest.fixture
def pg() -> Iterator[sa.Engine]:
    """빈 임시 PostgreSQL DB (마이그레이션 적용). 끝나면 DB를 지운다.

    DLP_DATABASE_URL(없으면 개발 기본값)의 서버에 `dlp_test_<임의>` DB를 만든다.
    """
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
    """실제 S3·DB로 수집·재수집·충돌을 확인한다 (서비스 필요).

    첫 실행은 inserted·업로드 8개·s3://dlp-raw URI, 두 번째는 unchanged. 동기화·프라이버시·
    생애주기가 바꾼 필드는 재수집 충돌이 아니며 바뀐 값이 유지된다. 작업자 ID가 바뀐 매니페스트는
    SessionConflictError.
    """
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

    # 수집 뒤 단계(동기화, 프라이버시, 생애주기)가 바꾼 필드는 재수집 충돌이 아니다
    third = first.session.stream("third_person").model_copy(
        update={"offset_ms": 120.0, "sync_method": SyncMethod.MANUAL, "manual_adjustment_ms": 3.0}
    )
    with pg.begin() as conn:
        update_stream_sync(conn, sid, third)
        set_privacy_state(conn, sid, PrivacyState.APPROVED)
        set_lifecycle(conn, sid, LifecycleState.PRIVACY_APPROVED)
    with pg.begin() as conn:
        third_run = ingest_session(manifest, base, store, conn)
    assert third_run.db == "unchanged"
    with pg.connect() as conn:
        assert get_session(conn, sid).stream("third_person") == third

    changed, _ = load_manifest(write_manifest(tmp_path, sync[1], session_id=sid, worker_id="w99"))
    with pytest.raises(SessionConflictError), pg.begin() as conn:
        ingest_session(changed, base, store, conn)
