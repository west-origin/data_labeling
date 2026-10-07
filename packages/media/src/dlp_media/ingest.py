"""세션 수집.

매니페스트(세션 메타데이터 + 스트림별 로컬 파일)를 받아 다음을 한다.

1. 원본 파일을 원본 버킷에 올린다 (`sessions/<세션>/raw/<스트림><확장자>`). 불변이며 멱등이다.
2. 영상 스트림: PTS 인덱스와 프록시 영상을 만들어 `sessions/<세션>/derived/`에 둔다.
3. 바디캠에 내장 IMU가 있고 매니페스트에 IMU 스트림이 없으면 추출해 IMU 스트림을 추가한다.
4. 센서 스트림(IMU, 장갑): 정규화된 Parquet을 만들고 스트림 uri가 이를 가리키게 한다.
5. 세션 계약 객체를 만들고 (DB 연결이 있으면) 등록한다.

같은 매니페스트로 다시 실행하면 아무것도 바꾸지 않는다. 파생 파일은 이미 있으면 다시 만들지
않는다 (인코딩 결과가 실행마다 바이트 단위로 같다는 보장이 없기 때문이다).
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import sqlalchemy as sa
import yaml
from pydantic import AwareDatetime, Field
from sqlalchemy.exc import NoResultFound

from dlp_media.glove import read_glove
from dlp_media.imu import ImuData, extract_embedded_imu, read_imu_table
from dlp_media.probe import probe
from dlp_media.proxy import make_proxy
from dlp_media.pts import build_pts_index
from dlp_media.storage import ObjectStore, put_immutable
from dlp_schema.common import Contract, Identifier, SemVer
from dlp_schema.db.repository import get_session, insert_session
from dlp_schema.session import Calibration, Domain, Session, Stream, StreamKind, SyncMethod

VIDEO_KINDS = {StreamKind.BODYCAM, StreamKind.THIRD_PERSON}
GLOVE_KINDS = {StreamKind.GLOVE_LEFT, StreamKind.GLOVE_RIGHT}


class ManifestStream(Contract):
    stream_id: Identifier
    kind: StreamKind
    path: str = Field(description="매니페스트 파일 기준 상대 경로 또는 절대 경로")


class SessionManifest(Contract):
    session_id: Identifier
    domain: Domain
    worker_id: Identifier
    site_id: Identifier
    consent_version: str
    recorded_at: AwareDatetime | None = Field(
        default=None, description="없으면 바디캠 컨테이너의 creation_time을 쓴다"
    )
    calibration: Calibration = Calibration()
    ontology_version: SemVer | None = None
    streams: tuple[ManifestStream, ...]


def load_manifest(path: Path) -> tuple[SessionManifest, Path]:
    data: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    return SessionManifest.model_validate(data), path.parent


class SessionConflictError(RuntimeError):
    pass


@dataclass
class IngestResult:
    session: Session
    uploaded: list[str] = field(default_factory=list[str])
    skipped: list[str] = field(default_factory=list[str])
    db: str = "skipped"  # inserted | unchanged | skipped


def ingest_session(
    manifest: SessionManifest,
    base_dir: Path,
    raw: ObjectStore,
    conn: sa.Connection | None = None,
) -> IngestResult:
    sid = manifest.session_id
    prefix = f"sessions/{sid}"
    kinds = [s.kind for s in manifest.streams]
    if kinds.count(StreamKind.BODYCAM) != 1:
        raise ValueError("매니페스트에는 바디캠 스트림이 정확히 하나 있어야 합니다")
    result_uploaded: list[str] = []
    result_skipped: list[str] = []

    def put(key: str, path: Path) -> str:
        (result_uploaded if put_immutable(raw, key, path) else result_skipped).append(key)
        return raw.uri(key)

    def derived(key: str, make: Callable[[Path], None]) -> str:
        """파생 파일은 없을 때만 만든다."""
        if raw.head(key) is not None:
            result_skipped.append(key)
            return raw.uri(key)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / Path(key).name
            make(out)
            return put(key, out)

    streams: list[Stream] = []
    duration_ms = 0.0
    recorded_at = manifest.recorded_at
    has_imu = StreamKind.IMU in kinds

    for ms in manifest.streams:
        src = Path(ms.path) if Path(ms.path).is_absolute() else base_dir / ms.path
        if not src.is_file():
            raise FileNotFoundError(src)
        raw_uri = put(f"{prefix}/raw/{ms.stream_id}{src.suffix}", src)

        if ms.kind in VIDEO_KINDS:
            info = probe(src)
            if info.video is None:
                raise ValueError(f"{src}: 비디오 트랙이 없습니다")
            index = build_pts_index(src)
            pts_uri = derived(f"{prefix}/derived/{ms.stream_id}.pts.parquet", index.write)
            derived(
                f"{prefix}/derived/{ms.stream_id}.proxy.mp4", lambda out, s=src: make_proxy(s, out)
            )
            is_body = ms.kind is StreamKind.BODYCAM
            streams.append(
                Stream(
                    stream_id=ms.stream_id,
                    kind=ms.kind,
                    uri=raw_uri,
                    pts_index_uri=pts_uri,
                    sync_method=SyncMethod.REFERENCE if is_body else SyncMethod.UNSYNCED,
                )
            )
            if is_body:
                duration_ms = index.duration_ms
                recorded_at = recorded_at or info.creation_time
                if not has_imu and (imu := extract_embedded_imu(src, info)) is not None:
                    streams.append(
                        _imu_stream("imu", imu, derived, prefix, SyncMethod.SHARED_CLOCK)
                    )
        elif ms.kind is StreamKind.IMU:
            imu = read_imu_table(src)
            streams.append(_imu_stream(ms.stream_id, imu, derived, prefix, SyncMethod.UNSYNCED))
        elif ms.kind in GLOVE_KINDS:
            glove = read_glove(src)
            uri = derived(f"{prefix}/derived/{ms.stream_id}.parquet", glove.write)
            streams.append(
                Stream(
                    stream_id=ms.stream_id,
                    kind=ms.kind,
                    uri=uri,
                    sample_rate_hz=glove.sample_rate_hz,
                    sync_method=SyncMethod.UNSYNCED,
                )
            )
        else:  # 별도 오디오 파일 등: 원본만 보관
            streams.append(Stream(stream_id=ms.stream_id, kind=ms.kind, uri=raw_uri))

    if recorded_at is None:
        raise ValueError("recorded_at이 매니페스트에도 바디캠 메타데이터에도 없습니다")
    session = Session(
        session_id=sid,
        domain=manifest.domain,
        worker_id=manifest.worker_id,
        site_id=manifest.site_id,
        consent_version=manifest.consent_version,
        recorded_at=recorded_at,
        duration_ms=round(duration_ms),
        streams=tuple(streams),
        calibration=manifest.calibration,
        ontology_version=manifest.ontology_version,
    )
    result = IngestResult(session, result_uploaded, result_skipped)
    if conn is not None:
        result.db = _register(conn, session)
    return result


def _imu_stream(
    stream_id: str,
    imu: ImuData,
    derived: Callable[[str, Callable[[Path], None]], str],
    prefix: str,
    sync: SyncMethod,
) -> Stream:
    uri = derived(f"{prefix}/derived/{stream_id}.parquet", imu.write)
    return Stream(
        stream_id=stream_id,
        kind=StreamKind.IMU,
        uri=uri,
        sample_rate_hz=imu.sample_rate_hz,
        sync_method=sync,
    )


def _register(conn: sa.Connection, session: Session) -> str:
    try:
        existing = get_session(conn, session.session_id)
    except NoResultFound:
        insert_session(conn, session)
        return "inserted"
    if existing != session:
        raise SessionConflictError(
            f"세션 {session.session_id}가 다른 내용으로 이미 등록되어 있습니다"
        )
    return "unchanged"
