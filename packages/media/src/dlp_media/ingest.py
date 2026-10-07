"""세션 수집.

매니페스트(세션 메타데이터 + 스트림별 로컬 파일)를 받아 다음을 한다.

1. 원본 파일을 원본 버킷에 올린다 (`sessions/<세션>/raw/<스트림><확장자>`). 불변이며 멱등이다.
2. 영상 스트림: PTS 인덱스와 프록시 영상을 만들어 `sessions/<세션>/derived/`에 둔다.
3. 바디캠에 내장 IMU가 있고 매니페스트에 IMU 스트림이 없으면 추출해 IMU 스트림을 추가한다.
4. 센서 스트림(IMU, 장갑): 정규화된 Parquet을 만들고 스트림 uri가 이를 가리키게 한다.
5. 세션 계약 객체를 만들고 (DB 연결이 있으면) 등록한다.

같은 매니페스트로 다시 실행하면 아무것도 바꾸지 않는다. 파생 파일은 이미 있으면 다시 만들지
않는다 (인코딩 결과가 실행마다 바이트 단위로 같다는 보장이 없기 때문이다).
이미 등록된 세션과는 매니페스트·미디어에서 나온 필드만 비교한다. 동기화 결과, 프라이버시·생애주기
상태, 온톨로지 이관처럼 수집 뒤 단계가 바꾸는 필드는 다시 수집해도 충돌로 보지 않는다.

프록시 인코딩 설정은 config/defaults.yaml media.proxy에서 읽는다.

WP3, ADR 0003·0020. 진입점: `dlp ingest <매니페스트>` (`dlp_cli.media_cmds`).
파이프라인의 첫 단계로, 결과 세션은 privacy_state=PENDING, lifecycle=RAW_INGESTED(계약
기본값)로 시작해 `dlp sync`, `dlp privacy detect`로 이어진다.

원본 저장소 `raw`는 CLI가 `dlp_cli.raw_access.raw_store`(DB 감사) 또는 `raw_store_offline`
(`--no-db`, 로컬 JSON Lines 감사)로 만든 감사 저장소다. 원본·파생 파일 업로드마다 write 기록이
남는다. 파생물(PTS 인덱스, 프록시, 정규화 Parquet)도 원본 버킷에 둔다 (블러 전 내용이므로).

시간: 세션 길이(`duration_ms`)는 바디캠 PTS 인덱스의 `duration_ms`를 반올림한 정수 ms다.
바디캠이 마스터 타임라인의 기준 스트림(sync_method=REFERENCE)이다 (ADR 0019).

- `SessionManifest` / `ManifestStream` / `load_manifest`: 매니페스트 계약과 로더.
- `ingest_session`: 수집 본체. `IngestResult` 반환.
- `SessionConflictError`: 같은 세션 ID가 다른 내용으로 이미 등록됨.
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
from dlp_schema.config import ProxyConfig, load_config, repo_root
from dlp_schema.db.repository import get_session, insert_session
from dlp_schema.session import (
    Calibration,
    ConsentVersion,
    Domain,
    Session,
    Stream,
    StreamKind,
    SyncMethod,
)

# PTS 인덱스·프록시를 만드는 영상 스트림 종류
VIDEO_KINDS = {StreamKind.BODYCAM, StreamKind.THIRD_PERSON}
# 장갑 정규화를 하는 스트림 종류
GLOVE_KINDS = {StreamKind.GLOVE_LEFT, StreamKind.GLOVE_RIGHT}


class ManifestStream(Contract):
    """매니페스트의 스트림 하나."""

    # 세션 안에서 고유한 스트림 ID (원본 키 이름에 쓰인다)
    stream_id: Identifier
    kind: StreamKind
    path: str = Field(description="매니페스트 파일 기준 상대 경로 또는 절대 경로")


class SessionManifest(Contract):
    """수집 매니페스트 (YAML). 세션 메타데이터와 스트림 파일 목록.

    바디캠 스트림이 정확히 하나 있어야 한다 (`ingest_session`이 확인).
    """

    session_id: Identifier
    domain: Domain
    # 작업자·장소 ID: 분할(골든·학습·검증이 겹치지 않게)과 내보내기 가명의 기준
    worker_id: Identifier
    site_id: Identifier
    consent_version: ConsentVersion = Field(description="동의서 버전 (1~64자)")
    recorded_at: AwareDatetime | None = Field(
        default=None, description="없으면 바디캠 컨테이너의 creation_time을 쓴다"
    )
    calibration: Calibration = Calibration()
    # 세션 라벨이 따를 온톨로지 버전 (없으면 나중에 정한다. 프라이버시 탐지에는 필요하다)
    ontology_version: SemVer | None = None
    streams: tuple[ManifestStream, ...]


def load_manifest(path: Path) -> tuple[SessionManifest, Path]:
    """매니페스트 YAML을 읽는다.

    Returns:
        (매니페스트, 상대 경로의 기준 디렉터리 = 매니페스트 파일이 있는 디렉터리).

    Raises:
        pydantic.ValidationError: 형식이 틀렸을 때 (모르는 키 포함).
    """
    data: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    return SessionManifest.model_validate(data), path.parent


# 같은 세션 ID가 매니페스트·미디어에서 나온 필드가 다른 내용으로 이미 DB에 등록돼 있다.
class SessionConflictError(RuntimeError):
    pass


@dataclass
class IngestResult:
    """`ingest_session` 결과."""

    session: Session
    # 이번에 올린 키 (원본·파생)
    uploaded: list[str] = field(default_factory=list[str])
    # 같은 내용이 이미 있거나 파생 파일이 이미 있어 건너뛴 키
    skipped: list[str] = field(default_factory=list[str])
    db: str = "skipped"  # inserted | unchanged | skipped


def ingest_session(
    manifest: SessionManifest,
    base_dir: Path,
    raw: ObjectStore,
    conn: sa.Connection | None = None,
    *,
    proxy: ProxyConfig | None = None,
) -> IngestResult:
    """proxy: 프록시 인코딩 설정. 없으면 저장소의 config/defaults.yaml media.proxy를 읽는다.

    Args:
        manifest: 매니페스트.
        base_dir: 매니페스트의 상대 경로 기준 디렉터리.
        raw: 원본 버킷 저장소 (CLI는 감사 저장소).
        conn: DB 연결 (호출자가 연 트랜잭션). None이면 DB 등록을 건너뛴다 (db="skipped").

    Returns:
        `IngestResult`. 세션 스트림 순서는 매니페스트 순서이고, 내장 IMU 스트림("imu")은 바디캠
        바로 뒤에 들어간다.

    Raises:
        ValueError: 바디캠이 하나가 아님, 영상에 비디오 트랙 없음, recorded_at을 정할 수 없음,
            센서 파일 형식 오류.
        FileNotFoundError: 매니페스트의 파일이 없을 때.
        ImmutableObjectError: 원본 키에 다른 내용이 이미 있을 때 (원본 파일이 바뀌었다).
        SessionConflictError: DB에 같은 세션이 다른 내용으로 있을 때.

    부작용: 원본 버킷 쓰기(감사 write 기록), DB `sessions`·스트림 행 추가.
    """
    proxy_cfg = proxy if proxy is not None else default_proxy_config()
    sid = manifest.session_id
    prefix = f"sessions/{sid}"
    kinds = [s.kind for s in manifest.streams]
    if kinds.count(StreamKind.BODYCAM) != 1:
        raise ValueError("매니페스트에는 바디캠 스트림이 정확히 하나 있어야 합니다")
    result_uploaded: list[str] = []
    result_skipped: list[str] = []

    def put(key: str, path: Path) -> str:
        """불변 업로드하고 결과를 uploaded/skipped에 적는다. URI를 돌려준다."""
        (result_uploaded if put_immutable(raw, key, path) else result_skipped).append(key)
        return raw.uri(key)

    def derived(key: str, make: Callable[[Path], None]) -> str:
        """파생 파일은 없을 때만 만든다.

        make(out)이 임시 경로 out에 파일을 쓰면 그것을 올린다. 이미 있으면 내용을 확인하지 않고
        그대로 쓴다 (재인코딩 결과가 바이트 단위로 같지 않을 수 있어 불변 비교를 하지 않는다).
        """
        if raw.head(key) is not None:
            result_skipped.append(key)
            return raw.uri(key)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / Path(key).name
            make(out)
            return put(key, out)

    streams: list[Stream] = []
    duration_ms = 0.0  # 바디캠 길이 (ms, 실수) → 세션 길이
    recorded_at = manifest.recorded_at
    has_imu = StreamKind.IMU in kinds

    # 주의: 반복 변수 ms는 "manifest stream"이다 (밀리초가 아니다)
    for ms in manifest.streams:
        src = Path(ms.path) if Path(ms.path).is_absolute() else base_dir / ms.path
        if not src.is_file():
            raise FileNotFoundError(src)
        # 1) 원본 업로드 (불변)
        raw_uri = put(f"{prefix}/raw/{ms.stream_id}{src.suffix}", src)

        if ms.kind in VIDEO_KINDS:
            # 2) 영상: PTS 인덱스와 프록시
            info = probe(src)
            if info.video is None:
                raise ValueError(f"{src}: 비디오 트랙이 없습니다")
            index = build_pts_index(src)
            pts_uri = derived(f"{prefix}/derived/{ms.stream_id}.pts.parquet", index.write)
            derived(
                f"{prefix}/derived/{ms.stream_id}.proxy.mp4",
                lambda out, s=src: make_proxy(s, out, proxy_cfg),
            )
            is_body = ms.kind is StreamKind.BODYCAM
            streams.append(
                Stream(
                    stream_id=ms.stream_id,
                    kind=ms.kind,
                    uri=raw_uri,
                    pts_index_uri=pts_uri,
                    # 바디캠이 마스터 타임라인 기준, 3인칭은 동기화 전
                    sync_method=SyncMethod.REFERENCE if is_body else SyncMethod.UNSYNCED,
                )
            )
            if is_body:
                duration_ms = index.duration_ms
                recorded_at = recorded_at or info.creation_time
                # 3) 바디캠 내장 IMU: 같은 시계이므로 SHARED_CLOCK
                if not has_imu and (imu := extract_embedded_imu(src, info)) is not None:
                    streams.append(
                        _imu_stream("imu", imu, derived, prefix, SyncMethod.SHARED_CLOCK)
                    )
        elif ms.kind is StreamKind.IMU:
            # 4) IMU 사이드카: 다른 시계일 수 있어 동기화 전
            imu = read_imu_table(src)
            streams.append(_imu_stream(ms.stream_id, imu, derived, prefix, SyncMethod.UNSYNCED))
        elif ms.kind in GLOVE_KINDS:
            # 4) 장갑: 정규화 Parquet이 스트림 uri가 된다 (원본은 raw/에 따로 보관)
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
    # 5) 세션 계약 객체 (상태 필드는 계약 기본값)
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


def default_proxy_config() -> ProxyConfig:
    """저장소 루트(이 파일에서 위로 찾은 루트)의 config/defaults.yaml `media.proxy`."""
    root = repo_root(Path(__file__).parent)
    return load_config(root / "config" / "defaults.yaml").media.proxy


def _imu_stream(
    stream_id: str,
    imu: ImuData,
    derived: Callable[[str, Callable[[Path], None]], str],
    prefix: str,
    sync: SyncMethod,
) -> Stream:
    """IMU 정규화 Parquet을 (없으면) 만들고 그것을 가리키는 IMU 스트림을 만든다."""
    uri = derived(f"{prefix}/derived/{stream_id}.parquet", imu.write)
    return Stream(
        stream_id=stream_id,
        kind=StreamKind.IMU,
        uri=uri,
        sample_rate_hz=imu.sample_rate_hz,
        sync_method=sync,
    )


def _register(conn: sa.Connection, session: Session) -> str:
    """DB에 세션을 등록한다 (이미 있으면 수집 필드가 같은지만 본다).

    Returns:
        "inserted" | "unchanged".

    Raises:
        SessionConflictError: 수집 필드가 다를 때.
    """
    try:
        existing = get_session(conn, session.session_id)
    except NoResultFound:
        insert_session(conn, session)
        return "inserted"
    if _ingest_fields(existing) != _ingest_fields(session):
        raise SessionConflictError(
            f"세션 {session.session_id}가 다른 내용으로 이미 등록되어 있습니다"
        )
    return "unchanged"


# 수집 뒤 단계가 바꾸는 필드 (동기화, 프라이버시·생애주기, 온톨로지 이관)
_LATER_SESSION_FIELDS = {"privacy_state", "lifecycle_state", "ontology_version"}
# 스트림 필드 중 동기화(`dlp sync`)가 바꾸는 것
_SYNC_FIELDS = {
    "offset_ms",
    "clock_scale",
    "sync_method",
    "sync_confidence",
    "manual_adjustment_ms",
}


def _ingest_fields(session: Session) -> dict[str, Any]:
    """매니페스트와 미디어에서 나온 필드만 (재수집 충돌 판단용)."""
    data = session.model_dump(mode="json", exclude=_LATER_SESSION_FIELDS)
    data["streams"] = [
        {k: v for k, v in s.items() if k not in _SYNC_FIELDS} for s in data["streams"]
    ]
    return data
