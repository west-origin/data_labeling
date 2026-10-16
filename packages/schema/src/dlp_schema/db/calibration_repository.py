"""카메라 내부 파라미터·작업자 손 크기 등록 저장·조회 (`camera_calibrations`, `worker_hand_sizes`).

둘 다 추가만 한다 (0022 트리거, ADR 0045). 등록 함수는 멱등이다: 같은 기기·유효 시작의 지금 등록과
내부 파라미터가 같으면(손 크기는 그 작업자의 최신 값·출처가 같으면) 새 행을 쓰지 않는다. 어느 등록을
쓸지는 `dlp_schema.calibration`의 순수 함수가 정한다.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import sqlalchemy as sa

from dlp_schema.calibration import (
    CalibrationMethod,
    CameraCalibration,
    HandSizeSource,
    ResolvedIntrinsics,
    ResolvedPalm,
    WorkerHandSize,
    calibration_id_for,
    hand_size_id_for,
    latest_hand_size,
    resolve_intrinsics,
    resolve_palm,
)
from dlp_schema.db.tables import camera_calibrations, sessions, worker_hand_sizes
from dlp_schema.session import CameraIntrinsics, Session


def _utc(t: datetime | None) -> datetime | None:
    """SQLite는 시간대를 저장하지 않는다 (UTC로 복원)."""
    if t is None:
        return None
    return t.replace(tzinfo=UTC) if t.utcoffset() is None else t


def _calibration(row: Any) -> CameraCalibration:
    return CameraCalibration.model_validate(
        {**row, "valid_from": _utc(row["valid_from"]), "created_at": _utc(row["created_at"])}
    )


def get_camera_calibrations(conn: sa.Connection, camera_id: str) -> list[CameraCalibration]:
    """그 기기의 등록 (등록 시각 순)."""
    rows = conn.execute(
        sa.select(camera_calibrations)
        .where(camera_calibrations.c.camera_id == camera_id)
        .order_by(camera_calibrations.c.created_at, camera_calibrations.c.calibration_id)
    ).mappings()
    return [_calibration(r) for r in rows]


def register_camera_calibration(
    conn: sa.Connection,
    camera_id: str,
    intrinsics: CameraIntrinsics,
    *,
    method: CalibrationMethod,
    now: datetime,
    rms_px: float | None = None,
    n_views: int | None = None,
    source_sha256: str | None = None,
    valid_from: datetime | None = None,
) -> tuple[CameraCalibration, bool]:
    """기기 캘리브레이션 한 건을 등록한다 (멱등).

    같은 기기·같은 valid_from의 가장 늦은 등록과 내부 파라미터가 같으면 쓰지 않고 그 등록을
    돌려준다.

    Returns:
        (등록, 새로 썼는지).
    """
    same_start = [r for r in get_camera_calibrations(conn, camera_id) if r.valid_from == valid_from]
    if same_start and same_start[-1].intrinsics == intrinsics:
        return same_start[-1], False
    record = CameraCalibration(
        calibration_id=calibration_id_for(camera_id, intrinsics, valid_from, now),
        camera_id=camera_id,
        intrinsics=intrinsics,
        method=method,
        rms_px=rms_px,
        n_views=n_views,
        source_sha256=source_sha256,
        valid_from=valid_from,
        created_at=now,
    )
    values = record.model_dump(mode="python")
    values["intrinsics"] = intrinsics.model_dump(mode="json")
    values["method"] = method.value
    values["created_at"] = now.astimezone(UTC)
    values["valid_from"] = valid_from.astimezone(UTC) if valid_from else None
    conn.execute(camera_calibrations.insert().values(**values))
    return record, True


def get_worker_hand_sizes(conn: sa.Connection, worker_id: str) -> list[WorkerHandSize]:
    """그 작업자의 손 크기 등록 (등록 시각 순)."""
    rows = conn.execute(
        sa.select(worker_hand_sizes)
        .where(worker_hand_sizes.c.worker_id == worker_id)
        .order_by(worker_hand_sizes.c.created_at, worker_hand_sizes.c.record_id)
    ).mappings()
    return [WorkerHandSize.model_validate({**r, "created_at": _utc(r["created_at"])}) for r in rows]


def register_hand_size(
    conn: sa.Connection,
    worker_id: str,
    palm_m: float,
    *,
    source: HandSizeSource,
    now: datetime,
    n_samples: int | None = None,
) -> tuple[WorkerHandSize, bool]:
    """작업자 손 크기 한 건을 등록한다 (멱등: 최신 등록과 값·출처가 같으면 쓰지 않는다).

    Returns:
        (등록, 새로 썼는지).

    Raises:
        pydantic.ValidationError: palm_m이 범위(0.04~0.16 m) 밖일 때.
    """
    latest = latest_hand_size(get_worker_hand_sizes(conn, worker_id), worker_id)
    if latest is not None and latest.palm_m == palm_m and latest.source is source:
        return latest, False
    record = WorkerHandSize(
        record_id=hand_size_id_for(worker_id, palm_m, source, now),
        worker_id=worker_id,
        palm_m=palm_m,
        source=source,
        n_samples=n_samples,
        created_at=now,
    )
    values = record.model_dump(mode="python")
    values["source"] = source.value
    values["created_at"] = now.astimezone(UTC)
    conn.execute(worker_hand_sizes.insert().values(**values))
    return record, True


def session_intrinsics(conn: sa.Connection, session: Session) -> ResolvedIntrinsics:
    """세션 3D 단계가 쓸 내부 파라미터 (`calibration.resolve_intrinsics`, 등록은 DB에서 읽는다)."""
    camera = session.calibration.camera_id
    records = get_camera_calibrations(conn, camera) if camera else []
    return resolve_intrinsics(session, records)


def session_palm(conn: sa.Connection, session: Session) -> ResolvedPalm:
    """세션 작업자의 손 크기 사전값 (`calibration.resolve_palm`)."""
    return resolve_palm(get_worker_hand_sizes(conn, session.worker_id), session.worker_id)


def worker_session_ids(conn: sa.Connection, worker_id: str) -> list[str]:
    """그 작업자의 세션 ID (ID 순). 손 크기 추정(`dlp calib palm-estimate`)이 쓴다."""
    rows = conn.execute(
        sa.select(sessions.c.session_id)
        .where(sessions.c.worker_id == worker_id)
        .order_by(sessions.c.session_id)
    ).scalars()
    return list(rows)
