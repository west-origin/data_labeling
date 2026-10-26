"""카메라 내부 파라미터·작업자 손 크기 등록 저장·조회 (ADR 0045, 0052).

테이블: `camera_calibrations`, `worker_hand_sizes`(+ 추정 출처 `worker_hand_size_sources`),
무효 기록 `worker_hand_size_invalidations`, 세션 기기 연결 `session_camera_assignments`. 모두
추가만 한다 (Alembic 리비전 0022·0024의 UPDATE·DELETE 금지 트리거).

멱등: 기기 등록은 같은 기기·유효 시작의 지금 등록과 내부 파라미터가 같으면, 손 크기는 그 작업자의
지금 쓰이는 값·출처·출처 세션이 같으면, 기기 연결은 그 세션의 최신 연결이 같은 기기면 새 행을 쓰지
않는다. 어느 등록을 쓸지는 `dlp_schema.calibration`의 순수 함수가 정한다.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import sqlalchemy as sa

from dlp_schema.calibration import (
    CalibrationMethod,
    CameraCalibration,
    HandSizeInvalidation,
    HandSizeSource,
    ResolvedIntrinsics,
    ResolvedPalm,
    SessionCameraAssignment,
    WorkerHandSize,
    assignment_id_for,
    calibration_id_for,
    effective_hand_size,
    hand_size_id_for,
    hand_size_invalid_reason,
    invalidation_id_for,
    resolve_intrinsics,
    resolve_palm,
    session_camera_id,
)
from dlp_schema.db.repository import withdrawn_session_ids
from dlp_schema.db.tables import (
    camera_calibrations,
    session_camera_assignments,
    sessions,
    worker_hand_size_invalidations,
    worker_hand_size_sources,
    worker_hand_sizes,
)
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
    돌려준다. 멱등 비교는 내부 파라미터만 본다 (method·rms_px·n_views·source_sha256이 달라도 새로
    쓰지 않는다). 값 검사(`dlp_media.calibration.check_intrinsics`)는 호출자가 먼저 한다.

    호출자 트랜잭션 안에서 읽고 쓴다. 동시 등록은 직렬화하지 않으므로 같은 값이 두 행 생길 수 있다
    (결정은 최신 등록을 쓴다).

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


def _sources(conn: sa.Connection, record_ids: list[str]) -> dict[str, tuple[str, ...]]:
    """등록 ID → 추정 출처 세션 ID (ID 순)."""
    if not record_ids:
        return {}
    rows = conn.execute(
        sa.select(worker_hand_size_sources.c.record_id, worker_hand_size_sources.c.session_id)
        .where(worker_hand_size_sources.c.record_id.in_(record_ids))
        .order_by(worker_hand_size_sources.c.session_id)
    ).all()
    out: dict[str, list[str]] = {}
    for record_id, session_id in rows:
        out.setdefault(str(record_id), []).append(str(session_id))
    return {k: tuple(v) for k, v in out.items()}


def get_worker_hand_sizes(conn: sa.Connection, worker_id: str) -> list[WorkerHandSize]:
    """그 작업자의 손 크기 등록 (등록 시각 순, 추정 출처 세션 포함)."""
    rows = list(
        conn.execute(
            sa.select(worker_hand_sizes)
            .where(worker_hand_sizes.c.worker_id == worker_id)
            .order_by(worker_hand_sizes.c.created_at, worker_hand_sizes.c.record_id)
        ).mappings()
    )
    sources = _sources(conn, [str(r["record_id"]) for r in rows])
    return [
        WorkerHandSize.model_validate(
            {
                **r,
                "source_session_ids": sources.get(str(r["record_id"]), ()),
                "created_at": _utc(r["created_at"]),
            }
        )
        for r in rows
    ]


def get_hand_size_invalidations(conn: sa.Connection, worker_id: str) -> list[HandSizeInvalidation]:
    """그 작업자의 손 크기 무효 기록 (기록 시각 순)."""
    t = worker_hand_size_invalidations
    rows = conn.execute(
        sa.select(t).where(t.c.worker_id == worker_id).order_by(t.c.created_at, t.c.invalidation_id)
    ).mappings()
    return [
        HandSizeInvalidation.model_validate({**r, "created_at": _utc(r["created_at"])})
        for r in rows
    ]


@dataclass(frozen=True)
class WorkerHandSizeState:
    """한 작업자의 손 크기 등록·무효 기록과 사용 중지 세션 (결정에 필요한 전부)."""

    records: list[WorkerHandSize]
    invalidations: list[HandSizeInvalidation]
    withdrawn: frozenset[str]

    def effective(self, worker_id: str) -> WorkerHandSize | None:
        """지금 쓰이는 등록 (`calibration.effective_hand_size`)."""
        return effective_hand_size(self.records, worker_id, self.invalidations, self.withdrawn)

    def invalid_reason(self, record: WorkerHandSize) -> str | None:
        """그 등록이 쓰이지 않는 이유 (유효하면 None)."""
        return hand_size_invalid_reason(record, self.invalidations, self.withdrawn)


def hand_size_state(conn: sa.Connection, worker_id: str) -> WorkerHandSizeState:
    """그 작업자의 손 크기 결정 입력을 DB에서 읽는다."""
    return WorkerHandSizeState(
        get_worker_hand_sizes(conn, worker_id),
        get_hand_size_invalidations(conn, worker_id),
        frozenset(withdrawn_session_ids(conn)),
    )


def register_hand_size(
    conn: sa.Connection,
    worker_id: str,
    palm_m: float,
    *,
    source: HandSizeSource,
    now: datetime,
    n_samples: int | None = None,
    source_session_ids: Iterable[str] = (),
    estimator: str | None = None,
) -> tuple[WorkerHandSize, bool]:
    """작업자 손 크기 한 건을 등록한다.

    멱등: 지금 쓰이는(유효한 최신) 등록과 값·출처·출처 세션이 같으면 쓰지 않는다. 무효가 된 등록과
    같은 값을 다시 등록하면 새 행을 쓴다 (무효 기록 뒤의 등록은 다시 쓰인다). 출처 세션은
    `worker_hand_size_sources`에 같은 트랜잭션으로 쓴다.

    호출자 트랜잭션 안에서 읽고 쓴다. 동시 등록은 직렬화하지 않으므로 중복 행이 생길 수 있다
    (결정은 유효한 최신 등록을 쓴다).

    Returns:
        (등록, 새로 썼는지).

    Raises:
        pydantic.ValidationError: palm_m이 범위(0.04~0.16 m) 밖이거나, 잰 값(measured)에 출처
            세션이나 estimator를 줄 때.
    """
    sessions_used = tuple(sorted(set(source_session_ids)))
    current = hand_size_state(conn, worker_id).effective(worker_id)
    if (
        current is not None
        and current.palm_m == palm_m
        and current.source is source
        and current.source_session_ids == sessions_used
    ):
        return current, False
    record = WorkerHandSize(
        record_id=hand_size_id_for(worker_id, palm_m, source, now),
        worker_id=worker_id,
        palm_m=palm_m,
        source=source,
        n_samples=n_samples,
        source_session_ids=sessions_used,
        estimator=estimator,
        created_at=now,
    )
    values = record.model_dump(mode="python", exclude={"source_session_ids"})
    values["source"] = source.value
    values["created_at"] = now.astimezone(UTC)
    conn.execute(worker_hand_sizes.insert().values(**values))
    if sessions_used:
        conn.execute(
            worker_hand_size_sources.insert(),
            [{"record_id": record.record_id, "session_id": s} for s in sessions_used],
        )
    return record, True


def invalidate_hand_sizes(
    conn: sa.Connection,
    worker_id: str,
    *,
    reason: str,
    now: datetime,
    record_id: str | None = None,
) -> tuple[HandSizeInvalidation | None, list[WorkerHandSize]]:
    """손 크기 등록을 무효로 한다 (무효 기록 한 행, 추가만).

    record_id가 있으면 그 등록 하나, 없으면 그 작업자의 지금까지 모든 등록. 멱등: 새로 무효가 될
    등록(지금 유효한 것)이 하나도 없으면 쓰지 않는다.

    Returns:
        (쓴 무효 기록 또는 None, 이번에 무효가 된 등록).

    Raises:
        KeyError: record_id가 그 작업자의 등록이 아닐 때.
    """
    state = hand_size_state(conn, worker_id)
    if record_id is not None and record_id not in {r.record_id for r in state.records}:
        raise KeyError(f"{worker_id}의 손 크기 등록이 아닙니다: {record_id}")
    targets = [
        r
        for r in state.records
        if (record_id is None or r.record_id == record_id) and state.invalid_reason(r) is None
    ]
    if not targets:
        return None, []
    record = HandSizeInvalidation(
        invalidation_id=invalidation_id_for(worker_id, record_id, now),
        worker_id=worker_id,
        record_id=record_id,
        reason=reason,
        created_at=now,
    )
    values = record.model_dump(mode="python")
    values["created_at"] = now.astimezone(UTC)
    conn.execute(worker_hand_size_invalidations.insert().values(**values))
    return record, targets


def get_camera_assignments(conn: sa.Connection, session_id: str) -> list[SessionCameraAssignment]:
    """그 세션의 기기 연결 기록 (기록 시각 순)."""
    t = session_camera_assignments
    rows = conn.execute(
        sa.select(t).where(t.c.session_id == session_id).order_by(t.c.created_at, t.c.assignment_id)
    ).mappings()
    return [
        SessionCameraAssignment.model_validate({**r, "created_at": _utc(r["created_at"])})
        for r in rows
    ]


class CameraAssignmentConflictError(ValueError):
    """매니페스트의 camera_id와 다른 기기로 세션을 이으려 했다."""


def assign_session_camera(
    conn: sa.Connection, session: Session, camera_id: str, *, reason: str, now: datetime
) -> tuple[SessionCameraAssignment | None, bool]:
    """세션을 바디캠 기기에 잇는다 (멱등: 지금 쓰이는 기기가 같으면 쓰지 않는다).

    매니페스트에 같은 camera_id가 있으면 기록이 필요 없다 (None, False).

    Raises:
        CameraAssignmentConflictError: 매니페스트에 다른 camera_id가 있을 때 (매니페스트 값이
            앞서므로 기록해도 쓰이지 않는다. 매니페스트가 틀렸으면 수집 기록을 바로잡는 별도
            절차가 필요하다).
    """
    manifest = session.calibration.camera_id
    if manifest is not None:
        if manifest != camera_id:
            raise CameraAssignmentConflictError(
                f"{session.session_id}: 매니페스트 기기 {manifest}가 있어 "
                f"{camera_id}로 이을 수 없습니다"
            )
        return None, False
    history = get_camera_assignments(conn, session.session_id)
    if session_camera_id(session, history) == camera_id:
        return history[-1], False
    record = SessionCameraAssignment(
        assignment_id=assignment_id_for(session.session_id, camera_id, now),
        session_id=session.session_id,
        camera_id=camera_id,
        reason=reason,
        created_at=now,
    )
    values = record.model_dump(mode="python")
    values["created_at"] = now.astimezone(UTC)
    conn.execute(session_camera_assignments.insert().values(**values))
    return record, True


def session_intrinsics(conn: sa.Connection, session: Session) -> ResolvedIntrinsics:
    """세션 3D 단계가 쓸 내부 파라미터 (`calibration.resolve_intrinsics`, 등록·연결은 DB)."""
    assignments = get_camera_assignments(conn, session.session_id)
    camera = session_camera_id(session, assignments)
    records = get_camera_calibrations(conn, camera) if camera else []
    return resolve_intrinsics(session, records, assignments)


def session_palm(conn: sa.Connection, session: Session) -> ResolvedPalm:
    """세션 작업자의 손 크기 사전값 (`calibration.resolve_palm`, 무효 기록·사용 중지 세션 반영)."""
    state = hand_size_state(conn, session.worker_id)
    return resolve_palm(state.records, session.worker_id, state.invalidations, state.withdrawn)


def worker_session_ids(conn: sa.Connection, worker_id: str) -> list[str]:
    """그 작업자의 세션 ID (ID 순). 손 크기 추정(`dlp calib palm-estimate`)이 쓴다."""
    rows = conn.execute(
        sa.select(sessions.c.session_id)
        .where(sessions.c.worker_id == worker_id)
        .order_by(sessions.c.session_id)
    ).scalars()
    return list(rows)


def hand_size_records_from_session(conn: sa.Connection, session_id: str) -> list[str]:
    """그 세션을 추정 출처로 쓴 손 크기 등록 ID (ID 순, 계보·사용 중지 안내용)."""
    rows = conn.execute(
        sa.select(worker_hand_size_sources.c.record_id)
        .where(worker_hand_size_sources.c.session_id == session_id)
        .order_by(worker_hand_size_sources.c.record_id)
    ).scalars()
    return [str(r) for r in rows]


@dataclass(frozen=True)
class HandSizeReviewItem:
    """보관·삭제 검토가 필요한 작업자 손 크기 (`hand_sizes_to_review`)."""

    worker_id: str
    record_id: str
    reason: str


def hand_sizes_to_review(conn: sa.Connection) -> list[HandSizeReviewItem]:
    """지금 쓰이는 손 크기 등록이 있는데 그 작업자의 세션이 모두 사용 중지된 작업자 (작업자 ID 순).

    동의 철회로 모든 세션을 사용 중지했다면 손 크기(신체 측정값)도 `dlp calib palm-revoke`로 무효로
    할지 정해야 한다 (`dlp ops retention`이 알린다, ADR 0052).
    """
    withdrawn = frozenset(withdrawn_session_ids(conn))
    workers = list(
        conn.execute(
            sa.select(worker_hand_sizes.c.worker_id)
            .distinct()
            .order_by(worker_hand_sizes.c.worker_id)
        ).scalars()
    )
    out: list[HandSizeReviewItem] = []
    for worker in workers:
        mine = worker_session_ids(conn, worker)
        if not mine or not set(mine) <= withdrawn:
            continue
        current = hand_size_state(conn, worker).effective(worker)
        if current is not None:
            out.append(HandSizeReviewItem(worker, current.record_id, "모든 세션 사용 중지"))
    return out
