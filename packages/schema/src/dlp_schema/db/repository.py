"""계약 타입 ↔ DB 행 변환과 기본 저장·조회. 연결(Connection)과 트랜잭션은 호출자가 관리한다."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any

import sqlalchemy as sa

from dlp_schema.dataset import DatasetVersion
from dlp_schema.db.tables import (
    dataset_split_assignments,
    dataset_versions,
    exports,
    golden_sets,
    label_records,
    model_versions,
    ontology_versions,
    review_assignments,
    review_tasks,
    sessions,
    streams,
    training_runs,
    withdrawals,
)
from dlp_schema.labels import LabelRecord, VerificationState
from dlp_schema.lineage import (
    ExportRecord,
    GoldenSet,
    ModelStatus,
    ModelVersion,
    TrainingRun,
    Withdrawal,
)
from dlp_schema.ontology import Ontology
from dlp_schema.review import (
    AssignmentStatus,
    ReviewAssignment,
    ReviewStage,
    ReviewTask,
    ReviewTaskStatus,
)
from dlp_schema.session import LifecycleState, PrivacyState, Session, Stream, can_transition


class TransitionError(ValueError):
    pass


# ---------------------------------------------------------------- 온톨로지


def register_ontology(conn: sa.Connection, ontology: Ontology) -> None:
    """온톨로지 버전을 등록한다. 같은 버전이 다른 내용으로 이미 있으면 오류."""
    content = ontology.model_dump(mode="json")
    existing = conn.execute(
        sa.select(ontology_versions.c.content).where(
            ontology_versions.c.version == ontology.version
        )
    ).scalar_one_or_none()
    if existing is None:
        conn.execute(
            ontology_versions.insert().values(
                version=ontology.version, status=ontology.status, content=content
            )
        )
    elif existing != content:
        raise ValueError(f"온톨로지 {ontology.version}이 다른 내용으로 이미 등록되어 있습니다")


# ---------------------------------------------------------------- 세션


def insert_session(conn: sa.Connection, session: Session) -> None:
    data = session.model_dump(mode="json")
    conn.execute(
        sessions.insert().values(
            session_id=session.session_id,
            domain=data["domain"],
            worker_id=session.worker_id,
            site_id=session.site_id,
            consent_version=session.consent_version,
            recorded_at=session.recorded_at,
            duration_ms=session.duration_ms,
            calibration=data["calibration"],
            privacy_state=data["privacy_state"],
            lifecycle_state=data["lifecycle_state"],
            ontology_version=session.ontology_version,
        )
    )
    rows = [
        {"session_id": session.session_id, "position": i, **s}
        for i, s in enumerate(data["streams"])
    ]
    conn.execute(streams.insert(), rows)


def get_session(conn: sa.Connection, session_id: str) -> Session:
    row = (
        conn.execute(sa.select(sessions).where(sessions.c.session_id == session_id))
        .mappings()
        .one()
    )
    stream_rows = (
        conn.execute(
            sa.select(streams)
            .where(streams.c.session_id == session_id)
            .order_by(streams.c.position)
        )
        .mappings()
        .all()
    )
    data = _without(row, "created_at")
    data["streams"] = [_without(s, "session_id", "position") for s in stream_rows]
    return Session.model_validate(data)


def set_lifecycle(conn: sa.Connection, session_id: str, target: LifecycleState) -> None:
    current = LifecycleState(
        conn.execute(
            sa.select(sessions.c.lifecycle_state).where(sessions.c.session_id == session_id)
        ).scalar_one()
    )
    if not can_transition(current, target):
        raise TransitionError(f"{session_id}: {current} → {target} 전이는 허용되지 않습니다")
    conn.execute(
        sessions.update()
        .where(sessions.c.session_id == session_id)
        .values(lifecycle_state=target.value)
    )


def set_privacy_state(conn: sa.Connection, session_id: str, state: PrivacyState) -> None:
    result = conn.execute(
        sessions.update()
        .where(sessions.c.session_id == session_id)
        .values(privacy_state=state.value)
    )
    if result.rowcount != 1:
        raise KeyError(session_id)


def update_stream_sync(conn: sa.Connection, session_id: str, stream: Stream) -> None:
    """스트림의 동기화 결과(오프셋, 드리프트, 방법, 신뢰도, 사람 조정값)만 갱신한다."""
    result = conn.execute(
        streams.update()
        .where(streams.c.session_id == session_id, streams.c.stream_id == stream.stream_id)
        .values(
            offset_ms=stream.offset_ms,
            clock_scale=stream.clock_scale,
            sync_method=stream.sync_method.value,
            sync_confidence=stream.sync_confidence,
            manual_adjustment_ms=stream.manual_adjustment_ms,
        )
    )
    if result.rowcount != 1:
        raise KeyError(f"{session_id}/{stream.stream_id}")


# ---------------------------------------------------------------- 라벨


def label_to_row(label: LabelRecord) -> dict[str, Any]:
    data = label.model_dump(mode="json")
    return {
        "label_id": label.label_id,
        "session_id": label.session_id,
        "stream_id": label.stream_id,
        "kind": label.kind,
        "t_start_ms": label.t_start_ms,
        "t_end_ms": label.t_end_ms,
        "ontology_version": label.ontology_version,
        "source": data["provenance"]["source"],
        "model_version": label.provenance.model_version,
        "sensor_id": label.provenance.sensor_id,
        "evidence": data["evidence"],
        "confidence": label.confidence,
        "verification_state": data["verification"]["state"],
        "reviewer_id": label.verification.reviewer_id,
        "reviewed_at": label.verification.reviewed_at,
        "parent_label_id": label.parent_label_id,
        "retracted": label.retracted,
        "seeded_error": label.seeded_error,
        "measurement": label.measurement,
        "created_at": label.created_at,
        "payload": data["payload"],
    }


def row_to_label(row: Mapping[Any, Any]) -> LabelRecord:
    return LabelRecord.model_validate(
        {
            "label_id": row["label_id"],
            "session_id": row["session_id"],
            "stream_id": row["stream_id"],
            "t_start_ms": row["t_start_ms"],
            "t_end_ms": row["t_end_ms"],
            "ontology_version": row["ontology_version"],
            "provenance": {
                "source": row["source"],
                "model_version": row["model_version"],
                "sensor_id": row["sensor_id"],
            },
            "evidence": row["evidence"],
            "confidence": row["confidence"],
            "verification": {
                "state": row["verification_state"],
                "reviewer_id": row["reviewer_id"],
                "reviewed_at": row["reviewed_at"],
            },
            "parent_label_id": row["parent_label_id"],
            "retracted": row["retracted"],
            "seeded_error": row["seeded_error"],
            "measurement": row["measurement"],
            "created_at": row["created_at"],
            "payload": row["payload"],
        }
    )


def insert_labels(conn: sa.Connection, labels: Iterable[LabelRecord]) -> int:
    rows = [label_to_row(x) for x in labels]
    if rows:
        conn.execute(label_records.insert(), rows)
    return len(rows)


def get_labels(
    conn: sa.Connection, session_id: str, kinds: Iterable[str] | None = None
) -> list[LabelRecord]:
    query = sa.select(label_records).where(label_records.c.session_id == session_id)
    if kinds is not None:
        query = query.where(label_records.c.kind.in_(list(kinds)))
    query = query.order_by(label_records.c.t_start_ms, label_records.c.label_id)
    return [row_to_label(r) for r in conn.execute(query).mappings()]


def record_review(
    conn: sa.Connection,
    label_id: str,
    state: VerificationState,
    reviewer_id: str,
    reviewed_at: datetime,
) -> None:
    """검수 상태만 갱신한다. 내용 수정은 새 레코드로 한다 (DB 트리거가 강제)."""
    if state is VerificationState.UNREVIEWED:
        raise ValueError("검수 기록에는 미검수 외의 상태가 필요합니다")
    result = conn.execute(
        label_records.update()
        .where(label_records.c.label_id == label_id)
        .values(verification_state=state.value, reviewer_id=reviewer_id, reviewed_at=reviewed_at)
    )
    if result.rowcount != 1:
        raise KeyError(label_id)


# ---------------------------------------------------------------- 검수 작업


def insert_review_task(conn: sa.Connection, task: ReviewTask) -> None:
    data = task.model_dump(mode="json")
    data["created_at"], data["collected_at"] = task.created_at, task.collected_at
    conn.execute(review_tasks.insert().values(**data))


def get_review_task(conn: sa.Connection, task_key: str) -> ReviewTask:
    row = (
        conn.execute(sa.select(review_tasks).where(review_tasks.c.task_key == task_key))
        .mappings()
        .one()
    )
    return ReviewTask.model_validate(dict(row))


def list_review_tasks(
    conn: sa.Connection, session_id: str, stage: ReviewStage | None = None
) -> list[ReviewTask]:
    query = sa.select(review_tasks).where(review_tasks.c.session_id == session_id)
    if stage is not None:
        query = query.where(review_tasks.c.stage == stage.value)
    rows = conn.execute(query.order_by(review_tasks.c.created_at)).mappings().all()
    return [ReviewTask.model_validate(dict(r)) for r in rows]


def mark_review_task_collected(conn: sa.Connection, task_key: str, at: datetime) -> None:
    result = conn.execute(
        review_tasks.update()
        .where(review_tasks.c.task_key == task_key)
        .values(status=ReviewTaskStatus.COLLECTED.value, collected_at=at)
    )
    if result.rowcount != 1:
        raise KeyError(task_key)


def insert_assignment(conn: sa.Connection, a: ReviewAssignment) -> None:
    data = a.model_dump(mode="json")
    data["created_at"], data["completed_at"] = a.created_at, a.completed_at
    conn.execute(review_assignments.insert().values(**data))


def get_assignment(conn: sa.Connection, assignment_id: str) -> ReviewAssignment:
    row = (
        conn.execute(
            sa.select(review_assignments).where(review_assignments.c.assignment_id == assignment_id)
        )
        .mappings()
        .one()
    )
    return ReviewAssignment.model_validate(dict(row))


def list_assignments(
    conn: sa.Connection, session_id: str | None = None, status: AssignmentStatus | None = None
) -> list[ReviewAssignment]:
    """우선순위 높은 순 (같으면 만든 순)."""
    query = sa.select(review_assignments)
    if session_id is not None:
        query = query.where(review_assignments.c.session_id == session_id)
    if status is not None:
        query = query.where(review_assignments.c.status == status.value)
    query = query.order_by(
        review_assignments.c.priority.desc(),
        review_assignments.c.created_at,
        review_assignments.c.assignment_id,
    )
    return [ReviewAssignment.model_validate(dict(r)) for r in conn.execute(query).mappings()]


def update_assignment(conn: sa.Connection, assignment_id: str, **values: object) -> None:
    """배정 상태·담당자·작업 키만 바꾼다."""
    allowed = {"assignee", "task_key", "status", "completed_at"}
    if not set(values) <= allowed:
        raise ValueError(f"바꿀 수 없는 필드: {set(values) - allowed}")
    data = {k: (v.value if isinstance(v, AssignmentStatus) else v) for k, v in values.items()}
    result = conn.execute(
        review_assignments.update()
        .where(review_assignments.c.assignment_id == assignment_id)
        .values(**data)
    )
    if result.rowcount != 1:
        raise KeyError(assignment_id)


# ---------------------------------------------------------------- 데이터셋 버전


def insert_dataset_version(conn: sa.Connection, version: DatasetVersion) -> None:
    data = version.model_dump(mode="json")
    conn.execute(
        dataset_versions.insert().values(
            version_id=version.version_id,
            parent_version_id=version.parent_version_id,
            ontology_version=version.ontology_version,
            created_at=version.created_at,
            snapshot_uri=version.snapshot_uri,
            golden_set_version=version.golden_set_version,
            excluded_sessions=data["excluded_sessions"],
        )
    )
    rows = [
        {"version_id": version.version_id, "session_id": sid, "split": split}
        for sid, split in data["splits"].items()
    ]
    if rows:
        conn.execute(dataset_split_assignments.insert(), rows)


def get_dataset_version(conn: sa.Connection, version_id: str) -> DatasetVersion:
    row = (
        conn.execute(sa.select(dataset_versions).where(dataset_versions.c.version_id == version_id))
        .mappings()
        .one()
    )
    splits = conn.execute(
        sa.select(dataset_split_assignments.c.session_id, dataset_split_assignments.c.split).where(
            dataset_split_assignments.c.version_id == version_id
        )
    ).all()
    data = _without(row)
    data["splits"] = {sid: split for sid, split in splits}
    return DatasetVersion.model_validate(data)


def _without(mapping: Mapping[Any, Any], *keys: str) -> dict[str, Any]:
    return {str(k): v for k, v in mapping.items() if k not in keys}


# ---------------------------------------------------------------- 세션 목록·계보


def list_session_ids(conn: sa.Connection, ontology_version: str | None = None) -> list[str]:
    query = sa.select(sessions.c.session_id).order_by(sessions.c.session_id)
    if ontology_version is not None:
        query = query.where(sessions.c.ontology_version == ontology_version)
    return [str(r) for r in conn.execute(query).scalars()]


def insert_golden_set(conn: sa.Connection, golden: GoldenSet) -> None:
    data = golden.model_dump(mode="json")
    data["created_at"] = golden.created_at
    conn.execute(golden_sets.insert().values(**data))


def get_golden_set(conn: sa.Connection, version: str) -> GoldenSet:
    row = (
        conn.execute(sa.select(golden_sets).where(golden_sets.c.version == version))
        .mappings()
        .one()
    )
    return GoldenSet.model_validate(dict(row))


def list_golden_sets(conn: sa.Connection, domain: str | None = None) -> list[GoldenSet]:
    query = sa.select(golden_sets).order_by(golden_sets.c.created_at)
    if domain is not None:
        query = query.where(golden_sets.c.domain == domain)
    return [GoldenSet.model_validate(dict(r)) for r in conn.execute(query).mappings()]


def insert_training_run(conn: sa.Connection, run: TrainingRun) -> None:
    conn.execute(training_runs.insert().values(**run.model_dump()))


def list_training_runs(conn: sa.Connection, dataset_version_ids: list[str]) -> list[TrainingRun]:
    query = (
        sa.select(training_runs)
        .where(training_runs.c.dataset_version_id.in_(dataset_version_ids))
        .order_by(training_runs.c.created_at)
    )
    return [TrainingRun.model_validate(dict(r)) for r in conn.execute(query).mappings()]


def get_training_run(conn: sa.Connection, run_id: str) -> TrainingRun:
    row = conn.execute(sa.select(training_runs).where(training_runs.c.run_id == run_id)).mappings()
    return TrainingRun.model_validate(dict(row.one()))


def insert_model_version(conn: sa.Connection, mv: ModelVersion) -> None:
    conn.execute(model_versions.insert().values(**mv.model_dump()))


def get_model_version(conn: sa.Connection, model_version: str) -> ModelVersion:
    query = sa.select(model_versions).where(model_versions.c.model_version == model_version)
    return ModelVersion.model_validate(dict(conn.execute(query).mappings().one()))


def list_model_versions(
    conn: sa.Connection, task: str | None = None, status: ModelStatus | None = None
) -> list[ModelVersion]:
    query = sa.select(model_versions).order_by(
        model_versions.c.created_at, model_versions.c.model_version
    )
    if task is not None:
        query = query.where(model_versions.c.task == task)
    if status is not None:
        query = query.where(model_versions.c.status == status.value)
    return [ModelVersion.model_validate(dict(r)) for r in conn.execute(query).mappings()]


def set_model_status(
    conn: sa.Connection,
    model_version: str,
    status: ModelStatus,
    at: datetime,
    report_uri: str | None = None,
) -> None:
    values: dict[str, Any] = {"status": status.value, "decided_at": at}
    if report_uri is not None:
        values["report_uri"] = report_uri
    conn.execute(
        model_versions.update()
        .where(model_versions.c.model_version == model_version)
        .values(**values)
    )


def insert_export(conn: sa.Connection, export: ExportRecord) -> None:
    data = export.model_dump(mode="json")
    data["created_at"] = export.created_at
    conn.execute(exports.insert().values(**data))


def list_exports(conn: sa.Connection, dataset_version_ids: list[str]) -> list[ExportRecord]:
    query = (
        sa.select(exports)
        .where(exports.c.dataset_version_id.in_(dataset_version_ids))
        .order_by(exports.c.created_at)
    )
    return [ExportRecord.model_validate(dict(r)) for r in conn.execute(query).mappings()]


def insert_withdrawal(conn: sa.Connection, withdrawal: Withdrawal) -> None:
    conn.execute(withdrawals.insert().values(**withdrawal.model_dump()))


def withdrawn_session_ids(conn: sa.Connection) -> set[str]:
    return {str(r) for r in conn.execute(sa.select(withdrawals.c.session_id)).scalars()}


def dataset_versions_with_session(conn: sa.Connection, session_id: str) -> list[str]:
    query = (
        sa.select(dataset_split_assignments.c.version_id)
        .where(dataset_split_assignments.c.session_id == session_id)
        .order_by(dataset_split_assignments.c.version_id)
    )
    return [str(r) for r in conn.execute(query).scalars()]
