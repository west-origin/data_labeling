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
    label_records,
    ontology_versions,
    sessions,
    streams,
)
from dlp_schema.labels import LabelRecord, VerificationState
from dlp_schema.ontology import Ontology
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
