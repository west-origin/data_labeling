"""테스트용 계약 객체 생성 도우미. 다른 패키지의 테스트에서도 쓴다."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from dlp_schema.labels import LabelPayload, LabelRecord, Provenance, Source
from dlp_schema.session import Session, StreamKind, SyncMethod

FIXED_TIME = datetime(2026, 10, 7, 9, 0, tzinfo=UTC)


def make_session(session_id: str = "s001", **overrides: Any) -> Session:
    data: dict[str, Any] = {
        "session_id": session_id,
        "domain": "cleaning",
        "worker_id": "w01",
        "site_id": "site01",
        "consent_version": "c1",
        "recorded_at": FIXED_TIME,
        "duration_ms": 60_000,
        "streams": [
            {
                "stream_id": "bodycam",
                "kind": StreamKind.BODYCAM,
                "uri": "s3://dlp-raw/s001/bodycam.mp4",
                "sync_method": SyncMethod.REFERENCE,
            },
            {
                "stream_id": "imu",
                "kind": StreamKind.IMU,
                "uri": "s3://dlp-raw/s001/imu.parquet",
                "sample_rate_hz": 200.0,
                "sync_method": SyncMethod.SHARED_CLOCK,
            },
        ],
        "ontology_version": "1.0.0",
    }
    data.update(overrides)
    return Session.model_validate(data)


def make_label(
    payload: LabelPayload | dict[str, Any],
    label_id: str = "l001",
    t_start_ms: int = 0,
    t_end_ms: int = 1_000,
    **overrides: Any,
) -> LabelRecord:
    data: dict[str, Any] = {
        "label_id": label_id,
        "session_id": "s001",
        "t_start_ms": t_start_ms,
        "t_end_ms": t_end_ms,
        "ontology_version": "1.0.0",
        "provenance": Provenance(source=Source.HUMAN),
        "created_at": FIXED_TIME,
        "payload": payload,
    }
    data.update(overrides)
    return LabelRecord.model_validate(data)


def action_payload(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "kind": "action",
        "action_id": "a001",
        "hand": "right",
        "verb": "grasp",
        "target_id": "rag_01",
        "t_approach_ms": 0,
        "t_contact_start_ms": 400,
        "t_contact_end_ms": 900,
        "t_end_ms": 1_000,
    }
    data.update(overrides)
    return data
