"""테스트용 계약 객체 생성 도우미. 다른 패키지의 테스트에서도 쓴다.

기본값으로 검증을 통과하는 최소한의 세션·라벨·행동 페이로드를 만들고, 키워드 인자로 필요한 필드만
덮어쓴다. 실제 영상·개인정보는 쓰지 않는다 (CLAUDE.md: 테스트는 합성 데이터로).

주요 이름
    - `FIXED_TIME`: 테스트 전반에서 쓰는 고정 시각 (UTC, 결정적 테스트용).
    - `make_session`: 바디캠(기준) + IMU 두 스트림을 가진 세션 `s001`.
    - `make_label`: 사람 출처 라벨 하나 (세션 `s001`, 온톨로지 1.0.0).
    - `action_payload`: 온톨로지 v1에서 유효한 원시 동작(grasp) 페이로드 사전.

주의
    - 기본 세션의 스트림 URI는 원본 버킷(`s3://dlp-raw/...`)을 가리킨다. 실제 저장소에 접근하지 않는
      순수 계약 객체일 뿐이다. 원본 URI 노출 검사 테스트에서는 이 점을 감안한다.
    - 공간 라벨(box_track 등)을 만들 때는 `stream_id`를 함께 넘겨야 검증을 통과한다.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from dlp_schema.labels import LabelPayload, LabelRecord, Provenance, Source
from dlp_schema.session import Session, StreamKind, SyncMethod

# 테스트 고정 시각 (2026-10-07 09:00 UTC). 시간대가 있어야 계약(AwareDatetime)을 통과한다.
FIXED_TIME = datetime(2026, 10, 7, 9, 0, tzinfo=UTC)


def make_session(session_id: str = "s001", **overrides: Any) -> Session:
    """검증을 통과하는 기본 세션을 만든다.

    기본값: 도메인 cleaning, 작업자 w01, 장소 site01, 길이 60초, 스트림 두 개
    (바디캠 = 기준 스트림 reference, IMU 200Hz = 같은 시계 shared_clock), 온톨로지 1.0.0.

    Args:
        session_id: 세션 ID. 스트림 URI의 경로(`s001/...`)는 바뀌지 않는다는 점에 주의.
        **overrides: 최상위 필드를 통째로 바꾼다 (예: `streams=[...]`, `domain="nursing"`).

    Returns:
        검증된 `Session`.

    Raises:
        pydantic.ValidationError: 덮어쓴 값이 계약을 어길 때.
    """
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
    """라벨 레코드 하나를 만든다 (기본: 사람 출처, 미검수, 세션 s001, 0~1000ms).

    Args:
        payload: 페이로드 객체 또는 `kind`가 든 사전.
        label_id: 라벨 ID.
        t_start_ms / t_end_ms: 라벨 구간 (ms). 행동 라벨은 페이로드의 접근 시작·종료와 같아야 한다.
        **overrides: 그 밖의 `LabelRecord` 필드 (예: `stream_id`, `provenance`, `confidence`,
            `parent_label_id`, `retracted`, `verification`).

    Returns:
        검증된 `LabelRecord`.

    Raises:
        pydantic.ValidationError: 계약 위반 (예: 공간 라벨에 stream_id 없음,
            모델 출처에 confidence 없음).
    """
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
    """오른손 grasp 원시 동작 페이로드 사전 (0~1000ms, 접촉 400~900ms, 대상 rag_01).

    `make_label`의 기본 구간(0~1000ms)과 접근 시작·종료가 맞춰져 있다. 시각을 바꾸면 라벨 구간도
    함께 바꿔야 한다.

    Args:
        **overrides: 페이로드 필드 덮어쓰기 (예: `verb="lift"`, `pre_state={"wetness": "dry"}`).
    """
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
