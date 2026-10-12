"""테스트용 계약 객체 생성 도우미. 다른 패키지의 테스트에서도 쓴다.

기본값으로 검증을 통과하는 최소한의 세션·라벨·행동 페이로드를 만들고, 키워드 인자로 필요한 필드만
덮어쓴다. 실제 영상·개인정보는 쓰지 않는다 (CLAUDE.md: 테스트는 합성 데이터로).

주요 이름
    - `FIXED_TIME`: 테스트 전반에서 쓰는 고정 시각 (UTC, 결정적 테스트용).
    - `make_session`: 바디캠(기준) + IMU 두 스트림을 가진 세션 `s001`.
    - `make_label`: 사람 출처 라벨 하나 (세션 `s001`, 온톨로지 1.0.0).
    - `action_payload`: 온톨로지 v1에서 유효한 원시 동작(grasp) 페이로드 사전.
    - `sqlite_db`: 시간대를 보존하는 SQLite 테스트 DB (서비스 없이 저장소 함수를 시험할 때).
    - AI 1차 검수 기록(WP17): `AI_VERSION`, `AI_TARGET_RULE`, `make_ai_verdict`, `make_ai_run`,
      `make_ai_unit`.

주의
    - 기본 세션의 스트림 URI는 원본 버킷(`s3://dlp-raw/...`)을 가리킨다. 실제 저장소에 접근하지 않는
      순수 계약 객체일 뿐이다. 원본 URI 노출 검사 테스트에서는 이 점을 감안한다.
    - 공간 라벨(box_track 등)을 만들 때는 `stream_id`를 함께 넘겨야 검증을 통과한다.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects.sqlite import DATETIME as SQLITE_DATETIME

from dlp_schema.aireview import AiReviewRun, AiReviewUnit, AiTargetRule, AiVerdict
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


# 테스트용 AI 판정 버전 (`aireview-<정책 해시 12자>+vlm-<모델>` 형식)
AI_VERSION = "aireview-0123456789ab+vlm-Qwen/Qwen3-VL-8B-Instruct"
# 판정 대상 조건의 정책 값 (config/policies/aireview.yaml kinds·temporal.long_max_span_ms와 같은 값,
# 검수 화면 판정률 분모 테스트용)
AI_TARGET_RULE = AiTargetRule(
    kinds=frozenset(
        ("action", "gap", "hand_state", "event", "segment", "object_state", "box_track",
         "keypoint_track")
    ),
    sources=frozenset({"model"}),
    long_max_span_ms=600_000,
)  # fmt: skip


def make_ai_verdict(label_id: str, value: str = "ok", **overrides: Any) -> AiVerdict:
    """AI 판정 하나 (기본: 행동 라벨, ok). wrong·suspect는 문제 wrong_verb, unsure는 not_visible.

    Args:
        label_id: 판정한 라벨.
        value: ok | suspect | wrong | unsure.
        **overrides: 그 밖의 `AiVerdict` 필드 (예: `kind`, `suggestion`, `answered`).
    """
    data: dict[str, Any] = {
        "label_id": label_id,
        "kind": "action",
        "verdict": value,
        "reason_ko": "AI: 문제 없음" if value == "ok" else "동사가 다름",
    }
    if value in ("wrong", "suspect"):
        data["problems"] = ("wrong_verb",)
    if value == "unsure":
        data["problems"] = ("not_visible",)
    data.update(overrides)
    return AiVerdict.model_validate(data)


def make_ai_run(
    session_id: str = "s001", run_id: str = "air-run-" + "0" * 24, **overrides: Any
) -> AiReviewRun:
    """AI 판정 실행 세대 하나 (바디캠 렌더 해시 a…, 센서 없음, 전량 선택, 시각 FIXED_TIME).

    Args:
        session_id: 세션.
        run_id: 실행 ID (`air-run-` + 16진수 24자).
        **overrides: 그 밖의 `AiReviewRun` 필드.
    """
    data: dict[str, Any] = {
        "run_id": run_id,
        "session_id": session_id,
        "version": AI_VERSION,
        "policy_digest": "0123456789ab",
        "prompt_version": 1,
        "backend": {
            "kind": "openai_compatible",
            "model": "Qwen/Qwen3-VL-8B-Instruct",
            "fingerprint": "vllm-bf16",
            "endpoints": ("spark-01",),
            "same_model_as_actions": False,
        },
        "ontology_version": "1.0.0",
        "render_hashes": {"bodycam": "a" * 64},
        "sensor_sha256": "absent",
        "selection": {"mode": "all", "audit_ratio": 0.05},
        "plan_digest": "0" * 16,
        "planned_units": 1,
        "created_at": FIXED_TIME,
    }
    data.update(overrides)
    return AiReviewRun.model_validate(data)


def make_ai_unit(
    unit_id: str,
    verdicts: Sequence[AiVerdict],
    *,
    run_id: str = "air-run-" + "0" * 24,
    session_id: str = "s001",
    **overrides: Any,
) -> AiReviewUnit:
    """AI 판정 단위 하나 (temporal, 바디캠 0~10초, 렌더 해시 a…, 대상 = 판정 순서).

    Args:
        unit_id: 단위 ID (`air-` + 16진수 24자).
        verdicts: 대상마다 판정 하나 (대상 라벨 ID는 이 순서).
        run_id: 처음 쓴 실행.
        session_id: 세션.
        **overrides: 그 밖의 `AiReviewUnit` 필드.
    """
    data: dict[str, Any] = {
        "unit_id": unit_id,
        "run_id": run_id,
        "session_id": session_id,
        "version": AI_VERSION,
        "group": "temporal",
        "stream_id": "bodycam",
        "t_start_ms": 0,
        "t_end_ms": 10_000,
        "target_label_ids": tuple(v.label_id for v in verdicts),
        "render_hash": "a" * 64,
        "sensor_sha256": "absent",
        "ontology_version": "1.0.0",
        "prompt_digest": "b" * 64,
        "outcome": "answered",
        "attempts": 1,
        "sub_requests": 1,
        "verdicts": tuple(verdicts),
        "created_at": FIXED_TIME,
    }
    data.update(overrides)
    return AiReviewUnit.model_validate(data)


class _AwareSqliteDateTime(SQLITE_DATETIME):
    """SQLite 시각 열: UTC로 바꿔 저장하고 읽을 때 UTC 시간대를 붙인다.

    SQLite는 시간대를 저장하지 않아 계약(AwareDatetime)이 읽기에서 실패한다. 테스트 DB에만 쓴다.
    """

    # 반환 타입은 SQLAlchemy의 비공개 처리기 타입이라 적지 않고 추론에 맡긴다
    def bind_processor(self, dialect: sa.Dialect):
        """시간대 있는 시각을 UTC 기준 시간대 없는 값으로 바꿔 넘긴다."""
        base = super().bind_processor(dialect)

        def process(value: Any) -> Any:
            if isinstance(value, datetime) and value.tzinfo is not None:
                value = value.astimezone(UTC).replace(tzinfo=None)
            return base(value) if base is not None else value

        return process

    def result_processor(self, dialect: sa.Dialect, coltype: object):
        """읽은 시각에 UTC 시간대를 붙인다."""
        base = super().result_processor(dialect, coltype)

        def process(value: Any) -> Any:
            out = base(value) if base is not None else value
            return out.replace(tzinfo=UTC) if isinstance(out, datetime) else out

        return process


def sqlite_db(url: str = "sqlite://", *, create: bool = True) -> sa.Engine:
    """시간대를 보존하는 SQLite 테스트 엔진. create면 `tables.metadata`로 표를 만든다.

    PostgreSQL 트리거(불변·추가 전용)와 행 잠금은 없다. 그런 성질은 서비스 테스트에서 본다.

    Args:
        url: SQLite URL (기본 메모리 DB, 한 스레드에서 같은 연결을 쓴다).
        create: 표를 만들지 (False면 마이그레이션 시험처럼 호출자가 만든다).
    """
    from dlp_schema.db.tables import metadata  # 계약만 쓰는 테스트가 DB 정의를 읽지 않게

    engine = sa.create_engine(url)
    # 이 엔진의 방언만 시각 열 처리를 바꾼다 (다른 엔진·PostgreSQL에는 영향 없음)
    engine.dialect.colspecs = {**engine.dialect.colspecs, sa.DateTime: _AwareSqliteDateTime}
    if create:
        metadata.create_all(engine)
    return engine
