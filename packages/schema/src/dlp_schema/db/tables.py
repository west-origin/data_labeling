"""PostgreSQL 테이블 정의 (SQLAlchemy Core). 스키마 변경은 반드시 Alembic 마이그레이션으로 한다.

역할
    DB 스키마의 "정답" 정의다. `db.repository`가 이 테이블 객체로 질의하고, Alembic 마이그레이션
    (`db/migrations/versions/*.py`)이 실제 DB를 같은 모양으로 만든다 (WP1, ADR 0002).

규칙 (CLAUDE.md)
    - 이 파일을 바꾸면 반드시 새 Alembic 리비전을 함께 추가한다. 테스트
      `test_migrations_match_table_definitions`가 SQLite에 마이그레이션을 적용한 결과와 이
      메타데이터를 비교해 다르면 실패한다.
    - 계약(`dlp_schema` Pydantic 타입)을 바꾸면 ADR + 마이그레이션 + 계약 테스트 + `make schemas`를
      함께 한다.

불변·추가 전용 테이블 (PostgreSQL 트리거, SQLite 테스트 DB에는 트리거 없음)
    - label_records: 검수 상태 열(verification_state, reviewer_id, reviewed_at) 외 수정 금지, 삭제
      금지 (0001, 0005), TRUNCATE 금지 (0010).
    - raw_access_log, review_work, privacy_audits, retention_decisions: 수정·삭제·TRUNCATE 금지
      (0009, 0010).
    - session_lifecycle_events: 수정·삭제·TRUNCATE 금지 (0011).

열 규약
    - 시각(`Ts`)은 모두 `TIMESTAMP WITH TIME ZONE`. 파이썬 쪽도 시간대 있는 datetime만 쓴다.
    - 라벨·세션의 ms 값은 BIGINT. 프레임 번호는 저장하지 않는다.
    - 목록·사전 값은 `Json`(PostgreSQL JSONB, 그 밖 JSON)에 계약의 JSON 직렬화 형태로 넣는다.
    - ID 열은 대부분 VARCHAR(128) = `common.IDENTIFIER_MAX`.
"""

from __future__ import annotations

from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# 제약 이름 규칙. Alembic 마이그레이션의 `op.f(...)` 이름과 같아야 비교 테스트를 통과한다.
metadata = sa.MetaData(
    naming_convention={
        "ix": "ix_%(column_0_label)s",
        "uq": "uq_%(table_name)s_%(column_0_name)s",
        "ck": "ck_%(table_name)s_%(constraint_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "pk": "pk_%(table_name)s",
    }
)

# JSON 열 타입: PostgreSQL에서는 JSONB(색인·비교 가능), SQLite 테스트에서는 일반 JSON.
Json: sa.types.TypeEngine[Any] = sa.JSON().with_variant(JSONB(), "postgresql")
# 시간대 있는 시각 열 타입.
Ts = sa.DateTime(timezone=True)

# 등록된 온톨로지 버전 (`register_ontology`). content는 `Ontology.model_dump(mode="json")` 전체.
# status: draft(덧붙이기 허용) | frozen(확정, 내용 변경 금지).
ontology_versions = sa.Table(
    "ontology_versions",
    metadata,
    sa.Column("version", sa.String(32), primary_key=True),
    sa.Column("status", sa.String(16), nullable=False),
    sa.Column("content", Json, nullable=False),
    sa.Column("created_at", Ts, nullable=False, server_default=sa.func.now()),
)

# 세션 (`insert_session`, `get_session`). 계약 `session.Session`과 1:1 (streams는 별도 테이블).
# calibration: `Calibration` JSON. privacy_state·lifecycle_state: 열거형 값 문자열.
# worker_id·site_id 색인: 분할기·계보 조회가 작업자·장소로 묶어 찾는다.
# created_at: DB가 매기는 등록 시각 (계약에는 없다. 0011 백필의 시각으로 쓰였다).
sessions = sa.Table(
    "sessions",
    metadata,
    sa.Column("session_id", sa.String(128), primary_key=True),
    sa.Column("domain", sa.String(32), nullable=False),
    sa.Column("worker_id", sa.String(128), nullable=False, index=True),
    sa.Column("site_id", sa.String(128), nullable=False, index=True),
    sa.Column("consent_version", sa.String(64), nullable=False),
    sa.Column("recorded_at", Ts, nullable=False),
    sa.Column("duration_ms", sa.BigInteger, nullable=False),
    sa.Column("calibration", Json, nullable=False),
    sa.Column("privacy_state", sa.String(32), nullable=False),
    sa.Column("lifecycle_state", sa.String(32), nullable=False),
    sa.Column(
        "ontology_version",
        sa.String(32),
        sa.ForeignKey("ontology_versions.version"),
        nullable=True,
    ),
    sa.Column("created_at", Ts, nullable=False, server_default=sa.func.now()),
)

# 세션 생애주기 전이 기록. 추가만 한다 (수정·삭제·TRUNCATE를 트리거로 막는다, ADR 0028)
session_lifecycle_events = sa.Table(
    "session_lifecycle_events",
    metadata,
    sa.Column(
        "event_id",
        sa.BigInteger().with_variant(sa.Integer(), "sqlite"),
        primary_key=True,
        autoincrement=True,
    ),
    sa.Column(
        "session_id",
        sa.String(128),
        sa.ForeignKey("sessions.session_id"),
        nullable=False,
        index=True,
    ),
    sa.Column("from_state", sa.String(32), nullable=True),
    sa.Column("to_state", sa.String(32), nullable=False),
    sa.Column("at", Ts, nullable=False),
    sa.Column("actor", sa.Text(), nullable=True),
)

# 세션의 스트림 (`Stream`). (session_id, stream_id)가
# 기본 키. 세션을 지우면 함께 지워진다 (CASCADE).
# position: 세션 안 스트림 순서 (0부터, 0002에서 추가). get_session이 이 순서로 복원한다.
# offset_ms·clock_scale·sync_*·manual_adjustment_ms만 `update_stream_sync`로 바뀐다.
streams = sa.Table(
    "streams",
    metadata,
    sa.Column(
        "session_id",
        sa.String(128),
        sa.ForeignKey("sessions.session_id", ondelete="CASCADE"),
        primary_key=True,
    ),
    sa.Column("stream_id", sa.String(128), primary_key=True),
    sa.Column("position", sa.Integer(), nullable=False, comment="세션 안 스트림 순서"),
    sa.Column("kind", sa.String(32), nullable=False),
    sa.Column("uri", sa.Text(), nullable=False),
    sa.Column("sample_rate_hz", sa.Float(), nullable=True),
    sa.Column("pts_index_uri", sa.Text(), nullable=True),
    sa.Column("offset_ms", sa.Float(), nullable=False),
    sa.Column("clock_scale", sa.Float(), nullable=False),
    sa.Column("sync_method", sa.String(32), nullable=False),
    sa.Column("sync_confidence", sa.Float(), nullable=True),
    sa.Column("manual_adjustment_ms", sa.Float(), nullable=False),
)

# 라벨 레코드 (`LabelRecord`, 변환은 `label_to_row`/`row_to_label`).
# 출처·검수 정보는 평탄화해 열로 두고(source, model_version, sensor_id, verification_state,
# reviewer_id, reviewed_at), 페이로드는 JSON으로 둔다.
# kind는 payload.kind의 사본 (종류별 조회 색인용).
# model_version은 TEXT (정책 해시가 붙어 128자를 넘을 수 있다, 0010).
# parent_label_id는 자기 참조 FK (수정·삭제 사슬). 부모가 먼저 들어가 있어야 한다.
# stream_id는 FK가 아니다 (streams와 묶이지 않음).
label_records = sa.Table(
    "label_records",
    metadata,
    sa.Column("label_id", sa.String(128), primary_key=True),
    sa.Column("session_id", sa.String(128), sa.ForeignKey("sessions.session_id"), nullable=False),
    sa.Column("stream_id", sa.String(128), nullable=True),
    sa.Column("kind", sa.String(32), nullable=False),
    sa.Column("t_start_ms", sa.BigInteger, nullable=False),
    sa.Column("t_end_ms", sa.BigInteger, nullable=False),
    sa.Column(
        "ontology_version",
        sa.String(32),
        sa.ForeignKey("ontology_versions.version"),
        nullable=False,
    ),
    sa.Column("source", sa.String(16), nullable=False),
    sa.Column("model_version", sa.Text(), nullable=True),
    sa.Column("sensor_id", sa.String(128), nullable=True),
    sa.Column("evidence", sa.String(16), nullable=False),
    sa.Column("confidence", sa.Float(), nullable=True),
    sa.Column("verification_state", sa.String(32), nullable=False),
    sa.Column("reviewer_id", sa.String(128), nullable=True),
    sa.Column("reviewed_at", Ts, nullable=True),
    sa.Column(
        "parent_label_id",
        sa.String(128),
        sa.ForeignKey("label_records.label_id"),
        nullable=True,
        index=True,
    ),
    sa.Column("retracted", sa.Boolean, nullable=False),
    sa.Column("seeded_error", sa.Boolean, nullable=False),
    sa.Column("measurement", sa.String(16), nullable=True),
    sa.Column("created_at", Ts, nullable=False),
    sa.Column("payload", Json, nullable=False),
    sa.CheckConstraint("t_start_ms <= t_end_ms", name="time_order"),
    sa.Index("ix_label_records_session_kind", "session_id", "kind"),
    sa.Index("ix_label_records_session_start", "session_id", "t_start_ms"),
)

# 외부 검수 도구 작업 (`ReviewTask`). 색인 (session_id, stage): 세션별·단계별 작업 조회.
review_tasks = sa.Table(
    "review_tasks",
    metadata,
    sa.Column("task_key", sa.String(128), primary_key=True),
    sa.Column("tool", sa.String(32), nullable=False),
    sa.Column("external_id", sa.String(64), nullable=False),
    sa.Column("session_id", sa.String(128), sa.ForeignKey("sessions.session_id"), nullable=False),
    sa.Column("stream_id", sa.String(128), nullable=False),
    sa.Column("stage", sa.String(16), nullable=False),
    sa.Column("assignee", sa.String(128), nullable=True),
    sa.Column("media_uri", sa.Text(), nullable=False),
    sa.Column("label_kinds", Json, nullable=False),
    sa.Column("mode", sa.String(16), nullable=False),
    sa.Column("assignment_id", sa.String(128), nullable=True),
    sa.Column("sent_label_ids", Json, nullable=True),
    sa.Column("status", sa.String(16), nullable=False),
    sa.Column("created_at", Ts, nullable=False),
    sa.Column("collected_at", Ts, nullable=True),
    sa.Index("ix_review_tasks_session", "session_id", "stage"),
)

# 검수 배정 (`ReviewAssignment`). 색인 queue (status, priority): 열린 배정을 우선순위 순으로 꺼낸다.
# 수정 가능한 열은 assignee·task_key·status·completed_at뿐이다 (`update_assignment`가 강제, DB
# 트리거 없음).
review_assignments = sa.Table(
    "review_assignments",
    metadata,
    sa.Column("assignment_id", sa.String(128), primary_key=True),
    sa.Column("session_id", sa.String(128), sa.ForeignKey("sessions.session_id"), nullable=False),
    sa.Column("stream_id", sa.String(128), nullable=True),
    sa.Column("label_kinds", Json, nullable=False),
    sa.Column("mode", sa.String(16), nullable=False),
    sa.Column("priority", sa.Float(), nullable=False),
    sa.Column("flagged", Json, nullable=False),
    sa.Column("assignee", sa.String(128), nullable=True),
    sa.Column("pair_id", sa.String(128), nullable=True),
    sa.Column("injected", Json, nullable=False),
    sa.Column("sample_label_ids", Json, nullable=False),
    sa.Column("withheld_label_ids", Json, nullable=False),
    sa.Column("only_label_ids", Json, nullable=False),
    sa.Column("task_key", sa.String(128), nullable=True),
    sa.Column("status", sa.String(16), nullable=False),
    sa.Column("created_at", Ts, nullable=False),
    sa.Column("completed_at", Ts, nullable=True),
    sa.Index("ix_review_assignments_session", "session_id"),
    sa.Index("ix_review_assignments_queue", "status", "priority"),
)

# 골든셋 버전 (`GoldenSet`). session_ids는 JSON 목록.
golden_sets = sa.Table(
    "golden_sets",
    metadata,
    sa.Column("version", sa.String(128), primary_key=True),
    sa.Column("domain", sa.String(32), nullable=False),
    sa.Column("session_ids", Json, nullable=False),
    sa.Column("created_at", Ts, nullable=False),
    sa.Column("note", sa.Text(), nullable=False),
)

# 학습 실행 (`TrainingRun`). 데이터셋 버전에 FK로 묶인다.
training_runs = sa.Table(
    "training_runs",
    metadata,
    sa.Column("run_id", sa.String(128), primary_key=True),
    sa.Column(
        "dataset_version_id",
        sa.String(128),
        sa.ForeignKey("dataset_versions.version_id"),
        nullable=False,
        index=True,
    ),
    sa.Column("model_name", sa.String(128), nullable=False),
    sa.Column("model_version", sa.String(128), nullable=False),
    sa.Column("mlflow_run_id", sa.String(64), nullable=True),
    sa.Column("created_at", Ts, nullable=False),
)

# 재학습 모델 레지스트리 (`ModelVersion`, 0007). status·decided_at·report_uri는
# `set_model_status`로 바뀐다.
model_versions = sa.Table(
    "model_versions",
    metadata,
    sa.Column("model_version", sa.String(128), primary_key=True),
    sa.Column("task", sa.String(64), nullable=False, index=True),
    sa.Column(
        "run_id",
        sa.String(128),
        sa.ForeignKey("training_runs.run_id"),
        nullable=False,
        index=True,
    ),
    sa.Column("trainer", sa.String(128), nullable=False),
    sa.Column("artifact_uri", sa.Text(), nullable=False),
    sa.Column("sha256", sa.String(64), nullable=False),
    sa.Column("train_examples", sa.Integer(), nullable=False),
    sa.Column("status", sa.String(16), nullable=False, index=True),
    sa.Column("report_uri", sa.Text(), nullable=True),
    sa.Column("created_at", Ts, nullable=False),
    sa.Column("decided_at", Ts, nullable=True),
)

# 내보내기 이력 (`ExportRecord`). label_states는 0008에서 추가 (기존 행은 빈 목록).
exports = sa.Table(
    "exports",
    metadata,
    sa.Column("export_id", sa.String(128), primary_key=True),
    sa.Column(
        "dataset_version_id",
        sa.String(128),
        sa.ForeignKey("dataset_versions.version_id"),
        nullable=False,
        index=True,
    ),
    sa.Column("target", sa.String(128), nullable=False),
    sa.Column("format", sa.String(64), nullable=False),
    sa.Column("uri", sa.Text(), nullable=False),
    sa.Column("session_ids", Json, nullable=False),
    sa.Column("label_states", Json, nullable=False, server_default="[]"),
    sa.Column("created_at", Ts, nullable=False),
)

# 세션 사용 중지 (`Withdrawal`). session_id가 기본 키라 세션당 한 번만 기록할 수 있다.
withdrawals = sa.Table(
    "withdrawals",
    metadata,
    sa.Column("session_id", sa.String(128), sa.ForeignKey("sessions.session_id"), primary_key=True),
    sa.Column("reason", sa.Text(), nullable=False),
    sa.Column("withdrawn_at", Ts, nullable=False),
)

# 데이터셋 버전 (`DatasetVersion`). splits는 dataset_split_assignments에 행으로 따로 둔다.
# parent_version_id는 자기 참조 FK. golden_set_version은 golden_sets.version과 같은 128자 (FK는
# 아님, 0010).
dataset_versions = sa.Table(
    "dataset_versions",
    metadata,
    sa.Column("version_id", sa.String(128), primary_key=True),
    sa.Column(
        "parent_version_id",
        sa.String(128),
        sa.ForeignKey("dataset_versions.version_id"),
        nullable=True,
    ),
    sa.Column(
        "ontology_version",
        sa.String(32),
        sa.ForeignKey("ontology_versions.version"),
        nullable=False,
    ),
    sa.Column("created_at", Ts, nullable=False),
    sa.Column("snapshot_uri", sa.Text(), nullable=False),
    sa.Column("golden_set_version", sa.String(128), nullable=True),
    sa.Column("excluded_sessions", Json, nullable=False),
)

# 데이터셋 버전의 세션별 분할 (version_id, session_id) → split. 버전을 지우면 함께 지워진다.
# 계보 조회(`dataset_versions_with_session`)가 세션 → 버전을 이 표로 찾는다.
dataset_split_assignments = sa.Table(
    "dataset_split_assignments",
    metadata,
    sa.Column(
        "version_id",
        sa.String(128),
        sa.ForeignKey("dataset_versions.version_id", ondelete="CASCADE"),
        primary_key=True,
    ),
    sa.Column("session_id", sa.String(128), sa.ForeignKey("sessions.session_id"), primary_key=True),
    sa.Column("split", sa.String(16), nullable=False),
)


# ---------------------------------------------------------------- 운영 기록 (WP16, 추가만 한다)

# 원본 버킷 접근 감사 기록 (`RawAccessEvent`, ADR 0020). 감사
# 저장소(`dlp_cli.raw_access.raw_store`)만 쓴다.
raw_access_log = sa.Table(
    "raw_access_log",
    metadata,
    sa.Column("event_id", sa.String(128), primary_key=True),
    sa.Column("at", Ts, nullable=False, index=True),
    sa.Column("actor", sa.String(128), nullable=False, index=True),
    sa.Column("purpose", sa.String(128), nullable=False),
    sa.Column("action", sa.String(16), nullable=False),
    sa.Column("bucket", sa.String(128), nullable=False),
    sa.Column("key", sa.Text(), nullable=False),
    sa.Column("session_id", sa.String(128), nullable=True, index=True),
)

# 검수 작업 시간 (`ReviewWork`). session_id는 FK가 아니다 (운영 기록 표들은 sessions와 독립적이다).
review_work = sa.Table(
    "review_work",
    metadata,
    sa.Column("work_id", sa.String(128), primary_key=True),
    sa.Column("task_key", sa.String(128), nullable=True),
    sa.Column("session_id", sa.String(128), nullable=False, index=True),
    sa.Column("reviewer", sa.String(128), nullable=False),
    sa.Column("stage", sa.String(16), nullable=False),
    sa.Column("seconds", sa.Float(), nullable=False),
    sa.Column("video_ms", sa.BigInteger(), nullable=False),
    sa.Column("source", sa.String(16), nullable=False),
    sa.Column("recorded_at", Ts, nullable=False, index=True),
)

# 잔여 블러 누락 감사 결과 (`PrivacyAuditRecord`).
privacy_audits = sa.Table(
    "privacy_audits",
    metadata,
    sa.Column("audit_id", sa.String(128), primary_key=True),
    sa.Column("session_id", sa.String(128), nullable=False, index=True),
    sa.Column("stream_id", sa.String(128), nullable=False),
    sa.Column("duration_ms", sa.BigInteger(), nullable=False),
    sa.Column("misses", sa.Integer(), nullable=False),
    sa.Column("auditor", sa.String(128), nullable=False),
    sa.Column("blur_reviewer", sa.String(128), nullable=False),
    sa.Column("audited_at", Ts, nullable=False, index=True),
)

# 원본 보관 결정 (`RetentionDecision`). until은 DATE (연장 기한).
retention_decisions = sa.Table(
    "retention_decisions",
    metadata,
    sa.Column("decision_id", sa.String(128), primary_key=True),
    sa.Column("session_id", sa.String(128), nullable=False, index=True),
    sa.Column("decision", sa.String(16), nullable=False),
    sa.Column("until", sa.Date(), nullable=True),
    sa.Column("reason", sa.Text(), nullable=False),
    sa.Column("decided_by", sa.String(128), nullable=False),
    sa.Column("decided_at", Ts, nullable=False),
)
