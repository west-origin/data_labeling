"""AI 1차 검수 기록 표 다섯 개 (추가 전용, ADR 0040).

Revision ID: 0019
Revises: 0017
Create Date: 2026-10-11

판정은 라벨이 아니다. `label_records` 밖의 표에만 쓴다 (ADR 0040 결정 1, I3·I4):
- ai_review_runs: 실행 세대 (계획을 고정한 기록)
- ai_review_run_events: 실행 이벤트 (started·completed·interrupted·aborted·waived)
- ai_review_units: 판정 단위 (계획 단위 하나 = 행 하나, 판정은 JSON으로 묶는다)
- ai_review_exposures: 검수 작업에 보인 판정 (작업마다 한 번)
- ai_review_outcomes: 사람 결과와 AI 판정의 연결 (작업마다 한 번)

다섯 표 모두 추가만 한다 (I5): PostgreSQL에서 0009의 `dlp_append_only()`(행 단위 UPDATE·DELETE 거부)와
0010의 `dlp_no_truncate()`(문장 단위 TRUNCATE 거부) 트리거를 건다. `INSERT … ON CONFLICT DO NOTHING`은
UPDATE 트리거를 부르지 않으므로 같은 단위를 다시 써도 거부되지 않고 무시된다 (I8).

리비전 순서: 0018(골든 정답 고정, fix/m-trn)도 0017을 잇는다. 둘이 메인에 함께 들어가면 나중에
병합하는 쪽이 이 파일의 `down_revision`을 "0018"로 바꾼다 (head가 둘이면 `upgrade`가 실패한다,
`test_migrations_have_single_head`).
"""

from __future__ import annotations

from typing import Any

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0019"
down_revision: str | None = "0017"
branch_labels: str | None = None
depends_on: str | None = None

# 추가 전용 트리거를 거는 표 (만드는 순서. 지울 때는 거꾸로)
TABLES = (
    "ai_review_runs",
    "ai_review_run_events",
    "ai_review_units",
    "ai_review_exposures",
    "ai_review_outcomes",
)


def _json() -> sa.types.TypeEngine[Any]:
    """JSON 열 타입 (PostgreSQL JSONB, 그 밖 JSON). tables.Json과 같다."""
    return sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def _create_runs() -> None:
    """실행 세대·이벤트 표."""
    ts = sa.DateTime(timezone=True)
    op.create_table(
        "ai_review_runs",
        sa.Column("run_id", sa.String(length=64), nullable=False),
        sa.Column("session_id", sa.String(length=128), nullable=False),
        sa.Column("version", sa.Text(), nullable=False),
        sa.Column("policy_digest", sa.String(length=12), nullable=False),
        sa.Column("prompt_version", sa.Integer(), nullable=False),
        sa.Column("backend", _json(), nullable=False),
        sa.Column("ontology_version", sa.String(length=32), nullable=False),
        sa.Column("render_hashes", _json(), nullable=False),
        sa.Column("blurred_sha256", _json(), nullable=False),
        sa.Column("sensor_sha256", sa.String(length=64), nullable=False),
        sa.Column("selection", _json(), nullable=False),
        sa.Column("plan_digest", sa.String(length=16), nullable=False),
        sa.Column("planned_units", sa.Integer(), nullable=False),
        sa.Column("requested_by", sa.String(length=128), nullable=True),
        sa.Column("created_at", ts, nullable=False),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["sessions.session_id"],
            name=op.f("fk_ai_review_runs_session_id_sessions"),
        ),
        sa.PrimaryKeyConstraint("run_id", name=op.f("pk_ai_review_runs")),
    )
    op.create_index(op.f("ix_ai_review_runs_session_id"), "ai_review_runs", ["session_id"])
    op.create_table(
        "ai_review_run_events",
        sa.Column(
            "event_id",
            sa.BigInteger().with_variant(sa.Integer(), "sqlite"),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column("session_id", sa.String(length=128), nullable=False),
        sa.Column("run_id", sa.String(length=64), nullable=True),
        sa.Column("event", sa.String(length=16), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("counts", _json(), nullable=False),
        sa.Column("actor", sa.String(length=128), nullable=True),
        sa.Column("at", ts, nullable=False),
        sa.CheckConstraint(
            "event IN ('started', 'completed', 'interrupted', 'aborted', 'waived')",
            name=op.f("ck_ai_review_run_events_event"),
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["ai_review_runs.run_id"],
            name=op.f("fk_ai_review_run_events_run_id_ai_review_runs"),
        ),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["sessions.session_id"],
            name=op.f("fk_ai_review_run_events_session_id_sessions"),
        ),
        sa.PrimaryKeyConstraint("event_id", name=op.f("pk_ai_review_run_events")),
    )
    op.create_index(
        op.f("ix_ai_review_run_events_session_id"), "ai_review_run_events", ["session_id"]
    )


def _create_units() -> None:
    """판정 단위 표 (group은 SQL 예약어라 CHECK 식에서 따옴표로 감싼다)."""
    op.create_table(
        "ai_review_units",
        sa.Column("unit_id", sa.String(length=32), nullable=False),
        sa.Column("run_id", sa.String(length=64), nullable=False),
        sa.Column("session_id", sa.String(length=128), nullable=False),
        sa.Column("version", sa.Text(), nullable=False),
        sa.Column("group", sa.String(length=16), nullable=False),
        sa.Column("stream_id", sa.String(length=128), nullable=False),
        sa.Column("t_start_ms", sa.BigInteger(), nullable=False),
        sa.Column("t_end_ms", sa.BigInteger(), nullable=False),
        sa.Column("target_label_ids", _json(), nullable=False),
        sa.Column("context_label_ids", _json(), nullable=False),
        sa.Column("render_hash", sa.String(length=64), nullable=False),
        sa.Column("blurred_sha256", sa.String(length=64), nullable=True),
        sa.Column("sensor_sha256", sa.String(length=64), nullable=False),
        sa.Column("ontology_version", sa.String(length=32), nullable=False),
        sa.Column("frames", _json(), nullable=False),
        sa.Column("prompt_digest", sa.String(length=64), nullable=False),
        sa.Column("outcome", sa.String(length=16), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("sub_requests", sa.Integer(), nullable=False),
        sa.Column("errors", _json(), nullable=False),
        sa.Column("verdicts", _json(), nullable=False),
        sa.Column("findings", _json(), nullable=False),
        sa.Column("response_sha256", sa.String(length=64), nullable=True),
        sa.Column("served_model", sa.Text(), nullable=True),
        sa.Column("endpoint", sa.String(length=64), nullable=True),
        sa.Column("prompt_tokens", sa.Integer(), nullable=True),
        sa.Column("cached_tokens", sa.Integer(), nullable=True),
        sa.Column("completion_tokens", sa.Integer(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("t_start_ms <= t_end_ms", name=op.f("ck_ai_review_units_time_order")),
        sa.CheckConstraint(
            "\"group\" IN ('temporal', 'temporal_long', 'spatial')",
            name=op.f("ck_ai_review_units_group"),
        ),
        sa.CheckConstraint(
            "outcome IN ('answered', 'fallback')", name=op.f("ck_ai_review_units_outcome")
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["ai_review_runs.run_id"],
            name=op.f("fk_ai_review_units_run_id_ai_review_runs"),
        ),
        sa.PrimaryKeyConstraint("unit_id", name=op.f("pk_ai_review_units")),
    )
    op.create_index(op.f("ix_ai_review_units_created_at"), "ai_review_units", ["created_at"])
    op.create_index(
        "ix_ai_review_units_session_version", "ai_review_units", ["session_id", "version"]
    )


def _create_review_links() -> None:
    """노출·결과 표 (검수 작업 키가 기본 키)."""
    ts = sa.DateTime(timezone=True)
    op.create_table(
        "ai_review_exposures",
        sa.Column("task_key", sa.String(length=128), nullable=False),
        sa.Column("session_id", sa.String(length=128), nullable=False),
        sa.Column("assignment_id", sa.String(length=128), nullable=True),
        sa.Column("arm", sa.String(length=16), nullable=False),
        sa.Column("arm_key", sa.String(length=128), nullable=False),
        sa.Column("versions", _json(), nullable=False),
        sa.Column("render_hash", sa.String(length=64), nullable=True),
        sa.Column("coverage", sa.Float(), nullable=False),
        sa.Column("shown", _json(), nullable=False),
        sa.Column("hidden", _json(), nullable=False),
        sa.Column("created_at", ts, nullable=False),
        sa.CheckConstraint(
            "arm IN ('shown', 'holdout', 'no_verdicts', 'mode_excluded', 'off')",
            name=op.f("ck_ai_review_exposures_arm"),
        ),
        sa.ForeignKeyConstraint(
            ["task_key"],
            ["review_tasks.task_key"],
            name=op.f("fk_ai_review_exposures_task_key_review_tasks"),
        ),
        sa.PrimaryKeyConstraint("task_key", name=op.f("pk_ai_review_exposures")),
    )
    op.create_index(
        op.f("ix_ai_review_exposures_session_id"), "ai_review_exposures", ["session_id"]
    )
    op.create_index(
        op.f("ix_ai_review_exposures_created_at"), "ai_review_exposures", ["created_at"]
    )
    op.create_table(
        "ai_review_outcomes",
        sa.Column("task_key", sa.String(length=128), nullable=False),
        sa.Column("session_id", sa.String(length=128), nullable=False),
        sa.Column("reviewer_id", sa.String(length=128), nullable=True),
        sa.Column("collected_at", ts, nullable=False),
        sa.Column("items", _json(), nullable=False),
        sa.Column("warnings", _json(), nullable=False),
        sa.ForeignKeyConstraint(
            ["task_key"],
            ["review_tasks.task_key"],
            name=op.f("fk_ai_review_outcomes_task_key_review_tasks"),
        ),
        sa.PrimaryKeyConstraint("task_key", name=op.f("pk_ai_review_outcomes")),
    )
    op.create_index(op.f("ix_ai_review_outcomes_session_id"), "ai_review_outcomes", ["session_id"])
    op.create_index(
        op.f("ix_ai_review_outcomes_collected_at"), "ai_review_outcomes", ["collected_at"]
    )


def upgrade() -> None:
    """다섯 표와 색인을 만들고, (PostgreSQL이면) 추가 전용·TRUNCATE 금지 트리거를 건다."""
    _create_runs()
    _create_units()
    _create_review_links()
    # 공용 트리거 함수는 0009(dlp_append_only)·0010(dlp_no_truncate) 소유다
    if op.get_bind().dialect.name == "postgresql":
        for table in TABLES:
            op.execute(
                f"CREATE TRIGGER {table}_append_only BEFORE UPDATE OR DELETE ON {table} "
                "FOR EACH ROW EXECUTE FUNCTION dlp_append_only()"
            )
            op.execute(
                f"CREATE TRIGGER {table}_no_truncate BEFORE TRUNCATE ON {table} "
                "FOR EACH STATEMENT EXECUTE FUNCTION dlp_no_truncate()"
            )


def downgrade() -> None:
    """표를 지운다. AI 기록이 하나라도 있으면 거부한다.

    추가 전용 감사 기록(판정·노출·결과)을 되돌리기로 지우면 품질 측정과 감사 근거가 사라진다. 기록이
    있는 DB를 되돌려야 하면 운영자가 백업한 뒤 직접 정리한다.
    """
    bind = op.get_bind()
    rows = sum(
        int(bind.execute(sa.text(f"SELECT count(*) FROM {table}")).scalar_one()) for table in TABLES
    )
    if rows:
        raise RuntimeError(
            f"AI 1차 검수 기록 {rows}행이 있어 0019를 되돌릴 수 없습니다 "
            "(추가 전용 감사 기록. 백업 뒤 운영자가 직접 정리한다)"
        )
    if bind.dialect.name == "postgresql":
        for table in TABLES:
            op.execute(f"DROP TRIGGER IF EXISTS {table}_no_truncate ON {table}")
            op.execute(f"DROP TRIGGER IF EXISTS {table}_append_only ON {table}")
    op.drop_index(op.f("ix_ai_review_outcomes_collected_at"), table_name="ai_review_outcomes")
    op.drop_index(op.f("ix_ai_review_outcomes_session_id"), table_name="ai_review_outcomes")
    op.drop_table("ai_review_outcomes")
    op.drop_index(op.f("ix_ai_review_exposures_created_at"), table_name="ai_review_exposures")
    op.drop_index(op.f("ix_ai_review_exposures_session_id"), table_name="ai_review_exposures")
    op.drop_table("ai_review_exposures")
    op.drop_index("ix_ai_review_units_session_version", table_name="ai_review_units")
    op.drop_index(op.f("ix_ai_review_units_created_at"), table_name="ai_review_units")
    op.drop_table("ai_review_units")
    op.drop_index(op.f("ix_ai_review_run_events_session_id"), table_name="ai_review_run_events")
    op.drop_table("ai_review_run_events")
    op.drop_index(op.f("ix_ai_review_runs_session_id"), table_name="ai_review_runs")
    op.drop_table("ai_review_runs")
