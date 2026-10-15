"""설명의 검수 물려받기 기록 (F-PAR-20, ADR 0043 3절 2026-10-15 개정)

Revision ID: 0021
Revises: 0020
Create Date: 2026-10-15

`description_inheritance`(추가 전용)를 만든다. 설명 라벨이 행동·상위 구간의 검수를 물려받을 때
(`apply_description_follow`) 그 `record_review`와 같은 트랜잭션에 한 행을 쓴다. 이 기록 전에는
"물려받은 검수만 있음"을 검수자·시각 일치로 추정했다. 그래서 행동과 설명을 같은 검수자·시각에 직접
검수하면 물려받은 것으로 잘못 보았고(F-PAR-20), 물려준 행동을 같은 레코드로 다시 승인하면(시각 바뀜)
물려받음을 잃었다(F-PAR-19 한계).

백필 (한 번, 이 리비전을 올릴 때만)
    이미 물려받은 설명은 기록이 없다. 예전 추정을 그대로 한 번 적용해 `mechanism=backfill-heuristic`
    으로 기록한다: 모델 출처 운영 설명이고 상태가 human_approved·sample_verified이며, 같은 세션의
    운영 주인 레코드(설명 segment_id = 행동 action_id·상위 구간 segment_id, 오류 삽입·측정 계보 제외)
    중 검수자가 같고 `max(주인 검수 시각, 설명 생성 시각)`이 설명 검수 시각과 같은 것이 있으면 그중
    가장 늦게 검수한(같으면 라벨 ID 큰) 주인을 준 주인으로 적는다. 추정이므로 예전과 같은 한계가
    남는다(같은 검수자·시각의 직접 검수를 물려받음으로 적음). 출처 값으로 구분해 감사할 수 있다.
    이 계산은 코드 모듈을 import하지 않고 여기에 고정해 둔다 (뒤 코드 변경과 무관하게 같은 결과).
    이 리비전 뒤의 직접 검수는 백필되지 않는다 (`dlp db upgrade`를 다시 해도 이 리비전은 다시 돌지
    않는다).

수정·삭제는 0009의 `dlp_append_only()`, TRUNCATE는 0010의 `dlp_no_truncate()` 트리거로 막는다
(PostgreSQL).
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0021"
down_revision: str | None = "0020"
branch_labels: str | None = None
depends_on: str | None = None

# 대상 테이블 이름
TABLE = "description_inheritance"
# 백필 기록의 출처·규칙 버전 (dlp_schema.review.InheritanceMechanism.BACKFILL, FOLLOW_VERSION)
BACKFILL = "backfill-heuristic"
RULE_VERSION = "review-follow-1"
INHERITED = ("human_approved", "sample_verified")

_labels = sa.table(
    "label_records",
    sa.column("label_id", sa.String()),
    sa.column("session_id", sa.String()),
    sa.column("kind", sa.String()),
    sa.column("source", sa.String()),
    sa.column("verification_state", sa.String()),
    sa.column("reviewer_id", sa.String()),
    sa.column("reviewed_at", sa.DateTime(timezone=True)),
    sa.column("parent_label_id", sa.String()),
    sa.column("seeded_error", sa.Boolean()),
    sa.column("measurement", sa.String()),
    sa.column("created_at", sa.DateTime(timezone=True)),
    sa.column("payload", sa.JSON()),
)


def _utc(t: datetime | None) -> datetime | None:
    """SQLite는 시간대를 저장하지 않는다 (UTC로 복원)."""
    if t is None:
        return None
    return t.replace(tzinfo=UTC) if t.utcoffset() is None else t.astimezone(UTC)


def _payload(row: Any) -> dict[str, Any]:
    p = row.payload
    return json.loads(p) if isinstance(p, str) else dict(p)


def _non_operational(rows: list[Any]) -> set[str]:
    """오류 삽입·측정 레코드와 그 후손 (`dlp_schema.episode.non_operational_ids`와 같은 계산)."""
    out = {r.label_id for r in rows if r.seeded_error or r.measurement is not None}
    children: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        if r.parent_label_id:
            children[r.parent_label_id].append(r.label_id)
    stack = list(out)
    while stack:
        for child in children.get(stack.pop(), []):
            if child not in out:
                out.add(child)
                stack.append(child)
    return out


def _owner_key(row: Any) -> str | None:
    p = _payload(row)
    if row.kind == "action":
        return p.get("action_id")
    if row.kind == "segment":
        return p.get("segment_id")
    return None


def _giver(d: Any, owners: list[Any]) -> Any | None:
    """예전 추정(`inherited_owner`, 0021 전): 검수자가 같고 max(주인 검수, 설명 생성) = 설명 검수."""
    at, created = _utc(d.reviewed_at), _utc(d.created_at)
    assert at is not None and created is not None
    found = [
        o
        for o in owners
        if (given := _utc(o.reviewed_at)) is not None
        and o.reviewer_id == d.reviewer_id
        and max(given, created) == at
    ]
    if not found:
        return None
    return max(found, key=lambda o: (_utc(o.reviewed_at), o.label_id))


def _backfill_rows(rows: list[Any], now: datetime) -> list[dict[str, Any]]:
    """한 세션의 라벨 행(설명·행동·상위 구간)에서 백필할 기록."""
    excluded = _non_operational(rows)
    owners: dict[str, list[Any]] = defaultdict(list)
    for r in rows:
        if r.label_id not in excluded and (k := _owner_key(r)) is not None:
            owners[k].append(r)
    out: list[dict[str, Any]] = []
    for d in rows:
        if (
            d.kind != "description"
            or d.label_id in excluded
            or d.source != "model"
            or d.verification_state not in INHERITED
            or d.reviewer_id is None
            or d.reviewed_at is None
        ):
            continue
        segment = _payload(d).get("segment_id")
        giver = _giver(d, owners.get(str(segment), [])) if segment is not None else None
        if giver is None:
            continue
        out.append(
            {
                "session_id": d.session_id,
                "description_label_id": d.label_id,
                "owner_label_ids": [giver.label_id],
                "giver_label_id": giver.label_id,
                "state": d.verification_state,
                "reviewer_id": d.reviewer_id,
                "reviewed_at": _utc(d.reviewed_at),
                "mechanism": BACKFILL,
                "rule_version": RULE_VERSION,
                "recorded_at": now,
            }
        )
    return out


def _backfill(bind: sa.Connection, table: sa.Table) -> None:
    """이 리비전 전에 물려받은 설명을 예전 추정으로 한 번 기록한다 (모듈 docstring)."""
    sessions = bind.execute(
        sa.select(_labels.c.session_id)
        .where(
            _labels.c.kind == "description",
            _labels.c.source == "model",
            _labels.c.verification_state.in_(INHERITED),
        )
        .distinct()
    ).scalars()
    now = datetime.now(UTC)
    for sid in sorted(sessions):
        rows = list(
            bind.execute(
                sa.select(_labels)
                .where(
                    _labels.c.session_id == sid,
                    _labels.c.kind.in_(("description", "action", "segment")),
                )
                .order_by(_labels.c.created_at, _labels.c.label_id)
            )
        )
        found = _backfill_rows(rows, now)
        if found:
            bind.execute(table.insert(), found)


def upgrade() -> None:
    """기록 테이블과 추가 전용·TRUNCATE 금지 트리거를 만들고 예전 물려받음을 한 번 메운다."""
    table = op.create_table(
        TABLE,
        sa.Column(
            "event_id",
            sa.BigInteger().with_variant(sa.Integer(), "sqlite"),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column("session_id", sa.String(length=128), nullable=False),
        sa.Column("description_label_id", sa.String(length=128), nullable=False),
        sa.Column(
            "owner_label_ids",
            sa.JSON().with_variant(postgresql.JSONB(), "postgresql"),
            nullable=False,
        ),
        sa.Column("giver_label_id", sa.String(length=128), nullable=False),
        sa.Column("state", sa.String(length=32), nullable=False),
        sa.Column("reviewer_id", sa.String(length=128), nullable=False),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("mechanism", sa.String(length=32), nullable=False),
        sa.Column("rule_version", sa.String(length=64), nullable=False),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["description_label_id"],
            ["label_records.label_id"],
            name=op.f("fk_description_inheritance_description_label_id_label_records"),
        ),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["sessions.session_id"],
            name=op.f("fk_description_inheritance_session_id_sessions"),
        ),
        sa.PrimaryKeyConstraint("event_id", name=op.f("pk_description_inheritance")),
    )
    assert table is not None
    op.create_index(
        op.f("ix_description_inheritance_description_label_id"),
        TABLE,
        ["description_label_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_description_inheritance_session_id"), TABLE, ["session_id"], unique=False
    )
    _backfill(op.get_bind(), table)
    if op.get_bind().dialect.name == "postgresql":
        op.execute(
            f"CREATE TRIGGER {TABLE}_append_only BEFORE UPDATE OR DELETE ON {TABLE} "
            "FOR EACH ROW EXECUTE FUNCTION dlp_append_only()"
        )
        op.execute(
            f"CREATE TRIGGER {TABLE}_no_truncate BEFORE TRUNCATE ON {TABLE} "
            "FOR EACH STATEMENT EXECUTE FUNCTION dlp_no_truncate()"
        )


def downgrade() -> None:
    """기록 테이블을 지운다. 이관 백필 외의 기록이 하나라도 있으면 거부한다.

    그 기록을 지우면 물려받은 설명이 직접 검수한 것처럼 보호돼(규칙 6이 지우지 못함) 다음 검수
    회차에서 낡은 설명이 승인된 채 남고, 물려받음 감사 기록도 잃는다. 공용 트리거 함수는 이전
    리비전 소유라 지우지 않는다.
    """
    count = (
        op.get_bind()
        .execute(sa.text(f"SELECT count(*) FROM {TABLE} WHERE mechanism <> :m"), {"m": BACKFILL})
        .scalar_one()
    )
    if count:
        raise RuntimeError(
            f"설명 물려받기 기록 {count}개가 있어 0021을 되돌릴 수 없습니다 "
            "(물려받은 설명의 규칙 6 판정과 감사 기록을 잃는다)"
        )
    op.drop_index(op.f("ix_description_inheritance_session_id"), table_name=TABLE)
    op.drop_index(op.f("ix_description_inheritance_description_label_id"), table_name=TABLE)
    op.drop_table(TABLE)
