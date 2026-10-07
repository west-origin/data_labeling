"""원본 보관 기간 만료 알림 (`dlp ops retention`).

기간은 config/defaults.yaml retention.raw_retention_days(미정이면 알림을 내지 않는다).
기산점은 블러본·라벨 확정 시각, 곧 사람 검증을 마친 세션의 마지막 라벨 변경(작성·검수) 시각이다.
만료가 alert_days_before 안이거나 지났는데 결정(연장·삭제)이 없으면 알린다.
연장은 기한까지 유효하다.
사용 중지된 세션은 기간과 무관하게 삭제 대상으로 보인다. 원본 삭제 자체는 사람이 결정을 기록한 뒤
운영 절차로 한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Literal

import sqlalchemy as sa

from dlp_schema.db.repository import (
    get_labels,
    get_session,
    list_retention_decisions,
    list_session_ids,
)
from dlp_schema.session import LifecycleState

FINAL = (LifecycleState.HUMAN_VERIFIED, LifecycleState.SPLIT_ASSIGNED, LifecycleState.EXPORTED)
Status = Literal["ok", "due_soon", "expired", "extended", "delete_decided", "withdrawn"]


@dataclass(frozen=True)
class RetentionItem:
    session_id: str
    status: Status
    finalized_at: datetime | None
    expires_on: date | None
    note: str = ""

    @property
    def alert(self) -> bool:
        return self.status in ("due_soon", "expired", "withdrawn")


def finalized_at(conn: sa.Connection, session_id: str) -> datetime | None:
    labels = get_labels(conn, session_id)
    times = [x.verification.reviewed_at or x.created_at for x in labels]
    return max(times, default=None)


def retention_status(
    conn: sa.Connection, today: date, retention_days: int | None, alert_days_before: int
) -> list[RetentionItem]:
    out: list[RetentionItem] = []
    for sid in list_session_ids(conn):
        session = get_session(conn, sid)
        decisions = list_retention_decisions(conn, sid)
        latest = decisions[-1] if decisions else None
        if session.lifecycle_state is LifecycleState.WITHDRAWN:
            out.append(
                RetentionItem(sid, "withdrawn", None, None, "사용 중지: 원본 삭제 절차 확인")
            )
            continue
        if retention_days is None or session.lifecycle_state not in FINAL:
            continue
        done = finalized_at(conn, sid)
        if done is None:
            continue
        expires = done.date() + timedelta(days=retention_days)
        if latest is not None and latest.decision == "delete":
            out.append(RetentionItem(sid, "delete_decided", done, expires, latest.reason))
            continue
        if latest is not None and latest.until is not None and latest.until >= today:
            out.append(
                RetentionItem(
                    sid, "extended", done, latest.until, f"{latest.until}까지: {latest.reason}"
                )
            )
            continue
        if latest is not None and latest.until is not None:
            expires = max(expires, latest.until)
        if expires < today:
            status: Status = "expired"
        elif expires <= today + timedelta(days=alert_days_before):
            status = "due_soon"
        else:
            status = "ok"
        out.append(RetentionItem(sid, status, done, expires))
    return out
