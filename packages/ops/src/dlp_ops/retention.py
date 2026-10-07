"""원본 보관 기간 만료 알림 (`dlp ops retention`).

기간은 config/defaults.yaml retention.raw_retention_days(미정이면 알림을 내지 않는다).
기산점은 블러본·라벨 확정 시각, 곧 사람 검증을 마친 세션의 마지막 사람 검수 변화(승인·수정·추가·
삭제·표본 검증, 블러 포함) 시각이다. 운영 라벨만 본다: 모델 버전 교체로 지운 레코드, 오류 삽입·측정
레코드, 검수 전 모델 라벨은 기산점을 늦추지 않는다 (dlp_schema.history.review_changes).
만료가 alert_days_before 안이거나 지났는데 결정(연장·삭제)이 없으면 알린다.
연장은 기한까지 유효하다.
사용 중지된 세션은 기간과 무관하게 삭제 대상으로 보인다. 삭제 결정을 기록하면 더 알리지 않는다.
원본 삭제 자체는 사람이 결정을 기록한 뒤 운영 절차로 한다.

주의: 검수 도구(CVAT)에 블러 검수용으로 올린 원본 프록시는 이 명령이 찾거나 지우지 않는다.
원본을 지울 때 그 세션의 원본 검수 작업(CVAT 작업·데이터)도 운영 절차에서 함께 지운다
(RETENTION_NOTE).
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
from dlp_schema.history import review_changes
from dlp_schema.labels import VerificationState
from dlp_schema.session import LifecycleState

FINAL = (LifecycleState.HUMAN_VERIFIED, LifecycleState.SPLIT_ASSIGNED, LifecycleState.EXPORTED)
# 검수로 보는 상태 (사람 승인·수정·표본 검증)
REVIEW_STATES = (
    VerificationState.HUMAN_APPROVED,
    VerificationState.HUMAN_CORRECTED,
    VerificationState.SAMPLE_VERIFIED,
)
RETENTION_NOTE = (
    "검수 도구(CVAT)에 올린 원본 프록시는 이 명령이 지우지 않는다: "
    "원본을 지울 때 그 세션의 원본 검수 작업도 함께 지운다"
)
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
    """마지막 사람 검수 변화 시각 (운영 라벨만, 블러 포함)."""
    changes = review_changes(get_labels(conn, session_id), REVIEW_STATES)
    return max((c.at for c in changes), default=None)


def retention_status(
    conn: sa.Connection, today: date, retention_days: int | None, alert_days_before: int
) -> list[RetentionItem]:
    out: list[RetentionItem] = []
    for sid in list_session_ids(conn):
        session = get_session(conn, sid)
        decisions = list_retention_decisions(conn, sid)
        latest = decisions[-1] if decisions else None
        if session.lifecycle_state is LifecycleState.WITHDRAWN:
            # 삭제 결정을 기록했으면 더 알리지 않는다 (결정 확인이 사용 중지보다 먼저)
            if latest is not None and latest.decision == "delete":
                out.append(RetentionItem(sid, "delete_decided", None, None, latest.reason))
            else:
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
