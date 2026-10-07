"""원본 보관 기간 만료 알림 (`dlp ops retention`, WP16, ADR 0020).

기간은 config/defaults.yaml retention.raw_retention_days(미정이면 알림을 내지 않는다). 기산점은
블러본·라벨 확정 시각, 곧 사람 검증을 마친 세션의 마지막 사람 검수 변화(승인·수정·추가·
삭제·표본 검증, 블러 포함) 시각이다. 운영 라벨만 본다: 모델 버전 교체로 지운 레코드, 오류
삽입·측정 레코드, 검수 전 모델 라벨은 기산점을 늦추지 않는다 (dlp_schema.history.review_changes).
만료가 alert_days_before 안이거나 지났는데 결정(연장·삭제)이 없으면 알린다. 연장은 기한까지
유효하다. 사용 중지된 세션은 기간과 무관하게 삭제 대상으로 보인다. 삭제 결정을 기록하면 더 알리지
않는다. 원본 삭제 자체는 사람이 결정을 기록한 뒤 운영 절차로 한다.

주의: 검수 도구(CVAT)에 블러 검수용으로 올린 원본 프록시는 이 명령이 찾거나 지우지 않는다.
원본을 지울 때 그 세션의 원본 검수 작업(CVAT 작업·데이터)도 운영 절차에서 함께 지운다
(RETENTION_NOTE).

상태 값(`Status`):
- `ok` — 만료까지 `alert_days_before`일보다 많이 남음.
- `due_soon` — 만료가 `alert_days_before`일 안 (알림).
- `expired` — 만료일이 지남 (알림).
- `extended` — 가장 최근 결정이 아직 유효한 연장 (기한 ≥ 오늘).
- `delete_decided` — 가장 최근 결정이 삭제 (더 알리지 않음).
- `withdrawn` — 사용 중지 세션이며 삭제 결정이 아직 없음 (알림).

공개: `retention_status(conn, today, retention_days, alert_days_before)`, `finalized_at`,
`RetentionItem`, `RETENTION_NOTE`. 모두 읽기 전용이다.
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

# 보관 기간을 세기 시작하는 생애주기 상태 (사람 검증을 마친 뒤)
FINAL = (LifecycleState.HUMAN_VERIFIED, LifecycleState.SPLIT_ASSIGNED, LifecycleState.EXPORTED)
# 검수로 보는 상태 (사람 승인·수정·표본 검증)
REVIEW_STATES = (
    VerificationState.HUMAN_APPROVED,
    VerificationState.HUMAN_CORRECTED,
    VerificationState.SAMPLE_VERIFIED,
)
# 만료·사용 중지·삭제 결정 세션이 있을 때 `dlp ops retention`이 덧붙이는 운영 절차 안내
RETENTION_NOTE = (
    "검수 도구(CVAT)에 올린 원본 프록시는 이 명령이 지우지 않는다: "
    "원본을 지울 때 그 세션의 원본 검수 작업도 함께 지운다"
)
# 보관 상태 (의미는 모듈 docstring의 "상태 값")
Status = Literal["ok", "due_soon", "expired", "extended", "delete_decided", "withdrawn"]


@dataclass(frozen=True)
class RetentionItem:
    """세션 하나의 보관 상태.

    필드:
        session_id: 세션 ID.
        status: 위 `Status` 중 하나.
        finalized_at: 기산점(마지막 사람 검수 변화 시각, 시간대 있음). 사용 중지 세션은 None.
        expires_on: 만료일(기산점 날짜 + 보관 일수, 지난 연장 기한이 더 늦으면 그 날). 연장 중이면
            연장 기한.
        note: 출력에 붙일 메모 (결정 이유, 사용 중지 안내).
    """

    session_id: str
    status: Status
    finalized_at: datetime | None
    expires_on: date | None
    note: str = ""

    @property
    def alert(self) -> bool:
        """사람에게 알려야 하는 상태인지 (`due_soon`, `expired`, `withdrawn`)."""
        return self.status in ("due_soon", "expired", "withdrawn")


def finalized_at(conn: sa.Connection, session_id: str) -> datetime | None:
    """마지막 사람 검수 변화 시각 (운영 라벨만, 블러 포함).

    반환: 사람 검수 변화가 하나도 없으면 None (그 세션은 아직 기산점이 없어 알림 대상이 아니다).
    """
    changes = review_changes(get_labels(conn, session_id), REVIEW_STATES)
    return max((c.at for c in changes), default=None)


def retention_status(
    conn: sa.Connection, today: date, retention_days: int | None, alert_days_before: int
) -> list[RetentionItem]:
    """모든 세션의 보관 상태를 계산한다. 읽기 전용.

    인자:
        conn: DB 연결 (`sessions`, `label_records`, `retention_decisions`를 읽는다).
        today: 기준일 (보통 오늘 UTC 날짜).
        retention_days: 보관 일수. None(미정)이면 사용 중지 세션만 보고한다.
        alert_days_before: 만료 며칠 전부터 `due_soon`으로 볼지.

    판정 순서 (세션마다, 결정은 가장 최근 것 하나만 본다):
    1. 사용 중지 세션: 삭제 결정이 있으면 `delete_decided`, 없으면 `withdrawn`.
    2. 보관 일수가 미정이거나 생애주기가 검수 완료 이후(`FINAL`)가 아니면 건너뛴다.
    3. 기산점이 없으면 건너뛴다.
    4. 삭제 결정 → `delete_decided`. 기한이 오늘 이후인 연장 → `extended`.
    5. 지난 연장 기한이 원래 만료일보다 늦으면 그 날을 만료일로 삼고 `expired`/`due_soon`/`ok`.

    반환: `RetentionItem` 목록 (세션 ID 순서는 `list_session_ids`를 따른다). 건너뛴 세션은 들어가지
    않는다.
    """
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
        # 기산점의 날짜(기록된 시간대, 보통 UTC) + 보관 일수
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
        # 기한이 지난 연장: 원래 만료일과 연장 기한 중 늦은 날을 만료일로 본다
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
