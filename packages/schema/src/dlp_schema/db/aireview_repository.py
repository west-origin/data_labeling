"""AI 1차 검수 기록 저장소 (WP17.4, ADR 0040). 연결·트랜잭션·커밋은 호출자가 관리한다.

판정은 라벨이 아니다. 이 모듈은 `label_records`를 읽거나 쓰지 않고, AI 표(`ai_review_*`, Alembic
0019)만 다룬다. 다섯 표 모두 추가만 한다 (PostgreSQL 트리거가 UPDATE·DELETE·TRUNCATE를 막는다). 그래서
이 모듈에는 수정·삭제 함수가 없다.

절별 함수
    - 실행 세대·이벤트: `insert_ai_run`, `get_ai_run`, `list_ai_runs`, `insert_ai_run_event`,
      `list_ai_run_events`, `latest_run_state`(세션별 마지막 이벤트·완료 버전·건너뜀).
    - 판정 단위: `insert_ai_units`(같은 단위 ID는 무시: PostgreSQL·SQLite 모두
      `ON CONFLICT DO NOTHING`), `done_unit_ids`(재개·멱등), `list_ai_units`, `get_ai_unit`.
    - 노출·결과: `insert_ai_exposure`, `get_ai_exposure`, `list_ai_exposures`, `insert_ai_outcome`,
      `get_ai_outcome`, `list_ai_outcomes`.

누가 쓰나
    - 실행기(`dlp_aireview.runner`, WP17.5)가 실행·이벤트·단위를 쓴다. 판정 버전마다 단위가 남고, 표시할
      버전은 읽는 쪽이 고른다 (`dlp_schema.aireview.valid_verdicts`).
    - 검수 연동(`dlp_review`, WP17.6~17.7)이 노출·결과를 쓰고 단위를 읽는다. 운영 지표(WP17.8)가
      노출·결과를 기간으로 읽는다.
    - 내보내기·학습·데이터셋·액티브 러닝은 이 모듈을 import하지 않는다 (정적 검사, I4).

JSON 열에는 계약의 `model_dump(mode="json")` 값을, 시각 열에는 시간대 있는 datetime 객체를 넣는다.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql, sqlite

from dlp_schema.aireview import (
    AiExposure,
    AiOutcome,
    AiReviewRun,
    AiReviewUnit,
    AiRunEvent,
    AiRunEventKind,
)
from dlp_schema.common import Contract
from dlp_schema.db.tables import (
    ai_review_exposures,
    ai_review_outcomes,
    ai_review_run_events,
    ai_review_runs,
    ai_review_units,
)


def _aware(at: datetime) -> datetime:
    """시간대 없는 시각이면 ValueError (시각 열은 TIMESTAMP WITH TIME ZONE)."""
    if at.tzinfo is None or at.utcoffset() is None:
        raise ValueError(f"시간대 없는 시각은 쓰지 않습니다: {at!r}")
    return at


def _row(obj: Contract, times: Iterable[str]) -> dict[str, Any]:
    """계약 → 행 값 (JSON 직렬화, 시각 열은 datetime 객체 그대로)."""
    data = obj.model_dump(mode="json")
    for name in times:
        data[name] = getattr(obj, name)
    return data


def _plain(row: Mapping[str, Any]) -> dict[str, Any]:
    """행 → 계약 검증용 사전."""
    return dict(row)


# ---------------------------------------------------------------- 실행 세대·이벤트


def insert_ai_run(conn: sa.Connection, run: AiReviewRun) -> None:
    """실행 세대 1행을 추가한다. 같은 run_id가 있으면 IntegrityError (세션은 sessions에 있어야 한다)."""
    conn.execute(ai_review_runs.insert().values(**_row(run, ("created_at",))))


def get_ai_run(conn: sa.Connection, run_id: str) -> AiReviewRun:
    """실행 세대 하나. Raises: sqlalchemy.exc.NoResultFound (없을 때)."""
    row = (
        conn.execute(sa.select(ai_review_runs).where(ai_review_runs.c.run_id == run_id))
        .mappings()
        .one()
    )
    return AiReviewRun.model_validate(_plain(row))


def list_ai_runs(conn: sa.Connection, session_id: str) -> list[AiReviewRun]:
    """세션의 실행 세대 (만든 순, 같은 시각이면 run_id 순)."""
    rows = conn.execute(
        sa.select(ai_review_runs)
        .where(ai_review_runs.c.session_id == session_id)
        .order_by(ai_review_runs.c.created_at, ai_review_runs.c.run_id)
    ).mappings()
    return [AiReviewRun.model_validate(_plain(r)) for r in rows]


def insert_ai_run_event(conn: sa.Connection, event: AiRunEvent) -> int:
    """실행 이벤트 1행을 추가하고 DB가 매긴 event_id를 돌려준다.

    `event.event_id`는 무시한다 (자동 증가). run_id가 있으면 그 실행이 먼저 있어야 한다 (FK).
    """
    data = _row(event, ("at",))
    data.pop("event_id")
    result = conn.execute(
        ai_review_run_events.insert().values(**data).returning(ai_review_run_events.c.event_id)
    )
    return int(result.scalar_one())


def list_ai_run_events(conn: sa.Connection, session_id: str) -> list[AiRunEvent]:
    """세션의 실행 이벤트 (시각 순, 같으면 event_id 순)."""
    rows = conn.execute(
        sa.select(ai_review_run_events)
        .where(ai_review_run_events.c.session_id == session_id)
        .order_by(ai_review_run_events.c.at, ai_review_run_events.c.event_id)
    ).mappings()
    return [AiRunEvent.model_validate(_plain(r)) for r in rows]


@dataclass(frozen=True)
class AiRunState:
    """세션 하나의 AI 판정 실행 상태 (`latest_run_state`, `dlp aireview status`와 required 게이트용).

    Attributes:
        session_id: 세션.
        last_event: 가장 최근 이벤트 (시각, event_id 순).
        last_run: 가장 최근 실행 세대 (만든 순). 실행 없이 건너뛰기만 있으면 None.
        completed_versions: completed 이벤트가 있는 실행들의 판정 버전.
        waived_at: 가장 최근 waived 이벤트 시각 (없으면 None).
    """

    session_id: str
    last_event: AiRunEvent
    last_run: AiReviewRun | None
    completed_versions: frozenset[str]
    waived_at: datetime | None

    @property
    def open(self) -> bool:
        """마지막 이벤트가 started다 (실행 중이거나, 마감 이벤트 없이 끝나 중단된 것으로 추정)."""
        return self.last_event.event is AiRunEventKind.STARTED


def latest_run_state(conn: sa.Connection, session_ids: Iterable[str]) -> dict[str, AiRunState]:
    """세션별 실행 상태. 이벤트가 하나도 없는 세션은 결과에 없다.

    Args:
        session_ids: 세션 ID (중복 무시).
    """
    sids = sorted(set(session_ids))
    if not sids:
        return {}
    events: dict[str, list[AiRunEvent]] = {}
    rows = conn.execute(
        sa.select(ai_review_run_events)
        .where(ai_review_run_events.c.session_id.in_(sids))
        .order_by(ai_review_run_events.c.at, ai_review_run_events.c.event_id)
    ).mappings()
    for r in rows:
        e = AiRunEvent.model_validate(_plain(r))
        events.setdefault(e.session_id, []).append(e)
    runs: dict[str, list[AiReviewRun]] = {}
    run_rows = conn.execute(
        sa.select(ai_review_runs)
        .where(ai_review_runs.c.session_id.in_(sids))
        .order_by(ai_review_runs.c.created_at, ai_review_runs.c.run_id)
    ).mappings()
    for r in run_rows:
        run = AiReviewRun.model_validate(_plain(r))
        runs.setdefault(run.session_id, []).append(run)
    return {sid: _state(sid, evs, runs.get(sid, [])) for sid, evs in events.items()}


def _state(sid: str, events: list[AiRunEvent], runs: list[AiReviewRun]) -> AiRunState:
    """한 세션의 이벤트(시각 순)·실행(만든 순) → 상태."""
    version = {r.run_id: r.version for r in runs}
    completed = frozenset(
        version[e.run_id]
        for e in events
        if e.event is AiRunEventKind.COMPLETED and e.run_id in version
    )
    waived = [e.at for e in events if e.event is AiRunEventKind.WAIVED]
    return AiRunState(
        session_id=sid,
        last_event=events[-1],
        last_run=runs[-1] if runs else None,
        completed_versions=completed,
        waived_at=waived[-1] if waived else None,
    )


# ---------------------------------------------------------------- 판정 단위


def _insert_ignore(conn: sa.Connection) -> Any:
    """방언별 `INSERT … ON CONFLICT (unit_id) DO NOTHING` 문 (PostgreSQL·SQLite)."""
    if conn.dialect.name == "postgresql":
        return postgresql.insert(ai_review_units).on_conflict_do_nothing(
            index_elements=["unit_id"]
        )
    if conn.dialect.name == "sqlite":
        return sqlite.insert(ai_review_units).on_conflict_do_nothing(index_elements=["unit_id"])
    raise NotImplementedError(f"지원하지 않는 DB 방언: {conn.dialect.name}")


def insert_ai_units(conn: sa.Connection, units: Iterable[AiReviewUnit]) -> int:
    """판정 단위를 추가한다. 이미 있는 unit_id는 건드리지 않고 건너뛴다 (I8).

    같은 단위 ID는 같은 입력(버전·대상·프레임·렌더 해시…)이라 먼저 쓴 행을 남긴다. 실행기 둘이
    잠금을 뚫고 같은 단위를 써도 중복 행이 생기지 않는다. 실행(run_id)이 먼저 있어야 한다 (FK).

    Returns:
        새로 추가한 행 수.
    """
    rows = [_row(u, ("created_at",)) for u in units]
    if not rows:
        return 0
    stmt = _insert_ignore(conn).returning(ai_review_units.c.unit_id)
    return len(conn.execute(stmt, rows).all())


def done_unit_ids(conn: sa.Connection, session_id: str, version: str) -> set[str]:
    """세션·판정 버전의 이미 기록한 단위 ID (재실행은 이 단위들을 서버에 다시 묻지 않는다)."""
    rows = conn.execute(
        sa.select(ai_review_units.c.unit_id).where(
            ai_review_units.c.session_id == session_id, ai_review_units.c.version == version
        )
    )
    return {str(r) for r in rows.scalars()}


def list_ai_units(
    conn: sa.Connection, session_id: str, versions: Collection[str] | None = None
) -> list[AiReviewUnit]:
    """세션의 판정 단위 (만든 순, 같으면 unit_id 순).

    Args:
        versions: 이 판정 버전들만 (None이면 모두, 빈 목록이면 결과도 빈 목록).
    """
    query = sa.select(ai_review_units).where(ai_review_units.c.session_id == session_id)
    if versions is not None:
        if not versions:
            return []
        query = query.where(ai_review_units.c.version.in_(sorted(versions)))
    rows = conn.execute(
        query.order_by(ai_review_units.c.created_at, ai_review_units.c.unit_id)
    ).mappings()
    return [AiReviewUnit.model_validate(_plain(r)) for r in rows]


def get_ai_unit(conn: sa.Connection, unit_id: str) -> AiReviewUnit | None:
    """판정 단위 하나 (없으면 None)."""
    row = (
        conn.execute(sa.select(ai_review_units).where(ai_review_units.c.unit_id == unit_id))
        .mappings()
        .one_or_none()
    )
    return None if row is None else AiReviewUnit.model_validate(_plain(row))


# ---------------------------------------------------------------- 노출·결과


def insert_ai_exposure(conn: sa.Connection, exposure: AiExposure) -> None:
    """작업 노출 1행 (작업 생성과 같은 트랜잭션). 같은 task_key가 있으면 IntegrityError."""
    conn.execute(ai_review_exposures.insert().values(**_row(exposure, ("created_at",))))


def get_ai_exposure(conn: sa.Connection, task_key: str) -> AiExposure | None:
    """작업 노출 (없으면 None: AI 단계 전에 만든 작업 등)."""
    row = (
        conn.execute(
            sa.select(ai_review_exposures).where(ai_review_exposures.c.task_key == task_key)
        )
        .mappings()
        .one_or_none()
    )
    return None if row is None else AiExposure.model_validate(_plain(row))


def list_ai_exposures(conn: sa.Connection, since: datetime, until: datetime) -> list[AiExposure]:
    """[since, until) 동안 만든 작업 노출 (만든 순, 같으면 작업 키 순).

    Raises:
        ValueError: 시간대 없는 시각.
    """
    c = ai_review_exposures.c
    rows = conn.execute(
        sa.select(ai_review_exposures)
        .where(c.created_at >= _aware(since), c.created_at < _aware(until))
        .order_by(c.created_at, c.task_key)
    ).mappings()
    return [AiExposure.model_validate(_plain(r)) for r in rows]


def insert_ai_outcome(conn: sa.Connection, outcome: AiOutcome) -> None:
    """작업 결과 1행 (수집 트랜잭션). 같은 task_key가 있으면 IntegrityError (작업마다 한 번)."""
    conn.execute(ai_review_outcomes.insert().values(**_row(outcome, ("collected_at",))))


def get_ai_outcome(conn: sa.Connection, task_key: str) -> AiOutcome | None:
    """작업 결과 (없으면 None)."""
    row = (
        conn.execute(sa.select(ai_review_outcomes).where(ai_review_outcomes.c.task_key == task_key))
        .mappings()
        .one_or_none()
    )
    return None if row is None else AiOutcome.model_validate(_plain(row))


def list_ai_outcomes(conn: sa.Connection, since: datetime, until: datetime) -> list[AiOutcome]:
    """[since, until) 동안 수집한 작업 결과 (수집 순, 같으면 작업 키 순).

    Raises:
        ValueError: 시간대 없는 시각.
    """
    c = ai_review_outcomes.c
    rows = conn.execute(
        sa.select(ai_review_outcomes)
        .where(c.collected_at >= _aware(since), c.collected_at < _aware(until))
        .order_by(c.collected_at, c.task_key)
    ).mappings()
    return [AiOutcome.model_validate(_plain(r)) for r in rows]
