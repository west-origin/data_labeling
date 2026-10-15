"""판정 진행 상태와 `--pending` 세션 고르기 (WP17.5, ADR 0040 5.2). DB를 읽기만 한다.

"판정됨" = 지금 판정 버전이고 그 스트림의 지금 렌더 해시인 단위에 대상으로 든 적이 있다
(`plan.uncovered_targets`). 맥락 변화는 보지 않는다 (사람 검수가 진행될 때마다 다시 고르지 않게).
영상을 받지 않고 계획 창(`plan_windows`)만으로 센다.

- `pending_sessions`: `dlp aireview run --pending`이 고르는 세션 (C12).
  1. 생애주기 prelabeled, 프라이버시 승인, 승인 보류 아님 (사용 중지 세션은 prelabeled가 아니다).
  2. 조용한 시간: 마지막 라벨 기록(`created_at`)이 `pending.quiet_minutes`보다 오래됐다
     (프리라벨 뒤 relations·actions가 쓰는 중에 판정하지 않게).
  3. 다음 중 하나: 판정 없는 현재 대상이 남았다, 또는 지금 판정 버전의 가장 최근 실행이 completed로
     끝나지 않았다 (`unfinished_run`: 중단·멈춤·마감 없음). 둘째 조건이 없으면 공간 트랙처럼 여러
     창에 드는 대상은 한 창만 판정돼도 "판정됨"이라, 중단된 실행의 남은 창이 다시 골라지지 않는다.
     좁힌 실행(`--groups`, 정책보다 좁은 선택 모드)의 completed는 `partial:`로 마감해 이 판정에서
     건너뛴다 (좁힌 실행이 앞선 전체 실행의 미완료를 덮지 않게).
  4. 순서: 최근 시도가 실패하지 않은 세션 먼저(실패 = 지금 버전 최근 실행이 장애·멈춤·마감 없음으로
     끝남, 시간·Ctrl-C 중단은 실패가 아니다), 그 안에서 prelabeled 전이 시각 순 (같으면 세션 ID).
     늘 실패하는 세션이 앞자리를 차지해 뒤 세션이 판정되지 않는 일을 막는다. 실행 기록 전 입력
     단계에서 실패하는 세션(손상된 장갑 Parquet 등)은 기록이 없어 이 정렬로는 뒤로 가지 않는다:
     CLI가 그런 세션을 `--limit`에 세지 않는다 (`runner.InputError`).
- `session_status`: `dlp aireview status`의 세션 한 줄 (판정률, 단위, fallback, 판정 분포, 마지막
  이벤트, 최근 처리량 기반 남은 시간, 백로그).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import sqlalchemy as sa

from dlp_aireview.plan import PlannedWindow, plan_windows, uncovered_targets
from dlp_aireview.policy import AiReviewPolicy
from dlp_privacy.errors import RenderNotCurrentError
from dlp_privacy.policy import PrivacyPolicy
from dlp_privacy.rendering import expected_render_hash
from dlp_schema.aireview import (
    AiGroup,
    AiReviewUnit,
    AiRunEvent,
    AiRunEventKind,
    is_full_completion,
    valid_verdicts,
)
from dlp_schema.db.aireview_repository import list_ai_run_events, list_ai_runs, list_ai_units
from dlp_schema.db.ontology_repository import get_session_ontology
from dlp_schema.db.repository import (
    active_privacy_hold,
    get_labels,
    get_session,
    list_lifecycle_events,
    list_session_ids,
)
from dlp_schema.episode import current_labels
from dlp_schema.labels import LabelRecord
from dlp_schema.ontology import ontology_digest
from dlp_schema.review import FlaggedSpan
from dlp_schema.session import LifecycleState, PrivacyState, Session

# flagged 모드의 우선순위 사유 구간 계산 (CLI가 검수 운영 정책의 flag_unit으로)
FlaggedFor = Callable[[sa.Connection, Session, list[LabelRecord]], Sequence[FlaggedSpan]]


@dataclass(frozen=True)
class Coverage:
    """세션의 판정 범위 (지금 버전·렌더 기준).

    Attributes:
        windows: 계획 창 (선택 모드·묶음 적용).
        render_hashes: 스트림 → 지금 렌더 해시 (승인 상태가 아니면 비었다).
        approved: 지금 승인 상태라 렌더 해시를 계산했는가.
        units: 지금 버전 단위 중 그 스트림의 지금 렌더 해시인 것.
        targets: 계획 창의 대상 라벨 ID.
        uncovered: 그중 판정되지 않은 라벨 ID.
    """

    windows: tuple[PlannedWindow, ...]
    render_hashes: dict[str, str]
    approved: bool
    units: tuple[AiReviewUnit, ...]
    targets: frozenset[str]
    uncovered: tuple[str, ...]

    @property
    def remaining_windows(self) -> int:
        """판정되지 않은 대상이 든 창 수 (남은 단위 근사)."""
        left = set(self.uncovered)
        return sum(1 for w in self.windows if any(x.label_id in left for x in w.targets))


def coverage(
    conn: sa.Connection,
    session: Session,
    current: Sequence[LabelRecord],
    policy: AiReviewPolicy,
    privacy_policy: PrivacyPolicy,
    *,
    groups: Collection[AiGroup] | None = None,
    flagged_for: FlaggedFor | None = None,
) -> Coverage:
    """세션의 판정 범위 (모듈 docstring의 "판정됨")."""
    sid, version = session.session_id, policy.verdict_version
    o = get_session_ontology(conn, sid)
    flagged = (
        tuple(flagged_for(conn, session, list(current)))
        if flagged_for is not None and policy.selection.mode == "flagged"
        else ()
    )
    windows = tuple(
        w
        for w in plan_windows(session, current, policy, o, flagged=flagged)
        if groups is None or w.group in groups
    )
    try:
        hashes = {
            s: expected_render_hash(conn, sid, s, privacy_policy)
            for s in sorted({w.stream_id for w in windows})
        }
        approved = True
    except RenderNotCurrentError:
        hashes, approved = {}, False
    # 지금 온톨로지 내용으로 판정한 단위만 센다 (F-AIR-32: 같은 버전 draft 덧붙이기)
    digest = ontology_digest(o)
    units = tuple(
        u
        for u in list_ai_units(conn, sid, [version])
        if hashes.get(u.stream_id) == u.render_hash and u.ontology_digest == digest
    )
    left = uncovered_targets(
        windows, units, version=version, render_hashes=hashes, ontology_digest=digest
    )
    targets = frozenset(x.label_id for w in windows for x in w.targets)
    return Coverage(windows, hashes, approved, units, targets, tuple(left))


# ---------------------------------------------------------------- --pending


@dataclass(frozen=True)
class PendingSession:
    """`--pending`이 고른 세션.

    Attributes:
        session_id: 세션. prelabeled_at: prelabeled로 전이한 시각 (기록이 없으면 None).
        uncovered: 판정되지 않은 대상 수. duration_ms: 영상 길이 (백로그 시간).
        last_label_at: 마지막 라벨 기록 시각.
        unfinished: 지금 버전 최근 실행의 마지막 이벤트 (`이벤트:사유`, completed면 None).
    """

    session_id: str
    prelabeled_at: datetime | None
    uncovered: int
    duration_ms: int
    last_label_at: datetime | None
    unfinished: str | None = None

    @property
    def failed(self) -> bool:
        """최근 시도가 실패했는가 (장애·멈춤·마감 없음. 시간·Ctrl-C 중단은 실패가 아니다)."""
        return self.unfinished is not None and self.unfinished not in _NOT_FAILED


# 끝나지 않았지만 실패는 아닌 마감 (다음 실행이 이어서 하면 된다)
_NOT_FAILED = frozenset({"interrupted:max_minutes", "interrupted:signal"})


def unfinished_run(conn: sa.Connection, session_id: str, version: str) -> str | None:
    """이 판정 버전의 가장 최근 실행이 completed로 끝나지 않았으면 그 마지막 이벤트(`이벤트:사유`,
    마감 이벤트가 없으면 started), 끝났거나 실행이 없으면 None.

    좁힌 실행(`--groups` 등, completed:partial 마감)은 건너뛰고 그 앞 실행을 본다: 좁힌 계획이
    끝났어도 앞선 전체 실행이 남긴 다른 묶음 단위는 판정되지 않았다 (공간 트랙처럼 여러 창에 드는
    대상은 판정 범위만으로는 알 수 없다).
    """
    runs = [r for r in list_ai_runs(conn, session_id) if r.version == version]
    events = list_ai_run_events(conn, session_id)
    for run in reversed(runs):
        mine = [e for e in events if e.run_id == run.run_id]
        if not mine:
            return AiRunEventKind.STARTED.value
        last = mine[-1]
        if last.event is AiRunEventKind.COMPLETED and not is_full_completion(last):
            continue  # 좁힌 실행의 완료: 그 앞 실행을 본다
        return None if last.event is AiRunEventKind.COMPLETED else _event_text(last)
    return None


def prelabeled_at(conn: sa.Connection, session_id: str) -> datetime | None:
    """세션이 마지막으로 prelabeled로 전이한 시각 (없으면 None)."""
    times = [
        e.at
        for e in list_lifecycle_events(conn, session_id)
        if e.to_state is LifecycleState.PRELABELED
    ]
    return times[-1] if times else None


def _pending_one(
    conn: sa.Connection,
    session_id: str,
    policy: AiReviewPolicy,
    privacy_policy: PrivacyPolicy,
    now: datetime,
    groups: Collection[AiGroup] | None,
    flagged_for: FlaggedFor | None,
) -> PendingSession | None:
    """세션 하나가 `--pending` 조건을 만족하면 그 요약, 아니면 None."""
    session = get_session(conn, session_id)
    if (
        session.lifecycle_state is not LifecycleState.PRELABELED
        or session.privacy_state is not PrivacyState.APPROVED
        or active_privacy_hold(conn, session_id) is not None
    ):
        return None
    history = get_labels(conn, session_id)
    last = max((x.created_at for x in history), default=None)
    if last is not None and now - last < timedelta(minutes=policy.pending.quiet_minutes):
        return None
    cov = coverage(
        conn, session, current_labels(history), policy, privacy_policy,
        groups=groups, flagged_for=flagged_for,
    )  # fmt: skip
    unfinished = unfinished_run(conn, session_id, policy.verdict_version)
    if not cov.uncovered and unfinished is None:
        return None
    return PendingSession(
        session_id, prelabeled_at(conn, session_id), len(cov.uncovered), session.duration_ms, last,
        unfinished,
    )  # fmt: skip


def pending_sessions(
    conn: sa.Connection,
    policy: AiReviewPolicy,
    privacy_policy: PrivacyPolicy,
    now: datetime,
    *,
    groups: Collection[AiGroup] | None = None,
    flagged_for: FlaggedFor | None = None,
) -> list[PendingSession]:
    """`--pending`이 고르는 세션 (모듈 docstring의 조건과 순서).

    Args:
        now: 지금 시각 (시간대 필수, 조용한 시간 계산).
        groups: 이 묶음만 본다 (`--groups`).
        flagged_for: flagged 모드의 우선순위 사유 구간 계산.
    """
    found = [
        p
        for sid in list_session_ids(conn)
        if (p := _pending_one(conn, sid, policy, privacy_policy, now, groups, flagged_for))
    ]
    far = datetime.max.replace(tzinfo=now.tzinfo)
    return sorted(found, key=lambda p: (p.failed, p.prelabeled_at or far, p.session_id))


# ---------------------------------------------------------------- status


@dataclass(frozen=True)
class SessionStatus:
    """`dlp aireview status`의 세션 한 줄.

    Attributes:
        lifecycle, privacy: 세션 상태 값. held: 승인 보류 중인가.
        version: 지금 판정 버전. approved: 렌더 해시를 계산했는가 (승인 상태).
        targets, covered: 계획 대상 수와 그중 판정된 수.
        units, fallback: 지금 버전·렌더의 단위 수와 그중 fallback 수.
        verdicts: 표시 프로필의 유효한 판정 분포 (라벨마다 가장 나쁜 판정).
        remaining_windows: 판정되지 않은 대상이 든 창 수.
        last_event: 마지막 이벤트 (`이벤트:사유`). last_event_at: 그 시각. last_run_id: 그 실행.
        presumed_interrupted: 마지막 이벤트가 started인데 실행 중이 아니다 (마감 없이 끝남).
        running: 실행기가 잠금을 쥐고 있는가 (모르면 None, SQLite).
        units_per_s: 최근 마감 이벤트의 처리량. eta_s: 남은 창 / 처리량.
        waived: 건너뛰기(waived) 이벤트가 있는가. backlog_hours: 판정할 대상이 남으면 영상 시간.
    """

    session_id: str
    lifecycle: str
    privacy: str
    held: bool
    version: str
    approved: bool
    targets: int
    covered: int
    units: int
    fallback: int
    verdicts: dict[str, int] = field(default_factory=dict[str, int])
    remaining_windows: int = 0
    last_event: str | None = None
    last_event_at: datetime | None = None
    last_run_id: str | None = None
    presumed_interrupted: bool = False
    running: bool | None = None
    units_per_s: float | None = None
    eta_s: float | None = None
    waived: bool = False
    backlog_hours: float = 0.0

    @property
    def coverage(self) -> float | None:
        """판정률 (대상이 없으면 None)."""
        return self.covered / self.targets if self.targets else None


def _rate(events: Sequence[AiRunEvent]) -> float | None:
    """가장 최근 마감 이벤트(완료·중단)의 단위/s (기록한 단위·걸린 초가 있을 때)."""
    for e in reversed(events):
        written, elapsed = e.counts.get("written", 0), e.counts.get("elapsed_s", 0)
        if (
            e.event in (AiRunEventKind.COMPLETED, AiRunEventKind.INTERRUPTED)
            and written
            and elapsed
        ):
            return written / elapsed
    return None


def _event_text(e: AiRunEvent) -> str:
    """`이벤트:사유` (사유가 없으면 이벤트만)."""
    return f"{e.event.value}:{e.reason}" if e.reason else e.event.value


def session_status(
    conn: sa.Connection,
    session_id: str,
    policy: AiReviewPolicy,
    privacy_policy: PrivacyPolicy,
    *,
    groups: Collection[AiGroup] | None = None,
    flagged_for: FlaggedFor | None = None,
    running: bool | None = None,
) -> SessionStatus:
    """세션 하나의 판정 상태 (모듈 docstring).

    Args:
        running: 실행기가 지금 이 세션을 처리 중인가 (CLI가 잠금으로 확인, 모르면 None).
    """
    session = get_session(conn, session_id)
    current = current_labels(get_labels(conn, session_id))
    cov = coverage(
        conn, session, current, policy, privacy_policy, groups=groups, flagged_for=flagged_for
    )
    shown = valid_verdicts(
        cov.units, current, versions=[policy.verdict_version], render_hashes=cov.render_hashes
    )
    events = list_ai_run_events(conn, session_id)
    last = events[-1] if events else None
    rate = _rate(events)
    remaining = cov.remaining_windows
    return SessionStatus(
        session_id=session_id, lifecycle=session.lifecycle_state.value,
        privacy=session.privacy_state.value,
        held=active_privacy_hold(conn, session_id) is not None,
        version=policy.verdict_version, approved=cov.approved, targets=len(cov.targets),
        covered=len(cov.targets) - len(cov.uncovered), units=len(cov.units),
        fallback=sum(u.outcome == "fallback" for u in cov.units),
        verdicts=dict(Counter(v.verdict.value for v, _ in shown.values())),
        remaining_windows=remaining,
        last_event=_event_text(last) if last else None,
        last_event_at=last.at if last else None,
        last_run_id=last.run_id if last else None,
        presumed_interrupted=bool(
            last and last.event is AiRunEventKind.STARTED and running is not True
        ),
        running=running, units_per_s=rate,
        eta_s=remaining / rate if rate and remaining else None,
        waived=any(e.event is AiRunEventKind.WAIVED for e in events),
        backlog_hours=session.duration_ms / 3_600_000 if cov.uncovered else 0.0,
    )  # fmt: skip
