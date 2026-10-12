"""판정 실행기: 계획 → 생산자·작업자·기록자 → 마감 (WP17.5, ADR 0040 5.1·5.3·5.4).

```
[계획]   세션·현재 운영 라벨·온톨로지 (짧은 읽기 트랜잭션) → 계획 창
         → 판정할 창이 있는 스트림만 `open_blurred` → `plan_session`
         → 기록된 단위(`done_unit_ids`) 제외 → 실행 기록 + started 이벤트
[생산자] 스트림마다 스레드 하나: 고른 프레임을 순차 디코드하며 단위의 프레임이 다 모이면
         표식·조각·JPEG를 만들어 제한 큐(작업자 수의 2배, 역압)에 넣는다. 다 쓴 프레임은
         바로 버린다.
[작업자] `options.workers`개 스레드: `judge_unit`(재시도·부분 수용·단위 안 문맥 초과 분할).
[기록자] 스레드 하나: `commit_every`개 또는 `commit_interval_s`마다 단위를 쓴다.
         `render_check_interval_s`마다 렌더 해시를 다시 확인해 바뀌면 멈춘다.
         `progress_interval_s`마다 진행 한 줄.
[마감]   completed(계획 단위가 모두 기록됨) | interrupted(일부) | aborted(설정·렌더 변경·버그)
```

- VLM 호출 중에는 DB 트랜잭션을 열지 않는다. 기록자는 자기 연결로 짧게 쓴다 (`DbSink`).
- 같은 세션에는 실행기 하나만 돈다: 실행 내내 `session_lock`(세션 수준 advisory lock, 전용
  연결)을 쥔다. 잠금이 뚫려도 단위 기록은 `ON CONFLICT DO NOTHING`이라 중복 행이 생기지 않는다.
- 재개·멱등 (I8): 단위 키가 판정 입력(대상·맥락 ID·프레임·렌더·센서·온톨로지)의 해시라, 기록된
  단위는 다시 묻지 않는다 (맥락 라벨의 검수 상태는 키 밖, `plan` 모듈 설명). 같은 버전·입력이면
  서버 호출이 0번이다. 판정은 라벨이 아니라 `retractions()`를 쓰지 않는다.
- 장애 분류 (5.3)
  - 모든 엔드포인트가 오래 막힘(`AllEndpointsDownError`): 새 요청을 멈추고 진행 중 요청을 끝낸 뒤
    기록자를 비우고 interrupted:unavailable (종료 3).
  - 요청 하나의 재시도가 끝나도 장애: 그 단위만 기록하지 않고 넘어간다. 끝에
    interrupted:unavailable이고 다음 실행이 다시 묻는다. 장애를 unsure로 기록하지 않는다.
  - 설정 오류(4xx, 서버 모델 불일치): aborted:config (종료 5).
  - 렌더 변경: aborted:render_changed (종료 6). 이미 쓴 단위는 표시 프로필에서 빠진다.
  - 실행 중 세션 사용 중지·승인 보류: aborted:session_not_ready (종료 6, 주기 확인).
  - `options.deadline`(--max-minutes): interrupted:max_minutes (종료 8).
  - `options.cancel`(CLI 공유 신호, 여러 세션을 함께 돌리다 Ctrl-C) 또는 이 스레드의 Ctrl-C:
    interrupted:signal (종료 130).
  - 버그(예외): aborted:error (종료 1). 어느 스레드의 예외든 멈춤 사유를 남기고 멈춘다 (스레드가
    조용히 죽어 다른 스레드가 영영 기다리지 않게).
  시간·Ctrl-C로 멈춰도 계획 단위가 모두 기록됐으면 completed다. 멈춘 뒤에는 새 요청을 보내지 않는다:
  엔드포인트 묶음에서 보낼 곳을 기다리거나 백오프하던 요청도 멈춤 신호를 보고 그만둔다
  (`CancellableBackend`). 이미 보낸 요청은 끝까지 받아 기록한다.
- 응답 원문은 DB에 쓰지 않는다 (sha256만). `options.debug_dir`가 있으면 형식이 맞지 않던 응답만
  로컬 파일로 남긴다 (쓰지 못하면 경고하고 그만 남긴다).

`UnitSink`(기록 대상) 구현은 `DbSink`(AI 표)와 `MemorySink`(테스트) 두 가지다. 프레임 공급
(`RunInput.frames`)은 `BlurredFrame`만 낸다 (실제 경로는 `media.frames_for`, 블러본 경계 I1).
이 모듈은 라벨·검수 기록을 쓰지 않는다 (정적 검사, I3).
"""

from __future__ import annotations

import hashlib
import json
import queue
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import sqlalchemy as sa
from numpy.typing import NDArray
from sqlalchemy.exc import NoResultFound

from dlp_aireview.backend import (
    AllEndpointsDownError,
    CancellableBackend,
    JudgeBackend,
    JudgeCancelledError,
    JudgeReply,
    JudgeRequest,
)
from dlp_aireview.judge import JudgedUnit, judge_unit
from dlp_aireview.media import (
    BlurredFrame,
    MediaBoundaryError,
    check_blurred,
    frames_for,
    open_blurred,
    render_hashes_now,
)
from dlp_aireview.overlay import render_unit_images
from dlp_aireview.plan import (
    PlannedUnit,
    PlannedWindow,
    StreamMedia,
    plan_session,
    plan_windows,
)
from dlp_aireview.policy import AiReviewPolicy
from dlp_aireview.prompt import PromptImage
from dlp_aireview.sensors import SensorContext
from dlp_media.storage import ObjectStore
from dlp_models.vlm import VlmRequestError, VlmUnavailableError
from dlp_privacy.errors import RenderNotCurrentError
from dlp_privacy.policy import PrivacyPolicy
from dlp_schema.aireview import (
    PARTIAL_COMPLETION,
    AiBackendRecord,
    AiGroup,
    AiReviewRun,
    AiReviewUnit,
    AiRunEvent,
    AiRunEventKind,
    AiSelectionRecord,
    is_full_completion,
)
from dlp_schema.db.aireview_repository import (
    AiRunState,
    done_unit_ids,
    insert_ai_run,
    insert_ai_run_event,
    insert_ai_units,
    latest_run_state,
)
from dlp_schema.db.locks import try_lock_aireview, unlock_aireview
from dlp_schema.db.ontology_repository import get_session_ontology
from dlp_schema.db.repository import active_privacy_hold, get_labels, get_session
from dlp_schema.episode import current_labels
from dlp_schema.labels import HandStatePayload, LabelRecord
from dlp_schema.ontology import Ontology
from dlp_schema.review import FlaggedSpan
from dlp_schema.session import LifecycleState, PrivacyState, Session

# 프레임 공급: (스트림, 프레임 번호들) → 승인된 블러본 프레임 (번호 오름차순)
FrameSource = Callable[[str, Sequence[int]], Iterable[BlurredFrame]]
# 렌더 재확인: 스트림 → 지금 렌더 해시 (승인 상태가 아니면 RenderNotCurrentError, 세션이 사용
# 중지·승인 보류면 SessionNotReadyError)
RenderCheck = Callable[[], Mapping[str, str]]
Status = Literal["dry_run", "nothing_to_do", "completed", "interrupted", "aborted"]
# 사유 → (마감 이벤트, 종료 코드)
_REASONS: dict[str, tuple[AiRunEventKind, int]] = {
    "unavailable": (AiRunEventKind.INTERRUPTED, 3),
    "max_minutes": (AiRunEventKind.INTERRUPTED, 8),
    "signal": (AiRunEventKind.INTERRUPTED, 130),
    "config": (AiRunEventKind.ABORTED, 5),
    "render_changed": (AiRunEventKind.ABORTED, 6),
    "session_not_ready": (AiRunEventKind.ABORTED, 6),
    "error": (AiRunEventKind.ABORTED, 1),
}
# 시간·Ctrl-C 멈춤: 계획 단위가 모두 기록됐으면 완료로 본다. 더 심각한 사유가 오면 그것으로 바꾼다
_SOFT = frozenset({"max_minutes", "signal"})
# 큐에서 기다릴 때 멈춤 신호를 보는 간격 (초)
_POLL_S = 0.1


class SessionNotReadyError(RuntimeError):
    """세션 전제 실패: 사용 중지, 승인 보류, 프라이버시 승인 전, 장갑 Parquet 없음 (종료 6)."""


class InputError(RuntimeError):
    """판정 입력을 만들다 예상하지 못한 오류가 났다 (실행 기록 전: 아무것도 쓰지 않았다).

    손상된 장갑 Parquet·블러본, 저장소 오류처럼 매번 같은 오류가 날 수 있다. 실행 기록·이벤트가 없어
    `--pending`의 실패 정렬이 작동하지 않으므로, CLI는 이 세션을 `--limit` 자리로 세지 않는다
    (늘 실패하는 세션이 뒤 세션을 막지 않게). 원래 예외는 `__cause__`에 있다.
    """


# ---------------------------------------------------------------- 입력·설정·결과


@dataclass(frozen=True)
class RunInput:
    """판정할 세션 하나의 입력 (계획에 쓰는 값과 프레임 공급).

    Attributes:
        session: 세션. current: 현재 운영 라벨 (`current_labels`). ontology: 세션 온톨로지.
        media: 스트림 → 블러본 PTS 인덱스·렌더 해시 (판정에 필요한 스트림만).
        blurred_sha256: 스트림 → 렌더 기록의 블러본 sha (이전 형식이면 None, 감사용).
        frames: 프레임 공급 (`BlurredFrame`만).
        sensor: 장갑 접촉 구간 (없으면 `SensorContext.absent()`).
        flagged: flagged 모드의 우선순위 사유 구간 (CLI가 `flag_unit`으로 계산).
        targets: 대상 지정 (골든 평가 `evaluate`만, None이면 `plan.targets(current)`).
    """

    session: Session
    current: tuple[LabelRecord, ...]
    ontology: Ontology
    media: Mapping[str, StreamMedia]
    blurred_sha256: Mapping[str, str | None]
    frames: FrameSource
    sensor: SensorContext = field(default_factory=SensorContext.absent)
    flagged: tuple[FlaggedSpan, ...] = ()
    targets: tuple[LabelRecord, ...] | None = None


@dataclass(frozen=True)
class RunOptions:
    """실행 설정 (CLI 인자).

    Attributes:
        workers: 작업자 스레드 수 (엔드포인트 수 * `server.concurrency_per_endpoint`).
        groups: 이 묶음만 판정한다 (None이면 전부, `--groups`).
        deadline: 이 시각(`clock` 기준)이 지나면 새 요청을 멈춘다 (`--max-minutes`).
        dry_run: 계획만 세고 서버에 묻거나 쓰지 않는다.
        debug_dir: 형식이 맞지 않던 응답 원문을 남길 로컬 디렉터리 (DB에는 쓰지 않는다).
        actor: 실행자 (`DLP_ACTOR`, 실행 기록 requested_by·이벤트 actor).
        same_model_as_actions: 판정 모델이 행동 분류 VLM과 같은가 (실행 기록에 남긴다).
        json_progress: 진행 줄을 JSON으로 낸다.
        endpoint_states: 엔드포인트 이름 → 상태 (진행 줄, `EndpointPool.states`).
        cancel: 바깥 멈춤 신호 (CLI가 여러 세션에 함께 준다). 켜지면 interrupted:signal로 멈춘다
            (여러 세션을 스레드로 돌릴 때 Ctrl-C는 주 스레드에만 오므로 이 신호로 알린다).
        narrowed: 정책 기본 계획보다 좁힌 실행의 설명 (예: `groups=temporal`, CLI가 정한다). 있으면
            completed 마감 사유를 `partial:<설명>`으로 남긴다 (`dlp_schema.aireview.
            PARTIAL_COMPLETION`: `--pending`·required 게이트는 이 완료를 전체 완료로 보지 않는다).
    """

    workers: int = 1
    groups: frozenset[AiGroup] | None = None
    deadline: float | None = None
    dry_run: bool = False
    debug_dir: Path | None = None
    actor: str | None = None
    same_model_as_actions: bool = False
    json_progress: bool = False
    endpoint_states: Callable[[], Mapping[str, str]] | None = None
    cancel: threading.Event | None = None
    narrowed: str | None = None


@dataclass(frozen=True)
class RunSummary:
    """세션 하나의 실행 결과 (`to_json`으로 출력한다).

    Attributes:
        status: dry_run | nothing_to_do | completed | interrupted | aborted.
        reason: interrupted면 unavailable·max_minutes·signal, aborted면 config·render_changed·
            session_not_ready·error.
        detail: 운영자에게 보일 설명 (DB에 쓰지 않는다: 서버 오류 본문·주소가 섞일 수 있다).
        planned: 계획 단위 수. todo: 그중 기록되지 않았던 단위 수.
        written: 이번에 판정해 기록 대상에 넘긴 단위 수. inserted: 그중 실제로 새로 들어간 행 수
            (잠금이 뚫려 다른 실행이 먼저 쓴 단위는 `ON CONFLICT DO NOTHING`으로 빠져 written보다
            작다).
        skipped: 장애로 이번에 기록하지 못한 단위 수. fallback: 형식에 맞는 응답이 없던 단위 수.
        groups: 묶음 → 계획 단위 수. verdicts: 이번 판정 분포. elapsed_s: 걸린 초.
    """

    session_id: str
    version: str
    status: Status
    reason: str | None = None
    detail: str | None = None
    run_id: str | None = None
    planned: int = 0
    todo: int = 0
    written: int = 0
    inserted: int = 0
    skipped: int = 0
    fallback: int = 0
    groups: dict[str, int] = field(default_factory=dict[str, int])
    verdicts: dict[str, int] = field(default_factory=dict[str, int])
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    elapsed_s: float = 0.0

    @property
    def exit_code(self) -> int:
        """종료 코드: 0 완료·할 일 없음·계획만 / 3 장애 / 5 설정 / 6 렌더 변경·세션 사용 중지·보류 /
        8 시간 / 130 Ctrl-C / 1 버그."""
        if self.reason is None:
            return 0
        return _REASONS[self.reason][1]

    def to_json(self) -> dict[str, Any]:
        """JSON 출력용 사전 (설명 포함)."""
        return {
            "session_id": self.session_id, "version": self.version, "status": self.status,
            "reason": self.reason, "detail": self.detail, "run_id": self.run_id,
            "planned": self.planned, "todo": self.todo, "written": self.written,
            "inserted": self.inserted, "skipped": self.skipped, "fallback": self.fallback,
            "groups": self.groups,
            "verdicts": self.verdicts, "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens, "completion_tokens": self.completion_tokens,
            "elapsed_s": round(self.elapsed_s, 3), "exit_code": self.exit_code,
        }  # fmt: skip


# ---------------------------------------------------------------- 기록 대상


class UnitSink(Protocol):
    """판정 기록 대상 (`DbSink`, `MemorySink`). 여러 스레드가 부를 수 있다."""

    def done_unit_ids(self, session_id: str, version: str) -> set[str]:
        """이미 기록한 단위 ID."""
        ...

    def is_complete(self, session_id: str, version: str, plan_digest: str, *, full: bool) -> bool:
        """가장 최근 실행이 이 버전·이 계획으로 completed인가 (같은 계획의 재실행은 아무것도
        쓰지 않는다). full(좁히지 않은 실행)이면 좁힌 실행의 `partial:` 마감은 완료가 아니다
        (`_completed`)."""
        ...

    def start(self, run: AiReviewRun, event: AiRunEvent) -> None:
        """실행 기록과 started 이벤트를 쓴다 (한 트랜잭션)."""
        ...

    def write(self, units: Sequence[AiReviewUnit]) -> int:
        """단위를 쓴다 (이미 있는 단위 ID는 건너뛴다). 새로 쓴 수."""
        ...

    def event(self, event: AiRunEvent) -> None:
        """실행 이벤트를 쓴다."""
        ...


def _completed(state: AiRunState | None, version: str, plan_digest: str, *, full: bool) -> bool:
    """가장 최근 이벤트가 가장 최근 실행의 completed이고 그 실행이 이 버전·계획인가.

    full(좁히지 않은 실행)이면 그 completed가 전체 완료여야 한다 (`is_full_completion`). 좁힌
    실행(`--groups`·`--mode flagged`)의 계획이 전체 계획과 같은 단위 집합이어도 그 마감은
    `partial:`이라 완료 버전(`latest_run_state.completed_versions`)에 들지 않으므로, 뒤이은 전체
    실행이 할 일 없음으로 끝나면 전체 완료 기록이 영영 생기지 않는다 (required 게이트가
    `max_wait_h` 뒤 waived:timeout으로 열린다). 그때는 새 단위 없이 실행·마감만 남긴다.
    """
    if state is None or state.last_run is None:
        return False
    run, last = state.last_run, state.last_event
    finished = is_full_completion(last) if full else last.event is AiRunEventKind.COMPLETED
    return (
        finished
        and last.run_id == run.run_id
        and run.version == version
        and run.plan_digest == plan_digest
    )


class DbSink:
    """AI 표에 쓰는 기록 대상. 호출마다 짧은 트랜잭션을 연다 (VLM 호출 중에 열어 두지 않는다)."""

    def __init__(self, engine: sa.Engine) -> None:
        """engine: DB 엔진 (기록자 스레드가 자기 연결을 꺼내 쓴다)."""
        self.engine = engine

    def done_unit_ids(self, session_id: str, version: str) -> set[str]:
        """`aireview_repository.done_unit_ids`."""
        with self.engine.connect() as conn:
            return done_unit_ids(conn, session_id, version)

    def is_complete(self, session_id: str, version: str, plan_digest: str, *, full: bool) -> bool:
        """`UnitSink.is_complete` (실행 상태 `latest_run_state`로 본다)."""
        with self.engine.connect() as conn:
            state = latest_run_state(conn, [session_id]).get(session_id)
        return _completed(state, version, plan_digest, full=full)

    def start(self, run: AiReviewRun, event: AiRunEvent) -> None:
        """실행 기록 + started 이벤트 (한 트랜잭션)."""
        with self.engine.begin() as conn:
            insert_ai_run(conn, run)
            insert_ai_run_event(conn, event)

    def write(self, units: Sequence[AiReviewUnit]) -> int:
        """`insert_ai_units` (ON CONFLICT DO NOTHING)."""
        with self.engine.begin() as conn:
            return insert_ai_units(conn, units)

    def event(self, event: AiRunEvent) -> None:
        """`insert_ai_run_event`."""
        with self.engine.begin() as conn:
            insert_ai_run_event(conn, event)


class MemorySink:
    """메모리 기록 대상 (테스트). DB 저장소와 같은 규칙: 같은 단위 ID는 처음 것을 남긴다.

    Attributes:
        runs, events: 쓴 실행·이벤트 (순서대로). units: 단위 ID → 단위. writes: write 호출 수.
    """

    def __init__(self) -> None:
        """빈 기록."""
        self.runs: list[AiReviewRun] = []
        self.events: list[AiRunEvent] = []
        self.units: dict[str, AiReviewUnit] = {}
        self.writes = 0
        self._lock = threading.Lock()

    def done_unit_ids(self, session_id: str, version: str) -> set[str]:
        """세션·버전의 기록된 단위 ID."""
        with self._lock:
            return {
                u.unit_id
                for u in self.units.values()
                if u.session_id == session_id and u.version == version
            }

    def is_complete(self, session_id: str, version: str, plan_digest: str, *, full: bool) -> bool:
        """`UnitSink.is_complete`."""
        with self._lock:
            events = [e for e in self.events if e.session_id == session_id]
            runs = [r for r in self.runs if r.session_id == session_id]
        if not events or not runs:
            return False
        state = AiRunState(session_id, events[-1], runs[-1], frozenset(), None)
        return _completed(state, version, plan_digest, full=full)

    def start(self, run: AiReviewRun, event: AiRunEvent) -> None:
        """실행과 started 이벤트를 남긴다."""
        with self._lock:
            self.runs.append(run)
            self.events.append(event)

    def write(self, units: Sequence[AiReviewUnit]) -> int:
        """새 단위만 남긴다."""
        with self._lock:
            self.writes += 1
            new = [u for u in units if u.unit_id not in self.units]
            self.units.update({u.unit_id: u for u in new})
            return len(new)

    def event(self, event: AiRunEvent) -> None:
        """이벤트를 남긴다."""
        with self._lock:
            self.events.append(event)


# ---------------------------------------------------------------- 계획 도우미


def plan_digest(units: Iterable[PlannedUnit]) -> str:
    """계획한 단위 ID 집합의 해시 16자 (실행 기록 plan_digest)."""
    ids = sorted(u.unit_id for u in units)
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]


def run_id_for(version: str, session_id: str, digest: str, started_at: datetime) -> str:
    """실행 ID `air-run-<sha256(버전|세션|계획 해시|시작 시각) 24자>`."""
    text = f"{version}|{session_id}|{digest}|{started_at.isoformat()}"
    return "air-run-" + hashlib.sha256(text.encode()).hexdigest()[:24]


def planned_units(inp: RunInput, policy: AiReviewPolicy, options: RunOptions) -> list[PlannedUnit]:
    """입력의 판정 단위 (`plan_session`, `options.groups` 묶음만)."""
    units = plan_session(
        inp.session, list(inp.current), policy, inp.ontology,
        media=inp.media, sensor=inp.sensor, flagged=inp.flagged, only=inp.targets,
    )  # fmt: skip
    return [u for u in units if options.groups is None or u.group in options.groups]


def _passes(units: Sequence[PlannedUnit]) -> list[list[PlannedUnit]]:
    """처리 순서 묶음: 지적·미상 단위 → 그 밖 (flagged 모드의 감사 단위). 빈 묶음은 뺀다."""
    first = [u for u in units if u.window.selection in ("flagged", "unknown_gap")]
    rest = [u for u in units if u.window.selection not in ("flagged", "unknown_gap")]
    return [b for b in (first, rest) if b]


# ---------------------------------------------------------------- 실행 상태


@dataclass(frozen=True)
class _Job:
    """작업자에게 넘길 단위 하나 (보낼 이미지까지 만든 것)."""

    unit: PlannedUnit
    images: tuple[PromptImage, ...]


@dataclass
class _Stats:
    """기록자가 모으는 통계 (기록자 스레드만 바꾼다, 진행 줄은 잠금 안에서 읽는다)."""

    written: int = 0
    fallback: int = 0
    verdicts: Counter[str] = field(default_factory=Counter[str])
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    latencies: list[float] = field(default_factory=list[float])

    def add(self, judged: JudgedUnit, seconds: float) -> None:
        """판정 단위 하나를 센다."""
        self.written += 1
        self.fallback += int(judged.outcome == "fallback")
        self.verdicts.update(v.verdict.value for v in judged.verdicts)
        self.prompt_tokens += judged.prompt_tokens or 0
        self.cached_tokens += judged.cached_tokens or 0
        self.completion_tokens += judged.completion_tokens or 0
        self.latencies.append(seconds)


class _Recorder:
    """디버그용 감싸기: 단위 하나의 응답 원문을 모은다 (`options.debug_dir`)."""

    def __init__(self, inner: JudgeBackend) -> None:
        """inner: 실제 백엔드."""
        self.inner = inner
        self.info = inner.info
        self.replies: list[JudgeReply] = []

    def judge(self, request: JudgeRequest) -> JudgeReply:
        """그대로 보내고 응답을 남긴다."""
        reply = self.inner.judge(request)
        self.replies.append(reply)
        return reply


class _Execution:
    """한 실행의 스레드들과 공유 상태 (모듈 docstring의 생산자·작업자·기록자)."""

    def __init__(
        self,
        inp: RunInput,
        backend: JudgeBackend,
        sink: UnitSink,
        policy: AiReviewPolicy,
        options: RunOptions,
        render_check: RenderCheck | None,
        clock: Callable[[], float],
        now: Callable[[], datetime],
        progress: Callable[[str], None] | None,
    ) -> None:
        """공유 상태를 만든다 (스레드는 `run`이 띄운다).

        작업자는 백엔드를 `CancellableBackend`(이 실행의 멈춤 신호 `stop`)로 감싸 쓴다: 멈춘 뒤에는
        엔드포인트 묶음에서 기다리던 요청도 보내지 않는다.
        """
        self.stop = threading.Event()
        self.inp, self.sink, self.policy = inp, sink, policy
        self.backend: JudgeBackend = CancellableBackend(backend, self.stop)
        self.options, self.render_check = options, render_check
        self.clock, self.now, self.progress = clock, now, progress
        self.jobs: queue.Queue[_Job | None] = queue.Queue(maxsize=max(2, 2 * options.workers))
        self.results: queue.Queue[tuple[JudgedUnit, float] | None] = queue.Queue()
        self.stats = _Stats()
        self.inserted = 0
        self.skipped = 0
        self.debug_failed = False
        self.reason: str | None = None
        self.detail: str | None = None
        self.error: BaseException | None = None
        self._lock = threading.Lock()
        self.run_id = ""
        self.total = 0
        self.already = 0
        self.started = clock()

    def halt(
        self, reason: str, detail: str | None = None, error: BaseException | None = None
    ) -> None:
        """멈춤 사유를 남기고 새 작업을 멈춘다. 처음 사유를 남기되, 시간·Ctrl-C 사유(`_SOFT`) 뒤에
        더 심각한 사유(설정·장애·렌더·세션·버그)가 오면 그것으로 바꾼다."""
        with self._lock:
            if self.reason is None or (self.reason in _SOFT and reason not in _SOFT):
                self.reason, self.detail, self.error = reason, detail, error
        self.stop.set()

    def poll(self) -> None:
        """바깥 멈춤 신호(`options.cancel`)와 마감 시각(`options.deadline`)을 본다. 작업자가 다음
        작업을 받기 전에, 기록자가 `_POLL_S`마다 부른다 (작업자가 모두 엔드포인트를 기다리고 있어도
        멈춘다)."""
        cancel, deadline = self.options.cancel, self.options.deadline
        if cancel is not None and cancel.is_set():
            self.halt("signal")
        if deadline is not None and self.clock() >= deadline:
            self.halt("max_minutes")

    # ------------------------------------------------------------ 생산자

    def _put(self, job: _Job | None) -> bool:
        """작업 큐에 넣는다 (가득 차면 기다린다). 멈췄으면 False."""
        while not self.stop.is_set():
            try:
                self.jobs.put(job, timeout=_POLL_S)
                return True
            except queue.Full:
                continue
        return False

    def produce(self, stream: str, units: Sequence[PlannedUnit]) -> None:
        """스트림 하나의 생산자: 처리 순서 묶음마다 순차 디코드하며 단위를 만든다."""
        try:
            for batch in _passes(units):
                if not self._produce_batch(stream, batch):
                    return
        except BaseException as exc:  # 디코드·그리기 실패는 버그나 영상 손상이다
            self.halt("error", f"{stream}: {type(exc).__name__}: {exc}", exc)

    def _emit(self, unit: PlannedUnit, cache: Mapping[int, NDArray[np.uint8]]) -> bool:
        """단위의 프레임으로 보낼 이미지를 만들어 큐에 넣는다."""
        images = render_unit_images(unit, [cache[i] for i in unit.frame_idx], self.policy)
        return self._put(_Job(unit, images))

    def _produce_batch(self, stream: str, batch: Sequence[PlannedUnit]) -> bool:
        """한 묶음: 단위를 마지막 프레임 번호 순으로 내보낸다 (프레임이 다 모이면 바로)."""
        order = sorted(range(len(batch)), key=lambda k: (max(batch[k].frame_idx, default=-1), k))
        uses = Counter(i for u in batch for i in u.frame_idx)
        cache: dict[int, NDArray[np.uint8]] = {}
        k = 0
        while k < len(order) and not batch[order[k]].frame_idx:  # 프레임 없는 단위 (방어)
            if not self._emit(batch[order[k]], cache):
                return False
            k += 1
        for frame in self.inp.frames(stream, sorted(uses)):
            if self.stop.is_set():
                return False
            cache[frame.index] = frame.rgb
            while k < len(order) and max(batch[order[k]].frame_idx) <= frame.index:
                unit = batch[order[k]]
                k += 1
                if not self._emit(unit, cache):
                    return False
                for i in unit.frame_idx:
                    uses[i] -= 1
                    if uses[i] == 0:
                        cache.pop(i, None)
        return True

    # ------------------------------------------------------------ 작업자

    def _next(self) -> _Job | None:
        """다음 작업 (멈췄거나 마감 시각이 지났거나 바깥 멈춤 신호가 켜졌거나 끝이면 None)."""
        while not self.stop.is_set():
            self.poll()
            if self.stop.is_set():
                return None
            try:
                return self.jobs.get(timeout=_POLL_S)
            except queue.Empty:
                continue
        return None

    def work(self) -> None:
        """작업자: 단위를 판정해 기록자에게 넘긴다 (모듈 docstring의 장애 분류).

        어떤 예외도 멈춤 사유 없이 이 스레드를 끝내지 않는다 (작업자가 모두 조용히 죽으면 생산자가
        가득 찬 큐 앞에서 영영 기다리고 실행이 세션 잠금을 쥔 채 멈춘다).
        """
        try:
            while (job := self._next()) is not None:
                if not self._judge(job):
                    return
        except BaseException as exc:
            self.halt("error", f"{type(exc).__name__}: {exc}", exc)

    def _judge(self, job: _Job) -> bool:
        """단위 하나를 판정해 기록자에게 넘긴다. 이 작업자가 그만둬야 하면 False."""
        start = self.clock()
        rec = _Recorder(self.backend) if self.options.debug_dir else None
        try:
            judged = judge_unit(
                rec or self.backend, job.unit, job.images, ontology=self.inp.ontology,
                policy=self.policy, sensor=self.inp.sensor,
            )  # fmt: skip
        except JudgeCancelledError:
            return False  # 이미 멈췄다 (사유는 멈춘 쪽이 남겼다). 이 단위는 다음 실행이 묻는다
        except AllEndpointsDownError as exc:
            self.halt("unavailable", str(exc))
            return False
        except VlmRequestError as exc:
            self.halt("config", str(exc))
            return False
        except VlmUnavailableError as exc:
            with self._lock:
                self.skipped += 1
                self.detail = self.detail or str(exc)
            return True
        self.results.put((judged, self.clock() - start))
        if rec is not None:
            self._debug(judged, rec)
        return True

    def _debug(self, judged: JudgedUnit, rec: _Recorder) -> None:
        """형식이 맞지 않던 응답이 있으면 원문을 로컬 파일로 남긴다 (DB에는 쓰지 않는다).

        디버그 기록은 판정 결과가 아니므로, 쓰지 못하면(디렉터리가 파일, 권한, 디스크) 한 번
        경고하고 이 실행에서는 더 남기지 않는다 (실행은 계속한다).
        """
        folder = self.options.debug_dir
        if folder is None or self.debug_failed:
            return
        if not (judged.outcome == "fallback" or judged.invalid_responses):
            return
        path = folder / self.inp.session.session_id / f"{judged.unit.unit_id}.json"
        data = {
            "unit_id": judged.unit.unit_id, "outcome": judged.outcome,
            "errors": list(judged.errors),
            "responses": [{"endpoint": r.endpoint, "content": r.content} for r in rec.replies],
        }  # fmt: skip
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as exc:
            with self._lock:
                first, self.debug_failed = not self.debug_failed, True
            if first and self.progress is not None:
                self.progress(f"[경고] 디버그 응답을 남기지 못해 더 남기지 않습니다: {exc}")

    # ------------------------------------------------------------ 기록자

    def write_loop(self) -> None:
        """기록자: 판정을 모아 쓰고 렌더를 다시 확인하고 진행을 알린다."""
        server = self.policy.server
        buf: list[AiReviewUnit] = []
        last = {"commit": self.clock(), "check": self.clock(), "progress": self.clock()}
        try:
            while True:
                try:
                    item = self.results.get(timeout=_POLL_S)
                except queue.Empty:
                    pass
                else:
                    if item is None:
                        break
                    buf.append(self._record(*item))
                self.poll()
                now = self.clock()
                if len(buf) >= server.commit_every or (
                    buf and now - last["commit"] >= server.commit_interval_s
                ):
                    self._flush(buf)
                    last["commit"] = now
                if now - last["check"] >= server.render_check_interval_s:
                    self.check_render()
                    last["check"] = now
                if now - last["progress"] >= server.progress_interval_s:
                    self.report()
                    last["progress"] = now
            self._flush(buf)
        except BaseException as exc:
            self.halt("error", f"기록 실패: {type(exc).__name__}: {exc}", exc)

    def _record(self, judged: JudgedUnit, seconds: float) -> AiReviewUnit:
        """판정 → 기록 행 (통계도 센다)."""
        with self._lock:
            self.stats.add(judged, seconds)
        stream = judged.unit.stream_id
        return judged.record(self.run_id, self.now(), self.inp.blurred_sha256.get(stream))

    def _flush(self, buf: list[AiReviewUnit]) -> None:
        """모은 단위를 쓰고 비운다 (새로 들어간 행 수를 센다)."""
        if buf:
            self.inserted += self.sink.write(list(buf))
            buf.clear()

    def check_render(self) -> bool:
        """렌더 해시가 계획 때와 같고 세션이 판정할 수 있는 상태인가. 렌더가 바뀌었거나 승인이
        풀렸으면 render_changed, 세션이 사용 중지·승인 보류면 session_not_ready로 멈추고 False."""
        if self.render_check is None:
            return True
        try:
            now = self.render_check()
        except SessionNotReadyError as exc:
            self.halt("session_not_ready", str(exc))
            return False
        except RenderNotCurrentError as exc:
            self.halt("render_changed", str(exc))
            return False
        changed = sorted(s for s, m in self.inp.media.items() if now.get(s) != m.render_hash)
        if changed:
            self.halt("render_changed", f"블러가 바뀐 스트림: {', '.join(changed)}")
            return False
        return True

    # ------------------------------------------------------------ 진행

    def snapshot(self) -> dict[str, Any]:
        """진행 값 (진행 줄·JSON)."""
        with self._lock:
            s = self.stats
            lat = sorted(s.latencies)
            written, skipped = s.written, self.skipped
            tokens = (s.prompt_tokens, s.cached_tokens, s.completion_tokens)
            fallback = s.fallback
        elapsed = max(self.clock() - self.started, 1e-9)
        rate = written / elapsed
        remaining = self.total - self.already - written - skipped
        return {
            "session_id": self.inp.session.session_id, "done": self.already + written,
            "planned": self.total, "units_per_s": rate,
            "p50_s": float(np.percentile(lat, 50)) if lat else None,
            "p95_s": float(np.percentile(lat, 95)) if lat else None,
            "prefill_tok_s": (tokens[0] - tokens[1]) / elapsed, "decode_tok_s": tokens[2] / elapsed,
            "cache_ratio": tokens[1] / tokens[0] if tokens[0] else None,
            "fallback_ratio": fallback / written if written else None,
            "endpoints": dict(states()) if (states := self.options.endpoint_states) else {},
            "eta_s": remaining / rate if rate > 0 else None, "skipped": skipped,
        }  # fmt: skip

    def report(self) -> None:
        """진행 한 줄 (JSON Lines 또는 글)."""
        if self.progress is None:
            return
        snap = self.snapshot()
        if self.options.json_progress:
            self.progress(json.dumps(snap, ensure_ascii=False))
        else:
            self.progress(progress_line(snap))

    # ------------------------------------------------------------ 실행

    def run(self, todo: Sequence[PlannedUnit]) -> None:
        """스레드를 띄우고 끝나기를 기다린다 (Ctrl-C면 진행 중 요청을 끝낸 뒤 올린다)."""
        streams: dict[str, list[PlannedUnit]] = {}
        for u in todo:
            streams.setdefault(u.stream_id, []).append(u)
        writer = start_thread(self.write_loop)
        workers = [start_thread(self.work) for _ in range(self.options.workers)]
        producers = [start_thread(self.produce, s, us) for s, us in streams.items()]
        try:
            wait_all(producers)
            for _ in workers:
                self._put(None)
            wait_all(workers)
        except KeyboardInterrupt:
            self.halt("signal")
            wait_all(workers)
            raise
        finally:
            self.results.put(None)
            wait_all([writer])


def start_thread(target: Callable[..., None], *args: Any) -> threading.Event:
    """데몬 스레드를 띄우고, 그 스레드가 끝나면 켜지는 신호를 돌려준다 (`wait_all`로 기다린다).

    `Thread.join`으로 기다리지 않는 이유: Ctrl-C(KeyboardInterrupt)가 join을 끊으면 CPython이 아직
    살아 있는 스레드를 끝난 것으로 표시할 수 있다 (`is_alive()`가 False). 그러면 Ctrl-C 뒤 다시
    기다릴 때 기다리지 않고 지나가, 세션이 마감 이벤트를 쓰기 전에 명령이 끝난다.
    """
    finished = threading.Event()

    def body() -> None:
        try:
            target(*args)
        finally:
            finished.set()

    threading.Thread(target=body, daemon=True).start()
    return finished


def wait_all(finished: Sequence[threading.Event]) -> None:
    """`start_thread`로 띄운 스레드가 모두 끝나기를 기다린다 (Ctrl-C를 받을 수 있게 짧게 나눠
    기다린다)."""
    for done in finished:
        while not done.wait(timeout=0.5):
            pass


def progress_line(snap: Mapping[str, Any]) -> str:
    """진행 한 줄 (ADR 0040 5.4 모양)."""
    done, planned = snap["done"], snap["planned"]
    pct = f"{done / planned:.0%}" if planned else "-"
    parts = [f"[aireview] {snap['session_id']} {done}/{planned} 단위({pct})"]
    parts.append(f"{snap['units_per_s']:.2f} 단위/s")
    if snap["p50_s"] is not None:
        parts.append(f"p50 {snap['p50_s']:.1f}s p95 {snap['p95_s']:.1f}s")
    parts.append(
        f"프리필 {snap['prefill_tok_s']:.0f} tok/s · 생성 {snap['decode_tok_s']:.0f} tok/s"
    )
    if snap["cache_ratio"] is not None:
        parts.append(f"캐시 {snap['cache_ratio']:.0%}")
    if snap["fallback_ratio"] is not None:
        parts.append(f"fallback {snap['fallback_ratio']:.1%}")
    if snap["endpoints"]:
        parts.append(", ".join(f"{k} {v}" for k, v in snap["endpoints"].items()))
    if snap["skipped"]:
        parts.append(f"장애로 건너뜀 {snap['skipped']}")
    if snap["eta_s"] is not None:
        parts.append(f"남은 약 {max(1, round(snap['eta_s'] / 60))}분")
    return " · ".join(parts)


# ---------------------------------------------------------------- 실행 진입점


def _counts(summary: RunSummary) -> dict[str, int]:
    """마감 이벤트의 정수 집계."""
    return {
        "planned": summary.planned, "todo": summary.todo, "written": summary.written,
        "inserted": summary.inserted, "skipped": summary.skipped, "fallback": summary.fallback,
        **{f"verdict_{k}": v for k, v in sorted(summary.verdicts.items())},
        "prompt_tokens": summary.prompt_tokens, "cached_tokens": summary.cached_tokens,
        "completion_tokens": summary.completion_tokens, "elapsed_s": round(summary.elapsed_s),
    }  # fmt: skip


def _run_record(
    inp: RunInput,
    backend: JudgeBackend,
    policy: AiReviewPolicy,
    options: RunOptions,
    units: Sequence[PlannedUnit],
    started_at: datetime,
) -> AiReviewRun:
    """실행 기록 (계획을 고정한 기록)."""
    digest = plan_digest(units)
    version = policy.verdict_version
    sel = policy.selection
    streams = sorted(inp.media)
    return AiReviewRun(
        run_id=run_id_for(version, inp.session.session_id, digest, started_at),
        session_id=inp.session.session_id, version=version, policy_digest=policy.digest,
        prompt_version=policy.prompt_version,
        backend=AiBackendRecord(
            kind=policy.backend.kind, model=policy.backend.model,
            served_model=policy.served_model, fingerprint=policy.backend.fingerprint,
            endpoints=backend.info.endpoints, same_model_as_actions=options.same_model_as_actions,
        ),
        ontology_version=inp.ontology.version,
        render_hashes={s: inp.media[s].render_hash for s in streams},
        blurred_sha256={s: inp.blurred_sha256.get(s) for s in streams},
        sensor_sha256=inp.sensor.sha256,
        selection=AiSelectionRecord(
            mode=sel.mode, reasons=tuple(r.value for r in sel.reasons), audit_ratio=sel.audit_ratio
        ),
        plan_digest=digest, planned_units=len(units), requested_by=options.actor,
        created_at=started_at,
    )  # fmt: skip


def _event(
    run: AiReviewRun | None,
    session_id: str,
    kind: AiRunEventKind,
    at: datetime,
    *,
    reason: str | None = None,
    counts: Mapping[str, int] | None = None,
    actor: str | None = None,
) -> AiRunEvent:
    """실행 이벤트 하나."""
    return AiRunEvent(
        session_id=session_id, run_id=run.run_id if run else None, event=kind, reason=reason,
        counts=dict(counts or {}), actor=actor, at=at,
    )  # fmt: skip


def _final_reason(ex: _Execution, run: AiReviewRun, plan_ids: set[str]) -> str | None:
    """마감 사유: 멈춤 사유 → 장애로 건너뛴 단위 → 마지막 확인(렌더·세션) → 기록 누락 (없으면 None).

    시간(`max_minutes`)·Ctrl-C(`signal`)로 멈췄어도 마지막 확인을 통과하고 계획 단위가 모두
    기록됐으면 완료다 (마지막 단위를 쓴 직후에 멈춘 경우).
    """
    if ex.reason is not None and ex.reason not in _SOFT:
        return ex.reason
    if ex.skipped:
        return ex.reason or "unavailable"
    try:
        if not ex.check_render():
            return ex.reason  # check_render가 render_changed·session_not_ready로 바꿨다
        missing = plan_ids - ex.sink.done_unit_ids(run.session_id, run.version)
    except Exception as exc:  # 마지막 확인의 DB 오류
        ex.halt("error", f"마감 확인 실패: {type(exc).__name__}: {exc}", exc)
        return "error"
    if missing:
        ex.detail = ex.detail or f"기록되지 않은 계획 단위 {len(missing)}개가 남았습니다"
        return ex.reason or "unavailable"
    return None


def _finish(ex: _Execution, run: AiReviewRun, base: RunSummary, plan_ids: set[str]) -> RunSummary:
    """마감: 사유를 정하고(`_final_reason`) 마감 이벤트를 쓴다."""
    reason = _final_reason(ex, run, plan_ids)
    kind = AiRunEventKind.COMPLETED if reason is None else _REASONS[reason][0]
    status: Status = (
        "completed"
        if kind is AiRunEventKind.COMPLETED
        else "interrupted"
        if kind is AiRunEventKind.INTERRUPTED
        else "aborted"
    )
    s = ex.stats
    summary = replace(
        base,
        status=status, reason=reason, detail=ex.detail if reason else None, run_id=run.run_id,
        written=s.written, inserted=ex.inserted, skipped=ex.skipped, fallback=s.fallback,
        verdicts=dict(s.verdicts),
        prompt_tokens=s.prompt_tokens, cached_tokens=s.cached_tokens,
        completion_tokens=s.completion_tokens, elapsed_s=ex.clock() - ex.started,
    )  # fmt: skip
    narrowed = ex.options.narrowed
    partial = kind is AiRunEventKind.COMPLETED and narrowed
    recorded = f"{PARTIAL_COMPLETION}{narrowed}" if partial else reason
    ex.sink.event(
        _event(run, run.session_id, kind, ex.now(), reason=recorded, counts=_counts(summary),
               actor=ex.options.actor)
    )  # fmt: skip
    return summary


def run_session(
    inp: RunInput,
    backend: JudgeBackend,
    sink: UnitSink,
    policy: AiReviewPolicy,
    *,
    options: RunOptions | None = None,
    render_check: RenderCheck | None = None,
    clock: Callable[[], float] = time.monotonic,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    progress: Callable[[str], None] | None = None,
) -> RunSummary:
    """세션 하나를 판정한다 (모듈 docstring). 외부 서비스 호출.

    Args:
        inp: 입력 (`load_input` 또는 테스트가 만든 것).
        backend: 판정 백엔드 (보통 `EndpointPool`).
        sink: 기록 대상.
        policy: 정책.
        options: 실행 설정.
        render_check: 렌더 재확인 (None이면 하지 않는다, `render_hashes_now`).
        clock, now: 단조 시계·기록 시각 (테스트).
        progress: 진행 한 줄 출력 (None이면 내지 않는다).

    바깥 멈춤 신호(`options.cancel`)가 시작 전에 이미 켜졌으면 아무것도 쓰지 않고
    interrupted:signal 요약을 돌려준다.

    Raises:
        BaseException: 버그·DB 오류는 마감 이벤트(aborted:error)를 쓴 뒤 다시 올린다.
        KeyboardInterrupt: Ctrl-C (interrupted:signal을 쓴 뒤).
    """
    opts = options or RunOptions()
    sid, version = inp.session.session_id, policy.verdict_version
    start = clock()
    units = planned_units(inp, policy, opts)
    groups = Counter(u.group.value for u in units)
    done = sink.done_unit_ids(sid, version)
    todo = [u for u in units if u.unit_id not in done]
    base = RunSummary(
        sid, version, "dry_run", planned=len(units), todo=len(todo), groups=dict(groups)
    )
    if opts.dry_run:
        return base
    full = opts.narrowed is None  # 좁힌 실행의 partial 완료는 전체 실행의 완료가 아니다
    if not todo and sink.is_complete(sid, version, plan_digest(units), full=full):
        return replace(base, status="nothing_to_do", elapsed_s=clock() - start)
    if opts.cancel is not None and opts.cancel.is_set():
        detail = "시작하기 전에 멈췄습니다 (아무것도 쓰지 않았습니다)"
        return replace(base, status="interrupted", reason="signal", detail=detail)
    started_at = now()
    run = _run_record(inp, backend, policy, opts, units, started_at)
    sink.start(run, _event(run, sid, AiRunEventKind.STARTED, started_at, actor=opts.actor,
                           counts={"planned": len(units), "todo": len(todo)}))  # fmt: skip
    ex = _Execution(inp, backend, sink, policy, opts, render_check, clock, now, progress)
    ex.run_id, ex.total, ex.already = run.run_id, len(units), len(units) - len(todo)
    try:
        ex.run(todo)
    except KeyboardInterrupt:
        _finish(ex, run, base, {u.unit_id for u in units})
        raise
    summary = _finish(ex, run, base, {u.unit_id for u in units})
    ex.report()
    if ex.error is not None:
        raise ex.error
    return summary


# ---------------------------------------------------------------- DB 경로


def session_problem(conn: sa.Connection, session: Session, *, approval: bool = True) -> str | None:
    """판정을 거부할 세션 상태의 설명 (괜찮으면 None): 사용 중지, 승인 보류, 승인 전.

    Args:
        approval: 승인 상태도 보는가. 실행 중 확인은 False다 (승인이 풀린 것은 렌더 확인이
            render_changed로 알린다: 블러 변경은 세션을 승인 전으로 되돌린다).
    """
    sid = session.session_id
    if session.lifecycle_state is LifecycleState.WITHDRAWN:
        return f"{sid}: 사용 중지된 세션은 판정하지 않습니다"
    held = active_privacy_hold(conn, sid)
    if held is not None:
        return f"{sid}: 승인 보류 중입니다 ({held.actor}: {held.reason})"
    if approval and session.privacy_state is not PrivacyState.APPROVED:
        return f"{sid}: 프라이버시 승인 전입니다 ({session.privacy_state.value})"
    return None


@contextmanager
def session_lock(engine: sa.Engine, session_id: str) -> Generator[bool]:
    """세션의 판정 실행 잠금을 실행 내내 쥔다 (전용 연결, 얻지 못하면 False를 낸다).

    잠금을 얻은 뒤 커밋해 연결을 트랜잭션 밖에 둔다 (세션 수준 잠금은 커밋해도 남는다, 몇 시간짜리
    실행이 열린 트랜잭션을 쥐지 않게). 끝나면 풀고 연결을 닫는다.
    """
    conn = engine.connect()
    got = False
    try:
        got = try_lock_aireview(conn, session_id)
        conn.commit()
        yield got
    finally:
        try:
            if got:
                unlock_aireview(conn, session_id)
                conn.commit()
        finally:
            conn.close()


def needs_sensor(windows: Iterable[PlannedWindow], policy: AiReviewPolicy) -> bool:
    """장갑 접촉 문장이 필요한가: 센서를 켰고, 판정할 시간 창에 손 상태 대상이 있다.

    문장은 손 상태 대상이 든 시간 단위에만 들어가므로(`plan.unit_sensor_sha`) 그런 대상이 없으면
    장갑을 읽지 않아도 단위 키·요청이 같다.
    """
    return policy.sensor.enabled and any(
        w.group is not AiGroup.SPATIAL
        and any(isinstance(x.payload, HandStatePayload) for x in w.targets)
        for w in windows
    )


def load_input(
    engine: sa.Engine,
    labeling: ObjectStore,
    session_id: str,
    policy: AiReviewPolicy,
    privacy_policy: PrivacyPolicy,
    work: Path,
    *,
    raw_bucket: str,
    groups: frozenset[AiGroup] | None = None,
    flagged_for: Callable[[sa.Connection, Session, list[LabelRecord]], Sequence[FlaggedSpan]]
    | None = None,
    sensor_for: Callable[[Session], SensorContext] | None = None,
    sensor_check: Callable[[Session], None] | None = None,
) -> RunInput:
    """DB와 라벨링 버킷에서 판정 입력을 만든다.

    짧은 읽기 트랜잭션으로 세션·현재 운영 라벨·세션 온톨로지를 읽고(세션 상태를 확인하고),
    판정할 창이 있는 스트림만 `open_blurred`로 받는다. 장갑 문장(`sensor_for`, 장갑 Parquet 원본
    접근)은 모든 블러본 게이트를 통과한 뒤, 손 상태 대상이 있을 때만 부른다: 게이트에서 거부될 세션
    (렌더 기록 없음·낡음 등)에 원본을 읽지 않는다. 계획만 하는 실행(`--dry-run`)도 단위 키에 센서
    해시가 들어가므로 같은 조건이면 장갑을 읽는다.

    Args:
        engine: DB 엔진. labeling: 라벨링 버킷 저장소. session_id: 세션.
        policy, privacy_policy: AI 판정·프라이버시 정책.
        work: 실행 임시 디렉터리 (블러본을 받는다).
        raw_bucket: 원본 버킷 이름 (`open_blurred`가 거부한다).
        groups: 이 묶음만.
        flagged_for: flagged 모드의 우선순위 사유 구간 계산 (CLI, 같은 읽기 트랜잭션).
        sensor_for: 장갑 접촉 구간 계산 (CLI, 감사 원본 저장소 `aireview.run`).
        sensor_check: 장갑이 필요할 때 큰 블러본을 받기 전에 부르는 전제 확인 (장갑 Parquet 존재,
            내용은 읽지 않는다). 블러본 게이트(받기 전 확인)를 모두 통과한 뒤에 부른다: 장갑이
            없어 거부될 세션이 cron마다 블러본(GB 단위)을 받았다 버리지 않게.

    Raises:
        SessionNotReadyError: 사용 중지·승인 보류·승인 전 (또는 `sensor_for`가 낸 것).
        MediaBoundaryError: `labeling`이 원본 버킷 저장소.
        RenderNotCurrentError: 블러본 렌더 게이트 실패.
    """
    with engine.connect() as conn:
        session = get_session(conn, session_id)
        problem = session_problem(conn, session)
        if problem is not None:
            raise SessionNotReadyError(problem)
        current = current_labels(get_labels(conn, session_id))
        ontology = get_session_ontology(conn, session_id)
        flagged = (
            tuple(flagged_for(conn, session, current))
            if flagged_for is not None and policy.selection.mode == "flagged"
            else ()
        )
    windows = [
        w
        for w in plan_windows(session, current, policy, ontology, flagged=flagged)
        if groups is None or w.group in groups
    ]
    streams = sorted({w.stream_id for w in windows})
    sensing = sensor_for is not None and needs_sensor(windows, policy)
    if sensing and sensor_check is not None:
        # 게이트(받기 전 확인) → 장갑 존재 확인 → 받기 순서: 원본은 게이트 통과 뒤에만 본다
        for s in streams:
            check_blurred(engine, labeling, session_id, s, privacy_policy, raw_bucket=raw_bucket)
        sensor_check(session)
    medias = {
        s: open_blurred(
            engine, labeling, session_id, s, privacy_policy, work, raw_bucket=raw_bucket
        )
        for s in streams
    }
    sensor = (  # 블러본 게이트를 모두 통과한 뒤에만 원본(장갑)을 읽는다
        sensor_for(session) if sensor_for is not None and sensing else SensorContext.absent()
    )

    def frames(stream: str, idx: Sequence[int]) -> Iterator[BlurredFrame]:
        return frames_for(medias[stream], idx)

    return RunInput(
        session=session, current=tuple(current), ontology=ontology,
        media={s: m.stream_media for s, m in medias.items()},
        blurred_sha256={s: m.blurred_sha256 for s, m in medias.items()},
        frames=frames, sensor=sensor, flagged=flagged,
    )  # fmt: skip


# 입력 단계의 예상한 거부 (CLI가 종료 코드로 나눈다). 그 밖의 예외는 `InputError`로 감싼다
_EXPECTED_INPUT_ERRORS = (
    SessionNotReadyError,
    MediaBoundaryError,
    RenderNotCurrentError,
    NoResultFound,
)


def run_from_db(
    engine: sa.Engine,
    labeling: ObjectStore,
    session_id: str,
    backend: JudgeBackend,
    policy: AiReviewPolicy,
    privacy_policy: PrivacyPolicy,
    *,
    raw_bucket: str,
    options: RunOptions | None = None,
    flagged_for: Callable[[sa.Connection, Session, list[LabelRecord]], Sequence[FlaggedSpan]]
    | None = None,
    sensor_for: Callable[[Session], SensorContext] | None = None,
    progress: Callable[[str], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
    sensor_check: Callable[[Session], None] | None = None,
) -> RunSummary | None:
    """세션 잠금 → 입력(`load_input`) → `run_session`(`DbSink`, 렌더 재확인). 블러본은 실행 임시
    디렉터리에 받고 끝나면 지운다.

    주기 확인(`render_check_interval_s`)과 마감 확인은 렌더 해시와 함께 세션 상태(`session_problem`:
    사용 중지·승인 보류)도 다시 본다 (aborted:session_not_ready). 실행 중 사용 중지(`dlp dataset
    withdraw`)는 생애주기만 바꾸고 승인·렌더 기록은 그대로라, 렌더 해시만 보면 놓친다.

    Returns:
        실행 결과. 다른 실행기가 이 세션을 처리 중이면 None (종료 7).

    Raises:
        SessionNotReadyError, MediaBoundaryError, RenderNotCurrentError: `load_input`과 같다.
        sqlalchemy.exc.NoResultFound: 세션이 없을 때.
        InputError: 입력을 만들다 그 밖의 예외가 났다 (실행 기록 전).
    """
    opts = options or RunOptions()
    with session_lock(engine, session_id) as locked:
        if not locked:
            return None
        with tempfile.TemporaryDirectory(prefix="aireview-") as tmp:
            try:
                inp = load_input(
                    engine, labeling, session_id, policy, privacy_policy, Path(tmp),
                    raw_bucket=raw_bucket, groups=opts.groups, flagged_for=flagged_for,
                    sensor_for=sensor_for, sensor_check=sensor_check,
                )  # fmt: skip
            except _EXPECTED_INPUT_ERRORS:
                raise
            except Exception as exc:
                raise InputError(f"{type(exc).__name__}: {exc}") from exc
            streams = list(inp.media)

            def check() -> Mapping[str, str]:
                with engine.connect() as conn:
                    session = get_session(conn, session_id)
                    problem = session_problem(conn, session, approval=False)
                if problem is not None:
                    raise SessionNotReadyError(problem)
                return render_hashes_now(engine, session_id, streams, privacy_policy)

            return run_session(
                inp, backend, DbSink(engine), policy, options=opts, render_check=check,
                clock=clock, progress=progress,
            )  # fmt: skip
