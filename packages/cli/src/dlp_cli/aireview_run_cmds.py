"""AI 1차 검수 실행·상태 하위 명령 (WP17.5, ADR 0040 5.2~5.5). `aireview_cmds.add_commands`가
등록한다.

등록하는 명령
- `dlp aireview run <세션>… | --pending [--limit N] [--parallel-sessions K] [--vlm-url URL …]
  [--store s3] [--groups …] [--mode all|flagged] [--max-minutes M] [--no-sensor] [--dry-run]
  [--debug-dir DIR] [--json-progress] [--url]`: 계획 → 판정 → AI 표 기록 (`dlp_aireview.runner`).
  멱등이고 재개된다 (기록된 단위는 다시 묻지 않는다). `--dry-run`은 계획만 센다 (서버 불필요,
  블러본 게이트는 거친다. 손 상태 대상이 있으면 단위 키의 센서 해시를 위해 장갑 Parquet은 읽는다).
  끝나면 세션마다 실행 요약 JSON 한 줄을 표준 출력에, 진행 줄은 표준 오류에 쓴다
  (`--json-progress`면 JSON Lines).
  종료 코드
  - 0: 완료, 할 일 없음, 계획만.
  - 1: 세션 실행 중 예상하지 못한 오류 (버그·DB 오류. aborted:error를 남기고, 다음 세션은 계속한다).
  - 2: 서버 주소 없음 (아무것도 쓰지 않음).
  - 3: 서버 장애로 중단 (완료한 단위는 남는다. 시작 전 모든 엔드포인트가 응답하지 않으면 아무것도
    쓰지 않음).
  - 5: 설정·라이선스 불일치 (4xx, 서버 모델이 정책 모델과 다름, 사내 주소 아님, 원본 버킷 저장소).
  - 6: 세션 전제 실패 (승인 전, 승인 보류, 사용 중지, 렌더 게이트, 장갑 Parquet 없음) 또는 실행 중
    블러 변경·사용 중지·승인 보류.
  - 7: 다른 실행기가 그 세션을 처리 중.
  - 8: `--max-minutes` 도달.
  - 130: Ctrl-C로 멈춘 세션 (`--parallel-sessions`로 함께 돌던 세션). Ctrl-C면 진행 중 요청을 끝내고
    세션마다 interrupted:signal을 남긴 뒤 KeyboardInterrupt로 끝난다 (셸 종료 코드 130).
  세션이 여럿이면 가장 심각한 코드(5 > 3 > 130 > 1 > 6 > 7 > 8, `_SEVERITY`)를 낸다. 5·8·130이 나면
  남은 세션은 시작하지 않는다 (다음 세션도 같다). 3은 엔드포인트 묶음 전체가 멈췄을 때만
  (`EndpointPool.dead`) 남은 세션을 멈추고, 단위 몇 개만 장애였으면 다음 세션을 계속한다 (그 세션은
  다음 실행이 이어서 한다). `--pending`은 전제가 안 맞는 세션(6)과 처리 중인 세션(7)을 알리고 0으로
  세며, `--limit`은 실제로 실행한 세션(6·7이 아닌 것)만 센다: 늘 전제가 안 맞는 세션이 앞자리를
  차지해 뒤 세션이 영영 판정되지 않는 일을 막는다.
  `--parallel-sessions K`: 세션 K개를 함께 돌린다. 세션 하나가 DB 연결을 최대 2개(실행 잠금 +
  기록자) 쥐므로 연결 풀을 K에 맞춰 키운다 (`engine_options`). K는 PostgreSQL `max_connections`의
  절반보다 작게 둔다.
  4는 회수 대기(`dlp review revoke`)의 코드라 쓰지 않는다.
- `dlp aireview status [<세션>… | --pending | --all] [--groups …] [--json] [--url]`: 세션별
  판정 버전, 판정률(판정된 대상/계획 대상), 단위·fallback 수, 판정 분포, 마지막 이벤트, 최근
  처리량 기반 남은 시간, 백로그(영상 시간). started만 있고 실행 중이 아니면 "중단 추정"이다.
  실행 중인지는 `pg_locks`만 읽어 본다 (잠금을 잡지 않는다, `aireview_lock_held`).
  종료 0 (없는 세션을 주면 1).
- `dlp aireview show <unit_id> | <세션>:<label_id> [--url]`: 감사 기록 전체 (프레임 PTS·sha,
  렌더 해시, 프롬프트 다이제스트, 판정, 응답 sha, 서버 모델, 엔드포인트 이름, 실행자). 라벨을
  주면 그 라벨이 대상으로 든 단위들과 지금 유효한(표시 프로필) 판정. 종료 0 (없으면 1).
- `dlp aireview waive <세션> --reason TEXT [--url]`: required 모드(`review.yaml ai_review.mode`,
  WP17.6)에서 이 세션의 AI 단계를 건너뛰기로 한 결정을 남긴다 (`waived` 이벤트, 사유
  `manual: <사유>`, 실행자 `DLP_ACTOR`). 종료 0 (세션이 없으면 1).

원본 접근: 장갑 접촉 문장(프리라벨과 같은 `glove_contact_intervals`, 같은 입력)을 위해 장갑
Parquet만 감사 원본 저장소(`raw_store`, 용도 `aireview.run`)로 읽는다. 손 상태 대상이 있고 센서를
켰을 때만 읽는다. 판정 프레임은 라벨링 버킷의 승인된 블러본에서만 얻는다 (`open_blurred`).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import threading
import time
import traceback
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import sqlalchemy as sa
from sqlalchemy.exc import NoResultFound

from dlp_actions.policy import load_policy as load_actions_policy
from dlp_aireview.backend import (
    BackendInfo,
    EndpointPool,
    JudgeBackend,
    JudgeReply,
    JudgeRequest,
    OpenAICompatibleJudge,
    PoolMember,
    check_hosts,
)
from dlp_aireview.media import MediaBoundaryError
from dlp_aireview.policy import AiReviewPolicy, load_policy
from dlp_aireview.runner import (
    RunOptions,
    RunSummary,
    SessionNotReadyError,
    run_from_db,
    start_thread,
    wait_all,
)
from dlp_aireview.sensors import SensorContext
from dlp_aireview.status import SessionStatus, pending_sessions, session_status
from dlp_cli.aireview_common import (
    CONFIG,
    UNAVAILABLE,
    add_url_arg,
    flagged_spans,
    no_url,
    parse_groups,
    vlm_api_key,
    vlm_endpoints,
)
from dlp_cli.raw_access import raw_store
from dlp_cli.schema_cmds import database_url
from dlp_media.audit import AuditedStore, current_actor
from dlp_media.storage import ObjectStore, store_from_spec
from dlp_models.vlm import VlmRequestError, VlmUnavailableError
from dlp_prelabel.contact import glove_contact_intervals
from dlp_prelabel.policy import GloveContactPolicy
from dlp_prelabel.policy import load_policy as load_prelabel_policy
from dlp_privacy.errors import RenderNotCurrentError
from dlp_privacy.policy import load_policy as load_privacy_policy
from dlp_privacy.rendering import expected_render_hash
from dlp_schema import load_config, repo_root
from dlp_schema.aireview import (
    AiGroup,
    AiRunEvent,
    AiRunEventKind,
    valid_verdicts,
)
from dlp_schema.db.aireview_repository import (
    get_ai_run,
    get_ai_unit,
    insert_ai_run_event,
    list_ai_units,
)
from dlp_schema.db.locks import aireview_lock_held
from dlp_schema.db.repository import get_labels, get_session, list_session_ids
from dlp_schema.episode import current_labels
from dlp_schema.labels import Hand
from dlp_schema.session import LifecycleState, Session, StreamKind, SyncMethod
from dlp_sync.policy import load_policy as load_sync_policy
from dlp_sync.signals import glove_series

ERROR = 1
SESSION = 6
LOCKED = 7
MAX_MINUTES = 8
SIGNAL = 130
# 여러 세션의 종료 코드를 하나로 (앞일수록 심각하다, 모듈 docstring)
_SEVERITY = (CONFIG, UNAVAILABLE, SIGNAL, ERROR, SESSION, LOCKED, MAX_MINUTES)
# 이 코드가 나면 남은 세션을 시작하지 않는다 (설정·시간·Ctrl-C는 다음 세션도 같다). 장애(3)는
# 엔드포인트 묶음 전체가 멈췄을 때만 (`_stops`)
_STOP = frozenset({CONFIG, MAX_MINUTES, SIGNAL})
# 실행하지 않은 세션 (전제 실패·처리 중): --pending의 --limit에 세지 않는다
_NOT_RUN = frozenset({SESSION, LOCKED})
# 세션 하나의 실행이 함께 쥐는 DB 연결 수 상한: 실행 잠금(실행 내내) + 기록자(쓰기·렌더 확인)
_CONNECTIONS_PER_SESSION = 2
_GLOVES = {StreamKind.GLOVE_LEFT: Hand.LEFT, StreamKind.GLOVE_RIGHT: Hand.RIGHT}


# ---------------------------------------------------------------- 서버


class _NoServer:
    """계획만(`--dry-run`) 하는 실행의 백엔드: 부르면 버그다."""

    def __init__(self, policy: AiReviewPolicy) -> None:
        b = policy.backend
        self.info = BackendInfo(kind=b.kind, model=b.model, fingerprint=b.fingerprint, endpoints=())

    def judge(self, request: JudgeRequest) -> JudgeReply:
        """부르지 않는다."""
        raise AssertionError("계획만 하는 실행은 서버를 부르지 않습니다")


def _check_servers(members: Sequence[PoolMember]) -> int | None:
    """시작 전 확인: 엔드포인트마다 `/v1/models`에 정책 모델이 있는가.

    없거나 4xx면 5, 모두 응답하지 않으면 3 (아무것도 쓰기 전). 일부만 응답하지 않으면 경고하고
    진행한다 (차단기가 그 엔드포인트를 뺀다).
    """
    down: list[str] = []
    for m in members:
        if m.ping is None:
            continue
        try:
            m.ping()
        except VlmRequestError as exc:
            print(f"[VLM 설정 오류] {m.name}: 서버 주소·모델·키를 확인하세요 ({exc})")
            return CONFIG
        except VlmUnavailableError as exc:
            print(f"[VLM 서버 장애] {m.name}: {exc}")
            down.append(m.name)
    if down and len(down) == len(members):
        print("[VLM 서버 장애] 응답하는 엔드포인트가 없습니다. 아무것도 쓰지 않았습니다")
        return UNAVAILABLE
    return None


def open_pool(args: argparse.Namespace, policy: AiReviewPolicy) -> EndpointPool | int:
    """엔드포인트 묶음 (또는 종료 코드: 2 주소 없음, 5 설정, 3 모두 장애)."""
    try:
        endpoints = vlm_endpoints(args)
    except ValueError as exc:
        print(f"[설정 오류] {exc}")
        return CONFIG
    if not endpoints:
        return no_url()
    outside = check_hosts(endpoints, policy.server.internal_hosts)
    if outside:
        print(f"[설정 오류] 사내 대역 밖 주소에는 보내지 않습니다: {'; '.join(outside)}")
        return CONFIG
    members: list[PoolMember] = []
    for name, url in endpoints.items():
        judge = OpenAICompatibleJudge(name, url, policy, api_key=vlm_api_key())
        members.append(PoolMember(name, judge, judge.ping))
    code = _check_servers(members)
    return code if code is not None else EndpointPool(members, policy.server)


# ---------------------------------------------------------------- 장갑 문장


def glove_context(
    session: Session,
    raw: AuditedStore,
    glove_policy: GloveContactPolicy,
    pressure_prefixes: Sequence[str],
) -> SensorContext:
    """세션 장갑의 손별 접촉 구간 → `SensorContext` (ADR 0040 2.5).

    프리라벨 접촉 단계와 같은 입력·같은 함수다: 동기화된 장갑 스트림만, 원 샘플 시각을
    `to_master_ms`로 마스터 ms로 바꿔 `glove_contact_intervals`(`prelabel.yaml contact.glove`)에
    넣는다. 장갑 Parquet은 감사 원본 저장소(용도 `aireview.run`)로 임시 디렉터리에 받고 지운다.

    Raises:
        SessionNotReadyError: 선언된 장갑 스트림의 Parquet이 원본 버킷에 없을 때.
    """
    gloves = {
        _GLOVES[s.kind]: s
        for s in session.streams
        if s.kind in _GLOVES and s.sync_method is not SyncMethod.UNSYNCED
    }
    if not gloves:
        return SensorContext.absent()
    spans: dict[Hand, list[tuple[float, float]]] = {}
    with tempfile.TemporaryDirectory(prefix="aireview-glove-") as tmp:
        for hand, stream in sorted(gloves.items()):
            key = stream.uri.removeprefix(raw.uri(""))
            if raw.head(key) is None:
                raise SessionNotReadyError(
                    f"{session.session_id}: 장갑 Parquet이 없습니다 ({stream.stream_id}). "
                    "센서 없이 판정하려면 --no-sensor"
                )
            path = Path(tmp) / f"{stream.stream_id}.parquet"
            raw.get_file(key, path)
            series = glove_series(path, pressure_prefixes)
            master = np.array([stream.to_master_ms(float(t)) for t in series.t_ms])
            found = glove_contact_intervals(master, series.values, glove_policy)
            spans[hand] = [(c.start_ms, c.end_ms) for c in found]
    return SensorContext.of(spans)


def _sensor_source(args: argparse.Namespace) -> Callable[[Session], SensorContext] | None:
    """장갑 문장 계산 함수 (`--no-sensor`면 None: 센서 없이 판정)."""
    if args.no_sensor:
        return None
    root = repo_root()
    glove = load_prelabel_policy(root).contact.glove
    prefixes = load_sync_policy(root / "config/policies/sync.yaml").glove.pressure_prefixes
    raw = raw_store(args.store, args.url, "aireview.run")

    def compute(session: Session) -> SensorContext:
        return glove_context(session, raw, glove, prefixes)

    return compute


# ---------------------------------------------------------------- run


@dataclass(frozen=True)
class _RunCtx:
    """세션마다 같은 실행 준비물."""

    engine: sa.Engine
    labeling: ObjectStore
    raw_bucket: str
    backend: JudgeBackend
    policy: AiReviewPolicy
    options: RunOptions
    sensor_for: Callable[[Session], SensorContext] | None


def _progress(line: str) -> None:
    """진행 줄은 표준 오류로."""
    print(line, file=sys.stderr, flush=True)


def _print_summary(s: RunSummary) -> None:
    """세션 결과: 계획만이면 묶음별 표, 사람이 읽는 한 줄, 그리고 요약 JSON 한 줄."""
    if s.status == "dry_run":
        groups = " · ".join(f"{g} {n}" for g, n in s.groups.items()) or "판정할 단위 없음"
        print(f"{s.session_id} 계획: {groups} (단위 {s.planned}, 남은 {s.todo})")
    elif s.reason:
        print(f"[{s.status}:{s.reason}] {s.session_id}: {s.detail or ''}".rstrip())
    print(json.dumps(s.to_json(), ensure_ascii=False))


def _run_one(sid: str, ctx: _RunCtx) -> int:
    """세션 하나 (모듈 docstring의 종료 코드). 부작용: AI 표 기록, VLM 호출, 라벨링 버킷 읽기.

    예상하지 못한 오류(버그·DB 오류)는 알리고 1로 센다 (실행기는 aborted:error를 남겼다). 다음
    세션은 계속한다. Ctrl-C(KeyboardInterrupt)는 올린다.
    """
    try:
        summary = run_from_db(
            ctx.engine, ctx.labeling, sid, ctx.backend, ctx.policy,
            load_privacy_policy(repo_root()), raw_bucket=ctx.raw_bucket, options=ctx.options,
            flagged_for=flagged_spans, sensor_for=ctx.sensor_for, progress=_progress,
        )  # fmt: skip
    except NoResultFound:
        print(f"[전제 실패] {sid}: 세션이 없습니다")
        return SESSION
    except SessionNotReadyError as exc:
        print(f"[전제 실패] {exc}")
        return SESSION
    except RenderNotCurrentError as exc:
        print(f"[블러본 게이트] {sid}: {exc}")
        return SESSION
    except MediaBoundaryError as exc:
        print(f"[설정 오류] {exc}")
        return CONFIG
    except Exception as exc:
        print(f"[실행 오류] {sid}: {type(exc).__name__}: {exc}")
        traceback.print_exc(file=sys.stderr)
        return ERROR
    if summary is None:
        print(f"[처리 중] {sid}: 다른 실행기가 이 세션을 처리하고 있습니다")
        return LOCKED
    _print_summary(summary)
    return summary.exit_code


def _worst(codes: Sequence[int], pending: bool) -> int:
    """여러 세션의 종료 코드 → 하나 (`--pending`은 6·7을 0으로 센다)."""
    seen = {c for c in codes if not (pending and c in _NOT_RUN)}
    return next((c for c in _SEVERITY if c in seen), 0)


def _stops(code: int, backend: JudgeBackend) -> bool:
    """이 코드가 나면 남은 세션을 시작하지 않는가: 설정·시간·Ctrl-C, 또는 엔드포인트 묶음 전체가
    멈춘 장애 (단위 몇 개만 장애였으면 다음 세션을 계속한다)."""
    if code in _STOP:
        return True
    return code == UNAVAILABLE and isinstance(backend, EndpointPool) and backend.dead


class _Picker:
    """돌릴 세션을 차례로 내준다 (여러 작업 스레드가 함께 쓴다).

    - `limit`(`--pending --limit`): 실제로 실행한 세션(6·7이 아닌 것)과 실행 중인 세션의 합이
      이만큼이면 더 내주지 않는다. 실행 중인 세션이 6·7로 끝나 자리가 나면 다음 세션을 내준다.
    - 멈춤: `_stops`인 코드가 나오거나 `halt`(Ctrl-C)면 남은 세션을 내주지 않는다.

    Attributes:
        left: 아직 내주지 않은 세션. ran: 실행한 세션. not_run: 6·7로 끝난 세션.
        codes: 끝난 세션의 종료 코드 (끝난 순서). stopped: 멈췄는가.
    """

    def __init__(self, sessions: Sequence[str], limit: int | None) -> None:
        """sessions: 돌릴 순서. limit: 실행 세션 수 상한 (None이면 없음)."""
        self.left = deque(sessions)
        self.limit = limit
        self.ran: list[str] = []
        self.not_run: list[str] = []
        self.codes: list[int] = []
        self.stopped = False
        self._running = 0
        self._cond = threading.Condition()

    def next(self) -> str | None:
        """다음 세션 (없거나 멈췄거나 상한을 채웠으면 None, 자리가 날 수 있으면 기다린다)."""
        with self._cond:
            while True:
                if self.stopped or not self.left:
                    return None
                if self.limit is None or len(self.ran) + self._running < self.limit:
                    self._running += 1
                    return self.left.popleft()
                if self._running == 0:
                    return None
                self._cond.wait()

    def done(self, sid: str, code: int, stop: bool) -> None:
        """세션 하나가 끝났다."""
        with self._cond:
            self._running -= 1
            self.codes.append(code)
            (self.not_run if code in _NOT_RUN else self.ran).append(sid)
            self.stopped = self.stopped or stop
            self._cond.notify_all()

    def halt(self) -> None:
        """남은 세션을 내주지 않는다 (Ctrl-C)."""
        with self._cond:
            self.stopped = True
            self._cond.notify_all()


def _work(picker: _Picker, ctx: _RunCtx) -> None:
    """작업 스레드 하나: 세션을 받아 끝날 때까지 돌린다."""
    while (sid := picker.next()) is not None:
        code = ERROR
        try:
            code = _run_one(sid, ctx)
        finally:
            picker.done(sid, code, _stops(code, ctx.backend))


def _run_all(
    sessions: Sequence[str], ctx: _RunCtx, parallel: int, limit: int | None = None
) -> _Picker:
    """세션들을 차례로(또는 K개씩 함께) 돌린다 (`_Picker`의 상한·멈춤 규칙).

    여러 세션을 함께 돌릴 때 Ctrl-C는 주 스레드에만 온다. 그때 남은 세션을 내주지 않고 공유 멈춤
    신호(`RunOptions.cancel`)를 켜서, 돌던 세션마다 진행 중 요청을 끝낸 뒤 interrupted:signal로
    닫게 하고, 모두 끝나면 KeyboardInterrupt를 다시 올린다.
    """
    picker = _Picker(sessions, limit)
    if parallel <= 1:
        _work(picker, ctx)  # Ctrl-C는 실행기가 이 스레드에서 받아 interrupted:signal로 닫는다
        return picker
    threads = [start_thread(_work, picker, ctx) for _ in range(min(parallel, len(sessions)))]
    try:
        wait_all(threads)
    except KeyboardInterrupt:
        picker.halt()
        if ctx.options.cancel is not None:
            ctx.options.cancel.set()
        print("[중단] Ctrl-C: 진행 중 요청을 끝내고 세션마다 멈춤을 기록합니다", file=sys.stderr)
        wait_all(threads)
        raise
    return picker


def _report_left(picker: _Picker, limit: int | None) -> None:
    """시작하지 않은 세션을 알린다 (멈춤 또는 --limit)."""
    if not picker.left:
        return
    if picker.stopped:
        for sid in picker.left:
            print(f"[건너뜀] {sid}: 앞 세션에서 실행을 멈췄습니다")
    elif limit is not None:
        print(
            f"[pending] 실행 상한 {limit}개를 채웠습니다. 다음 실행으로 미룬 세션 "
            f"{len(picker.left)}개: {_names(picker.left)}"
        )


def _names(ids: Sequence[str] | deque[str], most: int = 20) -> str:
    """세션 이름 목록 (많으면 앞 `most`개와 나머지 수)."""
    head = ", ".join(list(ids)[:most])
    return head + (f" 외 {len(ids) - most}개" if len(ids) > most else "")


def _narrowed(args: argparse.Namespace, base: AiReviewPolicy) -> str | None:
    """정책 기본 계획보다 좁힌 실행의 설명 (`RunOptions.narrowed`, 좁히지 않았으면 None).

    `--groups`가 모든 묶음이 아니거나, `--mode flagged`가 정책 기본(all)보다 좁을 때다 (`--mode
    all`은 넓히는 쪽이라 좁힌 것이 아니다). 좁힌 실행의 completed는 `partial:`로 마감해
    `--pending`과 required 게이트가 전체 완료로 보지 않는다.
    """
    parts: list[str] = []
    if args.groups and set(args.groups) != set(AiGroup):
        parts.append("groups=" + ",".join(sorted(g.value for g in set(args.groups))))
    if args.mode == "flagged" and base.selection.mode != "flagged":
        parts.append("mode=flagged")
    return ";".join(parts) or None


def _run_policy(args: argparse.Namespace) -> AiReviewPolicy:
    """정책 (`--mode`면 선택 모드만 바꾼 사본, 판정 버전은 그대로)."""
    policy = load_policy(repo_root())
    if args.mode:
        data = policy.model_dump()
        data["selection"]["mode"] = args.mode
        policy = AiReviewPolicy.model_validate(data)
    return policy


def _sessions(
    args: argparse.Namespace,
    engine: sa.Engine,
    policy: AiReviewPolicy,
    groups: frozenset[AiGroup] | None,
) -> list[str]:
    """판정할 세션: 지정한 세션, 또는 `--pending` 조건을 만족하는 세션 (고른 순서).

    `--limit`은 여기서 자르지 않는다: 실행할 때 실제로 실행한 세션만 센다 (`_Picker`).
    """
    if not args.pending:
        return list(args.sessions)
    with engine.connect() as conn:
        found = pending_sessions(
            conn, policy, load_privacy_policy(repo_root()), datetime.now(UTC),
            groups=groups, flagged_for=flagged_spans,
        )  # fmt: skip
    ids = [p.session_id for p in found]
    cap = f" (실행 상한 {args.limit}개)" if args.limit else ""
    print(f"[pending] 판정할 세션 {len(ids)}개{cap}: {_names(ids)}")
    return ids


def engine_options(parallel: int) -> dict[str, int]:
    """`--parallel-sessions K`에 맞춘 DB 연결 풀 크기 (`sa.create_engine` 인자).

    세션 하나가 연결을 최대 `_CONNECTIONS_PER_SESSION`개 함께 쥔다 (실행 잠금은 실행 내내). 기본 풀
    (5 + 넘침 10)이면 K가 8 이상일 때 연결을 기다리다 30초 뒤 실패한다.
    """
    return {"pool_size": _CONNECTIONS_PER_SESSION * max(1, parallel) + 1, "max_overflow": 10}


def _options(args: argparse.Namespace, policy: AiReviewPolicy, backend: JudgeBackend) -> RunOptions:
    """실행 설정 (작업자 = 엔드포인트 수 * 엔드포인트별 동시성)."""
    endpoints = max(1, len(backend.info.endpoints))
    actions_model = load_actions_policy(repo_root()).vlm.model
    same = actions_model == policy.backend.model
    if same:
        print(
            f"[경고] 판정 모델이 행동 분류 모델과 같습니다 ({actions_model}): 자기 검수 상관 오류"
        )
    states = backend.states if isinstance(backend, EndpointPool) else None
    return RunOptions(
        workers=endpoints * policy.server.concurrency_per_endpoint,
        groups=frozenset(args.groups) if args.groups else None,
        deadline=time.monotonic() + args.max_minutes * 60 if args.max_minutes else None,
        dry_run=args.dry_run,
        debug_dir=Path(args.debug_dir) if args.debug_dir else None,
        actor=current_actor(),
        same_model_as_actions=same,
        json_progress=args.json_progress,
        endpoint_states=states,
        cancel=threading.Event(),
        narrowed=_narrowed(args, load_policy(repo_root())),
    )


def cmd_run(args: argparse.Namespace) -> int:
    """`dlp aireview run` (모듈 docstring).

    부작용: AI 표(ai_review_runs·run_events·units) 기록, VLM 서버 호출(사내 주소만), 라벨링
    버킷 블러본 읽기(임시 디렉터리), 장갑이 필요하면 원본 버킷 장갑 Parquet 읽기(감사 기록
    `aireview.run`). 라벨 이력(`label_records`)에는 쓰지 않는다.
    """
    if not args.sessions and not args.pending:
        print("[설정 오류] 세션을 주거나 --pending을 쓰세요")
        return CONFIG
    policy = _run_policy(args)
    backend: JudgeBackend
    if args.dry_run:
        backend = _NoServer(policy)
    else:
        opened = open_pool(args, policy)
        if isinstance(opened, int):
            return opened
        backend = opened
    root = repo_root()
    buckets = load_config(root / "config/defaults.yaml").buckets
    engine = sa.create_engine(database_url(args.url), **engine_options(args.parallel_sessions))
    try:
        options = _options(args, policy, backend)
        sessions = _sessions(args, engine, policy, options.groups)
        ctx = _RunCtx(
            engine=engine, labeling=store_from_spec(args.store, buckets.labeling),
            raw_bucket=buckets.raw, backend=backend, policy=policy, options=options,
            sensor_for=_sensor_source(args),
        )  # fmt: skip
        limit = args.limit if args.pending else None
        picker = _run_all(sessions, ctx, args.parallel_sessions, limit)
        _report_left(picker, limit)
    finally:
        engine.dispose()
    return _worst(picker.codes, args.pending)


# ---------------------------------------------------------------- status


def _running(conn: sa.Connection, session_id: str) -> bool | None:
    """실행기가 이 세션의 잠금을 쥐고 있는가 (`pg_locks`만 읽는다, SQLite면 None)."""
    return aireview_lock_held(conn, session_id)


def _status_line(s: SessionStatus) -> str:
    """세션 상태 한 줄."""
    cov = f"{s.coverage:.0%}" if s.coverage is not None else "-"
    v = s.verdicts
    parts = [
        s.session_id,
        s.last_event or "실행 없음",
        f"판정 {s.covered}/{s.targets}({cov})",
        f"단위 {s.units} · fallback {s.fallback}",
        f"틀림 {v.get('wrong', 0)} 의심 {v.get('suspect', 0)} 판단불가 {v.get('unsure', 0)} "
        f"맞음 {v.get('ok', 0)}",
    ]
    if s.units_per_s:
        eta = f" · 남은 약 {max(1, round(s.eta_s / 60))}분" if s.eta_s else ""
        parts.append(f"{s.units_per_s:.2f} 단위/s{eta}")
    notes = [
        text
        for flag, text in (
            (s.presumed_interrupted, "중단 추정"), (s.running is True, "실행 중"),
            (not s.approved, "승인 전"), (s.held, "승인 보류"), (s.waived, "건너뜀 기록"),
        )
        if flag
    ]  # fmt: skip
    if notes:
        parts.append(", ".join(notes))
    return " · ".join(parts)


def _status_ids(args: argparse.Namespace, conn: sa.Connection, policy: AiReviewPolicy) -> list[str]:
    """상태를 볼 세션: 지정, `--pending`, `--all`(사용 중지 제외)."""
    if args.sessions:
        return list(args.sessions)
    if args.pending:
        groups = frozenset(args.groups) if args.groups else None
        found = pending_sessions(
            conn, policy, load_privacy_policy(repo_root()), datetime.now(UTC),
            groups=groups, flagged_for=flagged_spans,
        )  # fmt: skip
        return [p.session_id for p in found]
    if args.all:
        return [
            sid
            for sid in list_session_ids(conn)
            if get_session(conn, sid).lifecycle_state is not LifecycleState.WITHDRAWN
        ]
    return []


def _json_default(value: object) -> str:
    """JSON 직렬화: 시각은 ISO 8601."""
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def cmd_status(args: argparse.Namespace) -> int:
    """`dlp aireview status` (모듈 docstring). 부작용: 없음 (DB 읽기, 잠금은 보기만)."""
    policy = load_policy(repo_root())
    privacy = load_privacy_policy(repo_root())
    groups = frozenset(args.groups) if args.groups else None
    engine = sa.create_engine(database_url(args.url))
    try:
        with engine.connect() as conn:
            ids = _status_ids(args, conn, policy)
            known = set(list_session_ids(conn))
            missing = [sid for sid in ids if sid not in known]
            if missing:
                print(f"세션이 없습니다: {', '.join(missing)}")
                return 1
            rows = [
                session_status(
                    conn, sid, policy, privacy, groups=groups, flagged_for=flagged_spans,
                    running=_running(conn, sid),
                )
                for sid in ids
            ]  # fmt: skip
    finally:
        engine.dispose()
    if args.json:
        data = [asdict(r) | {"coverage": r.coverage} for r in rows]
        print(json.dumps(data, ensure_ascii=False, indent=2, default=_json_default))
        return 0
    print(f"판정 버전 {policy.verdict_version}")
    if not rows:
        print("볼 세션이 없습니다 (<세션>…, --pending, --all)")
    for r in rows:
        print(_status_line(r))
    backlog = sum(r.backlog_hours for r in rows)
    if len(rows) > 1:
        print(f"세션 {len(rows)}개 · 판정할 대상이 남은 영상 {backlog:.2f}시간")
    return 0


# ---------------------------------------------------------------- show


def _unit_record(conn: sa.Connection, unit_id: str) -> dict[str, Any] | None:
    """단위 감사 기록과 그 실행 (없으면 None)."""
    unit = get_ai_unit(conn, unit_id)
    if unit is None:
        return None
    run = get_ai_run(conn, unit.run_id)
    return {
        "unit": unit.model_dump(mode="json"),
        "run": {
            "run_id": run.run_id, "requested_by": run.requested_by,
            "backend": run.backend.model_dump(mode="json"),
            "created_at": run.created_at.isoformat(), "plan_digest": run.plan_digest,
        },
    }  # fmt: skip


def _split_ref(conn: sa.Connection, ref: str) -> tuple[str, str] | None:
    """`<세션>:<라벨>` → (세션, 라벨). 세션 ID에도 `:`가 있을 수 있어 앞에서부터 있는 세션을
    찾는다."""
    known = set(list_session_ids(conn))
    for k, ch in enumerate(ref):
        if ch == ":" and ref[:k] in known:
            return ref[:k], ref[k + 1 :]
    return None


def _label_record(
    conn: sa.Connection, session_id: str, label_id: str, policy: AiReviewPolicy
) -> dict[str, Any]:
    """라벨이 대상으로 든 단위들과 그 라벨의 판정, 지금 유효한 판정 (지금 버전·표시 프로필)."""
    units = [u for u in list_ai_units(conn, session_id) if label_id in u.target_label_ids]
    current = current_labels(get_labels(conn, session_id))
    try:
        hashes: dict[str, str] | None = {
            s: _render_now(conn, session_id, s) for s in sorted({u.stream_id for u in units})
        }
    except RenderNotCurrentError:
        hashes = None
    valid = (
        valid_verdicts(units, current, versions=[policy.verdict_version], render_hashes=hashes)
        if hashes is not None
        else {}
    )
    found = valid.get(label_id)
    rows = [
        {
            "unit_id": u.unit_id, "version": u.version, "created_at": u.created_at.isoformat(),
            "stream_id": u.stream_id, "render_hash": u.render_hash,
            "verdict": v.model_dump(mode="json"),
        }
        for u in units
        for v in u.verdicts
        if v.label_id == label_id
    ]  # fmt: skip
    shown = None if found is None else {"unit_id": found[1], **found[0].model_dump(mode="json")}
    return {"session_id": session_id, "label_id": label_id, "valid": shown, "units": rows}


def _render_now(conn: sa.Connection, session_id: str, stream_id: str) -> str:
    """지금 렌더 해시 (승인 상태가 아니면 RenderNotCurrentError)."""
    return expected_render_hash(conn, session_id, stream_id, load_privacy_policy(repo_root()))


def cmd_show(args: argparse.Namespace) -> int:
    """`dlp aireview show` (모듈 docstring). 부작용: 없음 (DB 읽기)."""
    policy = load_policy(repo_root())
    engine = sa.create_engine(database_url(args.url))
    try:
        with engine.connect() as conn:
            record = _unit_record(conn, args.ref) if args.ref.startswith("air-") else None
            if record is None:
                split = _split_ref(conn, args.ref)
                record = _label_record(conn, *split, policy) if split else None
    finally:
        engine.dispose()
    if record is None:
        print(f"찾지 못했습니다: {args.ref} (<unit_id> 또는 <세션>:<라벨 ID>)")
        return 1
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


# ---------------------------------------------------------------- waive


def cmd_waive(args: argparse.Namespace) -> int:
    """`dlp aireview waive <세션> --reason` (모듈 docstring). 부작용: waived 이벤트 1행."""
    reason = args.reason.strip()
    if not reason:
        print("[설정 오류] 건너뛰는 사유가 필요합니다 (--reason)")
        return CONFIG
    engine = sa.create_engine(database_url(args.url))
    try:
        with engine.begin() as conn:
            try:
                get_session(conn, args.session_id)
            except NoResultFound:
                print(f"세션이 없습니다: {args.session_id}")
                return 1
            event = AiRunEvent(
                session_id=args.session_id, event=AiRunEventKind.WAIVED,
                reason=f"manual: {reason}", actor=current_actor(), at=datetime.now(UTC),
            )  # fmt: skip
            event_id = insert_ai_run_event(conn, event)
    finally:
        engine.dispose()
    print(
        f"{args.session_id}: AI 단계를 건너뛰기로 기록했습니다 (이벤트 {event_id}). "
        "review.yaml ai_review.mode가 required일 때 검수 배정이 이 세션을 미루지 않습니다"
    )
    return 0


# ---------------------------------------------------------------- 등록


def _positive(text: str) -> int:
    """1 이상의 정수 (argparse 형식 함수)."""
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"정수가 아닙니다: {text}") from exc
    if value < 1:
        raise argparse.ArgumentTypeError(f"1 이상이어야 합니다: {text}")
    return value


def _run_parser(asub: Any) -> None:
    """`aireview run` 인자."""
    r = asub.add_parser("run", help="판정 실행 (멱등·재개, 참고용 판정만 AI 표에 기록)")
    r.add_argument("sessions", nargs="*")
    r.add_argument("--pending", action="store_true", help="판정할 대상이 남은 prelabeled 세션")
    r.add_argument(
        "--limit", type=_positive, help="--pending에서 실제로 실행할 세션 수 상한 (6·7은 세지 않음)"
    )
    r.add_argument(
        "--parallel-sessions", type=_positive, default=1,
        help="함께 돌릴 세션 수 (세션마다 DB 연결 최대 2개)",
    )  # fmt: skip
    add_url_arg(r)
    r.add_argument("--store", default="s3", help="'s3' 또는 'local:<디렉터리>'")
    r.add_argument("--groups", type=parse_groups, help="이 묶음만 (temporal,temporal_long,spatial)")
    r.add_argument("--mode", choices=["all", "flagged"], help="선택 모드 (기본: 정책)")
    r.add_argument("--max-minutes", type=float, help="이 시간이 지나면 새 요청을 멈춘다 (종료 8)")
    r.add_argument("--no-sensor", action="store_true", help="장갑 접촉 문장 없이 판정")
    r.add_argument("--dry-run", action="store_true", help="계획만 센다 (서버 불필요)")
    r.add_argument("--debug-dir", help="형식이 맞지 않던 응답 원문을 남길 로컬 디렉터리")
    r.add_argument("--json-progress", action="store_true", help="진행 줄을 JSON Lines로")
    r.add_argument("--url", help="DB URL (기본: DLP_DATABASE_URL)")
    r.set_defaults(func=cmd_run)


def add_run_commands(asub: Any) -> None:
    """`aireview run|status|show|waive` 하위 명령을 등록한다."""
    _run_parser(asub)
    s = asub.add_parser("status", help="세션별 판정 진행 상태")
    s.add_argument("sessions", nargs="*")
    s.add_argument("--pending", action="store_true", help="--pending이 고를 세션")
    s.add_argument("--all", action="store_true", help="사용 중지가 아닌 모든 세션")
    s.add_argument("--groups", type=parse_groups)
    s.add_argument("--json", action="store_true")
    s.add_argument("--url", help="DB URL (기본: DLP_DATABASE_URL)")
    s.set_defaults(func=cmd_status)
    w = asub.add_parser("show", help="판정 단위 감사 기록 (<unit_id> | <세션>:<라벨 ID>)")
    w.add_argument("ref")
    w.add_argument("--url", help="DB URL (기본: DLP_DATABASE_URL)")
    w.set_defaults(func=cmd_show)
    v = asub.add_parser("waive", help="required 모드에서 이 세션의 AI 단계를 건너뛴다")
    v.add_argument("session_id")
    v.add_argument("--reason", required=True, help="건너뛰는 사유")
    v.add_argument("--url", help="DB URL (기본: DLP_DATABASE_URL)")
    v.set_defaults(func=cmd_waive)
