"""AI 1차 검수 실행·상태 하위 명령 (WP17.5, ADR 0040 5.2~5.5). `aireview_cmds.add_commands`가
등록한다.

등록하는 명령
- `dlp aireview run <세션>… | --pending [--limit N] [--parallel-sessions K] [--vlm-url URL …]
  [--store s3] [--groups …] [--mode all|flagged] [--max-minutes M] [--no-sensor] [--dry-run]
  [--debug-dir DIR] [--json-progress] [--url]`: 계획 → 판정 → AI 표 기록 (`dlp_aireview.runner`).
  멱등이고 재개된다 (기록된 단위는 다시 묻지 않는다). `--dry-run`은 계획만 센다 (서버 불필요,
  블러본 게이트는 거친다). 끝나면 세션마다 실행 요약 JSON 한 줄을 표준 출력에, 진행 줄은 표준
  오류에 쓴다 (`--json-progress`면 JSON Lines).
  종료 코드
  - 0: 완료, 할 일 없음, 계획만.
  - 2: 서버 주소 없음 (아무것도 쓰지 않음).
  - 3: 서버 장애로 중단 (완료한 단위는 남는다. 시작 전 모든 엔드포인트가 응답하지 않으면 아무것도
    쓰지 않음).
  - 5: 설정·라이선스 불일치 (4xx, 서버 모델이 정책 모델과 다름, 사내 주소 아님, 원본 버킷 저장소).
  - 6: 세션 전제 실패 (승인 전, 승인 보류, 사용 중지, 렌더 게이트, 장갑 Parquet 없음) 또는 실행 중
    블러 변경.
  - 7: 다른 실행기가 그 세션을 처리 중.
  - 8: `--max-minutes` 도달.
  세션이 여럿이면 가장 심각한 코드(5 > 3 > 6 > 7 > 8)를 낸다. 5·3·8이 나면 남은 세션은 시작하지
  않는다. `--pending`은 전제가 안 맞는 세션(6)과 처리 중인 세션(7)을 목록으로 알리고 0으로 센다.
  4는 회수 대기(`dlp review revoke`)의 코드라 쓰지 않는다.
- `dlp aireview status [<세션>… | --pending | --all] [--groups …] [--json] [--url]`: 세션별
  판정 버전, 판정률(판정된 대상/계획 대상), 단위·fallback 수, 판정 분포, 마지막 이벤트, 최근
  처리량 기반 남은 시간, 백로그(영상 시간). started만 있고 실행 중이 아니면 "중단 추정"이다.
  종료 0.
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
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
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
from dlp_aireview.runner import RunOptions, RunSummary, SessionNotReadyError, run_from_db
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
from dlp_schema.db.locks import try_lock_aireview, unlock_aireview
from dlp_schema.db.repository import get_labels, get_session, list_session_ids
from dlp_schema.episode import current_labels
from dlp_schema.labels import Hand
from dlp_schema.session import LifecycleState, Session, StreamKind, SyncMethod
from dlp_sync.policy import load_policy as load_sync_policy
from dlp_sync.signals import glove_series

SESSION = 6
LOCKED = 7
MAX_MINUTES = 8
# 여러 세션의 종료 코드를 하나로 (앞일수록 심각하다)
_SEVERITY = (CONFIG, UNAVAILABLE, 130, 1, SESSION, LOCKED, MAX_MINUTES)
# 이 코드가 나면 남은 세션을 시작하지 않는다 (서버·설정·시간 문제는 다음 세션도 같다)
_STOP = frozenset({CONFIG, UNAVAILABLE, MAX_MINUTES, 130})
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


def _open_pool(args: argparse.Namespace, policy: AiReviewPolicy) -> EndpointPool | int:
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
    """세션 하나 (모듈 docstring의 종료 코드). 부작용: AI 표 기록, VLM 호출, 라벨링 버킷 읽기."""
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
    if summary is None:
        print(f"[처리 중] {sid}: 다른 실행기가 이 세션을 처리하고 있습니다")
        return LOCKED
    _print_summary(summary)
    return summary.exit_code


def _worst(codes: Sequence[int], pending: bool) -> int:
    """여러 세션의 종료 코드 → 하나 (`--pending`은 6·7을 0으로 센다)."""
    seen = {c for c in codes if not (pending and c in (SESSION, LOCKED))}
    return next((c for c in _SEVERITY if c in seen), 0)


def _run_all(sessions: Sequence[str], ctx: _RunCtx, parallel: int) -> list[int]:
    """세션들을 차례로(또는 K개씩 함께) 돌린다. 5·3·8이 나면 남은 세션은 시작하지 않는다."""
    stop = threading.Event()
    codes: list[int] = []
    lock = threading.Lock()

    def one(sid: str) -> None:
        if stop.is_set():
            print(f"[건너뜀] {sid}: 앞 세션에서 실행을 멈췄습니다")
            return
        code = _run_one(sid, ctx)
        with lock:
            codes.append(code)
        if code in _STOP:
            stop.set()

    if parallel <= 1:
        for sid in sessions:
            one(sid)
        return codes
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        list(pool.map(one, sessions))
    return codes


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
    """판정할 세션: 지정한 세션, 또는 `--pending` 조건을 만족하는 세션 (`--limit`개까지)."""
    if not args.pending:
        return list(args.sessions)
    with engine.connect() as conn:
        found = pending_sessions(
            conn, policy, load_privacy_policy(repo_root()), datetime.now(UTC),
            groups=groups, flagged_for=flagged_spans,
        )  # fmt: skip
    picked = found[: args.limit] if args.limit else found
    names = ", ".join(p.session_id for p in picked)
    print(f"[pending] 판정할 세션 {len(found)}개 중 {len(picked)}개: {names}")
    return [p.session_id for p in picked]


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
        opened = _open_pool(args, policy)
        if isinstance(opened, int):
            return opened
        backend = opened
    root = repo_root()
    buckets = load_config(root / "config/defaults.yaml").buckets
    engine = sa.create_engine(database_url(args.url))
    try:
        options = _options(args, policy, backend)
        sessions = _sessions(args, engine, policy, options.groups)
        ctx = _RunCtx(
            engine=engine, labeling=store_from_spec(args.store, buckets.labeling),
            raw_bucket=buckets.raw, backend=backend, policy=policy, options=options,
            sensor_for=_sensor_source(args),
        )  # fmt: skip
        codes = _run_all(sessions, ctx, args.parallel_sessions)
    finally:
        engine.dispose()
    return _worst(codes, args.pending)


# ---------------------------------------------------------------- status


def _running(engine: sa.Engine, session_id: str) -> bool | None:
    """실행기가 이 세션의 잠금을 쥐고 있는가 (PostgreSQL만, 아니면 None)."""
    if engine.dialect.name != "postgresql":
        return None
    with engine.connect() as conn:
        got = try_lock_aireview(conn, session_id)
        if got:
            unlock_aireview(conn, session_id)
        conn.commit()
    return not got


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
    """`dlp aireview status` (모듈 docstring). 부작용: 없음 (DB 읽기, 잠금 확인)."""
    policy = load_policy(repo_root())
    privacy = load_privacy_policy(repo_root())
    groups = frozenset(args.groups) if args.groups else None
    engine = sa.create_engine(database_url(args.url))
    try:
        with engine.connect() as conn:
            ids = _status_ids(args, conn, policy)
            rows = [
                session_status(
                    conn,
                    sid,
                    policy,
                    privacy,
                    groups=groups,
                    flagged_for=flagged_spans,
                    running=_running(engine, sid),
                )
                for sid in ids
            ]
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


def _run_parser(asub: Any) -> None:
    """`aireview run` 인자."""
    r = asub.add_parser("run", help="판정 실행 (멱등·재개, 참고용 판정만 AI 표에 기록)")
    r.add_argument("sessions", nargs="*")
    r.add_argument("--pending", action="store_true", help="판정할 대상이 남은 prelabeled 세션")
    r.add_argument("--limit", type=int, help="--pending에서 고를 세션 수 상한")
    r.add_argument("--parallel-sessions", type=int, default=1, help="함께 돌릴 세션 수")
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
