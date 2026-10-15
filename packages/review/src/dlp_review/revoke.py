"""회수 요청된 라벨링 작업 매체를 실제로 끊는다: 라벨링 버킷 사본 삭제 → 도구 작업 삭제 (ADR 0039).

회수 요청(outbox)은 블러를 바꾼 트랜잭션이 `review_tasks`에 남긴다 (`dlp_privacy.revocation`).
이 실행기는 그 커밋 **뒤에** 돌며, 요청된 행만 다룬다. 채널마다 멱등이고 서로 독립이다. 한 채널이
실패해도 다음 채널로 가고, 성공한 채널만 짧은 트랜잭션으로 완료를 표시한다. 남은 일은 대기로 남아
다음 실행(`dlp review revoke`, 수집·웹훅 직후, create·assign 시작, cron `--pending`)이 다시 한다.

순서는 "넓고 확실한 것부터"다:
1. 매체: 서로 다른 매체 URI마다 라벨링 버킷의 워터마크 사본을 지운다. Label Studio 작업의 서명
   URL은 이 객체를 가리키므로 객체를 지우면 바로 끊긴다. 같은 키를 쓰는 작업은 함께 표시한다.
   이어서 범위 안의 사용 중지(withdrawn) 세션마다 검수 시계열 CSV(`tasks.timeseries_key`)를 확인해
   있으면 지운다 (`timeseries_sessions`, 회수 대기 유무와 무관, 멱등): 동의를 철회한 작업자의
   장갑·IMU 값이고, 이미 발급한 Label Studio 서명 URL이 가리킨다. 사용 중지 전에 블러 변경 등으로
   모든 작업의 회수가 끝났으면 사용 중지가 새로 요청할 작업이 없어도 CSV는 남아 있다. 블러 변경
   회수는 CSV를 둔다(얼굴이 없다). CSV에는 작업 행 표시가 없어, 지우지 못하면 `timeseries_left`에
   남겨 실행을 미완료로 끝내고 다음 실행이 다시 확인한다.
2. CVAT: 작업 행 잠금(SKIP LOCKED) → 주인 대조 → 배정 해제(최선 노력) → 스냅숏(최선 노력, 이미
   있으면 두기) → 삭제(404도 성공) → 없음 확인 → 표시. CVAT는 영상을 자기 안에 올리므로 작업을
   지워야 끊긴다.
3. Label Studio: 주인 대조 → 스냅숏 → 삭제 → 없음 확인 → 표시.
노출 차단을 우선한다: 배정 해제·스냅숏 실패는 경고만 남기고 삭제를 계속한다. 지우기 전에 판정하는
것(버킷·접두사·키 모양·원본 URI·같은 키를 쓰는 살아 있는 작업·삭제 지원 저장소·도구 클라이언트·
도구 작업의 주인)이 하나라도 맞지 않으면 지우지 않고 대기로 남긴다 (실패 쪽으로 닫는다).
주인 대조: 도구에 저장된 세션·담당자(CVAT는 작업 이름의 스트림도)가 작업 행과 같아야 지운다. 도구
인스턴스를 새로 만들고 DB를 남기면 같은 외부 ID가 다른 세션·라벨러의 작업일 수 있다. 다르면 대기로
두고, 운영자가 원래 작업이 없어진 것을 확인해 `mark_tool_gone`으로 정리한다.

매체 삭제만 세션 잠금을 잡는다 (F-REV-29): 작업 생성이 같은 잠금을 쥐고 사본을 올리므로, 잠금
아래에서 "같은 키를 쓰는 살아 있는 작업이 없고 아직 표시되지 않았다"를 다시 확인한 뒤 지우고
표시한다. 잠금 밖 확인만으로는 다른 실행기가 같은 회수를 끝내 생성 게이트가 열린 사이 같은 렌더
해시로 다시 만든 작업의 새 사본(같은 키)을 지울 수 있었다. 도구 단계는 작업 행만 잠가(SKIP
LOCKED) 같은 작업의 스냅숏·삭제가 겹치지 않게 한다. "NULL인 열만 표시"는 멱등이다.
블러본 파일(`blurred_key`)은 지우지 않는다 (ADR 0024: 렌더 기록 무효로 막는다).

스냅숏: 라벨링 버킷 `sessions/<세션>/revoked/<도구>-<외부 ID>.json`. 라벨러가 진행 중이던 주석을
보존한다 (자동으로 이어받지는 않는다). 영상·서명 URL은 담지 않는다.

공개 이름:
- `RevocationReport`, `RevocationIncompleteError`: 실행 결과와 "대기가 남았다" 예외.
- `timeseries_sessions`: 시계열 CSV를 확인해 지울 사용 중지 세션.
- `snapshot_key`: 스냅숏 객체 키.
- `ToolLike`, `CvatLike`: 실행기가 쓰는 도구 클라이언트 메서드 (실제 `clients`가 만족한다).
- `execute_media_revocations`: 대기 회수를 실행한다.
- `mark_tool_gone`: 도구 인스턴스 폐기 등으로 도구 작업을 확인할 수 없을 때의 수동 완료 표시.
"""

from __future__ import annotations

import json
import re
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import httpx
import sqlalchemy as sa

from dlp_media.storage import STORE_ERRORS, DeletableStore, ObjectStore, sha256_file
from dlp_review.clients import TaskOwner, ToolError
from dlp_review.roles import RawAccessError, check_stage_uris
from dlp_review.tasks import MEDIA_HASH_CHARS, TaskError, object_key, timeseries_key
from dlp_schema.db.repository import (
    get_session,
    list_review_tasks,
    lock_session,
    mark_task_media_revoked,
    mark_task_tool_revoked,
    pending_revocations,
    withdrawn_session_ids,
)
from dlp_schema.db.tables import review_tasks
from dlp_schema.review import ReviewStage, ReviewTask, ReviewTool
from dlp_schema.session import LifecycleState

# 도구 호출 실패로 보는 예외 (응답 4xx·5xx, 연결·시간 초과)
_TOOL_ERRORS = (ToolError, httpx.HTTPError)
# 저장소 호출 실패로 보는 예외 (S3, 로컬 파일)
_STORE_ERRORS = STORE_ERRORS


class RevocationIncompleteError(RuntimeError):
    """회수 대기가 남았다 (웹훅 서버는 503으로 답한다. DB의 수집·요청은 이미 커밋됐다)."""


@dataclass
class RevocationReport:
    """회수 실행 결과. pending이 비면 이 범위의 회수가 모두 끝났다."""

    # 지운 라벨링 버킷 매체 키 (삭제 뒤 없음까지 확인)
    media_done: list[str] = field(default_factory=list[str])
    # 도구 작업이 없음을 확인한 작업 키
    tools_done: list[str] = field(default_factory=list[str])
    # 이번에 쓴 스냅숏 키
    snapshots: list[str] = field(default_factory=list[str])
    # 실패 (그 채널은 대기로 남는다). 매체 실패가 먼저 온다: Label Studio 영상 노출이 남는다
    errors: list[str] = field(default_factory=list[str])
    # 경고 (배정 해제·스냅숏 실패. 삭제는 계속했다)
    warnings: list[str] = field(default_factory=list[str])
    # 실행 뒤 DB에서 다시 읽은 대기 작업 키
    pending: list[str] = field(default_factory=list[str])
    # 지우지 못한 사용 중지 세션의 시계열 CSV 키 (작업 행 표시가 없어 pending에 들지 않는다.
    # 다음 실행이 다시 확인해 지운다)
    timeseries_left: list[str] = field(default_factory=list[str])

    @property
    def complete(self) -> bool:
        """대기가 남지 않았다 (작업 대기도, 지우지 못한 시계열 CSV도 없다)."""
        return not self.pending and not self.timeseries_left

    @property
    def remaining(self) -> list[str]:
        """남은 일: 대기 작업 키, 이어서 지우지 못한 시계열 CSV 키."""
        return [*self.pending, *self.timeseries_left]


class ToolLike(Protocol):
    """실행기가 쓰는 도구 메서드 (`CvatClient`, `LabelStudioClient`)."""

    def task_exists(self, task_id: int) -> bool:
        """작업이 있으면 True, 404면 False."""
        ...

    def task_owner(self, task_id: int) -> TaskOwner:
        """도구에 저장된 작업의 세션·스트림·담당자 (모르는 칸은 None)."""
        ...

    def task_snapshot(self, task_id: int) -> dict[str, Any]:
        """지우기 전 사본 (주석 등, 매체 URL 없음)."""
        ...

    def delete_task(self, task_id: int) -> bool:
        """지웠으면 True, 이미 없으면 False."""
        ...


class CvatLike(ToolLike, Protocol):
    """CVAT는 지우기 전에 배정을 먼저 푼다."""

    def unassign(self, task_id: int) -> None:
        """작업과 모든 job의 담당자를 비운다."""
        ...


def snapshot_key(task: ReviewTask) -> str:
    """작업 스냅숏 키: 라벨링 버킷 `sessions/<세션>/revoked/<도구>-<외부 ID>.json`.

    `review/` 밖에 두어 매체 회수 대상 키 모양과 섞이지 않는다.
    """
    return f"sessions/{task.session_id}/revoked/{task.tool.value}-{task.external_id}.json"


def _media_pattern(task: ReviewTask) -> re.Pattern[str]:
    """지워도 되는 워터마크 사본 키 모양: `sessions/<세션>/review/<담당자>/<스트림>[-<해시>].mp4`.

    해시 글자 수는 키를 만드는 쪽과 같은 `tasks.MEDIA_HASH_CHARS`다 (`tasks.labeling_media_key`).
    시계열 CSV·블러본·스냅숏은 이 모양이 아니라 지우지 않는다.
    """
    sid, stream = re.escape(task.session_id), re.escape(task.stream_id)
    digest = rf"(-[0-9a-f]{{{MEDIA_HASH_CHARS}}})?"
    return re.compile(rf"^sessions/{sid}/review/[^/]+/{stream}{digest}\.mp4$")


def _media_key(labeling: ObjectStore, raw_bucket: str, task: ReviewTask) -> str:
    """지울 매체 키. 라벨링 버킷 밖·원본 URI·키 모양 밖이면 `TaskError`/`RawAccessError`."""
    key = object_key(labeling, task.media_uri)
    check_stage_uris(ReviewStage.LABELING, [task.media_uri], raw_bucket)
    if not _media_pattern(task).match(key):
        raise TaskError(f"{task.task_key}: 회수할 수 있는 워터마크 사본 키가 아닙니다 ({key})")
    return key


def _delete_media(
    engine: sa.Engine,
    labeling: ObjectStore,
    raw_bucket: str,
    group: list[ReviewTask],
    clock: Callable[[], datetime],
) -> list[str]:
    """같은 매체 URI를 쓰는 대기 작업 묶음의 사본을 지우고 표시한다. 지운 키.

    시계열 CSV는 여기서 지우지 않는다 (`_timeseries_phase`: 대기 묶음이 없는 사용 중지 세션도 본다).

    Raises:
        TaskError, RawAccessError: 지우기 전 판정 실패 (살아 있는 작업이 같은 키를 씀 등).
        `STORE_ERRORS`: 저장소 실패.
    """
    first = group[0]
    key = _media_key(labeling, raw_bucket, first)
    if not isinstance(labeling, DeletableStore):
        raise TaskError(f"{type(labeling).__name__}는 삭제를 지원하지 않습니다")
    # 작업 생성(`tasks.create_labeling_tasks`)과 같은 세션 잠금 아래에서 다시 확인하고 지운다
    # (F-REV-29). 잠금 밖 확인은 낡을 수 있다: 다른 실행기가 같은 회수를 끝내 생성 게이트가 열리면
    # 같은 렌더 해시로 다시 만든 작업이 같은 키에 새 사본을 올린다
    with engine.begin() as conn:
        lock_session(conn, first.session_id)
        same = [
            t for t in list_review_tasks(conn, first.session_id) if t.media_uri == first.media_uri
        ]
        if not any(t.revoke_requested_at is not None and t.media_revoked_at is None for t in same):
            return []  # 다른 실행기가 이미 지우고 표시했다
        live = [t.task_key for t in same if t.revoke_requested_at is None]
        if live:
            raise TaskError(f"{key}: 회수 요청 없는 작업 {live}이 같은 사본을 써서 지우지 않습니다")
        labeling.delete(key)
        if labeling.head(key) is not None:
            raise TaskError(f"{key}: 지운 뒤에도 객체가 남아 있습니다")
        mark_task_media_revoked(conn, first.session_id, first.media_uri, clock())
    return [key]


def _media_phase(
    engine: sa.Engine,
    pending: list[ReviewTask],
    labeling: ObjectStore,
    raw_bucket: str,
    report: RevocationReport,
    clock: Callable[[], datetime],
) -> None:
    """매체 단계: 서로 다른 매체 URI마다 사본을 지운다. 실패는 그 매체만 대기로 남긴다."""
    groups: dict[tuple[str, str], list[ReviewTask]] = {}
    for t in pending:
        if t.media_revoked_at is None:
            groups.setdefault((t.session_id, t.media_uri), []).append(t)
    for group in groups.values():
        try:
            report.media_done += _delete_media(engine, labeling, raw_bucket, group, clock)
        except (TaskError, RawAccessError, *_STORE_ERRORS) as exc:
            keys = ", ".join(t.task_key for t in group)
            report.errors.append(f"[매체 노출 남음] {keys}: {exc}")


def timeseries_sessions(conn: sa.Connection, session_id: str | None) -> list[str]:
    """시계열 CSV를 확인해 지울 사용 중지 세션 (세션 ID 순).

    Label Studio 작업 행이 있는(만든 적 있는) 사용 중지 세션이다: CSV는 Label Studio 작업을 만들
    때만 올린다 (`tasks.timeseries_key`). session_id가 있으면 그 세션만, None이면 사용 중지 기록
    (`withdrawals`)이 있는 모든 세션을 본다. 회수 대기 유무와 무관하다: 사용 중지 전에 모든 작업의
    회수가 끝났으면 사용 중지가 새로 요청할 작업이 없지만 CSV는 남아 있을 수 있다 (이 변경 전에
    사용 중지한 세션도 같다).
    """
    sids = [session_id] if session_id is not None else sorted(withdrawn_session_ids(conn))
    found: list[str] = []
    for sid in sids:
        if not any(t.tool is ReviewTool.LABEL_STUDIO for t in list_review_tasks(conn, sid)):
            continue
        if get_session(conn, sid).lifecycle_state is LifecycleState.WITHDRAWN:
            found.append(sid)
    return found


def _delete_timeseries(labeling: ObjectStore, sid: str) -> bool:
    """사용 중지 세션의 시계열 CSV를 지운다. 지웠으면 True, 이미 없으면 False.

    Raises:
        TaskError: 삭제를 지원하지 않는 저장소이거나 지운 뒤에도 남아 있을 때.
        `STORE_ERRORS`: 저장소 실패.
    """
    csv = timeseries_key(sid)
    if labeling.head(csv) is None:
        return False
    if not isinstance(labeling, DeletableStore):
        raise TaskError(f"{type(labeling).__name__}는 삭제를 지원하지 않습니다")
    labeling.delete(csv)
    if labeling.head(csv) is not None:
        raise TaskError("사용 중지 세션의 시계열을 지운 뒤에도 남아 있습니다")
    return True


def _timeseries_phase(
    engine: sa.Engine, session_id: str | None, labeling: ObjectStore, report: RevocationReport
) -> None:
    """범위 안 사용 중지 세션의 시계열 CSV를 지운다 (`timeseries_sessions`, 멱등).

    지운 키는 media_done에, 실패는 errors(매체 노출 남음)와 timeseries_left에 남긴다.
    """
    with engine.connect() as conn:
        sids = timeseries_sessions(conn, session_id)
    for sid in sids:
        csv = timeseries_key(sid)
        try:
            deleted = _delete_timeseries(labeling, sid)
        except (TaskError, *_STORE_ERRORS) as exc:
            report.errors.append(f"[매체 노출 남음] {csv}: {exc}")
            report.timeseries_left.append(csv)
            continue
        if deleted:
            report.media_done.append(csv)


def _write_snapshot(
    labeling: ObjectStore, task: ReviewTask, client: ToolLike, clock: Callable[[], datetime]
) -> str | None:
    """스냅숏을 라벨링 버킷에 쓴다. 이미 있으면 덮어쓰지 않고 None (이전 실행의 사본을 지킨다)."""
    key = snapshot_key(task)
    if labeling.head(key) is not None:
        return None
    body = {
        "task_key": task.task_key, "tool": task.tool.value, "session_id": task.session_id,
        "stream_id": task.stream_id, "assignee": task.assignee,
        "assignment_id": task.assignment_id, "mode": task.mode.value,
        "revoke_reason": task.revoke_reason, "snapshot_at": clock().isoformat(),
        "tool_task": client.task_snapshot(int(task.external_id)),
    }  # fmt: skip
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "snapshot.json"
        path.write_text(json.dumps(body, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        labeling.put_file(key, path, sha256_file(path))
    return key


def _locked(conn: sa.Connection, task_key: str) -> bool:
    """도구 완료 표시 전인 회수 요청 행을 잠근다 (다른 실행기가 잡고 있으면 건너뛴다)."""
    row = conn.execute(
        sa.select(review_tasks.c.task_key)
        .where(
            review_tasks.c.task_key == task_key,
            review_tasks.c.revoke_requested_at.is_not(None),
            review_tasks.c.tool_revoked_at.is_(None),
        )
        .with_for_update(skip_locked=True)
    ).first()
    return row is not None


def _owner_mismatch(task: ReviewTask, owner: TaskOwner) -> str | None:
    """도구 작업의 주인이 작업 행과 다르면 이유 문장, 같으면 None.

    세션·담당자는 같아야 하고, 스트림은 도구가 담을 때만(CVAT) 비교한다. 모르는 칸(None)은 다르다고
    본다 (실패 쪽으로 닫는다).
    """
    same = (
        owner.session_id == task.session_id
        and owner.assignee is not None
        and owner.assignee == task.assignee
        and owner.stream_id in (None, task.stream_id)
    )
    if same:
        return None
    mine = f"{task.session_id}/{task.stream_id}/{task.assignee}"
    return (
        f"{task.task_key}: 도구 작업({owner.label})이 이 작업 행({mine})의 것이 아니라 지우지 "
        f"않습니다 (도구 인스턴스를 바꿨으면 원래 작업이 없어진 것을 확인한 뒤 "
        f"dlp review revoke --mark-tool-gone {task.task_key} --note <사유>)"
    )


def _revoke_tool_task(
    engine: sa.Engine,
    task: ReviewTask,
    client: ToolLike,
    unassign: Callable[[int], None] | None,
    labeling: ObjectStore,
    report: RevocationReport,
    clock: Callable[[], datetime],
) -> None:
    """도구 작업 하나: 잠금 → (있으면) 주인 대조 → 배정 해제·스냅숏 → 삭제 → 없음 확인 → 표시.

    주인이 다르면 아무것도 바꾸지 않고 errors에 남긴다 (대기).
    """
    tid = int(task.external_id)
    with engine.begin() as conn:
        if not _locked(conn, task.task_key):
            return
        if client.task_exists(tid):
            mismatch = _owner_mismatch(task, client.task_owner(tid))
            if mismatch is not None:
                report.errors.append(mismatch)
                return
            if unassign is not None:
                try:
                    unassign(tid)
                except _TOOL_ERRORS as exc:
                    report.warnings.append(f"{task.task_key}: 배정 해제 실패 ({exc})")
            try:
                written = _write_snapshot(labeling, task, client, clock)
            except (*_TOOL_ERRORS, *_STORE_ERRORS) as exc:
                report.warnings.append(f"{task.task_key}: 스냅숏 실패, 삭제는 계속 ({exc})")
            else:
                report.snapshots += [written] if written else []
            client.delete_task(tid)
            if client.task_exists(tid):
                report.errors.append(f"{task.task_key}: 지운 뒤에도 도구 작업이 남아 있습니다")
                return
        mark_task_tool_revoked(conn, task.task_key, clock())
    report.tools_done.append(task.task_key)


def _tool_phase(
    engine: sa.Engine,
    pending: list[ReviewTask],
    tool: ReviewTool,
    client: ToolLike | None,
    unassign: Callable[[int], None] | None,
    labeling: ObjectStore,
    report: RevocationReport,
    clock: Callable[[], datetime],
) -> None:
    """도구 단계: 그 도구의 대기 작업마다 지운다. 실패는 그 작업만 대기로 남긴다."""
    for task in (t for t in pending if t.tool is tool and t.tool_revoked_at is None):
        if client is None:
            report.errors.append(f"{task.task_key}: {tool.value} 클라이언트 없음")
            continue
        try:
            _revoke_tool_task(engine, task, client, unassign, labeling, report, clock)
        except _TOOL_ERRORS as exc:
            report.errors.append(f"{task.task_key}: 도구 작업 삭제 실패 ({exc})")


def execute_media_revocations(
    engine: sa.Engine,
    session_id: str | None,
    *,
    labeling: ObjectStore,
    raw_bucket: str,
    cvat: CvatLike | None,
    label_studio: ToolLike | None,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> RevocationReport:
    """세션(None이면 모든 세션)의 회수 대기를 실행한다. 커밋된 요청만 다룬다.

    Args:
        engine: DB 엔진 (단계마다 짧은 트랜잭션을 연다. 호출자 트랜잭션 안에서 부르지 않는다).
        labeling: 라벨링 버킷 저장소 (서비스 자격 증명, `DeletableStore`여야 지운다).
        raw_bucket: 원본 버킷 이름. 매체 URI가 원본을 가리키거나 labeling이 원본 버킷이면 거부한다.
        cvat, label_studio: 도구 클라이언트. 없으면 그 도구 작업은 대기로 남는다.
        clock: 완료 표시 시각 (시간대 필수).

    Returns:
        `RevocationReport`. 예외로 끝내지 않는다 (실패는 errors, 남은 일은 pending·
        timeseries_left).

    부작용: 라벨링 버킷 사본·사용 중지 세션 시계열 CSV 삭제(대기가 없어도 `timeseries_sessions`의
    CSV는 확인한다)·스냅숏 쓰기, CVAT·Label Studio 작업 배정 해제·삭제, `review_tasks` 완료 표시.
    외부 삭제는 되돌릴 수 없다.
    """
    report = RevocationReport()
    with engine.connect() as conn:
        pending = pending_revocations(conn, session_id)
    if labeling.bucket == raw_bucket:
        report.errors.append(f"라벨링 저장소가 원본 버킷({raw_bucket})입니다 (설정 오류)")
    else:
        _media_phase(engine, pending, labeling, raw_bucket, report, clock)
        _timeseries_phase(engine, session_id, labeling, report)
        unassign = cvat.unassign if cvat is not None else None
        _tool_phase(engine, pending, ReviewTool.CVAT, cvat, unassign, labeling, report, clock)
        _tool_phase(
            engine, pending, ReviewTool.LABEL_STUDIO, label_studio, None, labeling, report, clock
        )
    with engine.connect() as conn:
        report.pending = [t.task_key for t in pending_revocations(conn, session_id)]
    return report


def mark_tool_gone(
    conn: sa.Connection, task_key: str, *, note: str, actor: str, now: datetime
) -> bool:
    """도구 작업을 확인할 수 없을 때(도구 인스턴스 폐기, 관리자 수동 삭제) 수동으로 완료 표시한다.

    사유는 필수이며 `"<actor>: <note>"`로 `revoke_note`에 남는다. 매체 사본 회수는 따로 끝나야 한다.

    Returns:
        표시했으면 True (회수 요청이 없거나 이미 도구 완료 표시가 있으면 False).

    Raises:
        ValueError: note나 actor가 비었을 때, now에 시간대가 없을 때.
    """
    if not note.strip() or not actor.strip():
        raise ValueError("수동 완료 표시에는 행위자와 사유가 필요합니다")
    return mark_task_tool_revoked(conn, task_key, now, note=f"{actor}: {note.strip()}")
