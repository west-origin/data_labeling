"""DB 기록 없이 남은 검수 도구 작업·라벨링 버킷 사본(고아)을 찾고 지운다 (F-REV-15, R4, ADR 0039).

작업 생성(`tasks.create_labeling_tasks`·`create_privacy_tasks`)은 워터마크 사본을 라벨링 버킷에
올리고 CVAT·Label Studio 작업을 만든 **뒤** DB `review_tasks`에 삽입한다. 그 트랜잭션이 롤백되면
도구 작업과 사본은 DB 기록 없이 남고, DB 기준 회수(`revoke.execute_media_revocations`)는 이를 찾지
못한다 (CVAT 작업은 담당자에게 배정된 채 남을 수 있다). 이 모듈은 도구·버킷을 나열해 DB와 대조한다.
진입점: `dlp review revoke --orphans [<세션>] [--apply]` (기본은 보고만 한다).

대상 (이 플랫폼이 만든 이름 규칙에 맞는 것만, 다른 시스템의 것은 건드리지 않는다):
- 라벨링 버킷: DB 세션마다 `sessions/<세션>/review/` 아래 워터마크 사본 키
  (`tasks.parse_labeling_media_key`). 시계열 CSV·블러본·회수 스냅숏은 보지 않는다.
- CVAT: 이름이 `dlp-privacy`·`dlp-labeling-spatial` 프로젝트
  (`ontology_projects.is_ontology_project`, `-demo`처럼 다른 글자가 붙은 이름은 아님)의 작업 중
  이름이 `<세션>/<스트림>/<담당자>`(`clients.cvat_task_name`)인 것.
- Label Studio: `dlp-labeling-temporal` 프로젝트의 작업 중 data에 `session_id`·`assignee`가 있는 것.
- 세션이 이 DB에 없으면 다른 배포(같은 도구를 쓰는 다른 DB)의 것일 수 있어 보지 않는다 (경고에 수만
  남긴다).

판정 (`Reason`):
- `no_db_row`: 그 키(`cvat:<ID>`·`label_studio:<ID>`, 매체 URI)의 작업 행이 없다.
- `other_owner`: (도구) 같은 키의 행이 있지만 세션·스트림·담당자가 다르다. 그 행은 다른 작업(도구
  인스턴스를 바꾸기 전)의 것이고 키가 유일하므로 이 도구 작업에는 행이 없다.
- `revoked`: 그 키를 쓰는 행이 모두 회수 요청됐고 그 채널의 완료 표시(`media_revoked_at`·
  `tool_revoked_at`)까지 있는데 남아 있다 (백업 복원, 잘못된 `--mark-tool-gone` 등).
회수 요청만 있고 완료 표시가 없는 것은 실행기 몫이라 고아로 보지 않는다. 살아 있는 행(회수 요청
없음)이 하나라도 쓰면 남긴다.

생성 직후 경합 (DB 커밋 전 작업):
1. 도구·저장소가 기록한 생성 시각에서 정책 `review.yaml orphans.min_age_s`가 지난 것만 후보다
   (생성 시각을 모르면 후보가 아니다).
2. 지우기 전에 세션 행을 SKIP LOCKED로 잠근다 (`try_lock_session`). 작업 생성은 같은 행을 먼저
   잠그므로, 잠그지 못하면(생성 중) 그 세션을 건너뛴다. 잠근 뒤 DB를 다시 읽어 판정이 그대로일 때만
   지우고, 지우는 동안 잠금을 유지해 새 생성이 같은 키에 사본을 올리지 못하게 한다.

삭제는 실행기와 같은 순서·방식이다: 매체(삭제 → 없음 확인) → CVAT(배정 해제·스냅숏은 최선 노력 →
삭제 → 없음 확인) → Label Studio(스냅숏 → 삭제 → 없음 확인). 스냅숏은 라벨링 버킷
`sessions/<세션>/revoked/orphan-<도구>-<ID>.json`(이미 있으면 두기)이다. DB에는 쓰지 않는다 (지울
행이 없다). 지운 것은 되돌릴 수 없다.

공개 이름:
- `Reason`, `Orphan`, `OrphanReport`: 판정, 고아 하나, 찾기·지우기 결과.
- `OrphanTool`, `OrphanCvat`: 쓰는 도구 메서드 (실제 `clients`가 만족한다).
- `tool_reason`, `media_reason`: 판정 규칙 (DB 행만 보고 판정한다).
- `orphan_snapshot_key`: 고아 도구 작업 스냅숏 키.
- `find_orphans`: 나열·대조 (읽기만). `delete_orphans`: 찾은 고아를 다시 확인하고 지운다.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Protocol

import httpx
import sqlalchemy as sa

from dlp_media.storage import (
    STORE_ERRORS,
    DeletableStore,
    ListableStore,
    ObjectStore,
    sha256_file,
)
from dlp_review.clients import TaskOwner, ToolError, ToolTask
from dlp_review.ontology_projects import is_ontology_project
from dlp_review.tasks import (
    PRIVACY_PROJECT,
    SPATIAL_PROJECT,
    TEMPORAL_PROJECT,
    parse_labeling_media_key,
)
from dlp_schema.db.repository import list_review_tasks, list_session_ids, try_lock_session
from dlp_schema.review import ReviewStage, ReviewTask

# 채널: 라벨링 버킷 사본, CVAT 작업, Label Studio 작업 (지우는 순서)
Channel = Literal["media", "cvat", "label_studio"]
# 고아로 본 이유 (모듈 설명 참조)
Reason = Literal["no_db_row", "other_owner", "revoked"]
_ORDER: dict[str, int] = {"media": 0, "cvat": 1, "label_studio": 2}
# 도구별로 훑는 프로젝트 기본 이름 (`ontology_projects.ontology_project`의 base)
_PROJECTS: dict[str, tuple[str, ...]] = {
    "cvat": (PRIVACY_PROJECT, SPATIAL_PROJECT),
    "label_studio": (TEMPORAL_PROJECT,),
}
# 도구 호출 실패로 보는 예외 (응답 4xx·5xx, 연결·시간 초과)
_TOOL_ERRORS = (ToolError, httpx.HTTPError)


class OrphanError(RuntimeError):
    """지운 뒤에도 남아 있다 등, 고아 하나를 지우지 못했다 (보고서 errors에 남는다)."""


@dataclass(frozen=True)
class Orphan:
    """고아 하나. ref는 매체 키(`sessions/…`) 또는 작업 키(`cvat:<ID>`·`label_studio:<ID>`)."""

    channel: Channel
    ref: str
    session_id: str
    reason: Reason
    # 도구·저장소가 기록한 생성(매체는 마지막 수정) 시각. 모르면 None (후보가 아니다)
    created_at: datetime | None
    # 출력용: 매체 키 또는 도구 작업 이름·data 요약
    label: str
    # 도구 작업의 주인 (지우기 전 다시 판정할 때 쓴다). 매체는 None
    owner: TaskOwner | None = None


@dataclass
class OrphanReport:
    """찾기·지우기 결과. clean이면 이 범위에 고아가 없고 모든 채널을 확인했다."""

    # 나이 하한을 넘긴 고아 (지우기 대상)
    found: list[Orphan] = field(default_factory=list[Orphan])
    # 나이 하한보다 어리거나 생성 시각을 몰라 이번에 보지 않은 고아 후보
    young: list[Orphan] = field(default_factory=list[Orphan])
    # 지운 것 (없음까지 확인한 ref)
    deleted: list[str] = field(default_factory=list[str])
    # 이번에 쓴 스냅숏 키
    snapshots: list[str] = field(default_factory=list[str])
    # 세션 행을 잠그지 못해(작업 생성 중) 건너뛴 ref
    busy: list[str] = field(default_factory=list[str])
    # 지우지 못한 이유
    errors: list[str] = field(default_factory=list[str])
    # 경고 (다른 배포의 작업 수, 다시 읽으니 DB 기록이 생긴 것, 배정 해제·스냅숏 실패 등)
    warnings: list[str] = field(default_factory=list[str])
    # 확인하지 못한 채널과 이유 (클라이언트 없음, 목록 실패, 나열을 지원하지 않는 저장소)
    unscanned: list[str] = field(default_factory=list[str])

    @property
    def remaining(self) -> list[str]:
        """찾았지만 지우지 않은 고아의 ref (보고만 했거나 실패·건너뜀)."""
        done = set(self.deleted)
        return [o.ref for o in self.found if o.ref not in done]

    @property
    def clean(self) -> bool:
        """남은 고아·실패·건너뜀·확인하지 못한 채널이 없다."""
        return not (self.remaining or self.errors or self.busy or self.unscanned)


class OrphanTool(Protocol):
    """고아 찾기·지우기에 쓰는 도구 메서드 (`CvatClient`, `LabelStudioClient`)."""

    def list_tasks(self, project: Callable[[str], bool]) -> list[ToolTask]:
        """이름이 project → True인 프로젝트들의 작업 전부."""
        ...

    def task_exists(self, task_id: int) -> bool:
        """작업이 있으면 True, 404면 False."""
        ...

    def task_snapshot(self, task_id: int) -> dict[str, Any]:
        """지우기 전 사본 (주석 등, 매체 URL 없음)."""
        ...

    def delete_task(self, task_id: int) -> bool:
        """지웠으면 True, 이미 없으면 False."""
        ...


class OrphanCvat(OrphanTool, Protocol):
    """CVAT는 지우기 전에 배정을 먼저 푼다 (라벨러 화면에서 먼저 뺀다)."""

    def unassign(self, task_id: int) -> None:
        """작업과 모든 job의 담당자를 비운다."""
        ...


def _owns(row: ReviewTask, owner: TaskOwner) -> bool:
    """도구 작업의 주인이 이 행과 같다. 블러 검수 작업 이름의 담당자 자리는 `privacy`다.

    스트림은 도구가 담을 때만(CVAT) 비교한다 (Label Studio는 None).
    """
    who = "privacy" if row.stage is ReviewStage.PRIVACY else row.assignee
    return (
        owner.session_id == row.session_id
        and owner.assignee == who
        and owner.stream_id in (None, row.stream_id)
    )


def tool_reason(row: ReviewTask | None, owner: TaskOwner) -> Reason | None:
    """도구 작업 하나의 판정. row는 같은 작업 키의 DB 행 (없으면 None). 고아가 아니면 None."""
    if row is None:
        return "no_db_row"
    if not _owns(row, owner):
        return "other_owner"
    if row.revoke_requested_at is not None and row.tool_revoked_at is not None:
        return "revoked"
    return None


def media_reason(rows: Sequence[ReviewTask]) -> Reason | None:
    """라벨링 버킷 사본 하나의 판정. rows는 그 매체 URI를 쓰는 DB 행. 고아가 아니면 None."""
    if not rows:
        return "no_db_row"
    if all(r.revoke_requested_at is not None and r.media_revoked_at is not None for r in rows):
        return "revoked"
    return None


def orphan_snapshot_key(session_id: str, channel: str, task_id: int) -> str:
    """고아 도구 작업 스냅숏 키 (라벨링 버킷, `review/` 밖). 회수 스냅숏과 이름이 겹치지 않는다."""
    return f"sessions/{session_id}/revoked/orphan-{channel}-{task_id}.json"


def _platform_project(channel: str) -> Callable[[str], bool]:
    """그 도구에서 이 플랫폼이 만든 프로젝트 이름인지 보는 함수."""
    bases = _PROJECTS[channel]
    return lambda name: any(is_ontology_project(name, b) for b in bases)


def _ours(channel: str, owner: TaskOwner) -> bool:
    """작업 이름·data가 이 플랫폼 규칙인지 (CVAT는 세 칸 모두, LS는 세션·담당자)."""
    if owner.session_id is None or owner.assignee is None:
        return False
    return channel != "cvat" or owner.stream_id is not None


def _scan_media(
    labeling: ObjectStore,
    sessions: Iterable[str],
    rows: dict[str, list[ReviewTask]],
    report: OrphanReport,
) -> list[Orphan]:
    """라벨링 버킷의 세션별 `review/` 아래 워터마크 사본을 DB 매체 URI와 대조한다."""
    if not isinstance(labeling, ListableStore):
        report.unscanned.append(f"매체: {type(labeling).__name__}는 나열을 지원하지 않습니다")
        return []
    out: list[Orphan] = []
    for sid in sessions:
        try:
            listed = labeling.list_objects(f"sessions/{sid}/review/")
        except STORE_ERRORS as exc:
            report.unscanned.append(f"매체 {sid}: 나열 실패 ({exc})")
            continue
        for obj in listed:
            parsed = parse_labeling_media_key(obj.key)
            if parsed is None or parsed[0] != sid:
                continue
            uri = labeling.uri(obj.key)
            reason = media_reason([r for r in rows[sid] if r.media_uri == uri])
            if reason is not None:
                out.append(Orphan("media", obj.key, sid, reason, obj.modified_at, obj.key))
    return out


def _scan_tool(
    channel: Literal["cvat", "label_studio"],
    client: OrphanTool | None,
    scope: set[str],
    known: set[str],
    by_key: dict[str, ReviewTask],
    report: OrphanReport,
) -> list[Orphan]:
    """도구의 플랫폼 프로젝트 작업을 DB 작업 키와 대조한다. scope 밖 세션은 건너뛴다.

    known(이 DB의 모든 세션)에 없는 세션의 작업은 다른 배포의 것일 수 있어 수만 경고로 남긴다.
    """
    if client is None:
        report.unscanned.append(f"{channel}: 클라이언트 없음")
        return []
    try:
        listed = client.list_tasks(_platform_project(channel))
    except _TOOL_ERRORS as exc:
        report.unscanned.append(f"{channel}: 작업 목록 실패 ({exc})")
        return []
    out: list[Orphan] = []
    foreign = 0
    for t in listed:
        sid = t.owner.session_id
        if not _ours(channel, t.owner) or sid is None:
            continue
        if sid not in known:
            foreign += 1
            continue
        if sid not in scope:
            continue
        key = f"{channel}:{t.task_id}"
        reason = tool_reason(by_key.get(key), t.owner)
        if reason is not None:
            out.append(Orphan(channel, key, sid, reason, t.created_at, t.owner.label, t.owner))
    if foreign:
        report.warnings.append(
            f"{channel}: 이 DB에 없는 세션의 작업 {foreign}개는 다른 배포의 것일 수 있어 보지 "
            "않았습니다"
        )
    return out


def find_orphans(
    engine: sa.Engine,
    *,
    labeling: ObjectStore,
    cvat: OrphanTool | None,
    label_studio: OrphanTool | None,
    min_age: timedelta,
    now: datetime,
    session_id: str | None = None,
) -> OrphanReport:
    """고아를 찾는다 (읽기만, DB·도구·버킷 어디에도 쓰지 않는다).

    Args:
        engine: DB 엔진 (짧은 읽기 연결만 연다).
        labeling: 라벨링 버킷 저장소 (`ListableStore`가 아니면 매체 채널을 확인하지 못한다).
        cvat, label_studio: 도구 클라이언트. 없으면 그 채널은 `unscanned`.
        min_age: 생성 시각에서 이만큼 지나야 후보다 (`review.yaml orphans.min_age_s`).
        now: 기준 시각 (시간대 필수).
        session_id: 이 세션만 본다 (None이면 DB의 모든 세션).

    Returns:
        `OrphanReport` (found는 채널 순·ref 순, 어린 후보는 young).

    Raises:
        ValueError: now에 시간대가 없을 때.
    """
    if now.utcoffset() is None:
        raise ValueError("기준 시각(now)에는 시간대가 있어야 합니다")
    report = OrphanReport()
    with engine.connect() as conn:
        known = set(list_session_ids(conn))
        scope = known if session_id is None else known & {session_id}
        rows = {sid: list_review_tasks(conn, sid) for sid in sorted(scope)}
    by_key = {r.task_key: r for rs in rows.values() for r in rs}
    candidates = [
        *_scan_media(labeling, sorted(scope), rows, report),
        *_scan_tool("cvat", cvat, scope, known, by_key, report),
        *_scan_tool("label_studio", label_studio, scope, known, by_key, report),
    ]
    cutoff = now - min_age
    for o in sorted(candidates, key=lambda o: (_ORDER[o.channel], o.ref)):
        old = o.created_at is not None and o.created_at <= cutoff
        (report.found if old else report.young).append(o)
    return report


def _recheck(o: Orphan, rows: list[ReviewTask], labeling: ObjectStore) -> Reason | None:
    """세션을 잠근 뒤 다시 읽은 행으로 같은 판정을 한다 (그 사이 커밋된 생성이면 None)."""
    if o.channel == "media":
        uri = labeling.uri(o.ref)
        return media_reason([r for r in rows if r.media_uri == uri])
    assert o.owner is not None
    return tool_reason(next((r for r in rows if r.task_key == o.ref), None), o.owner)


def _delete_media(labeling: ObjectStore, key: str) -> None:
    """사본을 지우고 없어졌는지 확인한다. 실패면 `OrphanError` 또는 저장소 예외."""
    if not isinstance(labeling, DeletableStore):
        raise OrphanError(f"{type(labeling).__name__}는 삭제를 지원하지 않습니다")
    labeling.delete(key)
    if labeling.head(key) is not None:
        raise OrphanError(f"{key}: 지운 뒤에도 객체가 남아 있습니다")


def _snapshot(
    labeling: ObjectStore, o: Orphan, client: OrphanTool, tid: int, now: datetime
) -> str | None:
    """고아 도구 작업 스냅숏을 쓴다. 이미 있으면 덮어쓰지 않고 None."""
    key = orphan_snapshot_key(o.session_id, o.channel, tid)
    if labeling.head(key) is not None:
        return None
    body = {
        "ref": o.ref, "session_id": o.session_id, "reason": o.reason, "label": o.label,
        "snapshot_at": now.isoformat(), "tool_task": client.task_snapshot(tid),
    }  # fmt: skip
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "snapshot.json"
        path.write_text(json.dumps(body, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        labeling.put_file(key, path, sha256_file(path))
    return key


def _delete_tool(
    o: Orphan,
    client: OrphanTool,
    unassign: Callable[[int], None] | None,
    labeling: ObjectStore,
    report: OrphanReport,
    now: datetime,
) -> None:
    """도구 작업 하나: (있으면) 배정 해제·스냅숏(최선 노력) → 삭제 → 없음 확인."""
    tid = int(o.ref.split(":", 1)[1])
    if not client.task_exists(tid):
        return
    if unassign is not None:
        try:
            unassign(tid)
        except _TOOL_ERRORS as exc:
            report.warnings.append(f"{o.ref}: 배정 해제 실패 ({exc})")
    try:
        written = _snapshot(labeling, o, client, tid, now)
    except (*_TOOL_ERRORS, *STORE_ERRORS) as exc:
        report.warnings.append(f"{o.ref}: 스냅숏 실패, 삭제는 계속 ({exc})")
    else:
        report.snapshots += [written] if written else []
    client.delete_task(tid)
    if client.task_exists(tid):
        raise OrphanError(f"{o.ref}: 지운 뒤에도 도구 작업이 남아 있습니다")


def _delete_one(
    o: Orphan,
    labeling: ObjectStore,
    cvat: OrphanCvat | None,
    label_studio: OrphanTool | None,
    report: OrphanReport,
    now: datetime,
) -> None:
    """채널에 맞게 지운다. 실패는 errors에 남기고 다음으로 간다."""
    try:
        if o.channel == "media":
            _delete_media(labeling, o.ref)
        else:
            client = cvat if o.channel == "cvat" else label_studio
            if client is None:
                raise OrphanError(f"{o.channel} 클라이언트 없음")
            unassign = cvat.unassign if cvat is not None and o.channel == "cvat" else None
            _delete_tool(o, client, unassign, labeling, report, now)
    except (OrphanError, *_TOOL_ERRORS, *STORE_ERRORS) as exc:
        report.errors.append(f"{o.ref}: {exc}")
        return
    report.deleted.append(o.ref)


def delete_orphans(
    engine: sa.Engine,
    report: OrphanReport,
    *,
    labeling: ObjectStore,
    raw_bucket: str,
    cvat: OrphanCvat | None,
    label_studio: OrphanTool | None,
    now: datetime,
) -> None:
    """`find_orphans`가 찾은 고아(found)를 세션마다 다시 확인하고 지운다. report를 채운다.

    세션마다 한 트랜잭션: 세션 행을 SKIP LOCKED로 잠그고(못 잠그면 busy) 행을 다시 읽어 판정이
    그대로인 것만 매체 → CVAT → Label Studio 순으로 지운다 (found가 그 순서다). 다시 읽으니 행이
    생긴 것은 found에서 빼고 경고로 남긴다. 잠금은 그 세션을 다 지울 때까지 유지한다. DB에는 쓰지
    않는다. 어린 후보(young)는 지우지 않는다.

    Args:
        raw_bucket: 원본 버킷 이름. labeling이 원본 버킷이면 아무것도 지우지 않는다 (설정 오류).

    부작용: 라벨링 버킷 사본 삭제·스냅숏 쓰기, CVAT 배정 해제·작업 삭제, Label Studio 작업 삭제.
    되돌릴 수 없다.
    """
    if labeling.bucket == raw_bucket:
        report.errors.append(f"라벨링 저장소가 원본 버킷({raw_bucket})입니다 (설정 오류)")
        return
    sessions: dict[str, list[Orphan]] = {}
    for o in report.found:
        sessions.setdefault(o.session_id, []).append(o)
    for sid, items in sorted(sessions.items()):
        with engine.begin() as conn:
            if not try_lock_session(conn, sid):
                report.busy += [o.ref for o in items]
                continue
            rows = list_review_tasks(conn, sid)
            for o in items:
                if _recheck(o, rows, labeling) is None:
                    # 나열한 뒤 생성이 커밋됐다: 고아가 아니다 (남은 것으로 세지 않는다)
                    report.found.remove(o)
                    report.warnings.append(f"{o.ref}: 다시 읽으니 DB 기록이 있어 지우지 않았습니다")
                    continue
                _delete_one(o, labeling, cvat, label_studio, report, now)
