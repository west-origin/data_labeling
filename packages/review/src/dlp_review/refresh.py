"""열린 Label Studio 작업의 서명 URL을 다시 서명한다 (F-REV-08).

Label Studio 작업 data에는 라벨링 버킷 워터마크 영상·시계열 CSV의 서명 URL이 들어 있다 (라벨러
자격 증명 = 라벨링 버킷 읽기 전용, 유효 시간 `review.yaml media.signed_url_ttl_s`, S3 SigV4 상한
7일). 작업이 그보다 오래 열려 있으면 검수 화면이 영상을 열지 못한다. 이 모듈은 열린 작업의 URL을
같은 자격 증명·유효 시간으로 다시 서명해 작업 data를 바꾼다 (`dlp review refresh-urls`, cron으로
유효 시간보다 짧은 주기).

갱신하지 않는 작업 (서명 URL이 만료되며 노출이 줄어드는 쪽으로 둔다):
- Label Studio 작업 라벨(labeling) 작업이 아닌 것 (CVAT 작업은 영상을 CVAT에 올려 URL이 없다).
- 수집한 작업, 회수한(revoked) 작업, 회수 요청이 있는 작업.
- 매체가 낡은 작업: 렌더 해시가 지금 운영 블러와 다르다 (`dlp_privacy.runner.stale_media_tasks`).
  다시 서명하면 블러가 덜 된 영상의 노출을 늘리므로 건너뛰고 `dlp review revoke <세션>`을 안내한다.
갱신해야 하는데 못 한 작업은 오류로 남기고 다른 작업은 계속한다 (실패 쪽으로 닫는다: data를 바꾸지
않는다):
- 도구에 그 작업이 없거나 응답이 실패한다.
- 도구에 저장된 주인(data의 session_id·assignee)이 작업 행과 다르다 (도구 인스턴스를 새로 만든 뒤
  같은 번호가 다른 작업일 수 있다, `revoke`와 같은 대조).
- 매체 URI가 라벨링 버킷이 아니거나 원본 버킷이다, 라벨링 버킷에 영상·시계열 객체가 없다.
- 새 URL이 원본 버킷을 가리킨다 (`roles.check_stage_uris`).

동시성: 세션 행을 잠근 채 고르고 갱신한다 (`lock_session`). 회수 요청(블러 재검수 수집·탐지·렌더)과
작업 생성도 같은 잠금을 쓰므로, 요청이 먼저면 이 함수가 건너뛰고 나중이면 회수 실행기가 매체 객체와
도구 작업을 지운다 (다시 서명한 URL도 함께 끊긴다). 도구 호출이 잠금 안에서 돈다.

공개 이름:
- `RefreshReport`: 갱신 결과.
- `sessions_with_open_ls_tasks`: 열린 Label Studio 작업 라벨 작업이 있는 세션.
- `refresh_session_urls`: 세션 하나의 열린 작업을 다시 서명한다.

DB 트랜잭션은 호출자가 연다. DB에는 쓰지 않는다 (잠금만). 원본 버킷은 읽지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx
import sqlalchemy as sa

from dlp_media.storage import STORE_ERRORS
from dlp_privacy.runner import stale_media_tasks
from dlp_review.clients import LabelStudioClient, ToolError
from dlp_review.roles import RawAccessError, check_stage_uris
from dlp_review.tasks import (
    ReviewSetup,
    TaskError,
    label_studio_task_data,
    object_key,
    timeseries_key,
)
from dlp_schema.db.repository import list_review_tasks, lock_session
from dlp_schema.db.tables import review_tasks
from dlp_schema.review import ReviewStage, ReviewTask, ReviewTaskStatus, ReviewTool

# 작업 하나의 갱신 실패로 보는 예외 (도구 응답·연결, 전제 조건, 원본 URI, 저장소)
_TASK_ERRORS = (ToolError, httpx.HTTPError, TaskError, RawAccessError, *STORE_ERRORS)


@dataclass
class RefreshReport:
    """갱신 결과. 항목은 `"<작업 키>: <이유>"` 형식이다 (refreshed는 키만)."""

    # 다시 서명해 도구 작업 data를 바꾼 작업 키
    refreshed: list[str] = field(default_factory=list[str])
    # 일부러 갱신하지 않은 열린 작업 (낡은 매체·회수 요청)
    skipped: list[str] = field(default_factory=list[str])
    # 갱신해야 하는데 못 한 작업 (data를 바꾸지 않았다)
    errors: list[str] = field(default_factory=list[str])


def _open_ls_labeling(task: ReviewTask) -> bool:
    """갱신 후보: Label Studio 작업 라벨 작업이고 열려 있다."""
    return (
        task.tool is ReviewTool.LABEL_STUDIO
        and task.stage is ReviewStage.LABELING
        and task.status is ReviewTaskStatus.OPEN
    )


def sessions_with_open_ls_tasks(conn: sa.Connection) -> list[str]:
    """열린 Label Studio 작업 라벨 작업이 있는 세션 ID (정렬). 읽기만 한다."""
    query = (
        sa.select(review_tasks.c.session_id)
        .where(
            review_tasks.c.tool == ReviewTool.LABEL_STUDIO.value,
            review_tasks.c.stage == ReviewStage.LABELING.value,
            review_tasks.c.status == ReviewTaskStatus.OPEN.value,
        )
        .distinct()
        .order_by(review_tasks.c.session_id)
    )
    return [str(sid) for sid in conn.execute(query).scalars()]


def _refresh_one(ls: LabelStudioClient, task: ReviewTask, setup: ReviewSetup) -> None:
    """작업 하나를 다시 서명해 도구 작업 data를 바꾼다. 못 하면 `_TASK_ERRORS`의 예외."""
    if task.assignee is None:
        raise TaskError("담당자가 없는 작업")
    check_stage_uris(ReviewStage.LABELING, [task.media_uri], setup.raw.bucket)
    video = object_key(setup.labeling, task.media_uri)
    csv = timeseries_key(task.session_id)
    missing = [key for key in (video, csv) if setup.labeling.head(key) is None]
    if missing:
        raise TaskError(f"라벨링 버킷에 매체가 없습니다: {missing}")
    tid = int(task.external_id)
    owner = ls.task_owner(tid)
    if (owner.session_id, owner.assignee) != (task.session_id, task.assignee):
        raise TaskError(f"도구 작업의 주인이 다릅니다 ({owner.label})")
    ls.update_task_data(
        tid, label_studio_task_data(setup, task.session_id, task.assignee, video, csv)
    )


def refresh_session_urls(conn: sa.Connection, session_id: str, setup: ReviewSetup) -> RefreshReport:
    """세션의 열린 Label Studio 작업 라벨 작업을 다시 서명한다 (모듈 설명의 규칙).

    인자:
    - conn: DB 연결 (호출자 트랜잭션). 세션 행을 잠근다.
    - setup: 라벨링 저장소·라벨러 서명기·Label Studio 클라이언트·매체 정책(유효 시간)·프라이버시
      정책(낡은 매체 판정).

    반환: `RefreshReport`. 클라이언트가 없으면 갱신 후보마다 오류다.
    부작용: Label Studio 작업 data PATCH (외부 호출). DB에는 쓰지 않는다.
    """
    lock_session(conn, session_id)
    report = RefreshReport()
    candidates = [t for t in list_review_tasks(conn, session_id) if _open_ls_labeling(t)]
    if not candidates:
        return report
    stale = {m.task.task_key for m in stale_media_tasks(conn, session_id, setup.privacy_policy())}
    for task in candidates:
        if task.revoke_requested_at is not None:
            report.skipped.append(f"{task.task_key}: 회수 요청됨 (dlp review revoke {session_id})")
            continue
        if task.task_key in stale:
            report.skipped.append(
                f"{task.task_key}: 매체가 지금 블러와 다름 — 갱신하지 않고 회수한다 "
                f"(dlp review revoke {session_id})"
            )
            continue
        if setup.label_studio is None:
            report.errors.append(f"{task.task_key}: Label Studio 클라이언트가 없습니다")
            continue
        try:
            _refresh_one(setup.label_studio, task, setup)
        except _TASK_ERRORS as exc:
            report.errors.append(f"{task.task_key}: {exc}")
            continue
        report.refreshed.append(task.task_key)
    return report
