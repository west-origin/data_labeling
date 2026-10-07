"""검수 결과 수집.

도구에서 결과를 받아 reconcile하고 DB에 이력으로 쓴다. 한 작업은 한 번만 수집한다
(작업 행을 SELECT … FOR UPDATE로 잠그고 상태를 본다. 동시에 수집해도 두 번째는 None).

- 수집하는 검수자는 작업 담당자여야 한다 (웹훅의 도구 사용자 ID가 아니라 작업의 담당자로 기록).
- 작업을 보낸 뒤 다른 단계가 지웠거나(retracted) 다른 작업이 먼저 고친(자식 레코드가 생긴) 라벨은
  검수 결과로 되살리거나 두 갈래 이력을 만들지 않는다.
- CVAT는 트랙과 모양(Shape 모드, CVAT 기본) 주석을 모두 받는다. 옮길 수 없는 주석(태그, 다각형 등)이
  있으면 수집하지 않는다 (TaskError).
- 운영 블러 라벨이 바뀌면 프라이버시 승인을 풀어 다시 승인·렌더하게 하고, 그 스트림의 이전 블러본
  렌더 기록을 무효로 둔다 (ADR 0024).
"""

from __future__ import annotations

import hashlib
import tempfile
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import sqlalchemy as sa

from dlp_privacy.runner import invalidate_render
from dlp_review.cvat import (
    UNIT_SCALE,
    CvatFormatError,
    CvatSchema,
    annotation_tracks,
    from_cvat_tracks,
    quantize,
)
from dlp_review.labelstudio import from_ls_results
from dlp_review.reconcile import ReviewOutcome, reconcile
from dlp_review.tasks import (
    ReviewSetup,
    TaskError,
    current_for_review,
    frame_times,
    label_scale,
    object_key,
)
from dlp_schema.db.repository import (
    get_assignment,
    get_labels,
    get_review_task,
    get_session,
    insert_labels,
    insert_review_work,
    mark_review_task_collected,
    record_review,
    set_privacy_state,
)
from dlp_schema.db.tables import review_tasks
from dlp_schema.labels import LabelRecord, VerificationState
from dlp_schema.ops import ReviewWork
from dlp_schema.review import ReviewMode, ReviewStage, ReviewTaskStatus, ReviewTool
from dlp_schema.session import PrivacyState

if TYPE_CHECKING:
    from dlp_review.ops.policy import ReviewOpsPolicy

OPERATIONAL_MODES = (ReviewMode.STANDARD, ReviewMode.QA)


def _lock_task(conn: sa.Connection, task_key: str) -> ReviewTaskStatus:
    status = conn.execute(
        sa.select(review_tasks.c.status)
        .where(review_tasks.c.task_key == task_key)
        .with_for_update()
    ).scalar_one()
    return ReviewTaskStatus(status)


def _seeded_new_id(assignment_id: str, label_id: str) -> str:
    """오류 삽입 과제에서 검수자가 새로 그린 레코드 ID. 배정의 사본 접두사를 붙여 발견 판정을
    그 배정으로 한정한다 (dlp_review.ops.seeding.detected)."""
    from dlp_review.ops.seeding import seed_prefix  # 순환 import를 피한다

    digest = hashlib.sha256(label_id.encode()).hexdigest()[:16]
    return f"{seed_prefix(assignment_id)}new-{digest}"


def drop_retracted(outcome: ReviewOutcome, labels: list[LabelRecord]) -> list[str]:
    """보낸 뒤 바뀐 라벨에 대한 검수 결과(승인·수정·삭제)를 뺀다.

    - 다른 단계가 지운 라벨: 그것을 parent로 하는 수정 레코드를 쓰면 지운 라벨이 되살아난다.
    - 다른 작업(같은 라벨을 보낸 다른 검수)이 먼저 고친 라벨: 또 고치면 한 라벨에 자식이 둘인
      갈래 이력이 생겨 두 레코드가 모두 현재 라벨이 된다.
    어느 쪽이든 이 라벨은 이미 자식 레코드가 있다. 뺀 원래 라벨 ID를 돌려준다.
    """
    retracted = {x.parent_label_id for x in labels if x.parent_label_id}
    dropped = sorted(
        {i for i in outcome.approved if i in retracted}
        | {
            r.parent_label_id
            for r in outcome.new_records
            if r.parent_label_id is not None and r.parent_label_id in retracted
        }
    )
    outcome.approved = [i for i in outcome.approved if i not in retracted]
    outcome.new_records = [
        r
        for r in outcome.new_records
        if r.parent_label_id is None or r.parent_label_id not in retracted
    ]
    return dropped


def collect_task(
    conn: sa.Connection,
    task_key: str,
    setup: ReviewSetup,
    reviewer_id: str,
    now: datetime,
    ops_policy: ReviewOpsPolicy | None = None,
) -> ReviewOutcome | None:
    """수집했으면 결과를, 이미 수집한 작업이면 None을 돌려준다.

    검수 결과는 작업에 실제로 보낸 라벨(sent_label_ids)과 비교한다. 그 사이 다른 단계가 라벨을
    바꿨어도 보내지 않은 라벨을 검수자가 지운 것으로 보지 않는다.
    reviewer_id는 작업 담당자와 같아야 한다 (담당자가 있는 작업).
    ops_policy는 배정 마무리(표본 판정)에 쓴다.
    """
    if _lock_task(conn, task_key) is ReviewTaskStatus.COLLECTED:
        return None
    task = get_review_task(conn, task_key)
    if task.assignee is not None and reviewer_id != task.assignee:
        raise TaskError(
            f"{task_key}: 담당자({task.assignee})가 아닌 {reviewer_id}의 검수 결과는 받지 않습니다"
        )
    session = get_session(conn, task.session_id)
    if session.ontology_version is None:
        raise TaskError(f"{task.session_id}: 세션에 온톨로지 버전이 없습니다")
    stream_filter = task.stream_id if task.tool is ReviewTool.CVAT else None
    assignment = get_assignment(conn, task.assignment_id) if task.assignment_id else None
    all_labels = get_labels(conn, task.session_id)
    if task.sent_label_ids is not None:
        history = {x.label_id: x for x in all_labels}
        originals = [history[i] for i in task.sent_label_ids if i in history]
    elif assignment is not None:  # sent_label_ids 이전에 만든 작업
        from dlp_review.ops.selection import assignment_selector  # 순환 import를 피한다

        originals = assignment_selector(conn, assignment)(stream_filter, task.label_kinds)
    else:
        originals = current_for_review(conn, task.session_id, stream_filter, task.label_kinds)
    if task.mode in (ReviewMode.BLIND, ReviewMode.DOUBLE):
        # 측정용: 검수자가 낸 모든 라벨을 새 측정 레코드로 남기고 운영 라벨은 건드리지 않는다
        originals = []

    if task.tool is ReviewTool.CVAT:
        if setup.cvat is None:
            raise TaskError("CVAT 클라이언트가 없습니다")
        privacy = task.stage is ReviewStage.PRIVACY
        store = setup.raw if task.media_uri.startswith(setup.raw.uri("")) else setup.labeling
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            video = work / "media.mp4"
            store.get_file(object_key(store, task.media_uri), video)
            times = frame_times(video)
            scale = UNIT_SCALE
            if privacy:  # 블러 라벨은 원본 화소 좌표, 화면은 축소 프록시
                stream = next(s for s in session.streams if s.stream_id == task.stream_id)
                scale = label_scale(setup.raw, stream, video, work)
        project = setup.cvat.http.get(f"/api/tasks/{task.external_id}").json()["project_id"]
        schema = CvatSchema.from_labels(setup.cvat.project_labels(int(project)))
        try:
            # 모양(Shape) 모드 직사각형도 트랙으로 받는다 (버리면 놓친 얼굴이 블러 없이 남는다)
            tracks = annotation_tracks(
                setup.cvat.get_annotations(int(task.external_id)), len(times), schema
            )
            reviewed = from_cvat_tracks(
                tracks,
                times,
                schema,
                task.stream_id,
                new_box_kind="blur_track" if privacy else "box_track",
                scale=scale,
            )
        except CvatFormatError as exc:
            raise TaskError(f"{task_key}: {exc}") from exc
        normalize = partial(quantize, scale=scale)
    else:
        if setup.label_studio is None:
            raise TaskError("Label Studio 클라이언트가 없습니다")
        results = setup.label_studio.latest_results(int(task.external_id))
        if results is None:
            # 프리라벨은 예측(prediction)으로 보내므로 사람이 제출한 주석이 없으면 결과가 없다
            raise TaskError(f"{task_key}: 검수자가 제출한 주석이 아직 없습니다")
        reviewed = from_ls_results(results, {x.label_id for x in originals})
        normalize = None
        # 검수 시간 (운영 지표, WP16). CVAT는 작업 시간을 재지 않아 dlp ops log-work로 기록한다
        seconds = setup.label_studio.lead_seconds(int(task.external_id))
        if seconds > 0:
            insert_review_work(
                conn,
                ReviewWork(
                    work_id=f"{task_key}:work",
                    task_key=task_key,
                    session_id=task.session_id,
                    reviewer=reviewer_id,
                    stage=task.stage.value,
                    seconds=seconds,
                    video_ms=session.duration_ms,
                    source="label_studio",
                    recorded_at=now,
                ),
            )

    outcome = reconcile(
        originals,
        reviewed,
        session_id=task.session_id,
        ontology_version=session.ontology_version,
        reviewer_id=reviewer_id,
        now=now,
        normalize=normalize,
    )
    outcome.stale = drop_retracted(outcome, all_labels)
    if task.mode is ReviewMode.BLIND or task.mode is ReviewMode.DOUBLE:
        measurement = "blind" if task.mode is ReviewMode.BLIND else "double"
        outcome.new_records = [
            x.model_copy(update={"measurement": measurement}) for x in outcome.new_records
        ]
    elif task.mode is ReviewMode.SEEDED_ERROR:
        # 오류 삽입 과제에서 나온 모든 레코드는 학습에서 빠진다 (새로 그린 것 포함).
        # 새로 그린 레코드는 배정의 사본 접두사를 붙여 그 배정의 결과로만 센다.
        aid = task.assignment_id
        outcome.new_records = [
            x.model_copy(
                update={
                    "seeded_error": True,
                    **(
                        {"label_id": _seeded_new_id(aid, x.label_id)}
                        if aid is not None and x.parent_label_id is None
                        else {}
                    ),
                }
            )
            for x in outcome.new_records
        ]
    insert_labels(conn, outcome.new_records)
    for label_id in outcome.approved:
        record_review(conn, label_id, VerificationState.HUMAN_APPROVED, reviewer_id, now)
    mark_review_task_collected(conn, task_key, now)
    if task.stage is ReviewStage.PRIVACY and task.mode in OPERATIONAL_MODES and outcome.new_records:
        if session.privacy_state is PrivacyState.APPROVED:
            # 승인 뒤 블러가 바뀌었다: 다시 승인해야 블러본을 새로 렌더한다
            set_privacy_state(conn, task.session_id, PrivacyState.AUTO_BLURRED)
        # 이전 블러본을 어떤 단계도 현재 것으로 보지 않게 렌더 기록을 무효로 둔다
        with tempfile.TemporaryDirectory() as tmp:
            invalidate_render(
                setup.labeling, task.session_id, task.stream_id, f"collected:{task_key}", Path(tmp)
            )
    if assignment is not None:
        from dlp_review.ops.runner import finish_assignment

        finish_assignment(conn, assignment, now, ops_policy)
    return outcome
