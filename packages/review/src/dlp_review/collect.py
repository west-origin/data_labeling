"""검수 결과 수집.

도구에서 결과를 받아 reconcile하고 DB에 이력으로 쓴다. 한 작업은 한 번만 수집한다.
"""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import sqlalchemy as sa

from dlp_review.cvat import CvatSchema, from_cvat_tracks, quantize
from dlp_review.labelstudio import from_ls_results
from dlp_review.reconcile import ReviewOutcome, reconcile
from dlp_review.tasks import (
    ReviewSetup,
    TaskError,
    current_for_review,
    frame_times,
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
)
from dlp_schema.labels import VerificationState
from dlp_schema.ops import ReviewWork
from dlp_schema.review import ReviewMode, ReviewTaskStatus, ReviewTool

if TYPE_CHECKING:
    from dlp_review.ops.policy import ReviewOpsPolicy


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
    ops_policy는 배정 마무리(표본 판정)에 쓴다.
    """
    task = get_review_task(conn, task_key)
    if task.status is ReviewTaskStatus.COLLECTED:
        return None
    session = get_session(conn, task.session_id)
    if session.ontology_version is None:
        raise TaskError(f"{task.session_id}: 세션에 온톨로지 버전이 없습니다")
    stream_filter = task.stream_id if task.tool is ReviewTool.CVAT else None
    assignment = get_assignment(conn, task.assignment_id) if task.assignment_id else None
    if task.sent_label_ids is not None:
        history = {x.label_id: x for x in get_labels(conn, task.session_id)}
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
        store = setup.raw if task.media_uri.startswith(setup.raw.uri("")) else setup.labeling
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "media.mp4"
            store.get_file(object_key(store, task.media_uri), video)
            times = frame_times(video)
        project = setup.cvat.http.get(f"/api/tasks/{task.external_id}").json()["project_id"]
        schema = CvatSchema.from_labels(setup.cvat.project_labels(int(project)))
        reviewed = from_cvat_tracks(
            setup.cvat.get_tracks(int(task.external_id)), times, schema, task.stream_id
        )
        normalize = quantize
    else:
        if setup.label_studio is None:
            raise TaskError("Label Studio 클라이언트가 없습니다")
        results = setup.label_studio.latest_results(int(task.external_id))
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
    if task.mode is ReviewMode.BLIND or task.mode is ReviewMode.DOUBLE:
        measurement = "blind" if task.mode is ReviewMode.BLIND else "double"
        outcome.new_records = [
            x.model_copy(update={"measurement": measurement}) for x in outcome.new_records
        ]
    elif task.mode is ReviewMode.SEEDED_ERROR:
        # 오류 삽입 과제에서 나온 모든 레코드는 학습에서 빠진다 (새로 그린 것 포함)
        outcome.new_records = [
            x.model_copy(update={"seeded_error": True}) for x in outcome.new_records
        ]
    insert_labels(conn, outcome.new_records)
    for label_id in outcome.approved:
        record_review(conn, label_id, VerificationState.HUMAN_APPROVED, reviewer_id, now)
    mark_review_task_collected(conn, task_key, now)
    if assignment is not None:
        from dlp_review.ops.runner import finish_assignment

        finish_assignment(conn, assignment, now, ops_policy)
    return outcome
