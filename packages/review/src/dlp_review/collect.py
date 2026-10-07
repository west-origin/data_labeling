"""검수 결과 수집.

도구에서 결과를 받아 reconcile하고 DB에 이력으로 쓴다. 한 작업은 한 번만 수집한다.
"""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

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
    get_review_task,
    get_session,
    insert_labels,
    mark_review_task_collected,
    record_review,
)
from dlp_schema.labels import VerificationState
from dlp_schema.review import ReviewTaskStatus, ReviewTool


def collect_task(
    conn: sa.Connection, task_key: str, setup: ReviewSetup, reviewer_id: str, now: datetime
) -> ReviewOutcome | None:
    """수집했으면 결과를, 이미 수집한 작업이면 None을 돌려준다."""
    task = get_review_task(conn, task_key)
    if task.status is ReviewTaskStatus.COLLECTED:
        return None
    session = get_session(conn, task.session_id)
    if session.ontology_version is None:
        raise TaskError(f"{task.session_id}: 세션에 온톨로지 버전이 없습니다")
    stream_filter = task.stream_id if task.tool is ReviewTool.CVAT else None
    originals = current_for_review(conn, task.session_id, stream_filter, task.label_kinds)

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

    outcome = reconcile(
        originals,
        reviewed,
        session_id=task.session_id,
        ontology_version=session.ontology_version,
        reviewer_id=reviewer_id,
        now=now,
        normalize=normalize,
    )
    insert_labels(conn, outcome.new_records)
    for label_id in outcome.approved:
        record_review(conn, label_id, VerificationState.HUMAN_APPROVED, reviewer_id, now)
    mark_review_task_collected(conn, task_key, now)
    return outcome
