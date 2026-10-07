"""세션에서 검수 작업을 만든다.

- 프라이버시 검수 (CVAT, 원본 접근 권한자): 원본 프록시 영상 + 현재 블러 트랙.
- 작업 라벨 검수 (일반 라벨러): 라벨러 워터마크를 입힌 블러본.
  - CVAT: 객체·도구 박스 트랙과 키포인트 트랙
  - Label Studio: 시간 구간 라벨, 영상과 장갑·IMU 시계열 동기 재생
  일반 라벨러 경로의 URL은 라벨러 자격 증명(라벨링 버킷 읽기 전용)으로 서명한다.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import sqlalchemy as sa

from dlp_media.pts import build_pts_index
from dlp_media.storage import ObjectStore, S3Store, sha256_file
from dlp_review.clients import CvatClient, LabelStudioClient
from dlp_review.cvat import CvatSchema, label_spec, to_cvat_tracks
from dlp_review.labelstudio import LS_KINDS, label_config, to_ls_results
from dlp_review.roles import check_stage_uris
from dlp_review.timeseries import write_timeseries_csv
from dlp_review.watermark import burn_watermark
from dlp_schema.db.repository import get_labels, get_session, insert_review_task
from dlp_schema.episode import current_labels
from dlp_schema.labels import LabelRecord
from dlp_schema.ontology import Ontology
from dlp_schema.review import ReviewMode, ReviewStage, ReviewTask, ReviewTool
from dlp_schema.session import PrivacyState, Session, StreamKind
from dlp_sync.signals import Series, glove_series, imu_series

VIDEO_KINDS = {StreamKind.BODYCAM, StreamKind.THIRD_PERSON}
PRIVACY_PROJECT = "dlp-privacy"
SPATIAL_PROJECT = "dlp-labeling-spatial"
TEMPORAL_PROJECT = "dlp-labeling-temporal"


class TaskError(RuntimeError):
    pass


@dataclass
class ReviewSetup:
    raw: ObjectStore
    labeling: ObjectStore
    labeling_reader: S3Store  # 라벨러 자격 증명. 서명 URL용
    ontology: Ontology
    cvat: CvatClient | None = None
    label_studio: LabelStudioClient | None = None


def frame_times(video: Path) -> list[int]:
    return [round(t) for t in build_pts_index(video).ms]


def object_key(store: ObjectStore, uri: str) -> str:
    prefix = store.uri("")
    if not uri.startswith(prefix):
        raise TaskError(f"{uri}는 {store.bucket}에 있지 않습니다")
    return uri.removeprefix(prefix)


def current_for_review(
    conn: sa.Connection, session_id: str, stream_id: str | None, kinds: tuple[str, ...]
) -> list[LabelRecord]:
    labels = current_labels(get_labels(conn, session_id, kinds=list(kinds)))
    return [
        x
        for x in labels
        if not x.seeded_error and (stream_id is None or x.stream_id in (stream_id, None))
    ]


# 작업에 넣을 라벨을 고르는 함수: (스트림 또는 None, 라벨 종류) → 라벨
Selector = Callable[[str | None, tuple[str, ...]], list[LabelRecord]]


def default_selector(conn: sa.Connection, session_id: str) -> Selector:
    """현재 운영 라벨 (오류 삽입·측정 레코드 제외)."""

    def pick(stream_id: str | None, kinds: tuple[str, ...]) -> list[LabelRecord]:
        return current_for_review(conn, session_id, stream_id, kinds)

    return pick


def _cvat_project(cvat: CvatClient, name: str, labels: list[str]) -> tuple[int, CvatSchema]:
    project = cvat.find_project(name)
    pid = int(project["id"]) if project else cvat.create_project(name, label_spec(labels))
    return pid, CvatSchema.from_labels(cvat.project_labels(pid))


def create_privacy_tasks(
    conn: sa.Connection,
    session_id: str,
    setup: ReviewSetup,
    now: datetime,
    *,
    select: Selector | None = None,
    mode: ReviewMode = ReviewMode.STANDARD,
    assignment_id: str | None = None,
    assignee: str | None = None,
    streams: set[str] | None = None,
) -> list[ReviewTask]:
    """select가 없으면 현재 블러 트랙을 넣는다 (배정 방식에 따라 다른 라벨을 넣을 때 쓴다)."""
    pick = select or default_selector(conn, session_id)
    if setup.cvat is None:
        raise TaskError("CVAT 클라이언트가 없습니다")
    session = get_session(conn, session_id)
    if session.privacy_state is PrivacyState.PENDING:
        raise TaskError(f"{session_id}: 블러 자동 탐지를 먼저 실행하세요 (dlp privacy detect)")
    pid, schema = _cvat_project(setup.cvat, PRIVACY_PROJECT, list(setup.ontology.privacy_targets))
    out: list[ReviewTask] = []
    with tempfile.TemporaryDirectory() as tmp:
        for stream in (s for s in session.streams if s.kind in VIDEO_KINDS):
            if streams is not None and stream.stream_id not in streams:
                continue
            key = f"sessions/{session_id}/derived/{stream.stream_id}.proxy.mp4"
            video = Path(tmp) / f"{stream.stream_id}.mp4"
            setup.raw.get_file(key, video)
            labels = pick(stream.stream_id, ("blur_track",))
            tid = setup.cvat.create_task(f"{session_id}/{stream.stream_id}/privacy", pid, video)
            setup.cvat.put_tracks(tid, to_cvat_tracks(labels, frame_times(video), schema))
            task = ReviewTask(
                task_key=f"cvat:{tid}", tool=ReviewTool.CVAT, external_id=str(tid),
                session_id=session_id, stream_id=stream.stream_id, stage=ReviewStage.PRIVACY,
                media_uri=setup.raw.uri(key), label_kinds=("blur_track",), created_at=now,
                mode=mode, assignment_id=assignment_id, assignee=assignee,
                sent_label_ids=tuple(x.label_id for x in labels),
            )  # fmt: skip
            insert_review_task(conn, task)
            out.append(task)
    return out


def _series(session: Session, raw: ObjectStore, work: Path) -> dict[str, Series]:
    out: dict[str, Series] = {}
    for s in session.streams:
        if s.kind in (StreamKind.GLOVE_LEFT, StreamKind.GLOVE_RIGHT, StreamKind.IMU):
            path = work / f"{s.stream_id}.parquet"
            raw.get_file(object_key(raw, s.uri), path)
            out[s.stream_id] = imu_series(path) if s.kind is StreamKind.IMU else glove_series(path)
    return out


def create_labeling_tasks(
    conn: sa.Connection,
    session_id: str,
    setup: ReviewSetup,
    assignee: str,
    now: datetime,
    *,
    select: Selector | None = None,
    mode: ReviewMode = ReviewMode.STANDARD,
    assignment_id: str | None = None,
    streams: set[str] | None = None,
    tools: set[ReviewTool] | None = None,
) -> list[ReviewTask]:
    """select가 없으면 현재 운영 라벨을 넣는다. tools·streams로 만들 작업을 좁힌다."""
    pick = select or default_selector(conn, session_id)
    use_cvat = setup.cvat is not None and (tools is None or ReviewTool.CVAT in tools)
    use_ls = setup.label_studio is not None and (tools is None or ReviewTool.LABEL_STUDIO in tools)
    session = get_session(conn, session_id)
    if session.privacy_state is not PrivacyState.APPROVED:
        raise TaskError(f"{session_id}: 프라이버시 승인 전에는 작업 라벨 검수를 만들 수 없습니다")
    raw_bucket = setup.raw.bucket
    out: list[ReviewTask] = []
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for stream in (s for s in session.streams if s.kind in VIDEO_KINDS):
            if streams is not None and stream.stream_id not in streams:
                continue
            if not use_cvat and not (use_ls and stream.kind is StreamKind.BODYCAM):
                continue
            blurred = work / f"{stream.stream_id}-blurred.mp4"
            setup.labeling.get_file(
                f"sessions/{session_id}/blurred/{stream.stream_id}.mp4", blurred
            )
            marked = work / f"{stream.stream_id}-{assignee}.mp4"
            burn_watermark(blurred, marked, f"{assignee} {session_id}")
            vkey = f"sessions/{session_id}/review/{assignee}/{stream.stream_id}.mp4"
            setup.labeling.put_file(vkey, marked, sha256_file(marked))
            media_uri = setup.labeling.uri(vkey)

            if use_cvat:
                assert setup.cvat is not None
                labels = pick(stream.stream_id, ("box_track", "keypoint_track"))
                names = [*setup.ontology.objects, "kp_hand21", "kp_coco17", "kp_wholebody133"]
                pid, schema = _cvat_project(setup.cvat, SPATIAL_PROJECT, names)
                tid = setup.cvat.create_task(
                    f"{session_id}/{stream.stream_id}/{assignee}", pid, marked
                )
                setup.cvat.put_tracks(tid, to_cvat_tracks(labels, frame_times(marked), schema))
                spatial = ("box_track", "keypoint_track")
                out.append(
                    _task(f"cvat:{tid}", ReviewTool.CVAT, tid, session_id, stream.stream_id,
                          assignee, media_uri, spatial, now, mode, assignment_id, labels)
                )  # fmt: skip

            if use_ls and stream.kind is StreamKind.BODYCAM:
                assert setup.label_studio is not None
                csv = work / "timeseries.csv"
                write_timeseries_csv(session, _series(session, setup.raw, work), csv)
                ckey = f"sessions/{session_id}/review/timeseries.csv"
                setup.labeling.put_file(ckey, csv, sha256_file(csv))
                labels = pick(None, LS_KINDS)
                ls = setup.label_studio
                project = ls.find_project(TEMPORAL_PROJECT)
                lpid = (
                    int(project["id"])
                    if project
                    else ls.create_project(TEMPORAL_PROJECT, label_config(setup.ontology))
                )
                data = {
                    "video": setup.labeling_reader.presign(vkey),
                    "timeseries": setup.labeling_reader.presign(ckey),
                    "session_id": session_id,
                    "assignee": assignee,
                }
                check_stage_uris(ReviewStage.LABELING, [str(v) for v in data.values()], raw_bucket)
                tid = ls.create_task(lpid, data, to_ls_results(labels))
                out.append(_task(f"label_studio:{tid}", ReviewTool.LABEL_STUDIO, tid, session_id,
                                 stream.stream_id, assignee, media_uri, LS_KINDS, now, mode,
                                 assignment_id, labels))  # fmt: skip
    check_stage_uris(ReviewStage.LABELING, [t.media_uri for t in out], raw_bucket)
    for task in out:
        insert_review_task(conn, task)
    return out


def _task(
    key: str, tool: ReviewTool, tid: int, session_id: str, stream_id: str, assignee: str,
    media_uri: str, kinds: tuple[str, ...], now: datetime,
    mode: ReviewMode = ReviewMode.STANDARD, assignment_id: str | None = None,
    sent: Sequence[LabelRecord] = (),
) -> ReviewTask:  # fmt: skip
    return ReviewTask(
        task_key=key, tool=tool, external_id=str(tid), session_id=session_id, stream_id=stream_id,
        stage=ReviewStage.LABELING, assignee=assignee, media_uri=media_uri, label_kinds=kinds,
        created_at=now, mode=mode, assignment_id=assignment_id,
        sent_label_ids=tuple(x.label_id for x in sent),
    )  # fmt: skip
