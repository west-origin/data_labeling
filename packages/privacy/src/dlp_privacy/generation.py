"""프라이버시 운영 이력과 검수 세대 판정 (WP5·WP12, ADR 0033).

탐지 실행·렌더·도구 호출 없이 현재 운영 블러와 승인 대상 검수 작업을 고른다.
검수 작업은 완료 탐지 ID로 연결하고, DB 완료 이력 이전 데이터만 시각 규약을 사용한다.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import sqlalchemy as sa

from dlp_schema.db.repository import get_labels, latest_privacy_detection, list_review_tasks
from dlp_schema.episode import current_labels, non_operational_ids
from dlp_schema.labels import LabelRecord, Source
from dlp_schema.review import ReviewMode, ReviewStage, ReviewTask, ReviewTaskStatus
from dlp_schema.session import Session, StreamKind

VIDEO_KINDS = {StreamKind.BODYCAM, StreamKind.THIRD_PERSON}


def operational_blur(history: Sequence[LabelRecord], stream_id: str) -> list[LabelRecord]:
    """스트림의 운영 블러 라벨 (오류 삽입·측정 레코드와 그 후손 제외). 블러본은 이것으로 만든다.

    Args:
        history: 세션 라벨 전체 이력 (다른 종류가 섞여도 된다).
        stream_id: 영상 스트림 ID.

    Returns:
        그 스트림의 현재(최신·삭제되지 않은) 운영 blur_track 레코드.
    """
    # current_labels는 기본으로 운영 라벨만 남긴다. seeded_error 검사는 이중 안전장치다.
    return [
        x
        for x in current_labels([x for x in history if x.kind == "blur_track"])
        if x.stream_id == stream_id and not x.seeded_error
    ]


def last_detection(
    labels: Sequence[LabelRecord], stream_id: str, completed_at: datetime | None = None
) -> datetime | None:
    """마지막 자동 탐지 시각: 완료 이력(0건 포함)과 구버전 모델 블러 레코드 중 가장 늦은 것.

    오류 삽입 사본·측정 레코드와 그 후손은 운영 블러가 아니므로 세지 않는다 (오류 삽입 계획이
    모델 출처 사본을 지금 시각으로 만들어도 승인 조건이 움직이지 않게).

    Args:
        labels: 세션 블러 라벨 전체 이력.
        stream_id: 영상 스트림 ID.
        completed_at: DB 완료 이력의 최신 시각 (0건 탐지 포함). 구버전이면 None.

    Returns:
        가장 늦은 시각. 완료 이력과 모델 출처 운영 레코드가 모두 없으면 None.
    """
    excluded = non_operational_ids(list(labels))
    times = [
        x.created_at
        for x in labels
        if x.stream_id == stream_id
        and x.provenance.source is Source.MODEL
        and x.label_id not in excluded
    ]
    if completed_at is not None:
        times.append(completed_at)
    return max(times) if times else None


def collected_privacy_reviews(
    conn: sa.Connection,
    session_id: str,
    stream_id: str,
    labels: Sequence[LabelRecord] | None = None,
) -> list[ReviewTask]:
    """최신 완료 세대에 연결된 수집 완료 운영 검수. 구 탐지만 시각으로 판단한다.

    완료 이력이 생긴 뒤에는 세대 없는 구 작업을 인정하지 않는다. 생성 시각이나 시계 역행으로
    이전 탐지의 작업이 새 탐지 검수로 인정되는 일을 막는다 (ADR 0033)."""
    history = labels if labels is not None else get_labels(conn, session_id, kinds=["blur_track"])
    run = latest_privacy_detection(conn, session_id, stream_id)
    since = last_detection(history, stream_id, run.at if run else None)
    return [
        t
        for t in list_review_tasks(conn, session_id, ReviewStage.PRIVACY)
        if t.stream_id == stream_id
        and t.status is ReviewTaskStatus.COLLECTED
        and t.mode in (ReviewMode.STANDARD, ReviewMode.QA)
        and (
            t.privacy_detection_id == run.event_id
            if run is not None
            else t.privacy_detection_id is None and (since is None or t.created_at >= since)
        )
    ]


def missing_privacy_reviews(conn: sa.Connection, session: Session) -> list[str]:
    """최신 완료 탐지 세대에 연결된 수집 완료 운영 검수가 없는 영상 스트림.

    탐지 결과가 하나도 없어도(놓친 대상이 있을 수 있다) 사람이 영상 전체를 본 작업이 있어야 한다.
    운영 작업(표준·QA)만 센다. 블라인드·이중·오류 삽입 작업은 운영 블러를 검수하지 않는다.

    최신 완료 탐지 ID에 연결된 작업만 인정한다. DB 완료 기록 이전 구 데이터에만
    created_at ≥ 마지막 모델 라벨 시각을 사용한다 (ADR 0033).

    Returns:
        조건을 만족하지 못한 영상 스트림 ID 목록 (세션 스트림 순서).
    """
    labels = get_labels(conn, session.session_id, kinds=["blur_track"])
    missing: list[str] = []
    for s in session.streams:
        if s.kind not in VIDEO_KINDS:
            continue
        if not collected_privacy_reviews(conn, session.session_id, s.stream_id, labels):
            missing.append(s.stream_id)
    return missing
