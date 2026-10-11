"""블러가 바뀐 스트림의 배포 매체 회수 요청 (ADR 0039).

시간대 보존 SQLite와 LocalStore로 돈다 (서비스 없음).

정답 근거: 회수 요청은 지금 운영 블러의 렌더 해시와 작업의 렌더 해시가 다를 때만 남는다. 탐지가
블러 레코드를 쓴 스트림만 요청하고(구 작업 포함), 결과 0건이고 바뀐 것이 없는 스트림은 요청하지
않는다. 탐지 결과는 합성 블러 시나리오의 정답 라벨(모델 출처로 바꾼 것)이다.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import sqlalchemy as sa

from dlp_fixtures.video import generate_blur_scenario
from dlp_media.storage import LocalStore, sha256_file
from dlp_privacy.pipeline import StreamResult
from dlp_privacy.policy import PrivacyPolicy
from dlp_privacy.runner import detect_session, operational_blur, render_hash, stale_media_tasks
from dlp_schema import load_ontology, repo_root
from dlp_schema.db.repository import (
    get_labels,
    get_review_task,
    insert_review_task,
    insert_session,
    register_ontology,
)
from dlp_schema.labels import LabelRecord, Provenance, Source
from dlp_schema.review import ReviewStage, ReviewTask, ReviewTaskStatus, ReviewTool
from dlp_schema.session import PrivacyState
from dlp_schema.testing import FIXED_TIME, make_session, sqlite_db

SID = "s001"
NOW = FIXED_TIME + timedelta(hours=1)


@pytest.fixture
def engine() -> Iterator[sa.Engine]:
    """표·온톨로지 v1을 만든 메모리 SQLite."""
    eng = sqlite_db()
    with eng.begin() as conn:
        register_ontology(conn, load_ontology(repo_root() / "config/ontology/v1"))
    yield eng
    eng.dispose()


def _setup(engine: sa.Engine, tmp_path: Path, policy: PrivacyPolicy) -> LocalStore:
    """승인된 2스트림 세션(원본은 원본 버킷)과 라벨링 작업 넷을 넣는다. 원본 저장소.

    작업: 바디캠 CVAT(지금 해시), 바디캠 구 작업(해시 없음), 3인칭 CVAT(지금 해시),
    바디캠 블러 검수.
    """
    raw = LocalStore(tmp_path / "store", "dlp-raw")
    video = tmp_path / "v.bin"
    video.write_bytes(b"synthetic")  # 탐지는 가짜라 영상 내용은 읽지 않는다
    streams: list[dict[str, Any]] = []
    for sid, kind, sync in (
        ("bodycam", "bodycam", "reference"),
        ("third", "third_person", "manual"),
    ):
        key = f"sessions/{SID}/raw/{sid}.mp4"
        raw.put_file(key, video, sha256_file(video))
        streams.append({"stream_id": sid, "kind": kind, "sync_method": sync, "uri": raw.uri(key)})
    empty = render_hash([], policy)  # 아직 블러가 없는 세션의 블러본 해시
    labeling = "local://dlp-labeling/sessions/s001/review"
    tasks = [
        (
            "cvat:2",
            "bodycam",
            f"{labeling}/a/bodycam-{empty[:16]}.mp4",
            empty,
            ReviewStage.LABELING,
        ),
        ("cvat:3", "bodycam", f"{labeling}/b/bodycam.mp4", None, ReviewStage.LABELING),
        ("cvat:4", "third", f"{labeling}/a/third-{empty[:16]}.mp4", empty, ReviewStage.LABELING),
        ("cvat:1", "bodycam", "local://dlp-raw/x.proxy.mp4", None, ReviewStage.PRIVACY),
    ]
    with engine.begin() as conn:
        insert_session(
            conn, make_session(SID, privacy_state=PrivacyState.APPROVED, streams=streams)
        )
        for key, stream, uri, rhash, stage in tasks:
            insert_review_task(
                conn,
                ReviewTask(
                    task_key=key, tool=ReviewTool.CVAT, external_id=key[5:], session_id=SID,
                    stream_id=stream, stage=stage, media_uri=uri, label_kinds=("box_track",),
                    created_at=FIXED_TIME, render_hash=rhash,
                ),
            )  # fmt: skip
    return raw


def _model_blur() -> list[LabelRecord]:
    """합성 시나리오의 바디캠 정답 블러를 모델 출처(미검수)로 바꾼 탐지 결과."""
    model = Provenance(source=Source.MODEL, model_version="m1")
    return [
        x.model_copy(update={"provenance": model, "confidence": 0.9})
        for x in generate_blur_scenario(0, session_id=SID, duration_ms=200).labels
    ]


def test_detect_requests_revocation_only_on_changed_streams(
    engine: sa.Engine, tmp_path: Path, policy: PrivacyPolicy
) -> None:
    """바디캠에서만 블러가 생기면 바디캠 라벨링 작업(해시 있음·구 작업)만 회수 요청된다.

    결과 0건인 3인칭은 블러 집합이 그대로라 요청하지 않는다. 블러 검수 작업은 회수하지 않는다.
    """
    raw = _setup(engine, tmp_path, policy)

    def infer(video: Path, *, stream_id: str, **_: object) -> StreamResult:
        return StreamResult(_model_blur() if stream_id == "bodycam" else [], [], {}, 5)

    with engine.begin() as conn, patch("dlp_privacy.runner.detect_video", side_effect=infer):
        summary = detect_session(conn, SID, raw, {}, {}, policy, NOW)
    assert summary.changed_streams == ["bodycam"]
    assert sorted(summary.revocations) == ["cvat:2", "cvat:3"]
    with engine.connect() as conn:
        for key in ("cvat:2", "cvat:3"):
            task = get_review_task(conn, key)
            assert (
                task.status is ReviewTaskStatus.REVOKED and task.revoke_reason == "privacy_detect"
            )
        assert get_review_task(conn, "cvat:4").status is ReviewTaskStatus.OPEN
        assert get_review_task(conn, "cvat:1").revoke_requested_at is None
        # 3인칭은 지금 해시가 작업 해시와 같다 (판정 불변식)
        history = get_labels(conn, SID, kinds=["blur_track"])
        assert render_hash(operational_blur(history, "third"), policy) == render_hash([], policy)
        assert stale_media_tasks(conn, SID, policy) == []


def test_zero_result_redetect_requests_nothing(
    engine: sa.Engine, tmp_path: Path, policy: PrivacyPolicy
) -> None:
    """두 스트림 모두 결과 0건(블러 집합 그대로)이면 승인은 풀려도 회수 요청은 없다."""
    raw = _setup(engine, tmp_path, policy)
    empty = StreamResult([], [], {}, 5)
    with engine.begin() as conn, patch("dlp_privacy.runner.detect_video", return_value=empty):
        summary = detect_session(conn, SID, raw, {}, {}, policy, NOW)
    assert summary.reran and summary.changed_streams == [] and summary.revocations == []
    with engine.connect() as conn:
        assert all(
            get_review_task(conn, k).status is ReviewTaskStatus.OPEN
            for k in ("cvat:2", "cvat:3", "cvat:4")
        )
