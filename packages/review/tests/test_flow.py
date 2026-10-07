from __future__ import annotations

import copy
import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
import sqlalchemy as sa
import yaml

from dlp_fixtures.actions import generate_action_scenario
from dlp_fixtures.sync import generate_sync_scenario
from dlp_fixtures.video import generate_blur_scenario
from dlp_media.ingest import ingest_session, load_manifest
from dlp_media.storage import S3Store
from dlp_privacy.detection import FrameDetector
from dlp_privacy.detectors.oracle import OracleDetector
from dlp_privacy.policy import TargetPolicy, load_policy
from dlp_privacy.runner import approve_session, detect_session, render_session
from dlp_review.clients import CvatClient, LabelStudioClient
from dlp_review.collect import collect_task
from dlp_review.labelstudio import LS_KINDS
from dlp_review.tasks import ReviewSetup, create_labeling_tasks, create_privacy_tasks
from dlp_schema.db.migrate import upgrade
from dlp_schema.db.repository import get_labels, insert_labels, list_review_tasks, register_ontology
from dlp_schema.episode import current_labels
from dlp_schema.labels import BlurTrackPayload, LabelRecord, VerificationState
from dlp_schema.ontology import load_ontology
from dlp_schema.review import ReviewStage, ReviewTaskStatus
from dlp_schema.testing import FIXED_TIME

pytestmark = pytest.mark.services
ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def pg() -> Iterator[sa.Engine]:
    url = sa.make_url(
        os.environ.get(
            "DLP_DATABASE_URL", "postgresql+psycopg://dlp:dlp-dev-password@localhost:5432/dlp"
        )
    )
    name = f"dlp_test_{uuid.uuid4().hex[:8]}"
    admin = sa.create_engine(url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(sa.text(f'CREATE DATABASE "{name}"'))
    test_url = url.set(database=name).render_as_string(hide_password=False)
    upgrade(test_url)
    engine = sa.create_engine(test_url)
    with engine.begin() as conn:
        register_ontology(conn, load_ontology(ROOT / "config" / "ontology" / "v1"))
    try:
        yield engine
    finally:
        engine.dispose()
        with admin.connect() as conn:
            conn.execute(sa.text(f'DROP DATABASE "{name}" WITH (FORCE)'))
        admin.dispose()


@pytest.fixture
def setup() -> ReviewSetup:
    """CVAT는 이미지가 커서 CI 서비스 작업에서 띄우지 않는다. 닿지 않으면 건너뛴다."""
    try:
        cvat = CvatClient.from_env()
    except httpx.HTTPError as exc:
        pytest.skip(f"CVAT에 연결할 수 없습니다 (make cvat-up): {exc}")
    return ReviewSetup(
        raw=S3Store.from_env("dlp-raw"),
        labeling=S3Store.from_env("dlp-labeling"),
        labeling_reader=S3Store.labeler_from_env("dlp-labeling"),
        ontology=load_ontology(ROOT / "config" / "ontology" / "v1"),
        cvat=cvat,
        label_studio=LabelStudioClient.from_env(),
    )


def _session(pg: sa.Engine, setup: ReviewSetup, tmp_path: Path) -> tuple[str, list[LabelRecord]]:
    """블러 픽스처 영상 + 장갑 데이터로 세션을 만들고 오라클로 블러 프리라벨까지 한다."""
    blur = generate_blur_scenario(5)
    blur.write(tmp_path / "bodycam.mp4")
    generate_sync_scenario(1, recorded_at=FIXED_TIME, duration_ms=3_000).write(
        tmp_path, videos=False
    )
    sid = f"rev-{uuid.uuid4().hex[:8]}"
    manifest = {
        "session_id": sid, "domain": "cleaning", "worker_id": "w01", "site_id": "site01",
        "consent_version": "c1", "recorded_at": FIXED_TIME.isoformat(), "ontology_version": "1.0.0",
        "streams": [
            {"stream_id": "bodycam", "kind": "bodycam", "path": "bodycam.mp4"},
            {"stream_id": "glove_right", "kind": "glove_right", "path": "glove_right.parquet"},
        ],
    }  # fmt: skip
    (tmp_path / "m.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    policy = load_policy(ROOT)
    policy = policy.model_copy(
        update={
            "targets": {
                t: TargetPolicy(margin=tp.margin, detectors=("oracle",))
                for t, tp in policy.targets.items()
            }
        }
    )
    with pg.begin() as conn:
        ingest_session(*load_manifest(tmp_path / "m.yaml"), setup.raw, conn)
        oracle: dict[str, FrameDetector] = {"oracle": OracleDetector("oracle", blur.labels)}
        detect_session(conn, sid, setup.raw, oracle, {}, policy, FIXED_TIME)
        return sid, current_labels(get_labels(conn, sid, kinds=["blur_track"]))


def test_unchanged_review_round_trips_losslessly_through_both_tools(
    pg: sa.Engine, setup: ReviewSetup, tmp_path: Path
) -> None:
    """완료 기준: 도구를 거쳐도 ID·시각·속성이 그대로라 고치지 않으면 모두 승인만 된다."""
    sid, blur = _session(pg, setup, tmp_path)
    with pg.begin() as conn:
        [privacy] = create_privacy_tasks(conn, sid, setup, FIXED_TIME)
        outcome = collect_task(conn, privacy.task_key, setup, "rev01", FIXED_TIME)
    assert outcome is not None and not outcome.new_records
    assert sorted(outcome.approved) == sorted(x.label_id for x in blur)

    with pg.begin() as conn:
        approve_session(conn, sid)
        render_session(conn, sid, setup.raw, setup.labeling, load_policy(ROOT))
        temporal = [
            x.model_copy(update={"session_id": sid, "label_id": f"{sid}-{x.label_id}"})
            for x in generate_action_scenario(4).labels
            if x.kind in LS_KINDS
        ]
        insert_labels(conn, temporal)
        tasks = create_labeling_tasks(conn, sid, setup, "labeler01", FIXED_TIME)
        ls_task = next(t for t in tasks if t.task_key.startswith("label_studio:"))
        outcome = collect_task(conn, ls_task.task_key, setup, "labeler01", FIXED_TIME)
    assert outcome is not None and not outcome.new_records
    assert sorted(outcome.approved) == sorted(x.label_id for x in temporal)


def test_privacy_review_edits_become_label_history(
    pg: sa.Engine, setup: ReviewSetup, tmp_path: Path
) -> None:
    sid, blur = _session(pg, setup, tmp_path)
    assert setup.cvat is not None
    with pg.begin() as conn:
        [task] = create_privacy_tasks(conn, sid, setup, FIXED_TIME)
    assert task.stage is ReviewStage.PRIVACY and task.media_uri.startswith("s3://dlp-raw/")

    # 검수자: 첫 트랙을 5 px 옮기고, 하나를 지우고, 하나를 새로 그린다
    tid = int(task.external_id)
    tracks = setup.cvat.get_tracks(tid)
    for shape in tracks[0]["shapes"]:
        shape["points"] = [
            shape["points"][0] + 5,
            shape["points"][1],
            shape["points"][2] + 5,
            shape["points"][3],
        ]
    new_track = copy.deepcopy(tracks[1])
    new_track["attributes"] = []
    for t in (new_track, *tracks):
        t.pop("id", None)
        for s in t["shapes"]:
            s.pop("id", None)
    kept = [tracks[0], *tracks[1:-1], new_track]  # 마지막 트랙을 지운다
    setup.cvat.put_tracks(tid, kept)

    with pg.begin() as conn:
        outcome = collect_task(conn, task.task_key, setup, "rev01", FIXED_TIME)
        assert collect_task(conn, task.task_key, setup, "rev01", FIXED_TIME) is None  # 한 번만
        [stored] = list_review_tasks(conn, sid, ReviewStage.PRIVACY)
        current = current_labels(get_labels(conn, sid, kinds=["blur_track"]))
    assert outcome is not None
    assert (outcome.corrected, outcome.retracted, outcome.added) == (1, 1, 1)
    assert len(outcome.approved) == len(blur) - 2
    assert stored.status is ReviewTaskStatus.COLLECTED
    assert len(current) == len(blur)  # 수정 1(대체), 삭제 1, 추가 1
    assert all(x.verification.state is not VerificationState.UNREVIEWED for x in current)
    moved = next(x for x in current if x.parent_label_id == tracks[0]["attributes"][0]["value"])
    assert isinstance(moved.payload, BlurTrackPayload)
    with pg.begin() as conn:
        assert approve_session(conn, sid).privacy_state.value == "approved"


def test_labelers_only_get_watermarked_blurred_media(
    pg: sa.Engine, setup: ReviewSetup, tmp_path: Path
) -> None:
    sid, _ = _session(pg, setup, tmp_path)
    with pg.begin() as conn:
        [privacy] = create_privacy_tasks(conn, sid, setup, FIXED_TIME)
        collect_task(conn, privacy.task_key, setup, "rev01", FIXED_TIME)
        with pytest.raises(Exception, match="프라이버시 승인"):
            create_labeling_tasks(conn, sid, setup, "labeler01", FIXED_TIME)
        approve_session(conn, sid)
        render_session(conn, sid, setup.raw, setup.labeling, load_policy(ROOT))
        tasks = create_labeling_tasks(conn, sid, setup, "labeler01", FIXED_TIME)
    assert {t.tool.value for t in tasks} == {"cvat", "label_studio"}
    assert all(
        t.media_uri.startswith(f"s3://dlp-labeling/sessions/{sid}/review/labeler01/") for t in tasks
    )

    # Label Studio 작업의 URL은 라벨러 자격 증명으로 서명돼 라벨링 버킷만 열린다
    assert setup.label_studio is not None
    ls_task = next(t for t in tasks if t.task_key.startswith("label_studio:"))
    data = setup.label_studio.http.get(f"/api/tasks/{ls_task.external_id}").json()["data"]
    assert "dlp-raw" not in str(data)
    assert httpx.get(data["video"]).status_code == 200
    assert httpx.get(data["timeseries"]).text.startswith("time_ms,")
    raw_url = S3Store.labeler_from_env("dlp-raw").presign(f"sessions/{sid}/raw/bodycam.mp4")
    assert httpx.get(raw_url).status_code == 403
