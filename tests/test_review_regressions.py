"""전수 검사 후 회귀: 저장소 오류 경계, 0건 검수와 실제 PostgreSQL 동시성 (ADR 0032)."""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import uuid
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock, patch

import httpx
import numpy as np
import pytest
import sqlalchemy as sa
from sqlalchemy.exc import DBAPIError, IntegrityError

from dlp_cli.main import main
from dlp_cli.ops_cmds import cmd_privacy_audit
from dlp_datasets.build import build_dataset_version
from dlp_datasets.policy import load_policy as dataset_policy
from dlp_datasets.snapshot import (
    LakeFSSnapshotStore,
    LocalSnapshotStore,
    SnapshotError,
    SnapshotNotFoundError,
)
from dlp_export.policy import load_policy as export_policy
from dlp_export.runner import run_export
from dlp_export.source import ExportError, load_source
from dlp_fixtures.video import write_video
from dlp_media.storage import LocalStore, S3Store, sha256_file
from dlp_prelabel.adapters.owl_objects import OwlObjects
from dlp_prelabel.policy import load_policy as prelabel_policy
from dlp_privacy.detectors import build_detectors
from dlp_privacy.pipeline import StreamResult
from dlp_privacy.policy import load_policy as privacy_policy
from dlp_privacy.runner import (
    PrivacyGateError,
    RenderMeta,
    approve_session,
    detect_session,
    missing_privacy_reviews,
    render_hash,
    render_session,
    write_render_meta,
)
from dlp_review.clients import LabelStudioClient
from dlp_review.collect import collect_task
from dlp_review.labelstudio import to_ls_results
from dlp_review.ops.policy import load_policy as review_policy
from dlp_review.ops.runner import plan_session
from dlp_review.tasks import ReviewSetup
from dlp_schema.dataset import DatasetVersion, Split
from dlp_schema.db import repository as repo
from dlp_schema.db.migrate import upgrade
from dlp_schema.db.tables import model_versions
from dlp_schema.episode import current_labels
from dlp_schema.labels import Provenance, Source
from dlp_schema.lineage import ModelStatus, ModelVersion, TrainingRun
from dlp_schema.ontology import load_ontology
from dlp_schema.review import (
    AssignmentStatus,
    ReviewStage,
    ReviewTask,
    ReviewTaskStatus,
    ReviewTool,
)
from dlp_schema.session import PrivacyState, Session
from dlp_schema.testing import FIXED_TIME, make_label, make_session
from dlp_train.loop import TrainingError, deploy

ROOT = Path(__file__).resolve().parents[1]
NOW = FIXED_TIME + timedelta(hours=2)


@pytest.fixture
def pg() -> Iterator[sa.Engine]:
    """마이그레이션을 적용한 일회용 PostgreSQL. 실제 서비스 데이터는 건드리지 않는다."""
    url = sa.make_url(
        os.environ.get(
            "DLP_DATABASE_URL",
            "postgresql+psycopg://dlp:dlp-dev-password@localhost:5432/dlp",
        )
    )
    name = f"dlp_test_{uuid.uuid4().hex[:8]}"
    admin = sa.create_engine(url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(sa.text(f'CREATE DATABASE "{name}"'))
    test_url = url.set(database=name).render_as_string(hide_password=False)
    upgrade(test_url)
    engine = sa.create_engine(test_url)
    try:
        with engine.begin() as conn:
            repo.register_ontology(conn, load_ontology(ROOT / "config/ontology/v1"))
        yield engine
    finally:
        engine.dispose()
        with admin.connect() as conn:
            conn.execute(sa.text(f'DROP DATABASE "{name}" WITH (FORCE)'))
        admin.dispose()


def session(sid: str, privacy: PrivacyState = PrivacyState.APPROVED) -> Session:
    """바디캠 하나, 작업자·장소가 세션마다 다른 합성 메타데이터."""
    return make_session(
        sid,
        privacy_state=privacy,
        worker_id=f"w-{sid}",
        site_id=f"site-{sid}",
        streams=[
            {
                "stream_id": "bodycam",
                "kind": "bodycam",
                "sync_method": "reference",
                "uri": f"local://raw/sessions/{sid}/bodycam.mp4",
            }
        ],
    )


def task(
    sid: str,
    key: str,
    *,
    privacy: bool = False,
    sent: tuple[str, ...] = (),
    collected: bool = False,
    at: datetime = FIXED_TIME,
) -> ReviewTask:
    """검수 도구 대신 수집 조건만 기록한다. 담당자는 reviewer다."""
    return ReviewTask(
        task_key=key,
        tool=ReviewTool.CVAT if privacy else ReviewTool.LABEL_STUDIO,
        external_id=key.split(":")[1],
        session_id=sid,
        stream_id="bodycam",
        stage=ReviewStage.PRIVACY if privacy else ReviewStage.LABELING,
        assignee="reviewer",
        media_uri="local://raw/media.mp4",
        label_kinds=("blur_track",) if privacy else ("gap",),
        created_at=at,
        sent_label_ids=sent,
        status=ReviewTaskStatus.COLLECTED if collected else ReviewTaskStatus.OPEN,
        collected_at=at if collected else None,
    )


@pytest.mark.parametrize("status", [404, 401, 503])
def test_snapshot_http_errors_only_allow_404_fallback(status: int, tmp_path: Path) -> None:
    """lakeFS 객체 부재만 구버전으로 볼 수 있고 인증·서버 장애는 별도 오류다."""
    client = httpx.Client(
        base_url="http://lakefs/api/v1",
        transport=httpx.MockTransport(
            lambda req: httpx.Response(201 if req.method == "POST" else status),
        ),
    )
    with client, patch("dlp_datasets.snapshot.httpx.Client", return_value=client):
        store = LakeFSSnapshotStore(
            "http://lakefs",
            "test",
            "test",
            repository="r",
            branch="main",
            storage_namespace="local://data",
        )
        error = SnapshotNotFoundError if status == 404 else SnapshotError
        with pytest.raises(error) as caught:
            store.read("lakefs://r/commit/dv1", "sessions.jsonl", tmp_path / "s.jsonl")
        assert isinstance(caught.value, FileNotFoundError) is (status == 404)


def test_cli_rejects_evaluation_without_model(tmp_path: Path) -> None:
    """모델 생략 시 DB 조회·리포트 기록 전에 CLI 사용 오류로 실패한다."""
    out = tmp_path / "empty.json"
    with pytest.raises(SystemExit) as caught:
        main(["eval", "golden", "nonexistent", "--out", str(out)])
    assert caught.value.code == 2 and not out.exists()


def test_tokenizer_changes_both_owl_versions(tmp_path: Path) -> None:
    """가중치·정책이 같아도 토크나이저 버전이 바뀌면 프리라벨과 블러 재탐지 버전이 바뀐다."""
    versions: list[tuple[str, str]] = []

    def make_owl(_model: Path, _tokenizer: Path, queries: list[str]) -> Mock:
        """무거운 추론만 대신하고 질의·버전 결합은 실제 탐지기를 쓴다."""
        return Mock(queries=queries)

    for token_version in ("token-v1", "token-v2"):

        def resolve(_root: Path, name: str, token_version: str = token_version) -> tuple[Path, str]:
            return tmp_path / name, token_version if "tokenizer" in name else "weights-v1"

        with (
            patch("dlp_prelabel.adapters.owl_objects.resolve", side_effect=resolve),
            patch("dlp_privacy.detectors.resolve", side_effect=resolve),
            patch(
                "dlp_privacy.detectors.Owlv2",
                side_effect=make_owl,
            ),
            patch("dlp_privacy.detectors.YuNetFaceDetector"),
        ):
            prelabel = OwlObjects(
                ROOT, prelabel_policy(ROOT), ontology_version="1.0.0", now=FIXED_TIME
            )
            detectors, _ = build_detectors(privacy_policy(ROOT), ROOT)
            detector = next(d for name, d in detectors.items() if "open_vocab" in name)
            versions.append((prelabel.version, detector.version))
    assert versions[0][0] != versions[1][0] and versions[0][1] != versions[1][1]


@pytest.mark.services
def test_zero_detection_requires_new_review_and_allows_independent_audit(
    pg: sa.Engine,
    tmp_path: Path,
) -> None:
    """0건 재탐지는 승인·배정 세대를 바꾸며 오래된 검수를 거부한다. 새 검수 뒤 감사도 기록된다."""
    raw = LocalStore(tmp_path, "raw")
    video = tmp_path / "video.mp4"
    video.write_bytes(b"synthetic-video-stub")
    raw.put_file("sessions/zero/bodycam.mp4", video, sha256_file(video))
    policy = privacy_policy(ROOT)
    ontology = load_ontology(ROOT / "config/ontology/v1")
    review = review_policy(ROOT)
    review = review.model_copy(
        update={
            "reviewers": review.reviewers.model_copy(update={"privacy": ("reviewer",)}),
        }
    )
    with pg.begin() as conn:
        repo.insert_session(conn, session("zero", PrivacyState.PENDING))
    with (
        pg.begin() as conn,
        patch(
            "dlp_privacy.runner.detect_video", return_value=StreamResult([], [], {}, 10)
        ) as infer,
    ):
        first = detect_session(conn, "zero", raw, {}, {}, policy, FIXED_TIME)
        assert first.reran and first.detected == {"bodycam": 0}
        run = repo.latest_privacy_detection(conn, "zero", "bodycam")
        assert run is not None
        repo.insert_review_task(
            conn,
            task("zero", "cvat:1", privacy=True, collected=True).model_copy(
                update={"privacy_detection_id": run.event_id}
            ),
        )
        approve_session(conn, "zero")
        planned = plan_session(
            conn,
            "zero",
            ["reviewer"],
            review,
            ontology,
            seed=0,
            privacy=True,
            now=FIXED_TIME,
        )
        again = detect_session(conn, "zero", raw, {}, {}, policy, NOW)
        assert not again.reran and again.skipped == ["bodycam"] and infer.call_count == 1
        assert repo.get_session(conn, "zero").privacy_state is PrivacyState.APPROVED
    changed = policy.model_copy(update={"detection_threshold": policy.detection_threshold / 2})
    with (
        pg.begin() as conn,
        patch("dlp_privacy.runner.detect_video", return_value=StreamResult([], [], {}, 10)),
    ):
        rerun = detect_session(conn, "zero", raw, {}, {}, changed, NOW)
        assert rerun.reran and rerun.detected == {"bodycam": 0}
        assert repo.get_session(conn, "zero").privacy_state is PrivacyState.AUTO_BLURRED
        assert missing_privacy_reviews(conn, repo.get_session(conn, "zero")) == ["bodycam"]
        with pytest.raises(PrivacyGateError, match="검수"):
            approve_session(conn, "zero")
        new = plan_session(
            conn, "zero", ["reviewer"], review, ontology, seed=0, privacy=True, now=NOW
        )
        assert planned and new
        assert {a.assignment_id for a in planned}.isdisjoint(a.assignment_id for a in new)
        run = repo.latest_privacy_detection(conn, "zero", "bodycam")
        assert run is not None
        repo.insert_review_task(
            conn,
            task("zero", "cvat:2", privacy=True, collected=True, at=NOW).model_copy(
                update={"privacy_detection_id": run.event_id}
            ),
        )
        approve_session(conn, "zero")
        runs = repo.list_privacy_detections(conn, "zero")
        assert len(runs) == 2 and runs[1].at >= runs[0].at and runs[1].event_id > runs[0].event_id
        assert repo.latest_privacy_detection(conn, "zero", "bodycam") == runs[1]
    args = argparse.Namespace(
        session_id="zero",
        stream_id="bodycam",
        auditor="independent",
        misses=1,
        store="unused",
        url=pg.url.render_as_string(hide_password=False),
    )
    assert cmd_privacy_audit(args) == 0
    with pg.connect() as conn:
        audits = repo.list_privacy_audits(conn)
        assert len(audits) == 1 and audits[0].blur_reviewer == "reviewer"
    args.auditor = "reviewer"
    with pytest.raises(SystemExit, match="블러 검수자"):
        cmd_privacy_audit(args)
    for statement in (
        "UPDATE privacy_detection_runs SET at = now()",
        "DELETE FROM privacy_detection_runs",
        "TRUNCATE privacy_detection_runs",
    ):
        with pytest.raises(DBAPIError), pg.begin() as conn:
            conn.execute(sa.text(statement))


@pytest.mark.services
def test_completed_assignment_can_plan_new_pending_model(pg: sa.Engine) -> None:
    """완료 뒤 새 미검수 모델 라벨은 새 배정, 열린 배정과 검수 완료 입력은 중복 계획하지 않는다."""
    policy = review_policy(ROOT)
    ontology = load_ontology(ROOT / "config/ontology/v1")
    with pg.begin() as conn:
        repo.insert_session(conn, session("replan"))
        repo.insert_labels(
            conn,
            [
                make_label(
                    {"kind": "gap", "hand": "right", "gap_type": "unknown"},
                    label_id="plan-v1",
                    session_id="replan",
                )
            ],
        )
        first = plan_session(conn, "replan", ["reviewer"], policy, ontology, seed=0, now=FIXED_TIME)
        assert first
        for a in first:
            repo.update_assignment(
                conn, a.assignment_id, status=AssignmentStatus.DONE, completed_at=FIXED_TIME
            )
        assert not plan_session(conn, "replan", ["reviewer"], policy, ontology, seed=0, now=NOW)
        repo.insert_labels(
            conn,
            [
                make_label(
                    {"kind": "gap", "hand": "left", "gap_type": "unknown"},
                    label_id="plan-v2",
                    session_id="replan",
                    confidence=0.1,
                    provenance=Provenance(source=Source.MODEL, model_version="model-v2"),
                )
            ],
        )
        second = plan_session(conn, "replan", ["reviewer"], policy, ontology, seed=0, now=NOW)
        assert second and {a.assignment_id for a in first}.isdisjoint(
            a.assignment_id for a in second
        )
        assert not plan_session(conn, "replan", ["reviewer"], policy, ontology, seed=0, now=NOW)


def wait_for_db_lock(engine: sa.Engine, pid: int) -> None:
    """경쟁 스레드가 실제 DB 잠금을 기다리는지 확인한다 (API 장벽을 잠금 안에 두지 않는다)."""
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        with engine.connect() as conn:
            waiting = conn.execute(
                sa.text("SELECT wait_event_type FROM pg_stat_activity WHERE pid = :pid"),
                {"pid": pid},
            ).scalar_one()
        if waiting == "Lock":
            return
        time.sleep(0.01)
    pytest.fail("경쟁 수집이 세션 잠금을 기다리지 않았습니다")


@pytest.mark.services
def test_distinct_review_tasks_do_not_fork_parent(pg: sa.Engine, tmp_path: Path) -> None:
    """다른 두 작업이 같은 부모를 수정해도 둘째는 최신 이력을 보고 이전 결과를 버린다."""
    origin = make_label(
        {"kind": "gap", "hand": "right", "gap_type": "unknown"},
        label_id="race-parent",
        session_id="race",
        confidence=0.5,
        provenance=Provenance(source=Source.MODEL, model_version="model-1"),
    )
    with pg.begin() as conn:
        repo.insert_session(conn, session("race"))
        repo.insert_labels(conn, [origin])
        for n in (1, 2):
            repo.insert_review_task(
                conn, task("race", f"label_studio:{n}", sent=(origin.label_id,))
            )
    entered, release, second_started = threading.Event(), threading.Event(), threading.Event()
    pids: list[int] = []

    def latest_results(n: int):
        items = to_ls_results([origin])
        items[0]["value"]["end"] = 0.8 + n / 100
        if n == 1:
            entered.set()
            assert release.wait(10)
        return items

    ls = Mock(spec=LabelStudioClient)
    ls.latest_results.side_effect = latest_results
    ls.lead_seconds.return_value = 0
    setup = ReviewSetup(
        raw=LocalStore(tmp_path, "raw"),
        labeling=LocalStore(tmp_path, "labeling"),
        labeling_reader=Mock(spec=S3Store),
        ontology=load_ontology(ROOT / "config/ontology/v1"),
        label_studio=ls,
    )

    def collect(n: int):
        with pg.begin() as conn:
            if n == 2:
                pids.append(conn.execute(sa.text("SELECT pg_backend_pid()")).scalar_one())
                second_started.set()
            return collect_task(conn, f"label_studio:{n}", setup, "reviewer", NOW)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(collect, 1)
        try:
            assert entered.wait(5)
            second = pool.submit(collect, 2)
            assert second_started.wait(5)
            wait_for_db_lock(pg, pids[0])
        finally:
            release.set()
        outcomes = [first.result(timeout=10), second.result(timeout=10)]
    assert all(o is not None for o in outcomes)
    with pg.connect() as conn:
        current = current_labels(repo.get_labels(conn, "race"))
        assert len(current) == 1 and current[0].parent_label_id == origin.label_id
        assert all(
            repo.get_review_task(conn, f"label_studio:{n}").status is ReviewTaskStatus.COLLECTED
            for n in (1, 2)
        )


@pytest.mark.services
@pytest.mark.parametrize("status", [ModelStatus.CANDIDATE, ModelStatus.PASSED])
def test_concurrent_first_deployment_rejects_stale_baseline(
    pg: sa.Engine,
    tmp_path: Path,
    status: ModelStatus,
) -> None:
    """두 첫 배포는 과제 잠금 안에서 비교 기준을 다시 읽어 하나만 배포된다. 자동 배포도 같다.

    두 모델 모두 자기 후보 ID와 게이트 통과가 적힌 리포트를 가진다 (deploy가 검사한다, T4).
    """
    artifacts = LocalStore(tmp_path, "artifacts")
    for n in (1, 2):
        report = tmp_path / f"golden-{n}.json"
        report.write_text(
            json.dumps(
                {
                    "model_versions": {"objects": f"model-{n}"},
                    "gate": {"passed": True, "tasks": {"objects": {"passed": True}}},
                    "deployed_baseline": None,
                }
            )
        )
        artifacts.put_file(f"golden-{n}.json", report, sha256_file(report))
    with pg.begin() as conn:
        repo.insert_dataset_version(
            conn,
            DatasetVersion(
                version_id="dv1",
                ontology_version="1.0.0",
                created_at=FIXED_TIME,
                snapshot_uri="local-snapshot://unused/dv1",
                splits={},
            ),
        )
        for n in (1, 2):
            repo.insert_training_run(
                conn,
                TrainingRun(
                    run_id=f"train-{n}",
                    dataset_version_id="dv1",
                    model_name="objects",
                    model_version=f"model-{n}",
                    created_at=FIXED_TIME,
                ),
            )
            repo.insert_model_version(
                conn,
                ModelVersion(
                    model_version=f"model-{n}",
                    task="objects",
                    run_id=f"train-{n}",
                    trainer="oracle-stub",
                    artifact_uri="local://artifacts/model",
                    sha256="a" * 64,
                    train_examples=1,
                    status=status,
                    created_at=FIXED_TIME,
                    report_uri=artifacts.uri(f"golden-{n}.json"),
                    decided_at=None if status is ModelStatus.CANDIDATE else FIXED_TIME,
                ),
            )
    barrier = threading.Barrier(2)

    def deploy_one(n: int) -> bool:
        barrier.wait(timeout=5)
        try:
            with pg.begin() as conn:
                deploy(
                    conn,
                    f"model-{n}",
                    now=NOW,
                    report_uri=artifacts.uri(f"golden-{n}.json"),
                    artifacts=artifacts,
                )
            return True
        except TrainingError as exc:
            assert "지금 배포 모델" in str(exc)
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sum(pool.map(deploy_one, (1, 2))) == 1
    with pg.connect() as conn:
        [deployed] = repo.list_model_versions(conn, "objects", ModelStatus.DEPLOYED)
    # 저장소 함수 밖의 쓰기도 부분 유일 색인이 거부한다.
    with pytest.raises(IntegrityError), pg.begin() as conn:
        conn.execute(
            model_versions.update()
            .where(model_versions.c.model_version != deployed.model_version)
            .values(status="deployed")
        )


@pytest.mark.services
@pytest.mark.parametrize("failure", ["missing", "outage", "incomplete"])
def test_snapshot_metadata_fallback_is_only_for_absent_file(
    pg: sa.Engine,
    tmp_path: Path,
    failure: str,
) -> None:
    """없던 구버전 메타데이터만 DB로 대체한다. 장애·존재하는 파일의 세션 누락은 실패한다."""

    class MetadataStore(LocalSnapshotStore):
        def read(self, snapshot_uri: str, path: str, dest: Path) -> None:
            if path == "sessions.jsonl":
                if failure == "missing":
                    raise SnapshotNotFoundError("legacy missing metadata")
                if failure == "outage":
                    raise SnapshotError("HTTP 503")
                dest.write_text("")
                return
            dest.write_text("")  # 이 시나리오는 라벨이 없는 세션이다.

    with pg.begin() as conn:
        repo.insert_session(conn, session("snapshot"))
        repo.insert_dataset_version(
            conn,
            DatasetVersion(
                version_id="dv1",
                ontology_version="1.0.0",
                created_at=FIXED_TIME,
                snapshot_uri="local-snapshot://unused/dv1",
                splits={"snapshot": Split.TRAIN},
            ),
        )
    with pg.connect() as conn:
        args = (conn, MetadataStore(tmp_path), "dv1", export_policy(ROOT))
        if failure == "missing":
            assert (
                load_source(*args, include_unreviewed=False).sessions[0].session.session_id
                == "snapshot"
            )
        else:
            with pytest.raises(SnapshotError if failure == "outage" else ExportError):
                load_source(*args, include_unreviewed=False)


@pytest.mark.services
@pytest.mark.parametrize("change", ["approval", "blur"])
@pytest.mark.parametrize("fmt", ["intervals", "coco"])
def test_export_does_not_publish_after_privacy_changes_during_upload(
    pg: sa.Engine,
    tmp_path: Path,
    change: str,
    fmt: str,
) -> None:
    """업로드 중 승인 취소 또는 새 블러가 생기면 마지막 재검사로 완료 manifest를 차단한다."""
    snapshots = LocalSnapshotStore(tmp_path / "snap")
    labeling = LocalStore(tmp_path / "store", "labeling")
    with pg.begin() as conn:
        repo.insert_session(conn, session("export"))
        repo.insert_labels(
            conn,
            [
                make_label(
                    {"kind": "gap", "hand": "right", "gap_type": "unknown"},
                    label_id="export-gap",
                    session_id="export",
                )
            ],
        )
        if fmt == "coco":
            repo.insert_labels(
                conn,
                [
                    make_label(
                        {
                            "kind": "box_track",
                            "entity_id": "cup_1",
                            "class_id": "cup",
                            "keyframes": [{"t_ms": 0, "x": 1, "y": 1, "w": 10, "h": 10}],
                        },
                        label_id="export-box",
                        session_id="export",
                        stream_id="bodycam",
                        t_start_ms=0,
                        t_end_ms=0,
                    )
                ],
            )
        build_dataset_version(
            conn,
            snapshots,
            dataset_policy(ROOT),
            version_id="dv1",
            ontology_version="1.0.0",
            golden_set_version=None,
            now=FIXED_TIME,
        )
    if fmt == "coco":
        clip = tmp_path / "blurred.mp4"
        write_video(
            clip,
            [
                (0, np.zeros((48, 64, 3), dtype=np.uint8)),
                (100, np.zeros((48, 64, 3), dtype=np.uint8)),
            ],
            width=64,
            height=48,
        )
        labeling.put_file("sessions/export/blurred/bodycam.mp4", clip, sha256_file(clip))
        write_render_meta(
            labeling,
            "export",
            "bodycam",
            RenderMeta(render_hash([], privacy_policy(ROOT)), sha256_file(clip)),
            tmp_path,
        )

    class ChangeDuringUpload(LocalStore):
        changed = False

        def put_file(self, key: str, path: Path, sha256: str) -> None:
            super().put_file(key, path, sha256)
            if self.changed:
                return
            self.changed = True
            with pg.begin() as conn:
                if change == "approval":
                    repo.set_privacy_state(conn, "export", PrivacyState.AUTO_BLURRED)
                else:
                    repo.insert_labels(
                        conn,
                        [
                            make_label(
                                {
                                    "kind": "blur_track",
                                    "target": "face",
                                    "keyframes": [{"t_ms": 0, "x": 1, "y": 1, "w": 10, "h": 10}],
                                },
                                label_id="new-blur",
                                session_id="export",
                                stream_id="bodycam",
                            )
                        ],
                    )

    datasets = ChangeDuringUpload(tmp_path / "store", "datasets")
    with pytest.raises(ExportError):
        run_export(
            pg,
            root=ROOT,
            version_id="dv1",
            fmt=fmt,  # type: ignore[arg-type]
            target="synthetic",
            snapshots=snapshots,
            labeling=labeling,
            datasets=datasets,
            raw_bucket="raw",
            policy=export_policy(ROOT),
            ontology=load_ontology(ROOT / "config/ontology/v1"),
            include_unreviewed=False,
            splits=tuple(Split),
            now=NOW,
            id_secret=b"synthetic",
        )
    assert datasets.changed
    assert not list((tmp_path / "store" / "datasets" / "exports").rglob("manifest.json"))
    if fmt == "coco":
        assert list((tmp_path / "store" / "datasets" / "exports").rglob("*.jpg"))


@pytest.mark.services
def test_rolled_back_zero_detection_marker_requires_new_review(
    pg: sa.Engine,
    tmp_path: Path,
) -> None:
    """DB 롤백 뒤 외부 0건 표시만 남아도 과거 작업으로 승인하지 않고 확인 시각을 기록한다."""
    raw = LocalStore(tmp_path, "raw")
    video = tmp_path / "video.mp4"
    video.write_bytes(b"synthetic-video-stub")
    raw.put_file("sessions/rollback/bodycam.mp4", video, sha256_file(video))
    policy = privacy_policy(ROOT)
    with pg.begin() as conn:
        repo.insert_session(conn, session("rollback", PrivacyState.PENDING))
        repo.insert_review_task(conn, task("rollback", "cvat:old", privacy=True, collected=True))
    with (
        pytest.raises(RuntimeError, match="rollback"),
        pg.begin() as conn,
        patch(
            "dlp_privacy.runner.detect_video",
            return_value=StreamResult([], [], {}, 10),
        ),
    ):
        detect_session(conn, "rollback", raw, {}, {}, policy, FIXED_TIME)
        raise RuntimeError("rollback")
    with pg.begin() as conn, patch("dlp_privacy.runner.detect_video") as infer:
        summary = detect_session(conn, "rollback", raw, {}, {}, policy, NOW)
        assert summary.reran and summary.detected == {"bodycam": 0}
        infer.assert_not_called()
        [run] = repo.list_privacy_detections(conn, "rollback")
        assert run.at > NOW
        with pytest.raises(PrivacyGateError, match="검수"):
            approve_session(conn, "rollback")


@pytest.mark.services
def test_export_holds_session_lock_until_manifest_is_published(
    pg: sa.Engine,
    tmp_path: Path,
) -> None:
    """최종 재검사 뒤의 승인 취소도 manifest 게시까지 DB에서 기다린다."""
    snapshots = LocalSnapshotStore(tmp_path / "snap")
    with pg.begin() as conn:
        repo.insert_session(conn, session("publish"))
        repo.insert_labels(
            conn,
            [
                make_label(
                    {"kind": "gap", "hand": "right", "gap_type": "unknown"},
                    label_id="publish-gap",
                    session_id="publish",
                )
            ],
        )
        build_dataset_version(
            conn,
            snapshots,
            dataset_policy(ROOT),
            version_id="dv1",
            ontology_version="1.0.0",
            golden_set_version=None,
            now=FIXED_TIME,
        )
    started = threading.Event()
    pids: list[int] = []

    def revoke() -> None:
        with pg.begin() as conn:
            pids.append(conn.execute(sa.text("SELECT pg_backend_pid()")).scalar_one())
            started.set()
            repo.set_privacy_state(conn, "publish", PrivacyState.AUTO_BLURRED)

    with ThreadPoolExecutor(max_workers=1) as pool:

        class ObservePublication(LocalStore):
            revocation: Future[None] | None = None

            def put_file(self, key: str, path: Path, sha256: str) -> None:
                if key.endswith("/manifest.json"):
                    self.revocation = pool.submit(revoke)
                    assert started.wait(5)
                    wait_for_db_lock(pg, pids[0])
                    assert not self.revocation.done()
                super().put_file(key, path, sha256)

        datasets = ObservePublication(tmp_path / "store", "datasets")
        result = run_export(
            pg,
            root=ROOT,
            version_id="dv1",
            fmt="intervals",
            target="synthetic",
            snapshots=snapshots,
            labeling=LocalStore(tmp_path / "store", "labeling"),
            datasets=datasets,
            raw_bucket="raw",
            policy=export_policy(ROOT),
            ontology=load_ontology(ROOT / "config/ontology/v1"),
            include_unreviewed=False,
            splits=tuple(Split),
            now=NOW,
            id_secret=b"synthetic",
        )
        assert datasets.revocation is not None
        datasets.revocation.result(timeout=5)
    assert datasets.head(f"exports/{result.record.export_id}/manifest.json") is not None
    with pg.connect() as conn:
        assert repo.get_session(conn, "publish").privacy_state is PrivacyState.AUTO_BLURRED


def privacy_setup(tmp_path: Path, sid: str) -> ReviewSetup:
    """2프레임 합성 원본·프록시와 모의 CVAT. DB·작업 생성 경로는 실제다."""
    from dlp_review.clients import CvatClient

    raw = LocalStore(tmp_path, "raw")
    video = tmp_path / "tiny.mp4"
    write_video(
        video,
        [(0, np.zeros((48, 64, 3), dtype=np.uint8)), (100, np.zeros((48, 64, 3), dtype=np.uint8))],
        width=64,
        height=48,
    )
    for key in (f"sessions/{sid}/bodycam.mp4", f"sessions/{sid}/derived/bodycam.proxy.mp4"):
        raw.put_file(key, video, sha256_file(video))
    cvat = Mock(spec=CvatClient)
    cvat.find_user_id.return_value = 1
    cvat.create_task.return_value = 1
    cvat.jobs.return_value = []
    conf = review_policy(ROOT).cvat.model_copy(update={"users": {"reviewer": "cvat-reviewer"}})
    return ReviewSetup(
        raw=raw,
        labeling=LocalStore(tmp_path, "labeling"),
        labeling_reader=Mock(spec=S3Store),
        ontology=load_ontology(ROOT / "config/ontology/v1"),
        cvat=cvat,
        cvat_config=conf,
    )


@pytest.mark.services
def test_review_during_model_loading_cannot_approve_new_detection(
    pg: sa.Engine,
    tmp_path: Path,
) -> None:
    """CLI 시작 뒤 모델 로딩 중 끝난 이전 세대 검수는 그 뒤의 새 0건 탐지를 승인하지 못한다."""
    from types import SimpleNamespace

    from dlp_cli.privacy_cmds import cmd_detect
    from dlp_review.cvat import UNIT_SCALE, CvatSchema
    from dlp_review.tasks import create_privacy_tasks

    sid = "model-loading"
    setup = privacy_setup(tmp_path, sid)
    policy = privacy_policy(ROOT)
    previous = policy.model_copy(update={"detection_threshold": policy.detection_threshold / 2})
    with (
        pg.begin() as conn,
        patch("dlp_privacy.runner.detect_video", return_value=StreamResult([], [], {}, 2)),
    ):
        repo.insert_session(conn, session(sid, PrivacyState.PENDING))
        detect_session(conn, sid, setup.raw, {}, {}, previous, FIXED_TIME)
        old = repo.latest_privacy_detection(conn, sid, "bodycam")
        assert old is not None
        repo.insert_review_task(
            conn,
            task(sid, "cvat:99", privacy=True, collected=True).model_copy(
                update={"privacy_detection_id": old.event_id}
            ),
        )
        approve_session(conn, sid)

    def loaded(*_args: object, **_kwargs: object) -> SimpleNamespace:
        with pg.begin() as conn:
            [review] = create_privacy_tasks(
                conn, sid, setup, NOW, assignee="reviewer", privacy_reviewers=["reviewer"]
            )
            repo.mark_review_task_collected(conn, review.task_key, NOW)
        return SimpleNamespace(notes=[], predictors=[])

    args = argparse.Namespace(
        session_id=sid, store="unused", url=pg.url.render_as_string(hide_password=False)
    )
    with (
        patch("dlp_cli.privacy_cmds.raw_store", return_value=setup.raw),
        patch("dlp_cli.privacy_cmds.build_detectors", return_value=({}, {})),
        patch("dlp_cli.privacy_cmds.store_from_spec", return_value=setup.labeling),
        patch("dlp_cli.privacy_cmds.deployed_predictors", side_effect=loaded),
        patch("dlp_review.tasks._cvat_project", return_value=(1, CvatSchema.from_labels([]))),
        patch("dlp_review.tasks.label_scale", return_value=UNIT_SCALE),
        patch("dlp_privacy.runner.detect_video", return_value=StreamResult([], [], {}, 2)),
    ):
        assert cmd_detect(args) == 0
    with pg.begin() as conn:
        review = repo.get_review_task(conn, "cvat:1")
        new = repo.latest_privacy_detection(conn, sid, "bodycam")
        assert new is not None and new.event_id != review.privacy_detection_id
        assert review.privacy_detection_id == old.event_id
        assert new.at > review.created_at
        with pytest.raises(PrivacyGateError, match="검수"):
            approve_session(conn, sid)
        # 시각을 미래로 올려도 이전 세대라는 사실은 바뀌지 않는다.
        repo.insert_review_task(
            conn,
            review.model_copy(
                update={
                    "task_key": "cvat:2",
                    "created_at": new.at + timedelta(days=10),
                }
            ),
        )
        assert missing_privacy_reviews(conn, repo.get_session(conn, sid)) == ["bodycam"]


@pytest.mark.services
def test_task_creation_waits_for_detection_and_binds_committed_generation(
    pg: sa.Engine,
    tmp_path: Path,
) -> None:
    """추론 중 작업 생성은 DB 잠금을 기다린 뒤 새 세대·라벨을 함께 보낸다."""
    from dlp_review.cvat import UNIT_SCALE, CvatSchema
    from dlp_review.tasks import create_privacy_tasks

    sid = "detect-create"
    setup = privacy_setup(tmp_path, sid)
    policy = privacy_policy(ROOT)
    with pg.begin() as conn:
        repo.insert_session(conn, session(sid, PrivacyState.PENDING))
    entered, release, started = threading.Event(), threading.Event(), threading.Event()
    pids: list[int] = []

    def infer(*_args: object, **_kwargs: object) -> StreamResult:
        entered.set()
        assert release.wait(10)
        return StreamResult([], [], {}, 2)

    def detect() -> None:
        with pg.begin() as conn:
            detect_session(conn, sid, setup.raw, {}, {}, policy, FIXED_TIME)

    def create() -> list[ReviewTask]:
        with pg.begin() as conn:
            pids.append(conn.execute(sa.text("SELECT pg_backend_pid()")).scalar_one())
            started.set()
            return create_privacy_tasks(
                conn, sid, setup, FIXED_TIME, assignee="reviewer", privacy_reviewers=["reviewer"]
            )

    with (
        patch("dlp_privacy.runner.detect_video", side_effect=infer),
        patch("dlp_review.tasks._cvat_project", return_value=(1, CvatSchema.from_labels([]))),
        patch("dlp_review.tasks.label_scale", return_value=UNIT_SCALE),
        ThreadPoolExecutor(max_workers=2) as pool,
    ):
        detecting = pool.submit(detect)
        try:
            assert entered.wait(5)
            creating = pool.submit(create)
            assert started.wait(5)
            wait_for_db_lock(pg, pids[0])
        finally:
            release.set()
        detecting.result(timeout=10)
        [review] = creating.result(timeout=10)
    with pg.begin() as conn:
        run = repo.latest_privacy_detection(conn, sid, "bodycam")
        assert run is not None and review.privacy_detection_id == run.event_id
        repo.mark_review_task_collected(conn, review.task_key, NOW)
        assert approve_session(conn, sid).privacy_state is PrivacyState.APPROVED


@pytest.mark.services
@pytest.mark.parametrize("empty", [False, True])
def test_privacy_policy_rollback_creates_new_generation(
    pg: sa.Engine,
    tmp_path: Path,
    empty: bool,
) -> None:
    """v1→v2→v1→v2 복귀는 새 세대로 탐지한다. 0건 표시도 과거 결과로 재활성화하지 않는다."""
    from dlp_privacy.pipeline import model_version

    sid = "policy-rollback"
    setup = privacy_setup(tmp_path, sid)
    v1 = privacy_policy(ROOT)
    v2 = v1.model_copy(update={"detection_threshold": v1.detection_threshold / 2})
    with pg.begin() as conn:
        repo.insert_session(conn, session(sid, PrivacyState.PENDING))
    with patch("dlp_privacy.runner.detect_video") as infer:
        for i, policy in enumerate((v1, v2, v1, v2)):
            version = model_version({}, policy)
            label = make_label(
                {
                    "kind": "blur_track",
                    "target": "face",
                    "keyframes": [{"t_ms": 0, "x": 1, "y": 1, "w": 10, "h": 10}],
                },
                label_id="blur-" + version.split("+p")[-1],
                session_id=sid,
                stream_id="bodycam",
                confidence=0.9,
                provenance=Provenance(source=Source.MODEL, model_version=version),
            )
            infer.return_value = StreamResult([] if empty else [label], [], {}, 2)
            with pg.begin() as conn:
                summary = detect_session(
                    conn, sid, setup.raw, {}, {}, policy, FIXED_TIME + timedelta(minutes=i)
                )
                assert summary.reran and not summary.skipped
                current = current_labels(repo.get_labels(conn, sid))
                assert not current if empty else len(current) == 1
                assert all(x.provenance.model_version == version for x in current)
                run = repo.latest_privacy_detection(conn, sid, "bodycam")
                assert run is not None and run.model_versions == (version,)
        assert infer.call_count == 4
        with pg.begin() as conn:
            summary = detect_session(conn, sid, setup.raw, {}, {}, v2, NOW)
            assert summary.skipped == ["bodycam"] and not summary.reran
            assert len(repo.list_privacy_detections(conn, sid)) == 4
            assert infer.call_count == 4


@pytest.mark.services
def test_cli_partial_requested_evaluation_fails(pg: sa.Engine, tmp_path: Path) -> None:
    """objects 완벽 정답·예측만 있는 골든셋에 hands도 요청하면 실패·미평가 이유를 쓴다."""
    from dlp_schema.lineage import GoldenSet
    from dlp_schema.session import Domain

    sid = "partial-evaluation"
    truth = make_label(
        {
            "kind": "box_track",
            "entity_id": "cup_01",
            "class_id": "cup",
            "keyframes": [{"t_ms": t, "x": 1, "y": 1, "w": 10, "h": 10} for t in (0, 100, 200)],
        },
        session_id=sid,
        stream_id="bodycam",
        t_end_ms=200,
    )
    prediction = truth.model_copy(
        update={
            "label_id": "model-box",
            "confidence": 0.9,
            "provenance": Provenance(source=Source.MODEL, model_version="objects-v1"),
        }
    )
    with pg.begin() as conn:
        repo.insert_session(conn, session(sid))
        repo.insert_labels(conn, [truth, prediction])
        repo.insert_golden_set(
            conn,
            GoldenSet(
                version="partial-g1",
                domain=Domain.CLEANING,
                session_ids=(sid,),
                created_at=FIXED_TIME,
            ),
        )
    out = tmp_path / "partial.json"
    code = main(
        [
            "eval",
            "golden",
            "partial-g1",
            "--model",
            "objects=objects-v1",
            "--model",
            "hands=hands-v1",
            "--out",
            str(out),
            "--url",
            pg.url.render_as_string(hide_password=False),
        ]
    )
    result = json.loads(out.read_text())
    assert code == 1 and not result["gate"]["passed"]
    assert result["overall"]["objects"]["metrics"]["hota"] == 1.0
    # 정답을 고정하지 않은 이전 형식 골든셋은 리포트에 그렇게 남는다 (ADR 0047)
    assert result["golden_truth"]["truth_frozen"] is False
    assert result["gate"]["tasks"]["objects"]["passed"]
    assert not result["gate"]["tasks"]["hands"]["passed"]
    assert "미평가" in result["gate"]["tasks"]["hands"]["reasons"][0]


@pytest.mark.services
def test_review_detection_reference_is_scoped_and_clock_independent(pg: sa.Engine) -> None:
    """세대 FK는 다른 세션·스트림을 거부한다. 완료 시각이 역행해도 최신 일련번호를 고른다."""
    with pg.begin() as conn:
        repo.insert_session(conn, session("generation-a"))
        repo.insert_session(conn, session("generation-b"))
        first = repo.record_privacy_detection(conn, "generation-a", "bodycam", ["v1"], NOW)
        second = repo.record_privacy_detection(conn, "generation-a", "bodycam", ["v2"], FIXED_TIME)
        latest = repo.latest_privacy_detection(conn, "generation-a", "bodycam")
        assert latest is not None and latest.event_id == second > first
        bound = task("generation-a", "cvat:bound", privacy=True, collected=True).model_copy(
            update={"privacy_detection_id": second}
        )
        repo.insert_review_task(conn, bound)
        assert repo.get_review_task(conn, bound.task_key) == bound
        assert missing_privacy_reviews(conn, repo.get_session(conn, "generation-a")) == []
    for changed in ({"session_id": "generation-b"}, {"stream_id": "missing"}):
        with pytest.raises(IntegrityError), pg.begin() as conn:
            repo.insert_review_task(
                conn,
                bound.model_copy(
                    update={
                        "task_key": "cvat:invalid",
                        **changed,
                    }
                ),
            )


@pytest.mark.services
def test_removing_and_restoring_extra_model_updates_active_generation(
    pg: sa.Engine,
    tmp_path: Path,
) -> None:
    """배포 모델 제거는 결과·우선 구간을 철회한다. 복귀는 추가 모델만 재추론한다."""
    from dlp_privacy.pipeline import model_version
    from dlp_privacy.runner import read_review_segments
    from dlp_schema.labels import LabelRecord
    from dlp_schema.predictor import Clip

    sid = "extra-model"
    setup = privacy_setup(tmp_path, sid)
    policy = privacy_policy(ROOT)

    class Extra:
        name = "extra"
        version = "privacy-extra-1"
        calls = 0

        def run(self, clip: Clip) -> list[LabelRecord]:
            self.calls += 1
            return [
                make_label(
                    {
                        "kind": "blur_track",
                        "target": "face",
                        "keyframes": [{"t_ms": 0, "x": 1, "y": 1, "w": 10, "h": 10}],
                    },
                    label_id="extra-fixed",
                    session_id=clip.session_id,
                    stream_id=clip.stream_id,
                    confidence=0.9,
                    provenance=Provenance(source=Source.MODEL, model_version=self.version),
                )
            ]

    model = Extra()
    with (
        pg.begin() as conn,
        patch("dlp_privacy.runner.detect_video", return_value=StreamResult([], [], {}, 2)) as infer,
    ):
        repo.insert_session(conn, session(sid, PrivacyState.PENDING))
        detect_session(conn, sid, setup.raw, {}, {}, policy, FIXED_TIME, extra=[model])
        assert model.calls == 1 and infer.call_count == 1
        run = repo.latest_privacy_detection(conn, sid, "bodycam")
        assert run is not None and set(run.model_versions) == {
            model.version,
            model_version({}, policy),
        }
        summary = detect_session(conn, sid, setup.raw, {}, {}, policy, NOW)
        assert summary.reran and summary.changed
        assert not current_labels(repo.get_labels(conn, sid))
        assert not read_review_segments(setup.raw, sid, "bodycam", tmp_path)
        assert model.calls == 1 and infer.call_count == 1
        restored = detect_session(conn, sid, setup.raw, {}, {}, policy, NOW, extra=[model])
        assert restored.reran and model.calls == 2 and infer.call_count == 1
        current = current_labels(repo.get_labels(conn, sid))
        assert len(current) == 1 and current[0].provenance.model_version == model.version
        assert current[0].label_id != "extra-fixed"


@pytest.mark.services
@pytest.mark.parametrize("first", ["create", "collect"])
def test_direct_labeling_creation_and_blur_collect_never_miss_revocation(
    pg: sa.Engine,
    tmp_path: Path,
    first: str,
) -> None:
    """F-REV-02(ADR 0039): 직접 라벨링 작업 생성과 블러 수집이 겹쳐도 회수에서 빠지는 작업이 없다.

    생성이 먼저 세션을 잡으면 수집이 기다렸다가 그 작업까지 회수 요청한다 (REVOKED). 수집이 먼저면
    생성이 기다렸다가 승인이 풀린 것을 보고 TaskError. 둘 다 실제 PostgreSQL 잠금으로 직렬화된다.
    """
    from dlp_review import collect as collect_module
    from dlp_review.cvat import CvatSchema
    from dlp_review.reconcile import ReviewedItem
    from dlp_review.tasks import TaskError, create_labeling_tasks, proxy_key
    from dlp_schema.labels import BlurTrackPayload, BoxKeyframe

    sid = f"race-revoke-{first}"
    setup = privacy_setup(tmp_path, sid)
    policy = privacy_policy(ROOT)
    setup = ReviewSetup(
        raw=setup.raw, labeling=setup.labeling, labeling_reader=setup.labeling_reader,
        ontology=setup.ontology, cvat=setup.cvat, cvat_config=setup.cvat_config, privacy=policy,
    )  # fmt: skip
    assert setup.cvat is not None
    cvat = cast(Mock, setup.cvat)
    cvat.http = Mock()  # spec에 없는 인스턴스 속성 (수집이 작업의 프로젝트 ID를 읽는다)
    cvat.http.get.return_value.json.return_value = {"project_id": 1}
    with pg.begin() as conn:
        repo.insert_session(conn, session(sid))
        repo.insert_review_task(
            conn,
            task(sid, "cvat:500", privacy=True).model_copy(
                update={"media_uri": setup.raw.uri(proxy_key(sid, "bodycam"))}
            ),
        )
        render_session(conn, sid, setup.raw, setup.labeling, policy)
    entered, release, started = threading.Event(), threading.Event(), threading.Event()
    pids: list[int] = []

    def hold(*_args: object, **_kwargs: object) -> Any:
        entered.set()
        assert release.wait(10)
        return 42 if first == "create" else {"tracks": [], "shapes": [], "tags": []}

    face = BlurTrackPayload(target="face", keyframes=(BoxKeyframe(t_ms=0, x=1, y=1, w=8, h=8),))

    def run_create() -> Any:
        with pg.begin() as conn:
            if first == "collect":
                pids.append(conn.execute(sa.text("SELECT pg_backend_pid()")).scalar_one())
                started.set()
            return create_labeling_tasks(conn, sid, setup, "reviewer", NOW, tools={ReviewTool.CVAT})

    def run_collect() -> Any:
        with pg.begin() as conn:
            if first == "create":
                pids.append(conn.execute(sa.text("SELECT pg_backend_pid()")).scalar_one())
                started.set()
            return collect_task(conn, "cvat:500", setup, "reviewer", NOW)

    if first == "create":
        cvat.create_task.side_effect = hold
    else:
        cvat.create_task.return_value = 42
        cvat.get_annotations.side_effect = hold
    with (
        patch("dlp_review.tasks._cvat_project", return_value=(1, CvatSchema.from_labels([]))),
        patch.object(collect_module, "CvatSchema", Mock()),
        patch.object(collect_module, "annotation_tracks", Mock(return_value=[])),
        patch.object(
            collect_module,
            "from_cvat_tracks",
            Mock(return_value=[ReviewedItem(None, "bodycam", 0, 100, face)]),
        ),
        patch("dlp_review.collect.label_scale", return_value=(1.0, 1.0)),
        ThreadPoolExecutor(max_workers=2) as pool,
    ):
        leader = pool.submit(run_create if first == "create" else run_collect)
        try:
            assert entered.wait(10)
            follower = pool.submit(run_collect if first == "create" else run_create)
            assert started.wait(5)
            wait_for_db_lock(pg, pids[0])
        finally:
            release.set()
        made: list[str] = []
        if first == "create":
            made = [t.task_key for t in leader.result(timeout=20)]
            follower.result(timeout=20)
        else:
            leader.result(timeout=20)
            with pytest.raises(TaskError, match="승인"):
                follower.result(timeout=20)
    with pg.connect() as conn:
        tasks = [t for t in repo.list_review_tasks(conn, sid) if t.stage is ReviewStage.LABELING]
        assert repo.get_session(conn, sid).privacy_state is PrivacyState.AUTO_BLURRED
    # 생성이 먼저면 그 작업이 회수되고(REVOKED), 수집이 먼저면 작업이 생기지 않는다
    assert [t.task_key for t in tasks] == made
    assert all(t.status is ReviewTaskStatus.REVOKED for t in tasks)
    assert len(made) == (1 if first == "create" else 0)
