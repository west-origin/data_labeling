"""검수 운영 흐름 (Label Studio): 계획 → 블라인드·오류 삽입 작업 → 검수 → 수집 → 품질 리포트."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

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
from dlp_review.cvat import CvatSchema, to_cvat_tracks
from dlp_review.labelstudio import LS_KINDS, to_ls_results
from dlp_review.ops.policy import CvatPolicy
from dlp_review.ops.policy import load_policy as load_ops_policy
from dlp_review.ops.runner import (
    AccessError,
    create_assignment_tasks,
    plan_session,
    quality_report,
)
from dlp_review.ops.seeding import seed_prefix
from dlp_review.tasks import ReviewSetup, frame_times, object_key
from dlp_schema.db.migrate import upgrade
from dlp_schema.db.repository import (
    get_assignment,
    get_labels,
    insert_golden_set,
    insert_labels,
    insert_review_task,
    list_review_tasks,
    record_review,
    register_ontology,
)
from dlp_schema.episode import current_labels
from dlp_schema.labels import LabelRecord, Provenance, Source, VerificationState
from dlp_schema.lineage import GoldenSet
from dlp_schema.ontology import load_ontology
from dlp_schema.review import (
    AssignmentStatus,
    ReviewMode,
    ReviewStage,
    ReviewTask,
    ReviewTaskStatus,
    ReviewTool,
)
from dlp_schema.session import Domain
from dlp_schema.testing import FIXED_TIME

pytestmark = pytest.mark.services
ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def pg() -> Iterator[sa.Engine]:
    """일회용 PostgreSQL DB (마이그레이션 + 온톨로지 v1 등록). 테스트 뒤 지운다.

    `DLP_DATABASE_URL`(기본: 개발 compose의 DB)에 접속해 무작위 이름의 DB를 만든다.
    """
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
def setup(cvat_config: CvatPolicy) -> ReviewSetup:
    """개발 서비스(S3·Label Studio)와 시험용 CVAT 계정 연결로 만든 `ReviewSetup`.

    CVAT 클라이언트는 넣지 않는다 (CVAT 테스트는 따로 붙인다)."""
    return ReviewSetup(
        raw=S3Store.from_env("dlp-raw"),
        labeling=S3Store.from_env("dlp-labeling"),
        labeling_reader=S3Store.labeler_from_env("dlp-labeling"),
        ontology=load_ontology(ROOT / "config" / "ontology" / "v1"),
        label_studio=LabelStudioClient.from_env(),
        cvat_config=cvat_config,
    )


def _session(
    pg: sa.Engine, setup: ReviewSetup, tmp: Path, sid: str, human: bool
) -> list[LabelRecord]:
    """블러 승인·렌더까지 끝난 세션에 시간 라벨을 넣는다.

    human이면 사람 라벨, 아니면 모델 프리라벨이다.
    """
    tmp.mkdir()
    blur = generate_blur_scenario(5)
    blur.write(tmp / "bodycam.mp4")
    generate_sync_scenario(1, recorded_at=FIXED_TIME, duration_ms=3_000).write(tmp, videos=False)
    manifest = {
        "session_id": sid, "domain": "cleaning", "worker_id": "w01", "site_id": "site01",
        "consent_version": "c1", "recorded_at": FIXED_TIME.isoformat(), "ontology_version": "1.0.0",
        "streams": [
            {"stream_id": "bodycam", "kind": "bodycam", "path": "bodycam.mp4"},
            {"stream_id": "glove_right", "kind": "glove_right", "path": "glove_right.parquet"},
        ],
    }  # fmt: skip
    (tmp / "m.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    policy = load_policy(ROOT)
    policy = policy.model_copy(
        update={
            "targets": {
                t: TargetPolicy(margin=tp.margin, detectors=("oracle",))
                for t, tp in policy.targets.items()
            }
        }
    )
    temporal = [
        x.model_copy(update={"session_id": sid, "label_id": f"{sid}-{x.label_id}"})
        for x in generate_action_scenario(4).labels
        if x.kind in LS_KINDS
    ]
    if not human:
        temporal = [
            x.model_copy(
                update={
                    "provenance": Provenance(source=Source.MODEL, model_version="vlm-1"),
                    "confidence": 0.7,
                }
            )
            for x in temporal
        ]
    with pg.begin() as conn:
        ingest_session(*load_manifest(tmp / "m.yaml"), setup.raw, conn)
        oracle: dict[str, FrameDetector] = {"oracle": OracleDetector("oracle", blur.labels)}
        detect_session(conn, sid, setup.raw, oracle, {}, policy, FIXED_TIME)
        for x in get_labels(conn, sid, kinds=["blur_track"]):  # 블러 검수는 끝났다고 둔다
            record_review(
                conn, x.label_id, VerificationState.HUMAN_APPROVED, "privacy01", FIXED_TIME
            )
        insert_review_task(  # 수집까지 끝난 블러 검수 작업 기록 (승인 조건)
            conn,
            ReviewTask(
                task_key=f"cvat:{sid}-offline", tool=ReviewTool.CVAT, external_id="0",
                session_id=sid, stream_id="bodycam", stage=ReviewStage.PRIVACY,
                assignee="privacy01",
                media_uri=f"s3://dlp-raw/sessions/{sid}/derived/bodycam.proxy.mp4",
                label_kinds=("blur_track",), created_at=FIXED_TIME,
                status=ReviewTaskStatus.COLLECTED, collected_at=FIXED_TIME,
            ),
        )  # fmt: skip
        approve_session(conn, sid)
        render_session(conn, sid, setup.raw, setup.labeling, load_policy(ROOT))
        insert_labels(conn, temporal)
    return temporal


def _post(ls: LabelStudioClient, task_id: str, results: list[dict[str, Any]]) -> None:
    """Label Studio 작업에 사람 주석(results)을 제출한다 (검수자 제출 흉내)."""
    resp = ls.http.post(f"/api/tasks/{task_id}/annotations", json={"result": results})
    resp.raise_for_status()


def test_blind_and_seeded_tasks_through_label_studio(
    pg: sa.Engine, setup: ReviewSetup, tmp_path: Path
) -> None:
    """Label Studio로 표준·블라인드·오류 삽입 배정을 끝까지 돌리고 품질 리포트를 본다.

    시나리오: 골든(사람 라벨) 세션과 작업(모델 프리라벨, 신뢰도 0.7) 세션. 블라인드·오류 삽입 비율
    1.0으로 계획(재계획은 빈 목록, 멱등) → 작업 만들기(같은 배정으로 다시 만들면 같은 작업) →
    블라인드는 정답의 절반을 그림, 오류 삽입은 넣은 오류를 모두 원래 값으로 되돌림, 표준은
    프리라벨을 그대로 제출 → 수집.
    정답 근거: 블라인드 작업에 예측 없음, 블라인드 결과는 모두 measurement=blind, 오류 삽입 결과는
    모두 seeded_error, 운영 라벨에 측정 레코드 없음, 골든 운영 라벨 불변, 발견율 100%, 편향 > 0.
    """
    gold_sid, work_sid = f"gold-{uuid.uuid4().hex[:6]}", f"work-{uuid.uuid4().hex[:6]}"
    _session(pg, setup, tmp_path / "gold", gold_sid, human=True)
    work = _session(pg, setup, tmp_path / "work", work_sid, human=False)
    with pg.begin() as conn:
        insert_golden_set(
            conn,
            GoldenSet(
                version=f"g-{gold_sid}",
                domain=Domain.CLEANING,
                session_ids=(gold_sid,),
                created_at=FIXED_TIME,
            ),
        )

    ops = load_ops_policy(ROOT)
    ops = ops.model_copy(
        update={
            "ratios": ops.ratios.model_copy(
                update={
                    "blind_task_ratio": 1.0,
                    "seeded_error_task_ratio": 1.0,
                    "double_annotation_ratio": 0.0,
                }
            )
        }
    )
    ontology = load_ontology(ROOT / "config" / "ontology" / "v1")
    with pg.begin() as conn:
        planned = plan_session(
            conn,
            work_sid,
            ["r1", "r2"],
            ops,
            ontology,
            seed=3,
            now=FIXED_TIME,
            seed_sessions=[gold_sid],
            seed_groups={"temporal"},
        )
        again = plan_session(
            conn,
            work_sid,
            ["r1", "r2"],
            ops,
            ontology,
            seed=3,
            now=FIXED_TIME,
            seed_sessions=[gold_sid],
            seed_groups={"temporal"},
        )
    assert again == []  # 멱등
    modes = {a.mode: a for a in planned}
    assert set(modes) == {ReviewMode.STANDARD, ReviewMode.BLIND, ReviewMode.SEEDED_ERROR}
    standard, blind, seeded = (
        modes[ReviewMode.STANDARD],
        modes[ReviewMode.BLIND],
        modes[ReviewMode.SEEDED_ERROR],
    )
    assert blind.assignee != standard.assignee and seeded.session_id == gold_sid
    assert seeded.injected and {e.error_type for e in seeded.injected} <= {
        "boundary_shift",
        "class_swap",
    }
    ls = setup.label_studio
    assert ls is not None

    with pg.begin() as conn:
        [t_std] = create_assignment_tasks(conn, standard, setup, ops, FIXED_TIME)
        [t_blind] = create_assignment_tasks(conn, blind, setup, ops, FIXED_TIME)
        [t_seed] = create_assignment_tasks(conn, seeded, setup, ops, FIXED_TIME)
    # 회귀: 같은 배정으로 다시 만들면 (배정 행을 잠그고) 이미 만든 작업을 돌려준다
    with pg.begin() as conn:
        before = len(list_review_tasks(conn, work_sid))
        assert create_assignment_tasks(conn, standard, setup, ops, FIXED_TIME) == [t_std]
        assert len(list_review_tasks(conn, work_sid)) == before
    assert ls.prediction_results(int(t_blind.external_id)) == []  # 블라인드: 프리라벨 없음
    assert ls.latest_results(int(t_std.external_id)) is None  # 프리라벨은 사람 주석이 아니다
    sent = ls.prediction_results(int(t_seed.external_id))
    assert all(r["id"].startswith("seed-") for r in sent)

    # 블라인드 검수자: 처음부터 그림 (여기서는 정답의 절반)
    blind_results = to_ls_results(work[::2])
    for i, r in enumerate(blind_results):
        r["id"] = f"b{i}"
    _post(ls, t_blind.external_id, blind_results)
    # 오류 삽입 검수자: 넣은 오류를 모두 되돌린다
    gold = {x.label_id: x for x in get_labels_sync(pg, gold_sid) if not x.seeded_error}
    fixed: list[dict[str, Any]] = []
    for r in sent:
        err = next((e for e in seeded.injected if e.seeded_label_id == r["id"]), None)
        if err is not None:
            original = gold[err.original_label_id]
            r = to_ls_results([original])[0] | {"id": r["id"]}
        fixed.append(r)
    _post(ls, t_seed.external_id, fixed)
    # 표준 검수자: 프리라벨을 고치지 않고 제출한다
    _post(ls, t_std.external_id, ls.prediction_results(int(t_std.external_id)))

    with pg.begin() as conn:
        collect_task(conn, t_std.task_key, setup, standard.assignee or "r1", FIXED_TIME)
        b = collect_task(conn, t_blind.task_key, setup, blind.assignee or "r2", FIXED_TIME)
        s = collect_task(conn, t_seed.task_key, setup, seeded.assignee or "r1", FIXED_TIME)
        assert all(
            get_assignment(conn, a.assignment_id).status is AssignmentStatus.DONE for a in planned
        )
        report = quality_report(conn, ops)
        work_labels = get_labels(conn, work_sid)
        gold_labels = get_labels(conn, gold_sid)
    assert b is not None and b.added == len(blind_results)
    assert all(x.measurement == "blind" for x in b.new_records)
    assert s is not None and s.corrected == len(seeded.injected)
    assert all(x.seeded_error for x in s.new_records)
    # 운영 라벨: 블라인드 결과와 오류 삽입 계보는 빠지고, 표준 검수(승인)만 반영된다
    assert {x.label_id for x in current_labels(work_labels)} >= {x.label_id for x in work}
    assert not any(x.measurement for x in current_labels(work_labels))
    assert {x.label_id for x in current_labels(gold_labels) if x.kind in LS_KINDS} == {
        x.label_id for x in gold.values() if x.kind in LS_KINDS
    }
    [rate] = report.detection
    assert (rate.injected, rate.detected) == (len(seeded.injected), len(seeded.injected))
    assert report.blind_bias[blind.assignment_id] > 0  # 표준 검수는 프리라벨을 그대로 승인했다


def get_labels_sync(pg: sa.Engine, sid: str) -> list[LabelRecord]:
    """새 연결로 세션의 모든 라벨 레코드를 읽는다 (트랜잭션 밖에서 확인용)."""
    with pg.connect() as conn:
        return get_labels(conn, sid)


def test_seeded_blur_deletion_through_cvat(
    pg: sa.Engine, setup: ReviewSetup, tmp_path: Path
) -> None:
    """블러 삭제 오류 삽입 과제 (CVAT). CI 서비스 작업에는 CVAT가 없어 닿지 않으면 건너뛴다."""
    try:
        cvat = CvatClient.from_env()
    except httpx.HTTPError as exc:
        pytest.skip(f"CVAT에 연결할 수 없습니다 (make cvat-up): {exc}")
    setup = replace(setup, cvat=cvat, label_studio=None)
    gold_sid, work_sid = f"gold-{uuid.uuid4().hex[:6]}", f"work-{uuid.uuid4().hex[:6]}"
    _session(pg, setup, tmp_path / "gold", gold_sid, human=True)
    _session(pg, setup, tmp_path / "work", work_sid, human=False)
    ops = load_ops_policy(ROOT)
    ratios = {
        "blind_task_ratio": 0.0,
        "seeded_error_task_ratio": 1.0,
        "double_annotation_ratio": 0.0,
    }
    reviewers = ops.reviewers.model_copy(update={"privacy": ("p1", "p2")})  # 원본 접근 권한자
    ops = ops.model_copy(
        update={"ratios": ops.ratios.model_copy(update=ratios), "reviewers": reviewers}
    )
    ontology = load_ontology(ROOT / "config" / "ontology" / "v1")
    # 권한 없는 검수자에게는 블러 검수를 배정하지 않는다
    with pg.begin() as conn, pytest.raises(AccessError):
        plan_session(conn, work_sid, ["r1"], ops, ontology, seed=5, now=FIXED_TIME, privacy=True)
    with pg.begin() as conn:
        planned = plan_session(
            conn, work_sid, ["p1", "p2"], ops, ontology, seed=5, now=FIXED_TIME,
            seed_sessions=[gold_sid], seed_groups={"privacy"}, privacy=True,
        )  # fmt: skip
    [seeded] = [a for a in planned if a.mode is ReviewMode.SEEDED_ERROR]
    assert seeded.label_kinds == ("blur_track",) and seeded.session_id == gold_sid
    assert seeded.injected and {e.error_type for e in seeded.injected} == {"blur_deletion"}
    # 계획 단계에서 오류를 넣은 사본이 이미 들어가 있으므로 원래 라벨만 고른다
    gold = {x.label_id: x for x in get_labels_sync(pg, gold_sid) if not x.seeded_error}

    with pg.begin() as conn:
        [task] = create_assignment_tasks(conn, seeded, setup, ops, FIXED_TIME)
    tid = int(task.external_id)
    sent = cvat.get_tracks(tid)
    assert len(sent) == len([x for x in gold.values() if x.kind == "blur_track"]) - len(
        seeded.injected
    )

    # 검수자: 빠진 블러를 다시 그린다
    project = cvat.http.get(f"/api/tasks/{tid}").json()["project_id"]
    schema = CvatSchema.from_labels(cvat.project_labels(int(project)))
    video = tmp_path / "proxy.mp4"
    setup.raw.get_file(object_key(setup.raw, task.media_uri), video)
    redrawn = to_cvat_tracks(
        [gold[e.original_label_id] for e in seeded.injected], frame_times(video), schema
    )
    for t in redrawn:
        t["attributes"] = []  # 새로 그린 트랙에는 원래 라벨 ID가 없다
    for t in [*sent, *redrawn]:
        t.pop("id", None)
        for s in t["shapes"]:
            s.pop("id", None)
    cvat.put_tracks(tid, [*sent, *redrawn])

    with pg.begin() as conn:
        outcome = collect_task(conn, task.task_key, setup, seeded.assignee or "p1", FIXED_TIME)
        report = quality_report(conn, ops)
        gold_after = get_labels(conn, gold_sid)
    assert outcome is not None and outcome.added == len(seeded.injected)
    assert all(x.seeded_error for x in outcome.new_records)
    # 새로 그린 블러는 이 배정의 사본 접두사가 붙어 발견 판정이 이 배정으로 한정된다
    assert all(
        x.label_id.startswith(seed_prefix(seeded.assignment_id)) for x in outcome.new_records
    )
    [rate] = report.detection
    assert (rate.injected, rate.detected) == (len(seeded.injected), len(seeded.injected))
    # 원래 골든 블러 라벨(운영 라벨)은 그대로다
    before = {i for i, x in gold.items() if x.kind == "blur_track" and not x.seeded_error}
    assert {x.label_id for x in current_labels(gold_after) if x.kind == "blur_track"} == before
