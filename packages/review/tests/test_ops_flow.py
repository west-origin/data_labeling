"""검수 운영 흐름 (Label Studio): 계획 → 블라인드·오류 삽입 작업 → 검수 → 수집 → 품질 리포트."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

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
from dlp_review.clients import LabelStudioClient
from dlp_review.collect import collect_task
from dlp_review.labelstudio import LS_KINDS, to_ls_results
from dlp_review.ops.policy import load_policy as load_ops_policy
from dlp_review.ops.runner import create_assignment_tasks, plan_session, quality_report
from dlp_review.tasks import ReviewSetup
from dlp_schema.db.migrate import upgrade
from dlp_schema.db.repository import (
    get_assignment,
    get_labels,
    insert_golden_set,
    insert_labels,
    record_review,
    register_ontology,
)
from dlp_schema.episode import current_labels
from dlp_schema.labels import LabelRecord, Provenance, Source, VerificationState
from dlp_schema.lineage import GoldenSet
from dlp_schema.ontology import load_ontology
from dlp_schema.review import AssignmentStatus, ReviewMode
from dlp_schema.session import Domain
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
    return ReviewSetup(
        raw=S3Store.from_env("dlp-raw"),
        labeling=S3Store.from_env("dlp-labeling"),
        labeling_reader=S3Store.labeler_from_env("dlp-labeling"),
        ontology=load_ontology(ROOT / "config" / "ontology" / "v1"),
        label_studio=LabelStudioClient.from_env(),
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
        approve_session(conn, sid)
        render_session(conn, sid, setup.raw, setup.labeling, load_policy(ROOT))
        insert_labels(conn, temporal)
    return temporal


def _post(ls: LabelStudioClient, task_id: str, results: list[dict[str, Any]]) -> None:
    resp = ls.http.post(f"/api/tasks/{task_id}/annotations", json={"result": results})
    resp.raise_for_status()


def test_blind_and_seeded_tasks_through_label_studio(
    pg: sa.Engine, setup: ReviewSetup, tmp_path: Path
) -> None:
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
        [t_std] = create_assignment_tasks(conn, standard, setup, FIXED_TIME)
        [t_blind] = create_assignment_tasks(conn, blind, setup, FIXED_TIME)
        [t_seed] = create_assignment_tasks(conn, seeded, setup, FIXED_TIME)
    assert ls.latest_results(int(t_blind.external_id)) == []  # 블라인드: 프리라벨 없음
    sent = ls.latest_results(int(t_seed.external_id))
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
    with pg.connect() as conn:
        return get_labels(conn, sid)
