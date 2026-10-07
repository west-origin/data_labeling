"""종단 검증 (계획서 "종단 검증").

합성 세션 하나를 끝까지 돌리고 내보낸 결과를 픽스처 정답과 비교한다.

수집 → 동기화 → 블러 탐지·검수·승인·렌더 → 프리라벨(stub) → 관계 → 행동(VLM stub, 블러본) →
검수 시뮬레이터(승인, 흔들린 박스와 행동은 정답으로 수정) → 데이터셋 버전(lakeFS) →
재학습(stub)·골든 평가·게이트 → 내보내기(구간 JSON, COCO).

서비스(make up)가 필요하다. 실제 모델 대신 정답을 아는 stub을 쓰므로, 이 테스트가 보는 것은
단계 사이의 계약·멱등·검증 정책·시각 처리다.

CLI(`dlp …`)를 거치지 않고 각 패키지의 실행 함수를 직접 부른다. 저장소는 감사 없는 `S3Store`를
직접 쓴다 (원본 접근 감사는 `test_raw_access_db.py`가 따로 검증한다).
정답 출처: `dlp_fixtures.actions.generate_action_scenario`(손 키포인트·행동 구간·객체 위치)와
`truth_boxes`(그 객체 위치로 만든 정답 박스).
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
import sqlalchemy as sa
import yaml

from dlp_actions.clients import OracleVlm
from dlp_actions.policy import load_policy as load_actions_policy
from dlp_actions.runner import run_actions
from dlp_datasets.build import build_dataset_version
from dlp_datasets.policy import load_policy as load_dataset_policy
from dlp_datasets.snapshot import LakeFSSnapshotStore
from dlp_eval.policy import load_policy as load_eval_policy
from dlp_export.policy import load_policy as load_export_policy
from dlp_export.runner import run_export
from dlp_fixtures.actions import ENTITIES, generate_action_scenario
from dlp_fixtures.video import read_frame_times, write_video
from dlp_media.ingest import ingest_session, load_manifest
from dlp_media.storage import S3Store
from dlp_prelabel.adapters.stubs import OraclePredictor
from dlp_prelabel.policy import load_policy as load_prelabel_policy
from dlp_prelabel.runner import run_prelabel
from dlp_privacy.detectors.oracle import OracleDetector
from dlp_privacy.policy import TargetPolicy
from dlp_privacy.policy import load_policy as load_privacy_policy
from dlp_privacy.runner import approve_session, detect_session, render_session
from dlp_relations.policy import load_policy as load_relations_policy
from dlp_relations.runner import run_relations
from dlp_review.verify import verify_session
from dlp_schema.db.migrate import upgrade
from dlp_schema.db.repository import (
    get_labels,
    get_session,
    insert_golden_set,
    insert_labels,
    insert_review_task,
    insert_session,
    record_review,
    register_ontology,
)
from dlp_schema.episode import current_labels
from dlp_schema.labels import (
    BoxKeyframe,
    BoxTrackPayload,
    LabelRecord,
    Provenance,
    Source,
    Verification,
    VerificationState,
)
from dlp_schema.lineage import GoldenSet, ModelStatus
from dlp_schema.ontology import load_ontology
from dlp_schema.predictor import Clip
from dlp_schema.review import ReviewStage, ReviewTask, ReviewTaskStatus, ReviewTool
from dlp_schema.session import Domain, PrivacyState, StreamKind
from dlp_schema.testing import FIXED_TIME, make_label, make_session
from dlp_sync.policy import load_policy as load_sync_policy
from dlp_sync.runner import run_sync
from dlp_train.loop import TrainingJob, run_training_job
from dlp_train.policy import load_policy as load_training_policy
from dlp_train.tracking import MemoryTracker

pytestmark = pytest.mark.services
ROOT = Path(__file__).resolve().parents[1]
OBJECTS = {e[0] for e in ENTITIES}


@pytest.fixture
def pg() -> Iterator[sa.Engine]:
    """임시 PostgreSQL 데이터베이스 (`make up` 필요).

    `DLP_DATABASE_URL`(없으면 개발 기본값) 서버에 `dlp_test_<임의>` DB를 만들고 Alembic 최신까지
    올린 엔진을 준다. 끝나면 DB를 강제로 지운다. (같은 픽스처가 `test_raw_access_db.py`,
    `packages/ops/tests/test_ops.py`에도 있다 — 공용 conftest로 모을 리팩토링 후보.)
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
    try:
        yield engine
    finally:
        engine.dispose()
        with admin.connect() as conn:
            conn.execute(sa.text(f'DROP DATABASE "{name}" WITH (FORCE)'))
        admin.dispose()


def truth_boxes(sid: str, frame_times: list[int], prefix: str) -> list[LabelRecord]:
    """정답 객체 박스: 영상 프레임 시각마다 키프레임 (공간 라벨 = 스트림 PTS 시각, ADR 0019).

    인자:
        sid: 세션 ID.
        frame_times: 바디캠 영상의 프레임 PTS 시각(ms) 목록. 키프레임마다 하나씩 쓴다.
        prefix: 라벨 ID 접두 (`<prefix>-<entity_id>`).
    반환: 픽스처 `ENTITIES` 중 객체마다 사람 출처 `box_track` 라벨 하나 (중심 ±15px, 30x30 고정
    박스).
    """
    return [
        make_label(
            BoxTrackPayload(
                entity_id=eid,
                class_id=cls,
                keyframes=tuple(
                    BoxKeyframe(t_ms=t, x=p[0] - 15, y=p[1] - 15, w=30, h=30) for t in frame_times
                ),
            ),
            label_id=f"{prefix}-{eid}",
            session_id=sid,
            stream_id="bodycam",
            t_start_ms=frame_times[0],
            t_end_ms=frame_times[-1],
        )
        for eid, cls, _, p, _ in ENTITIES
        if eid in OBJECTS
    ]


def approve_all(conn: sa.Connection, sid: str, kinds: list[str], reviewer: str) -> int:
    """검수 시뮬레이터: `kinds`의 운영 현재 모델 라벨 중 미검수인 것을 `reviewer`가 모두 승인한다.

    반환: 승인한 라벨 수. 부작용: `record_review`로 검수 상태 갱신 (호출자 트랜잭션 안).
    """
    n = 0
    for x in current_labels(get_labels(conn, sid, kinds=kinds)):
        if (
            x.provenance.source is Source.MODEL
            and x.verification.state is VerificationState.UNREVIEWED
        ):
            record_review(conn, x.label_id, VerificationState.HUMAN_APPROVED, reviewer, FIXED_TIME)
            n += 1
    return n


def test_synthetic_session_end_to_end(pg: sa.Engine, tmp_path: Path) -> None:
    """합성 세션 하나를 수집부터 내보내기까지 돌리고 결과를 픽스처 정답과 맞춘다.

    단계별 확인 (본문 번호 주석과 같다):
    1. 수집·동기화: 바디캠만이라 기준 스트림 오프셋 0.
    2. 블러: 정답 탐지기 → 원본 권한 검수자 승인 → 수거된 블러 검수 작업(가짜) → 승인 → 렌더.
       블러본 URI에 원본 버킷 이름이 없다.
    3. 프리라벨: 손(정답), 객체(정답을 4px 흔든 것). 두 번 돌리면 둘 다 건너뛴다(멱등).
    4. 관계·행동: VLM stub이 블러본을 받아 정답 구간 분류. 오른손 행동 수 ≥ 정답 - 1.
    5. 검수 시뮬레이터: 박스는 정답으로 수정, 행동 타임라인은 지우고 정답으로 다시 그림, 나머지
       승인. `verify_session`으로 human_verified.
    6. 다른 작업자·장소의 골든 세션(사람 정답 + 기본 어댑터 예측)과 데이터셋 버전.
    7. 재학습(oracle stub): 기본 어댑터보다 나아 게이트 통과 → 배포. 학습 예제는 고친 박스뿐.
    8. 내보내기(구간 JSON, COCO): 골든 세션 없음, 미검수 없음, 세션 ID는 가명. 행동
       동사·대상·시각과 박스가 픽스처 정답과 정확히 같다.
    마지막으로 계보(세션 → 데이터셋 버전 → 학습 실행 → 내보내기)를 확인한다.
    """
    sid = f"e2e-{uuid.uuid4().hex[:8]}"
    actions = generate_action_scenario(2, session_id=sid, n_units=4)
    # 바디캠: 행동 픽스처의 프레임 시각으로 쓴 영상
    # (손 키포인트·객체 박스·블러가 같은 프레임 시각에 있다)
    write_video(
        tmp_path / "bodycam.mp4",
        ((t, np.full((240, 320, 3), 90, np.uint8)) for t in actions.frame_times),
        width=320,
        height=240,
    )
    frame_times = read_frame_times(tmp_path / "bodycam.mp4")
    assert frame_times == actions.frame_times
    face = make_label(
        {"kind": "blur_track", "target": "face",
         "keyframes": [{"t_ms": t, "x": 10, "y": 10, "w": 30, "h": 30} for t in frame_times[:30]]},
        label_id="face", session_id=sid, stream_id="bodycam",
        t_start_ms=frame_times[0], t_end_ms=frame_times[29],
    )  # fmt: skip
    manifest = {
        "session_id": sid, "domain": "cleaning", "worker_id": "w-e2e", "site_id": "site-e2e",
        "consent_version": "c1", "recorded_at": FIXED_TIME.isoformat(), "ontology_version": "1.0.0",
        "streams": [{"stream_id": "bodycam", "kind": "bodycam", "path": "bodycam.mp4"}],
    }  # fmt: skip
    (tmp_path / "m.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    raw, labeling = S3Store.from_env("dlp-raw"), S3Store.from_env("dlp-labeling")
    datasets, artifacts = S3Store.from_env("dlp-datasets"), S3Store.from_env("dlp-mlflow")
    ontology = load_ontology(ROOT / "config/ontology/v1")
    now = FIXED_TIME

    # 1. 수집·동기화 (바디캠뿐이라 기준 스트림만)
    with pg.begin() as conn:
        register_ontology(conn, ontology)
        ingest_session(*load_manifest(tmp_path / "m.yaml"), raw, conn)
        session, _ = run_sync(conn, sid, raw, load_sync_policy(ROOT / "config/policies/sync.yaml"))
    assert session.reference_stream.offset_ms == 0

    # 2. 블러: 탐지(정답 탐지기) → 원본 권한 검수자 승인 → 세션 승인 → 블러본 렌더
    privacy = load_privacy_policy(ROOT)
    oracle_policy = privacy.model_copy(
        update={"targets": {t: TargetPolicy(margin=tp.margin, detectors=("oracle",))
                            for t, tp in privacy.targets.items()}}
    )  # fmt: skip
    with pg.begin() as conn:
        detect_session(
            conn, sid, raw, {"oracle": OracleDetector("oracle", [face])}, {}, oracle_policy, now
        )
        approve_all(conn, sid, ["blur_track"], "privacy-reviewer")
        # 영상 스트림마다 수거된 블러 검수 작업이 있어야 승인된다 (CVAT 없이 수거된 것으로 둔다)
        for st in get_session(conn, sid).streams:
            if st.kind not in (StreamKind.BODYCAM, StreamKind.THIRD_PERSON):
                continue
            insert_review_task(
                conn,
                ReviewTask(
                    task_key=f"cvat:{sid}-{st.stream_id}-e2e", tool=ReviewTool.CVAT,
                    external_id="0", session_id=sid, stream_id=st.stream_id,
                    stage=ReviewStage.PRIVACY, assignee="privacy-reviewer",
                    media_uri=f"s3://dlp-raw/sessions/{sid}/derived/{st.stream_id}.proxy.mp4",
                    label_kinds=("blur_track",), created_at=now,
                    status=ReviewTaskStatus.COLLECTED, collected_at=now,
                ),
            )  # fmt: skip
        approve_session(conn, sid)
        blurred = render_session(conn, sid, raw, labeling, oracle_policy)
    assert all("dlp-raw" not in u for u in blurred.values())

    # 3. 프리라벨: 손(정답), 객체(정답을 흔든 것 — 검수자가 고친다), 접촉
    boxes = truth_boxes(sid, frame_times, "gt")
    hands = OraclePredictor("hands", actions.labels, ("keypoint_track",), now=now)
    objects = OraclePredictor("objects", boxes, ("box_track",), jitter_px=4.0, now=now)
    prelabel_policy = load_prelabel_policy(ROOT)
    with pg.begin() as conn:
        pre = run_prelabel(conn, sid, raw, [hands, objects], prelabel_policy, ontology, now)
        again = run_prelabel(conn, sid, raw, [hands, objects], prelabel_policy, ontology, now)
    assert (
        pre.produced == {"bodycam/hands": 1, "bodycam/objects": len(OBJECTS)} and pre.contacts > 0
    )
    assert sorted(again.skipped) == ["bodycam/hands", "bodycam/objects"]  # 멱등

    # 4. 관계, 행동(VLM stub은 블러본을 받는다)
    truth_segments = [x for x in actions.labels if x.kind in ("action", "gap")]
    with pg.begin() as conn:
        run_relations(conn, sid, ontology, load_relations_policy(ROOT), now)
        acts = run_actions(
            conn, sid, OracleVlm(truth_segments), ontology, load_actions_policy(ROOT), now, labeling
        )
    assert acts.hands["right"]["actions"] >= len(actions.actions) - 1

    # 5. 검수 시뮬레이터: 흔들린 박스는 정답으로 고치고, 나머지 모델 라벨은 승인한다
    with pg.begin() as conn:
        model_boxes = [
            x for x in current_labels(get_labels(conn, sid, kinds=["box_track"]))
            if x.provenance.source is Source.MODEL
        ]  # fmt: skip
        by_entity = {b.payload.entity_id: b for b in boxes}  # type: ignore[union-attr]
        insert_labels(
            conn,
            [
                by_entity[m.payload.entity_id].model_copy(  # type: ignore[union-attr]
                    update={
                        "label_id": f"{m.label_id}:fix",
                        "parent_label_id": m.label_id,
                        "verification": Verification(
                            state=VerificationState.HUMAN_CORRECTED,
                            reviewer_id="labeler-1",
                            reviewed_at=now,
                        ),
                    }
                )
                for m in model_boxes
            ],
        )
        # 행동 타임라인: 영상 접촉 휴리스틱의 경계는 정답과 다를 수 있다.
        # 검수자가 모델 행동·공백·설명을 지우고 정답 타임라인으로 다시 그린다 (사람 출처 레코드)
        reviewed = Verification(
            state=VerificationState.HUMAN_CORRECTED, reviewer_id="labeler-1", reviewed_at=now
        )
        timeline = [
            x for x in current_labels(get_labels(conn, sid, kinds=["action", "gap", "description"]))
            if x.provenance.source is Source.MODEL
        ]  # fmt: skip
        insert_labels(
            conn,
            [
                x.model_copy(
                    update={
                        "label_id": f"{x.label_id}:del",
                        "parent_label_id": x.label_id,
                        "retracted": True,
                        "provenance": Provenance(source=Source.HUMAN),
                        "confidence": None,
                        "verification": reviewed,
                    }
                )
                for x in timeline
            ]
            + [
                x.model_copy(update={"label_id": f"{x.label_id}-h", "verification": reviewed})
                for x in truth_segments
            ],
        )
        approve_all(
            conn, sid, ["keypoint_track", "hand_state", "relation", "coverage"], "labeler-1"
        )
        # 검수 완료 판정: 블러 승인·수거·모든 모델 라벨 검수가 끝나야 human_verified로 간다
        verified = verify_session(conn, sid, now, "labeler-1")
        assert verified.verified, verified.reasons

    # 6. 골든 세션(사람이 처음부터 라벨링, 다른 작업자·장소)과 데이터셋 버전
    golden_sid = f"{sid}-golden"
    with pg.begin() as conn:
        insert_session(
            conn,
            make_session(golden_sid, worker_id="w-gold", site_id="site-gold").model_copy(
                update={"privacy_state": PrivacyState.APPROVED}
            ),
        )
        golden_truth = truth_boxes(golden_sid, frame_times, "golden")
        insert_labels(conn, golden_truth)
        # 골든 세션에도 기본 어댑터(흔들린 객체 탐지) 예측이 있다 (프리라벨을 돌린 것처럼).
        # 게이트의 비교 기준이다
        base_on_golden = OraclePredictor(
            "objects", golden_truth, ("box_track",), jitter_px=4.0, now=now
        )
        assert base_on_golden.version == objects.version
        insert_labels(
            conn, base_on_golden.run(Clip(golden_sid, "bodycam", tmp_path / "bodycam.mp4"))
        )
        insert_golden_set(
            conn,
            GoldenSet(version=f"g-{sid}", domain=Domain.CLEANING, session_ids=(golden_sid,),
                      created_at=now),
        )  # fmt: skip
        dpol = load_dataset_policy(ROOT)
        snapshots = LakeFSSnapshotStore.from_env(
            repository=dpol.lakefs.repository, branch=dpol.lakefs.branch,
            storage_namespace=dpol.lakefs.storage_namespace,
        )  # fmt: skip
        build_dataset_version(
            conn, snapshots, dpol, version_id=f"dv-{sid}", ontology_version="1.0.0",
            golden_set_version=f"g-{sid}", now=now,
        )  # fmt: skip

    # 7. 재학습(stub) → 골든 평가 → 게이트 → 배포 (기본 어댑터 예측과 비교)
    training = load_training_policy(ROOT)
    small = training.tasks["objects"].model_copy(update={"min_examples": 1, "min_new_examples": 0})
    training = training.model_copy(update={"tasks": {**training.tasks, "objects": small}})
    with pg.begin() as conn:
        trained = run_training_job(
            conn,
            TrainingJob(
                "objects",
                f"dv-{sid}",
                params={"confidence": 1.0},  # stub이 정확하므로 신뢰도도 1 (보정 오차 0)
                baseline_versions=(objects.version,),
            ),
            snapshots=snapshots, artifacts=artifacts, tracker=MemoryTracker(),
            clips=lambda s, st, work: tmp_path / "bodycam.mp4", policy=training,
            eval_policy=load_eval_policy(ROOT), now=now + timedelta(minutes=1), stub_truth=True,
        )  # fmt: skip
    assert trained.status == "deployed", trained.reason
    # 고친 박스만 학습 예제 (자동 원본과의 차이).
    # 세션 하나라 학습·검증 어느 분할에 들어갈지는 분할기가 정한다
    assert sum(trained.examples.values()) == len(OBJECTS)
    assert {k.split("/")[1] for k in trained.examples} == {"corrected"}

    # 8. 내보내기 → 픽스처 정답과 비교
    export_policy = load_export_policy(ROOT)
    out: dict[str, Path] = {}
    out_ids: dict[str, str] = {}
    # 트랜잭션은 run_export가 연다 (이력을 올리기 전에 따로 커밋)
    for fmt in ("intervals", "coco"):
        r = run_export(
            pg, root=ROOT, version_id=f"dv-{sid}", fmt=fmt, target="e2e",  # type: ignore[arg-type]
            snapshots=snapshots, labeling=labeling, datasets=datasets, raw_bucket="dlp-raw",
            policy=export_policy, ontology=ontology, include_unreviewed=False, splits=None,
            now=now + timedelta(minutes=2),
        )  # fmt: skip
        assert r.record.session_ids == (sid,)  # 골든 세션은 기본 내보내기에 없다
        assert not [k for k in r.label_counts if k.endswith("/unreviewed")]
        dest = tmp_path / "export" / fmt
        # 결과 파일의 세션 ID는 내보내기마다 다른 가명이다
        pseudo = r.session_pseudonyms[sid]
        for key in ("manifest.json", f"intervals/{pseudo}.json", "coco/annotations.json"):
            if (fmt == "coco") == key.startswith("coco") or key == "manifest.json":
                path = dest / key
                path.parent.mkdir(parents=True, exist_ok=True)
                datasets.get_file(f"exports/{r.record.export_id}/{key}", path)
        out[fmt] = dest
        out_ids[fmt] = pseudo

    intervals = json.loads(
        (out["intervals"] / "intervals" / f"{out_ids['intervals']}.json").read_text()
    )
    exported_actions = [
        x["payload"] for x in intervals["labels"] if x["payload"]["kind"] == "action"
    ]
    assert [(a["verb"], a["target_id"]) for a in exported_actions] == [
        (a.verb, a.target_id) for a in actions.actions
    ]
    for got, want in zip(exported_actions, actions.actions, strict=True):
        assert (got["t_approach_ms"], got["t_end_ms"]) == (want.t_approach_ms, want.t_end_ms)
        assert (got["t_contact_start_ms"], got["t_contact_end_ms"]) == (
            want.t_contact_start_ms,
            want.t_contact_end_ms,
        )
    assert {x["verification"] for x in intervals["labels"]} <= {"human_approved", "human_corrected"}

    coco = json.loads((out["coco"] / "coco" / "annotations.json").read_text())
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    images = {i["id"]: i for i in coco["images"]}
    got_boxes = sorted(
        (a["track_id"], images[a["image_id"]]["t_ms"], tuple(a["bbox"]))
        for a in coco["annotations"]
        if cats[a["category_id"]] != "hand"
    )
    want_boxes: list[tuple[str, int, tuple[float, float, float, float]]] = []
    for b in boxes:
        assert isinstance(b.payload, BoxTrackPayload)
        want_boxes += [
            (b.payload.entity_id, k.t_ms, (k.x, k.y, k.w, k.h)) for k in b.payload.keyframes
        ]
    want_boxes.sort()
    assert got_boxes == want_boxes  # 검수자가 고친 정답 박스 그대로, 블러본 프레임 시각에
    assert {a["verification"] for a in coco["annotations"]} <= {"human_corrected", "human_approved"}

    # 계보: 세션 → 데이터셋 버전 → 학습 실행 → 내보내기
    from dlp_datasets.lineage import session_lineage
    from dlp_schema.db.repository import list_model_versions

    with pg.connect() as conn:
        lineage = session_lineage(conn, sid)
        assert lineage.dataset_versions == [f"dv-{sid}"]
        assert [r.model_version for r in lineage.training_runs] == [trained.model_version]
        assert {e.format for e in lineage.exports} == {"intervals", "coco"}
        assert [m.status for m in list_model_versions(conn, "objects")] == [ModelStatus.DEPLOYED]
