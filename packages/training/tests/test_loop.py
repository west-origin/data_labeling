"""stub 모델로 재학습 루프 전체: 누적 → 학습 → 골든셋 평가 → 게이트 실패 시 미배포 / 통과 시 배포.

DB는 PostgreSQL(make up)이 필요하다. 실험 기록은 MemoryTracker, MLflow 서버 연동은
test_mlflow.py가 따로 확인한다.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa

from dlp_datasets.build import build_dataset_version
from dlp_datasets.lineage import session_lineage, withdraw_session
from dlp_datasets.policy import load_policy as load_dataset_policy
from dlp_datasets.snapshot import LocalSnapshotStore
from dlp_eval.policy import load_policy as load_eval_policy
from dlp_media.storage import LocalStore
from dlp_schema.db.migrate import upgrade
from dlp_schema.db.repository import (
    get_labels,
    insert_golden_set,
    insert_labels,
    insert_session,
    list_model_versions,
    register_ontology,
)
from dlp_schema.labels import LabelRecord, Provenance, Source, Verification, VerificationState
from dlp_schema.lineage import GoldenSet, ModelStatus
from dlp_schema.ontology import load_ontology
from dlp_schema.session import Domain, PrivacyState, Session, Stream, StreamKind, SyncMethod
from dlp_schema.testing import FIXED_TIME, make_label, make_session
from dlp_train.deployed import deployed_predictors
from dlp_train.loop import LoopResult, TrainingError, TrainingJob, deploy, run_training_job
from dlp_train.policy import TrainingPolicy, load_policy
from dlp_train.tracking import MemoryTracker
from dlp_train.trainers import LoadContext

ROOT = Path(__file__).resolve().parents[3]
pytestmark = pytest.mark.services
BASE = ("base-v1",)
CLASSES = ("cup", "bucket", "mop")


@pytest.fixture
def engine() -> Iterator[sa.Engine]:
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
    e = sa.create_engine(test_url)
    try:
        yield e
    finally:
        e.dispose()
        with admin.connect() as conn:
            conn.execute(sa.text(f'DROP DATABASE "{name}" WITH (FORCE)'))
        admin.dispose()


def box(entity: str, cls: str, n: int = 5, dx: float = 0.0) -> dict[str, Any]:
    i = CLASSES.index(cls)
    return {
        "kind": "box_track",
        "entity_id": entity,
        "class_id": cls,
        "keyframes": [
            {"t_ms": t * 100, "x": 40 + 120 * i + 3 * t + dx, "y": 60, "w": 60, "h": 60}
            for t in range(n)
        ],
    }


def blur(entity: str, n: int = 5) -> dict[str, Any]:
    return {
        "kind": "blur_track",
        "target": "face",
        "keyframes": [
            {"t_ms": t * 100, "x": 300 + 2 * t, "y": 20, "w": 40, "h": 40} for t in range(n)
        ],
    }


def label(sid: str, lid: str, payload: dict[str, Any], **kw: Any) -> LabelRecord:
    return make_label(
        payload, label_id=f"{sid}-{lid}", session_id=sid, stream_id="bodycam", t_end_ms=400, **kw
    )


def reviewed(state: VerificationState) -> Verification:
    return Verification(state=state, reviewer_id="r1", reviewed_at=FIXED_TIME)


def base_model(sid: str, lid: str, payload: dict[str, Any], **kw: Any) -> LabelRecord:
    return label(
        sid,
        lid,
        payload,
        provenance=Provenance(source=Source.MODEL, model_version="base-v1"),
        confidence=0.7,
        **kw,
    )


def train_labels(sid: str) -> list[LabelRecord]:
    """학습 세션: 승인·수정·추가·삭제가 섞인 검수 이력."""
    return [
        base_model(sid, "m-cup", box("cup_1", "cup"),
                   verification=reviewed(VerificationState.HUMAN_APPROVED)),
        base_model(sid, "m-bucket", box("bucket_1", "bucket", dx=25)),
        label(sid, "h-bucket", box("bucket_1", "bucket"), parent_label_id=f"{sid}-m-bucket",
              verification=reviewed(VerificationState.HUMAN_CORRECTED)),
        label(sid, "h-mop", box("mop_1", "mop"),
              verification=reviewed(VerificationState.HUMAN_CORRECTED)),
        label(sid, "h-face", blur("face_1"),
              verification=reviewed(VerificationState.HUMAN_CORRECTED)),
    ]  # fmt: skip


def golden_labels(sid: str) -> list[LabelRecord]:
    """골든 세션: 사람 정답과 기존(base) 모델 예측 (많이 어긋남)."""
    out = [label(sid, f"t-{c}", box(f"{c}_1", c)) for c in CLASSES]
    out.append(label(sid, "t-face", blur("face_1")))
    out += [base_model(sid, f"b-{c}", box(f"{c}_1", c, dx=45)) for c in CLASSES]
    return out


def populate(conn: sa.Connection) -> None:
    register_ontology(conn, load_ontology(ROOT / "config/ontology/v1"))
    sessions: list[Session] = []
    for i in range(8):
        sessions.append(make_session(f"tr{i}", worker_id=f"w{i}", site_id=f"site{i}"))
    for i in range(2):
        sessions.append(make_session(f"gd{i}", worker_id=f"gw{i}", site_id=f"gsite{i}"))
    for s in sessions:
        insert_session(conn, s.model_copy(update={"privacy_state": PrivacyState.APPROVED}))
        sid = s.session_id
        insert_labels(conn, golden_labels(sid) if sid.startswith("gd") else train_labels(sid))
    insert_golden_set(
        conn,
        GoldenSet(
            version="golden-v1",
            domain=Domain.CLEANING,
            session_ids=("gd0", "gd1"),
            created_at=FIXED_TIME,
        ),
    )


def small(policy: TrainingPolicy) -> TrainingPolicy:
    """테스트 분량에 맞춘 누적 기준 (예제 10개 이상, 직전보다 5개 이상 늘어야 재학습)."""
    tasks = {
        t: s.model_copy(update={"min_examples": 10, "min_new_examples": 5})
        for t, s in policy.tasks.items()
    }
    return policy.model_copy(update={"tasks": tasks})


def test_training_loop_gate_blocks_or_deploys(engine: sa.Engine, tmp_path: Path) -> None:
    policy = small(load_policy(ROOT))
    eval_policy = load_eval_policy(ROOT)
    snapshots = LocalSnapshotStore(tmp_path / "snapshots")
    artifacts = LocalStore(tmp_path / "store", "dlp-mlflow")
    tracker = MemoryTracker()
    video = tmp_path / "video.mp4"
    video.write_bytes(b"stub")

    def run(job: TrainingJob, now_s: int = 0) -> LoopResult:
        with engine.begin() as conn:
            return run_training_job(
                conn,
                job,
                snapshots=snapshots,
                artifacts=artifacts,
                tracker=tracker,
                clips=lambda session, stream, work: video,
                policy=policy,
                eval_policy=eval_policy,
                now=FIXED_TIME.replace(second=now_s),
                stub_truth=True,
            )

    with engine.begin() as conn:
        populate(conn)
        build_dataset_version(
            conn,
            snapshots,
            load_dataset_policy(ROOT),
            version_id="dv1",
            ontology_version="1.0.0",
            golden_set_version="golden-v1",
            now=FIXED_TIME,
        )
        golden_before = {sid: len(get_labels(conn, sid)) for sid in ("gd0", "gd1")}

    # 누적 부족: 기준을 올리면 학습하지 않는다
    strict = policy.tasks["objects"].model_copy(update={"min_examples": 1000})
    policy_strict = policy.model_copy(update={"tasks": {**policy.tasks, "objects": strict}})
    with engine.begin() as conn:
        r0 = run_training_job(
            conn, TrainingJob("objects", "dv1"), snapshots=snapshots, artifacts=artifacts,
            tracker=tracker, clips=lambda session, stream, work: video, policy=policy_strict,
            eval_policy=eval_policy, now=FIXED_TIME, stub_truth=True,
        )  # fmt: skip
    assert r0.status == "skipped" and "min_examples" in r0.reason
    assert not tracker.runs

    # 1) 흔들림이 큰 후보: 첫 배포 기준(HOTA) 미달 → 미배포
    # 기본 어댑터(objects·tools)를 대신하는 과제라 비교할 기본 예측 버전 없이는 평가하지 않는다
    with pytest.raises(TrainingError, match="baseline-version"):
        run(TrainingJob("objects", "dv1", params={"jitter_px": 0.0}), 1)
    # 잘못 적은 기존 버전: 골든셋에 예측이 없으면 기존 지표가 0이 되어 아무 후보나 통과하므로 멈춘다
    with pytest.raises(TrainingError, match="base-v9"):
        run(TrainingJob("objects", "dv1", baseline_versions=(*BASE, "base-v9"), force=True), 1)
    assert not tracker.runs
    r1 = run(TrainingJob("objects", "dv1", params={"jitter_px": 200.0}, baseline_versions=BASE), 1)
    assert r1.status == "rejected", r1.reason
    # 학습 세션 8개마다 (승인 cup, 수정 bucket, 추가 mop). 골든 세션은 들어가지 않는다
    assert sum(r1.examples.values()) == 24
    assert {k.split("/")[1] for k in r1.examples} == {"accepted", "corrected", "added"}
    with engine.connect() as conn:
        assert not list_model_versions(conn, "objects", ModelStatus.DEPLOYED)

    # 2) 새 예제가 없으면 재학습하지 않는다 (force로만)
    r_skip = run(TrainingJob("objects", "dv1", baseline_versions=BASE), 2)
    assert r_skip.status == "skipped" and "min_new_examples" in r_skip.reason

    # 3) 정확한 후보: 배포 모델이 없으니 DB에 있는 기존(base) 모델 예측과 비교 → 통과 → 배포
    r2 = run(
        TrainingJob(
            "objects", "dv1", params={"jitter_px": 0.0}, baseline_versions=BASE, force=True
        ),
        3,
    )
    assert r2.status == "deployed", r2.reason
    assert r2.baseline is not None and r2.baseline.model_versions["objects"] == "base-v1"
    assert r2.candidate is not None
    assert r2.candidate.overall["objects"].metrics["hota"] > 0.9

    # 4) 배포 모델보다 나쁜 후보: 기존(배포 모델) 대비 하락 → 미배포, 기존 배포 유지
    r3 = run(TrainingJob("objects", "dv1", params={"jitter_px": 8.0}, force=True), 4)
    assert r3.status == "rejected"
    assert r3.baseline is not None and r3.baseline.model_versions["objects"] == r2.model_version
    with engine.connect() as conn:
        deployed = list_model_versions(conn, "objects", ModelStatus.DEPLOYED)
        assert [m.model_version for m in deployed] == [r2.model_version]
        statuses = {m.model_version: m.status for m in list_model_versions(conn, "objects")}
        assert statuses[r1.model_version or ""] is ModelStatus.REJECTED
        assert statuses[r3.model_version or ""] is ModelStatus.REJECTED
        # 골든셋 세션에는 아무것도 쓰지 않았다
        assert {sid: len(get_labels(conn, sid)) for sid in golden_before} == golden_before
        # 계보: 학습 세션 → 데이터셋 버전 → 학습 실행
        runs = session_lineage(conn, "tr0").training_runs
        assert {r.model_version for r in runs} == {
            r1.model_version,
            r2.model_version,
            r3.model_version,
        }

    # MLflow 기록 (메모리): 실행마다 파라미터·학습 지표·골든 지표·산출물, 배포 모델만 레지스트리에
    rec = tracker.runs[r2.mlflow_run_id or ""]
    assert rec.status == "FINISHED" and rec.params["jitter_px"] == "0.0"
    assert rec.metrics["gate/passed"] == 1.0 and "golden/hota" in rec.metrics
    assert {"model/oracle-stub.json", "golden/golden.json", "golden/golden.md"} <= set(
        rec.artifacts
    )
    # 레지스트리 등록은 DB 커밋 뒤에 CLI가 한다 (배포 결과가 알려 준다)
    assert r2.registration is not None and r1.registration is None and r3.registration is None
    tracker.register(*r2.registration)
    assert tracker.registry == {"dlp-objects": [r2.mlflow_run_id]}
    assert tracker.aliases[("dlp-objects", "deployed")] == "1"

    # 5) privacy는 deploy: approve → 통과해도 사람 승인 전에는 배포하지 않는다
    r4 = run(TrainingJob("privacy", "dv1", force=True), 5)
    assert r4.status == "passed", r4.reason
    with engine.begin() as conn:
        assert not list_model_versions(conn, "privacy", ModelStatus.DEPLOYED)
        deploy(conn, r4.model_version or "", now=FIXED_TIME, artifacts=artifacts)
        assert [
            m.model_version for m in list_model_versions(conn, "privacy", ModelStatus.DEPLOYED)
        ] == [r4.model_version]
    # 승인 대기 모델은, 그 게이트 판정 뒤에 다른 모델이 배포됐으면 승인할 수 없다
    r6 = run(TrainingJob("privacy", "dv1", force=True), 7)
    r7 = run(TrainingJob("privacy", "dv1", force=True), 8)
    assert (r6.status, r7.status) == ("passed", "passed")
    with engine.begin() as conn:
        deploy(conn, r6.model_version or "", now=FIXED_TIME.replace(second=9), artifacts=artifacts)
    with engine.begin() as conn, pytest.raises(TrainingError, match="다시 학습"):
        deploy(conn, r7.model_version or "", now=FIXED_TIME.replace(second=10), artifacts=artifacts)
    r5 = run(TrainingJob("hands", "dv1", force=True), 6)
    assert r5.status == "skipped" and "정답이 없어" in r5.reason

    # 배포 모델을 프리라벨 단계에 붙인다: objects·tools 기본 어댑터를 빼고 재학습 모델을 넣는다
    class Base:
        def __init__(self, name: str) -> None:
            self.name, self.version = name, f"{name}-base"

        def run(self, clip: object) -> list[LabelRecord]:
            return []

    with engine.connect() as conn:
        truth = {x.label_id: x for x in get_labels(conn, "gd0")}
        ctx = LoadContext(
            now=FIXED_TIME,
            ontology_version="1.0.0",
            truth=lambda sid, stream: [x for x in truth.values() if x.session_id == sid],
        )
        d = deployed_predictors(conn, artifacts, policy, "prelabel", ctx, tmp_path / "work")
        assert [p.name for p in d.apply([Base("hands"), Base("objects"), Base("tools")])] == [
            "hands",
            "trained-objects",
        ]
        # 운영(정답 없음)에서 oracle-stub은 쓸 수 없으니 기본 어댑터를 그대로 쓴다
        prod = deployed_predictors(
            conn,
            artifacts,
            policy,
            "prelabel",
            LoadContext(now=FIXED_TIME, ontology_version="1.0.0"),
            tmp_path / "work2",
        )
        assert not prod.predictors and "기본 어댑터" in prod.notes[0]
        # 산출물이 바뀌면(해시 불일치) 쓰지 않는다
        mv = list_model_versions(conn, "objects", ModelStatus.DEPLOYED)[0]
        artifacts._path(mv.artifact_uri.removeprefix(artifacts.uri(""))).write_text("{}")  # pyright: ignore[reportPrivateUsage]
        tampered = deployed_predictors(conn, artifacts, policy, "prelabel", ctx, tmp_path / "w3")
        assert not tampered.predictors and "해시" in tampered.notes[0]


# ---------------------------------------------------------------- 타임라인 과제·동의 철회·승인 배포

THIRD: dict[str, Any] = {
    "stream_id": "third", "kind": StreamKind.THIRD_PERSON,
    "uri": "s3://dlp-raw/x/third.mp4", "sync_method": SyncMethod.QR_SLATE,
}  # fmt: skip


def timeline_labels(sid: str, *, shift: int = 0, model: str | None = None) -> list[LabelRecord]:
    """접촉 5개와 행동 5개 (마스터 타임라인, stream_id 없음). model이면 그 버전의 모델 예측."""
    kw: dict[str, Any] = (
        {"provenance": Provenance(source=Source.MODEL, model_version=model), "confidence": 0.6}
        if model
        else {}
    )
    tag = "m" if model else "h"
    out: list[LabelRecord] = []
    for i in range(5):
        s, e = 1000 * i + shift, 1000 * i + 500 + shift
        contact = {"kind": "hand_state", "hand": "right", "contact_target_kind": "object",
                   "target_id": "cup_01", "grasp_type": "power", "role": "active"}  # fmt: skip
        action = {"kind": "action", "action_id": f"a{i}", "hand": "right", "verb": "grasp",
                  "t_approach_ms": s, "t_end_ms": e}  # fmt: skip
        out.append(make_label(contact, label_id=f"{sid}-{tag}-c{i}", session_id=sid,
                              t_start_ms=s, t_end_ms=e, **kw))  # fmt: skip
        out.append(make_label(action, label_id=f"{sid}-{tag}-a{i}", session_id=sid,
                              t_start_ms=s, t_end_ms=e, **kw))  # fmt: skip
    return out


def test_timeline_tasks_withdrawal_and_approval_baseline(engine: sa.Engine, tmp_path: Path) -> None:
    policy = small(load_policy(ROOT))
    snapshots = LocalSnapshotStore(tmp_path / "snapshots")
    artifacts = LocalStore(tmp_path / "store", "dlp-mlflow")
    video = tmp_path / "video.mp4"
    video.write_bytes(b"stub")

    with engine.begin() as conn:
        register_ontology(conn, load_ontology(ROOT / "config/ontology/v1"))
        for i in range(8):
            s = make_session(f"tr{i}", worker_id=f"w{i}", site_id=f"site{i}")
            insert_session(conn, s.model_copy(update={"privacy_state": PrivacyState.APPROVED}))
            insert_labels(conn, timeline_labels(f"tr{i}"))
        for i in range(2):
            # 골든 세션에는 3인칭 영상도 있다 (영상 스트림 두 개)
            s = make_session(f"gd{i}", worker_id=f"gw{i}", site_id=f"gsite{i}")
            s = s.model_copy(
                update={"streams": (*s.streams, Stream.model_validate(THIRD)),
                        "privacy_state": PrivacyState.APPROVED}
            )  # fmt: skip
            insert_session(conn, s)
            insert_labels(conn, timeline_labels(f"gd{i}"))
            # 휴리스틱 접촉 예측: 버전이 세션마다 다르고 300 ms 늦다 (허용 150 ms 밖)
            heuristic = timeline_labels(f"gd{i}", shift=300, model=f"contact-heuristic-1+pX+igd{i}")
            insert_labels(conn, [x for x in heuristic if x.kind == "hand_state"])
        insert_golden_set(
            conn,
            GoldenSet(version="golden-v1", domain=Domain.CLEANING, session_ids=("gd0", "gd1"),
                      created_at=FIXED_TIME),
        )  # fmt: skip
        version = build_dataset_version(
            conn, snapshots, load_dataset_policy(ROOT), version_id="dv1",
            ontology_version="1.0.0", golden_set_version="golden-v1", now=FIXED_TIME,
        ).version  # fmt: skip
        trained = sorted(sid for sid, sp in version.splits.items() if sp in policy.splits)
        # 버전을 만든 뒤 동의 철회: 버전 분할에는 남아 있지만 학습 예제에서 빠져야 한다
        withdrawn = trained[0]
        withdraw_session(conn, withdrawn, "consent", FIXED_TIME)

    def run(job: TrainingJob, now_s: int) -> LoopResult:
        with engine.begin() as conn:
            return run_training_job(
                conn, job, snapshots=snapshots, artifacts=artifacts, tracker=MemoryTracker(),
                clips=lambda session, stream, work: video, policy=policy,
                eval_policy=load_eval_policy(ROOT), now=FIXED_TIME.replace(second=now_s),
                stub_truth=True,
            )  # fmt: skip

    heuristic_base = ("contact-heuristic-1*",)
    c1 = run(TrainingJob("contact", "dv1", baseline_versions=heuristic_base, force=True), 1)
    # 철회한 세션의 접촉 5개는 빠졌다
    assert sum(c1.examples.values()) == 5 * (len(trained) - 1)
    # 접촉은 타임라인 라벨이라 세션마다 한 번만 예측한다 (3인칭 스트림에서 겹치면 오탐이 된다)
    assert c1.status == "passed", c1.reason
    assert c1.candidate is not None
    m = c1.candidate.overall["contact"].metrics
    assert m["contact_start_f1"] == 1.0 and m["contact_end_f1"] == 1.0
    assert c1.baseline is not None
    assert c1.baseline.overall["contact"].metrics["contact_start_f1"] == 0.0

    a1 = run(TrainingJob("actions", "dv1", force=True), 2)
    assert a1.status == "passed", a1.reason
    assert a1.candidate is not None
    assert a1.candidate.overall["actions"].metrics["segment_f1_0.5"] == 1.0
    assert sum(a1.examples.values()) == 5 * (len(trained) - 1)

    # 승인 배포: 리포트에 적힌 비교 배포 모델(없음)이 지금과 같으니 배포한다.
    # 리포트 저장소 없이는 확인할 수 없다
    with engine.begin() as conn, pytest.raises(TrainingError, match="저장소"):
        deploy(conn, c1.model_version or "", now=FIXED_TIME.replace(second=50))
    with engine.begin() as conn:
        deploy(conn, c1.model_version or "", now=FIXED_TIME.replace(second=50), artifacts=artifacts)
    # 배포(50초)보다 이른 시각에 시작한 학습도 c1과 비교했으면 승인할 수 있다
    # (판정 시각은 학습 시작 시각이라 비교에 쓰지 않는다)
    c2 = run(TrainingJob("contact", "dv1", force=True), 20)
    c3 = run(TrainingJob("contact", "dv1", force=True), 21)
    assert (c2.status, c3.status) == ("passed", "passed")
    assert c2.baseline is not None and c2.baseline.model_versions["contact"] == c1.model_version
    with engine.begin() as conn:
        deploy(conn, c2.model_version or "", now=FIXED_TIME.replace(second=55), artifacts=artifacts)
    # c3는 c1과 비교했는데 지금 배포 모델은 c2다 → 다시 평가해야 한다
    with engine.begin() as conn, pytest.raises(TrainingError, match="다시 학습"):
        deploy(conn, c3.model_version or "", now=FIXED_TIME.replace(second=56), artifacts=artifacts)
    with engine.connect() as conn:
        deployed = list_model_versions(conn, "contact", ModelStatus.DEPLOYED)
        assert [d.model_version for d in deployed] == [c2.model_version]
