from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import sqlalchemy as sa
import yaml

from dlp_fixtures.actions import ENTITIES, generate_action_scenario
from dlp_fixtures.io import write_parquet
from dlp_fixtures.video import generate_blur_scenario
from dlp_media.ingest import ingest_session, load_manifest
from dlp_media.storage import S3Store
from dlp_prelabel.adapters.stubs import OraclePredictor
from dlp_prelabel.lift3d import DepthLifter
from dlp_prelabel.policy import load_policy
from dlp_prelabel.runner import CONTACT_PREFIX, CONTACT_STEP, run_prelabel
from dlp_schema.db.migrate import upgrade
from dlp_schema.db.repository import (
    get_labels,
    get_session,
    insert_labels,
    record_review,
    register_ontology,
    set_lifecycle,
    set_privacy_state,
    update_stream_sync,
)
from dlp_schema.episode import current_labels
from dlp_schema.labels import (
    BoxKeyframe,
    BoxTrackPayload,
    HandStatePayload,
    Keypoint,
    KeypointFrame,
    KeypointTrackPayload,
    LabelRecord,
    Provenance,
    Source,
    VerificationState,
)
from dlp_schema.ontology import load_ontology
from dlp_schema.predictor import ModelUnavailableError, Predictor
from dlp_schema.session import LifecycleState, PrivacyState, SyncMethod
from dlp_schema.testing import FIXED_TIME, make_label
from dlp_schema.validation import check_label

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


def test_prelabel_session_with_glove_contacts(pg: sa.Engine, tmp_path: Path) -> None:
    actions = generate_action_scenario(1, n_units=10)
    generate_blur_scenario(1).write(tmp_path / "bodycam.mp4")
    write_parquet(
        tmp_path / "glove_right.parquet",
        {"t_ms": actions.glove_t_ms, "pressure_0": actions.glove_pressure},
    )
    sid = f"pre-{uuid.uuid4().hex[:8]}"
    manifest = {
        "session_id": sid,
        "domain": "cleaning",
        "worker_id": "w01",
        "site_id": "site01",
        "consent_version": "c1",
        "recorded_at": FIXED_TIME.isoformat(),
        "ontology_version": "1.0.0",
        "streams": [
            {"stream_id": "bodycam", "kind": "bodycam", "path": "bodycam.mp4"},
            {"stream_id": "glove_right", "kind": "glove_right", "path": "glove_right.parquet"},
        ],
    }
    (tmp_path / "m.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    raw = S3Store.from_env("dlp-raw")
    with pg.begin() as conn:
        session = ingest_session(*load_manifest(tmp_path / "m.yaml"), raw, conn).session
        glove = session.stream("glove_right").model_copy(
            update={"sync_method": SyncMethod.TAP_EVENT}
        )
        update_stream_sync(conn, sid, glove)  # 장갑이 이미 동기화되었다고 둔다 (오프셋 0)
        set_privacy_state(conn, sid, PrivacyState.APPROVED)
        set_lifecycle(conn, sid, LifecycleState.PRIVACY_APPROVED)

    # 고정 객체 박스 정답을 객체 탐지 stub의 출처로 쓴다
    boxes = [
        make_label(
            BoxTrackPayload(
                entity_id=eid,
                class_id=cls,
                keyframes=tuple(
                    BoxKeyframe(t_ms=t, x=p[0] - 15, y=p[1] - 15, w=30, h=30)
                    for t in actions.frame_times
                ),
            ),
            label_id=f"gt-{eid}",
            session_id=sid,
            stream_id="bodycam",
            t_start_ms=actions.frame_times[0],
            t_end_ms=actions.frame_times[-1],
        )
        for eid, cls, _, p, _ in ENTITIES
        if eid in {"drawer_01", "bucket_01", "sink_01"}
    ]
    predictors: list[Predictor] = [
        OraclePredictor("hands", actions.labels, ("keypoint_track",), now=FIXED_TIME),
        OraclePredictor("objects", boxes, ("box_track",), now=FIXED_TIME),
    ]
    policy, ontology = load_policy(ROOT), load_ontology(ROOT / "config/ontology/v1")
    try:  # 실제 깊이 모델이 있으면 3D 궤적 단계도 돈다 (make export-models)
        lifter: DepthLifter | None = DepthLifter(ROOT, policy, now=FIXED_TIME)
    except ModelUnavailableError:
        lifter = None
    with pg.begin() as conn:
        summary = run_prelabel(
            conn, sid, raw, predictors, policy, ontology, FIXED_TIME, lifter=lifter
        )
        lifted = get_labels(conn, sid, kinds=["trajectory3d"])
    assert summary.produced == {"bodycam/hands": 1, "bodycam/objects": 3}
    if lifter is not None:
        # 손 관절 3개 + 객체 3개. 합성 영상의 깊이 값 자체는 의미가 없어 형식만 본다
        assert summary.lifted == len(lifted) == 6
        assert all(check_label(x, ontology) == [] for x in lifted)
    truth = [
        x
        for x in actions.labels
        if isinstance(x.payload, HandStatePayload) and x.payload.contact_target_kind != "none"
    ]
    assert summary.contacts == len(truth)

    with pg.begin() as conn:
        again = run_prelabel(conn, sid, raw, predictors, policy, ontology, FIXED_TIME)
        contacts = _builtin_contacts(get_labels(conn, sid, kinds=["hand_state"]))
        assert get_session(conn, sid).lifecycle_state is LifecycleState.PRELABELED
    assert sorted(again.skipped) == ["bodycam/hands", "bodycam/objects"] and again.contacts == 0
    assert again.lifted == 0
    for c, gt in zip(contacts, truth, strict=True):
        assert abs(c.t_start_ms - gt.t_start_ms) <= 33
        gp = gt.payload
        assert isinstance(gp, HandStatePayload) and isinstance(c.payload, HandStatePayload)
        if gp.target_id in {"drawer_01", "bucket_01", "sink_01"}:
            assert c.payload.target_id == gp.target_id
            assert c.payload.contact_target_kind == (
                "fixed_surface" if gp.target_id == "sink_01" else "object"
            )
    assert np.isfinite([c.confidence or 0 for c in contacts]).all()

    # 감사 회귀 (ADR 0015) 1: 예측기 버전이 바뀌면 새 ID로 넣고, 검수 전인 이전 버전만 지운다
    jittered = OraclePredictor("objects", boxes, ("box_track",), jitter_px=2.0, now=FIXED_TIME)
    with pg.begin() as conn:
        changed = run_prelabel(conn, sid, raw, [jittered], policy, ontology, FIXED_TIME)
        objects = current_labels(get_labels(conn, sid, kinds=["box_track"]))
    assert changed.produced == {"bodycam/objects": 3} and changed.retracted == 3
    assert {x.provenance.model_version for x in objects} == {jittered.version}
    # 감사 회귀: 접촉 단계 입력(객체 박스)이 바뀌었으므로 접촉도 다시 만들고,
    # 검수 전 이전 접촉은 지운다
    with pg.begin() as conn:
        history = get_labels(conn, sid, kinds=["hand_state"])
        live_contacts = _builtin_contacts(current_labels(history))
    assert changed.contacts == len(truth) == len(live_contacts)
    assert not {x.label_id for x in contacts} & {x.label_id for x in live_contacts}
    assert {x.provenance.model_version for x in live_contacts} != {
        x.provenance.model_version for x in contacts
    }
    contacts = live_contacts

    # 2: 검수자가 접촉 라벨을 모두 지운 뒤 다시 돌려도 되살리지 않는다 (이력으로 멱등 판단)
    with pg.begin() as conn:
        deletions = [
            x.model_copy(
                update={
                    "label_id": f"{x.label_id}:rev",
                    "parent_label_id": x.label_id,
                    "retracted": True,
                    "provenance": Provenance(source=Source.HUMAN),
                    "confidence": None,
                }
            )
            for x in contacts
        ]
        insert_labels(conn, deletions)
        third = run_prelabel(conn, sid, raw, predictors, policy, ontology, FIXED_TIME)
        live = _builtin_contacts(current_labels(get_labels(conn, sid, kinds=["hand_state"])))
    assert third.contacts == 0 and live == []

    # 3: 배포된 재학습 모델이 objects 어댑터를 대신하면, objects가 낸 검수 전 라벨은 지운다
    trained = OraclePredictor("trained-objects", boxes, ("box_track",), now=FIXED_TIME)
    with pg.begin() as conn:
        swapped = run_prelabel(
            conn, sid, raw, [trained], policy, ontology, FIXED_TIME, replaced=["objects"]
        )
        objects = current_labels(get_labels(conn, sid, kinds=["box_track"]))
    assert swapped.retracted == 3
    assert {x.provenance.model_version for x in objects} == {trained.version}
    with pg.begin() as conn:
        live = _builtin_contacts(current_labels(get_labels(conn, sid, kinds=["hand_state"])))
    assert swapped.contacts == len(live) == len(truth)  # 입력이 바뀌어 다시 만들었다

    # 4: 배포된 재학습 접촉 모델이 기본 접촉 단계를 대신하면,
    # 기본 단계는 돌지 않고 검수 전 접촉을 지운다
    with pg.begin() as conn:
        replaced = run_prelabel(
            conn, sid, raw, [trained], policy, ontology, FIXED_TIME, replaced=[CONTACT_STEP]
        )
        live = _builtin_contacts(current_labels(get_labels(conn, sid, kinds=["hand_state"])))
    assert replaced.contacts == 0 and replaced.retracted == len(truth) and live == []


def _builtin_contacts(labels: list[LabelRecord]) -> list[LabelRecord]:
    return [x for x in labels if (x.provenance.model_version or "").startswith(CONTACT_PREFIX)]


def _coco_person(entity: str, t: np.ndarray, motion: np.ndarray, seed: int) -> LabelRecord:
    rng = np.random.default_rng(seed)
    x = np.cumsum(motion * 8 + rng.normal(0, 0.3, t.size)) + 100
    frames: list[KeypointFrame] = []
    for i, tm in enumerate(t):
        pts = [Keypoint(x=50.0, y=50.0, visibility=2) for _ in range(17)]
        pts[9] = pts[10] = Keypoint(x=float(x[i]), y=120.0, visibility=2)
        frames.append(KeypointFrame(t_ms=int(tm), points=tuple(pts)))
    return make_label(
        KeypointTrackPayload(entity_id=entity, skeleton="coco17", keyframes=tuple(frames)),
        label_id=f"gt-{entity}",
        stream_id="third",
        t_start_ms=int(t[0]),
        t_end_ms=int(t[-1]),
    )


def test_wearer_copy_is_unreviewed_and_rematched_after_body_model_change(
    pg: sa.Engine, tmp_path: Path
) -> None:
    """감사 회귀: 착용자 사본은 원래 트랙의 검수 상태를 물려받지 않고, 전신 모델 버전이 바뀌어
    사본이 지워지면 새 트랙으로 다시 찾는다 (이력의 착용자 표시가 다시 찾기를 막지 않는다)."""
    rng = np.random.default_rng(0)
    t = np.arange(0, 20_000, 33.0)
    bursts = np.zeros(t.size)
    for s in rng.uniform(0, 19_000, 12):
        bursts += np.exp(-(((t - s) / 150) ** 2))
    people = [
        _coco_person("wearer_track", t, bursts, 1),
        _coco_person("other", t, np.convolve(rng.random(t.size), np.ones(15) / 15, "same"), 2),
    ]
    imu_t = np.arange(0, 20_000, 5.0)
    zeros = np.zeros(imu_t.size)
    write_parquet(
        tmp_path / "imu.parquet",
        {
            "t_ms": imu_t, "ax": zeros, "ay": zeros,
            "az": 9.81 + np.interp(imu_t, t, bursts) * 3, "gx": zeros, "gy": zeros, "gz": zeros,
        },
    )  # fmt: skip
    video = generate_blur_scenario(1)
    video.write(tmp_path / "bodycam.mp4")
    video.write(tmp_path / "third.mp4")
    sid = f"wear-{uuid.uuid4().hex[:8]}"
    manifest = {
        "session_id": sid, "domain": "caregiving", "worker_id": "w01", "site_id": "site01",
        "consent_version": "c1", "recorded_at": FIXED_TIME.isoformat(), "ontology_version": "1.0.0",
        "streams": [
            {"stream_id": "bodycam", "kind": "bodycam", "path": "bodycam.mp4"},
            {"stream_id": "third", "kind": "third_person", "path": "third.mp4"},
            {"stream_id": "imu", "kind": "imu", "path": "imu.parquet"},
        ],
    }  # fmt: skip
    (tmp_path / "m.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    raw = S3Store.from_env("dlp-raw")
    with pg.begin() as conn:
        session = ingest_session(*load_manifest(tmp_path / "m.yaml"), raw, conn).session
        third = session.stream("third").model_copy(update={"sync_method": SyncMethod.TAP_EVENT})
        update_stream_sync(conn, sid, third)
    policy, ontology = load_policy(ROOT), load_ontology(ROOT / "config/ontology/v1")
    body = OraclePredictor("body", people, ("keypoint_track",), now=FIXED_TIME)
    with pg.begin() as conn:  # IMU가 아직 동기화 전이라 착용자 매칭은 건너뛴다
        first = run_prelabel(conn, sid, raw, [body], policy, ontology, FIXED_TIME)
        tracks = [
            x for x in get_labels(conn, sid, kinds=["keypoint_track"]) if x.stream_id == "third"
        ]
        assert first.wearer is None and len(tracks) == 2
        for x in tracks:  # 검수자가 3인칭 인물 트랙을 승인했다
            record_review(conn, x.label_id, VerificationState.HUMAN_APPROVED, "rev01", FIXED_TIME)
        imu = session.stream("imu").model_copy(update={"sync_method": SyncMethod.SHARED_CLOCK})
        update_stream_sync(conn, sid, imu)
        matched = run_prelabel(conn, sid, raw, [body], policy, ontology, FIXED_TIME)
        wearer = [x for x in get_labels(conn, sid) if x.label_id.endswith(":wearer")]
    assert matched.wearer is not None and len(wearer) == 1
    w = wearer[0]
    assert isinstance(w.payload, KeypointTrackPayload) and w.payload.entity_id == "wearer"
    # 모델 출력이므로 원래 트랙의 승인 상태를 물려받지 않는다
    assert w.verification.state is VerificationState.UNREVIEWED

    # 원래 트랙이 승인되어 있으면 전신 모델이 바뀌어도 착용자 사본을 지우지 않는다 (승인 보존)
    body2 = OraclePredictor("body", people, ("keypoint_track",), jitter_px=0.5, now=FIXED_TIME)
    with pg.begin() as conn:
        run_prelabel(conn, sid, raw, [body2], policy, ontology, FIXED_TIME)
        live = [x for x in current_labels(get_labels(conn, sid)) if x.label_id.endswith(":wearer")]
    assert [x.label_id for x in live] == [w.label_id]

    # 검수 전 트랙이면: 새 전신 모델 버전이 사본을 지우고, 새 트랙으로 착용자를 다시 찾는다
    sid2 = f"{sid}-b"
    manifest["session_id"] = sid2
    (tmp_path / "m.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    with pg.begin() as conn:
        s2 = ingest_session(*load_manifest(tmp_path / "m.yaml"), raw, conn).session
        for name, method in (("third", SyncMethod.TAP_EVENT), ("imu", SyncMethod.SHARED_CLOCK)):
            update_stream_sync(
                conn, sid2, s2.stream(name).model_copy(update={"sync_method": method})
            )
        a = run_prelabel(conn, sid2, raw, [body], policy, ontology, FIXED_TIME)
        b = run_prelabel(conn, sid2, raw, [body2], policy, ontology, FIXED_TIME)
        live = [x for x in current_labels(get_labels(conn, sid2)) if x.label_id.endswith(":wearer")]
    assert a.wearer is not None and b.wearer is not None and a.wearer != b.wearer
    assert len(live) == 1 and live[0].parent_label_id == b.wearer
    assert live[0].verification.state is VerificationState.UNREVIEWED
