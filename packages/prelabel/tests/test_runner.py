"""프리라벨 실행기 통합 테스트 (`@pytest.mark.services`: PostgreSQL·SeaweedFS S3 필요, `make up`).

테스트마다 새 DB를 만들고 지운다(`pg`). 합성 세션을 실제 수집 경로(`ingest_session`)로 원본
버킷에 올리고, 모델은 Oracle stub으로 대신한다. 멱등성, 버전 변경 시 검수 전 라벨만 지우기,
검수자 삭제 존중, 배포 모델 교체, 착용자 다시 찾기, 타임라인 모델 한 번 실행을 본다 (ADR 0015,
0026).
"""

from __future__ import annotations

import itertools
import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import sqlalchemy as sa
import yaml
from numpy.typing import NDArray

from dlp_fixtures.actions import ENTITIES, ActionScenario, generate_action_scenario
from dlp_fixtures.io import write_parquet
from dlp_fixtures.video import generate_blur_scenario, write_video
from dlp_media.ingest import ingest_session, load_manifest
from dlp_media.storage import S3Store
from dlp_prelabel.adapters.stubs import OraclePredictor
from dlp_prelabel.common import Image, iter_frames
from dlp_prelabel.lift3d import DepthLifter, lift_tracks
from dlp_prelabel.policy import PrelabelPolicy, load_policy
from dlp_prelabel.runner import CONTACT_PREFIX, CONTACT_STEP, pick_lift_tracks, run_prelabel
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
    Hand,
    HandStatePayload,
    Keypoint,
    KeypointFrame,
    KeypointTrackPayload,
    LabelRecord,
    Provenance,
    Source,
    Trajectory3DPayload,
    Trajectory3DSample,
    VerificationState,
)
from dlp_schema.ontology import load_ontology
from dlp_schema.predictor import Clip, ModelUnavailableError, Predictor
from dlp_schema.session import LifecycleState, PrivacyState, SyncMethod
from dlp_schema.testing import FIXED_TIME, make_label
from dlp_schema.validation import check_label

pytestmark = pytest.mark.services
ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def pg() -> Iterator[sa.Engine]:
    """테스트 전용 PostgreSQL DB를 만들고 마이그레이션·온톨로지 v1 등록 후 엔진을 넘긴다. 끝나면
    지운다.
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


def _write_bodycam(path: Path, actions: ActionScenario) -> None:
    """행동 시나리오 전체(마지막 손 키프레임 뒤까지)를 덮는 합성 바디캠 영상을 쓴다.

    세션 길이(`Session.duration_ms`)는 바디캠 PTS로 정해지고 접촉은 그 안으로 잘린다 (F-PAR-02).
    영상이 시나리오보다 짧으면 영상 끝 뒤 접촉이 빠져 정답 수와 맞지 않는다.
    """
    generate_blur_scenario(1, duration_ms=actions.frame_times[-1] + 200).write(path)


def test_prelabel_session_with_glove_contacts(pg: sa.Engine, tmp_path: Path) -> None:
    """장갑이 있는 바디캠 세션에서 프리라벨 전체 흐름과 감사 회귀 네 가지를 본다.

    1차: 손·객체 stub 결과가 들어가고 접촉 수가 정답과 같으며 경계가 한 프레임 안, 대상·종류가
    정답과 같다 (깊이 가중치가 있으면 3D 궤적 6개). 재실행은 모두 건너뛰고 생애주기는 prelabeled.
    회귀 1: 객체 예측기 버전이 바뀌면 새 ID로 넣고 검수 전 이전 버전만 지우며, 입력이 바뀐 접촉도
    다시 만든다. 회귀 2: 검수자가 접촉을 모두 지우면 다시 돌려도 되살리지 않는다. 회귀 3: 배포
    모델이 objects를 대신하면 objects의 검수 전 라벨을 지운다. 회귀 4: 배포 접촉 모델이 접촉
    단계를 대신하면 기본 접촉 단계는 돌지 않고 검수 전 접촉을 지운다.
    """
    actions = generate_action_scenario(1, n_units=10)
    _write_bodycam(tmp_path / "bodycam.mp4", actions)
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
    """기본 접촉 단계(contact-heuristic)가 만든 라벨만 고른다."""
    return [x for x in labels if (x.provenance.model_version or "").startswith(CONTACT_PREFIX)]


def _coco_person(entity: str, t: np.ndarray, motion: np.ndarray, seed: int) -> LabelRecord:
    """3인칭 coco17 인물 트랙 라벨: 두 손목 x가 motion에 비례해 움직인다 (seed로 잡음 고정)."""
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
    사본이 지워지면 새 트랙으로 다시 찾는다 (이력의 착용자 표시가 다시 찾기를 막지 않는다).

    정답 근거: IMU 가속도에 착용자 운동 패턴(bursts)을 넣었으므로 "wearer_track"이 뽑힌다. 원래
    트랙이 승인됐으면 전신 모델이 바뀌어도 사본을 지우지 않는다.
    """
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


def test_reviewed_contacts_are_not_duplicated_when_inputs_change(
    pg: sa.Engine, tmp_path: Path
) -> None:
    """감사 회귀 (4차): 접촉을 승인한 뒤 입력(객체 박스)이 바뀌어 접촉 단계가 다시 돌아도, 승인된
    접촉과 같은 손에서 겹치는 새 접촉을 넣지 않는다. 장갑 압력 채널 접두사가 바뀌어도 다시 돈다.

    접두사 "pressure"는 기본값과 같아 버전이 같고(0개), "pressure_"는 다른 버전이라 다시 만든다.
    """
    actions = generate_action_scenario(1, n_units=10)
    _write_bodycam(tmp_path / "bodycam.mp4", actions)
    write_parquet(
        tmp_path / "glove_right.parquet",
        {"t_ms": actions.glove_t_ms, "pressure_0": actions.glove_pressure},
    )
    sid = f"dup-{uuid.uuid4().hex[:8]}"
    manifest = {
        "session_id": sid, "domain": "cleaning", "worker_id": "w01", "site_id": "site01",
        "consent_version": "c1", "recorded_at": FIXED_TIME.isoformat(), "ontology_version": "1.0.0",
        "streams": [
            {"stream_id": "bodycam", "kind": "bodycam", "path": "bodycam.mp4"},
            {"stream_id": "glove_right", "kind": "glove_right", "path": "glove_right.parquet"},
        ],
    }  # fmt: skip
    (tmp_path / "m.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    raw = S3Store.from_env("dlp-raw")
    with pg.begin() as conn:
        session = ingest_session(*load_manifest(tmp_path / "m.yaml"), raw, conn).session
        glove = session.stream("glove_right")
        update_stream_sync(
            conn, sid, glove.model_copy(update={"sync_method": SyncMethod.TAP_EVENT})
        )
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
    with pg.begin() as conn:
        first = run_prelabel(conn, sid, raw, predictors, policy, ontology, FIXED_TIME)
        contacts = _builtin_contacts(current_labels(get_labels(conn, sid, kinds=["hand_state"])))
        approved = contacts[::2]  # 검수자가 접촉 절반을 승인했다
        for x in approved:
            record_review(conn, x.label_id, VerificationState.HUMAN_APPROVED, "rev01", FIXED_TIME)
    assert first.contacts == len(contacts) > 1
    # 검수자가 박스를 고친 것과 같은 입력 변화 (객체 예측기 버전 변경)
    jittered = OraclePredictor("objects", boxes, ("box_track",), jitter_px=2.0, now=FIXED_TIME)
    with pg.begin() as conn:
        second = run_prelabel(conn, sid, raw, [jittered], policy, ontology, FIXED_TIME)
        live = _builtin_contacts(current_labels(get_labels(conn, sid, kinds=["hand_state"])))
    assert second.contacts == len(contacts) - len(approved)  # 승인 구간과 겹치는 새 접촉은 버린다
    assert {x.label_id for x in approved} <= {x.label_id for x in live}
    assert len(live) == len(contacts)
    for a, b in itertools.combinations(live, 2):
        assert isinstance(a.payload, HandStatePayload) and isinstance(b.payload, HandStatePayload)
        same_hand = a.payload.hand is b.payload.hand
        assert not (same_hand and a.t_start_ms < b.t_end_ms and b.t_start_ms < a.t_end_ms)

    # 장갑 압력 채널 접두사(sync.yaml)는 접촉 모델 버전에 들어간다: 바뀌면 다시 돈다
    with pg.begin() as conn:
        same = run_prelabel(
            conn, sid, raw, [jittered], policy, ontology, FIXED_TIME, pressure_prefixes=["pressure"]
        )
        other = run_prelabel(
            conn, sid, raw, [jittered], policy, ontology, FIXED_TIME,
            pressure_prefixes=["pressure_"],
        )  # fmt: skip
    assert same.contacts == 0 and other.contacts == second.contacts


class _TimelineContacts:
    """배포된 재학습 접촉 모델 흉내: 마스터 타임라인 접촉 구간(stream_id 없음)을 낸다."""

    name = "trained-contact"
    version = "trained-contact-v1"

    def __init__(self, truth: list[LabelRecord]) -> None:
        """정답 중 손 상태 라벨만 들고, 불린 스트림을 `calls`에 기록한다."""
        self.truth = [x for x in truth if isinstance(x.payload, HandStatePayload)]
        self.calls: list[str] = []

    def run(self, clip: Clip) -> list[LabelRecord]:
        """정답 손 상태를 이 세션·버전의 모델 출처, stream_id=None 구간으로 다시 쓴다."""
        self.calls.append(clip.stream_id)
        return [
            x.model_copy(
                update={
                    "label_id": f"{clip.session_id}-{clip.stream_id}-{self.name}-{i:04d}",
                    "session_id": clip.session_id,
                    "stream_id": None,
                    "provenance": Provenance(source=Source.MODEL, model_version=self.version),
                    "confidence": 0.9,
                }
            )
            for i, x in enumerate(self.truth)
        ]


def test_timeline_model_runs_once_on_the_reference_stream(pg: sa.Engine, tmp_path: Path) -> None:
    """감사 회귀 (4차): 마스터 타임라인 구간을 내는 배포 모델(접촉)은 3인칭 스트림이 있어도 기준
    스트림에서 세션당 한 번만 돈다 (같은 접촉이 두 번 남지 않는다). 다시 돌려도 건너뛴다.
    """
    actions = generate_action_scenario(1, n_units=4)
    video = generate_blur_scenario(1)
    video.write(tmp_path / "bodycam.mp4")
    video.write(tmp_path / "third.mp4")
    sid = f"tl-{uuid.uuid4().hex[:8]}"
    manifest = {
        "session_id": sid, "domain": "cleaning", "worker_id": "w01", "site_id": "site01",
        "consent_version": "c1", "recorded_at": FIXED_TIME.isoformat(), "ontology_version": "1.0.0",
        "streams": [
            {"stream_id": "bodycam", "kind": "bodycam", "path": "bodycam.mp4"},
            {"stream_id": "third", "kind": "third_person", "path": "third.mp4"},
        ],
    }  # fmt: skip
    (tmp_path / "m.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    raw = S3Store.from_env("dlp-raw")
    with pg.begin() as conn:
        ingest_session(*load_manifest(tmp_path / "m.yaml"), raw, conn)
    policy, ontology = load_policy(ROOT), load_ontology(ROOT / "config/ontology/v1")
    assert "trained-contact" in policy.timeline_predictors
    model = _TimelineContacts([x.model_copy(update={"session_id": sid}) for x in actions.labels])
    with pg.begin() as conn:
        first = run_prelabel(
            conn, sid, raw, [model], policy, ontology, FIXED_TIME, replaced=[CONTACT_STEP]
        )
        again = run_prelabel(
            conn, sid, raw, [model], policy, ontology, FIXED_TIME, replaced=[CONTACT_STEP]
        )
        states = current_labels(get_labels(conn, sid, kinds=["hand_state"]))
    assert model.calls == ["bodycam"]
    assert first.produced == {"bodycam/trained-contact": len(model.truth)}
    assert again.skipped == ["bodycam/trained-contact"]
    assert len(states) == len(model.truth) > 0


class _UnitDepth:
    """깊이를 늘 1 m로 내는 가짜 깊이 모델."""

    def predict(self, image: Image) -> NDArray[np.float32]:
        """(H, W) 1 m 깊이 맵."""
        return np.ones(image.shape[:2], dtype=np.float32)


class _UnitLifter(DepthLifter):
    """가중치 파일 없이 `_UnitDepth`로 도는 깊이 3D 단계 (가중치 확인·ONNX 세션을 건너뛴다)."""

    def __init__(self, policy: PrelabelPolicy) -> None:  # pyright: ignore[reportMissingSuperCall]
        """`DepthLifter.__init__`의 가중치 확인 대신 가짜 모델과 버전을 직접 둔다."""
        self.path = Path("unused.onnx")
        self.version = f"depth3d-unit+p{policy.digest('depth')}"
        self.policy, self.now = policy, FIXED_TIME
        self.model = _UnitDepth()


def _bodycam_only_session(
    pg: sa.Engine, tmp_path: Path, actions: ActionScenario, prefix: str
) -> tuple[str, S3Store]:
    """손 키프레임 시각 그대로 쓴 회색 바디캠 영상 하나뿐인 세션을 수집한다 (3D 단계 테스트용).

    깊이를 읽는 프레임이 손 키프레임 시각과 같아 `lift_tracks` 결과를 정답으로 쓸 수 있다.

    Returns:
        (세션 ID, 원본 저장소).
    """
    write_video(
        tmp_path / "bodycam.mp4",
        ((t, np.full((240, 320, 3), 90, np.uint8)) for t in actions.frame_times),
        width=320,
        height=240,
    )
    sid = f"{prefix}-{uuid.uuid4().hex[:8]}"
    manifest = {
        "session_id": sid, "domain": "cleaning", "worker_id": "w01", "site_id": "site01",
        "consent_version": "c1", "recorded_at": FIXED_TIME.isoformat(), "ontology_version": "1.0.0",
        "streams": [{"stream_id": "bodycam", "kind": "bodycam", "path": "bodycam.mp4"}],
    }  # fmt: skip
    (tmp_path / "m.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    raw = S3Store.from_env("dlp-raw")
    with pg.begin() as conn:
        ingest_session(*load_manifest(tmp_path / "m.yaml"), raw, conn)
    return sid, raw


def test_depth_lift_uses_the_reviewed_track_when_hand_versions_coexist(
    pg: sa.Engine, tmp_path: Path
) -> None:
    """F-PAR-01 회귀: 승인된 이전 버전 손 트랙과 새 버전 손 트랙(같은 entity_id)이 함께 현재 라벨일
    때 3D 단계가 프리라벨 전체를 되돌리지 않고, 승인 트랙으로 궤적을 만든다.

    예전에는 두 트랙의 같은 시각 표본이 한 궤적에 두 번 들어가 "키프레임 시각은 중복 없이
    오름차순" 검증이 실패하고 트랜잭션 전체가 롤백됐다 (다시 돌려도 같았다).

    정답: 바디캠은 손 키프레임 시각 그대로 쓴 영상이라 깊이를 읽는 프레임이 정해진다. 3D 표본은
    같은 가짜 깊이로 승인 트랙만 올린 결과와 같고, 흔들린 새 트랙으로 올린 결과와는 다르다. 다시
    돌리면 같은 입력이라 건너뛴다.
    """
    actions = generate_action_scenario(1, n_units=2)
    sid, raw = _bodycam_only_session(pg, tmp_path, actions, "lift")
    policy, ontology = load_policy(ROOT), load_ontology(ROOT / "config/ontology/v1")
    hands = OraclePredictor("hands", actions.labels, ("keypoint_track",), now=FIXED_TIME)
    hands2 = OraclePredictor(
        "hands", actions.labels, ("keypoint_track",), jitter_px=3.0, now=FIXED_TIME
    )
    with pg.begin() as conn:
        run_prelabel(conn, sid, raw, [hands], policy, ontology, FIXED_TIME)
        [old] = get_labels(conn, sid, kinds=["keypoint_track"])
        record_review(conn, old.label_id, VerificationState.HUMAN_APPROVED, "rev01", FIXED_TIME)
        # 손 모델 버전이 바뀌었다: 승인된 이전 트랙은 남고 새 트랙이 같은 entity_id로 들어온다
        run_prelabel(conn, sid, raw, [hands2], policy, ontology, FIXED_TIME)
        tracks = current_labels(get_labels(conn, sid, kinds=["keypoint_track"]))
    entities = {x.payload.entity_id for x in tracks if isinstance(x.payload, KeypointTrackPayload)}
    assert len(tracks) == 2 and len(entities) == 1
    [new] = [x for x in tracks if x.label_id != old.label_id]

    lifter = _UnitLifter(policy)
    with pg.begin() as conn:
        summary = run_prelabel(
            conn, sid, raw, [hands2], policy, ontology, FIXED_TIME, lifter=lifter
        )
        again = run_prelabel(conn, sid, raw, [hands2], policy, ontology, FIXED_TIME, lifter=lifter)
        lifted = get_labels(conn, sid, kinds=["trajectory3d"])
    assert summary.lifted == len(lifted) == len(policy.depth.hand_points)
    assert again.lifted == 0
    assert all(check_label(x, ontology) == [] for x in lifted)
    video = tmp_path / "bodycam.mp4"
    from_old = lift_tracks(iter_frames(video), _UnitDepth(), [old], None, policy.depth)
    from_new = lift_tracks(iter_frames(video), _UnitDepth(), [new], None, policy.depth)
    for x in lifted:
        assert isinstance(x.payload, Trajectory3DPayload)
        key = (x.payload.entity_id, x.payload.part)
        assert list(x.payload.samples) == from_old[key] != from_new[key]


def _lifted_from(
    video: Path, track: LabelRecord, policy: PrelabelPolicy
) -> dict[tuple[str, str | None], list[Trajectory3DSample]]:
    """정답: 가짜 깊이(1 m)로 트랙 하나만 올린 (개체, 부위) → 3D 표본."""
    return lift_tracks(iter_frames(video), _UnitDepth(), [track], None, policy.depth)


def _two_unreviewed_hand_tracks(actions: ActionScenario) -> list[Predictor]:
    """같은 entity_id 손 트랙을 내는 두 어댑터 ("hands", 흔들림을 넣은 "hands-alt").

    어댑터 이름이 달라 라벨 ID 접두사가 다르므로 서로의 이전 버전으로 지워지지 않고, 둘 다 검수 전
    모델 트랙으로 함께 현재 라벨이 된다.
    """
    kinds = ("keypoint_track",)
    return [
        OraclePredictor("hands", actions.labels, kinds, now=FIXED_TIME),
        OraclePredictor("hands-alt", actions.labels, kinds, jitter_px=3.0, now=FIXED_TIME),
    ]


def test_depth_lift_ignores_body_tracks_it_does_not_lift(pg: sa.Engine, tmp_path: Path) -> None:
    """검증 지적 회귀 (F-PAR-09): 3D 입력 해시는 실제로 올리는 트랙(`pick_lift_tracks`)으로 만든다.
    바디캠에 coco17 전신 트랙이 더해져도 3D로 올리지 않으므로 버전이 같아 다시 만들지 않는다.

    예전 해시(바디캠 키포인트·박스 트랙 전체)는 전신 트랙 ID가 더해져 버전이 바뀌고, 같은 손 궤적을
    지웠다가 다시 만들었다.

    정답: 3D 궤적은 hand_points 수만큼 한 번만 만들어지고, 전신 어댑터를 더해 다시 돌려도 이력이
    그대로다. 표본은 고른 손 트랙만 올린 `lift_tracks` 결과와 같다.
    """
    actions = generate_action_scenario(1, n_units=2)
    sid, raw = _bodycam_only_session(pg, tmp_path, actions, "body3d")
    policy, ontology = load_policy(ROOT), load_ontology(ROOT / "config/ontology/v1")
    hands = _two_unreviewed_hand_tracks(actions)
    t = np.asarray(actions.frame_times, dtype=np.float64)
    body = OraclePredictor(
        "body", [_coco_person("worker", t, np.zeros(t.size), 3)], ("keypoint_track",),
        now=FIXED_TIME,
    )  # fmt: skip
    lifter = _UnitLifter(policy)
    with pg.begin() as conn:
        first = run_prelabel(conn, sid, raw, hands, policy, ontology, FIXED_TIME, lifter=lifter)
        lifted = get_labels(conn, sid, kinds=["trajectory3d"])
        with_body = run_prelabel(
            conn, sid, raw, [*hands, body], policy, ontology, FIXED_TIME, lifter=lifter
        )
        after = get_labels(conn, sid, kinds=["trajectory3d"])
        tracks = current_labels(get_labels(conn, sid, kinds=["keypoint_track"]))
    skeletons = sorted(
        x.payload.skeleton for x in tracks if isinstance(x.payload, KeypointTrackPayload)
    )
    assert skeletons == ["coco17", "hand21", "hand21"]
    assert first.lifted == len(lifted) == len(policy.depth.hand_points)
    assert with_body.produced == {"bodycam/body": 1}
    assert with_body.lifted == 0 and after == lifted
    [picked] = pick_lift_tracks(tracks, "bodycam")
    want = _lifted_from(tmp_path / "bodycam.mp4", picked, policy)
    for x in lifted:
        assert isinstance(x.payload, Trajectory3DPayload)
        assert list(x.payload.samples) == want[(x.payload.entity_id, x.payload.part)]


def test_approving_the_other_hand_track_relifts_and_keeps_reviewed_trajectories(
    pg: sa.Engine, tmp_path: Path
) -> None:
    """검증 지적 회귀: 같은 entity_id의 검수 전 손 트랙 둘 중 고르지 않은 쪽을 승인하면 고르는
    트랙이 바뀐다 (승인은 라벨 ID를 바꾸지 않는다). 3D는 고른 트랙으로 만든 입력 해시가 바뀌어 새
    버전으로 다시 돌고, 이전 버전 3D 중 검수 전인 것만 지운다.

    예전 해시(후보 트랙 ID 전체)는 승인 전후가 같아 3D가 다시 돌지 않았고, 승인 트랙이 아닌
    트랙으로 올린 궤적이 남았다.

    시나리오: 첫 실행에서 고른 트랙(`pick_lift_tracks`)으로 3D 궤적(hand_points 수)이 생긴다. 그중
    하나를 검수자가 승인하고, 고르지 않은 손 트랙도 승인한 뒤 다시 돌린다.
    정답: 새 궤적은 승인한 손 트랙만 올린 `lift_tracks` 결과와 같고, 승인한 이전 궤적의 (개체,
    부위)는 새 궤적을 만들지 않는다 (hand_points 수 - 1개). 지운 것은 승인하지 않은 이전 궤적뿐이다.
    다시 돌리면 같은 버전이라 건너뛴다.
    """
    actions = generate_action_scenario(1, n_units=2)
    sid, raw = _bodycam_only_session(pg, tmp_path, actions, "relift")
    policy, ontology = load_policy(ROOT), load_ontology(ROOT / "config/ontology/v1")
    hands = _two_unreviewed_hand_tracks(actions)
    lifter = _UnitLifter(policy)
    video = tmp_path / "bodycam.mp4"
    with pg.begin() as conn:
        first = run_prelabel(conn, sid, raw, hands, policy, ontology, FIXED_TIME, lifter=lifter)
        before = get_labels(conn, sid, kinds=["trajectory3d"])
        tracks = current_labels(get_labels(conn, sid, kinds=["keypoint_track"]))
    assert len(tracks) == 2 and all(
        x.verification.state is VerificationState.UNREVIEWED for x in tracks
    )
    [picked] = pick_lift_tracks(tracks, "bodycam")
    [other] = [x for x in tracks if x.label_id != picked.label_id]
    from_picked, from_other = (_lifted_from(video, x, policy) for x in (picked, other))
    n = len(policy.depth.hand_points)
    assert first.lifted == len(before) == n
    for x in before:
        assert isinstance(x.payload, Trajectory3DPayload)
        key = (x.payload.entity_id, x.payload.part)
        assert list(x.payload.samples) == from_picked[key] != from_other[key]

    reviewed = before[0]
    with pg.begin() as conn:
        record_review(
            conn, reviewed.label_id, VerificationState.HUMAN_APPROVED, "rev01", FIXED_TIME
        )
        record_review(conn, other.label_id, VerificationState.HUMAN_APPROVED, "rev01", FIXED_TIME)
        relift = run_prelabel(conn, sid, raw, hands, policy, ontology, FIXED_TIME, lifter=lifter)
        again = run_prelabel(conn, sid, raw, hands, policy, ontology, FIXED_TIME, lifter=lifter)
        history = get_labels(conn, sid, kinds=["trajectory3d"])
    assert relift.lifted == n - 1 and again.lifted == 0
    removed = {x.parent_label_id for x in history if x.retracted}
    assert removed == {x.label_id for x in before} - {reviewed.label_id}
    live = current_labels(history)
    assert reviewed.label_id in {x.label_id for x in live} and len(live) == n
    new = [x for x in live if x.label_id != reviewed.label_id]
    assert all(x.provenance.model_version != reviewed.provenance.model_version for x in new)
    for x in new:
        assert isinstance(x.payload, Trajectory3DPayload)
        assert list(x.payload.samples) == from_other[(x.payload.entity_id, x.payload.part)]
    assert all(check_label(x, ontology) == [] for x in new)


def test_glove_started_before_the_bodycam_keeps_only_contacts_inside_the_session(
    pg: sa.Engine, tmp_path: Path
) -> None:
    """F-PAR-02 회귀: 장갑을 바디캠보다 2초 먼저 켜(offset -2000 ms) 마스터로 바꾼 장갑 접촉이 0
    앞이나 세션 끝 뒤에 놓여도 프리라벨이 롤백되지 않고, 세션 [0, 길이] 안 구간만 손 상태가 된다.

    예전에는 음수 시각 손 상태 라벨이 Ms(ge=0) 검증에서 실패해 프리라벨 트랜잭션 전체가 롤백됐다.

    장갑 시계 접촉(압력 1, ms), 두 장갑 모두 offset -2000 ms:
    - 오른손: 500~1200 → 마스터 -1500~-800 (버림), 1600~2600 → -400~600 (0~600), 3000~4000 →
      1000~2000 (그대로), 4500~5800 → 2500~3800 (2500~세션 길이).
    - 왼손: 1300~2030 → -700~30 (0~30은 장갑 최소 길이 60 ms보다 짧은 조각이라 버림), 3200~3500 →
      1200~1500 (그대로).
    세션 길이는 바디캠 PTS로 정한 `Session.duration_ms` (약 3000 ms 합성 영상).
    """
    generate_blur_scenario(1).write(tmp_path / "bodycam.mp4")
    t = np.arange(0, 6_000, 10.0)
    contacts_ms = {
        "glove_right": ((500, 1_200), (1_600, 2_600), (3_000, 4_000), (4_500, 5_800)),
        "glove_left": ((1_300, 2_030), (3_200, 3_500)),
    }
    for name, spans_ms in contacts_ms.items():
        pressure = np.zeros(t.size)
        for a, b in spans_ms:
            pressure[(t >= a) & (t < b)] = 1.0
        write_parquet(tmp_path / f"{name}.parquet", {"t_ms": t, "pressure_0": pressure})
    sid = f"neg-{uuid.uuid4().hex[:8]}"
    manifest = {
        "session_id": sid, "domain": "cleaning", "worker_id": "w01", "site_id": "site01",
        "consent_version": "c1", "recorded_at": FIXED_TIME.isoformat(), "ontology_version": "1.0.0",
        "streams": [
            {"stream_id": "bodycam", "kind": "bodycam", "path": "bodycam.mp4"},
            *({"stream_id": n, "kind": n, "path": f"{n}.parquet"} for n in contacts_ms),
        ],
    }  # fmt: skip
    (tmp_path / "m.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    raw = S3Store.from_env("dlp-raw")
    policy, ontology = load_policy(ROOT), load_ontology(ROOT / "config/ontology/v1")
    assert 30 < policy.contact.glove.min_duration_ms <= 300  # 왼손 0~30 조각만 최소 길이 밑이다
    with pg.begin() as conn:
        session = ingest_session(*load_manifest(tmp_path / "m.yaml"), raw, conn).session
        for name in contacts_ms:
            glove = session.stream(name).model_copy(
                update={"sync_method": SyncMethod.TAP_EVENT, "offset_ms": -2_000.0}
            )
            update_stream_sync(conn, sid, glove)
    duration = session.duration_ms
    assert 2_500 + 300 < duration < 3_800  # 마지막 접촉이 세션 끝에 걸치고 잘려도 충분히 길다
    with pg.begin() as conn:
        summary = run_prelabel(conn, sid, raw, [], policy, ontology, FIXED_TIME)
        contacts = _builtin_contacts(current_labels(get_labels(conn, sid, kinds=["hand_state"])))
    spans: dict[Hand, list[tuple[int, int]]] = {}
    for x in contacts:
        assert isinstance(x.payload, HandStatePayload)
        spans.setdefault(x.payload.hand, []).append((x.t_start_ms, x.t_end_ms))
    assert {h: sorted(v) for h, v in spans.items()} == {
        Hand.RIGHT: [(0, 600), (1_000, 2_000), (2_500, duration)],
        Hand.LEFT: [(1_200, 1_500)],
    }
    assert summary.contacts == len(contacts)
    assert all(check_label(x, ontology) == [] for x in contacts)


def test_person_box_contacts_are_left_for_review(pg: sa.Engine, tmp_path: Path) -> None:
    """F-PAR-03 회귀: 바디캠 사람 박스(온톨로지 person)와의 접촉은 object 손 상태로 쓰지 않는다.

    `test_prelabel_session_with_glove_contacts`와 같은 장갑 세션에서 버킷 자리 박스를 COCO person
    박스(entity person_01, class person)로 바꾼다. 사람 접촉은 person + body_part가 필수인데
    박스로는 부위를 알 수 없어(온톨로지에 "모름" 부위가 없다) 접촉 라벨을 만들지 않고 요약에 센다.
    예전에는 contact_target_kind=object, target_id=person_01로 기록했다.

    정답: 버킷 대상 접촉 수만큼 `person_contacts`로 빠지고, 나머지 접촉은 그대로이며, 남은 손 상태는
    모두 온톨로지 검증을 통과한다.
    """
    actions = generate_action_scenario(1, n_units=10)
    _write_bodycam(tmp_path / "bodycam.mp4", actions)
    write_parquet(
        tmp_path / "glove_right.parquet",
        {"t_ms": actions.glove_t_ms, "pressure_0": actions.glove_pressure},
    )
    sid = f"psn-{uuid.uuid4().hex[:8]}"
    manifest = {
        "session_id": sid, "domain": "cleaning", "worker_id": "w01", "site_id": "site01",
        "consent_version": "c1", "recorded_at": FIXED_TIME.isoformat(), "ontology_version": "1.0.0",
        "streams": [
            {"stream_id": "bodycam", "kind": "bodycam", "path": "bodycam.mp4"},
            {"stream_id": "glove_right", "kind": "glove_right", "path": "glove_right.parquet"},
        ],
    }  # fmt: skip
    (tmp_path / "m.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    raw = S3Store.from_env("dlp-raw")
    with pg.begin() as conn:
        session = ingest_session(*load_manifest(tmp_path / "m.yaml"), raw, conn).session
        glove = session.stream("glove_right").model_copy(
            update={"sync_method": SyncMethod.TAP_EVENT}
        )
        update_stream_sync(conn, sid, glove)
    # 버킷 자리에 사람 박스를 둔다 (서랍·세면대는 그대로)
    renamed = {"bucket_01": ("person_01", "person")}
    boxes = [
        make_label(
            BoxTrackPayload(
                entity_id=renamed.get(eid, (eid, cls))[0],
                class_id=renamed.get(eid, (eid, cls))[1],
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
    with pg.begin() as conn:
        summary = run_prelabel(conn, sid, raw, predictors, policy, ontology, FIXED_TIME)
        contacts = _builtin_contacts(current_labels(get_labels(conn, sid, kinds=["hand_state"])))
    truth = [
        x.payload
        for x in actions.labels
        if isinstance(x.payload, HandStatePayload) and x.payload.contact_target_kind != "none"
    ]
    on_person = sum(p.target_id == "bucket_01" for p in truth)
    assert on_person > 0  # 시나리오에 버킷(지금은 사람) 접촉이 있어야 회귀를 본다
    for x in contacts:
        assert isinstance(x.payload, HandStatePayload)
        assert x.payload.target_id != "person_01"
        assert check_label(x, ontology) == []
    assert summary.contacts == len(contacts) == len(truth) - on_person
    assert summary.person_contacts == on_person
    # 재실행은 같은 버전이라 아무것도 하지 않는다 (사람 접촉도 다시 세지 않는다)
    with pg.begin() as conn:
        again = run_prelabel(conn, sid, raw, predictors, policy, ontology, FIXED_TIME)
    assert again.contacts == 0 and again.person_contacts == 0
