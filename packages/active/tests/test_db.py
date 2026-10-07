"""DB(make up)의 세션에서 순위를 매기고, 고른 세션의 블러본으로 FiftyOne 샘플을 만든다."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import sqlalchemy as sa

from dlp_active.curation import build_samples
from dlp_active.policy import load_policy
from dlp_active.select import rank_sessions
from dlp_datasets.lineage import withdraw_session
from dlp_fixtures.video import write_video
from dlp_media.storage import LocalStore, sha256_file
from dlp_schema.db.migrate import upgrade
from dlp_schema.db.repository import (
    insert_golden_set,
    insert_labels,
    insert_session,
    register_ontology,
)
from dlp_schema.labels import LabelRecord, Provenance, Source, Verification, VerificationState
from dlp_schema.lineage import GoldenSet
from dlp_schema.ontology import load_ontology
from dlp_schema.session import Domain, LifecycleState, PrivacyState
from dlp_schema.testing import FIXED_TIME, make_label, make_session

pytestmark = pytest.mark.services
ROOT = Path(__file__).resolve().parents[3]


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


def box(cls: str) -> dict[str, Any]:
    return {"kind": "box_track", "entity_id": f"{cls}_1", "class_id": cls,
            "keyframes": [{"t_ms": 0, "x": 8, "y": 8, "w": 16, "h": 16}]}  # fmt: skip


def lab(sid: str, lid: str, cls: str, *, model: bool = True, **kw: Any) -> LabelRecord:
    return make_label(
        box(cls),
        label_id=f"{sid}-{lid}",
        session_id=sid,
        stream_id="bodycam",
        provenance=Provenance(source=Source.MODEL, model_version="m1")
        if model
        else Provenance(source=Source.HUMAN),
        confidence=0.8 if model else None,
        **kw,
    )


def test_rank_candidates_and_build_fiftyone_samples(engine: sa.Engine, tmp_path: Path) -> None:
    policy = load_policy(ROOT)
    done = Verification(
        state=VerificationState.HUMAN_CORRECTED, reviewer_id="r", reviewed_at=FIXED_TIME
    )
    ok = Verification(
        state=VerificationState.HUMAN_APPROVED, reviewer_id="r", reviewed_at=FIXED_TIME
    )
    with engine.begin() as conn:
        register_ontology(conn, load_ontology(ROOT / "config/ontology/v1"))

        def add(sid: str, state: LifecycleState, labels: list[LabelRecord], **kw: Any) -> None:
            insert_session(
                conn,
                make_session(sid, worker_id=f"w-{sid}", site_id=f"s-{sid}", **kw).model_copy(
                    update={"lifecycle_state": state, "privacy_state": PrivacyState.APPROVED}
                ),
            )
            insert_labels(conn, labels)

        # 검수가 끝난 세션: mop은 자주 고치고(2/2), cup은 그대로 승인(0/2)
        add("rev", LifecycleState.HUMAN_VERIFIED, [
            lab("rev", "m1", "mop"), lab("rev", "h1", "mop", model=False, verification=done,
                                         parent_label_id="rev-m1"),
            lab("rev", "m2", "mop"), lab("rev", "h2", "mop", model=False, verification=done,
                                         parent_label_id="rev-m2"),
            lab("rev", "c1", "cup", verification=ok), lab("rev", "c2", "cup", verification=ok),
        ])  # fmt: skip
        add("mops", LifecycleState.PRELABELED, [lab("mops", f"{i}", "mop") for i in range(2)])
        # 검수 수가 적어 평활이 크다: mop 0.545, cup 0.455 → 예상 수정 mops 1.09 > cups 0.91
        add("cups", LifecycleState.PRELABELED, [lab("cups", f"{i}", "cup") for i in range(2)])
        # 골든 세션: 사람이 처음부터 만든 cup 정답 (수정률에 "추가"로 세면 안 된다)
        add(
            "gold",
            LifecycleState.PRELABELED,
            [lab("gold", f"{i}", "mop") for i in range(9)]
            + [lab("gold", f"t{i}", "cup", model=False, verification=done) for i in range(20)],
        )
        add("gone", LifecycleState.PRELABELED, [lab("gone", f"{i}", "mop") for i in range(9)])
        add("raw", LifecycleState.PRIVACY_APPROVED, [])
        insert_golden_set(
            conn,
            GoldenSet(
                version="g1", domain=Domain.CLEANING, session_ids=("gold",), created_at=FIXED_TIME
            ),
        )
        withdraw_session(conn, "gone", "동의 철회", FIXED_TIME + timedelta(days=1))

    with engine.connect() as conn:
        ranked, rates = rank_sessions(conn, policy)
    assert [s.session_id for s in ranked] == ["mops", "cups"]  # 골든·사용 중지·검수 전 단계 제외
    assert rates.classes["box_track/mop"].changed == 2
    assert (rates.classes["box_track/cup"].reviewed, rates.classes["box_track/cup"].changed) == (
        2,
        0,
    )
    assert ranked[0].top_classes[0][0] == "box_track/mop"

    # 블러본은 라벨링 버킷에만 있다 (cups는 아직 렌더 전)
    labeling = LocalStore(tmp_path / "store", "dlp-labeling")
    video = tmp_path / "mops.mp4"
    write_video(
        video,
        ((t, np.zeros((32, 48, 3), np.uint8)) for t in (0, 33, 66)),
        width=48,
        height=32,
    )
    labeling.put_file("sessions/mops/blurred/bodycam.mp4", video, sha256_file(video))
    with engine.connect() as conn:
        samples, notes = build_samples(conn, labeling, tmp_path / "cache", ranked, policy)
    assert [s.session_id for s in samples] == ["mops"]
    assert notes == ["cups/bodycam: 블러본이 없어 건너뜀"]
    s = samples[0]
    assert s.filepath == tmp_path / "cache" / "mops" / "bodycam.mp4" and s.filepath.exists()
    assert s.fields["active_rank"] == 1
    assert [d.label for d in s.frames[1]] == ["mop", "mop"]
    assert s.frames[1][0].box == (8 / 48, 8 / 32, 16 / 48, 16 / 32)
