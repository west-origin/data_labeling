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
from dlp_privacy.policy import load_policy as load_privacy_policy
from dlp_privacy.runner import RenderMeta, expected_render_hash, write_render_meta
from dlp_schema.db.migrate import upgrade
from dlp_schema.db.repository import (
    insert_golden_set,
    insert_labels,
    insert_session,
    register_ontology,
    set_privacy_state,
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
    """테스트마다 새 PostgreSQL 데이터베이스를 만들고 마이그레이션한다. 끝나면 지운다."""
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
    """클래스 cls의 박스 트랙 페이로드 (t=0 키프레임 하나)."""
    return {"kind": "box_track", "entity_id": f"{cls}_1", "class_id": cls,
            "keyframes": [{"t_ms": 0, "x": 8, "y": 8, "w": 16, "h": 16}]}  # fmt: skip


def lab(sid: str, lid: str, cls: str, *, model: bool = True, **kw: Any) -> LabelRecord:
    """세션 sid의 박스 라벨 "<sid>-<lid>" (model이면 모델 출처·신뢰도 0.8, 아니면 사람)."""
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
    """DB 세션 순위와 FiftyOne 샘플 만들기 (골든·사용 중지·검수 전 단계 제외, 렌더 확인).

    정답: 후보는 prelabeled 중 골든(gold)·사용 중지(gone)를 뺀 mops·cups. 수정률은 rev 세션에서
    mop 2/2, cup 0/2 (골든의 사람 cup 정답은 세지 않는다) → mops가 먼저. 블러본은 렌더 기록이
    있고 지금 승인 상태일 때만 쓴다 (cups는 블러본 없음, 승인을 풀면 건너뜀).
    """
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
            """작업자·장소가 세션마다 다른 승인 세션을 생애주기 state로 넣고 라벨을 넣는다."""
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
    sha = sha256_file(video)
    labeling.put_file("sessions/mops/blurred/bodycam.mp4", video, sha)
    privacy = load_privacy_policy(ROOT)
    with engine.connect() as conn:
        # 렌더 기록이 없는 블러본은 쓰지 않는다 (지금 승인된 블러 라벨로 렌더했는지 모른다)
        samples, notes = build_samples(conn, labeling, tmp_path / "cache", ranked, policy)
        assert samples == [] and "렌더 기록" in notes[0]
        digest = expected_render_hash(conn, "mops", "bodycam", privacy)
    write_render_meta(labeling, "mops", "bodycam", RenderMeta(digest, sha), tmp_path)
    with engine.connect() as conn:
        samples, notes = build_samples(conn, labeling, tmp_path / "cache", ranked, policy)
    assert [s.session_id for s in samples] == ["mops"]
    assert notes == ["cups/bodycam: 블러본이 없어 건너뜀"]
    # 회귀(감사 4-2): 승인이 풀린 세션의 이전 블러본은 큐레이션에 넣지 않는다
    with engine.begin() as conn:
        set_privacy_state(conn, "mops", PrivacyState.AUTO_BLURRED)
        stale, why = build_samples(conn, labeling, tmp_path / "cache2", ranked, policy)
        set_privacy_state(conn, "mops", PrivacyState.APPROVED)
    assert stale == [] and "승인 상태가 아닙니다" in why[0]
    s = samples[0]
    assert s.filepath == tmp_path / "cache" / "mops" / "bodycam.mp4" and s.filepath.exists()
    assert s.fields["active_rank"] == 1
    assert [d.label for d in s.frames[1]] == ["mop", "mop"]
    assert s.frames[1][0].box == (8 / 48, 8 / 32, 16 / 48, 16 / 32)
