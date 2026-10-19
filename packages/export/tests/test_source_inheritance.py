"""물려받은 3D의 기본 내보내기 경로를 DB·스냅샷을 거쳐 검사한다 (ADR 0048·0050, F-COE-55).

SQLite(`dlp_schema.testing.sqlite_db`)와 `LocalSnapshotStore`로 서비스 없이 돈다. 2D 손 트랙 승인 →
궤적 따르기(물려받기 기록) → 데이터셋 빌드 → `load_source`가 지금 DB의 기록으로 3D를 고르고, 2D를
고친 뒤에는 새 스냅샷에서 끊긴 3D를 빼는지 본다. 다른 세션의 같은 라벨 ID 기록을 섞지 않는지도 본다.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

from dlp_datasets.build import build_dataset_version
from dlp_datasets.policy import load_policy as load_dataset_policy
from dlp_datasets.snapshot import LocalSnapshotStore
from dlp_export.policy import load_policy as load_export_policy
from dlp_export.runner import trajectory_manifest
from dlp_export.source import ExportSource, load_source
from dlp_schema import load_ontology
from dlp_schema.db.repository import (
    insert_labels,
    insert_session,
    record_review,
    register_ontology,
)
from dlp_schema.db.trajectory_repository import apply_trajectory_follow
from dlp_schema.labels import Provenance, Source, Verification, VerificationState
from dlp_schema.review import TrajectoryInheritanceMechanism
from dlp_schema.session import PrivacyState
from dlp_schema.testing import (
    FIXED_TIME,
    hand_track_payload,
    make_label,
    make_session,
    sqlite_db,
    trajectory_payload,
)

from .conftest import ROOT

LATER = FIXED_TIME + timedelta(hours=1)


def _model(sid: str) -> dict[str, Any]:
    """모델 출처 바디캠 라벨 공통 값."""
    return {
        "provenance": Provenance(source=Source.MODEL, model_version="m-1"),
        "confidence": 0.8,
        "stream_id": "bodycam",
        "session_id": sid,
    }


def _ids(src: ExportSource, sid: str) -> set[str]:
    """그 세션에서 고른 라벨 ID."""
    es = next(e for e in src.sessions if e.session.session_id == sid)
    return {x.label_id for x in es.labels}


def test_load_source_uses_current_inheritance_records_per_session(tmp_path: Path) -> None:
    """2D 승인에서 물려받은 3D는 기본 정책으로 들어가고 manifest 근거가 inherited_from_2d다. 2D를
    고치면 새 스냅샷은 (따르기 전에도, 미검수 포함 옵션이어도) 끊긴 3D를 빼고, 옛 스냅샷은 옛 2D와
    함께 그대로 둔다. 다른 세션(s002)의 같은 모양 라벨은 그 세션 기록으로만 판정한다."""
    engine = sqlite_db()
    snaps = LocalSnapshotStore(tmp_path / "snap")
    epol = load_export_policy(ROOT)
    dpol = load_dataset_policy(ROOT)
    try:
        with engine.begin() as conn:
            register_ontology(conn, load_ontology(ROOT / "config" / "ontology" / "v1"))
            for sid in ("s001", "s002"):
                session = make_session(sid).model_copy(
                    update={"privacy_state": PrivacyState.APPROVED}
                )
                insert_session(conn, session)
                insert_labels(
                    conn,
                    [
                        make_label(hand_track_payload(), label_id=f"{sid}-g1", **_model(sid)),
                        make_label(
                            trajectory_payload(derived_from=(f"{sid}-g1",)),
                            label_id=f"{sid}-c1",
                            **_model(sid),
                        ),
                    ],
                )
            # s001만 2D를 승인하고 따르기를 적용한다 (s002는 기록 없음)
            record_review(conn, "s001-g1", VerificationState.HUMAN_APPROVED, "rev01", FIXED_TIME)
            apply_trajectory_follow(
                conn, "s001", FIXED_TIME, TrajectoryInheritanceMechanism.COLLECT
            )
            build_dataset_version(
                conn, snaps, dpol, version_id="dv1", ontology_version="1.0.0",
                golden_set_version=None, now=FIXED_TIME,
            )  # fmt: skip
        with engine.connect() as conn:
            src = load_source(conn, snaps, "dv1", epol, include_unreviewed=False)
        assert {"s001-g1", "s001-c1"} <= _ids(src, "s001")
        assert "s002-c1" not in _ids(src, "s002")  # 기록 없는 세션: 물려받지 않음
        es = next(e for e in src.sessions if e.session.session_id == "s001")
        manifest = trajectory_manifest(es.labels, {"s001": es.trajectory_inheritance})
        assert manifest["review_basis"] == {"inherited_from_2d": 1}

        fixed = make_label(
            hand_track_payload(times=(0, 500, 1_000)),
            label_id="s001-g2",
            stream_id="bodycam",
            session_id="s001",
            parent_label_id="s001-g1",
            verification=Verification(
                state=VerificationState.HUMAN_CORRECTED, reviewer_id="lead", reviewed_at=LATER
            ),
        )
        with engine.begin() as conn:
            insert_labels(conn, [fixed])
            build_dataset_version(
                conn, snaps, dpol, version_id="dv2", ontology_version="1.0.0",
                golden_set_version=None, parent_version_id="dv1", now=LATER,
            )  # fmt: skip
        with engine.connect() as conn:
            for include in (False, True):
                new = load_source(conn, snaps, "dv2", epol, include_unreviewed=include)
                assert "s001-c1" not in _ids(new, "s001")
            old = load_source(conn, snaps, "dv1", epol, include_unreviewed=False)
            assert "s001-c1" in _ids(old, "s001")  # 옛 스냅샷은 옛 2D(g1)와 함께 (설계)
        with engine.begin() as conn:
            follow = apply_trajectory_follow(
                conn, "s001", LATER, TrajectoryInheritanceMechanism.VERIFY
            )
        assert [x.label_id for x in follow.retract] == ["s001-c1"]
        with engine.connect() as conn:
            new = load_source(conn, snaps, "dv2", epol, include_unreviewed=True)
        assert "s001-c1" not in _ids(new, "s001")
    finally:
        engine.dispose()
