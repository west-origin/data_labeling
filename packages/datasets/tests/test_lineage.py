"""데이터셋 빌드·사용 중지·계보 통합 테스트 (PostgreSQL·lakeFS, `make up`, WP7).

합성 세션 300개(`generate_sessions(300, seed=5)`)를 프라이버시 승인·사람 검증 완료로 넣고 세션마다
행동 라벨 하나(`<세션>-a`)를 둔다. 정답 근거: 라벨 수(300), 분할 격리, 사용 중지 전후의
버전·내보내기 차이.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

import pytest
import sqlalchemy as sa

from dlp_datasets.build import DatasetBuildError, build_dataset_version, propose_golden_set
from dlp_datasets.lineage import (
    record_export,
    register_training_run,
    session_lineage,
    withdraw_session,
)
from dlp_datasets.policy import DatasetPolicy, load_policy
from dlp_datasets.snapshot import LakeFSSnapshotStore
from dlp_datasets.splitter import check_isolation, propose_golden
from dlp_fixtures.sessions import generate_sessions
from dlp_schema.dataset import Split
from dlp_schema.db.migrate import upgrade
from dlp_schema.db.repository import (
    get_dataset_version,
    get_session,
    insert_golden_set,
    insert_labels,
    insert_session,
    insert_withdrawal,
    list_lifecycle_events,
    register_ontology,
    set_privacy_state,
)
from dlp_schema.labels import LabelRecord
from dlp_schema.lineage import GoldenSet, TrainingRun, Withdrawal
from dlp_schema.ontology import load_ontology
from dlp_schema.session import Domain, LifecycleState, PrivacyState
from dlp_schema.testing import FIXED_TIME, action_payload, make_label

pytestmark = pytest.mark.services
ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def pg() -> Iterator[sa.Engine]:
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
    engine = sa.create_engine(test_url)
    try:
        yield engine
    finally:
        engine.dispose()
        with admin.connect() as conn:
            conn.execute(sa.text(f'DROP DATABASE "{name}" WITH (FORCE)'))
        admin.dispose()


@pytest.fixture(scope="module")
def policy() -> DatasetPolicy:
    """저장소의 config/policies/dataset.yaml."""
    return load_policy(ROOT)


@pytest.fixture
def snapshots(policy: DatasetPolicy) -> LakeFSSnapshotStore:
    """개발 compose의 lakeFS (환경 변수 또는 기본값)."""
    lp = policy.lakefs
    return LakeFSSnapshotStore.from_env(
        repository=lp.repository, branch=lp.branch, storage_namespace=lp.storage_namespace
    )


def _populate(pg: sa.Engine) -> list[str]:
    """합성 세션 300개 (프라이버시 승인, 사람 검증 완료)와 세션마다 라벨 하나."""
    sessions = generate_sessions(300, seed=5)
    with pg.begin() as conn:
        register_ontology(conn, load_ontology(ROOT / "config" / "ontology" / "v1"))
        for s in sessions:
            insert_session(
                conn,
                s.model_copy(
                    update={
                        "privacy_state": PrivacyState.APPROVED,
                        "lifecycle_state": LifecycleState.HUMAN_VERIFIED,
                    }
                ),
            )
            insert_labels(
                conn,
                [
                    make_label(
                        action_payload(), label_id=f"{s.session_id}-a", session_id=s.session_id
                    )
                ],
            )
        golden = propose_golden(sessions, "cleaning", 20, seed=5)
        insert_golden_set(
            conn,
            GoldenSet(
                version="golden-cleaning-v1",
                domain=Domain.CLEANING,
                session_ids=tuple(golden),
                created_at=FIXED_TIME,
            ),
        )
    return golden


def test_build_withdraw_rebuild_export_and_trace(
    pg: sa.Engine, policy: DatasetPolicy, snapshots: LakeFSSnapshotStore, tmp_path: Path
) -> None:
    """빌드 → 학습·내보내기 기록 → 사용 중지 → 재빌드·재내보내기 → 계보 (WP7 완료 기준).

    정답: v1은 격리 위반 0·골든 그대로·lakeFS URI·라벨 300개. 사용 중지한 train 세션의 계보는
    v1·run-1·exp-1. 골든·holdout 세션 계보에는 학습 실행이 없다. v2와 이후 내보내기에는 그 세션이
    없고(v2 excluded_sessions에 기록), 옛 v1은 그대로 남는다.
    """
    golden = _populate(pg)
    v1_id = f"ds-{uuid.uuid4().hex[:6]}-v1"
    with pg.begin() as conn:
        result = build_dataset_version(
            conn,
            snapshots,
            policy,
            version_id=v1_id,
            ontology_version="1.0.0",
            golden_set_version="golden-cleaning-v1",
            now=FIXED_TIME,
        )
        sessions = [get_session(conn, sid) for sid in result.version.splits]
    v1 = result.version
    assert check_isolation(sessions, dict(v1.splits)) == []  # 완료 기준: 교집합 0
    assert all(v1.splits[g] is Split.GOLDEN for g in golden)
    assert v1.snapshot_uri.startswith("lakefs://dlp-datasets/")
    snap = tmp_path / "manifest.json"
    snapshots.read(v1.snapshot_uri, "manifest.json", snap)
    manifest = json.loads(snap.read_text(encoding="utf-8"))
    assert manifest["split_counts"] == result.report.counts
    assert sum(result.label_counts.values()) == 300

    victim = next(sid for sid, sp in v1.splits.items() if sp is Split.TRAIN)
    with pg.begin() as conn:
        assert get_session(conn, victim).lifecycle_state is LifecycleState.SPLIT_ASSIGNED
        register_training_run(
            conn,
            TrainingRun(
                run_id="run-1",
                dataset_version_id=v1_id,
                model_name="blur",
                model_version="b1",
                created_at=FIXED_TIME,
            ),
        )
        e1 = record_export(
            conn,
            export_id="exp-1",
            dataset_version_id=v1_id,
            target="buyer-a",
            format="lerobot",
            uri="s3://dlp-datasets/exports/1",
            now=FIXED_TIME,
        )
        assert victim in e1.session_ids

        lineage = withdraw_session(
            conn, victim, "동의 철회", FIXED_TIME + timedelta(days=1), actor="dpo01"
        )
        events = list_lifecycle_events(conn, victim)
    # 감사 회귀: 빌드와 사용 중지의 생애주기 기록에 시각·실행자가 남는다
    # (예전에는 둘 다 빠져 DB 시각·실행자 없음으로 기록됐다)
    assert [(e.from_state, e.to_state, e.at, e.actor) for e in events[-2:]] == [
        (
            LifecycleState.HUMAN_VERIFIED,
            LifecycleState.SPLIT_ASSIGNED,
            FIXED_TIME,
            f"dataset:{v1_id}",
        ),
        (
            LifecycleState.SPLIT_ASSIGNED,
            LifecycleState.WITHDRAWN,
            FIXED_TIME + timedelta(days=1),
            "dpo01",
        ),
    ]
    # 완료 기준: 계보 조회가 세션 → 버전 → 학습 실행 → 내보내기를 모두 돌려준다
    assert lineage.lifecycle is LifecycleState.WITHDRAWN
    assert lineage.dataset_versions == [v1_id]
    assert [r.run_id for r in lineage.training_runs] == ["run-1"]
    assert [e.export_id for e in lineage.exports] == ["exp-1"]
    assert lineage.golden_sets == []

    # 감사 회귀: 골든·holdout 세션의 계보에는 그 버전의 학습 실행이 없고,
    # 골든 세션은 골든셋이 나온다
    holdout = next(sid for sid, sp in v1.splits.items() if sp is Split.HOLDOUT)
    with pg.connect() as conn:
        gl = session_lineage(conn, golden[0])
        hl = session_lineage(conn, holdout)
    assert gl.dataset_versions == [v1_id] and gl.training_runs == []
    assert gl.golden_sets == ["golden-cleaning-v1"]
    assert hl.dataset_versions == [v1_id] and hl.training_runs == [] and hl.golden_sets == []

    v2_id = v1_id.replace("-v1", "-v2")
    with pg.begin() as conn:
        v2 = build_dataset_version(
            conn,
            snapshots,
            policy,
            version_id=v2_id,
            ontology_version="1.0.0",
            golden_set_version="golden-cleaning-v1",
            parent_version_id=v1_id,
            now=FIXED_TIME + timedelta(days=2),
        ).version
        e2 = record_export(
            conn,
            export_id="exp-2",
            dataset_version_id=v1_id,
            target="buyer-b",
            format="coco",
            uri="s3://dlp-datasets/exports/2",
            now=FIXED_TIME,
        )
        stored_v1 = get_dataset_version(conn, v1_id)
    # 완료 기준: 사용 중지 후 새 버전과 내보내기에 그 세션이 없다 (옛 버전은 재현성을 위해 그대로)
    assert victim not in v2.splits and victim in v2.excluded_sessions
    assert victim not in e2.session_ids
    assert victim in stored_v1.splits
    assert v2.snapshot_uri != v1.snapshot_uri


def test_build_refuses_empty_or_wrong_domain(
    pg: sa.Engine, policy: DatasetPolicy, snapshots: LakeFSSnapshotStore
) -> None:
    """빌드 실패 두 경우: 골든셋(cleaning)과 다른 도메인(nursing) 요청, 후보가 없는 온톨로지."""
    _populate(pg)
    with pg.begin() as conn, pytest.raises(DatasetBuildError, match="도메인"):
        build_dataset_version(
            conn,
            snapshots,
            policy,
            version_id="x1",
            ontology_version="1.0.0",
            golden_set_version="golden-cleaning-v1",
            domain="nursing",
            now=FIXED_TIME,
        )
    with pg.begin() as conn, pytest.raises(DatasetBuildError, match="세션이 없습니다"):
        build_dataset_version(
            conn,
            snapshots,
            policy,
            version_id="x2",
            ontology_version="9.9.9",
            golden_set_version=None,
            now=FIXED_TIME,
        )


def test_seeded_errors_and_measurements_never_reach_training(
    pg: sa.Engine, policy: DatasetPolicy, snapshots: LakeFSSnapshotStore, tmp_path: Path
) -> None:
    """완료 기준: 오류 삽입 레코드와 그 후손, 측정용 레코드가 학습 분할에 0건."""
    _populate(pg)
    with pg.begin() as conn:
        sids = [s.session_id for s in generate_sessions(300, seed=5)]
        extra: list[LabelRecord] = []
        for sid in sids:
            seeded = make_label(
                action_payload(), label_id=f"seed-x-{sid}", session_id=sid, seeded_error=True
            )
            fix = make_label(
                action_payload(), label_id=f"fix-{sid}", session_id=sid,
                parent_label_id=seeded.label_id,
            )  # fmt: skip
            blind = make_label(
                action_payload(), label_id=f"blind-{sid}", session_id=sid, measurement="blind"
            )
            extra += [seeded, fix, blind]
        insert_labels(conn, extra)
        result = build_dataset_version(
            conn, snapshots, policy, version_id=f"ds-{uuid.uuid4().hex[:6]}-seed",
            ontology_version="1.0.0", golden_set_version="golden-cleaning-v1", now=FIXED_TIME,
        )  # fmt: skip
    path = tmp_path / "labels.jsonl"
    snapshots.read(result.version.snapshot_uri, "labels.jsonl", path)
    ids = [json.loads(line)["label_id"] for line in path.read_text("utf-8").splitlines()]
    assert len(ids) == 300 and all(i.endswith("-a") for i in ids)
    train = {sid for sid, sp in result.version.splits.items() if sp is Split.TRAIN}
    assert train and not any(i.startswith(("seed-", "fix-", "blind-")) for i in ids)


def test_golden_session_outside_candidates_still_blocks_its_worker_and_site(
    pg: sa.Engine, policy: DatasetPolicy, snapshots: LakeFSSnapshotStore
) -> None:
    """감사 회귀: 프라이버시 미승인으로 후보에서 빠진 골든 세션의 작업자·장소도

    학습·검증에 못 든다.
    """
    golden = _populate(pg)
    with pg.begin() as conn:
        by_id = {g: get_session(conn, g) for g in golden}
        worker = by_id[golden[0]].worker_id
        hidden = [g for g in golden if by_id[g].worker_id == worker]
        sites = {by_id[g].site_id for g in hidden}
        for g in hidden:
            set_privacy_state(conn, g, PrivacyState.PENDING)
        result = build_dataset_version(
            conn, snapshots, policy, version_id=f"ds-{uuid.uuid4().hex[:6]}-g",
            ontology_version="1.0.0", golden_set_version="golden-cleaning-v1", now=FIXED_TIME,
        )  # fmt: skip
        sessions = {sid: get_session(conn, sid) for sid in result.version.splits}
    assert not set(hidden) & set(result.version.splits)
    for sid, sp in result.version.splits.items():
        if sp in (Split.TRAIN, Split.VAL):
            s = sessions[sid]
            assert s.worker_id != worker and s.site_id not in sites, sid


def test_golden_proposal_skips_unapproved_and_withdrawn_sessions(
    pg: sa.Engine, policy: DatasetPolicy
) -> None:
    """골든 제안은 프라이버시 미승인·사용 중지 세션을 고르지 않는다.

    처음 제안의 첫 세션을 미승인으로, 둘째 세션을 사용 중지로 바꾼 뒤 다시 제안해 확인한다.
    """
    sessions = generate_sessions(300, seed=5)
    with pg.begin() as conn:
        register_ontology(conn, load_ontology(ROOT / "config" / "ontology" / "v1"))
        for s in sessions:
            insert_session(conn, s.model_copy(update={"privacy_state": PrivacyState.APPROVED}))
        first = propose_golden_set(
            conn, policy, ontology_version="1.0.0", domain="cleaning", target=20, seed=5
        )
        pending, gone = first[0], first[1]
        set_privacy_state(conn, pending, PrivacyState.PENDING)
        insert_withdrawal(
            conn, Withdrawal(session_id=gone, reason="동의 철회", withdrawn_at=FIXED_TIME)
        )
        second = propose_golden_set(
            conn, policy, ontology_version="1.0.0", domain="cleaning", target=20, seed=5
        )
    assert first and pending not in second and gone not in second


def test_label_history_policy_controls_snapshot(
    pg: sa.Engine, policy: DatasetPolicy, snapshots: LakeFSSnapshotStore, tmp_path: Path
) -> None:
    """include_label_history: 참이면 수정 이력(원본과 수정본)을, 거짓이면 현재 라벨만 넣는다."""
    _populate(pg)
    with pg.begin() as conn:
        sids = [s.session_id for s in generate_sessions(300, seed=5)]
        insert_labels(
            conn,
            [
                make_label(
                    action_payload(),
                    label_id=f"{sid}-fix",
                    session_id=sid,
                    parent_label_id=f"{sid}-a",
                )
                for sid in sids
            ],
        )
    ids: dict[bool, list[str]] = {}
    for keep in (True, False):
        with pg.begin() as conn:
            result = build_dataset_version(
                conn, snapshots, policy.model_copy(update={"include_label_history": keep}),
                version_id=f"ds-{uuid.uuid4().hex[:6]}-h", ontology_version="1.0.0",
                golden_set_version="golden-cleaning-v1", now=FIXED_TIME,
            )  # fmt: skip
        path = tmp_path / f"labels-{keep}.jsonl"
        snapshots.read(result.version.snapshot_uri, "labels.jsonl", path)
        ids[keep] = [json.loads(x)["label_id"] for x in path.read_text("utf-8").splitlines()]
    assert len(ids[True]) == 600
    assert len(ids[False]) == 300 and all(i.endswith("-fix") for i in ids[False])
