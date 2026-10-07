"""DB 계층 테스트: 마이그레이션과 테이블 정의 일치(SQLite), 저장소 함수와 트리거(PostgreSQL).

`@pytest.mark.services` 테스트는 실행 중인 PostgreSQL이 필요하다
(`make up` 뒤 `make test-services`).
접속 URL은 환경 변수 DLP_DATABASE_URL 또는 개발 기본값. 테스트마다 임시 DB를 만들고 지운다.
정답 근거: 계약 객체 왕복 동일성, 트리거가 내는 예외, 마이그레이션 백필 규칙.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy.exc import DBAPIError

from dlp_schema.dataset import DatasetVersion, Split
from dlp_schema.db.migrate import downgrade, upgrade
from dlp_schema.db.repository import (
    TransitionError,
    get_dataset_version,
    get_labels,
    get_session,
    insert_dataset_version,
    insert_labels,
    insert_session,
    list_lifecycle_events,
    record_review,
    register_ontology,
    set_lifecycle,
    set_model_status,
    update_stream_sync,
)
from dlp_schema.db.tables import metadata, ontology_versions
from dlp_schema.labels import Provenance, Source, VerificationState
from dlp_schema.lineage import ModelStatus
from dlp_schema.ontology import Ontology
from dlp_schema.session import LifecycleState, SyncMethod
from dlp_schema.testing import FIXED_TIME, action_payload, make_label, make_session

# 개발 compose(services/)의 PostgreSQL 기본 접속 정보 (로컬 개발 전용 비밀번호)
DEFAULT_URL = "postgresql+psycopg://dlp:dlp-dev-password@localhost:5432/dlp"


def test_migrations_match_table_definitions(tmp_path: Path) -> None:
    """SQLite에 모든 마이그레이션을 적용한 스키마가 db.tables 메타데이터와 같다.

    차이가 있으면 diff 목록이 실패 메시지에 나온다. 끝으로 base까지 내릴 수 있는지도 본다.
    """
    url = f"sqlite:///{tmp_path / 'm.db'}"
    upgrade(url)
    engine = sa.create_engine(url)
    with engine.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn), metadata)
    engine.dispose()
    assert diff == [], diff
    downgrade(url, "base")


@pytest.fixture
def pg_url() -> Iterator[str]:
    """테스트마다 새 데이터베이스를 만들고 마이그레이션을 적용한다.

    관리 DB(postgres)에 AUTOCOMMIT으로 붙어 `dlp_test_<임의 8자>` DB를 만들고, 끝나면 강제로 지운다.
    """
    admin_url = sa.make_url(os.environ.get("DLP_DATABASE_URL", DEFAULT_URL))
    name = f"dlp_test_{uuid.uuid4().hex[:8]}"
    admin = sa.create_engine(admin_url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(sa.text(f'CREATE DATABASE "{name}"'))
    url = admin_url.set(database=name).render_as_string(hide_password=False)
    upgrade(url)
    try:
        yield url
    finally:
        with admin.connect() as conn:
            conn.execute(sa.text(f'DROP DATABASE "{name}" WITH (FORCE)'))
        admin.dispose()


@pytest.fixture
def pg(pg_url: str, ontology: Ontology) -> Iterator[sa.Engine]:
    """마이그레이션된 임시 DB 엔진. 온톨로지 v1과 기본 세션 s001을 미리 등록한다."""
    engine = sa.create_engine(pg_url)
    with engine.begin() as conn:
        register_ontology(conn, ontology)
        insert_session(conn, make_session())
    yield engine
    engine.dispose()


@pytest.mark.services
def test_session_and_label_roundtrip(pg: sa.Engine) -> None:
    """세션과 라벨(모델 행동, 공백, 블러)이 DB 왕복 후 같다.

    get_labels는 (시작, ID) 순서와 종류 필터를 지킨다.
    """
    model = Provenance(source=Source.MODEL, model_version="vlm-0.1")
    labels = [
        make_label(action_payload(), label_id="a1", provenance=model, confidence=0.7),
        make_label({"kind": "gap", "hand": "right", "gap_type": "idle"}, label_id="g1",
                   t_start_ms=1_000, t_end_ms=2_000),
        make_label({"kind": "blur_track", "target": "reflection",
                    "keyframes": [{"t_ms": 0, "x": 1, "y": 1, "w": 5, "h": 5}]},
                   label_id="b1", stream_id="bodycam"),
    ]  # fmt: skip
    with pg.begin() as conn:
        assert insert_labels(conn, labels) == 3
    with pg.connect() as conn:
        assert get_session(conn, "s001") == make_session()
        assert get_labels(conn, "s001") == sorted(labels, key=lambda x: (x.t_start_ms, x.label_id))
        assert [x.label_id for x in get_labels(conn, "s001", kinds=["gap"])] == ["g1"]


@pytest.mark.services
def test_stream_order_is_preserved(pg: sa.Engine) -> None:
    """스트림 순서(imu, bodycam)가 DB 왕복 후에도 유지된다 (streams.position)."""
    base = make_session("s002")
    reordered = base.model_copy(update={"streams": tuple(reversed(base.streams))})
    assert [s.stream_id for s in reordered.streams] == ["imu", "bodycam"]
    with pg.begin() as conn:
        insert_session(conn, reordered)
    with pg.connect() as conn:
        assert get_session(conn, "s002") == reordered


@pytest.mark.services
def test_labels_are_immutable_except_review(pg: sa.Engine) -> None:
    """record_review로 검수 상태만 바꿀 수 있고, 다른 열 수정·삭제는 트리거가 막는다."""
    with pg.begin() as conn:
        insert_labels(conn, [make_label(action_payload(), label_id="a1")])
    with pg.begin() as conn:
        record_review(conn, "a1", VerificationState.HUMAN_APPROVED, "rev01", FIXED_TIME)
    with pg.connect() as conn:
        [label] = get_labels(conn, "s001")
    assert label.verification.state is VerificationState.HUMAN_APPROVED
    assert label.verification.reviewer_id == "rev01"

    for statement in (
        "UPDATE label_records SET t_end_ms = 5 WHERE label_id = 'a1'",
        "UPDATE label_records SET payload = '{}'::jsonb WHERE label_id = 'a1'",
        "DELETE FROM label_records WHERE label_id = 'a1'",
        "UPDATE label_records SET measurement = 'blind' WHERE label_id = 'a1'",
        "UPDATE label_records SET seeded_error = true WHERE label_id = 'a1'",
    ):
        with pytest.raises(DBAPIError, match="label_records"), pg.begin() as conn:
            conn.execute(sa.text(statement))


# TRUNCATE를 막아야 하는 추가 전용 테이블 (0010, 0011)
APPEND_ONLY_TABLES = (
    "label_records", "raw_access_log", "review_work", "privacy_audits", "retention_decisions",
    "session_lifecycle_events",
)  # fmt: skip


@pytest.mark.services
@pytest.mark.parametrize("table", APPEND_ONLY_TABLES)
def test_append_only_tables_cannot_be_truncated(pg: sa.Engine, table: str) -> None:
    """행 트리거를 거치지 않는 TRUNCATE도 막는다 (CASCADE로 딸려 비우는 경우 포함)."""
    with pytest.raises(DBAPIError, match="TRUNCATE"), pg.begin() as conn:
        conn.execute(sa.text(f"TRUNCATE {table}"))
    if table == "label_records":
        with pytest.raises(DBAPIError, match="TRUNCATE"), pg.begin() as conn:
            conn.execute(sa.text("TRUNCATE sessions CASCADE"))


@pytest.mark.services
def test_long_model_version_is_stored(pg: sa.Engine) -> None:
    """정책 해시가 붙은 모델 버전은 128자를 넘을 수 있다 (프라이버시 파이프라인)."""
    version = "privacy-" + "+".join(f"detector{i}-1.0.0+p{'a' * 12}" for i in range(6))
    assert len(version) > 128
    model = Provenance(source=Source.MODEL, model_version=version)
    with pg.begin() as conn:
        insert_labels(conn, [make_label(action_payload(), provenance=model, confidence=0.5)])
    with pg.connect() as conn:
        [label] = get_labels(conn, "s001")
    assert label.provenance.model_version == version


@pytest.mark.services
def test_dataset_version_golden_set_version_fits_golden_set_ids(pg: sa.Engine) -> None:
    """golden_set_version에 128자 골든셋 ID를 저장할 수 있다 (0010에서 64 → 128)."""
    version = DatasetVersion(
        version_id="ds-0002", ontology_version="1.0.0", created_at=FIXED_TIME,
        snapshot_uri="lakefs://dlp/main@abc", golden_set_version="g" * 128, splits={},
    )  # fmt: skip
    with pg.begin() as conn:
        insert_dataset_version(conn, version)
    with pg.connect() as conn:
        assert get_dataset_version(conn, "ds-0002") == version


@pytest.mark.services
def test_set_model_status_rejects_unknown_version(pg: sa.Engine) -> None:
    """없는 모델 버전의 상태를 바꾸면 KeyError."""
    with pytest.raises(KeyError, match="nope"), pg.begin() as conn:
        set_model_status(conn, "nope", ModelStatus.DEPLOYED, FIXED_TIME)


@pytest.mark.services
def test_update_stream_sync_keeps_reference_clock(pg: sa.Engine) -> None:
    """기준 스트림 시계 변경과 다른 스트림의 reference 지정은 거부한다.

    같은 값을 다시 쓰는 것은 허용한다.
    """
    session = make_session()
    moved = session.reference_stream.model_copy(update={"offset_ms": 10.0})
    with pytest.raises(ValueError, match="기준 스트림"), pg.begin() as conn:
        update_stream_sync(conn, "s001", moved)
    as_reference = session.stream("imu").model_copy(update={"sync_method": SyncMethod.REFERENCE})
    with pytest.raises(ValueError, match="기준 스트림"), pg.begin() as conn:
        update_stream_sync(conn, "s001", as_reference)
    with pg.begin() as conn:  # 그대로 다시 쓰는 것은 허용
        update_stream_sync(conn, "s001", session.reference_stream)
    with pg.connect() as conn:
        assert get_session(conn, "s001") == session


@pytest.mark.services
def test_lifecycle_transitions_are_enforced(pg: sa.Engine) -> None:
    """건너뛰는 전이(privacy_approved → exported)는 TransitionError, withdrawn은 언제나 허용."""
    with pg.begin() as conn:
        set_lifecycle(conn, "s001", LifecycleState.PRIVACY_APPROVED)
        with pytest.raises(TransitionError):
            set_lifecycle(conn, "s001", LifecycleState.EXPORTED)
        set_lifecycle(conn, "s001", LifecycleState.WITHDRAWN)
        assert get_session(conn, "s001").lifecycle_state is LifecycleState.WITHDRAWN


@pytest.mark.services
def test_lifecycle_transitions_are_recorded(pg: sa.Engine) -> None:
    """전이마다 기록 1행: 등록·전진은 남고 멱등 호출은 남지 않으며, 시각·actor가 그대로 저장된다.
    시간대 없는 시각은 거부, 기록 테이블은 수정·삭제를 트리거가 막는다.
    """
    later = FIXED_TIME + timedelta(hours=1)
    with pg.begin() as conn:
        set_lifecycle(conn, "s001", LifecycleState.PRIVACY_APPROVED, at=later, actor="rev01")
        set_lifecycle(conn, "s001", LifecycleState.PRIVACY_APPROVED, at=later)  # 멱등: 기록 없음
        set_lifecycle(conn, "s001", LifecycleState.PRELABELED)  # 시각을 안 주면 DB 시각
        with pytest.raises(TransitionError):
            set_lifecycle(conn, "s001", LifecycleState.EXPORTED, at=later)
        with pytest.raises(ValueError, match="시간대"):
            set_lifecycle(conn, "s001", LifecycleState.HUMAN_VERIFIED, at=datetime(2026, 1, 1))
    with pg.connect() as conn:
        events = list_lifecycle_events(conn, "s001")
    assert [(e.from_state, e.to_state) for e in events] == [
        (None, LifecycleState.RAW_INGESTED),  # 세션 등록
        (LifecycleState.RAW_INGESTED, LifecycleState.PRIVACY_APPROVED),
        (LifecycleState.PRIVACY_APPROVED, LifecycleState.PRELABELED),
    ]
    assert events[1].at == later and events[1].actor == "rev01"
    assert all(e.at.utcoffset() is not None for e in events)
    # 추가만 한다
    for statement in (
        "UPDATE session_lifecycle_events SET actor = 'x'",
        "DELETE FROM session_lifecycle_events",
    ):
        with pytest.raises(DBAPIError, match="session_lifecycle_events"), pg.begin() as conn:
            conn.execute(sa.text(statement))


@pytest.mark.services
def test_lifecycle_events_are_backfilled_by_migration(pg_url: str) -> None:
    """0011 이전에 등록된 세션은 지금 상태로 한 번 기록된다 (시각 = 세션 등록 시각)."""
    downgrade(pg_url, "0010")
    engine = sa.create_engine(pg_url)
    try:
        with engine.begin() as conn:
            register_ontology_row = "INSERT INTO ontology_versions (version, status, content) "
            conn.execute(sa.text(register_ontology_row + "VALUES ('1.0.0', 'draft', '{}')"))
            conn.execute(
                sa.text(
                    "INSERT INTO sessions (session_id, domain, worker_id, site_id, consent_version,"
                    " recorded_at, duration_ms, calibration, privacy_state, lifecycle_state,"
                    " created_at) VALUES ('old1', 'cleaning', 'w1', 'p1', 'c1', :t, 1000, '{}',"
                    " 'approved', 'prelabeled', :t)"
                ),
                {"t": FIXED_TIME},
            )
        upgrade(pg_url)
        with engine.connect() as conn:
            [event] = list_lifecycle_events(conn, "old1")
        assert (event.from_state, event.to_state) == (None, LifecycleState.PRELABELED)
        assert event.at == FIXED_TIME and event.actor == "migration:0011"
    finally:
        engine.dispose()


@pytest.mark.services
def test_dataset_version_roundtrip(pg: sa.Engine) -> None:
    """데이터셋 버전과 분할이 DB 왕복 후 같다."""
    version = DatasetVersion(
        version_id="ds-0001", ontology_version="1.0.0", created_at=FIXED_TIME,
        snapshot_uri="lakefs://dlp/main@abc", splits={"s001": Split.GOLDEN},
    )  # fmt: skip
    with pg.begin() as conn:
        insert_dataset_version(conn, version)
    with pg.connect() as conn:
        assert get_dataset_version(conn, "ds-0001") == version


@pytest.mark.services
def test_registering_changed_ontology_under_same_version_fails(
    pg: sa.Engine, ontology: Ontology
) -> None:
    """같은 버전에 상태(draft → frozen)만 바꿔 등록해도 '다른 내용' 오류다.

    상태 변경은 덧붙이기가 아니다.
    """
    changed = ontology.model_copy(update={"status": "frozen"})
    with pytest.raises(ValueError, match="다른 내용"), pg.begin() as conn:
        register_ontology(conn, changed)


@pytest.mark.services
def test_draft_ontology_accepts_additive_update(pg_url: str, ontology: Ontology) -> None:
    """3차 검수 전 내용(hand_joints·surface_parts 없음)으로 등록된 DB에 현재 v1을 다시 등록한다."""
    old = ontology.model_copy(update={"hand_joints": {}, "surface_parts": {}})
    engine = sa.create_engine(pg_url)
    try:
        with engine.begin() as conn:
            register_ontology(conn, old)
        with engine.begin() as conn:
            register_ontology(conn, ontology)  # 덧붙이기만: 내용을 바꾼다
            register_ontology(conn, ontology)  # 같은 내용: 그대로
        with engine.connect() as conn:
            stored = conn.execute(
                sa.select(ontology_versions.c.content).where(
                    ontology_versions.c.version == ontology.version
                )
            ).scalar_one()
        assert stored == ontology.model_dump(mode="json")
        # 키를 지우는 변경(예전 내용으로 되돌리기 포함)은 덧붙이기가 아니다
        with pytest.raises(ValueError, match="덧붙이기가 아닌"), engine.begin() as conn:
            register_ontology(conn, old)
        renamed = dict(ontology.verbs)
        first = next(iter(renamed))
        renamed[first] = renamed[first].model_copy(update={"ko": "바뀐 이름"})
        with pytest.raises(ValueError, match="덧붙이기가 아닌"), engine.begin() as conn:
            register_ontology(conn, ontology.model_copy(update={"verbs": renamed}))
        # 확정된 버전은 덧붙이기도 받지 않는다
        with engine.begin() as conn:
            conn.execute(ontology_versions.update().values(status="frozen"))
        extra = {**ontology.hand_joints, "palm_center": next(iter(ontology.hand_joints.values()))}
        with pytest.raises(ValueError, match="확정된"), engine.begin() as conn:
            register_ontology(conn, ontology.model_copy(update={"hand_joints": extra}))
    finally:
        engine.dispose()


@pytest.mark.services
def test_update_stream_sync_changes_only_sync_fields(pg: sa.Engine) -> None:
    """IMU의 동기화 다섯 필드가 저장되고, 없는 스트림이면 KeyError."""
    imu = make_session().stream("imu")
    synced = imu.model_copy(
        update={"offset_ms": 12.5, "clock_scale": 1.00005, "sync_method": SyncMethod.TAP_EVENT,
                "sync_confidence": 0.9, "manual_adjustment_ms": -2.0}
    )  # fmt: skip
    with pg.begin() as conn:
        update_stream_sync(conn, "s001", synced)
    with pg.connect() as conn:
        assert get_session(conn, "s001").stream("imu") == synced
    with pytest.raises(KeyError), pg.begin() as conn:
        update_stream_sync(conn, "s001", synced.model_copy(update={"stream_id": "nope"}))
