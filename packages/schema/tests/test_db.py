"""DB 계층 테스트: 마이그레이션과 테이블 정의 일치(SQLite), 저장소 함수와 트리거(PostgreSQL).

`@pytest.mark.services` 테스트는 실행 중인 PostgreSQL이 필요하다
(`make up` 뒤 `make test-services`).
접속 URL은 환경 변수 DLP_DATABASE_URL 또는 개발 기본값. 테스트마다 임시 DB를 만들고 지운다.
정답 근거: 계약 객체 왕복 동일성, 트리거가 내는 예외, 마이그레이션 백필 규칙.
"""

from __future__ import annotations

import itertools
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy.exc import DBAPIError

from dlp_schema import repo_root
from dlp_schema.dataset import DatasetVersion, Split
from dlp_schema.db.migrate import alembic_config, downgrade, upgrade
from dlp_schema.db.repository import (
    TransitionError,
    active_privacy_hold,
    get_dataset_version,
    get_description_inheritance,
    get_labels,
    get_session,
    insert_dataset_version,
    insert_labels,
    insert_session,
    list_lifecycle_events,
    list_privacy_holds,
    record_privacy_hold,
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


def test_migrations_have_single_head() -> None:
    """마이그레이션 head가 하나다 (같은 리비전을 잇는 두 브랜치를 병합하면 head가 둘이 되어
    `upgrade(url)`이 실패한다. 예: 0018과 0019가 모두 0017을 이을 때, 나중에 병합하는 쪽이
    0019의 down_revision을 0018로 바꾼다 — ADR 0040)."""
    heads = ScriptDirectory.from_config(alembic_config("sqlite://")).get_heads()
    assert len(heads) == 1, f"head가 여럿이다: {heads} (나중 리비전의 down_revision을 고친다)"


# 기준 문서(ADR, 구현 계획)에서 마이그레이션의 down 리비전을 적은 표기. 읽는 값은 (리비전, down).
# - 괄호·쉼표: "Alembic 0019(down 0018", "`0019_ai_review.py`(down 0018", "Alembic 0018, down 0017",
#   "Alembic 0017, down_revision 0015"
# - 바꾼 기록: "0019의 `down_revision`을 0017에서 0018로" (바꾼 뒤 값을 읽는다)
# - 체인: "마이그레이션 체인은 0015 → 0017(설명) → 0018" (뒤 리비전의 down이 앞 리비전,
#   괄호 설명은 뺀다)
# 한글도 \w라 "0019다"·"down은"의 경계는 \b로 잡히지 않는다: 숫자 뒤는 (?!\d), down 앞뒤는
# ASCII 글자로 본다.
_DOC_DOWN = re.compile(
    r"(?:Alembic\s+`?|(?<!\d)(?=\d{4}_\w+\.py))(?P<rev>\d{4})(?:_\w+\.py)?`?"
    r"\s*[(,]\s*`?down(?:_revision)?`?\s*(?P<down>\d{4})(?!\d)"
)
_DOC_DOWN_CHANGE = re.compile(
    r"(?<!\d)(?P<rev>\d{4})의\s*`?down(?:_revision)?`?[을를]\s*\d{4}에서\s*(?P<down>\d{4})(?!\d)"
)
_DOC_CHAIN = re.compile(r"체인[은는이]?\s+(?P<chain>\d{4}(?:(?:\([^)]*\))?\s*→\s*\d{4}(?!\d))+)")
# down 리비전을 언급하는 자리: 같은 줄 뒤에 리비전 번호가 오는 down·down_revision, 번호 사이 화살표
# (ADR 번호 사이 화살표는 뺀다). 모두 위 표기 안에 있어야 한다 (새 표기를 조용히 건너뛰지 않는다).
_DOC_DOWN_MENTION = re.compile(
    r"(?<![A-Za-z])down(?:_revision)?(?![A-Za-z_])(?=[^\n]*\d{4})"
    r"|(?<!ADR )(?<!\d)\d{4}(?:\([^)]*\))?\s*→(?=\s*\d{4}(?!\d))"
)


def _documented_downs(text: str) -> tuple[list[tuple[str, str, int]], list[int]]:
    """문서 본문에서 적은 down 리비전을 읽는다.

    반환: ((리비전, 적은 down 리비전, 줄 번호) 목록, 위 표기로 읽지 못한 down 언급의 줄 번호 목록).
    """

    def line(pos: int) -> int:
        return text.count("\n", 0, pos) + 1

    found: list[tuple[str, str, int]] = []
    spans: list[tuple[int, int]] = []
    for pattern in (_DOC_DOWN, _DOC_DOWN_CHANGE):
        for m in pattern.finditer(text):
            found.append((m["rev"], m["down"], line(m.start())))
            spans.append(m.span())
    for m in _DOC_CHAIN.finditer(text):
        revs = re.findall(r"\d{4}", re.sub(r"\([^)]*\)", "", m["chain"]))
        found += [(rev, down, line(m.start())) for down, rev in itertools.pairwise(revs)]
        spans.append(m.span())
    unparsed = [
        line(m.start())
        for m in _DOC_DOWN_MENTION.finditer(text)
        if not any(start <= m.start() < end for start, end in spans)
    ]
    return found, unparsed


def test_documented_down_parser_reads_every_notation() -> None:
    """괄호·쉼표·`down_revision`·바꾼 기록·체인 표기를 (리비전, down) 쌍으로 읽고, 어느 표기에도
    맞지 않는 down 언급은 줄 번호로 낸다. ADR 번호 사이 화살표, markdown·downgrade는 언급이 아니다.
    정답은 문자열에 적은 값이다 (7번째 줄만 읽지 못한 언급)."""
    text = (
        "표 5개는 Alembic 0019(down 0018, WP17.4),\n"
        "Alembic `0019_ai_review.py`(down 0018, 5개 표)\n"
        "열(Alembic 0018, down 0017),\n"
        "테이블(추가 전용, Alembic 0017, down_revision 0015).\n"
        "마이그레이션 체인은 0015 → 0017(승인 보류, ADR 0042)\n"
        "  → 0018(골든, ADR 0041) → 0019다 (병합 때 0019의 `down_revision`을 0017에서 0018로\n"
        "0019는 down 리비전이 0016이다.\n"
        "ADR 0021 → 0027에서 바뀜. markdown 0001, downgrade 0002\n"
    )
    found, unparsed = _documented_downs(text)
    assert sorted(found) == sorted([
        ("0019", "0018", 1), ("0019", "0018", 2), ("0018", "0017", 3), ("0017", "0015", 4),
        ("0017", "0015", 5), ("0018", "0017", 5), ("0019", "0018", 5), ("0019", "0018", 6),
    ])  # fmt: skip
    assert unparsed == [7]


def test_documented_down_revisions_match_migrations() -> None:
    """ADR·구현 계획에 적은 "리비전(down 이전 리비전)"이 실제 `down_revision`과 같다.

    브랜치 병합으로 down_revision을 바꾸고 문서를 그대로 두면, 문서만 보고 downgrade·백포트 순서를
    잘못 잡는다 (ADR 0040이 0019를 "down 0017"로 남긴 일). 검수 기록(docs/handover)은 과거 상태를
    인용하므로 보지 않는다. 정답: down 리비전을 언급하는 줄은 모두 `_documented_downs`의 표기로
    읽히고 (ADR 0041 "Alembic 0018, down 0017", ADR 0042 "Alembic 0017, down_revision 0015"처럼
    예전 정규식이 건너뛴 꼴을 놓치지 않는다), 읽은 표기가 하나 이상 있으며, 모두 마이그레이션 파일과
    같다.
    """
    script = ScriptDirectory.from_config(alembic_config("sqlite://"))
    actual = {s.revision: s.down_revision for s in script.walk_revisions()}
    docs = repo_root() / "docs"
    found: list[tuple[str, int, str, str]] = []
    unparsed: list[tuple[str, int]] = []
    for path in [*sorted((docs / "adr").glob("*.md")), docs / "ai-implementation-plan.md"]:
        got, missed = _documented_downs(path.read_text(encoding="utf-8"))
        found += [(path.name, line, rev, down) for rev, down, line in got]
        unparsed += [(path.name, line) for line in missed]
    assert unparsed == [], f"down 리비전 언급을 읽지 못했다 (문서, 줄; 표기를 더한다): {unparsed}"
    assert found
    wrong = [
        (name, line, rev, down, actual.get(rev))
        for name, line, rev, down in found
        if actual.get(rev) != down
    ]
    assert wrong == [], f"문서의 down 리비전이 다르다 (문서, 줄, 리비전, 적은 값, 실제): {wrong}"


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


# TRUNCATE를 막아야 하는 추가 전용 테이블 (0010~0012, 0017, 0021, 0022, 0023)
APPEND_ONLY_TABLES = (
    "label_records", "raw_access_log", "review_work", "privacy_audits", "retention_decisions",
    "session_lifecycle_events", "privacy_detection_runs", "privacy_holds",
    "description_inheritance", "camera_calibrations", "worker_hand_sizes",
    "trajectory_inheritance",
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
def test_calibration_registries_are_append_only(pg: sa.Engine) -> None:
    """카메라 내부 파라미터·작업자 손 크기 등록(0022, ADR 0045)은 수정·삭제할 수 없다."""
    from dlp_schema.calibration import CalibrationMethod, HandSizeSource
    from dlp_schema.db.calibration_repository import (
        register_camera_calibration,
        register_hand_size,
    )
    from dlp_schema.session import CameraIntrinsics

    k = CameraIntrinsics(width=640, height=480, fx=500.0, fy=500.0, cx=320.0, cy=240.0)
    with pg.begin() as conn:
        register_camera_calibration(conn, "cam-1", k, method=CalibrationMethod.IMPORTED,
                                    now=FIXED_TIME)  # fmt: skip
        register_hand_size(conn, "w01", 0.08, source=HandSizeSource.MEASURED, now=FIXED_TIME)
    for table, statement in (
        ("camera_calibrations", "UPDATE camera_calibrations SET camera_id = 'x'"),
        ("camera_calibrations", "DELETE FROM camera_calibrations"),
        ("worker_hand_sizes", "UPDATE worker_hand_sizes SET palm_m = 0.1"),
        ("worker_hand_sizes", "DELETE FROM worker_hand_sizes"),
    ):
        with pytest.raises(DBAPIError, match=table), pg.begin() as conn:
            conn.execute(sa.text(statement))
    url = pg.url.render_as_string(hide_password=False)
    pg.dispose()
    with pytest.raises(RuntimeError, match="0022"):
        downgrade(url, "0021")


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


def test_set_model_status_rejects_candidate() -> None:
    """판정된 모델을 candidate로 되돌리면 ValueError (DB에 닿기 전에 거부).

    되돌리면 decided_at이 채워진 candidate가 되어 `ModelVersion` 검증기를 어기고, 이후
    `get_model_version`이 그 행을 읽지 못한다. 회귀 테스트: 예전에는 그대로 UPDATE했다.
    """
    # 연결 객체는 쓰이지 않아야 한다 (검사가 UPDATE보다 먼저). 쓰이면 AttributeError가 난다.
    conn: sa.Connection = object()  # type: ignore[assignment]
    with pytest.raises(ValueError, match="candidate"):
        set_model_status(conn, "m1", ModelStatus.CANDIDATE, FIXED_TIME)


@pytest.mark.services
def test_set_lifecycle_rejects_naive_time_before_update(pg: sa.Engine) -> None:
    """시간대 없는 at은 sessions를 바꾸기 전에 거부한다.

    예외를 잡고 같은 트랜잭션을 커밋해도 상태가 바뀌지 않고 기록도 남지 않는다.
    회귀 테스트: 예전에는 UPDATE 뒤 기록 단계에서야 거부해서 기록 없는 전이가 커밋될 수 있었다.
    """
    with pg.begin() as conn, pytest.raises(ValueError, match="시간대"):
        set_lifecycle(conn, "s001", LifecycleState.PRIVACY_APPROVED, at=datetime(2026, 1, 1))
    with pg.connect() as conn:
        assert get_session(conn, "s001").lifecycle_state is LifecycleState.RAW_INGESTED
        assert len(list_lifecycle_events(conn, "s001")) == 1  # 등록 기록만


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


@pytest.mark.services
def test_privacy_holds_are_append_only_and_block_downgrade(pg: sa.Engine) -> None:
    """승인 보류 기록(0017, ADR 0042)은 수정·삭제할 수 없고, 기록이 있으면 0017을 되돌리지 않는다.

    마지막 기록이 hold면 보류 중, release면 아니다 (일련번호 순, 같은 시각이어도).
    """
    with pg.begin() as conn:
        record_privacy_hold(conn, "s001", "hold", reason="report", actor="ops01", at=FIXED_TIME)
        assert active_privacy_hold(conn, "s001") is not None
        record_privacy_hold(conn, "s001", "release", reason="ok", actor="ops02", at=FIXED_TIME)
        assert active_privacy_hold(conn, "s001") is None
        assert [h.action for h in list_privacy_holds(conn, "s001")] == ["hold", "release"]
    for statement in (
        "UPDATE privacy_holds SET action = 'release'",
        "DELETE FROM privacy_holds",
    ):
        with pytest.raises(DBAPIError, match="privacy_holds"), pg.begin() as conn:
            conn.execute(sa.text(statement))
    url = pg.url.render_as_string(hide_password=False)
    pg.dispose()
    with pytest.raises(RuntimeError, match="0017"):
        downgrade(url, "0015")


@pytest.mark.services
def test_description_inheritance_is_append_only(pg: sa.Engine) -> None:
    """설명 물려받기 기록(0021, ADR 0043 3절)은 수정·삭제할 수 없다. 기록은 같은 트랜잭션의
    `record_review`와 함께 설명 따르기가 쓴다 (경로·준 주인·검수 상태가 그대로 읽힌다)."""
    from dlp_schema.db.description_repository import apply_description_follow
    from dlp_schema.review import InheritanceMechanism

    model = Provenance(source=Source.MODEL, model_version="actions-1")
    action = make_label(action_payload(), label_id="a1", provenance=model, confidence=0.9)
    desc = make_label({"kind": "description", "segment_id": "a001", "text": "설명"},
                      label_id="d1", provenance=model, confidence=0.9)  # fmt: skip
    with pg.begin() as conn:
        insert_labels(conn, [action, desc])
        record_review(conn, "a1", VerificationState.HUMAN_APPROVED, "r1", FIXED_TIME)
        apply_description_follow(conn, "s001", FIXED_TIME, InheritanceMechanism.VERIFY)
    with pg.connect() as conn:
        (rec,) = get_description_inheritance(conn, "s001")
    assert (rec.description_label_id, rec.giver_label_id, rec.reviewer_id) == ("d1", "a1", "r1")
    assert rec.mechanism is InheritanceMechanism.VERIFY and rec.reviewed_at == FIXED_TIME
    for statement in (
        "UPDATE description_inheritance SET reviewer_id = 'x'",
        "DELETE FROM description_inheritance",
    ):
        with pytest.raises(DBAPIError, match="description_inheritance"), pg.begin() as conn:
            conn.execute(sa.text(statement))


@pytest.mark.services
def test_trajectory_inheritance_is_append_only_and_blocks_downgrade(pg: sa.Engine) -> None:
    """3D 궤적 물려받기 기록(0023, ADR 0048)은 수정·삭제할 수 없고, 기록이 있으면 0023을 되돌릴
    수 없다. 기록은 같은 트랜잭션의 `record_review`와 함께 궤적 따르기가 쓴다."""
    from dlp_schema.db.repository import get_trajectory_inheritance
    from dlp_schema.db.trajectory_repository import apply_trajectory_follow
    from dlp_schema.review import TrajectoryInheritanceMechanism
    from dlp_schema.testing import hand_track_payload, trajectory_payload

    model = Provenance(source=Source.MODEL, model_version="m-1")
    track = make_label(hand_track_payload(), label_id="g1", stream_id="bodycam",
                       provenance=model, confidence=0.8)  # fmt: skip
    cam = make_label(trajectory_payload(derived_from=("g1",)), label_id="c1",
                     stream_id="bodycam", provenance=model, confidence=0.8)  # fmt: skip
    with pg.begin() as conn:
        insert_labels(conn, [track, cam])
        record_review(conn, "g1", VerificationState.HUMAN_APPROVED, "r1", FIXED_TIME)
        apply_trajectory_follow(conn, "s001", FIXED_TIME, TrajectoryInheritanceMechanism.VERIFY)
    with pg.connect() as conn:
        (rec,) = get_trajectory_inheritance(conn, "s001")
        [c1] = [x for x in get_labels(conn, "s001") if x.label_id == "c1"]
    assert (rec.trajectory_label_id, rec.source_label_id, rec.reviewer_id) == ("c1", "g1", "r1")
    assert c1.verification.state is VerificationState.HUMAN_APPROVED
    for statement in (
        "UPDATE trajectory_inheritance SET reviewer_id = 'x'",
        "DELETE FROM trajectory_inheritance",
    ):
        with pytest.raises(DBAPIError, match="trajectory_inheritance"), pg.begin() as conn:
            conn.execute(sa.text(statement))
    url = pg.url.render_as_string(hide_password=False)
    pg.dispose()
    with pytest.raises(RuntimeError, match="0023"):
        downgrade(url, "0022")


def test_adding_english_names_to_registered_draft_is_additive(ontology: Ontology) -> None:
    """en 없이 등록된 초안 v1에 en을 더한 사전을 다시 등록하면 덧붙이기로 받는다 (ADR 0028·0043).

    서비스 없이 시간대 보존 SQLite로 돈다. 저장 내용은 새 사전(en 포함)으로 바뀐다.
    """
    from dlp_schema.db.ontology_repository import LegacyOntologyContentError, get_ontology
    from dlp_schema.testing import sqlite_db

    def strip_en(value: Any) -> Any:
        if isinstance(value, dict):
            items: dict[str, Any] = value  # pyright: ignore[reportUnknownVariableType]
            return {k: strip_en(v) for k, v in items.items() if k != "en"}
        return value

    old = ontology.model_dump(mode="json")
    engine = sqlite_db()
    try:
        with engine.begin() as conn:
            # 예전 형식(en 없음) 그대로의 행을 넣는다 (지금 계약으로는 만들 수 없는 내용)
            conn.execute(
                ontology_versions.insert().values(
                    version=ontology.version, status="draft", content=strip_en(old)
                )
            )
        # 다시 등록하기 전에는 읽을 때 재등록 안내로 실패한다
        with engine.connect() as conn, pytest.raises(LegacyOntologyContentError, match="register"):
            get_ontology(conn, ontology.version)
        with engine.begin() as conn:
            register_ontology(conn, ontology)
        with engine.connect() as conn:
            stored = conn.execute(sa.select(ontology_versions.c.content)).scalar_one()
            assert get_ontology(conn, ontology.version) == ontology
        assert stored == old
    finally:
        engine.dispose()


def test_adding_physical_priors_to_registered_draft_is_additive(ontology: Ontology) -> None:
    """물리 사전값 없이 등록된 초안 v1에 사전값·재질·촉감 어휘를 더해 다시 등록하면 덧붙이기다.

    ADR 0049.

    예전 등록 내용은 새 코드로도 읽히고(사전값 없음, 어휘 빈 사전) 다시 등록하면 새 내용으로 바뀐다.
    내용 해시(`ontology_digest`)는 달라져 AI 1차 검수 단위·검수 화면 프로젝트가 새로 갈린다.
    """
    from dlp_schema.db.ontology_repository import get_ontology
    from dlp_schema.ontology import ontology_digest
    from dlp_schema.testing import sqlite_db

    old = ontology.model_dump(mode="json")
    for obj in old["objects"].values():
        del obj["physical_priors"]
    del old["materials"], old["tactile_properties"]
    engine = sqlite_db()
    try:
        with engine.begin() as conn:
            conn.execute(
                ontology_versions.insert().values(
                    version=ontology.version, status="draft", content=old
                )
            )
        with engine.connect() as conn:
            before = get_ontology(conn, ontology.version)
        assert before.materials == {} and before.objects["cup"].physical_priors is None
        assert ontology_digest(before) != ontology_digest(ontology)
        with engine.begin() as conn:
            register_ontology(conn, ontology)
        with engine.connect() as conn:
            assert get_ontology(conn, ontology.version) == ontology
    finally:
        engine.dispose()


def test_filling_null_priors_is_additive_but_changing_values_is_not(ontology: Ontology) -> None:
    """사전값 없이(null) 등록한 객체에 나중에 사전값을 채우는 것은 덧붙이기지만, 이미 등록된
    사전값의 숫자를 바꾸는 것은 덧붙이기가 아니라 거부된다 (ADR 0028·0049)."""
    from dlp_schema.testing import sqlite_db

    data = ontology.model_dump(mode="json")
    data["objects"]["cup"]["physical_priors"] = None
    without_cup = Ontology.model_validate(data)
    data = ontology.model_dump(mode="json")
    data["objects"]["cup"]["physical_priors"]["mass_kg"]["max"] = 9.0
    changed = Ontology.model_validate(data)
    engine = sqlite_db()
    try:
        with engine.begin() as conn:
            register_ontology(conn, without_cup)
        with engine.begin() as conn:
            register_ontology(conn, ontology)  # null → 사전값: 덧붙이기
        with pytest.raises(ValueError, match="덧붙이기가 아닌"), engine.begin() as conn:
            register_ontology(conn, changed)
    finally:
        engine.dispose()
