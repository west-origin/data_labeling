"""AI 1차 검수 기록 저장소 테스트 (SQLite, 서비스 없음. WP17.4, ADR 0040).

`dlp_schema.testing.sqlite_db`(시간대 보존 SQLite)로 저장소 함수의 왕복·멱등·필터·상태 계산을 본다.
PostgreSQL 트리거(추가 전용·TRUNCATE 금지)·advisory lock·`ON CONFLICT`의 PG 경로는
`test_aireview_tables.py`(services)가 본다.

정답 근거: 계약 객체 왕복 동일성, 같은 단위 ID는 먼저 쓴 행이 남음(입력이 같으면 ID가 같다),
이벤트 순서로 정한 실행 상태, 반열린 기간 [since, until).
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError

from dlp_schema import repo_root
from dlp_schema.aireview import (
    AiArm,
    AiExposure,
    AiHumanAction,
    AiOutcome,
    AiOutcomeItem,
    AiReviewUnit,
    AiRunEvent,
    AiRunEventKind,
    AiSuggestion,
    AiVerdictValue,
)
from dlp_schema.db.aireview_repository import (
    done_unit_ids,
    get_ai_exposure,
    get_ai_outcome,
    get_ai_run,
    get_ai_unit,
    insert_ai_exposure,
    insert_ai_outcome,
    insert_ai_run,
    insert_ai_run_event,
    insert_ai_units,
    latest_run_state,
    list_ai_exposures,
    list_ai_outcomes,
    list_ai_run_events,
    list_ai_runs,
    list_ai_units,
)
from dlp_schema.db.locks import try_lock_aireview, unlock_aireview
from dlp_schema.db.repository import (
    get_labels,
    insert_labels,
    insert_session,
    record_review,
    register_ontology,
)
from dlp_schema.db.tables import ai_review_units
from dlp_schema.labels import ActionPayload, Verification, VerificationState
from dlp_schema.ontology import load_ontology
from dlp_schema.testing import (
    AI_VERSION,
    FIXED_TIME,
    action_payload,
    make_ai_run,
    make_ai_unit,
    make_ai_verdict,
    make_label,
    make_session,
    sqlite_db,
)

RUN = "air-run-" + "0" * 24
RUN2 = "air-run-" + "1" * 24
OTHER_VERSION = "aireview-ba9876543210+vlm-Qwen/Qwen3-VL-8B-Instruct"


def uid(n: int) -> str:
    """단위 ID `air-` + 24자 (n을 16진수로 채운다)."""
    return f"air-{n:024x}"


@pytest.fixture
def engine() -> Iterator[sa.Engine]:
    """온톨로지 v1, 세션 s001·s002, 실행 RUN(s001)을 등록한 SQLite."""
    eng = sqlite_db()
    with eng.begin() as conn:
        register_ontology(conn, load_ontology(repo_root() / "config/ontology/v1"))
        insert_session(conn, make_session())
        insert_session(conn, make_session("s002"))
        insert_ai_run(conn, make_ai_run())
    yield eng
    eng.dispose()


def _suggested_unit() -> AiReviewUnit:
    """제안(완성 페이로드)이 든 wrong 판정과 ok 판정이 든 단위."""
    wrong = make_ai_verdict(
        "l1",
        "wrong",
        confidence=0.8,
        evidence_t_ms=1_200,
        suggestion=AiSuggestion(
            op="set",
            fields=("verb",),
            payload=ActionPayload.model_validate(action_payload(action_id="l1", verb="lift")),
        ),
    )
    return make_ai_unit(uid(1), [wrong, make_ai_verdict("l2")], context_label_ids=("l9",))


def test_run_event_unit_roundtrip(engine: sa.Engine) -> None:
    """실행·이벤트·단위(제안 페이로드 포함)가 DB 왕복 뒤 같다. event_id는 DB가 매긴다."""
    unit = _suggested_unit()
    with engine.begin() as conn:
        first = insert_ai_run_event(
            conn,
            AiRunEvent(session_id="s001", run_id=RUN, event=AiRunEventKind.STARTED, at=FIXED_TIME),
        )
        second = insert_ai_run_event(
            conn,
            AiRunEvent(
                event_id=999, session_id="s001", run_id=RUN, event=AiRunEventKind.COMPLETED,
                counts={"done": 1, "ok": 1, "wrong": 1}, actor="ops01",
                at=FIXED_TIME + timedelta(minutes=1),
            ),
        )  # fmt: skip
        assert insert_ai_units(conn, [unit]) == 1
    assert second > first
    with engine.connect() as conn:
        assert get_ai_run(conn, RUN) == make_ai_run()
        assert list_ai_runs(conn, "s001") == [make_ai_run()]
        events = list_ai_run_events(conn, "s001")
        assert [e.event for e in events] == [AiRunEventKind.STARTED, AiRunEventKind.COMPLETED]
        assert [e.event_id for e in events] == [first, second]
        assert events[1].counts == {"done": 1, "ok": 1, "wrong": 1}
        assert get_ai_unit(conn, uid(1)) == unit
        assert get_ai_unit(conn, uid(99)) is None
        assert list_ai_units(conn, "s001") == [unit]


def test_same_unit_id_is_ignored_and_first_row_kept(engine: sa.Engine) -> None:
    """같은 단위 ID를 다시 쓰면 건너뛴다 (멱등, I8). 먼저 쓴 행이 남는다."""
    unit = make_ai_unit(uid(1), [make_ai_verdict("l1")])
    later = make_ai_unit(
        uid(1), [make_ai_verdict("l1", "wrong")], created_at=FIXED_TIME + timedelta(hours=1)
    )
    with engine.begin() as conn:
        assert insert_ai_units(conn, [unit]) == 1
        assert (
            insert_ai_units(conn, [unit, later, make_ai_unit(uid(2), [make_ai_verdict("l2")])]) == 1
        )
        assert insert_ai_units(conn, []) == 0
    with engine.connect() as conn:
        assert get_ai_unit(conn, uid(1)) == unit
        assert done_unit_ids(conn, "s001", AI_VERSION) == {uid(1), uid(2)}


def test_done_units_and_listing_filter_by_session_and_version(engine: sa.Engine) -> None:
    """done_unit_ids는 세션·버전별, list_ai_units는 버전 필터(None=전부, 빈 목록=없음)와 만든 순."""
    with engine.begin() as conn:
        insert_ai_run(conn, make_ai_run("s002", RUN2))
        insert_ai_units(
            conn,
            [
                make_ai_unit(uid(3), [make_ai_verdict("l3")], created_at=FIXED_TIME + timedelta(1)),
                make_ai_unit(uid(1), [make_ai_verdict("l1")]),
                make_ai_unit(uid(2), [make_ai_verdict("l2")], version=OTHER_VERSION),
                make_ai_unit(uid(4), [make_ai_verdict("l4")], run_id=RUN2, session_id="s002"),
            ],
        )
    with engine.connect() as conn:
        assert done_unit_ids(conn, "s001", AI_VERSION) == {uid(1), uid(3)}
        assert done_unit_ids(conn, "s001", OTHER_VERSION) == {uid(2)}
        assert done_unit_ids(conn, "s002", OTHER_VERSION) == set()
        assert [u.unit_id for u in list_ai_units(conn, "s001")] == [uid(1), uid(2), uid(3)]
        assert [u.unit_id for u in list_ai_units(conn, "s001", [AI_VERSION])] == [uid(1), uid(3)]
        assert list_ai_units(conn, "s001", []) == []
        assert [u.unit_id for u in list_ai_units(conn, "s002")] == [uid(4)]


def _event(sid: str, kind: AiRunEventKind, minutes: int, run_id: str | None = RUN) -> AiRunEvent:
    """FIXED_TIME + minutes의 이벤트."""
    return AiRunEvent(
        session_id=sid, run_id=run_id, event=kind, at=FIXED_TIME + timedelta(minutes=minutes)
    )


def test_latest_run_state(engine: sa.Engine) -> None:
    """마지막 이벤트(시각 순), 완료 버전, 건너뜀 시각. started로 끝나면 열린(중단 추정) 실행이다."""
    with engine.begin() as conn:
        insert_ai_run(conn, make_ai_run(run_id=RUN2, version=OTHER_VERSION,
                                        created_at=FIXED_TIME + timedelta(hours=1)))  # fmt: skip
        insert_ai_run_event(conn, _event("s001", AiRunEventKind.STARTED, 0))
        insert_ai_run_event(conn, _event("s001", AiRunEventKind.COMPLETED, 5))
        insert_ai_run_event(conn, _event("s001", AiRunEventKind.STARTED, 61, RUN2))
        insert_ai_run_event(conn, _event("s002", AiRunEventKind.WAIVED, 3, None))
    with engine.connect() as conn:
        states = latest_run_state(conn, ["s001", "s002", "s001", "nope"])
        assert latest_run_state(conn, []) == {}
    assert set(states) == {"s001", "s002"}
    s1, s2 = states["s001"], states["s002"]
    assert s1.open and s1.last_event.run_id == RUN2
    assert s1.last_run is not None and s1.last_run.run_id == RUN2
    assert s1.completed_versions == {AI_VERSION}
    assert s1.waived_at is None
    assert not s2.open and s2.last_run is None
    assert s2.completed_versions == frozenset()
    assert s2.waived_at == FIXED_TIME + timedelta(minutes=3)


def _exposure(task_key: str, at: datetime) -> AiExposure:
    """보인 판정 하나, 숨긴 판정 하나인 노출."""
    return AiExposure(
        task_key=task_key, session_id="s001", assignment_id="as1", arm=AiArm.SHOWN,
        arm_key="as1", versions=(AI_VERSION,), render_hash="a" * 64, coverage=0.5,
        shown={"l1": uid(1)}, hidden={"l2": uid(1)}, created_at=at,
    )  # fmt: skip


def _outcome(task_key: str, at: datetime) -> AiOutcome:
    """판정 있는 항목과 없는 항목 하나씩."""
    items = (
        AiOutcomeItem(label_id="l1", unit_id=uid(1), verdict=AiVerdictValue.WRONG,
                      arm=AiArm.SHOWN,
                      human=AiHumanAction.CORRECTED, match="exact", result_label_ids=("l1-h",)),
        AiOutcomeItem(label_id="l3", arm=AiArm.SHOWN, human=AiHumanAction.APPROVED),
    )  # fmt: skip
    return AiOutcome(
        task_key=task_key, session_id="s001", reviewer_id="rev01", collected_at=at,
        items=items, warnings=("AI 영역 무시 1개",),
    )  # fmt: skip


def test_exposures_and_outcomes(engine: sa.Engine) -> None:
    """노출·결과 왕복, 작업마다 한 번(중복은 IntegrityError), 기간 [since, until) 조회."""
    t0 = FIXED_TIME
    with engine.begin() as conn:
        for k, key in enumerate(("ls:1", "ls:2", "ls:3")):
            insert_ai_exposure(conn, _exposure(key, t0 + timedelta(hours=k)))
            insert_ai_outcome(conn, _outcome(key, t0 + timedelta(hours=k)))
    with engine.connect() as conn:
        assert get_ai_exposure(conn, "ls:1") == _exposure("ls:1", t0)
        assert get_ai_exposure(conn, "ls:9") is None
        assert get_ai_outcome(conn, "ls:2") == _outcome("ls:2", t0 + timedelta(hours=1))
        assert get_ai_outcome(conn, "ls:9") is None
        span = (t0 + timedelta(minutes=30), t0 + timedelta(hours=2))
        assert [e.task_key for e in list_ai_exposures(conn, *span)] == ["ls:2"]
        assert [o.task_key for o in list_ai_outcomes(conn, t0, t0 + timedelta(hours=3))] == [
            "ls:1", "ls:2", "ls:3",
        ]  # fmt: skip
        with pytest.raises(ValueError, match="시간대"):
            list_ai_outcomes(conn, datetime(2026, 1, 1), t0)
    with pytest.raises(IntegrityError), engine.begin() as conn:
        insert_ai_exposure(conn, _exposure("ls:1", t0))
    with pytest.raises(IntegrityError), engine.begin() as conn:
        insert_ai_outcome(conn, _outcome("ls:1", t0))


def test_check_constraints_reject_unknown_values(engine: sa.Engine) -> None:
    """계약을 거치지 않은 SQL도 CHECK 제약이 막는다 (이벤트 종류, 묶음, 구간 순서)."""
    with pytest.raises(IntegrityError), engine.begin() as conn:
        conn.execute(
            sa.text(
                "INSERT INTO ai_review_run_events (session_id, event, counts, at) "
                "VALUES ('s001', 'deleted', '{}', '2026-10-07 09:00:00')"
            )
        )
    unit = make_ai_unit(uid(1), [make_ai_verdict("l1")])
    for change in ({"group": "frames"}, {"t_start_ms": 20_000}, {"outcome": "error"}):
        row = unit.model_dump(mode="json") | change | {"created_at": FIXED_TIME}
        with pytest.raises(IntegrityError), engine.begin() as conn:
            conn.execute(ai_review_units.insert().values(**row))


def test_ai_reviewer_cannot_record_review(engine: sa.Engine) -> None:
    """`aireview:` 검수자로는 검수 상태를 기록하지 못한다 (record_review, insert_labels). I3."""
    with engine.begin() as conn:
        insert_labels(conn, [make_label(action_payload(), label_id="a1")])
    with pytest.raises(ValueError, match="AI 판정은 검수가 아닙니다"), engine.begin() as conn:
        record_review(conn, "a1", VerificationState.HUMAN_APPROVED, "aireview:run1", FIXED_TIME)
    verified = Verification(
        state=VerificationState.HUMAN_APPROVED, reviewer_id="aireview:x", reviewed_at=FIXED_TIME
    )
    label = make_label(action_payload(action_id="a2"), label_id="a2", verification=verified)
    with pytest.raises(ValueError, match="AI 판정은 검수가 아닙니다"), engine.begin() as conn:
        insert_labels(conn, [label])
    with engine.begin() as conn:  # 사람 검수자는 그대로 된다
        record_review(conn, "a1", VerificationState.HUMAN_APPROVED, "rev01", FIXED_TIME)
    with engine.connect() as conn:
        [a1] = get_labels(conn, "s001")
    assert a1.verification.reviewer_id == "rev01"


def test_aireview_lock_is_noop_on_sqlite(engine: sa.Engine) -> None:
    """SQLite에는 advisory lock이 없어 늘 얻는다 (PostgreSQL 경합은 services 테스트)."""
    with engine.connect() as conn:
        assert try_lock_aireview(conn, "s001")
        assert try_lock_aireview(conn, "s001")
        unlock_aireview(conn, "s001")
