"""AI 1차 검수 표의 PostgreSQL 성질 (services, WP17.4, ADR 0040 I5·I8).

`make up` 뒤 `make test-services`로 돈다 (`pg` 픽스처: 마이그레이션된 일회용 DB, 온톨로지 v1,
세션 s001).

- 다섯 표 모두 UPDATE·DELETE·TRUNCATE를 트리거가 거부한다. 오류 메시지까지 확인한다 (F-SCH-05의
  교훈: match가 아무 오류에나 맞지 않게 표 이름과 트리거 문구를 함께 본다).
- AI 기록을 써도 `label_records`는 그대로다 (판정은 라벨이 아니다).
- 같은 단위 ID는 `ON CONFLICT DO NOTHING`으로 건너뛴다 (트리거에 걸리지 않는다).
- 세션별 실행 잠금(`try_lock_aireview`)은 다른 연결과 경합하고, 커밋해도 유지되며, 풀거나 연결을
  닫으면 풀린다.
- 기록이 있으면 0019를 되돌리지 않는다.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import DBAPIError

from dlp_schema.aireview import AiArm, AiExposure, AiOutcome, AiRunEvent, AiRunEventKind
from dlp_schema.db.aireview_repository import (
    done_unit_ids,
    get_ai_unit,
    insert_ai_exposure,
    insert_ai_outcome,
    insert_ai_run,
    insert_ai_run_event,
    insert_ai_units,
    latest_run_state,
    list_ai_units,
)
from dlp_schema.db.locks import try_lock_aireview, unlock_aireview
from dlp_schema.db.migrate import downgrade
from dlp_schema.db.repository import get_labels, insert_labels, insert_review_task
from dlp_schema.labels import Provenance, Source
from dlp_schema.review import ReviewStage, ReviewTask, ReviewTool
from dlp_schema.testing import (
    AI_VERSION,
    FIXED_TIME,
    action_payload,
    make_ai_run,
    make_ai_unit,
    make_ai_verdict,
    make_label,
)

pytestmark = pytest.mark.services

RUN = "air-run-" + "0" * 24
UNIT = "air-" + "1" * 24
TASK = "ls:7"
# 표 → 행을 하나 고르는 조건 (UPDATE·DELETE 대상)
TABLES = {
    "ai_review_runs": f"run_id = '{RUN}'",
    "ai_review_run_events": "event_id IS NOT NULL",
    "ai_review_units": f"unit_id = '{UNIT}'",
    "ai_review_exposures": f"task_key = '{TASK}'",
    "ai_review_outcomes": f"task_key = '{TASK}'",
}


def _fill(engine: sa.Engine) -> None:
    """모델 라벨 하나, 검수 작업 하나, 다섯 표에 한 행씩."""
    model = Provenance(source=Source.MODEL, model_version="prelabel-1")
    task = ReviewTask(
        task_key=TASK, tool=ReviewTool.LABEL_STUDIO, external_id="7", session_id="s001",
        stream_id="bodycam", stage=ReviewStage.LABELING, media_uri="s3://dlp-labeling/x.mp4",
        label_kinds=("action",), created_at=FIXED_TIME,
    )  # fmt: skip
    with engine.begin() as conn:
        insert_labels(conn, [make_label(action_payload(), label_id="a1", provenance=model,
                                        confidence=0.7)])  # fmt: skip
        insert_review_task(conn, task)
        insert_ai_run(conn, make_ai_run())
        insert_ai_run_event(
            conn,
            AiRunEvent(session_id="s001", run_id=RUN, event=AiRunEventKind.STARTED, at=FIXED_TIME),
        )
        insert_ai_units(conn, [make_ai_unit(UNIT, [make_ai_verdict("a1", "wrong")])])
        insert_ai_exposure(
            conn,
            AiExposure(
                task_key=TASK, session_id="s001", arm=AiArm.SHOWN, arm_key=TASK,
                versions=(AI_VERSION,), coverage=1.0, shown={"a1": UNIT}, created_at=FIXED_TIME,
            ),
        )  # fmt: skip
        insert_ai_outcome(
            conn, AiOutcome(task_key=TASK, session_id="s001", collected_at=FIXED_TIME)
        )


@pytest.mark.parametrize("table", sorted(TABLES))
def test_ai_tables_are_append_only(pg: sa.Engine, table: str) -> None:
    """UPDATE·DELETE·TRUNCATE 모두 그 표의 트리거 메시지로 거부한다 (I5)."""
    _fill(pg)
    where = TABLES[table]
    for statement in (
        f"UPDATE {table} SET session_id = 's001' WHERE {where}",
        f"DELETE FROM {table} WHERE {where}",
    ):
        with (
            pytest.raises(DBAPIError, match=f"{table} 기록은 추가만 할 수 있습니다"),
            pg.begin() as conn,
        ):
            conn.execute(sa.text(statement))
    with pytest.raises(DBAPIError, match=f"{table} 기록은 비울 수 없습니다"), pg.begin() as conn:
        conn.execute(sa.text(f"TRUNCATE {table} CASCADE"))
    with pg.connect() as conn:
        assert conn.execute(sa.text(f"SELECT count(*) FROM {table}")).scalar_one() == 1


def test_ai_records_leave_labels_untouched(pg: sa.Engine) -> None:
    """AI 기록(틀림 판정 포함)을 써도 라벨 이력은 그대로다 (행 수·검수 상태)."""
    with pg.begin() as conn:
        insert_labels(conn, [make_label(action_payload(), label_id="a0")])
    with pg.connect() as conn:
        before = get_labels(conn, "s001")
    _fill(pg)
    with pg.connect() as conn:
        after = get_labels(conn, "s001")
        count = conn.execute(sa.text("SELECT count(*) FROM label_records")).scalar_one()
    assert [x for x in after if x.label_id != "a1"] == before
    assert count == 2
    assert all(x.verification.reviewer_id is None for x in after)


def test_same_unit_is_skipped_without_trigger(pg: sa.Engine) -> None:
    """같은 단위 ID는 ON CONFLICT DO NOTHING으로 건너뛴다 (UPDATE 트리거에 걸리지 않는다, I8)."""
    _fill(pg)
    again = make_ai_unit(UNIT, [make_ai_verdict("a1")], created_at=FIXED_TIME + timedelta(hours=1))
    with pg.begin() as conn:
        assert insert_ai_units(conn, [again]) == 0
    with pg.connect() as conn:
        unit = get_ai_unit(conn, UNIT)
        assert done_unit_ids(conn, "s001", AI_VERSION) == {UNIT}
        assert [u.unit_id for u in list_ai_units(conn, "s001", [AI_VERSION])] == [UNIT]
        state = latest_run_state(conn, ["s001"])["s001"]
    assert unit is not None and unit.verdicts[0].verdict.value == "wrong"
    assert unit.created_at == FIXED_TIME
    assert state.open and state.completed_versions == frozenset()


def test_aireview_lock_contends_across_connections(pg: sa.Engine) -> None:
    """한 연결이 쥔 세션 잠금은 다른 연결이 못 얻고, 커밋해도 유지되며, 풀거나 DB 연결이 끊기면
    풀린다. (풀로 돌려보낸 연결은 DB 세션이 살아 있어 잠금이 남으므로 실행기는 반드시 푼다.)"""
    first, second = pg.connect(), pg.connect()
    try:
        assert try_lock_aireview(first, "s001")
        first.commit()  # 세션 수준 잠금은 트랜잭션 끝과 무관하다
        assert not try_lock_aireview(second, "s001")
        assert try_lock_aireview(second, "s002")  # 다른 세션은 따로
        unlock_aireview(first, "s001")
        assert try_lock_aireview(second, "s001")
        assert not try_lock_aireview(first, "s001")
        second.invalidate()  # DB 연결이 끊기면(프로세스 종료 등) 풀린다
        assert try_lock_aireview(first, "s001")
        unlock_aireview(first, "s001")
    finally:
        first.close()
        second.close()


def test_downgrade_refuses_with_ai_records(pg_url: str, pg: sa.Engine) -> None:
    """AI 기록이 있으면 0019를 되돌리지 않는다 (추가 전용 감사 기록 보존)."""
    _fill(pg)
    with pytest.raises(RuntimeError, match="0019를 되돌릴 수 없습니다"):
        downgrade(pg_url, "0015")
    with pg.connect() as conn:
        assert conn.execute(sa.text("SELECT count(*) FROM ai_review_units")).scalar_one() == 1
