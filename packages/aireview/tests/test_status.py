"""판정 진행 상태와 `--pending` 선택 테스트 (`status`, SQLite, WP17.5, ADR 0040 5.2 C12).

정답 근거: 같은 세션의 계획 창 대상 수, 판정 전후, 사람이 고친 라벨(맥락 변화)과 새 자동 라벨,
라벨 기록 시각(조용한 시간)으로 직접 정한 기대값.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from dlp_aireview.oracle import OracleJudge
from dlp_aireview.policy import AiReviewPolicy
from dlp_aireview.runner import RunOptions, run_from_db
from dlp_aireview.status import pending_sessions, session_status
from dlp_schema.aireview import AiRunEvent, AiRunEventKind
from dlp_schema.db.aireview_repository import insert_ai_run_event
from dlp_schema.db.repository import insert_labels
from dlp_schema.labels import LabelRecord, Provenance, Source
from dlp_schema.testing import FIXED_TIME

from .blurred import RAW, DbScene, db_scene

NOW = datetime(2026, 10, 12, 12, 0, tzinfo=UTC)


def _pending(db: DbScene, policy: AiReviewPolicy, now: datetime = NOW) -> dict[str, int]:
    """세션 → 판정 없는 대상 수 (`--pending`이 고른 순서)."""
    with db.engine.connect() as conn:
        return {p.session_id: p.uncovered for p in pending_sessions(conn, policy, db.privacy, now)}


def _run(db: DbScene, policy: AiReviewPolicy) -> None:
    """판정 (OracleJudge, 모두 ok)."""
    summary = run_from_db(
        db.engine, db.labeling, db.session_id, OracleJudge(list(db.scene.labels)), policy,
        db.privacy, raw_bucket=RAW, options=RunOptions(workers=2),
    )  # fmt: skip
    assert summary is not None and summary.status == "completed"


def _target(db: DbScene) -> LabelRecord:
    """판정 대상이 되는 행동 라벨 하나."""
    return next(x for x in db.scene.labels if x.kind == "action")


def test_pending_ignores_context_changes_but_picks_new_targets(
    tmp_path: Path, policy: AiReviewPolicy
) -> None:
    """판정 전에는 고르고, 판정 뒤에는 고르지 않는다. 사람이 대상을 고쳐도(맥락 변화) 다시 고르지
    않는다. 새 자동 라벨이 생기면 그 하나 때문에 다시 고른다. 방금 쓴 라벨이 있으면(조용한 시간 전)
    고르지 않는다."""
    db = db_scene(tmp_path)
    with db.engine.connect() as conn:
        before = session_status(conn, db.session_id, policy, db.privacy)
    assert _pending(db, policy) == {db.session_id: before.targets} and before.targets > 0
    _run(db, policy)
    assert _pending(db, policy) == {}
    target = _target(db)
    fixed = target.model_copy(
        update={
            "label_id": f"{target.label_id}-h", "parent_label_id": target.label_id,
            "provenance": Provenance(source=Source.HUMAN), "confidence": None,
        }
    )  # fmt: skip
    with db.engine.begin() as conn:
        insert_labels(conn, [fixed])
    assert _pending(db, policy) == {}
    box = next(x for x in db.scene.labels if x.stream_id == "tp1")
    payload = box.payload.model_copy(update={"entity_id": "bucket_02"})
    new = box.model_copy(update={"label_id": f"{db.session_id}-new-box", "payload": payload})
    with db.engine.begin() as conn:
        insert_labels(conn, [new])
    assert _pending(db, policy) == {db.session_id: 1}
    fresh_payload = box.payload.model_copy(update={"entity_id": "bucket_03"})
    fresh = box.model_copy(
        update={"label_id": f"{db.session_id}-fresh", "payload": fresh_payload, "created_at": NOW}
    )
    with db.engine.begin() as conn:
        insert_labels(conn, [fresh])
    quiet = timedelta(minutes=policy.pending.quiet_minutes)
    assert _pending(db, policy) == {}
    assert _pending(db, policy, NOW + quiet + timedelta(seconds=1)) == {db.session_id: 2}


def test_session_status_counts_and_interrupted_guess(
    tmp_path: Path, policy: AiReviewPolicy
) -> None:
    """판정 뒤 판정률 100%·판정 분포(모두 ok)·마지막 이벤트 completed. 마감 없이 started만 남은
    실행은 "중단 추정"이다 (실행 중이 아니면)."""
    db = db_scene(tmp_path)
    _run(db, policy)
    with db.engine.connect() as conn:
        s = session_status(conn, db.session_id, policy, db.privacy, running=False)
    assert s.coverage == 1.0 and s.covered == s.targets and s.units > 0 and s.fallback == 0
    assert s.verdicts == {"ok": s.targets} and s.last_event == "completed"
    assert s.approved and not s.presumed_interrupted and s.backlog_hours == 0.0
    assert s.units_per_s is None or s.units_per_s > 0
    with db.engine.begin() as conn:
        insert_ai_run_event(
            conn,
            AiRunEvent(
                session_id=db.session_id,
                event=AiRunEventKind.STARTED,
                at=FIXED_TIME + timedelta(days=9),
            ),
        )
    with db.engine.connect() as conn:
        open_ = session_status(conn, db.session_id, policy, db.privacy, running=False)
        busy = session_status(conn, db.session_id, policy, db.privacy, running=True)
    assert open_.presumed_interrupted and open_.last_event == "started"
    assert not busy.presumed_interrupted
