"""판정 진행 상태와 `--pending` 선택 테스트 (`status`, SQLite, WP17.5, ADR 0040 5.2 C12).

정답 근거: 같은 세션의 계획 창 대상 수, 판정 전후, 사람이 고친 라벨(맥락 변화)과 새 자동 라벨,
라벨 기록 시각(조용한 시간), 세션 상태(승인 보류·승인 전), 직접 쓴 실행 이벤트로 정한 기대값.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from dlp_aireview.backend import BackendInfo, JudgeBackend, JudgeReply, JudgeRequest
from dlp_aireview.oracle import OracleJudge
from dlp_aireview.policy import AiReviewPolicy
from dlp_aireview.runner import RunOptions, RunSummary, load_input, planned_units, run_from_db
from dlp_aireview.status import pending_sessions, session_status
from dlp_models.vlm import VlmUnavailableError
from dlp_schema.aireview import AiGroup, AiRunEvent, AiRunEventKind
from dlp_schema.db.aireview_repository import (
    insert_ai_run,
    insert_ai_run_event,
    latest_run_state,
)
from dlp_schema.db.repository import insert_labels, record_privacy_hold
from dlp_schema.labels import LabelRecord, Provenance, Source
from dlp_schema.session import PrivacyState
from dlp_schema.testing import FIXED_TIME, make_ai_run

from .blurred import RAW, DbScene, db_scene
from .conftest import with_policy

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


def test_pending_excludes_held_and_unapproved_sessions(
    tmp_path: Path, policy: AiReviewPolicy
) -> None:
    """--pending은 승인 보류 중인 세션과 프라이버시 승인 전 세션을 고르지 않는다 (판정 없는 대상이
    남아 있어도). 같은 DB의 승인된 세션만 고른다."""
    ok = db_scene(tmp_path, session_id="ok1")
    db_scene(tmp_path, session_id="held1", engine=ok.engine)
    db_scene(
        tmp_path, session_id="unapproved1", engine=ok.engine, privacy=PrivacyState.AUTO_BLURRED
    )
    with ok.engine.begin() as conn:
        record_privacy_hold(
            conn, "held1", "hold", reason="노출 신고", actor="ops", at=FIXED_TIME
        )  # 기록만 (승인 상태는 그대로): 보류 조건 자체를 본다
    assert list(_pending(ok, policy)) == ["ok1"]


class _DownFor:
    """정해 둔 단위만 늘 장애인 백엔드 감싸기 (재시도가 끝나도 장애)."""

    def __init__(self, inner: JudgeBackend, unit_ids: set[str]) -> None:
        self.inner, self.unit_ids = inner, unit_ids
        self.info = BackendInfo("stub", "m", "f", ("stub",))

    def judge(self, request: JudgeRequest) -> JudgeReply:
        if request.unit_id in self.unit_ids:
            raise VlmUnavailableError("이 단위만 장애")
        return self.inner.judge(request)


def test_pending_resumes_interrupted_run_whose_targets_look_covered(
    tmp_path: Path, policy: AiReviewPolicy
) -> None:
    """중단된 실행의 남은 단위가 다른 단위에 이미 든 대상만 담아도(공간 트랙은 여러 창에 든다)
    --pending은 그 세션을 다시 고른다: 지금 버전의 최근 실행이 completed가 아니다. 이어서 끝내면 더
    고르지 않는다."""
    p = with_policy(policy, spatial={"stride_ms": 2000, "snapshot_ms": 500})
    db = db_scene(tmp_path, n_units=6)
    work = tmp_path / "w"
    work.mkdir()
    inp = load_input(db.engine, db.labeling, db.session_id, p, db.privacy, work, raw_bucket=RAW)
    plan = planned_units(inp, p, RunOptions())
    covered_elsewhere = [
        u
        for u in plan
        if all(
            any(x.label_id in {y.label_id for y in v.window.targets} for v in plan if v != u)
            for x in u.window.targets
        )
    ]
    assert covered_elsewhere  # 대상이 모두 다른 단위에도 든 단위
    judge = _DownFor(OracleJudge(list(db.scene.labels)), {covered_elsewhere[0].unit_id})
    first = run_from_db(
        db.engine, db.labeling, db.session_id, judge, p, db.privacy, raw_bucket=RAW,
        options=RunOptions(workers=2),
    )  # fmt: skip
    assert first is not None and (first.status, first.reason) == ("interrupted", "unavailable")
    with db.engine.connect() as conn:
        [found] = pending_sessions(conn, p, db.privacy, NOW)
    assert found.uncovered == 0 and found.unfinished == "interrupted:unavailable" and found.failed
    _run(db, p)
    assert _pending(db, p) == {}


def test_pending_puts_failed_sessions_after_fresh_ones(
    tmp_path: Path, policy: AiReviewPolicy
) -> None:
    """최근 시도가 장애·멈춤으로 끝난 세션은 새 세션 뒤로 간다 (늘 실패하는 세션이 앞자리를 차지해
    뒤 세션이 판정되지 않는 일을 막는다). 시간 중단(max_minutes)은 실패가 아니라 순서를 지킨다."""
    a = db_scene(tmp_path, session_id="a1")
    db_scene(tmp_path, session_id="b1", engine=a.engine)
    db_scene(tmp_path, session_id="c1", engine=a.engine)
    version = policy.verdict_version
    with a.engine.begin() as conn:
        for sid, reason in (("a1", "unavailable"), ("b1", "max_minutes")):
            run = make_ai_run(sid, f"air-run-{sid[0] * 24}", version=version)
            insert_ai_run(conn, run)
            for kind, why in ((AiRunEventKind.STARTED, None), (AiRunEventKind.INTERRUPTED, reason)):
                insert_ai_run_event(
                    conn,
                    AiRunEvent(
                        session_id=sid, run_id=run.run_id, event=kind, reason=why, at=FIXED_TIME
                    ),
                )
    with a.engine.connect() as conn:
        found = pending_sessions(conn, policy, a.privacy, NOW)
    assert [(f.session_id, f.failed) for f in found] == [
        ("b1", False), ("c1", False), ("a1", True),
    ]  # fmt: skip


def test_narrowed_completed_run_does_not_hide_an_interrupted_full_run(
    tmp_path: Path, policy: AiReviewPolicy
) -> None:
    """좁힌 실행(`--groups temporal`)의 completed는 앞선 전체 실행의 미완료를 덮지 않는다.

    정답 근거: 전체 실행이 공간 단위 하나를 장애로 남기고 interrupted:unavailable로 끝났다.
    이어서 시간 묶음만 돈 실행은 할 일이 없어도 completed로 끝나지만 `partial:` 마감이라,
    --pending은 여전히 그 세션을 고르고 required 게이트의 완료 버전에도 들지 않는다. 전체 실행이
    남은 단위를 끝내야 완료다.
    """
    p = with_policy(policy, spatial={"stride_ms": 2000, "snapshot_ms": 500})
    db = db_scene(tmp_path, n_units=6)
    work = tmp_path / "w"
    work.mkdir()
    inp = load_input(db.engine, db.labeling, db.session_id, p, db.privacy, work, raw_bucket=RAW)
    spatial = [u for u in planned_units(inp, p, RunOptions()) if u.group is AiGroup.SPATIAL]
    judge = _DownFor(OracleJudge(list(db.scene.labels)), {spatial[0].unit_id})

    def run(j: JudgeBackend, options: RunOptions) -> RunSummary:
        summary = run_from_db(
            db.engine, db.labeling, db.session_id, j, p, db.privacy, raw_bucket=RAW,
            options=options,
        )  # fmt: skip
        assert summary is not None
        return summary

    def state() -> tuple[str | None, frozenset[str]]:
        with db.engine.connect() as conn:
            found = pending_sessions(conn, p, db.privacy, NOW)
            done = latest_run_state(conn, [db.session_id])[db.session_id].completed_versions
        return (found[0].unfinished if found else None), done

    first = run(judge, RunOptions(workers=2))
    assert (first.status, first.reason) == ("interrupted", "unavailable")
    temporal = frozenset({AiGroup.TEMPORAL, AiGroup.TEMPORAL_LONG})
    narrowed = run(
        OracleJudge(list(db.scene.labels)),
        RunOptions(workers=2, groups=temporal, narrowed="groups=temporal,temporal_long"),
    )
    assert narrowed.status == "completed"
    unfinished, done = state()
    assert unfinished == "interrupted:unavailable" and p.verdict_version not in done
    _run(db, p)
    assert state() == (None, frozenset({p.verdict_version}))
