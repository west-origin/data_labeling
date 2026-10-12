"""DB 경로 실행기 테스트 (`runner.run_from_db`·`load_input`·`DbSink`, SQLite, WP17.5).

시간대 보존 SQLite 파일 DB와 로컬 라벨링 버킷의 합성 블러본(`blurred.db_scene`)으로 실제 경로를
돈다: 세션 상태 확인 → 필요한 스트림만 `open_blurred` → 계획 → `OracleJudge` 판정 → AI 표 기록. PG
트리거·advisory lock 경합은 `test_runner_pg.py`(services)가 본다.

정답 근거: 같은 입력의 계획(`runner.planned_units`), DB 표 행 수, 판정 전후의 라벨 이력.
"""

from __future__ import annotations

import tempfile
import threading
import time
from pathlib import Path

import pytest
import sqlalchemy as sa

from dlp_aireview.backend import JudgeReply, JudgeRequest
from dlp_aireview.oracle import OracleJudge
from dlp_aireview.policy import AiReviewPolicy
from dlp_aireview.runner import (
    RunOptions,
    RunSummary,
    SessionNotReadyError,
    load_input,
    planned_units,
    run_from_db,
)
from dlp_aireview.sensors import SensorContext
from dlp_schema.aireview import AiGroup
from dlp_schema.db.aireview_repository import (
    latest_run_state,
    list_ai_run_events,
    list_ai_runs,
    list_ai_units,
)
from dlp_schema.db.repository import get_labels, record_privacy_hold, set_lifecycle
from dlp_schema.db.tables import label_records
from dlp_schema.episode import current_labels
from dlp_schema.labels import Hand, LabelRecord
from dlp_schema.review import FlaggedSpan
from dlp_schema.session import LifecycleState, PrivacyState, Session
from dlp_schema.testing import FIXED_TIME

from .blurred import RAW, DbScene, db_scene
from .conftest import with_policy


def _label_rows(engine: sa.Engine) -> list[tuple[object, ...]]:
    """라벨 이력 표 전체 (판정 전후 비교)."""
    with engine.connect() as conn:
        return [tuple(r) for r in conn.execute(sa.select(label_records).order_by("label_id"))]


def _run(db: DbScene, policy: AiReviewPolicy, judge: OracleJudge) -> RunSummary | None:
    """run_from_db (작업자 4개)."""
    return run_from_db(
        db.engine, db.labeling, db.session_id, judge, policy, db.privacy, raw_bucket=RAW,
        options=RunOptions(workers=4),
    )  # fmt: skip


def test_run_from_db_writes_ai_tables_only_and_is_idempotent(
    tmp_path: Path, policy: AiReviewPolicy, monkeypatch: pytest.MonkeyPatch
) -> None:
    """실제 경로: 계획 단위가 모두 AI 표에 들어가고 completed. 라벨 이력은 한 행도 바뀌지 않는다.
    재실행은 서버를 부르지 않고 아무것도 쓰지 않는다. 받은 블러본은 실행이 끝나면 지운다."""
    db = db_scene(tmp_path)
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    before = _label_rows(db.engine)
    judge = OracleJudge(list(db.scene.labels))
    summary = _run(db, policy, judge)
    assert summary is not None and summary.status == "completed", summary
    with db.engine.connect() as conn:
        units = list_ai_units(conn, db.session_id)
        runs = list_ai_runs(conn, db.session_id)
        state = latest_run_state(conn, [db.session_id])[db.session_id]
    assert len(units) == summary.planned == summary.written and len(runs) == 1
    assert state.completed_versions == {policy.verdict_version}
    assert {u.render_hash for u in units} == set(runs[0].render_hashes.values())
    assert all(u.blurred_sha256 == runs[0].blurred_sha256[u.stream_id] for u in units)
    assert _label_rows(db.engine) == before
    calls = judge.calls
    again = _run(db, policy, judge)
    assert again is not None and again.status == "nothing_to_do" and judge.calls == calls
    with db.engine.connect() as conn:
        assert len(list_ai_run_events(conn, db.session_id)) == 2
    assert list(scratch.iterdir()) == []  # 블러본 임시 디렉터리는 지워졌다


@pytest.mark.parametrize("state", ["withdrawn", "held", "unapproved"])
def test_session_preconditions_refuse(tmp_path: Path, policy: AiReviewPolicy, state: str) -> None:
    """사용 중지·승인 보류·승인 전 세션은 블러본을 받기 전에 거부한다 (종료 6, 아무것도 안 씀)."""
    db = db_scene(
        tmp_path,
        lifecycle=LifecycleState.WITHDRAWN if state == "withdrawn" else LifecycleState.PRELABELED,
        privacy=PrivacyState.AUTO_BLURRED if state == "unapproved" else PrivacyState.APPROVED,
    )
    if state == "held":
        with db.engine.begin() as conn:
            record_privacy_hold(
                conn, db.session_id, "hold", reason="노출 신고", actor="ops", at=FIXED_TIME
            )
    judge = OracleJudge(list(db.scene.labels))
    with pytest.raises(SessionNotReadyError):
        _run(db, policy, judge)
    with db.engine.connect() as conn:
        assert list_ai_runs(conn, db.session_id) == []
    assert judge.calls == 0


def test_load_input_opens_only_needed_streams_and_reads_sensor_when_needed(
    tmp_path: Path, policy: AiReviewPolicy
) -> None:
    """temporal만 판정하면 기준 스트림 블러본만 받는다. 장갑 문장은 손 상태 대상이 있을 때만
    계산하고(공간만이면 부르지 않는다), 우선순위 사유는 flagged 모드에서만 계산한다."""
    db = db_scene(tmp_path)
    seen: list[str] = []

    def sensor_for(session: Session) -> SensorContext:
        seen.append(session.session_id)
        return SensorContext.of({Hand.RIGHT: [(0.0, 500.0)]})

    flagged_calls: list[int] = []

    def flagged_for(
        conn: sa.Connection, session: Session, current: list[LabelRecord]
    ) -> list[FlaggedSpan]:
        flagged_calls.append(len(current))
        return []

    work = tmp_path / "w"
    work.mkdir()
    temporal = frozenset({AiGroup.TEMPORAL})
    inp = load_input(
        db.engine, db.labeling, db.session_id, policy, db.privacy, work, raw_bucket=RAW,
        groups=temporal, sensor_for=sensor_for, flagged_for=flagged_for,
    )  # fmt: skip
    assert sorted(inp.media) == ["bodycam"] and seen == [db.session_id]
    assert inp.sensor.present and flagged_calls == []
    with db.engine.connect() as conn:
        assert inp.current == tuple(current_labels(get_labels(conn, db.session_id)))
    hand_units = [
        u
        for u in planned_units(inp, policy, RunOptions(groups=temporal))
        if u.sensor_sha256 != "absent"
    ]
    assert hand_units and all(u.sensor_sha256 == inp.sensor.sha256 for u in hand_units)
    spatial = load_input(
        db.engine, db.labeling, db.session_id, policy, db.privacy, work, raw_bucket=RAW,
        groups=frozenset({AiGroup.SPATIAL}), sensor_for=sensor_for,
    )  # fmt: skip
    assert sorted(spatial.media) == ["bodycam", "tp1"] and seen == [db.session_id]
    assert not spatial.sensor.present
    flagged = with_policy(policy, selection={"mode": "flagged"})
    load_input(
        db.engine, db.labeling, db.session_id, flagged, db.privacy, work, raw_bucket=RAW,
        groups=temporal, flagged_for=flagged_for,
    )  # fmt: skip
    assert len(flagged_calls) == 1


def test_dry_run_downloads_but_writes_nothing(tmp_path: Path, policy: AiReviewPolicy) -> None:
    """계획만 (--dry-run): 블러본 게이트는 거치고 AI 표에는 쓰지 않는다."""
    db = db_scene(tmp_path)
    judge = OracleJudge(list(db.scene.labels))
    summary = run_from_db(
        db.engine, db.labeling, db.session_id, judge, policy, db.privacy, raw_bucket=RAW,
        options=RunOptions(dry_run=True),
    )  # fmt: skip
    assert summary is not None and summary.status == "dry_run" and summary.planned > 0
    assert judge.calls == 0
    with db.engine.connect() as conn:
        assert list_ai_runs(conn, db.session_id) == []


class _ChangeOnFirst:
    """첫 판정 요청 때 세션 상태를 바꾸고(사용 중지 또는 승인 보류 기록만) 요청마다 조금 기다리는
    백엔드 감싸기 (실행 중 다른 명령이 세션을 바꾼 상황)."""

    def __init__(self, inner: OracleJudge, db: DbScene, change: str) -> None:
        self.inner, self.db, self.change, self.info = inner, db, change, inner.info
        self.calls = 0
        self._lock = threading.Lock()

    def judge(self, request: JudgeRequest) -> JudgeReply:
        with self._lock:
            self.calls += 1
            first = self.calls == 1
        if first:
            with self.db.engine.begin() as conn:
                if self.change == "withdrawn":
                    set_lifecycle(conn, self.db.session_id, LifecycleState.WITHDRAWN, actor="t")
                else:
                    record_privacy_hold(
                        conn, self.db.session_id, "hold", reason="노출 신고", actor="ops",
                        at=FIXED_TIME,
                    )  # fmt: skip
        time.sleep(0.1)
        return self.inner.judge(request)


@pytest.mark.parametrize("change", ["withdrawn", "held"])
def test_session_change_during_run_aborts(
    tmp_path: Path, policy: AiReviewPolicy, change: str
) -> None:
    """실행 중 세션이 사용 중지되거나 승인 보류 기록이 생기면 (렌더 기록·승인 상태는 그대로라 렌더
    해시만 보면 놓친다) 주기 확인에서 aborted:session_not_ready (종료 6)로 멈추고 남은 단위를 묻지
    않는다."""
    db = db_scene(tmp_path)
    p = with_policy(policy, server={"render_check_interval_s": 0.05, "commit_interval_s": 0.05})
    judge = _ChangeOnFirst(OracleJudge(list(db.scene.labels)), db, change)
    summary = run_from_db(
        db.engine, db.labeling, db.session_id, judge, p, db.privacy, raw_bucket=RAW,
        options=RunOptions(workers=1),
    )  # fmt: skip
    assert summary is not None
    assert (summary.status, summary.reason, summary.exit_code) == (
        "aborted", "session_not_ready", 6,
    )  # fmt: skip
    assert judge.calls < summary.planned
    with db.engine.connect() as conn:
        last = list_ai_run_events(conn, db.session_id)[-1]
    assert (last.event.value, last.reason) == ("aborted", "session_not_ready")
