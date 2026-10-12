"""`dlp review`의 AI 1차 검수 연결: 판정 사용 설정, required 배정 게이트, 회수 뒤 안내, 큐 표시
(WP17.6, ADR 0040).

서비스 없이 시간대 보존 SQLite(CLI 엔진을 바꿔 끼운다)와 임시 검수 정책 파일
(`DLP_REVIEW_POLICY_FILE`)로 돈다.

정답 근거
- 허용 판정 버전이 비면 `config/policies/aireview.yaml`의 지금 판정 버전 하나다.
- required: 허용 버전 completed 실행 → 진행, waived → 진행, prelabeled 뒤 max_wait_h(24시간) 초과 →
  waived:timeout을 남기고 진행, 그 밖은 배정 0개·종료 6.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy
import yaml

from dlp_aireview.policy import load_policy as load_aireview_policy
from dlp_cli import aireview_run_cmds, review_cmds
from dlp_cli.main import main
from dlp_cli.review_config import ai_display_config
from dlp_review.ops.policy import load_policy as load_ops_policy
from dlp_review.reconcile import ReviewOutcome
from dlp_schema import repo_root
from dlp_schema.aireview import AiRunEvent, AiRunEventKind
from dlp_schema.db import aireview_repository, locks
from dlp_schema.db.aireview_repository import (
    insert_ai_run,
    insert_ai_run_event,
    list_ai_run_events,
)
from dlp_schema.db.repository import (
    insert_assignment,
    insert_labels,
    insert_session,
    list_assignments,
    register_ontology,
    set_lifecycle,
)
from dlp_schema.labels import Provenance, Source
from dlp_schema.ontology import load_ontology
from dlp_schema.review import (
    FlaggedSpan,
    ReviewAssignment,
    ReviewMode,
    ReviewReason,
    ReviewStage,
    ReviewTask,
    ReviewTool,
)
from dlp_schema.session import LifecycleState, PrivacyState
from dlp_schema.testing import (
    FIXED_TIME,
    action_payload,
    make_ai_run,
    make_label,
    make_session,
    sqlite_db,
)

ROOT = repo_root()


class _AwareSqlalchemy:
    """CLI 모듈의 `sa`를 대신한다: create_engine만 시간대 보존 SQLite 엔진으로 바꾼다."""

    def __getattr__(self, name: str) -> Any:
        return getattr(sqlalchemy, name)

    def create_engine(self, url: Any, **kwargs: Any) -> sqlalchemy.Engine:
        return sqlite_db(str(url), create=False)


def _policy_file(tmp_path: Path, **ai: Any) -> Path:
    """저장소 review.yaml에서 ai_review만 바꾼 정책 파일."""
    data = yaml.safe_load((ROOT / "config/policies/review.yaml").read_text("utf-8"))
    data["ai_review"] |= ai
    path = tmp_path / "review.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str, sqlalchemy.Engine]:
    """SQLite 파일 DB (온톨로지 v1). CLI 엔진은 시간대 보존 SQLite로 바꾼다."""
    url = f"sqlite:///{tmp_path / 'cli.db'}"
    engine = sqlite_db(url)
    with engine.begin() as conn:
        register_ontology(conn, load_ontology(ROOT / "config/ontology/v1"))
    monkeypatch.setattr(review_cmds, "sa", _AwareSqlalchemy())
    monkeypatch.setattr(aireview_run_cmds, "sa", _AwareSqlalchemy())
    return url, engine


def _session(engine: sqlalchemy.Engine, sid: str, prelabeled_at: datetime) -> None:
    """privacy_approved로 등록하고 prelabeled_at에 prelabeled로 옮긴 세션 + 모델 행동 둘."""
    session = make_session(
        sid, privacy_state=PrivacyState.APPROVED, lifecycle_state=LifecycleState.PRIVACY_APPROVED
    )
    labels = [
        make_label(
            action_payload(action_id=f"{sid}-a{i}"), label_id=f"{sid}-a{i}", session_id=sid,
            confidence=0.5, provenance=Provenance(source=Source.MODEL, model_version="m"),
        )
        for i in range(2)
    ]  # fmt: skip
    with engine.begin() as conn:
        insert_session(conn, session, at=prelabeled_at - timedelta(hours=1))
        set_lifecycle(conn, sid, LifecycleState.PRELABELED, at=prelabeled_at, actor="prelabel")
        insert_labels(conn, labels)


def _plan(url: str, sid: str) -> int:
    """`dlp review plan <세션> --reviewer r1 --reviewer r2`."""
    return main(["review", "plan", sid, "--reviewer", "r1", "--reviewer", "r2", "--url", url])


def _assignments(engine: sqlalchemy.Engine, sid: str) -> list[ReviewAssignment]:
    with engine.connect() as conn:
        return list_assignments(conn, session_id=sid)


def test_ai_display_config_follows_mode_and_versions() -> None:
    """끔이면 None, 버전이 비면 aireview.yaml 지금 버전 하나, 적으면 그 목록."""
    policy = load_ops_policy(ROOT)
    assert policy.ai_review.mode == "advisory"  # 저장소 값 (장비 도착 전)
    cfg = ai_display_config(policy)
    assert cfg is not None
    assert cfg.versions == (load_aireview_policy(ROOT).verdict_version,)
    # 판정률 분모는 판정 실행과 같은 대상 조건 (aireview.yaml 종류·출처·긴 상위 구간 상한)
    assert cfg.targets == load_aireview_policy(ROOT).target_rule
    off = policy.model_copy(
        update={"ai_review": policy.ai_review.model_copy(update={"mode": "off"})}
    )
    assert ai_display_config(off) is None
    pinned = ("aireview-0123456789ab+vlm-a", "aireview-ba9876543210+vlm-b")
    explicit = policy.model_copy(
        update={"ai_review": policy.ai_review.model_copy(update={"versions": pinned})}
    )
    cfg2 = ai_display_config(explicit)
    assert cfg2 is not None and cfg2.versions == pinned


def test_required_mode_defers_plan_until_judged_waived_or_timed_out(
    db: tuple[str, sqlalchemy.Engine],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """required: 판정 완료 기록이 없으면 배정 0개·종료 6. 지금 버전 completed가 생기면 계획한다.
    prelabeled 뒤 24시간이 지난 세션은 waived:timeout을 남기고 계획하고, 수동 건너뜀(waive)도
    진행시킨다. advisory는 막지 않는다."""
    url, engine = db
    monkeypatch.setenv("DLP_REVIEW_POLICY_FILE", str(_policy_file(tmp_path, mode="required")))
    now = datetime.now(UTC)
    _session(engine, "fresh", now - timedelta(hours=2))
    _session(engine, "old", now - timedelta(hours=30))
    _session(engine, "skip", now - timedelta(hours=2))

    assert _plan(url, "fresh") == review_cmds.AI_PENDING_EXIT
    out = capsys.readouterr().out
    assert "dlp aireview run fresh" in out and "dlp aireview waive fresh" in out
    assert _assignments(engine, "fresh") == []
    version = load_aireview_policy(ROOT).verdict_version
    run = make_ai_run("fresh", version=version, created_at=now)
    with engine.begin() as conn:
        insert_ai_run(conn, run)
        insert_ai_run_event(conn, AiRunEvent(
            session_id="fresh", run_id=run.run_id, event=AiRunEventKind.COMPLETED, at=now,
        ))  # fmt: skip
    assert _plan(url, "fresh") == 0 and _assignments(engine, "fresh")

    assert _plan(url, "old") == 0 and _assignments(engine, "old")
    assert "waived:timeout" in capsys.readouterr().out
    with engine.connect() as conn:
        [event] = list_ai_run_events(conn, "old")
    assert event.event is AiRunEventKind.WAIVED and (event.reason or "").startswith("timeout")

    assert main(["aireview", "waive", "skip", "--reason", "서버 점검", "--url", url]) == 0
    assert _plan(url, "skip") == 0 and _assignments(engine, "skip")

    monkeypatch.setenv("DLP_REVIEW_POLICY_FILE", str(_policy_file(tmp_path, mode="advisory")))
    _session(engine, "adv", now - timedelta(hours=1))
    assert _plan(url, "adv") == 0 and _assignments(engine, "adv")


def test_required_timeout_locks_session_before_reading_run_state(
    db: tuple[str, sqlalchemy.Engine],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """required 시간 초과 경로는 실행 상태를 읽고 waived를 쓰기 전에 세션 행을 잠근다.

    정답 근거: waived INSERT(세션 FK → FOR KEY SHARE)가 잠금보다 먼저면, 같은 세션을 동시에
    계획한 둘이 서로의 KEY SHARE를 기다리며 FOR UPDATE로 올리다 교착한다 (PostgreSQL). 잠금이
    먼저면 뒤 계획은 앞 계획이 커밋한 waived를 읽고 다시 쓰지 않는다.
    """
    url, engine = db
    monkeypatch.setenv("DLP_REVIEW_POLICY_FILE", str(_policy_file(tmp_path, mode="required")))
    _session(engine, "old", datetime.now(UTC) - timedelta(hours=30))
    order: list[str] = []

    def lock(conn: sqlalchemy.Connection, session_id: str) -> None:
        order.append(f"lock:{session_id}")
        locks.lock_session(conn, session_id)

    def state(conn: sqlalchemy.Connection, ids: list[str]) -> Any:
        order.append("state")
        return aireview_repository.latest_run_state(conn, ids)

    def event(conn: sqlalchemy.Connection, ev: AiRunEvent) -> Any:
        order.append(f"event:{ev.event.value}")
        return aireview_repository.insert_ai_run_event(conn, ev)

    monkeypatch.setattr(review_cmds, "lock_session", lock)
    monkeypatch.setattr(review_cmds, "latest_run_state", state)
    monkeypatch.setattr(review_cmds, "insert_ai_run_event", event)
    assert _plan(url, "old") == 0 and _assignments(engine, "old")
    assert order[:3] == ["lock:old", "state", "event:waived"]


def test_plan_without_required_mode_does_not_read_ai_tables(
    db: tuple[str, sqlalchemy.Engine], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """off·advisory의 계획은 AI 실행 기록·생애주기 기록을 읽지 않는다 (review.yaml: off는 AI 표를
    읽지 않는다). 읽으면 실패하도록 저장소 함수를 바꿔 끼운다."""
    url, engine = db

    def boom(*_: Any, **__: Any) -> Any:
        raise AssertionError("required가 아닌데 AI 게이트 기록을 읽었다")

    monkeypatch.setattr(review_cmds, "latest_run_state", boom)
    monkeypatch.setattr(review_cmds, "prelabeled_at", boom)
    now = datetime.now(UTC)
    for mode in ("off", "advisory"):
        monkeypatch.setenv("DLP_REVIEW_POLICY_FILE", str(_policy_file(tmp_path, mode=mode)))
        _session(engine, mode, now - timedelta(hours=1))
        assert _plan(url, mode) == 0 and _assignments(engine, mode)


def test_reissue_hint_includes_aireview_step(capsys: pytest.CaptureFixture[str]) -> None:
    """블러 변경으로 회수한 작업의 다시 만들기 안내: AI 판정을 쓰면 `dlp aireview run <세션>`을
    렌더 뒤에 넣는다 (이전 판정은 렌더 해시가 달라 보이지 않는다, I10)."""
    task = ReviewTask(
        task_key="cvat:1", tool=ReviewTool.CVAT, external_id="1", session_id="s001",
        stream_id="bodycam", stage=ReviewStage.PRIVACY, media_uri="s3://dlp-raw/p.mp4",
        label_kinds=("blur_track",), created_at=FIXED_TIME,
    )  # fmt: skip
    outcome = ReviewOutcome(revoked=["label_studio:2"], reissue_assignments=["A"])
    review_cmds._print_outcome(task, outcome, ai=True)  # pyright: ignore[reportPrivateUsage]
    with_ai = capsys.readouterr().out
    assert "dlp privacy render → dlp aireview run s001 뒤" in with_ai
    review_cmds._print_outcome(task, outcome, ai=False)  # pyright: ignore[reportPrivateUsage]
    assert "dlp aireview run" not in capsys.readouterr().out


def test_queue_shows_ai_reason_count(
    db: tuple[str, sqlalchemy.Engine], capsys: pytest.CaptureFixture[str]
) -> None:
    """열린 배정 큐에 AI 사유 구간 수를 덧붙인다 (사유가 없으면 붙이지 않는다)."""
    url, engine = db
    with engine.begin() as conn:
        insert_session(conn, make_session())
        for aid, reasons in (
            ("A", [ReviewReason.AI_WRONG, ReviewReason.LOW_CONFIDENCE]),
            ("B", []),
        ):
            insert_assignment(conn, ReviewAssignment(
                assignment_id=aid, session_id="s001", label_kinds=("action",),
                mode=ReviewMode.STANDARD, priority=1.0, created_at=FIXED_TIME,
                flagged=tuple(FlaggedSpan(reason=r, t_start_ms=0, t_end_ms=1, label_ids=("l",))
                              for r in reasons),
            ))  # fmt: skip
    assert main(["review", "queue", "--url", url]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert any(" A " in f" {line} " and line.endswith("AI 지적 1") for line in lines)
    assert not any("B" in line.split() and "AI" in line for line in lines)


def _holdout_outcome(engine: sqlalchemy.Engine) -> None:
    """대조군 작업 하나: 판정 wrong(확신도 0.9)인 라벨을 사람이 고쳤다, ok(0.8)인 라벨은 승인."""
    from dlp_schema.aireview import AiArm, AiExposure, AiHumanAction, AiOutcome, AiOutcomeItem
    from dlp_schema.aireview import AiVerdictValue as V
    from dlp_schema.db.aireview_repository import (
        insert_ai_exposure,
        insert_ai_outcome,
        insert_ai_units,
    )
    from dlp_schema.db.repository import insert_review_task
    from dlp_schema.testing import make_ai_unit, make_ai_verdict

    unit = "air-" + "5" * 24
    run = make_ai_run("s001")
    task = ReviewTask(
        task_key="label_studio:7", tool=ReviewTool.LABEL_STUDIO, external_id="7",
        session_id="s001", stream_id="bodycam", stage=ReviewStage.LABELING, assignee="r1",
        media_uri="s3://dlp-labeling/x.mp4", label_kinds=("action",), created_at=FIXED_TIME,
        sent_label_ids=("a", "b"),
    )  # fmt: skip
    verdicts = [
        make_ai_verdict("a", "wrong", confidence=0.9),
        make_ai_verdict("b", "ok", confidence=0.8),
    ]
    with engine.begin() as conn:
        insert_session(conn, make_session())
        insert_review_task(conn, task)
        insert_ai_run(conn, run)
        insert_ai_units(conn, [make_ai_unit(unit, verdicts, run_id=run.run_id)])
        insert_ai_exposure(conn, AiExposure(
            task_key=task.task_key, session_id="s001", arm=AiArm.HOLDOUT, arm_key=task.task_key,
            coverage=1.0, hidden={"a": unit, "b": unit}, created_at=FIXED_TIME,
        ))  # fmt: skip
        insert_ai_outcome(conn, AiOutcome(
            task_key=task.task_key, session_id="s001", reviewer_id="r1", collected_at=FIXED_TIME,
            items=(
                AiOutcomeItem(label_id="a", unit_id=unit, verdict=V.WRONG, arm=AiArm.HOLDOUT,
                              human=AiHumanAction.CORRECTED),
                AiOutcomeItem(label_id="b", unit_id=unit, verdict=V.OK, arm=AiArm.HOLDOUT,
                              human=AiHumanAction.APPROVED),
            ),
        ))  # fmt: skip


def test_quality_ai_report_text_json_and_thresholds(
    db: tuple[str, sqlalchemy.Engine], capsys: pytest.CaptureFixture[str]
) -> None:
    """`dlp review quality --ai`: 대조군 지표(지적 정밀도 1/1, ok 놓침 0/1), 확신도 조안(목표 0.5면
    wrong 0.90), `--json`은 AI 절 JSON(`--ai-thresholds`면 조안 포함). 기간 밖(`--since`가
    뒤)이면 행이 없다."""
    import json

    url, engine = db
    _holdout_outcome(engine)
    args = ["review", "quality", "--ai", "--ai-thresholds", "0.5", "--url", url]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "대조군(holdout, 편향 없음)] 행 2: 지적 정밀도 100.0% (1/1)" in out
    assert "ok 놓침률 0.0% (0/1), 95% 상한 95.0%" in out
    assert "wrong: 0.90 (표본 1, 정밀도 100.0%)" in out
    assert main(["review", "quality", "--ai", "--json", "--url", url]) == 0
    data = json.loads(capsys.readouterr().out)
    holdout = data["arms"]["holdout"]
    assert holdout["flag_precision"] == {
        "num": 1, "den": 1, "value": 1.0, "upper95": 1.0, "weighted": False, "n_eff": None,
    }  # fmt: skip
    assert data["exposure"]["tasks"] == {"holdout": 1}
    # --json과 --ai-thresholds를 함께 주면 조안을 JSON에 넣는다 (예전: 조용히 버렸다)
    both = ["review", "quality", "--ai", "--json", "--ai-thresholds", "0.5", "--url", url]
    assert main(both) == 0
    floors = json.loads(capsys.readouterr().out)["thresholds"]
    assert floors == {
        "target_precision": 0.5, "floors": {"wrong": {"floor": 0.9, "n": 1, "precision": 1.0}},
    }  # fmt: skip
    later = (FIXED_TIME + timedelta(days=1)).isoformat()
    assert main(["review", "quality", "--ai", "--json", "--since", later, "--url", url]) == 0
    assert json.loads(capsys.readouterr().out)["arms"] == {}


@pytest.mark.parametrize(
    ("extra", "message"),
    [(["--json"], "--json"), (["--ai-thresholds", "0.5"], "--ai-thresholds"),
     (["--since", "2026-10-01"], "--since"), (["--until", "2026-10-01T00:00:00+09:00"], "--until")],
)  # fmt: skip
def test_quality_ai_only_args_require_ai(
    extra: list[str], message: str, db: tuple[str, sqlalchemy.Engine]
) -> None:
    """AI 전용 인자를 --ai 없이 주면 조용히 무시하지 않고 알린다 (예전: --json이면 `{}`만 내고
    0)."""
    url, _ = db
    with pytest.raises(SystemExit, match=f"{message}.*--ai와 함께"):
        main(["review", "quality", *extra, "--url", url])


@pytest.mark.parametrize(
    "extra",
    [["--since", "어제"], ["--until", "2026-13-01"], ["--ai-thresholds", "1.5"],
     ["--ai-thresholds", "높음"]],
)  # fmt: skip
def test_quality_bad_time_or_precision_is_usage_error(
    extra: list[str], db: tuple[str, sqlalchemy.Engine], capsys: pytest.CaptureFixture[str]
) -> None:
    """잘못된 ISO 시각·정밀도는 traceback이 아니라 사용법 오류(종료 2)다."""
    url, _ = db
    with pytest.raises(SystemExit) as exc:
        main(["review", "quality", "--ai", *extra, "--url", url])
    assert exc.value.code == 2 and "error" in capsys.readouterr().err


def test_quality_since_after_until_is_refused(db: tuple[str, sqlalchemy.Engine]) -> None:
    """--since가 --until보다 뒤면 빈 기간이라 알린다."""
    url, _ = db
    with pytest.raises(SystemExit, match="앞이어야"):
        main(["review", "quality", "--ai", "--since", "2026-10-02", "--until", "2026-10-01",
              "--url", url])  # fmt: skip


def test_collect_output_shows_ai_counts_and_warnings(capsys: pytest.CaptureFixture[str]) -> None:
    """수집 출력: AI 결과 기록이 있으면 적용·기각·적용 못 함 수와 경고를 덧붙인다 (WP17.7)."""
    from dlp_schema.aireview import (
        AiApplyResult,
        AiArm,
        AiDecision,
        AiHumanAction,
        AiOutcome,
        AiOutcomeItem,
    )

    task = ReviewTask(
        task_key="label_studio:2", tool=ReviewTool.LABEL_STUDIO, external_id="2",
        session_id="s001", stream_id="bodycam", stage=ReviewStage.LABELING,
        media_uri="s3://dlp-labeling/p.mp4", label_kinds=("action",), created_at=FIXED_TIME,
    )  # fmt: skip
    items = (
        AiOutcomeItem(label_id="a", arm=AiArm.SHOWN, human=AiHumanAction.CORRECTED,
                      decision=AiDecision.APPLY, apply_result=AiApplyResult.APPLIED),
        AiOutcomeItem(label_id="b", arm=AiArm.SHOWN, human=AiHumanAction.APPROVED,
                      decision=AiDecision.APPLY, apply_result=AiApplyResult.INVALID_NOW),
        AiOutcomeItem(label_id="c", arm=AiArm.SHOWN, human=AiHumanAction.APPROVED,
                      decision=AiDecision.DISMISS),
    )  # fmt: skip
    record = AiOutcome(
        task_key=task.task_key, session_id="s001", collected_at=FIXED_TIME, items=items,
        warnings=("b: 제안을 적용하면 세션 검증(상위 구간 연결 등)에 걸려 적용하지 않았다",),
    )  # fmt: skip
    outcome = ReviewOutcome(approved=["b", "c"], ai=record)
    review_cmds._print_outcome(task, outcome)  # pyright: ignore[reportPrivateUsage]
    out = capsys.readouterr().out
    assert "AI 제안 적용 1, 기각 1, 적용 안 함(invalid_now 1)" in out
    assert "[AI 경고] b: 제안을 적용하면 세션 검증" in out
