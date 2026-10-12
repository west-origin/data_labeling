"""주간 운영 지표의 AI 1차 검수 절 (`dlp_ops.metrics.ai_weekly`, WP17.8 완료 기준).

서비스 없이 시간대를 보존하는 SQLite(`dlp_schema.testing.sqlite_db`)로 돈다. 정답은 테스트가 넣은
레코드에서 손으로 계산한 값이다 (각 단언 옆 주석).

장면 (2026-W41, 모든 세션 60초 = 1/60시간)
- s001: prelabeled·승인, 판정 실행 없음 → 백로그.
- s002: prelabeled·승인, 허용 버전 completed 실행(걸린 시간 120초) → 백로그 아님. 그 주 단위 4개
  (fallback 1개, 프롬프트 1,000토큰씩, 캐시 400토큰 둘, 나머지 둘은 서버가 캐시 적중을 보고하지
  않음) → fallback 25%, 캐시 800/2,000 = 40% (보고하지 않은 단위는 모름이라 분모에서 뺀다).
  대조군 작업 하나(ok·승인) → ok 놓침 0/1, 95% 상한 = CP(0,1) = 0.95.
- s003: prelabeled·승인, 건너뜀(waived) → 백로그 아님. s004: prelabeled, 승인 전 → 백로그 아님.
- s005: prelabeled·승인, 허용되지 않은 판정 버전의 completed 실행만 있음 → 백로그.
- s006: prelabeled·승인, 실행 없음이지만 사용 중지 → 백로그 아님.
→ 백로그 = s001 + s005 = 2/60시간. 그 주에 끝난 실행은 s002·s005 둘 (영상 2/60시간, 걸린 시간
  120 + 30초).
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

import pytest
import sqlalchemy as sa

from dlp_ops.metrics import AiWeekly, WeeklyMetrics, ai_weekly, alerts, markdown, week_range
from dlp_ops.policy import OpsPolicy, load_policy
from dlp_review.ai_display import AiDisplayConfig
from dlp_review.ops.policy import AiReviewDisplayPolicy
from dlp_schema import load_ontology, repo_root
from dlp_schema.aireview import (
    AiArm,
    AiExposure,
    AiHumanAction,
    AiOutcome,
    AiOutcomeItem,
    AiRunEvent,
    AiRunEventKind,
    AiVerdictValue,
)
from dlp_schema.db.aireview_repository import (
    insert_ai_exposure,
    insert_ai_outcome,
    insert_ai_run,
    insert_ai_run_event,
    insert_ai_units,
)
from dlp_schema.db.repository import (
    insert_review_task,
    insert_session,
    insert_withdrawal,
    register_ontology,
    set_lifecycle,
)
from dlp_schema.lineage import Withdrawal
from dlp_schema.review import ReviewStage, ReviewTask, ReviewTool
from dlp_schema.session import LifecycleState, PrivacyState
from dlp_schema.testing import (
    AI_TARGET_RULE,
    AI_VERSION,
    FIXED_TIME,
    make_ai_run,
    make_ai_unit,
    make_ai_verdict,
    make_session,
    sqlite_db,
)

ROOT = Path(__file__).resolve().parents[3]
WEEK = "2026-W41"
IN = FIXED_TIME  # 2026-10-07 09:00 UTC (W41 안)
AI = AiDisplayConfig(AiReviewDisplayPolicy(mode="advisory"), (AI_VERSION,), AI_TARGET_RULE)
OLD_VERSION = "aireview-" + "e" * 12 + "+vlm-old"  # 허용 버전(AI_VERSION)이 아닌 판정 버전


@pytest.fixture(scope="module")
def policy() -> OpsPolicy:
    """저장소의 실제 `config/policies/ops.yaml`."""
    return load_policy(ROOT)


@pytest.fixture
def engine() -> Iterator[sa.Engine]:
    """표와 온톨로지 v1을 만든 메모리 SQLite."""
    eng = sqlite_db()
    with eng.begin() as conn:
        register_ontology(conn, load_ontology(repo_root() / "config/ontology/v1"))
    yield eng
    eng.dispose()


def _session(conn: sa.Connection, sid: str, privacy: PrivacyState) -> None:
    """60초 세션을 넣고 그 주 전에 prelabeled로 옮긴다."""
    insert_session(conn, make_session(sid, privacy_state=privacy))
    for i, state in enumerate((LifecycleState.PRIVACY_APPROVED, LifecycleState.PRELABELED)):
        set_lifecycle(conn, sid, state, at=IN - timedelta(hours=3 - i))


def _scene(engine: sa.Engine) -> None:
    """모듈 docstring의 장면."""
    run = "air-run-" + "4" * 24
    units = [
        make_ai_unit(
            f"air-{i:024x}", [make_ai_verdict(f"s002-a{i}", "ok")], run_id=run,
            session_id="s002", created_at=IN, prompt_tokens=1000,
            cached_tokens=400 if i < 2 else None,
        )
        for i in range(3)
    ]  # fmt: skip
    units.append(make_ai_unit(
        "air-" + "f" * 24, [make_ai_verdict("s002-a9", "unsure", answered=False,
                                            problems=("response_invalid",))],
        run_id=run, session_id="s002", created_at=IN, outcome="fallback", prompt_tokens=1000,
    ))  # fmt: skip
    task = ReviewTask(
        task_key="label_studio:9", tool=ReviewTool.LABEL_STUDIO, external_id="9",
        session_id="s002", stream_id="bodycam", stage=ReviewStage.LABELING, assignee="r1",
        media_uri="s3://dlp-labeling/x.mp4", label_kinds=("action",), created_at=IN,
        sent_label_ids=("s002-a0",),
    )  # fmt: skip
    old_run = "air-run-" + "5" * 24
    with engine.begin() as conn:
        for sid in ("s001", "s002", "s003", "s005", "s006"):
            _session(conn, sid, PrivacyState.APPROVED)
        _session(conn, "s004", PrivacyState.AUTO_BLURRED)
        insert_ai_run(conn, make_ai_run("s005", old_run, created_at=IN, version=OLD_VERSION))
        insert_ai_run_event(conn, AiRunEvent(
            session_id="s005", run_id=old_run, event=AiRunEventKind.COMPLETED, at=IN,
            counts={"elapsed_s": 30},
        ))  # fmt: skip
        insert_withdrawal(conn, Withdrawal(session_id="s006", reason="동의 철회", withdrawn_at=IN))
        insert_ai_run(conn, make_ai_run("s002", run, created_at=IN))
        insert_ai_units(conn, units)
        insert_ai_run_event(conn, AiRunEvent(
            session_id="s002", run_id=run, event=AiRunEventKind.COMPLETED, at=IN,
            counts={"elapsed_s": 120},
        ))  # fmt: skip
        insert_ai_run_event(conn, AiRunEvent(
            session_id="s003", event=AiRunEventKind.WAIVED, reason="manual: 장비 없음", at=IN,
        ))  # fmt: skip
        insert_review_task(conn, task)
        insert_ai_exposure(conn, AiExposure(
            task_key=task.task_key, session_id="s002", arm=AiArm.HOLDOUT, arm_key=task.task_key,
            coverage=1.0, hidden={"s002-a0": units[0].unit_id}, created_at=IN,
        ))  # fmt: skip
        insert_ai_outcome(conn, AiOutcome(
            task_key=task.task_key, session_id="s002", reviewer_id="r1", collected_at=IN,
            items=(AiOutcomeItem(label_id="s002-a0", unit_id=units[0].unit_id,
                                 verdict=AiVerdictValue.OK, arm=AiArm.HOLDOUT,
                                 human=AiHumanAction.APPROVED),),
        ))  # fmt: skip


def test_ai_weekly_matches_hand_calculation(engine: sa.Engine) -> None:
    """백로그·fallback율·캐시 적중률·장비 시간·판정률·ok 놓침 상한 (모듈 docstring 손 계산)."""
    _scene(engine)
    start, end = week_range(WEEK)
    with engine.connect() as conn:
        ai = ai_weekly(conn, start, end, AI)
        assert ai_weekly(conn, start, end, None) is None  # AI 끔이면 절 없음
    assert ai is not None
    # s001 + s005 (허용되지 않은 버전의 완료 실행은 판정이 끝난 것이 아니다). s006은 사용 중지
    assert ai.backlog_hours == pytest.approx(2 / 60)
    assert ai.fallback_rate == pytest.approx(0.25)  # 1/4
    assert ai.cache_hit_ratio == pytest.approx(0.4)  # 800/2000 (보고 없는 두 단위 제외)
    assert ai.video_hours == pytest.approx(2 / 60)  # s002·s005 실행
    assert ai.device_seconds_per_video_hour == pytest.approx(150 * 30)  # 150초 / (2/60시간)
    assert ai.coverage == pytest.approx(1.0)
    assert ai.ok_miss_rate == 0.0 and ai.ok_miss_upper95 == pytest.approx(0.95)
    assert math.isnan(ai.flag_precision)  # 대조군 지적 없음
    assert ai.counts == {
        "units": 4, "fallback_units": 1, "cache_unreported_units": 2, "runs": 2, "outcome_rows": 1,
    }  # fmt: skip
    # 다음 주에는 그 주 자료가 없다 (백로그는 주 끝 시점: 아직 s001·s005가 남는다)
    with engine.connect() as conn:
        later = ai_weekly(conn, end, end + timedelta(days=7), AI)
    assert later is not None and math.isnan(later.fallback_rate)
    assert later.backlog_hours == pytest.approx(2 / 60)
    # s005의 옛 버전을 허용 버전에 넣으면 백로그에서 빠진다
    both = AiDisplayConfig(AI.policy, (AI_VERSION, OLD_VERSION), AI_TARGET_RULE)
    with engine.connect() as conn:
        allowed = ai_weekly(conn, start, end, both)
    assert allowed is not None and allowed.backlog_hours == pytest.approx(1 / 60)


def test_unreported_cache_hits_are_unknown_not_zero(engine: sa.Engine, policy: OpsPolicy) -> None:
    """서버가 캐시 적중 토큰을 보고하지 않은 단위(`cached_tokens` None)는 적중 0이 아니라 모름이다.

    장면: s002에 그 주 판정 단위 4개, 프롬프트 1,000토큰씩, 모두 캐시 적중 보고 없음 (vLLM 기본,
    `--enable-prompt-tokens-details` 꺼짐). 정답: 캐시 적중률 nan(모름)이고 캐시 경고가 없다.
    한 단위만 400토큰 적중을 보고하면 400/1,000 = 40%다 (보고 없는 셋이 희석하지 않는다).
    """
    run = "air-run-" + "6" * 24
    units = [
        make_ai_unit(
            f"air-{i:024x}", [make_ai_verdict(f"s002-a{i}", "ok")], run_id=run,
            session_id="s002", created_at=IN, prompt_tokens=1000,
        )
        for i in range(4)
    ]  # fmt: skip
    with engine.begin() as conn:
        _session(conn, "s002", PrivacyState.APPROVED)
        insert_ai_run(conn, make_ai_run("s002", run, created_at=IN))
        insert_ai_units(conn, units)
    start, end = week_range(WEEK)
    with engine.connect() as conn:
        ai = ai_weekly(conn, start, end, AI)
    assert ai is not None and math.isnan(ai.cache_hit_ratio)
    assert ai.counts["cache_unreported_units"] == 4
    assert not any("캐시" in w for w in alerts([WeeklyMetrics(WEEK, ai=ai)], policy))
    one = make_ai_unit(
        "air-" + "e" * 24, [make_ai_verdict("s002-a8", "ok")], run_id=run, session_id="s002",
        created_at=IN, prompt_tokens=1000, cached_tokens=400,
    )  # fmt: skip
    with engine.begin() as conn:
        insert_ai_units(conn, [one])
    with engine.connect() as conn:
        mixed = ai_weekly(conn, start, end, AI)
    assert mixed is not None and mixed.cache_hit_ratio == pytest.approx(0.4)  # 400/1000
    assert mixed.counts["cache_unreported_units"] == 4


def test_ai_alerts_and_markdown(policy: OpsPolicy) -> None:
    """`ops.yaml alerts.ai_*`: ok 놓침 상한 > 0.05, 백로그 > 328시간, fallback > 0.05, 캐시 < 0.3
    이면 경고 (마지막 주만). 자료가 없는(NaN) 지표는 경고하지 않는다. Markdown에 AI 절이 붙는다."""
    bad = AiWeekly(ok_miss_upper95=0.2, backlog_hours=400.0, fallback_rate=0.1, cache_hit_ratio=0.1)
    got = alerts([WeeklyMetrics(WEEK, ai=bad)], policy)
    assert sum("AI" in w for w in got) == 4
    assert alerts([WeeklyMetrics(WEEK, ai=AiWeekly())], policy) == [
        w for w in alerts([WeeklyMetrics(WEEK)], policy)
    ]  # 모두 NaN이면 AI 경고 없음
    good = AiWeekly(ok_miss_upper95=0.01, backlog_hours=10, fallback_rate=0.0, cache_hit_ratio=0.5)
    assert not any("AI" in w for w in alerts([WeeklyMetrics(WEEK, ai=good)], policy))
    md = markdown([WeeklyMetrics("2026-W40"), WeeklyMetrics(WEEK, ai=bad)], [], "KRW")
    assert "## AI 1차 검수" in md and "| ok 놓침률 95% 상한 | - | 20.0% |" in md
    assert "## AI 1차 검수" not in markdown([WeeklyMetrics(WEEK)], [], "KRW")
    data = WeeklyMetrics(WEEK, ai=bad).as_dict()
    assert data["ai"]["ok_miss_upper95"] == 0.2 and data["ai"]["coverage"] is None
