"""AI 1차 검수 판정 품질 지표 (`dlp_review.ops.ai_quality`, WP17.8 완료 기준: 손 계산 일치).

순수 함수는 손으로 만든 행으로, DB 읽기(`load_ai_rows`·`ai_quality`)는 시간대를 보존하는 SQLite
(`dlp_schema.testing.sqlite_db`)로 본다. 서비스가 필요 없다.

손 계산 (대조군 행 H1~H8, 보인 집단 행 S1~S6, 분모 밖 H9~H11·D1; 각 테스트 docstring)
- 지적 정밀도 = 바꿈∧지적 / 지적, 재현율 = 바꿈∧지적 / 바꿈, ok 놓침률 = 바꿈∧ok / ok.
- 층화: 전수 층 ok 2개 중 1개 바꿈(모집단 2), 표본 층 ok 2개 중 0(가중치 4씩, 모집단 8) →
  합 = (2·0.5 + 8·0)/10 = 0.1. 상한 = (2·CP(1,2) + 8·CP(0,2))/10, CP(1,2) = √0.95,
  CP(0,2) = 1-√0.05.
- κ: 대조군 unsure 제외 7행, 일치 5 → po = 5/7, pe = (4·4 + 3·3)/49 → κ = 10/24.
- ECE(15구간): 확신도 0.95(맞음)·0.85(틀림)·0.75(맞음)·0.65(틀림)가 각자 다른 구간 →
  (0.05 + 0.85 + 0.25 + 0.65)/4 = 0.45.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from datetime import timedelta
from typing import Any

import pytest
import sqlalchemy as sa

from dlp_review.ops.ai_quality import (
    AiRow,
    ai_agreement,
    ai_quality,
    confidence_floors,
    exposure_stats,
    load_ai_rows,
)
from dlp_schema import load_ontology, repo_root
from dlp_schema.aireview import (
    AiApplyResult,
    AiArm,
    AiDecision,
    AiExposure,
    AiHumanAction,
    AiOutcome,
    AiOutcomeItem,
    AiSuggestion,
    AiVerdictValue,
)
from dlp_schema.db.aireview_repository import (
    insert_ai_exposure,
    insert_ai_outcome,
    insert_ai_run,
    insert_ai_units,
)
from dlp_schema.db.repository import (
    insert_assignment,
    insert_labels,
    insert_review_task,
    insert_review_work,
    insert_session,
    register_ontology,
)
from dlp_schema.labels import Provenance, Source
from dlp_schema.ops import ReviewWork
from dlp_schema.review import (
    ReviewAssignment,
    ReviewMode,
    ReviewStage,
    ReviewTask,
    ReviewTaskStatus,
    ReviewTool,
)
from dlp_schema.testing import (
    FIXED_TIME,
    action_payload,
    make_ai_run,
    make_ai_unit,
    make_ai_verdict,
    make_label,
    make_session,
    sqlite_db,
)

V = AiVerdictValue
H = AiHumanAction


def row(
    label: str,
    verdict: AiVerdictValue | None,
    human: AiHumanAction,
    *,
    arm: AiArm = AiArm.HOLDOUT,
    **kw: Any,
) -> AiRow:
    """행 하나 (기본: 대조군, 표준 작업, 전수 층, 제안 없음, 결정 없음)."""
    data: dict[str, Any] = {
        "task_key": "label_studio:1", "label_id": label, "kind": "action", "arm": arm,
        "verdict": verdict, "confidence": None, "has_suggestion": False, "shown": False,
        "human": human, "decision": AiDecision.NONE, "apply_result": None, "match": "na",
    }  # fmt: skip
    data.update(kw)
    return AiRow(**data)


def _apply(result: AiApplyResult) -> dict[str, Any]:
    """적용을 고른 보인 제안 행의 필드."""
    return {"decision": AiDecision.APPLY, "apply_result": result}


def _rows() -> list[AiRow]:
    """모듈 docstring의 손 계산 행."""
    sug: dict[str, Any] = {"has_suggestion": True}
    shown: dict[str, Any] = {"arm": AiArm.SHOWN, "shown": True, "has_suggestion": True}
    return [
        # 대조군 (편향 없는 지표)
        row("h1", V.OK, H.APPROVED, confidence=0.95),
        row("h2", V.OK, H.CORRECTED, confidence=0.85),
        row("h3", V.OK, H.APPROVED, stratum="sample", weight=4.0),
        row("h4", V.OK, H.APPROVED, stratum="sample", weight=4.0),
        row("h5", V.WRONG, H.CORRECTED, kind="event", confidence=0.75, match="exact", **sug),
        row("h6", V.WRONG, H.APPROVED, kind="event", confidence=0.65, match="none", **sug),
        row("h7", V.SUSPECT, H.DELETED, kind="event"),
        row("h8", V.UNSURE, H.APPROVED, kind="event"),
        # 분모 밖: stale, 판정 없음, 재검수, 이중(측정)
        row("h9", V.OK, H.STALE),
        row("h10", None, H.APPROVED),
        row("h11", V.WRONG, H.CORRECTED, followup=True),
        row("d1", V.WRONG, H.NA, arm=AiArm.SHOWN, mode=ReviewMode.DOUBLE),
        # 보인 집단 (자동화 편향 감시·제안 적용)
        row("s1", V.WRONG, H.CORRECTED, match="exact", **shown, **_apply(AiApplyResult.APPLIED)),
        row("s2", V.WRONG, H.CORRECTED, match="none", **shown,
            **_apply(AiApplyResult.CONFLICT_EDITED)),
        row("s3", V.SUSPECT, H.APPROVED, match="none", **shown),
        row("s4", V.WRONG, H.APPROVED, match="none", decision=AiDecision.DISMISS, **shown),
        row("s5", V.OK, H.APPROVED, arm=AiArm.SHOWN),
        row("s6", V.WRONG, H.APPROVED, match="none", **shown,
            **_apply(AiApplyResult.INVALID_NOW)),
    ]  # fmt: skip


def test_agreement_matches_hand_calculation() -> None:
    """대조군·보인 집단 지표, 혼동 행렬, 층화 ok 놓침률과 상한, κ, ECE, 제안 적용, 분모 제외."""
    a = ai_agreement(_rows())
    h = a.arms["holdout"]
    assert h.rows == 8
    assert (h.flag_precision.num, h.flag_precision.den) == (2, 3)  # h5·h7 / h5·h6·h7
    assert (h.flag_recall.num, h.flag_recall.den) == (2, 3)  # h5·h7 / h2·h5·h7
    assert (h.ok_miss.num, h.ok_miss.den) == (1, 4)  # h2 / h1~h4
    assert (h.wrong_approved.num, h.wrong_approved.den) == (1, 2)  # h6 / h5·h6
    assert (h.suggestion_exact.num, h.suggestion_exact.den) == (1, 2)  # h5 / h5·h6
    assert h.confusion == {
        "ok": {"approved": 3, "corrected": 1}, "wrong": {"corrected": 1, "approved": 1},
        "suspect": {"deleted": 1}, "unsure": {"approved": 1},
    }  # fmt: skip
    # 층화 ok 놓침률 (전수 1/2 모집단 2, 표본 0/2 모집단 8)
    full, sample = a.ok_miss_strata["full"], a.ok_miss_strata["sample"]
    assert (full.rate.num, full.rate.den, full.population) == (1, 2, 2.0)
    assert (sample.rate.num, sample.rate.den, sample.population) == (0, 2, 8.0)
    assert a.ok_miss_rate == pytest.approx(0.1)
    assert full.rate.upper95 == pytest.approx(math.sqrt(0.95))
    assert sample.rate.upper95 == pytest.approx(1 - math.sqrt(0.05))
    want = (2 * math.sqrt(0.95) + 8 * (1 - math.sqrt(0.05))) / 10
    assert a.ok_miss_upper95 == pytest.approx(want)
    assert a.kappa == pytest.approx(10 / 24)
    assert a.ece == pytest.approx(0.45)
    # 보인 집단 (d1은 이중 측정이라 빠진다)
    s = a.arms["shown"]
    assert s.rows == 6
    assert (s.flag_precision.num, s.flag_precision.den) == (2, 5)
    assert (s.flag_recall.num, s.flag_recall.den) == (2, 2)
    assert (s.wrong_approved.num, s.wrong_approved.den) == (2, 4)
    assert (s.ok_miss.num, s.ok_miss.den) == (0, 1)
    p = a.apply
    assert p is not None and p.shown_suggestions == 5
    assert (p.apply_rate.num, p.apply_rate.den) == (3, 5)
    assert (p.accept_rate.num, p.accept_rate.den) == (1, 5)
    assert (p.conflict_edited.num, p.conflict_edited.den) == (1, 3)
    assert (p.invalid_now.num, p.invalid_now.den) == (1, 3)
    assert a.excluded == {"stale": 1, "no_verdict": 1, "followup": 1, "measured": 1}
    # 종류별 (대조군): action은 h1~h4, event는 h5~h8
    assert (a.by_kind["action"].ok_miss.num, a.by_kind["action"].ok_miss.den) == (1, 4)
    assert math.isnan(a.by_kind["action"].flag_precision.value)
    assert a.by_kind["event"].flag_precision.value == pytest.approx(2 / 3)


def test_empty_rows_give_nan_not_zero() -> None:
    """자료가 없으면 NaN이다 (0이면 통과처럼 보인다)."""
    a = ai_agreement([])
    assert a.arms == {} and math.isnan(a.ok_miss_rate) and math.isnan(a.ok_miss_upper95)
    assert math.isnan(a.kappa) and math.isnan(a.ece)
    assert a.apply is not None and math.isnan(a.apply.accept_rate.value)


def test_confidence_floors_pick_lowest_threshold_meeting_target() -> None:
    """wrong 확신도 0.65(승인)·0.75(바꿈): 목표 0.6이면 0.65 이상은 1/2라 모자라고 0.75 이상은
    1/1 → 0.75. 목표에 닿지 않으면 None과 전체 정밀도."""
    floors = confidence_floors(_rows(), 0.6)
    assert floors == {"wrong": (0.75, 1, 1.0)}
    assert confidence_floors(_rows(), 0.4) == {"wrong": (0.65, 2, 0.5)}
    unreachable = confidence_floors([row("x", V.WRONG, H.APPROVED, confidence=0.9)], 0.5)
    assert unreachable == {"wrong": (None, 1, 0.0)}


def _exposure(key: str, arm: AiArm, coverage: float) -> AiExposure:
    """노출 기록 하나."""
    return AiExposure(
        task_key=key, session_id="s001", arm=arm, arm_key=key, coverage=coverage,
        created_at=FIXED_TIME,
    )  # fmt: skip


def _task(
    key: str, mode: ReviewMode, status: ReviewTaskStatus = ReviewTaskStatus.OPEN
) -> ReviewTask:
    """Label Studio 작업 하나 (회수 상태면 회수 요청 시각·사유를 채운다)."""
    revoked: dict[str, Any] = (
        {"revoke_requested_at": FIXED_TIME, "revoke_reason": "blur_changed:x"}
        if status is ReviewTaskStatus.REVOKED
        else {}
    )
    return ReviewTask(
        task_key=key, tool=ReviewTool.LABEL_STUDIO, external_id=key.split(":")[1],
        session_id="s001", stream_id="bodycam", stage=ReviewStage.LABELING, assignee="r1",
        media_uri="s3://dlp-labeling/x.mp4", label_kinds=("action",), created_at=FIXED_TIME,
        mode=mode, status=status, sent_label_ids=(), **revoked,
    )  # fmt: skip


def test_exposure_stats_and_seeded_leak() -> None:
    """작업 수(회수 제외), 판정률 평균(제외 집단 빼고), 오류 삽입 누출(블라인드 제외).

    손 계산: 살아 있는 작업 e1 shown(1.0)·e2 holdout(0.5)·e3 no_verdicts(0.0)·e4 오류 삽입
    mode_excluded·e5 블라인드 mode_excluded → 판정률 (1+0.5+0)/3 = 0.5. 블라인드 아닌 4개 중 "AI
    없음" 화면 3개(e2·e3·e4)에 오류 삽입 1개 → 1/3, 전체 1/4.
    """
    exposures = [
        _exposure("label_studio:1", AiArm.SHOWN, 1.0),
        _exposure("label_studio:2", AiArm.HOLDOUT, 0.5),
        _exposure("label_studio:3", AiArm.NO_VERDICTS, 0.0),
        _exposure("label_studio:4", AiArm.MODE_EXCLUDED, 0.0),
        _exposure("label_studio:5", AiArm.MODE_EXCLUDED, 0.0),
        _exposure("label_studio:6", AiArm.SHOWN, 1.0),
    ]
    tasks = {
        "label_studio:1": _task("label_studio:1", ReviewMode.STANDARD),
        "label_studio:2": _task("label_studio:2", ReviewMode.STANDARD),
        "label_studio:3": _task("label_studio:3", ReviewMode.STANDARD),
        "label_studio:4": _task("label_studio:4", ReviewMode.SEEDED_ERROR),
        "label_studio:5": _task("label_studio:5", ReviewMode.BLIND),
        "label_studio:6": _task("label_studio:6", ReviewMode.STANDARD, ReviewTaskStatus.REVOKED),
    }
    e = exposure_stats(exposures, tasks)
    assert e.tasks == {"shown": 1, "holdout": 1, "no_verdicts": 1, "mode_excluded": 2}
    assert e.coverage == pytest.approx(0.5)
    assert (e.seeded_share_no_ai.num, e.seeded_share_no_ai.den) == (1, 3)
    assert (e.seeded_share_all.num, e.seeded_share_all.den) == (1, 4)


# ---------------------------------------------------------------- DB (SQLite)

SID = "s001"
UNIT = "air-" + "3" * 24
RUN = "air-run-" + "3" * 24


@pytest.fixture
def engine() -> Iterator[sa.Engine]:
    """표와 온톨로지 v1을 만든 메모리 SQLite."""
    eng = sqlite_db()
    with eng.begin() as conn:
        register_ontology(conn, load_ontology(repo_root() / "config/ontology/v1"))
    yield eng
    eng.dispose()


def test_load_rows_and_quality_from_db(engine: sa.Engine) -> None:
    """결과 기록 → 행: 라벨 종류, 판정 확신도·제안(단위에서), 화면 표시(노출 기록), 층·가중치(배정
    표본 2개 + 보류 6개 → 가중치 4), 재검수 작업 표시. 검수 시간은 집단별로 (재검수 제외).

    손 계산: 대조군 작업(holdout) 표본 라벨 2개 모두 ok·승인 → 표본 층 0/2, 모집단 8. 검수 시간
    대조군 30분 / 영상 1분(60,000 ms) → 영상 1시간당 1,800분. 재검수 작업의 행은 followup.
    """
    m: dict[str, Any] = {
        "provenance": Provenance(source=Source.MODEL, model_version="actions-1"), "confidence": 0.9,
    }  # fmt: skip
    labels = [
        make_label(action_payload(action_id=f"a{i}"), label_id=f"s001-a{i}", **m) for i in range(3)
    ]
    swap = labels[2].payload.model_copy(update={"verb": "press"})
    verdicts = [
        make_ai_verdict("s001-a0", "ok", confidence=0.8),
        make_ai_verdict("s001-a1", "ok"),
        make_ai_verdict("s001-a2", "wrong", confidence=0.7,
                        suggestion=AiSuggestion(op="set", fields=("verb",), payload=swap)),
    ]  # fmt: skip
    sample = ReviewAssignment(
        assignment_id="s001-std", session_id=SID, label_kinds=("action",), mode=ReviewMode.STANDARD,
        priority=1.0, sample_label_ids=("s001-a0", "s001-a1"),
        withheld_label_ids=tuple(f"w{i}" for i in range(6)), created_at=FIXED_TIME,
    )  # fmt: skip
    follow = sample.model_copy(
        update={"assignment_id": "s001-std:ai-flagged", "sample_label_ids": (),
                "withheld_label_ids": (), "only_label_ids": ("s001-a2",)}
    )  # fmt: skip
    tasks = [
        _task("label_studio:1", ReviewMode.STANDARD).model_copy(
            update={"assignment_id": "s001-std", "sent_label_ids": ("s001-a0", "s001-a1")}
        ),
        _task("label_studio:2", ReviewMode.STANDARD).model_copy(
            update={"assignment_id": "s001-std:ai-flagged", "sent_label_ids": ("s001-a2",)}
        ),
    ]
    items = {
        "label_studio:1": [
            AiOutcomeItem(label_id="s001-a0", unit_id=UNIT, verdict=V.OK, arm=AiArm.HOLDOUT,
                          human=H.APPROVED),
            AiOutcomeItem(label_id="s001-a1", unit_id=UNIT, verdict=V.OK, arm=AiArm.HOLDOUT,
                          human=H.APPROVED),
        ],
        "label_studio:2": [
            AiOutcomeItem(label_id="s001-a2", unit_id=UNIT, verdict=V.WRONG, arm=AiArm.SHOWN,
                          human=H.CORRECTED, match="exact"),
        ],
    }  # fmt: skip
    with engine.begin() as conn:
        insert_session(conn, make_session(SID))
        insert_labels(conn, labels)
        insert_assignment(conn, sample)
        insert_assignment(conn, follow)
        for t in tasks:
            insert_review_task(conn, t)
        insert_ai_run(conn, make_ai_run(SID, RUN))
        insert_ai_units(conn, [make_ai_unit(UNIT, verdicts, run_id=RUN, session_id=SID)])
        insert_ai_exposure(conn, AiExposure(
            task_key="label_studio:1", session_id=SID, assignment_id="s001-std",
            arm=AiArm.HOLDOUT, arm_key="s001-std", coverage=1.0,
            hidden={"s001-a0": UNIT, "s001-a1": UNIT}, created_at=FIXED_TIME,
        ))  # fmt: skip
        insert_ai_exposure(conn, AiExposure(
            task_key="label_studio:2", session_id=SID, assignment_id="s001-std:ai-flagged",
            arm=AiArm.SHOWN, arm_key="s001-std", coverage=1.0, shown={"s001-a2": UNIT},
            created_at=FIXED_TIME,
        ))  # fmt: skip
        for key, its in items.items():
            insert_ai_outcome(conn, AiOutcome(
                task_key=key, session_id=SID, reviewer_id="r1", collected_at=FIXED_TIME,
                items=tuple(its),
            ))  # fmt: skip
        for key, minutes in (("label_studio:1", 30), ("label_studio:2", 10)):
            insert_review_work(conn, ReviewWork(
                work_id=f"{key}:work", task_key=key, session_id=SID, reviewer="r1",
                stage="labeling", seconds=minutes * 60, video_ms=60_000, source="label_studio",
                recorded_at=FIXED_TIME,
            ))  # fmt: skip
    since, until = FIXED_TIME - timedelta(days=1), FIXED_TIME + timedelta(days=1)
    with engine.connect() as conn:
        rows = {r.label_id: r for r in load_ai_rows(conn, since, until)}
        q = ai_quality(conn, since, until)
    a0, a2 = rows["s001-a0"], rows["s001-a2"]
    assert (a0.kind, a0.confidence, a0.stratum, a0.weight, a0.followup) == (
        "action", 0.8, "sample", 4.0, False,
    )  # fmt: skip
    assert not a0.shown and a0.version is not None
    assert (a2.has_suggestion, a2.shown, a2.followup, a2.stratum) == (True, True, True, "full")
    strata = q.agreement.ok_miss_strata
    assert (strata["sample"].rate.num, strata["sample"].rate.den) == (0, 2)
    assert strata["sample"].population == 8.0 and "shown" not in q.agreement.arms
    assert q.agreement.excluded == {"followup": 1}
    assert q.review_minutes == {"holdout": pytest.approx(1800.0)}
    assert q.exposure.tasks == {"holdout": 1, "shown": 1}
    # 이중 일치·블라인드 편향은 짝 표준 배정의 노출 집단으로 묶는다 (이중은 자기 노출이 짝을 따른다)
    with engine.connect() as conn:
        paired = ai_quality(
            conn, since, until, double_kappa={"s001-std": 0.8, "x": 0.1},
            blind_bias={"b1": 0.2}, pairs={"b1": "s001-std"},
        )  # fmt: skip
    assert paired.double_kappa_by_arm == {"holdout": 0.8}  # 집단을 모르는 배정 x는 뺀다
    assert paired.blind_bias_by_arm == {"holdout": 0.2}
    # 기간 밖이면 아무것도 없다
    with engine.connect() as conn:
        assert load_ai_rows(conn, until, until + timedelta(days=1)) == []
