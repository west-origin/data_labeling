"""골든셋 판정 정확도 (`evaluate`, WP17.8 완료 기준: 손 계산 일치, DB에 쓰지 않음).

실제 경로(블러본 게이트 → 실행기 → 메모리 기록)는 시간대 보존 SQLite와 로컬 라벨링 버킷의 합성
블러본(`blurred.db_scene`)으로, 판정은 정답 라벨로 답하는 `OracleJudge`로 돈다. 지표·게이트는 손으로
만든 판정으로 본다.

정답 근거
- perturb: 넣은 오류는 모두 wrong(오라클이 정답과 다른 값을 지적), 손대지 않은 라벨은 ok →
  탐지 재현율 1, 오지적 0, ok 놓침 0, 게이트 통과. DB 표 행 수가 전후 같다.
- history: 사람이 승인한 모델 라벨은 ok, 고친 라벨은 wrong(제안 = 사람 수정본), 지운 박스는
  wrong(삭제). 사람 수정본은 맥락에 들지 않는다.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
import sqlalchemy as sa

from dlp_aireview.evaluate import (
    EvalCase,
    Judged,
    Truth,
    eval_input,
    history_case,
    judge_cases,
    perturb_case,
    score,
)
from dlp_aireview.oracle import OracleJudge
from dlp_aireview.policy import AiReviewPolicy
from dlp_aireview.runner import RunOptions
from dlp_schema.aireview import AiVerdict
from dlp_schema.db.repository import get_labels, insert_labels, record_review
from dlp_schema.db.tables import metadata
from dlp_schema.labels import ActionPayload, LabelRecord, Provenance, Source, VerificationState
from dlp_schema.ontology import Ontology
from dlp_schema.testing import FIXED_TIME, make_ai_verdict

from .blurred import RAW, DbScene, db_scene
from .conftest import make_scene


def _counts(engine: sa.Engine) -> dict[str, int]:
    """모든 표의 행 수 (평가 전후 비교: DB에 쓰지 않는다)."""
    with engine.connect() as conn:
        return {
            t.name: int(conn.execute(sa.select(sa.func.count()).select_from(t)).scalar_one())
            for t in metadata.sorted_tables
        }


def _human(x: LabelRecord) -> LabelRecord:
    """골든 정답처럼 사람이 만든 사본."""
    return x.model_copy(update={"provenance": Provenance(source=Source.HUMAN), "confidence": None})


def _judge(
    db: DbScene, cases: list[EvalCase], oracle: OracleJudge, policy: AiReviewPolicy, tmp: Path
) -> Judged:
    """사례마다 블러본 게이트로 입력을 만들고 판정한다."""
    work = tmp / "work"
    work.mkdir(exist_ok=True)
    inputs = [
        eval_input(db.engine, db.labeling, c, policy, db.privacy, work, raw_bucket=RAW)
        for c in cases
    ]
    return judge_cases(inputs, oracle, policy, options=RunOptions(workers=2))


def test_perturb_mode_catches_injected_errors_and_writes_nothing(
    tmp_path: Path, policy: AiReviewPolicy, ontology: Ontology
) -> None:
    """교란한 라벨은 모두 탐지, 깨끗한 라벨은 오지적 0, 게이트 통과. DB 표 행 수는 그대로다."""
    db = db_scene(tmp_path, n_units=3)
    truth = [_human(x) for x in db.scene.labels]
    rates = {"verb_swap": 0.5, "class_swap": 0.5, "false_event": 0.3, "action_to_gap": 0.3}
    case = perturb_case(db.scene.session, truth, rates, seed=3, o=ontology, policy=policy)
    injected = [t for t in case.truth.values() if t.wrong]
    assert len(injected) >= 3 and any(t.delete for t in injected)
    assert all(x.provenance.source is Source.MODEL for x in case.labels)  # 확정 표시 없음
    before = _counts(db.engine)
    judged = _judge(db, [case], OracleJudge(truth), policy, tmp_path)
    assert _counts(db.engine) == before
    report = score(
        [case], judged, policy, mode="perturb", golden_version="g1", same_model_as_actions=False
    )
    assert report.wrong_recall == 1.0 and report.false_flag_rate == 0.0
    assert report.ok_miss_rate == 0.0 and report.gate.passed and not report.gate.failures
    assert report.targets == len(case.targets) and report.units == len(judged.units) > 0
    assert report.by_error["clean"].flagged == 0
    swapped = report.by_error.get("verb_swap")
    assert swapped is None or swapped.exact == swapped.expected  # 동사 교란은 원래 값 제안
    assert report.by_error["false_event"].exact == report.by_error["false_event"].total


def test_history_mode_uses_reviewed_model_records_without_leaking_answers(
    tmp_path: Path, policy: AiReviewPolicy, ontology: Ontology
) -> None:
    """승인 → ok, 사람 수정 → wrong(제안 = 수정본), 사람이 지운 박스 → wrong(삭제). 수정본은 맥락에
    없다. 미검수 모델 라벨은 대상이 아니다."""
    db = db_scene(tmp_path, n_units=3)
    labels = list(db.scene.labels)
    actions = [x for x in labels if isinstance(x.payload, ActionPayload)]
    approved, fixed = actions[0], actions[1]
    box = next(x for x in labels if x.kind == "box_track")
    later = FIXED_TIME + timedelta(hours=1)
    assert isinstance(fixed.payload, ActionPayload)
    verb = "press" if fixed.payload.verb != "press" else "push"
    child = fixed.model_copy(update={
        "label_id": f"{fixed.label_id}:r1", "parent_label_id": fixed.label_id,
        "provenance": Provenance(source=Source.HUMAN), "confidence": None, "created_at": later,
        "payload": fixed.payload.model_copy(update={"verb": verb}),
    })  # fmt: skip
    gone = box.model_copy(update={
        "label_id": f"{box.label_id}:r1", "parent_label_id": box.label_id, "retracted": True,
        "provenance": Provenance(source=Source.HUMAN), "confidence": None, "created_at": later,
    })  # fmt: skip
    with db.engine.begin() as conn:
        record_review(conn, approved.label_id, VerificationState.HUMAN_APPROVED, "r1", later)
        insert_labels(conn, [child, gone])
        history = get_labels(conn, db.session_id)
    case = history_case(db.scene.session, history, o=ontology, policy=policy)
    assert {t.label_id: (t.wrong, t.error) for t in case.truth.values()} == {
        approved.label_id: (False, "approved"), fixed.label_id: (True, "corrected"),
        box.label_id: (True, "deleted"),
    }  # fmt: skip
    assert case.truth[fixed.label_id].expected == child.payload
    assert child.label_id not in {x.label_id for x in case.labels}  # 정답이 맥락으로 새지 않는다
    # 오라클의 정답: 승인 라벨은 그대로, 고친 라벨은 사람 수정본의 값, 지운 박스는 없음
    oracle_truth = [
        x if x.label_id != fixed.label_id else x.model_copy(update={"payload": child.payload})
        for x in labels if x.label_id != box.label_id
    ]  # fmt: skip
    before = _counts(db.engine)
    judged = _judge(db, [case], OracleJudge(oracle_truth), policy, tmp_path)
    assert _counts(db.engine) == before
    report = score(
        [case], judged, policy, mode="history", golden_version="g1", same_model_as_actions=True
    )
    assert report.wrong_recall == 1.0 and report.false_flag_rate == 0.0
    assert report.by_error["corrected"].exact == 1 and report.by_error["deleted"].exact == 1
    assert report.same_model_as_actions and "action" in report.by_kind


def _case(n_wrong: int, n_clean: int, o: Ontology) -> EvalCase:
    """판정만 보는 가짜 사례 (앞 n_wrong개가 틀린 라벨)."""
    scene = make_scene()
    targets = tuple(scene.labels[: n_wrong + n_clean])
    truth = {
        x.label_id: Truth(x.label_id, i < n_wrong, "verb_swap" if i < n_wrong else "clean")
        for i, x in enumerate(targets)
    }
    return EvalCase(scene.session, o, targets, targets, truth)


def test_score_and_gate_hand_calculation(policy: AiReviewPolicy, ontology: Ontology) -> None:
    """틀린 라벨 4개 중 3개 지적·1개 ok, 깨끗한 라벨 6개 중 1개 지적 → 탐지 재현율 0.75, 오지적률
    1/6, ok 놓침률 0.25. 게이트(재현율 ≥ 0.6, 오지적 ≤ 0.1, ok 놓침 ≤ 0.05)는 오지적·ok 놓침 둘이
    실패이고, block이 false라 차단하지 않는다. ECE: 확신도 0.9 판정 10개, 맞음 8개 → |0.8 - 0.9|."""
    case = _case(4, 6, ontology)
    values = ["wrong", "wrong", "suspect", "ok", "suspect", "ok", "ok", "ok", "ok", "ok"]
    verdicts: dict[str, AiVerdict] = {}
    for x, v in zip(case.targets, values, strict=True):
        extra = {"problems": ("wrong_verb",)} if v in ("wrong", "suspect") else {}
        verdicts[x.label_id] = make_ai_verdict(x.label_id, v, kind=x.kind, confidence=0.9, **extra)
    judged = Judged(verdicts, (), (), 2.0)
    report = score(
        [case], judged, policy, mode="perturb", golden_version="g1", same_model_as_actions=False
    )
    assert report.wrong_recall == pytest.approx(0.75)
    assert report.false_flag_rate == pytest.approx(1 / 6)
    assert report.ok_miss_rate == pytest.approx(0.25)
    assert report.ece == pytest.approx(0.1)
    assert not report.gate.passed and len(report.gate.failures) == 2 and not report.gate.block
    assert report.by_error["verb_swap"].flagged == 3 and report.by_error["clean"].flagged == 1
    # 판정이 하나도 없으면 지표는 자료 없음이고 게이트는 통과로 보지 않는다
    empty = score(
        [case], Judged({}, (), (), 0.0), policy, mode="perturb", golden_version="g1",
        same_model_as_actions=False,
    )  # fmt: skip
    assert empty.wrong_recall == 0.0 and empty.by_error["verb_swap"].unjudged == 4
    assert not empty.gate.passed and empty.units_per_s is None
