"""골든셋 판정 정확도 (`evaluate`, WP17.8 완료 기준: 손 계산 일치, DB에 쓰지 않음).

실제 경로(블러본 게이트 → 실행기 → 메모리 기록)는 시간대 보존 SQLite와 로컬 라벨링 버킷의 합성
블러본(`blurred.db_scene`)으로, 판정은 정답 라벨로 답하는 `OracleJudge`로 돈다. 지표·게이트는 손으로
만든 판정으로 본다.

정답 근거
- perturb: 넣은 오류는 모두 wrong(오라클이 정답과 다른 값을 지적), 손대지 않은 라벨은 ok →
  탐지 재현율 1, 오지적 0, ok 놓침 0, 게이트 통과. DB 표 행 수가 전후 같다.
- history: 사람이 승인한 모델 라벨은 ok, 고친 라벨은 wrong(제안 = 사람 수정본), 지운 박스는
  wrong(삭제). 정답은 후손 사슬의 지금 상태다 (QA가 다시 고쳤으면 마지막 수정본, QA가 지웠으면
  삭제, 원래 값으로 되돌렸으면 맞음). 같은 손·같은 구간을 다른 종류로 다시 그렸으면 replaced(제안 =
  다시 그린 레코드). 사람 수정본·다시 그린 레코드(정답)는 맥락에 들지 않는다.
- ok 놓침률 = P(틀린 라벨 | AI ok) = 틀린 라벨의 ok 수 / 모든 ok 수 (설계 7.2, 품질 보고와 같은
  정의).
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
import sqlalchemy as sa

from dlp_aireview.evaluate import (
    EvalCase,
    Judged,
    LabelKey,
    Truth,
    eval_input,
    history_case,
    judge_cases,
    perturb_case,
    restrict_groups,
    score,
)
from dlp_aireview.oracle import OracleJudge
from dlp_aireview.policy import AiReviewPolicy
from dlp_aireview.runner import RunOptions
from dlp_schema.aireview import AiGroup, AiVerdict
from dlp_schema.db.repository import get_labels, insert_labels, record_review
from dlp_schema.db.tables import metadata
from dlp_schema.labels import (
    ActionPayload,
    Evidence,
    GapPayload,
    Hand,
    HandStatePayload,
    LabelRecord,
    Provenance,
    Source,
    VerificationState,
)
from dlp_schema.ontology import Ontology
from dlp_schema.testing import FIXED_TIME, make_ai_verdict

from .blurred import RAW, DbScene, db_scene
from .conftest import SID, make_scene, with_policy


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


def _child(x: LabelRecord, suffix: str, hours: int, **update: object) -> LabelRecord:
    """x의 사람 자식 레코드 (수정·삭제, hours시간 뒤)."""
    return x.model_copy(update={
        "label_id": f"{x.label_id}:{suffix}", "parent_label_id": x.label_id,
        "provenance": Provenance(source=Source.HUMAN), "confidence": None,
        "created_at": FIXED_TIME + timedelta(hours=hours), **update,
    })  # fmt: skip


def test_history_truth_follows_descendant_chain_and_hides_answers(
    policy: AiReviewPolicy, ontology: Ontology
) -> None:
    """정답 = 후손 사슬의 지금 상태, 정답 레코드는 맥락에 없다 (판정 없이 사례만 본다).

    - twice: 사람이 press로 고친 뒤 QA가 다시 pull로 고침 → corrected, 제안 = pull (첫 수정본 아님).
    - dropped: 사람이 고친 뒤 QA가 지움 → deleted.
    - reverted: 사람이 고친 뒤 QA가 원래 동사로 되돌림 → 맞음(approved).
    - replaced: 사람이 지우고 같은 손·같은 구간을 사이 구간(idle)으로 다시 그림(parent 없음) →
      replaced, 제안 = 다시 그린 사이 구간. 회귀: 다시 그린 레코드는 대상의 후손이 아니라 맥락에
      그대로 들어가 정답이 새었다.
    - 같은 구간이지만 다른 손의 새 사이 구간은 replaced의 정답이 아니다 (그 대상은 deleted).
    """
    scene = make_scene()
    acts = [x for x in scene.labels if isinstance(x.payload, ActionPayload)]
    twice, dropped, reverted, replaced, other = acts[:5]
    for x in (twice, dropped, reverted, replaced, other):
        assert isinstance(x.payload, ActionPayload)

    def verb(x: LabelRecord, v: str) -> dict[str, object]:
        assert isinstance(x.payload, ActionPayload)
        return {"payload": x.payload.model_copy(update={"verb": v})}

    def alt(x: LabelRecord) -> str:
        assert isinstance(x.payload, ActionPayload)
        return "press" if x.payload.verb != "press" else "push"

    t1 = _child(twice, "r1", 1, **verb(twice, alt(twice)))
    t2 = _child(t1, "r2", 2, **verb(twice, "pull" if alt(twice) != "pull" else "lift"))
    d1 = _child(dropped, "r1", 1, **verb(dropped, alt(dropped)))
    d2 = _child(d1, "r2", 2, retracted=True)
    v1 = _child(reverted, "r1", 1, **verb(reverted, alt(reverted)))
    v2 = _child(v1, "r2", 2, payload=reverted.payload)
    gone = _child(replaced, "r1", 1, retracted=True)
    assert isinstance(replaced.payload, ActionPayload) and isinstance(other.payload, ActionPayload)
    gap = GapPayload(hand=replaced.payload.hand, gap_type="idle")
    redraw = replaced.model_copy(update={
        "label_id": f"{SID}-h:rnew", "parent_label_id": None, "stream_id": None, "payload": gap,
        "provenance": Provenance(source=Source.HUMAN), "confidence": None,
        "created_at": FIXED_TIME + timedelta(hours=1),
    })  # fmt: skip
    gone_other = _child(other, "r1", 1, retracted=True)
    flipped = Hand.LEFT if other.payload.hand is Hand.RIGHT else Hand.RIGHT
    wrong_hand = redraw.model_copy(update={
        "label_id": f"{SID}-h:rhand", "t_start_ms": other.t_start_ms, "t_end_ms": other.t_end_ms,
        "payload": GapPayload(hand=flipped, gap_type="idle"),
    })  # fmt: skip
    history = [*scene.labels, t1, t2, d1, d2, v1, v2, gone, redraw, gone_other, wrong_hand]
    case = history_case(scene.session, history, o=ontology, policy=policy)
    got = {x.label_id: (case.truth[x.label_id].wrong, case.truth[x.label_id].error)
           for x in (twice, dropped, reverted, replaced, other)}  # fmt: skip
    assert got == {
        twice.label_id: (True, "corrected"), dropped.label_id: (True, "deleted"),
        reverted.label_id: (False, "approved"), replaced.label_id: (True, "replaced"),
        other.label_id: (True, "deleted"),
    }  # fmt: skip
    assert case.truth[twice.label_id].expected == t2.payload
    assert case.truth[dropped.label_id].delete
    assert case.truth[replaced.label_id].expected == gap
    in_context = {x.label_id for x in case.labels}
    answers = {t1.label_id, t2.label_id, d1.label_id, d2.label_id, v1.label_id, v2.label_id,
               gone.label_id, redraw.label_id, gone_other.label_id}  # fmt: skip
    assert not in_context & answers  # 정답(후손·다시 그린 레코드)이 맥락으로 새지 않는다
    assert wrong_hand.label_id in in_context  # 정답으로 쓰지 않은 사람 추가는 맥락에 남는다


def test_perturb_case_excludes_glove_sensor_labels(
    policy: AiReviewPolicy, ontology: Ontology
) -> None:
    """장갑 센서 근거 라벨(센서 출처, 모델 출처 OBSERVED 손 상태)은 교란·판정 대상이 아니다. 사본은
    모두 모델 출처로 바뀌므로 원래 출처로 거른다 (사용자 결정). 맥락으로는 남는다."""
    scene = make_scene()
    hand = HandStatePayload(
        hand=Hand.LEFT, contact_target_kind="object", target_id="bucket_01", role="active"
    )
    sensor = scene.labels[0].model_copy(update={
        "label_id": f"{SID}-sensor-contact", "stream_id": None, "t_start_ms": 3_000,
        "t_end_ms": 3_500, "payload": hand, "confidence": None,
        "provenance": Provenance(source=Source.SENSOR, sensor_id="glove_left"),
    })  # fmt: skip
    glove = sensor.model_copy(update={
        "label_id": f"{SID}-glove-contact", "evidence": Evidence.OBSERVED,
        "provenance": Provenance(source=Source.MODEL, model_version="contact-heuristic-1"),
        "confidence": 0.9,
    })  # fmt: skip
    truth = [_human(x) for x in scene.labels] + [sensor, glove]
    case = perturb_case(scene.session, truth, {"verb_swap": 0.5}, seed=1, o=ontology,
                        policy=policy)  # fmt: skip
    targets = {x.label_id for x in case.targets}
    assert targets and not targets & {sensor.label_id, glove.label_id}
    assert not set(case.truth) & {sensor.label_id, glove.label_id}
    assert {sensor.label_id, glove.label_id} <= {x.label_id for x in case.labels}


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
    """틀린 라벨 4개 중 3개 지적·1개 ok, 깨끗한 라벨 6개 중 1개 지적·5개 ok → 탐지 재현율 0.75,
    오지적률 1/6, ok 놓침률 = P(틀림 | AI ok) = 1/(1+5) = 1/6 (설계 7.2 정의), 틀린 라벨을 ok라 한
    비율 = 1/4 (참고). 게이트(재현율 ≥ 0.6, 오지적 ≤ 0.1, ok 놓침 ≤ 0.05)는 오지적·ok 놓침 둘이
    실패이고, block이 false라 차단하지 않는다. ECE: 확신도 0.9 판정 10개, 맞음 8개 → |0.8 - 0.9|."""
    case = _case(4, 6, ontology)
    values = ["wrong", "wrong", "suspect", "ok", "suspect", "ok", "ok", "ok", "ok", "ok"]
    verdicts: dict[LabelKey, AiVerdict] = {}
    for x, v in zip(case.targets, values, strict=True):
        extra = {"problems": ("wrong_verb",)} if v in ("wrong", "suspect") else {}
        verdicts[(SID, x.label_id)] = make_ai_verdict(
            x.label_id, v, kind=x.kind, confidence=0.9, **extra
        )
    judged = Judged(verdicts, (), (), 2.0)
    report = score(
        [case], judged, policy, mode="perturb", golden_version="g1", same_model_as_actions=False
    )
    assert report.wrong_recall == pytest.approx(0.75)
    assert report.false_flag_rate == pytest.approx(1 / 6)
    assert report.ok_miss_rate == pytest.approx(1 / 6)
    assert report.wrong_called_ok_rate == pytest.approx(0.25)
    assert any(f.startswith("ok 놓침률") for f in report.gate.failures)
    assert report.ece == pytest.approx(0.1)
    assert not report.gate.passed and len(report.gate.failures) == 2 and not report.gate.block
    assert report.by_error["verb_swap"].flagged == 3 and report.by_error["clean"].flagged == 1
    # 판정이 하나도 없으면 지표는 자료 없음이고 게이트는 통과로 보지 않는다
    empty = score(
        [case], Judged({}, (), (), 0.0), policy, mode="perturb", golden_version="g1",
        same_model_as_actions=False,
    )  # fmt: skip
    assert empty.wrong_recall == 0.0 and empty.by_error["verb_swap"].unjudged == 4
    assert empty.ok_miss_rate is None  # AI ok가 없으면 분모 0 (자료 없음 → 게이트 실패)
    assert not empty.gate.passed and empty.units_per_s is None


def test_false_events_of_two_golden_sessions_are_scored_separately(
    policy: AiReviewPolicy, ontology: Ontology
) -> None:
    """골든 세션 둘의 거짓 이벤트는 ID가 다르고, 판정도 세션별로 따로 센다.

    정답 근거: 판정기가 g1의 거짓 이벤트는 모두 wrong(삭제 제안), g2의 것은 모두 ok라 하면
    false_event 지적은 g1 수, ok는 g2 수다. 예전에는 ID가 세션과 무관하게 `pert-ev-0000…`이고 판정을
    라벨 ID로만 모아, g2에서 놓친 거짓 이벤트가 g1의 wrong으로 '잡힌' 것이 되고 재현율이 부풀었다.
    """
    scene = make_scene()
    truth = [_human(x) for x in scene.labels]
    other = scene.session.model_copy(update={"session_id": "g2"})
    truth2 = [
        x.model_copy(update={"session_id": "g2", "label_id": f"g2-{x.label_id}"}) for x in truth
    ]
    rates = {"false_event": 0.5}
    cases = [
        perturb_case(scene.session, truth, rates, seed=1, o=ontology, policy=policy),
        perturb_case(other, truth2, rates, seed=1, o=ontology, policy=policy),
    ]
    fakes = [[x for x in c.targets if c.truth[x.label_id].error == "false_event"] for c in cases]
    assert fakes[0] and fakes[1]
    assert not {x.label_id for x in fakes[0]} & {x.label_id for x in fakes[1]}
    verdicts: dict[LabelKey, AiVerdict] = {}
    for case, fake, value in zip(cases, fakes, ("wrong", "ok"), strict=True):
        sid = case.session.session_id
        for x in case.targets:
            v = value if x in fake else "ok"
            extra = {"problems": ("false_event",)} if v == "wrong" else {}
            verdicts[(sid, x.label_id)] = make_ai_verdict(x.label_id, v, kind=x.kind, **extra)
    report = score(
        cases, Judged(verdicts, (), (), 1.0), policy, mode="perturb", golden_version="g",
        same_model_as_actions=False,
    )  # fmt: skip
    stat = report.by_error["false_event"]
    assert (stat.total, stat.flagged, stat.ok) == (
        len(fakes[0]) + len(fakes[1]), len(fakes[0]), len(fakes[1]),
    )  # fmt: skip
    assert report.wrong_recall == pytest.approx(len(fakes[0]) / stat.total)


def test_eval_ignores_selection_mode_and_counts_only_chosen_groups(
    tmp_path: Path, policy: AiReviewPolicy, ontology: Ontology
) -> None:
    """평가는 선택 모드(flagged)를 쓰지 않고, --groups면 그 묶음 창에 든 대상만 센다.

    정답 근거: selection은 판정 버전(digest) 밖이라 같은 버전인데, 평가 경로에는 우선순위 사유
    구간이 없어 flagged면 감사 표본 창만 남는다 (예전: 단위 0개, 모든 대상 미판정, 재현율 0).
    대상은 모두 판정해야 분모가 맞다. --groups temporal이면 공간 대상(박스)은 분모에서 빠진다
    (예전: 미판정으로 남아 재현율이 1.0에서 떨어지고 오지적률 분모도 커졌다). 오라클이면 탐지
    재현율 1, 미판정 0이다.
    """
    db = db_scene(tmp_path, n_units=3)
    truth = [_human(x) for x in db.scene.labels]
    rates = {"verb_swap": 0.5, "class_swap": 0.5}
    flagged = with_policy(policy, selection={"mode": "flagged", "audit_ratio": 0.0})
    assert flagged.verdict_version == policy.verdict_version
    case = perturb_case(db.scene.session, truth, rates, seed=3, o=ontology, policy=flagged)
    assert {t.error for t in case.truth.values()} >= {"verb_swap", "class_swap"}
    judged = _judge(db, [case], OracleJudge(truth), flagged, tmp_path)
    report = score(
        [case], judged, flagged, mode="perturb", golden_version="g1", same_model_as_actions=False
    )
    assert report.units > 0 and report.wrong_recall == 1.0
    assert all(s.unjudged == 0 and s.unplanned == 0 for s in report.by_error.values())

    temporal = restrict_groups(case, policy, {AiGroup.TEMPORAL, AiGroup.TEMPORAL_LONG})
    kinds = {x.kind for x in temporal.targets}
    assert (
        temporal.targets
        and "box_track" not in kinds
        and set(temporal.truth) == {x.label_id for x in temporal.targets}
    )
    options = RunOptions(workers=2, groups=frozenset({AiGroup.TEMPORAL, AiGroup.TEMPORAL_LONG}))
    work = tmp_path / "work-temporal"
    work.mkdir()
    inp = eval_input(db.engine, db.labeling, temporal, policy, db.privacy, work, raw_bucket=RAW)
    narrowed = judge_cases([inp], OracleJudge(truth), policy, options=options)
    only = score(
        [temporal], narrowed, policy, mode="perturb", golden_version="g1",
        same_model_as_actions=False,
    )  # fmt: skip
    assert "class_swap" not in only.by_error and only.wrong_recall == 1.0
    assert all(s.unjudged == 0 and s.unplanned == 0 for s in only.by_error.values())


def test_suggestion_exact_rate_counts_only_attainable_suggestions(
    tmp_path: Path, policy: AiReviewPolicy, ontology: Ontology
) -> None:
    """제안 정확 일치는 제안으로 표현할 수 있는 오류만 분모에 넣고, 제안 키 필드로 비교한다.

    정답 근거: 행동↔사이 구간 제안의 새 페이로드는 action_id(`h-<손>-<시작>`)·접근 시각·접촉 시각을
    정할 수 없고, 손(hand_swap)·경계는 제안 키가 아니다. 정답을 아는 오라클이면 표현할 수 있는
    오류(action_to_gap·verb_swap)의 일치율은 1이고, hand_swap은 분모에서 빠진다(unattainable).
    """
    db = db_scene(tmp_path, n_units=3)
    truth = [_human(x) for x in db.scene.labels]
    rates = {"action_to_gap": 0.4, "verb_swap": 0.3, "hand_swap": 0.3}
    case = perturb_case(db.scene.session, truth, rates, seed=4, o=ontology, policy=policy)
    errors = {t.error for t in case.truth.values()}
    assert {"action_to_gap", "hand_swap", "verb_swap"} <= errors, errors
    judged = _judge(db, [case], OracleJudge(truth), policy, tmp_path)
    report = score(
        [case], judged, policy, mode="perturb", golden_version="g1", same_model_as_actions=False
    )
    gap = report.by_error["action_to_gap"]
    assert gap.flagged == gap.total and gap.expected == gap.total and gap.exact == gap.expected
    verb = report.by_error["verb_swap"]
    assert verb.expected == verb.total and verb.exact == verb.expected
    swapped = report.by_error["hand_swap"]
    assert swapped.expected == 0 and swapped.unattainable == swapped.total
    assert report.suggestion_exact_rate == 1.0
