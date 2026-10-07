"""검수 운영 로직 (WP12) 단위 테스트."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from dlp_fixtures.actions import generate_action_scenario
from dlp_review.ops.assign import PlannedUnit, plan, plan_qa
from dlp_review.ops.measure import agreement, as_items, prelabel_bias
from dlp_review.ops.policy import ReviewOpsPolicy, load_policy
from dlp_review.ops.priority import Unit, flag_unit, unit_priority
from dlp_review.ops.sampling import Lot, draw_sample, judge, lots, sample_size
from dlp_review.ops.seeding import detected, seed_labels, seed_prefix
from dlp_schema.episode import current_labels, non_operational_ids
from dlp_schema.labels import (
    ActionPayload,
    BlurTrackPayload,
    BoxKeyframe,
    BoxTrackPayload,
    Hand,
    HandStatePayload,
    LabelRecord,
    Provenance,
    Source,
    Verification,
    VerificationState,
)
from dlp_schema.ontology import Ontology, load_ontology
from dlp_schema.review import AssignmentStatus, ReviewMode, ReviewReason
from dlp_schema.testing import FIXED_TIME, make_label

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def policy() -> ReviewOpsPolicy:
    """저장소의 실제 검수 운영 정책 (review.yaml + defaults.yaml review 비율)."""
    return load_policy(ROOT)


@pytest.fixture(scope="module")
def ontology() -> Ontology:
    """온톨로지 v1 (오류 삽입 class_swap 후보)."""
    return load_ontology(ROOT / "config/ontology/v1")


def model(label: LabelRecord, version: str, confidence: float) -> LabelRecord:
    """라벨을 모델 출처(model_version=version)와 신뢰도 confidence로 바꾼 사본."""
    return label.model_copy(
        update={
            "provenance": Provenance(source=Source.MODEL, model_version=version),
            "confidence": confidence,
        }
    )


def box(
    label_id: str, cls: str, x: float, start: int = 0, end: int = 1000, **kw: Any
) -> LabelRecord:
    """시험용 박스 트랙 라벨 (bodycam, 100 ms 간격 키프레임, 크기 50*50, y=10).

    x로 가로 위치를, start·end(ms)로 구간을 정한다. kw는 `make_label`에 넘긴다.
    """
    frames = tuple(BoxKeyframe(t_ms=t, x=x, y=10, w=50, h=50) for t in range(start, end + 1, 100))
    payload = BoxTrackPayload(entity_id=f"{cls}_{label_id}", class_id=cls, keyframes=frames)
    return make_label(
        payload, label_id=label_id, t_start_ms=start, t_end_ms=end, stream_id="bodycam", **kw
    )


# ---------------------------------------------------------------- 배정 비율


def _units(n: int) -> list[PlannedUnit]:
    """n개 공간 단위 (세션 s00000…, 우선순위 1~7 순환). 배정 비율 시험용."""
    return [
        PlannedUnit(Unit(f"s{i:05d}", "bodycam", "spatial", ("box_track",)), (), 1.0 + (i % 7))
        for i in range(n)
    ]


def test_assignment_ratios_follow_policy(policy: ReviewOpsPolicy) -> None:
    """완료 기준: 정책 YAML의 비율대로 배정된다 (이항 분포 4 표준편차 안)."""
    n = 20_000
    pool = [
        Unit("gold-1", "bodycam", "spatial", ("box_track",)),
        Unit("gold-2", None, "temporal", ("action",)),
    ]
    reviewers = [f"r{i}" for i in range(8)]
    out = plan(_units(n), reviewers, policy, seed=7, now=FIXED_TIME, seed_pool=pool)
    counts = Counter(a.mode for a in out)
    assert counts[ReviewMode.STANDARD] == n
    r = policy.ratios
    for mode, p in (
        (ReviewMode.BLIND, r.blind_task_ratio),
        (ReviewMode.DOUBLE, r.double_annotation_ratio),
        (ReviewMode.SEEDED_ERROR, r.seeded_error_task_ratio),
    ):
        sigma = math.sqrt(n * p * (1 - p))
        assert abs(counts[mode] - n * p) <= 4 * sigma, (mode, counts[mode], n * p)
    standard = {a.assignment_id: a.assignee for a in out if a.mode is ReviewMode.STANDARD}
    for a in out:
        if a.mode in (ReviewMode.BLIND, ReviewMode.DOUBLE):
            assert a.pair_id in standard and a.assignee != standard[a.pair_id]
        if a.mode is ReviewMode.SEEDED_ERROR:
            assert a.session_id in {"gold-1", "gold-2"}
    loads = Counter(a.assignee for a in out)
    assert max(loads.values()) - min(loads.values()) <= 1  # 고르게 나눔
    again = plan(_units(n), reviewers, policy, seed=7, now=FIXED_TIME, seed_pool=pool)
    assert [a.assignment_id for a in again] == [a.assignment_id for a in out]  # 결정적


def test_qa_ratio_and_senior(policy: ReviewOpsPolicy) -> None:
    """QA 배정이 qa_sample_ratio만큼(이항 분포 4 표준편차 안) 선임에게 가는지 본다.

    시나리오: 20,000개 단위의 표준 배정을 모두 끝난 것으로 두고 `plan_qa`. 선임은 "lead" 한 명.
    정답 근거: 정책 비율과 이항 분포 표준편차, 모든 QA 배정의 담당자가 "lead"이고 pair_id가 있다.
    """
    p = policy.model_copy(
        update={"reviewers": policy.reviewers.model_copy(update={"senior": ("lead",)})}
    )
    done = [
        a.model_copy(update={"status": AssignmentStatus.DONE})
        for a in plan(_units(20_000), ["r1", "r2"], p, seed=1, now=FIXED_TIME)
    ]
    qa = plan_qa(done, p, seed=1, now=FIXED_TIME)
    ratio = p.ratios.qa_sample_ratio
    assert abs(len(qa) - 20_000 * ratio) <= 4 * math.sqrt(20_000 * ratio * (1 - ratio))
    assert all(a.mode is ReviewMode.QA and a.assignee == "lead" and a.pair_id for a in qa)


# ---------------------------------------------------------------- 우선순위


def test_priority_reasons(policy: ReviewOpsPolicy) -> None:
    """우선순위 사유 네 가지가 모두 잡히고 단위 점수가 가중치 * 길이와 같은지 본다.

    시나리오: 같은 자리 다른 클래스 박스(두 모델 버전) → 불일치, 신뢰도 0.3 → 낮은 신뢰도,
    처음 보는 클래스(mop) → 새 객체, 장갑 세션의 신뢰도 0.6 접촉 → 접촉 불일치.
    정답 근거: 박스 구간 0~1000 ms(1초), 접촉 구간 100~600 ms(0.5초)로 계산한 가중 합.
    장갑 세션이 아니면 접촉 사유가 없고, 사유가 없으면 routine_priority.
    """
    a = model(box("a", "cup", 10), "det-1", 0.9)
    b = model(box("b", "bucket", 12), "det-2", 0.9)  # 같은 곳, 다른 클래스 → 불일치
    low = model(box("c", "cup", 200), "det-1", 0.3)  # 낮은 신뢰도
    new = model(box("d", "mop", 400), "det-1", 0.95)  # 처음 보는 클래스
    state = HandStatePayload(
        hand=Hand.RIGHT, contact_target_kind="object", target_id="cup_a", role="active"
    )
    touch = make_label(state, label_id="h1", t_start_ms=100, t_end_ms=600)
    touch = model(touch, "contact-1", 0.6)  # 장갑·영상 한쪽만
    labels = [a, b, low, new, touch]
    spans = flag_unit(labels, policy, known_classes={"cup", "bucket"}, glove_session=True)
    reasons = {s.reason: s for s in spans}
    assert set(reasons) == {
        ReviewReason.MODEL_DISAGREEMENT, ReviewReason.LOW_CONFIDENCE,
        ReviewReason.NEW_OBJECT, ReviewReason.CONTACT_MISMATCH,
    }  # fmt: skip
    assert set(reasons[ReviewReason.MODEL_DISAGREEMENT].label_ids) == {"a", "b"}
    assert reasons[ReviewReason.NEW_OBJECT].label_ids == ("d",)
    assert not flag_unit([touch], policy, known_classes=set(), glove_session=False)
    w = policy.priority.weights
    assert unit_priority(spans, policy) == pytest.approx(
        w["model_disagreement"] * 1.0
        + w["low_confidence"]
        + w["new_object"]
        + w["contact_mismatch"] * 0.5
    )
    assert unit_priority([], policy) == policy.priority.routine_priority


# ---------------------------------------------------------------- 표본 검수


def test_sampling_accepts_or_rejects_lot(policy: ReviewOpsPolicy) -> None:
    """표본 묶음 만들기·뽑기·합격 판정을 본다.

    시나리오: 신뢰도 0.95 박스 100개(+ 묶음 밖 0.5 하나) → 묶음 1개, 표본 10개(ratio 0.1).
    표본 검수 전 accepted=None, 표본 전부 승인 → 합격 + 나머지 90개 표본 검증,
    표본 2/10 수정 → 결함 비율 0.2 > 0.05 → 불합격.
    정답 근거: 정책 sampling 값과 같은 seed의 같은 표본(결정성).
    """
    sp = policy.sampling
    labels = [model(box(f"l{i:03d}", "cup", float(i)), "det-1", 0.95) for i in range(100)]
    labels.append(model(box("low", "cup", 0), "det-1", 0.5))  # 묶음 밖 (신뢰도 낮음)
    [lot] = lots(labels, sp)
    assert len(lot.label_ids) == 100 and sample_size(100, sp) == 10
    sample = draw_sample(lot, sp, seed=3)
    assert sample == draw_sample(lot, sp, seed=3) and len(sample) == 10
    assert judge(lot.label_ids, sample, labels, sp).accepted is None  # 아직 검수 전

    ok_review = Verification(
        state=VerificationState.HUMAN_APPROVED, reviewer_id="r1", reviewed_at=FIXED_TIME
    )
    approved = [
        x.model_copy(update={"verification": ok_review}) if x.label_id in sample else x
        for x in labels
    ]  # fmt: skip
    ok = judge(lot.label_ids, sample, approved, sp)
    assert ok.accepted and len(ok.to_verify) == 90

    fixed = approved + [
        box(f"fix{i}", "bucket", 0, parent_label_id=sample[i])
        for i in range(2)  # 표본 2/10 수정
    ]
    bad = judge(lot.label_ids, sample, fixed, sp)
    assert bad.accepted is False and bad.defects == 2 and bad.to_verify == ()


# ---------------------------------------------------------------- 오류 삽입


def _truth() -> list[LabelRecord]:
    """오류 삽입 정답 단위: 합성 행동 시나리오의 행동 라벨 + 박스 하나 + 얼굴 블러 하나.

    세 오류 종류(boundary_shift, class_swap, blur_deletion)의 후보가 모두 있다.
    """
    actions = [
        x for x in generate_action_scenario(0, session_id="gold").labels if x.kind == "action"
    ]
    frames = tuple(BoxKeyframe(t_ms=t, x=1, y=1, w=5, h=5) for t in (0, 500))
    blur = make_label(
        BlurTrackPayload(target="face", keyframes=frames),
        label_id="gold-blur", session_id="gold", t_start_ms=0, t_end_ms=500, stream_id="bodycam",
    )  # fmt: skip
    return [*actions, box("gold-box", "cup", 5, session_id="gold"), blur]


def test_seeded_task_injects_known_errors(policy: ReviewOpsPolicy, ontology: Ontology) -> None:
    """오류 삽입 사본이 규칙대로 만들어지고 운영 라벨에 섞이지 않는지 본다.

    정답 근거: 세 종류 오류가 하나씩 들어가고, 모든 사본은 seeded_error·parent 없음·배정 접두사 ID,
    블러 하나를 빼 사본 수 = 정답 수 - 1, 경계 이동량이 정책 범위 안, 클래스 교체 값이 원래와
    다르다. `current_labels`는 정답만 돌려준다.
    """
    truth = _truth()
    task = seed_labels(
        truth,
        assignment_id="gold:all:temporal:seeded_error",
        ontology=ontology,
        policy=policy.seeding,
        seed=1,
        now=FIXED_TIME,
    )
    assert {e.error_type for e in task.injected} == {
        "boundary_shift",
        "class_swap",
        "blur_deletion",
    }
    assert all(
        x.seeded_error
        and x.parent_label_id is None
        and x.label_id.startswith(seed_prefix("gold:all:temporal:seeded_error"))
        for x in task.labels
    )
    assert len(task.labels) == len(truth) - 1  # 블러 하나를 뺐다
    by_id = {x.label_id: x for x in task.labels}
    for e in task.injected:
        if e.error_type == "boundary_shift":
            shifted = by_id[e.seeded_label_id or ""]
            lo, hi = policy.seeding.boundary_shift_ms
            assert lo <= int(e.detail["shift_ms"]) <= hi
            assert (
                shifted.t_start_ms
                != next(x for x in truth if x.label_id == e.original_label_id).t_start_ms
                or e.detail["edge"] == "end"
            )
        if e.error_type == "class_swap":
            p = by_id[e.seeded_label_id or ""].payload
            assert getattr(p, str(e.detail["field"])) == e.detail["seeded"] != e.detail["original"]
    # 오류 삽입 레코드는 운영 라벨이 아니다
    assert current_labels([*truth, *task.labels]) == truth


def test_detection_of_seeded_errors(policy: ReviewOpsPolicy, ontology: Ontology) -> None:
    """검수자가 오류를 되돌리면 발견으로, 다른 검수자 결과는 세지 않는지 본다.

    시나리오: 아무것도 고치지 않으면 발견 0 → 블러를 배정 접두사 ID로 다시 그림, 클래스를 원래
    값으로, 경계를 원래 값 ±150 ms(허용 오차 200 ms 안)까지 되돌림 → r1 기준 모두 발견, r2 기준 0.
    고친 레코드도 오류 삽입 계보라 운영 라벨이 아니다 (`non_operational_ids`).
    """
    truth = _truth()
    task = seed_labels(
        truth, assignment_id="g:a", ontology=ontology, policy=policy.seeding, seed=1, now=FIXED_TIME
    )
    labels = [*truth, *task.labels]
    by_id = {x.label_id: x for x in labels}
    tol = policy.seeding.detect_tolerance_ms
    assert not any(detected(e, labels, "r1", tol, 0.5) for e in task.injected)  # 아무것도 안 고침

    human = Verification(
        state=VerificationState.HUMAN_CORRECTED, reviewer_id="r1", reviewed_at=FIXED_TIME
    )
    fixes: list[LabelRecord] = []
    for e in task.injected:
        original = by_id[e.original_label_id]
        base = {
            "provenance": Provenance(source=Source.HUMAN),
            "confidence": None,
            "seeded_error": True,
            "verification": human,
        }
        if e.error_type == "blur_deletion":
            # 수집은 오류 삽입 과제에서 새로 그린 레코드 ID에 배정의 사본 접두사를 붙인다
            fixes.append(
                original.model_copy(update={"label_id": f"{seed_prefix('g:a')}new-blur", **base})
            )
        elif e.error_type == "class_swap":
            # 검수자가 원래 값으로 되돌린다
            fixes.append(
                original.model_copy(
                    update={
                        "label_id": f"fix-{e.seeded_label_id}",
                        "parent_label_id": e.seeded_label_id,
                        **base,
                    }
                )
            )
        else:
            # 경계를 허용 오차 안(150 ms 차이)까지만 되돌린다
            o = original
            assert isinstance(o.payload, ActionPayload)
            start = (
                int(e.detail["original_ms"]) - 150 if e.detail["edge"] == "start" else o.t_start_ms
            )
            end = o.t_end_ms if e.detail["edge"] == "start" else int(e.detail["original_ms"]) + 150
            payload = o.payload.model_copy(update={"t_approach_ms": start, "t_end_ms": end})
            fixes.append(
                o.model_copy(
                    update={
                        "label_id": f"fix-{e.seeded_label_id}",
                        "parent_label_id": e.seeded_label_id,
                        "t_start_ms": start,
                        "t_end_ms": end,
                        "payload": payload,
                        **base,
                    }
                )
            )
    labels += fixes
    assert all(detected(e, labels, "r1", tol, 0.5) for e in task.injected)
    assert not any(
        detected(e, labels, "r2", tol, 0.5) for e in task.injected
    )  # 다른 검수자 결과는 세지 않음
    # 검수자가 고친 레코드도 오류 삽입 계보라 운영 라벨이 아니다
    assert {x.label_id for x in fixes} <= non_operational_ids(labels)


# ---------------------------------------------------------------- 일치도·편향


def test_agreement_and_prelabel_bias() -> None:
    """일치도·프리라벨 편향 계산을 본다.

    정답 근거: 같은 라벨끼리는 카파·구간 F1·경계 F1이 모두 1. 블라인드가 절반만 맞고 표준은 모델과
    같으면 편향 = 1 - F1(블라인드, 모델) > 0.
    """
    truth = [x for x in generate_action_scenario(0).labels if x.kind == "action"]
    items = as_items(truth)
    same = agreement(items, items, 200, 0.5)
    assert (same.kappa, same.segment_f1, same.boundary_f1) == (1.0, 1.0, 1.0) and same.pairs == len(
        items
    )
    # 블라인드 결과가 절반만 맞고 표준 결과는 모델과 같으면 편향은 양수
    half = [(k, c if i % 2 else "other", s, e) for i, (k, c, s, e) in enumerate(items)]
    bias = prelabel_bias(items, items, half, 200, 0.5)
    assert bias == pytest.approx(1.0 - agreement(half, items, 200, 0.5).segment_f1) and bias > 0


def test_lot_key_is_stable() -> None:
    """묶음 키 형식 `<세션>:<종류>:<모델 버전>`이 바뀌지 않는지 본다 (표본 시드 안정성)."""
    lot = Lot("s", "box_track", "det-1", ("a", "b"))
    assert lot.key == "s:box_track:det-1"


def test_blur_review_needs_privacy_reviewers_and_skips_qa(policy: ReviewOpsPolicy) -> None:
    """블러 검수 권한 검사와 블러 배정의 QA 제외를 본다.

    시나리오: 권한자 priv1은 통과, 일반 라벨러·미배정(None)은 AccessError.
    블러 단위 2,000개를 끝난 것으로 둬도 `plan_qa`는 빈 목록 (블러 검수는 일반 QA로 넘기지 않는다).
    """
    from dlp_review.ops.runner import AccessError, check_privacy_reviewers

    p = policy.model_copy(
        update={
            "reviewers": policy.reviewers.model_copy(
                update={"privacy": ("priv1",), "senior": ("lead",)}
            )
        }
    )
    check_privacy_reviewers(["priv1"], p)
    with pytest.raises(AccessError):
        check_privacy_reviewers(["labeler1"], p)
    with pytest.raises(AccessError):
        check_privacy_reviewers([None], p)  # 미배정도 막는다
    blur_units = [
        PlannedUnit(Unit(f"s{i}", "bodycam", "privacy", ("blur_track",)), (), 1.0)
        for i in range(2000)
    ]
    done = [
        a.model_copy(update={"status": AssignmentStatus.DONE})
        for a in plan(blur_units, ["priv1"], p, seed=1, now=FIXED_TIME)
    ]
    assert plan_qa(done, p, seed=1, now=FIXED_TIME) == []  # 블러 검수는 일반 QA로 넘기지 않는다


def test_untouched_seed_copies_do_not_count_as_detection(
    policy: ReviewOpsPolicy, ontology: Ontology
) -> None:
    """같은 대상(얼굴) 블러가 겹쳐 있을 때, 남은 사본만으로 빠진 블러를 찾은 것으로 세지 않는다."""
    frames = tuple(BoxKeyframe(t_ms=t, x=1, y=1, w=5, h=5) for t in (0, 500))
    blur = BlurTrackPayload(target="face", keyframes=frames)
    span: dict[str, Any] = {"session_id": "gold", "t_end_ms": 500, "stream_id": "bodycam"}
    truth = [make_label(blur, label_id=f"gold-blur{i}", **span) for i in range(2)]
    seeding = policy.seeding.model_copy(update={"errors_per_task": 1, "types": ("blur_deletion",)})
    task = seed_labels(
        truth, assignment_id="g:b", ontology=ontology, policy=seeding, seed=0, now=FIXED_TIME
    )
    [err] = task.injected
    labels = [*truth, *task.labels]
    assert not detected(err, labels, None, 200, 0.5) and not detected(err, labels, "r1", 200, 0.5)


def test_blur_deletion_credit_is_scoped_to_assignment_and_stream(
    policy: ReviewOpsPolicy, ontology: Ontology
) -> None:
    """회귀: 같은 세션의 다른 배정·다른 스트림에서 그린 블러로 블러 삭제 발견을 인정하지 않는다."""
    frames = tuple(BoxKeyframe(t_ms=t, x=1, y=1, w=5, h=5) for t in (0, 500))
    blur = BlurTrackPayload(target="face", keyframes=frames)
    span: dict[str, Any] = {"session_id": "gold", "t_end_ms": 500, "stream_id": "bodycam"}
    truth = [make_label(blur, label_id="gold-blur0", **span)]
    seeding = policy.seeding.model_copy(update={"errors_per_task": 1, "types": ("blur_deletion",)})
    task = seed_labels(
        truth, assignment_id="g:b", ontology=ontology, policy=seeding, seed=0, now=FIXED_TIME
    )
    [err] = task.injected
    assert err.detail["assignment_id"] == "g:b"
    human = Verification(
        state=VerificationState.HUMAN_CORRECTED, reviewer_id="r1", reviewed_at=FIXED_TIME
    )
    base = {"provenance": Provenance(source=Source.HUMAN), "seeded_error": True,
            "verification": human}  # fmt: skip

    def drawn(label_id: str, stream: str) -> LabelRecord:
        """검수자가 이 과제에서 새로 그린 것처럼 만든 블러 레코드 (ID·스트림 지정)."""
        return truth[0].model_copy(update={"label_id": label_id, "stream_id": stream, **base})

    other_assignment = drawn(f"{seed_prefix('g:c')}new-1", "bodycam")
    other_stream = drawn(f"{seed_prefix('g:b')}new-2", "third_person")
    labels = [*truth, *task.labels, other_assignment, other_stream]
    assert not detected(err, labels, "r1", 200, 0.5)
    assert detected(err, [*labels, drawn(f"{seed_prefix('g:b')}new-3", "bodycam")], "r1", 200, 0.5)
