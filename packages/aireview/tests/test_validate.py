"""응답 재검사·제안 완성 테스트 (`dlp_aireview.validate`·`suggest`, WP17.2, ADR 0040 3.2).

작은 요청(행동·사이 구간·이벤트·손 상태 4개, 개체 2개)에 손으로 만든 응답을 넣어 본다.
- 목록 밖·중복·빠진 번호 (빠진 대상은 unsure + response_invalid, answered=False)
- 온톨로지 밖 값, 개체표 밖 번호, 종류가 받지 않는 키, 헛제안, 타임라인 종류 삭제 → 항목 위반
- `check_label`에 걸리는 제안 → 제안만 버리고 판정은 남긴다
- 재시도 조건(파싱 실패 또는 위반 비율), 부분 수용
- `reason_guard` (URL·숫자열·프레임 참조·금지어·글자 수)
- gap→action 제안의 새 action_id는 `h-<손>-<시작>`
- 해시할 수 없는 값·bool·실수 제안 값도 예외 없이 위반, 기록 문장에 모델이 쓴 값이 없음
- 근거 프레임 번호는 단위 번호(문맥 초과로 줄인 요청도), 누락 후보 검사(범위·온톨로지·겹침·이유
  검사)
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from dlp_aireview.policy import AiReviewPolicy
from dlp_aireview.schema import SPATIAL_FIELDS, TEMPORAL_FIELDS
from dlp_aireview.suggest import NO_CHANGE, materialize
from dlp_aireview.validate import RequestView, reason_guard, validate_answer
from dlp_schema.aireview import AiGroup, AiProblem, AiVerdictValue
from dlp_schema.labels import (
    ActionPayload,
    BoxKeyframe,
    BoxTrackPayload,
    EventPayload,
    GapPayload,
    Hand,
    HandStatePayload,
    LabelRecord,
    Provenance,
    Source,
)
from dlp_schema.ontology import Ontology
from dlp_schema.testing import FIXED_TIME

ENTITIES = (("cup_01", "cup"), ("sink_01", "sink"))


def _label(label_id: str, payload: Any, start: int, end: int, **kw: Any) -> LabelRecord:
    """모델 출처 라벨."""
    data: dict[str, Any] = {
        "label_id": label_id, "session_id": "s001", "t_start_ms": start, "t_end_ms": end,
        "ontology_version": "1.0.0",
        "provenance": Provenance(source=Source.MODEL, model_version="m"), "confidence": 0.7,
        "created_at": FIXED_TIME, "payload": payload,
    }  # fmt: skip
    data.update(kw)
    return LabelRecord.model_validate(data)


ACTION = _label(
    "a1",
    ActionPayload(
        action_id="act-1", hand=Hand.RIGHT, verb="grasp", target_id="cup_01",
        t_approach_ms=1_000, t_contact_start_ms=1_300, t_contact_end_ms=1_800, t_end_ms=2_000,
    ),
    1_000, 2_000,
)  # fmt: skip
GAP = _label("g1", GapPayload(hand=Hand.RIGHT, gap_type="idle"), 2_000, 3_000)
EVENT = _label("e1", EventPayload(event_type="slip", severity=2), 2_500, 2_900)
HAND = _label(
    "h1",
    HandStatePayload(
        hand=Hand.RIGHT, contact_target_kind="object", target_id="cup_01", grasp_type="power",
        role="active",
    ),
    1_300, 1_800,
)  # fmt: skip
VIEW = RequestView(AiGroup.TEMPORAL, (ACTION, GAP, EVENT, HAND), tuple(range(6)), ENTITIES)


def _s(**values: Any) -> dict[str, Any]:
    """키 전부를 가진 시간 제안 객체 (나머지는 null, delete False)."""
    return {k: values.get(k, False if k == "delete" else None) for k in TEMPORAL_FIELDS}


def _item(i: int, v: str = "wrong", p: tuple[str, ...] = ("other",), **kw: Any) -> dict[str, Any]:
    """응답 항목."""
    return {"i": i, "v": v, "p": list(p), "r": kw.pop("r", "물체가 다르다"), "c": 0.8, "f": 1} | kw


def _answer(ok: list[int], items: list[dict[str, Any]], view: RequestView = VIEW, **kw: Any) -> Any:
    """응답 원문을 재검사한다."""
    policy, o = kw["policy"], kw["ontology"]
    return validate_answer(json.dumps({"ok": ok, "items": items, "missing": []}), view, o, policy)


def test_ok_and_valid_items_are_accepted(policy: AiReviewPolicy, ontology: Ontology) -> None:
    """맞는 번호는 ok, 지적 항목은 제안을 완성 페이로드로 만든다 (동사 rub, 대상 e1 = sink_01)."""
    s = _s(verb="rub", target=1)
    got = _answer(
        [1, 2, 3],
        [_item(0, p=("wrong_verb", "wrong_target"), s=s)],
        policy=policy,
        ontology=ontology,
    )
    assert got.parsed and got.invalid == 0 and not got.violations
    first = got.items[0]
    assert first.verdict is AiVerdictValue.WRONG and first.frame == 1
    assert first.suggestion is not None and first.suggestion.op == "set"
    payload = first.suggestion.payload
    assert isinstance(payload, ActionPayload)
    assert (payload.verb, payload.target_id, payload.action_id) == ("rub", "sink_01", "act-1")
    assert [d.verdict for d in got.items[1:]] == [AiVerdictValue.OK] * 3
    assert all(d.reason == "AI: 문제 없음" for d in got.items[1:])


def test_number_violations_and_missing(policy: AiReviewPolicy, ontology: Ontology) -> None:
    """목록 밖·중복 번호는 위반이다.

    빠진 번호와 중복 번호는 unsure + response_invalid(answered=False)다.
    """
    got = _answer([0, 0, 7], [_item(2)], policy=policy, ontology=ontology)
    assert got.parsed
    assert any("목록 밖" in v.text for v in got.violations)
    assert any("두 번" in v.text for v in got.violations)
    assert any("#1가 빠짐" in v.text or "#3가 빠짐" in v.text for v in got.violations)
    for k in (0, 1, 3):
        d = got.items[k]
        assert (d.verdict, d.problems, d.answered) == (
            AiVerdictValue.UNSURE, (AiProblem.RESPONSE_INVALID,), False,
        )  # fmt: skip
    assert got.items[2].verdict is AiVerdictValue.WRONG
    assert got.invalid == 4  # 목록 밖 1 + 중복 1 + 빠짐 2


@pytest.mark.parametrize(
    ("item", "message"),
    [
        (_item(0, s=_s(verb="dance")), "목록 밖 verb"),
        (_item(0, s=_s(target=5)), "개체표 밖"),
        (_item(2, p=("false_event",), s=_s(gap_type="idle")), "쓸 수 없는 제안 키"),
        (_item(0, s=_s(delete=True)), "삭제 제안"),
        (_item(1, s=_s(gap_type="idle")), NO_CHANGE),
        (_item(0, p=("wrong_class",)), "쓸 수 없는 문제 유형"),
        (_item(0, p=()), "문제 유형이 필요"),
        (_item(0, r="bad"), "한국어"),
        (_item(0, f=9), "보낸 프레임이 아니다"),
        (_item(0, v="ok"), "ok 목록"),
        (_item(0, v="maybe"), "판정 값"),
        (_item(0, s={"verb": "rub"}), "필수"),
    ],
)
def test_item_violations_downgrade_to_unsure(
    item: dict[str, Any], message: str, policy: AiReviewPolicy, ontology: Ontology
) -> None:
    """항목 위반: 그 대상은 unsure + response_invalid로 낮추고 위반 문장을 남긴다."""
    got = _answer([k for k in range(4) if k != item["i"]], [item], policy=policy, ontology=ontology)
    if message == "필수":  # 제안 객체의 키가 빠져도 받아 준다 (프롬프트 강제 서버 호환)
        assert got.invalid == 0
        return
    assert any(message in v.text for v in got.violations), got.violations
    d = got.items[item["i"]]
    assert d.verdict is AiVerdictValue.UNSURE and not d.answered


def test_ontology_check_failure_drops_only_the_suggestion(
    policy: AiReviewPolicy, ontology: Ontology
) -> None:
    """`check_label`에 걸리는 제안(관련 행동이 필요한 이벤트로 바꾸기)은 제안만 버린다.

    판정은 남긴다.
    """
    got = _answer(
        [0, 1, 3], [_item(2, p=("wrong_event_type",), s=_s(event_type="drop"))],
        policy=policy, ontology=ontology,
    )  # fmt: skip
    d = got.items[2]
    assert got.invalid == 0 and d.verdict is AiVerdictValue.WRONG
    assert d.suggestion is None and d.suggestion_error and "related_action_id" in d.suggestion_error


def test_kind_change_and_delete_suggestions(policy: AiReviewPolicy, ontology: Ontology) -> None:
    """gap→action: 새 action_id = h-<손>-<시작>, 접근·끝 = 라벨 구간. action→gap: 같은 손 gap.
    이벤트 삭제: 페이로드 없는 delete."""
    s_action = _s(kind="action", verb="push", target=0)
    s_gap = _s(kind="gap", gap_type="idle")
    got = _answer(
        [3],
        [
            _item(0, p=("should_be_gap",), s=s_gap),
            _item(1, p=("should_be_action",), s=s_action),
            _item(2, p=("false_event",), s=_s(delete=True)),
        ],
        policy=policy, ontology=ontology,
    )  # fmt: skip
    assert got.invalid == 0, got.violations
    to_gap, to_action, delete = (got.items[k].suggestion for k in range(3))
    assert to_gap is not None and to_gap.op == "change_kind"
    assert to_gap.payload == GapPayload(hand=Hand.RIGHT, gap_type="idle")
    assert to_action is not None and isinstance(to_action.payload, ActionPayload)
    assert to_action.payload.action_id == "h-right-2000"
    assert (to_action.payload.t_approach_ms, to_action.payload.t_end_ms) == (2_000, 3_000)
    assert (to_action.payload.verb, to_action.payload.target_id) == ("push", "cup_01")
    assert delete is not None and delete.op == "delete" and delete.payload is None


def test_materialize_clears_dependent_fields(ontology: Ontology) -> None:
    """종속 필드 정리: 손 상태 접촉 대상을 none으로 바꾸면 대상·파지를 비운다.

    접촉 동사끼리 바꾸면 접촉 시각은 그대로다 (v1에는 비접촉 원시 동작이 없다). 바꾸는 것이 없으면
    NO_CHANGE, 필요한 값이 빠진 종류 변경은 사유 문자열이다.
    """
    made = materialize(ACTION, {"verb": "push"}, ENTITIES, ontology)
    assert not isinstance(made, str) and isinstance(made.payload, ActionPayload)
    assert made.payload.t_contact_start_ms == 1_300
    hand = materialize(HAND, {"contact_kind": "none"}, ENTITIES, ontology)
    assert not isinstance(hand, str) and isinstance(hand.payload, HandStatePayload)
    assert (hand.payload.target_id, hand.payload.grasp_type) == (None, None)
    assert materialize(ACTION, {"verb": "grasp"}, ENTITIES, ontology) == NO_CHANGE
    assert isinstance(materialize(ACTION, {"kind": "gap"}, ENTITIES, ontology), str)


def test_spatial_hand_and_class_suggestions(policy: AiReviewPolicy, ontology: Ontology) -> None:
    """공간 제안: 박스 클래스 바꾸기, hand는 손 키포인트에만."""
    box = _label(
        "b1", BoxTrackPayload(entity_id="cup_01", class_id="cup",
                              keyframes=(BoxKeyframe(t_ms=0, x=1, y=1, w=5, h=5),)),
        0, 0, stream_id="bodycam",
    )  # fmt: skip
    view = RequestView(AiGroup.SPATIAL, (box,), tuple(range(4)), ())

    def sp(**values: Any) -> dict[str, Any]:
        return {k: values.get(k, False if k == "delete" else None) for k in SPATIAL_FIELDS}

    kw: dict[str, Any] = {"policy": policy, "ontology": ontology}
    ok = _answer([], [_item(0, p=("wrong_class",), s=sp(class_id="bucket"))], view, **kw)
    assert ok.invalid == 0 and ok.items[0].suggestion is not None
    bad = _answer([], [_item(0, p=("wrong_class",), s=sp(hand="left"))], view, **kw)
    assert any("hand" in v.text for v in bad.violations)


def test_retry_condition_and_parse_failure(policy: AiReviewPolicy, ontology: Ontology) -> None:
    """파싱 실패, 또는 위반 비율이 retry_invalid_ratio(0.5)를 넘을 때만 다시 묻는다."""
    ratio = policy.response.retry_invalid_ratio
    broken = validate_answer("이건 JSON이 아니다", VIEW, ontology, policy)
    assert not broken.parsed and broken.retry_worthy(4, ratio) and not broken.items
    two_bad = _answer([0, 1], [], policy=policy, ontology=ontology)  # 2/4 빠짐 = 0.5
    assert two_bad.invalid == 2 and not two_bad.retry_worthy(4, ratio)
    three_bad = _answer([0], [], policy=policy, ontology=ontology)
    assert three_bad.retry_worthy(4, ratio)
    extra_key = validate_answer(
        json.dumps({"ok": [], "items": [], "missing": [], "x": 1}), VIEW, ontology, policy
    )
    assert not extra_key.parsed


@pytest.mark.parametrize(
    "text",
    [
        "https://example.com 참고",
        "s3://bucket 참고",
        "메일 a@b",
        "번호 12345 보임",
        "F3에서 잡는다",
        "프레임C1 보면",
        "얼굴이 보인다",
        "가" * 61,
    ],
)
def test_reason_guard_replaces_unsafe_reasons(text: str, policy: AiReviewPolicy) -> None:
    """URL·@·숫자 4자리 이상·프레임/조각 참조·금지어·글자 수 초과는 고정 문장으로 바꾼다.

    고정 문장은 문제 유형에서 만든다.
    """
    out, hit = reason_guard(text, (AiProblem.WRONG_VERB,), policy)
    assert hit and out == "AI: 동사가 다름"
    clean, hit = reason_guard("  걸레로   닦는다 ", (AiProblem.WRONG_VERB,), policy)
    assert (clean, hit) == ("걸레로 닦는다", False)


def test_guarded_reason_keeps_the_verdict(policy: AiReviewPolicy, ontology: Ontology) -> None:
    """이유가 안전 검사에 걸려도 판정은 유지하고 이유만 바꾼다 (위반으로 세지 않는다)."""
    got = _answer(
        [1, 2, 3],
        [_item(0, p=("wrong_verb",), r="이름표 1234 보임")],
        policy=policy,
        ontology=ontology,
    )
    assert got.invalid == 0
    assert got.items[0].verdict is AiVerdictValue.WRONG and got.items[0].reason == "AI: 동사가 다름"


@pytest.mark.parametrize(
    ("index", "s"),
    [
        (0, _s(verb=["wipe"])),
        (0, _s(verb={"id": "rub"})),
        (0, _s(target=True)),
        (0, _s(target=1.0)),
        (0, _s(kind=["gap"], gap_type="idle")),
        (1, _s(gap_type=["idle"])),
        (2, _s(event_type={"x": 1})),
        (2, _s(severity=2.5)),
        (2, _s(severity=True)),
        (2, _s(delete="yes")),
        (3, _s(contact_kind=["object"])),
    ],
)
def test_unhashable_or_wrong_typed_suggestion_values_are_violations(
    index: int, s: dict[str, Any], policy: AiReviewPolicy, ontology: Ontology
) -> None:
    """제안 값이 목록·객체·실수·bool처럼 온톨로지 값이 될 수 없는 형태여도 예외 없이 항목 위반이다
    (서버가 enum을 강제하지 않거나 스키마를 프롬프트로만 줄 때)."""
    p = {0: "wrong_verb", 1: "wrong_gap_type", 2: "wrong_event_type", 3: "wrong_contact_kind"}
    item = _item(index, p=(p[index],), s=s)
    got = _answer([k for k in range(4) if k != index], [item], policy=policy, ontology=ontology)
    assert got.invalid == 1 and got.items[index].verdict is AiVerdictValue.UNSURE
    assert not got.items[index].answered


def test_violation_records_carry_no_model_written_values(
    policy: AiReviewPolicy, ontology: Ontology
) -> None:
    """기록용 위반 문장(`Violation.record`, 추가 전용 표에 남는다)에는 모델이 쓴 값·키 이름·목록 밖
    번호가 없다. 재시도 문장(`text`)에는 고칠 수 있게 값이 들어간다."""
    pii = "김철수 환자 302호 010-1234-5678"
    got = _answer(
        [1, 1012345678],
        [
            _item(0, v=pii),
            _item(2, p=(pii,)),
            _item(3, p=("wrong_contact_kind",), s={**_s(), pii: 1}),
        ],
        policy=policy, ontology=ontology,
    )  # fmt: skip
    records = " ".join(v.record for v in got.violations)
    texts = " ".join(v.text for v in got.violations)
    assert pii in texts and "1012345678" in texts
    assert pii not in records and "1012345678" not in records and "302" not in records
    broken = validate_answer(json.dumps({"ok": [], "items": [], pii: 1}), VIEW, ontology, policy)
    assert [v.record for v in broken.violations] == ["응답 구조 위반"]
    assert pii in broken.violations[0].text


def test_frame_numbers_are_unit_numbers_of_sent_frames(
    policy: AiReviewPolicy, ontology: Ontology
) -> None:
    """문맥 초과로 프레임을 줄인 요청(F0, F2, F4)의 f는 단위 프레임 번호다: 보낸 번호만 받는다."""
    halved = RequestView(AiGroup.TEMPORAL, VIEW.targets, (0, 2, 4), ENTITIES)
    rest = [1, 2, 3]
    kw: dict[str, Any] = {"policy": policy, "ontology": ontology}
    ok = _answer(rest, [_item(0, p=("wrong_verb",), f=4)], halved, **kw)
    assert ok.invalid == 0 and ok.items[0].frame == 4
    bad = _answer(rest, [_item(0, p=("wrong_verb",), f=1)], halved, **kw)
    assert bad.invalid == 1 and any("F0, F2, F4" in v.text for v in bad.violations)


# ---------------------------------------------------------------- 누락 후보 (max_findings > 0)

BOX = _label(
    "b-cup", BoxTrackPayload(entity_id="cup_01", class_id="cup",
                             keyframes=(BoxKeyframe(t_ms=0, x=100, y=100, w=40, h=40),
                                        BoxKeyframe(t_ms=900, x=100, y=100, w=40, h=40))),
    0, 900, stream_id="bodycam",
)  # fmt: skip
SPATIAL_VIEW = RequestView(
    AiGroup.SPATIAL, (BOX,), (0, 1, 2), (), frame_ms=(0, 300, 600), known=(BOX,),
    src_size=(400, 200),
)  # fmt: skip
TEMPORAL_VIEW = RequestView(
    AiGroup.TEMPORAL, VIEW.targets, (0, 1, 2), ENTITIES, frame_ms=(2_000, 2_600, 3_200),
    known=VIEW.targets,
)  # fmt: skip


def _findings_policy(policy: AiReviewPolicy) -> AiReviewPolicy:
    """누락 후보를 켠 정책 (응답 절만 바꾼 사본)."""
    response = policy.response.model_copy(update={"max_findings": 4})
    return policy.model_copy(update={"response": response})


def _missing(view: RequestView, missing: list[dict[str, Any]], **kw: Any) -> Any:
    """누락 후보가 든 응답 (대상은 모두 ok)."""
    raw = json.dumps({"ok": list(range(len(view.targets))), "items": [], "missing": missing})
    return validate_answer(raw, view, kw["ontology"], _findings_policy(kw["policy"]))


def test_spatial_findings_are_checked_not_crashing(
    policy: AiReviewPolicy, ontology: Ontology
) -> None:
    """공간 누락 후보: 범위 밖 상자, 온톨로지 밖·ID 형식이 아닌 클래스, 보내지 않은 프레임, 같은
    클래스 박스와 겹치는 후보는 버린다 (예외 없음). 이유는 안전 검사를 거친다."""
    kw: dict[str, Any] = {"policy": policy, "ontology": ontology}
    cup_box = [250, 500, 100, 200]  # 원 좌표 (100, 100, 40, 40) / (400, 200) → 정규화와 같다
    got = _missing(
        SPATIAL_VIEW,
        [
            {
                "class_id": "bucket",
                "f": 1,
                "box": [10, 20, 100, 30],
                "r": "김철수 씨 010-1234-5678",
            },
            {"class_id": "mop", "f": 0, "box": [10, 20, 1500, 30], "r": "빠진 물체가 있다"},
            {"class_id": "Not An Id!", "f": 0, "box": [10, 20, 30, 30], "r": "빠진 물체"},
            {"class_id": "mop", "f": 7, "box": [10, 20, 30, 30], "r": "빠진 물체"},
            {"class_id": "cup", "f": 1, "box": cup_box, "r": "컵이 빠졌다"},
            {"class_id": ["mop"], "f": 0, "box": [10, 20, 30, 30], "r": "빠진 물체"},
        ],
        **kw,
    )
    assert got.parsed and got.invalid == 0
    assert [(f.value, f.f0, f.box) for f in got.findings] == [("bucket", 1, (10, 20, 100, 30))]
    assert got.findings[0].reason == "AI: 빠진 물체"


def test_temporal_findings_drop_raw_access_and_already_labeled(
    policy: AiReviewPolicy, ontology: Ontology
) -> None:
    """시간 누락 후보: 원본 접근 이벤트(불편 반응)와 같은 유형 라벨이 이미 겹치는 구간은 버린다."""
    kw: dict[str, Any] = {"policy": policy, "ontology": ontology}
    got = _missing(
        TEMPORAL_VIEW,
        [
            {"event_type": "discomfort", "f0": 0, "f1": 1, "r": "불편해 보임"},
            {"event_type": "slip", "f0": 1, "f1": 2, "r": "미끄러짐"},  # EVENT(slip 2500~2900)
            {"event_type": "drop", "f0": 0, "f1": 1, "r": "http://evil 떨어뜨림"},
            {"event_type": "drop", "f0": 2, "f1": 1, "r": "거꾸로 된 구간"},
        ],
        **kw,
    )
    assert [(f.value, f.f0, f.f1, f.reason) for f in got.findings] == [
        ("drop", 0, 1, "AI: 빠진 이벤트")
    ]
