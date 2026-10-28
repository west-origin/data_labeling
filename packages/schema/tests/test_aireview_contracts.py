"""AI 1차 검수 계약 테스트 (`dlp_schema.aireview`, WP17.2, ADR 0040).

판정 값별 규칙(ok는 문제·제안 없음, response_invalid는 미응답만), 제안 형식(delete·change_kind·블러
금지), 단위 기록의 대상-판정 1:1, 노출 기록의 shown·hidden 규칙, `worst`·`valid_verdicts`의 두
프로필(표시 = 렌더 해시까지, 증가 = 렌더 무관)을 합성 객체로 확인한다. DB는 쓰지 않는다.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest
import sqlalchemy as sa
from pydantic import BaseModel, ValidationError

from dlp_schema.aireview import (
    AiArm,
    AiExposure,
    AiHumanAction,
    AiOutcome,
    AiOutcomeItem,
    AiProblem,
    AiReviewUnit,
    AiRunEvent,
    AiRunEventKind,
    AiSuggestion,
    AiVerdict,
    AiVerdictValue,
    split_verdict_ids,
    valid_verdicts,
    valid_verdicts_by_version,
    worst,
)
from dlp_schema.db import tables
from dlp_schema.labels import (
    GapPayload,
    Hand,
    LabelRecord,
    Provenance,
    Source,
    Verification,
    VerificationState,
)
from dlp_schema.review import ReviewReason
from dlp_schema.testing import (
    FIXED_TIME,
    action_payload,
    make_ai_run,
    make_ai_unit,
    make_ai_verdict,
    make_label,
)

H1 = "a" * 64
H2 = "b" * 64
VERSION = "aireview-0123456789ab+vlm-Qwen/Qwen3-VL-8B-Instruct"


def _model_label(label_id: str, **kw: Any) -> LabelRecord:
    """검수 전 모델 출처 행동 라벨."""
    data: dict[str, Any] = {
        "provenance": Provenance(source=Source.MODEL, model_version="m1"),
        "confidence": 0.9,
    }
    data.update(kw)
    return make_label(action_payload(action_id=label_id), label_id=label_id, **data)


def _verdict(label_id: str, value: str = "ok", **kw: Any) -> AiVerdict:
    """판정 하나 (wrong·suspect면 기본 문제 wrong_verb)."""
    data: dict[str, Any] = {
        "label_id": label_id,
        "kind": "action",
        "verdict": value,
        "reason_ko": "AI: 문제 없음" if value == "ok" else "동사가 다르다",
    }
    if value in ("wrong", "suspect"):
        data["problems"] = ("wrong_verb",)
    if value == "unsure":
        data["problems"] = ("not_visible",)
    data.update(kw)
    return AiVerdict.model_validate(data)


def _unit(unit_id: str, verdicts: list[AiVerdict], **kw: Any) -> AiReviewUnit:
    """판정 단위 하나 (바디캠, 렌더 해시 H1)."""
    data: dict[str, Any] = {
        "unit_id": unit_id,
        "run_id": "air-run-" + "0" * 24,
        "session_id": "s001",
        "version": VERSION,
        "group": "temporal",
        "stream_id": "bodycam",
        "t_start_ms": 0,
        "t_end_ms": 10_000,
        "target_label_ids": tuple(v.label_id for v in verdicts),
        "render_hash": H1,
        "sensor_sha256": "absent",
        "ontology_version": "1.0.0",
        "prompt_digest": H2,
        "outcome": "answered",
        "attempts": 1,
        "sub_requests": 1,
        "verdicts": tuple(verdicts),
        "created_at": FIXED_TIME,
    }
    data.update(kw)
    return AiReviewUnit.model_validate(data)


def test_verdict_rules_by_value() -> None:
    """ok는 문제·제안 없음, suspect·wrong은 문제 필요, unsure는 판단 불가 사유만,
    response_invalid는 미응답(answered=False)의 유일한 문제, 이유는 60자 이하, 블러 불가."""
    assert _verdict("l1").verdict is AiVerdictValue.OK
    with pytest.raises(ValidationError):
        _verdict("l1", "ok", problems=("wrong_verb",))
    with pytest.raises(ValidationError):
        _verdict("l1", "wrong", problems=())
    with pytest.raises(ValidationError):
        _verdict("l1", "unsure", problems=("wrong_verb",))
    with pytest.raises(ValidationError):
        _verdict("l1", "wrong", problems=("response_invalid",))
    missing = _verdict("l1", "unsure", problems=("response_invalid",), answered=False)
    assert not missing.answered
    with pytest.raises(ValidationError):
        _verdict("l1", "unsure", problems=("response_invalid",))  # answered=True인데 시스템 사유
    with pytest.raises(ValidationError):
        _verdict("l1", "unsure", problems=("not_visible",), answered=False)
    with pytest.raises(ValidationError):
        _verdict("l1", "wrong", reason_ko="가" * 61)
    with pytest.raises(ValidationError):
        _verdict("l1", kind="blur_track")
    with pytest.raises(ValidationError):
        _verdict("l1", "wrong", problems=("wrong_verb", "wrong_verb"))


def test_suggestion_shapes() -> None:
    """delete는 페이로드 없음, set·change_kind는 페이로드 필요, change_kind는 action↔gap만,
    set은 원 라벨과 같은 종류, 블러 페이로드 금지."""
    gap = GapPayload(hand=Hand.RIGHT, gap_type="idle")
    to_gap = AiSuggestion(op="change_kind", fields=("kind", "gap_type"), payload=gap)
    assert _verdict("l1", "wrong", problems=("should_be_gap",), suggestion=to_gap).suggestion
    with pytest.raises(ValidationError):
        AiSuggestion(op="delete", fields=("delete",), payload=gap)
    with pytest.raises(ValidationError):
        AiSuggestion(op="set", fields=("verb",))
    blur = {
        "kind": "blur_track",
        "target": "face",
        "keyframes": [{"t_ms": 0, "x": 0, "y": 0, "w": 1, "h": 1}],
    }
    with pytest.raises(ValidationError):
        AiSuggestion.model_validate({"op": "set", "fields": ["target"], "payload": blur})
    with pytest.raises(ValidationError):
        AiSuggestion.model_validate(
            {
                "op": "change_kind",
                "fields": ["kind"],
                "payload": {"kind": "event", "event_type": "drop"},
            }
        )
    set_gap = AiSuggestion(op="set", fields=("gap_type",), payload=gap)
    with pytest.raises(ValidationError):
        _verdict("l1", "wrong", suggestion=set_gap)  # action 라벨에 gap set 제안


def test_unit_requires_one_verdict_per_target_in_order() -> None:
    """대상마다 판정 정확히 하나(대상 순서), 대상과 맥락은 겹치지 않음, fallback은 모두 미응답."""
    unit = _unit("air-" + "1" * 24, [_verdict("l1"), _verdict("l2", "wrong")])
    assert unit.target_label_ids == ("l1", "l2")
    with pytest.raises(ValidationError):
        _unit("air-" + "1" * 24, [_verdict("l1")], target_label_ids=("l1", "l2"))
    with pytest.raises(ValidationError):
        _unit("air-" + "1" * 24, [_verdict("l1")], context_label_ids=("l1",))
    with pytest.raises(ValidationError):
        _unit("air-" + "1" * 24, [_verdict("l1")], outcome="fallback")
    with pytest.raises(ValidationError):
        _unit("bad-id", [_verdict("l1")])
    fallback = _verdict("l1", "unsure", problems=("response_invalid",), answered=False)
    assert _unit("air-" + "1" * 24, [fallback], outcome="fallback").outcome == "fallback"


def test_exposure_shown_only_for_shown_arm() -> None:
    """shown 집단만 보인 판정을 갖고, 같은 라벨이 shown·hidden 둘 다에 있을 수 없다."""
    uid = "air-" + "2" * 24
    base: dict[str, Any] = {
        "task_key": "label_studio:1", "session_id": "s001", "arm_key": "a1", "coverage": 1.0,
        "created_at": FIXED_TIME,
    }  # fmt: skip
    assert AiExposure(**base, arm=AiArm.SHOWN, shown={"l1": uid}).shown == {"l1": uid}
    assert AiExposure(**base, arm=AiArm.HOLDOUT, hidden={"l1": uid}).hidden
    with pytest.raises(ValidationError):
        AiExposure(**base, arm=AiArm.HOLDOUT, shown={"l1": uid})
    with pytest.raises(ValidationError):
        AiExposure(**base, arm=AiArm.SHOWN, shown={"l1": uid}, hidden={"l1": uid})


def test_worst_prefers_bad_then_latest() -> None:
    """wrong > suspect > unsure > ok, 같은 판정이면 최신 단위."""
    old = _unit("air-" + "a" * 24, [_verdict("l1", "suspect")])
    new = _unit(
        "air-" + "b" * 24, [_verdict("l1", "suspect")], created_at=FIXED_TIME + timedelta(hours=1)
    )
    ok = _unit("air-" + "c" * 24, [_verdict("l1")], created_at=FIXED_TIME + timedelta(hours=2))
    pairs = [(u.verdicts[0], u) for u in (old, new, ok)]
    assert worst(pairs)[1].unit_id == new.unit_id
    wrong = _unit("air-" + "d" * 24, [_verdict("l1", "wrong")])
    assert worst([*pairs, (wrong.verdicts[0], wrong)])[0].verdict is AiVerdictValue.WRONG
    with pytest.raises(ValueError):
        worst([])


def test_valid_verdicts_two_profiles() -> None:
    """표시 프로필은 렌더 해시까지 같아야 하고, 증가 프로필은 렌더를 보지 않는다.

    버전 밖, 이미 검수된 라벨, 운영 라벨에서 빠진 라벨의 판정은 없다. strict_context면 맥락도
    현재여야 한다.
    """
    labels = [_model_label("l1"), _model_label("l2"), _model_label("l3")]
    reviewed = labels[2].model_copy(
        update={
            "verification": Verification(
                state=VerificationState.HUMAN_APPROVED, reviewer_id="r1", reviewed_at=FIXED_TIME
            )
        }
    )
    current = [labels[0], labels[1], reviewed]
    fresh = _unit("air-" + "1" * 24, [_verdict("l1", "wrong")])
    stale_render = _unit("air-" + "2" * 24, [_verdict("l2", "suspect")], render_hash=H2)
    other_version = _unit(
        "air-" + "3" * 24, [_verdict("l1", "unsure")], version="aireview-ffffffffffff+vlm-x"
    )
    done = _unit("air-" + "4" * 24, [_verdict("l3", "wrong")])
    gone = _unit("air-" + "5" * 24, [_verdict("l9", "wrong")], context_label_ids=("zz",))
    with_context = _unit("air-" + "6" * 24, [_verdict("l2", "ok")], context_label_ids=("zz",))
    units = [fresh, stale_render, other_version, done, gone, with_context]

    shown = valid_verdicts(units, current, versions=[VERSION], render_hashes={"bodycam": H1})
    assert {k: (v.verdict.value, u) for k, (v, u) in shown.items()} == {
        "l1": ("wrong", fresh.unit_id),
        "l2": ("ok", with_context.unit_id),
    }
    escalate = valid_verdicts(units, current, versions=[VERSION], render_hashes=None)
    assert escalate["l2"][0].verdict is AiVerdictValue.SUSPECT  # 낡은 렌더의 지적도 증가에는 쓴다
    assert "l3" not in escalate and "l9" not in escalate
    strict = valid_verdicts(
        units, current, versions=[VERSION], render_hashes={"bodycam": H1}, strict_context=True
    )
    assert set(strict) == {"l1"}
    assert valid_verdicts(units, current, versions=[], render_hashes=None) == {}
    assert AiProblem.RESPONSE_INVALID.value == "response_invalid"


def test_valid_verdicts_by_version_and_panel_split() -> None:
    """판정단(ADR 0063): 버전별 유효한 판정은 `valid_verdicts`와 같은 조건(렌더·검수 상태)이고
    버전마다 따로 가장 나쁜 것이다. 갈림 = 버전 둘 이상이 판정했고 ok와 ok 아님이 섞임.

    정답 근거: l1은 A ok / B wrong(갈림), l2는 A suspect / B wrong(둘 다 ok 아님), l3은 A ok / B ok,
    l4는 A만 ok, l5는 B의 wrong이 낡은 렌더라 표시 프로필에서는 A ok 하나뿐(갈림 아님)이고 증가
    프로필에서는 갈림이다. l6은 검수된 라벨이라 판정이 유효하지 않다.
    """
    other = "aireview-0123456789ab+vlm-vllm-sr/d3-turbo"
    labels = [_model_label(f"l{i}") for i in range(1, 6)]
    reviewed = _model_label("l6").model_copy(
        update={
            "verification": Verification(
                state=VerificationState.HUMAN_APPROVED, reviewer_id="r1", reviewed_at=FIXED_TIME
            )
        }
    )
    a = _unit("air-" + "a" * 24, [_verdict("l1", "ok"), _verdict("l2", "suspect"),
              _verdict("l3", "ok"), _verdict("l4", "ok"), _verdict("l5", "ok"),
              _verdict("l6", "ok")])  # fmt: skip
    b = _unit("air-" + "b" * 24, [_verdict("l1", "wrong"), _verdict("l2", "wrong"),
              _verdict("l3", "ok"), _verdict("l6", "wrong")], version=other)  # fmt: skip
    b_stale = _unit("air-" + "c" * 24, [_verdict("l5", "wrong")], version=other, render_hash=H2)
    units = [a, b, b_stale]
    current = [*labels, reviewed]
    both = [VERSION, other]
    escalate = valid_verdicts_by_version(units, current, versions=both, render_hashes=None)
    values = {k: {ver: v.verdict.value for ver, (v, _) in d.items()} for k, d in escalate.items()}
    assert values == {
        "l1": {VERSION: "ok", other: "wrong"}, "l2": {VERSION: "suspect", other: "wrong"},
        "l3": {VERSION: "ok", other: "ok"}, "l4": {VERSION: "ok"},
        "l5": {VERSION: "ok", other: "wrong"},
    }  # fmt: skip
    assert split_verdict_ids(escalate) == {"l1", "l5"}
    shown = valid_verdicts_by_version(units, current, versions=both, render_hashes={"bodycam": H1})
    assert split_verdict_ids(shown) == {"l1"}
    # 버전 하나만 허용하면 갈림이 없다. 합친 가장 나쁜 판정은 valid_verdicts와 같다
    only_a = valid_verdicts_by_version(units, current, versions=[VERSION], render_hashes=None)
    assert split_verdict_ids(only_a) == set()
    merged = valid_verdicts(units, current, versions=both, render_hashes=None)
    rank = {"ok": 0, "unsure": 1, "suspect": 2, "wrong": 3}
    for label_id, per_version in escalate.items():
        top = max((v.verdict.value for v, _ in per_version.values()), key=rank.__getitem__)
        assert merged[label_id][0].verdict.value == top
    assert ReviewReason.AI_DISAGREEMENT.value == "ai_disagreement"


def test_ok_and_unsure_verdicts_never_carry_suggestions() -> None:
    """ok·unsure 판정에는 제안이 없다 (제안은 suspect·wrong만)."""
    gap = GapPayload(hand=Hand.RIGHT, gap_type="idle")
    to_gap = AiSuggestion(op="change_kind", fields=("kind", "gap_type"), payload=gap)
    with pytest.raises(ValidationError, match="ok"):
        _verdict("l1", "ok", suggestion=to_gap)
    with pytest.raises(ValidationError, match="unsure"):
        _verdict("l1", "unsure", suggestion=to_gap)


def test_valid_verdicts_skip_measurement_seeded_and_human_labels() -> None:
    """측정(블라인드·이중)·오류 삽입 레코드와 사람 라벨에는 판정을 주지 않는다 (지금 운영 라벨
    목록에 섞여 들어와도). 검수 전 자동 라벨만 판정이 유효하다."""
    plain = _model_label("l1")
    blind = _model_label("l2", measurement="blind")
    double = _model_label("l3", measurement="double")
    seeded = _model_label("l4", seeded_error=True)
    human = _model_label("l5", provenance=Provenance(source=Source.HUMAN), confidence=None)
    labels = [plain, blind, double, seeded, human]
    unit = _unit("air-" + "7" * 24, [_verdict(x.label_id, "wrong") for x in labels])
    for render in ({"bodycam": H1}, None):
        found = valid_verdicts([unit], labels, versions=[VERSION], render_hashes=render)
        assert set(found) == {"l1"}


def _table_examples() -> list[tuple[BaseModel, sa.Table]]:
    """AI 표마다 유효한 계약 객체 하나."""
    run = "air-run-" + "0" * 24
    item = AiOutcomeItem(label_id="l1", arm=AiArm.SHOWN, human=AiHumanAction.APPROVED)
    return [
        (make_ai_run(), tables.ai_review_runs),
        (
            AiRunEvent(session_id="s001", run_id=run, event=AiRunEventKind.STARTED,
                       reason="aborted:config", actor="ops", at=FIXED_TIME),
            tables.ai_review_run_events,
        ),
        (make_ai_unit("air-" + "1" * 24, [make_ai_verdict("l1")], endpoint="spark-01"),
         tables.ai_review_units),
        (
            AiExposure(task_key="label_studio:1", session_id="s001", assignment_id="as1",
                       arm=AiArm.SHOWN, arm_key="as1", coverage=1.0, created_at=FIXED_TIME),
            tables.ai_review_exposures,
        ),
        (
            AiOutcome(task_key="label_studio:1", session_id="s001", reviewer_id="rev01",
                      collected_at=FIXED_TIME, items=(item,)),
            tables.ai_review_outcomes,
        ),
    ]  # fmt: skip


@pytest.mark.parametrize(
    ("example", "table"), _table_examples(), ids=lambda v: getattr(v, "name", "")
)
def test_contract_rejects_values_longer_than_varchar_columns(
    example: BaseModel, table: sa.Table
) -> None:
    """VARCHAR(n) 열에 들어가는 계약 필드는 n자를 넘는 값을 계약 검증에서 거부한다.

    SQLite는 길이를 강제하지 않아 단위 테스트로 드러나지 않고, PostgreSQL에서야 DataError가 난다
    (예: 노출 arm_key). 값은 식별자 문자(영숫자)만 써서 길이로만 걸리게 한다.
    """
    data = example.model_dump()
    checked = 0
    for column in table.columns:
        length = getattr(column.type, "length", None)
        if length is None or column.name not in data:
            continue
        with pytest.raises(ValidationError):
            type(example).model_validate(data | {column.name: "a" * (length + 1)})
        checked += 1
    assert checked > 0


def test_valid_verdicts_display_needs_current_ontology_digest() -> None:
    """F-AIR-32: 표시 프로필에 온톨로지 내용 해시를 주면 같은 해시로 판정한 단위만 쓴다.

    같은 버전의 draft 덧붙이기 전 판정(다른 해시)과 내용 해시가 없는 이전 단위는 빠진다. 증가
    프로필(해시를 주지 않음)은 그대로 쓴다 (사람 검수를 늘리는 쪽).
    """
    current = [_model_label("l1"), _model_label("l2")]
    d_old, d_new = "a" * 64, "b" * 64
    old = _unit("air-" + "1" * 24, [_verdict("l1", "wrong")]).model_copy(
        update={"ontology_digest": d_old}
    )
    legacy = _unit("air-" + "2" * 24, [_verdict("l2", "suspect")])
    new = _unit("air-" + "3" * 24, [_verdict("l1", "ok")]).model_copy(
        update={"ontology_digest": d_new}
    )
    units = [old, legacy, new]
    shown = valid_verdicts(
        units, current, versions=[VERSION], render_hashes={"bodycam": H1}, ontology_digest=d_new
    )
    assert {k: (v.verdict.value, u) for k, (v, u) in shown.items()} == {"l1": ("ok", new.unit_id)}
    escalate = valid_verdicts(units, current, versions=[VERSION], render_hashes=None)
    assert escalate["l1"][0].verdict is AiVerdictValue.WRONG and "l2" in escalate
