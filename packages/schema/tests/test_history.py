"""검수 결과 읽기 (dlp_schema.history.review_changes).

정답 근거: 손으로 만든 작은 라벨 이력. 각 레코드의 출처·부모·삭제 표시가 곧 기대 결과를 정한다.
"""

from __future__ import annotations

from typing import Any

from dlp_schema.history import review_changes
from dlp_schema.labels import LabelRecord, Provenance, Source, Verification, VerificationState
from dlp_schema.testing import FIXED_TIME, make_label

# 모델 출처 (버전 m1), 검수 승인·수정 상태, 검수됨으로 볼 상태 집합
MODEL = Provenance(source=Source.MODEL, model_version="m1")
APPROVED = Verification(
    state=VerificationState.HUMAN_APPROVED, reviewer_id="r01", reviewed_at=FIXED_TIME
)
CORRECTED = Verification(
    state=VerificationState.HUMAN_CORRECTED, reviewer_id="r01", reviewed_at=FIXED_TIME
)
STATES = (VerificationState.HUMAN_APPROVED, VerificationState.HUMAN_CORRECTED)


def box(label_id: str, x: float, **overrides: Any) -> LabelRecord:
    """cup_1 박스 트랙 라벨 (x 좌표로 수정 여부를 구별). 모델 출처면 confidence 0.9를 채운다."""
    payload = {
        "kind": "box_track", "entity_id": "cup_1", "class_id": "cup",
        "keyframes": [{"t_ms": 0, "x": x, "y": 0, "w": 10, "h": 10}],
    }  # fmt: skip
    if overrides.get("provenance") is MODEL:
        overrides["confidence"] = 0.9
    return make_label(payload, label_id=label_id, stream_id="bodycam", **overrides)


def changes(history: list[LabelRecord]) -> set[tuple[str, str, str | None]]:
    """review_changes 결과를 (변화, 최종 라벨 ID, 원래 모델 라벨 ID) 집합으로 줄인다 (순서 무시)."""
    return {
        (c.change, c.label.label_id, c.origin.label_id if c.origin else None)
        for c in review_changes(history, STATES)
    }


def test_model_child_record_is_not_a_human_correction() -> None:
    """모델이 낸 자식 레코드(3인칭 착용자 사본)는 사람의 수정이 아니라 자기 검증 상태로 판단한다.

    시나리오: 원래 트랙(모델, 승인) → 사본(모델, parent=원래, 승인). 사본만 현재 라벨이므로
    accepted 하나.
    사본이 미검수면 states 밖이라 아무것도 세지 않는다.
    """
    # 원래 트랙(모델) → 3인칭 착용자 사본(모델, parent=원래). 사본은 사람이 고친 것이 아니다
    original = box("orig", 0, provenance=MODEL, verification=APPROVED)
    copy = box("copy", 5, provenance=MODEL, parent_label_id="orig", verification=APPROVED)
    assert changes([original, copy]) == {("accepted", "copy", "copy")}
    # 검수 전 모델 사본은 세지 않는다
    unreviewed = copy.model_copy(update={"verification": Verification()})
    assert changes([original, unreviewed]) == set()


def test_human_correction_and_addition_and_deletion() -> None:
    """네 가지 변화를 한 이력에서 모두 구별한다.

    - a → a-fix(사람): corrected, origin a
    - b → b-gone(사람 삭제): deleted, label·origin b
    - c(모델, 승인): accepted
    - new(사람, 부모 없음): added
    """
    a = box("a", 0, provenance=MODEL)
    a_fix = box("a-fix", 3, parent_label_id="a", verification=CORRECTED)
    b = box("b", 0, provenance=MODEL)
    b_gone = box("b-gone", 0, parent_label_id="b", retracted=True, verification=CORRECTED)
    c = box("c", 0, provenance=MODEL, verification=APPROVED)
    new = box("new", 9, verification=CORRECTED)
    assert changes([a, a_fix, b, b_gone, c, new]) == {
        ("corrected", "a-fix", "a"),
        ("deleted", "b", "b"),
        ("accepted", "c", "c"),
        ("added", "new", None),
    }


def test_sensor_label_is_not_counted_as_human_addition() -> None:
    """센서 출처 현재 라벨(states 안, 부모 없음)은 어떤 변화로도 세지 않는다.

    회귀 테스트: 예전에는 사슬 맨 앞이 모델이 아니라는 이유로 "added"(사람이 추가함)로 셌다.
    같은 이력의 사람 추가 라벨은 그대로 added다.
    """
    sensor = box(
        "glove",
        0,
        provenance=Provenance(source=Source.SENSOR, sensor_id="glove-l"),
        verification=APPROVED,
    )
    new = box("new", 9, verification=CORRECTED)
    assert changes([sensor, new]) == {("added", "new", None)}
