"""검수 결과 읽기 (dlp_schema.history.review_changes)."""

from __future__ import annotations

from typing import Any

from dlp_schema.history import review_changes
from dlp_schema.labels import LabelRecord, Provenance, Source, Verification, VerificationState
from dlp_schema.testing import FIXED_TIME, make_label

MODEL = Provenance(source=Source.MODEL, model_version="m1")
APPROVED = Verification(
    state=VerificationState.HUMAN_APPROVED, reviewer_id="r01", reviewed_at=FIXED_TIME
)
CORRECTED = Verification(
    state=VerificationState.HUMAN_CORRECTED, reviewer_id="r01", reviewed_at=FIXED_TIME
)
STATES = (VerificationState.HUMAN_APPROVED, VerificationState.HUMAN_CORRECTED)


def box(label_id: str, x: float, **overrides: Any) -> LabelRecord:
    payload = {
        "kind": "box_track", "entity_id": "cup_1", "class_id": "cup",
        "keyframes": [{"t_ms": 0, "x": x, "y": 0, "w": 10, "h": 10}],
    }  # fmt: skip
    if overrides.get("provenance") is MODEL:
        overrides["confidence"] = 0.9
    return make_label(payload, label_id=label_id, stream_id="bodycam", **overrides)


def changes(history: list[LabelRecord]) -> set[tuple[str, str, str | None]]:
    return {
        (c.change, c.label.label_id, c.origin.label_id if c.origin else None)
        for c in review_changes(history, STATES)
    }


def test_model_child_record_is_not_a_human_correction() -> None:
    # 원래 트랙(모델) → 3인칭 착용자 사본(모델, parent=원래). 사본은 사람이 고친 것이 아니다
    original = box("orig", 0, provenance=MODEL, verification=APPROVED)
    copy = box("copy", 5, provenance=MODEL, parent_label_id="orig", verification=APPROVED)
    assert changes([original, copy]) == {("accepted", "copy", "copy")}
    # 검수 전 모델 사본은 세지 않는다
    unreviewed = copy.model_copy(update={"verification": Verification()})
    assert changes([original, unreviewed]) == set()


def test_human_correction_and_addition_and_deletion() -> None:
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
