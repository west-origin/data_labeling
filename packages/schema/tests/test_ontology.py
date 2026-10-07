from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from dlp_schema.ontology import Ontology, VerbLevel, find_ontology_dir, load_ontology
from dlp_schema.testing import action_payload, make_label
from dlp_schema.validation import OntologyViolationError, check_label, validate_label


def test_v1_matches_plan(ontology: Ontology) -> None:
    assert ontology.version == "1.0.0"
    assert len(ontology.body_parts) == 26
    assert {t for t, p in ontology.privacy_targets.items() if p.required} == {
        "face", "reflection", "shipping_label", "document", "screen", "photo", "address_sign",
    }  # fmt: skip
    primitives = {v for v, t in ontology.verbs.items() if t.level is VerbLevel.PRIMITIVE}
    assert len(primitives) == 11
    assert set(ontology.gap_types) == {"idle", "unknown", "out_of_scope"}
    assert all(t.domain in ontology.domains for t in ontology.tasks.values())
    assert ontology.tool_parts("mop") == {"mop_head", "handle"}
    assert ontology.objects["sink"].surface


def test_find_ontology_dir(repo: Path) -> None:
    root = repo / "config" / "ontology"
    assert find_ontology_dir(root, "1.0.0").name == "v1"
    with pytest.raises(FileNotFoundError):
        find_ontology_dir(root, "9.9.9")


def _write(tmp: Path, files: dict[str, str]) -> Path:
    for name, text in files.items():
        (tmp / name).write_text(text, encoding="utf-8")
    return tmp


def test_loader_rejects_duplicate_keys_across_files(tmp_path: Path) -> None:
    _write(tmp_path, {"a.yaml": "verbs: {}\n", "b.yaml": "verbs: {}\n"})
    with pytest.raises(ValueError, match="중복"):
        load_ontology(tmp_path)


def test_cross_references_are_checked(ontology: Ontology) -> None:
    data: dict[str, Any] = ontology.model_dump(mode="json")
    data["tasks"]["bad_task"] = {"domain": "gardening", "ko": "정원"}
    data["objects"]["bad_obj"] = {"ko": "x", "physical_type": "plasma", "states": ["smell"]}
    with pytest.raises(ValidationError) as exc:
        Ontology.model_validate(data)
    text = str(exc.value)
    assert "gardening" in text and "plasma" in text and "smell" in text


def test_valid_labels_pass(ontology: Ontology) -> None:
    validate_label(make_label(action_payload(pre_state={"wetness": "dry"})), ontology)
    seg = {"kind": "segment", "segment_id": "t1", "level": "task", "ref_id": "toilet_cleaning"}
    validate_label(make_label(seg), ontology)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (action_payload(verb="dance"), "알 수 없는 동사"),
        (action_payload(verb="wipe"), "원시 동작"),
        (action_payload(post_state={"wetness": "soggy"}), "없는 값"),
        ({"kind": "segment", "segment_id": "x", "level": "skill", "ref_id": "grasp"}, "기술 동사"),
        ({"kind": "segment", "segment_id": "x", "level": "substep", "ref_id": "nope"}, "하위 단계"),
        ({"kind": "event", "event_type": "slip"}, "심각도"),
        ({"kind": "event", "event_type": "drop"}, "related_action_id"),
        ({"kind": "gap", "gap_type": "nap"}, "사이 구간"),
        ({"kind": "object_state", "entity_id": "s", "class_id": "sink", "attribute": "fold_state",
          "value": "folded"}, "없는 상태 속성"),
        ({"kind": "hand_state", "hand": "left", "contact_target_kind": "object", "target_id": "c",
          "grasp_type": "claw", "role": "active"}, "파지 유형"),
    ],
)  # fmt: skip
def test_ontology_violations(ontology: Ontology, payload: dict[str, Any], message: str) -> None:
    problems = check_label(make_label(payload), ontology)
    assert any(message in p for p in problems), problems


def test_contactless_verb_and_point_event(ontology: Ontology) -> None:
    data: dict[str, Any] = ontology.model_dump(mode="json")
    data["verbs"]["glance"] = {"level": "primitive", "ko": "보다", "contactless": True}
    o = Ontology.model_validate(data)
    assert any("비접촉" in p for p in check_label(make_label(action_payload(verb="glance")), o))
    drop = {"kind": "event", "event_type": "drop", "related_action_id": "a001"}
    assert any("시점" in p for p in check_label(make_label(drop, t_end_ms=10), o))
    assert not check_label(make_label(drop, t_end_ms=0), o)


def test_mask_part_must_belong_to_tool(ontology: Ontology) -> None:
    mask = {"kind": "mask_track", "entity_id": "mop_01", "class_id": "mop", "part": "bristles",
            "keyframes": [{"t_ms": 0, "outside": True}]}  # fmt: skip
    assert any("부분" in p for p in check_label(make_label(mask, stream_id="bodycam"), ontology))


def test_version_mismatch_raises(ontology: Ontology) -> None:
    label = make_label(action_payload(), ontology_version="2.0.0")
    with pytest.raises(OntologyViolationError, match="버전"):
        validate_label(label, ontology)
