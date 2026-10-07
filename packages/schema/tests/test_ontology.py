"""온톨로지 로더·교차 참조 검증과 라벨의 온톨로지 대조(`validation.check_label`) 테스트.

정답 근거: 실제 config/ontology/v1 내용(기준 문서 "온톨로지
v1" 절)과, 일부러 사전 밖 값을 넣은 라벨.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from dlp_schema.ontology import Ontology, VerbLevel, find_ontology_dir, load_ontology
from dlp_schema.testing import action_payload, make_label
from dlp_schema.validation import OntologyViolationError, check_label, validate_label


def test_v1_matches_plan(ontology: Ontology) -> None:
    """v1 YAML이 기준 문서의 숫자와 맞는지: 신체 부위 26개, 필수 블러 대상 7개, 원시 동작 11개,
    공백 유형 3개, 작업의 도메인 참조, 밀대 도구 부분, 세면대 표면 표시.
    """
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
    """manifest의 version으로 디렉터리를 찾고, 없는 버전은 FileNotFoundError."""
    root = repo / "config" / "ontology"
    assert find_ontology_dir(root, "1.0.0").name == "v1"
    with pytest.raises(FileNotFoundError):
        find_ontology_dir(root, "9.9.9")


def _write(tmp: Path, files: dict[str, str]) -> Path:
    """임시 디렉터리에 파일 이름 → 내용으로 YAML 파일들을 쓴다."""
    for name, text in files.items():
        (tmp / name).write_text(text, encoding="utf-8")
    return tmp


def test_loader_rejects_duplicate_keys_across_files(tmp_path: Path) -> None:
    """같은 최상위 키(verbs)가 두 파일에 있으면 로더가 거부한다."""
    _write(tmp_path, {"a.yaml": "verbs: {}\n", "b.yaml": "verbs: {}\n"})
    with pytest.raises(ValueError, match="중복"):
        load_ontology(tmp_path)


def test_cross_references_are_checked(ontology: Ontology) -> None:
    """없는 도메인·물리 유형·상태 속성을 참조하면 세 위반이 한 메시지에 모두 나온다."""
    data: dict[str, Any] = ontology.model_dump(mode="json")
    data["tasks"]["bad_task"] = {"domain": "gardening", "ko": "정원"}
    data["objects"]["bad_obj"] = {"ko": "x", "physical_type": "plasma", "states": ["smell"]}
    with pytest.raises(ValidationError) as exc:
        Ontology.model_validate(data)
    text = str(exc.value)
    assert "gardening" in text and "plasma" in text and "smell" in text


def test_valid_labels_pass(ontology: Ontology) -> None:
    """유효한 행동(사전 안 상태 포함)과 작업 수준 상위 구간은 예외 없이 통과한다."""
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
    """사전 밖 값·규칙 위반마다 기대 메시지 조각이 위반 목록에 있다 (매개변수 표 참고)."""
    problems = check_label(make_label(payload), ontology)
    assert any(message in p for p in problems), problems


def test_contactless_verb_and_point_event(ontology: Ontology) -> None:
    """비접촉 동사에 접촉 시각이 있으면 위반, 시점 이벤트는 t_start == t_end여야 한다.

    비접촉 원시 동사 glance를 테스트용으로 사전에 더해 확인한다.
    """
    data: dict[str, Any] = ontology.model_dump(mode="json")
    data["verbs"]["glance"] = {"level": "primitive", "ko": "보다", "contactless": True}
    o = Ontology.model_validate(data)
    assert any("비접촉" in p for p in check_label(make_label(action_payload(verb="glance")), o))
    drop = {"kind": "event", "event_type": "drop", "related_action_id": "a001"}
    assert any("시점" in p for p in check_label(make_label(drop, t_end_ms=10), o))
    assert not check_label(make_label(drop, t_end_ms=0), o)


def test_mask_part_must_belong_to_tool(ontology: Ontology) -> None:
    """밀대(mop) 마스크의 part가 밀대의 작용부·파지부가 아니면(bristles) 위반이다."""
    mask = {"kind": "mask_track", "entity_id": "mop_01", "class_id": "mop", "part": "bristles",
            "keyframes": [{"t_ms": 0, "outside": True}]}  # fmt: skip
    assert any("부분" in p for p in check_label(make_label(mask, stream_id="bodycam"), ontology))


def test_version_mismatch_raises(ontology: Ontology) -> None:
    """라벨 온톨로지 버전이 사전과 다르면 validate_label이 OntologyViolationError를 던진다."""
    label = make_label(action_payload(), ontology_version="2.0.0")
    with pytest.raises(OntologyViolationError, match="버전"):
        validate_label(label, ontology)


def test_trajectory_and_relation_parts_must_be_known(ontology: Ontology) -> None:
    """3D 궤적·관계의 part는 알려진 부분(도구 부분·표면 꼭짓점·손 관절) 안이어야 한다.

    corner_9, lid는 사전에 없어 위반이고, None과 알려진 부분은 통과한다.
    """
    assert {"corner_0", "corner_3", "wrist", "mop_head", "handle"} <= ontology.known_parts()

    def traj(part: str | None) -> dict[str, Any]:
        return {"kind": "trajectory3d", "entity_id": "table_01", "part": part,
                "frame": "camera", "source_3d": "mono_depth",
                "samples": [{"t_ms": 0, "x": 0, "y": 0, "z": 1}]}  # fmt: skip

    for part in ("corner_0", "cloth_face", "index_tip", None):
        assert check_label(make_label(traj(part), stream_id="bodycam"), ontology) == []
    bad = check_label(make_label(traj("corner_9"), stream_id="bodycam"), ontology)
    assert any("부분: corner_9" in p for p in bad)
    relation = {"kind": "relation", "subject_id": "rag_01", "subject_part": "cloth_face",
                "predicate": "contact", "object_id": "table_01", "object_part": "lid"}  # fmt: skip
    assert check_label(make_label(relation), ontology) == ["알 수 없는 부분: lid"]
