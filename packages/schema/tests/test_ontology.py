"""온톨로지 로더·교차 참조 검증과 라벨의 온톨로지 대조(`validation.check_label`) 테스트.

정답 근거: 실제 config/ontology/v1 내용(기준 문서 "온톨로지
v1" 절)과, 일부러 사전 밖 값을 넣은 라벨.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from dlp_schema.ontology import (
    Ontology,
    PriorSource,
    VerbLevel,
    find_ontology_dir,
    load_ontology,
)
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
    data["tasks"]["bad_task"] = {"domain": "gardening", "ko": "정원", "en": "Garden"}
    data["objects"]["bad_obj"] = {
        "ko": "x",
        "en": "x",
        "physical_type": "plasma",
        "states": ["smell"],
    }
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
    data["verbs"]["glance"] = {
        "level": "primitive",
        "ko": "보다",
        "en": "glance",
        "contactless": True,
    }
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
        """table_01의 3D 궤적 페이로드 (part만 바꿔 가며 쓴다)."""
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


# ---------------------------------------------------------------- 한·영 이름 (ADR 0043)

# v1 사전 항목 수 (도메인 3, 작업 17, 동사 34, 공백 3, 이벤트 15, 접촉 대상 6, 파지 8, 손 역할 3,
# 손 관절 6, 팔 관절 2 (ADR 0058), 물리 유형 5, 상태 속성 6, 객체 31 (손 마스크 hand, ADR 0059),
# 표면 부분 4, 프라이버시 대상 13, 신체 부위 26, 자세 9)
V1_TERMS = 191


def test_every_term_has_korean_and_english_names(ontology: Ontology) -> None:
    """v1의 모든 사전 항목(하위 단계 포함)에 빈칸 아닌 ko·en 이름이 있고, 키는 "범주:ID"다."""
    names = ontology.display_names()
    assert len(names) == V1_TERMS
    for key, name in names.items():
        assert name.ko.strip(), key
        assert name.en.strip(), key
        assert name.en.isascii(), f"{key}: 영어 이름에 비ASCII 글자 {name.en}"
    assert names["verbs:grasp"].ko == "잡다"
    assert names["verbs:grasp"].en == "grasp"
    assert names["objects:mop"].en == "Mop"
    # 같은 ID가 다른 범주에 있어도(person) 범주로 구별된다
    assert names["physical_types:person"].en == "Person"
    assert ontology.name("objects", "bucket", "en") == "Bucket"
    assert ontology.name("objects", "bucket", "ko") == "양동이"
    assert ontology.name("objects", "moon", "en") is None


def test_substep_names_are_included(ontology: Ontology) -> None:
    """작업의 하위 단계도 "substeps:<ID>"로 이름 표에 들어간다."""
    data = ontology.model_dump(mode="json")
    data["tasks"]["floor_sweep_mop"]["substeps"] = {
        "sweep_floor": {"ko": "바닥 쓸기", "en": "Sweep the floor"}
    }
    names = Ontology.model_validate(data).display_names()
    assert names["substeps:sweep_floor"].en == "Sweep the floor"
    assert len(names) == V1_TERMS + 1


@pytest.mark.parametrize("en", [None, "", "   "])
def test_term_without_english_name_is_rejected(ontology: Ontology, en: str | None) -> None:
    """en이 없거나 비면(공백만 포함) 사전 적재가 실패한다."""
    data = ontology.model_dump(mode="json")
    if en is None:
        del data["verbs"]["grasp"]["en"]
    else:
        data["verbs"]["grasp"]["en"] = en
    with pytest.raises(ValidationError):
        Ontology.model_validate(data)


def test_term_without_korean_name_is_rejected(ontology: Ontology) -> None:
    """ko가 비어도 사전 적재가 실패한다."""
    data = ontology.model_dump(mode="json")
    data["objects"]["mop"]["ko"] = " "
    with pytest.raises(ValidationError):
        Ontology.model_validate(data)


# ---------------------------------------------------------------- 객체 물리 사전값 (ADR 0049)


def test_v1_objects_carry_physical_priors(ontology: Ontology) -> None:
    """v1 객체는 person·hand(손 마스크 클래스, ADR 0059)를 빼고 모두 물리 사전값이 있고, 모두
    측정값이 아닌 추정(prior_estimate)이다.

    정답 근거: config/ontology/v1/objects.yaml (2026-10-17 덧붙임). 재질·촉감 이름은 표시 이름 표
    (`display_names`, 번역 용어집)에 들지 않고 `physics_names`에만 있다.
    """
    with_priors = {k for k, v in ontology.objects.items() if v.physical_priors is not None}
    assert with_priors == set(ontology.objects) - {"person", "hand"}
    sources = {v.physical_priors.source for v in ontology.objects.values() if v.physical_priors}
    assert sources == {PriorSource.PRIOR_ESTIMATE}
    bucket = ontology.objects["bucket"].physical_priors
    assert bucket is not None and bucket.mass_kg is not None
    assert bucket.mass_kg.by_state["fill_state"]["full"].min == 5.0
    names = ontology.physics_names()
    assert names["materials:plastic"].en == "Plastic"
    assert names["tactile_properties:slippery_when_wet"].ko == "젖으면 미끄러움"
    assert not set(names) & set(ontology.display_names())
    assert len(ontology.display_names()) == V1_TERMS  # 용어집(번역 버전)은 그대로


def _with_priors(ontology: Ontology, obj: str, priors: dict[str, Any]) -> dict[str, Any]:
    """obj의 물리 사전값을 priors로 바꾼 사전 데이터 (검증 전)."""
    data = ontology.model_dump(mode="json")
    data["objects"][obj]["physical_priors"] = priors
    return data


def _by_state(attr: str, value: str, lo: float, hi: float) -> dict[str, Any]:
    """기본 범위 0.1~1에 상태별 범위 하나를 붙인 질량 사전값."""
    return {"min": 0.1, "max": 1, "by_state": {attr: {value: {"min": lo, "max": hi}}}}


_EST = "prior_estimate"


@pytest.mark.parametrize(
    ("obj", "priors", "message"),
    [
        ("cup", {"source": _EST, "mass_kg": {"min": 2.0, "max": 1.0}}, "min"),
        ("cup", {"source": _EST, "mass_kg": {"min": -1.0, "max": 1.0}}, "greater"),
        ("cup", {"source": _EST, "mass_kg": {"min": 0.1, "max": "inf"}}, "finite"),
        ("cup", {"source": _EST, "static_friction": {"min": 0, "max": 1}}, "greater"),
        ("cup", {"source": "measured", "material": "ceramic"}, "source"),
        ("cup", {"material": "ceramic"}, "source"),
        ("cup", {"source": _EST}, "비었습니다"),
        ("cup", {"source": _EST, "material": "unobtainium"}, "알 수 없는 재질"),
        ("cup", {"source": _EST, "tactile": ["fluffy"]}, "알 수 없는 촉감"),
        ("cup", {"source": _EST, "tactile": ["hard", "hard"]}, "중복"),
        ("mop", {"source": _EST, "part_materials": {"handle": "unobtainium"}}, "알 수 없는 재질"),
        ("cup", {"source": _EST, "part_materials": {"handle": "plastic"}}, "도구 부분이 아닌"),
        ("cup", {"source": _EST, "mass_kg": _by_state("wetness", "wet", 1, 2)}, "states에 없음"),
        ("cup", {"source": _EST, "mass_kg": _by_state("fill_state", "brimming", 1, 2)}, "brimming"),
        ("cup", {"source": _EST, "mass_kg": _by_state("fill_state", "full", 3, 2)}, "min"),
        # ADR 0049 규칙과 기본 범위의 뜻 (F-SCH-22·23)
        ("cup", {"source": "prior_literature", "material": "ceramic"}, "note"),
        ("person", {"source": _EST, "mass_kg": {"min": 40, "max": 120}}, "물리 유형 person"),
        ("mop", {"source": _EST, "material": "metal", "part_materials": {"mop_head": "metal"}},
         "대표 재질과 같은"),
        ("towel", {"source": _EST, "mass_kg": _by_state("wetness", "dry", 0.2, 0.5)}, "기본"),
        ("cup", {"source": _EST, "mass_kg": _by_state("fill_state", "full", 0.5, 2)}, "모든 상태"),
        ("towel", {"source": _EST, "mass_kg": {"min": 0.1, "max": 1, "by_state": {
            "wetness": {"wet": {"min": 0.2, "max": 3}},
            "fold_state": {"folded": {"min": 0.1, "max": 1}}}}}, "하나만"),
        ("cup", {"source": _EST, "mass_kg": {"min": 0.1, "max": 1, "by_state": {"fill_state": {}}}},
         "상태 값이 없습니다"),
    ],
)  # fmt: skip
def test_physical_prior_validation(
    ontology: Ontology, obj: str, priors: dict[str, Any], message: str
) -> None:
    """잘못된 범위(min>max, 0 이하, 무한대), 측정값 출처, 빈 블록, 사전에 없는 재질·촉감, 중복 촉감,
    도구 부분이 아닌 재질 부분, 객체에 없는 상태 속성·값의 상태별 범위를 거부한다. 문헌 출처의 빈
    note, 사람·액체 유형의 사전값, 대표 재질과 같은 부분 재질, 기본 조건(건조)을 따로 적은 범위,
    모든 상태를 덮어야 하는 기본 범위 밖의 상태별 범위, 두 상태 속성, 빈 상태 값 사전도 거부한다."""
    with pytest.raises(ValidationError, match=message):
        Ontology.model_validate(_with_priors(ontology, obj, priors))


@pytest.mark.parametrize("category", ["materials", "tactile_properties"])
@pytest.mark.parametrize("missing", ["ko", "en"])
def test_physics_vocabulary_requires_both_names(
    ontology: Ontology, category: str, missing: str
) -> None:
    """재질·촉감 어휘도 ko·en이 모두 필요하다 (ADR 0043)."""
    data = ontology.model_dump(mode="json")
    del next(iter(data[category].values()))[missing]
    with pytest.raises(ValidationError):
        Ontology.model_validate(data)


def test_valid_state_dependent_priors_are_accepted(ontology: Ontology) -> None:
    """상태별 범위는 기본 범위 밖이어도 되고(젖은 물체의 마찰), 도구 부분 재질도 받는다."""
    friction = {"min": 0.3, "max": 0.8, "by_state": {"wetness": {"wet": {"min": 0.1, "max": 0.2}}}}
    priors: dict[str, Any] = {
        "source": "prior_literature",
        "note": "제품 사양",
        "material": "metal",
        "part_materials": {"mop_head": "fabric"},
        "tactile": ["hard"],
        "static_friction": friction,
    }
    o = Ontology.model_validate(_with_priors(ontology, "mop", priors))
    got = o.objects["mop"].physical_priors
    assert got is not None and got.source is PriorSource.PRIOR_LITERATURE
    assert got.static_friction is not None
    assert got.static_friction.by_state["wetness"]["wet"].max == 0.2
