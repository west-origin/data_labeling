from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from dlp_fixtures.actions import generate_action_scenario
from dlp_fixtures.wiping import generate_wiping_scenario
from dlp_relations.contact import SurfaceContact
from dlp_relations.coverage import coverage_ratio
from dlp_relations.derive import derive
from dlp_relations.geometry import SurfaceFrame, interpolate
from dlp_relations.policy import RelationsPolicy, load_policy
from dlp_relations.rules import apply_rules
from dlp_schema.labels import (
    CoordinateFrame,
    Hand,
    HandStatePayload,
    LabelRecord,
    RelationPayload,
    Trajectory3DPayload,
)
from dlp_schema.ontology import Ontology, load_ontology
from dlp_schema.testing import make_label

ROOT = Path(__file__).resolve().parents[3]
TOLERANCE_MS = 70  # 기준 문서의 접촉 경계 허용 오차 (장갑 ±70 ms, 영상 ±150 ms) 중 엄격한 쪽


@pytest.fixture(scope="module")
def policy() -> RelationsPolicy:
    return load_policy(ROOT)


@pytest.fixture(scope="module")
def ontology() -> Ontology:
    return load_ontology(ROOT / "config/ontology/v1")


def test_surface_coordinates_do_not_depend_on_camera_pose() -> None:
    corners = np.array([[0, 0, 0], [0.8, 0, 0], [0.8, 0.5, 0], [0, 0.5, 0]], dtype=float)
    point = np.array([0.2, 0.25, 0.03])
    angle = 0.7
    rot = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    ) @ np.array([[1, 0, 0], [0, np.cos(1.1), -np.sin(1.1)], [0, np.sin(1.1), np.cos(1.1)]])
    shift = np.array([0.3, -0.2, 1.5])
    a = SurfaceFrame.from_corners(corners).local(point)
    b = SurfaceFrame.from_corners(corners @ rot.T + shift).local(rot @ point + shift)
    assert a == pytest.approx((0.25, 0.5, 0.03)) and b == pytest.approx(a)
    assert SurfaceFrame.from_corners(corners).size_m == pytest.approx((0.8, 0.5))


def test_interpolation_respects_max_gap() -> None:
    times = np.array([0, 100, 400], dtype=np.int64)
    values = np.array([[0.0], [1.0], [4.0]])
    assert interpolate(times, values, 50, 60) == pytest.approx([0.5])
    assert interpolate(times, values, 250, 60) is None  # 양쪽 샘플이 150 ms 떨어져 있다
    assert interpolate(times, values, 450, 60) == pytest.approx([4.0])
    assert interpolate(times, values, 500, 60) is None


def test_coverage_of_one_stroke_matches_capsule_area() -> None:
    """길이 L 선분을 반지름 r 원으로 쓸면 넓이는 2rL + πr² 이다."""
    r, length, size = 0.05, 0.4, (1.0, 1.0)
    points = [(t, 0.3 + 0.4 * t / 1000, 0.5) for t in range(0, 1001, 33)]
    contact = SurfaceContact("rag_01", "cloth_face", "table_01", 0, 1000, points, [size] * 31)
    expected = 2 * r * length + np.pi * r * r
    assert coverage_ratio([contact], r, 0.005) == pytest.approx(expected, rel=0.03)


@pytest.mark.parametrize("seed", range(5))
def test_wiping_contacts_relations_and_coverage_match_truth(
    seed: int, policy: RelationsPolicy, ontology: Ontology
) -> None:
    w = generate_wiping_scenario(seed)
    d = derive(w.labels, ontology, policy)

    assert len(d.contacts) == len(w.contacts)
    for found, (start, end) in zip(d.contacts, w.contacts, strict=True):
        assert abs(found.start_ms - start) <= TOLERANCE_MS
        assert abs(found.end_ms - end) <= TOLERANCE_MS

    [cov] = d.coverage
    assert (cov.payload.surface_id, cov.payload.tool_id) == ("table_01", "rag_01")
    assert cov.payload.ratio == pytest.approx(w.coverage, abs=0.03)

    assert len(d.relations) == len(w.truth_relations)
    for got, truth in zip(d.relations, w.truth_relations, strict=True):
        assert isinstance(truth.payload, RelationPayload)
        expected = truth.payload.model_copy(update={"derived_by": got.payload.derived_by})
        assert got.payload == expected
        assert abs(got.start_ms - truth.t_start_ms) <= TOLERANCE_MS
        assert abs(got.end_ms - truth.t_end_ms) <= TOLERANCE_MS
    assert {r.payload.derived_by for r in d.relations} == {"hand_grasp", "tool_surface_contact"}


def test_tool_contact_needs_grasp_and_same_coordinate_frame(
    policy: RelationsPolicy, ontology: Ontology
) -> None:
    w = generate_wiping_scenario(0)
    no_grasp = [
        x
        for x in w.labels
        if not (isinstance(x.payload, HandStatePayload) and x.payload.target_id == "rag_01")
    ]
    assert derive(no_grasp, ontology, policy).contacts == []

    def to_world(x: LabelRecord) -> LabelRecord:
        p = x.payload
        if isinstance(p, Trajectory3DPayload) and p.entity_id == "rag_01":
            return x.model_copy(
                update={"payload": p.model_copy(update={"frame": CoordinateFrame.WORLD})}
            )
        return x

    assert derive([to_world(x) for x in w.labels], ontology, policy).contacts == []


def test_hand_state_rules_follow_grasp_type(policy: RelationsPolicy) -> None:
    labels = generate_action_scenario(2).labels
    states = [x for x in labels if isinstance(x.payload, HandStatePayload)]
    drafts = apply_rules(policy, states, [])
    touching = [
        x for x in states if isinstance(x.payload, HandStatePayload)
        and x.payload.contact_target_kind != "none"
    ]  # fmt: skip
    assert len(drafts) == len(touching)  # 손 상태 하나에 규칙 하나
    expected = {
        "tool_grip": "grasp", "power": "grasp", "hook": "grasp", "precision_pinch": "grasp",
        "palm_support": "support", "palm_push_wipe": "contact",
    }  # fmt: skip
    for state in touching:
        assert isinstance(state.payload, HandStatePayload)
        [d] = [d for d in drafts if d.start_ms == state.t_start_ms]
        assert d.payload.subject_id == f"{state.payload.hand.value}_hand"
        assert d.payload.object_id == state.payload.target_id
        assert d.payload.predicate.value == expected[state.payload.grasp_type or ""]


def test_rules_skip_missing_fields_and_merge_short_gaps(policy: RelationsPolicy) -> None:
    def state(start: int, end: int, **kw: Any) -> LabelRecord:
        payload = HandStatePayload(hand=Hand.LEFT, role="active", **kw)
        return make_label(payload=payload, t_start_ms=start, t_end_ms=end)

    patient = state(0, 500, contact_target_kind="person", body_part="forearm")
    a = state(0, 1000, contact_target_kind="object", target_id="cup_01", grasp_type="power")
    b = state(1050, 2000, contact_target_kind="object", target_id="cup_01", grasp_type="power")
    c = state(2500, 3000, contact_target_kind="object", target_id="cup_01", grasp_type="power")
    drafts = apply_rules(policy, [patient, a, b, c], [])
    assert [(d.start_ms, d.end_ms) for d in drafts] == [(0, 2000), (2500, 3000)]


def test_policy_digest_tracks_rule_changes(policy: RelationsPolicy) -> None:
    changed = policy.model_copy(update={"rules": policy.rules[1:]})
    assert changed.digest != policy.digest and policy.digest == load_policy(ROOT).digest
    with pytest.raises(ValueError):
        RelationsPolicy.model_validate(
            {
                **policy.model_dump(mode="json"),
                "rules": [policy.rules[0].model_dump(mode="json")] * 2,
            }
        )
