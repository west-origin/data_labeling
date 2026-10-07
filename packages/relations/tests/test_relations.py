"""관계·커버리지 순수 계산 테스트 (DB 없음).

정답은 합성 닦기 시나리오(`dlp_fixtures.wiping.generate_wiping_scenario`: 걸레 작용부·탁자 꼭짓점
3D 궤적, 손 파지 구간, 정답 접촉 구간·커버리지·관계)와 행동 시나리오의 손 상태다. 관련: WP9, ADR
0011.
"""

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
    """저장소 `relations.yaml` 정책 (모듈 범위)."""
    return load_policy(ROOT)


@pytest.fixture(scope="module")
def ontology() -> Ontology:
    """온톨로지 v1 (모듈 범위)."""
    return load_ontology(ROOT / "config/ontology/v1")


def test_surface_coordinates_do_not_depend_on_camera_pose() -> None:
    """표면 좌표가 카메라 자세(회전·이동)와 무관한지 본다.

    정답 근거: 0.8x0.5 m 탁자에서 (0.2, 0.25, 0.03) 점은 가로 0.25, 세로 0.5, 거리 0.03 m. 같은
    강체 변환을 꼭짓점과 점에 함께 적용해도 같다. 표면 크기는 (0.8, 0.5).
    """
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
    """보간 허용 거리: 가까운 샘플이 60 ms 안이면 보간·끝값 유지, 양쪽이 150 ms씩 떨어지거나
    끝에서 100 ms 밖이면 None.
    """
    times = np.array([0, 100, 400], dtype=np.int64)
    values = np.array([[0.0], [1.0], [4.0]])
    assert interpolate(times, values, 50, 60) == pytest.approx([0.5])
    assert interpolate(times, values, 250, 60) is None  # 양쪽 샘플이 150 ms 떨어져 있다
    assert interpolate(times, values, 450, 60) == pytest.approx([4.0])
    assert interpolate(times, values, 500, 60) is None


def test_coverage_of_one_stroke_matches_capsule_area() -> None:
    """길이 L 선분을 반지름 r 원으로 쓸면 넓이는 2rL + πr² 이다.

    1 m x 1 m 표면이라 비율 = 넓이. 격자 5 mm에서 3% 안.
    """
    r, length, size = 0.05, 0.4, (1.0, 1.0)
    points = [(t, 0.3 + 0.4 * t / 1000, 0.5) for t in range(0, 1001, 33)]
    contact = SurfaceContact("rag_01", "cloth_face", "table_01", 0, 1000, points, [size] * 31)
    expected = 2 * r * length + np.pi * r * r
    assert coverage_ratio([contact], r, 0.005) == pytest.approx(expected, rel=0.03)


@pytest.mark.parametrize("seed", range(5))
def test_wiping_contacts_relations_and_coverage_match_truth(
    seed: int, policy: RelationsPolicy, ontology: Ontology
) -> None:
    """닦기 시나리오 5개 시드에서 접촉 구간 수·경계(±70 ms), 커버리지(±0.03),
    관계(페이로드·경계)가 정답과 맞는지 본다. 규칙은 hand_grasp와 tool_surface_contact 두
    가지만 나온다.
    """
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
    """손이 걸레를 쥔 구간이 없으면(require_grasp) 접촉이 없고, 걸레 궤적 좌표계가 표면과 다르면
    (world vs camera) 접촉이 없는지 본다.
    """
    w = generate_wiping_scenario(0)
    no_grasp = [
        x
        for x in w.labels
        if not (isinstance(x.payload, HandStatePayload) and x.payload.target_id == "rag_01")
    ]
    assert derive(no_grasp, ontology, policy).contacts == []

    def to_world(x: LabelRecord) -> LabelRecord:
        """걸레 궤적만 world 좌표계로 바꾼 사본 (표면 꼭짓점은 camera 그대로)."""
        p = x.payload
        if isinstance(p, Trajectory3DPayload) and p.entity_id == "rag_01":
            return x.model_copy(
                update={"payload": p.model_copy(update={"frame": CoordinateFrame.WORLD})}
            )
        return x

    assert derive([to_world(x) for x in w.labels], ontology, policy).contacts == []


def test_hand_state_rules_follow_grasp_type(policy: RelationsPolicy) -> None:
    """행동 시나리오의 손 상태마다 규칙 하나가 맞고, grasp_type에 따라
    술어(grasp/support/contact)가 정해지는지 본다.
    """
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
    """대상 ID가 없는 사람 접촉(목적어 자리 비움)은 관계를 만들지 않고, 50 ms 끊긴 같은 관계는
    잇고 500 ms 끊긴 것은 따로 두는지 본다 (merge_gap_ms=100).
    """

    def state(start: int, end: int, **kw: Any) -> LabelRecord:
        """왼손 손 상태 라벨 (구간과 페이로드 필드를 받는다)."""
        payload = HandStatePayload(hand=Hand.LEFT, role="active", **kw)
        return make_label(payload=payload, t_start_ms=start, t_end_ms=end)

    patient = state(0, 500, contact_target_kind="person", body_part="forearm")
    a = state(0, 1000, contact_target_kind="object", target_id="cup_01", grasp_type="power")
    b = state(1050, 2000, contact_target_kind="object", target_id="cup_01", grasp_type="power")
    c = state(2500, 3000, contact_target_kind="object", target_id="cup_01", grasp_type="power")
    drafts = apply_rules(policy, [patient, a, b, c], [])
    assert [(d.start_ms, d.end_ms) for d in drafts] == [(0, 2000), (2500, 3000)]


def test_unresolved_contact_target_emits_no_relation(policy: RelationsPolicy) -> None:
    """감사 회귀: 장갑만 잡은 접촉(target_id=unresolved)이 관계를 만들지 않는다."""
    payload = HandStatePayload(
        hand=Hand.RIGHT, role="active", contact_target_kind="object", target_id="unresolved"
    )
    assert "unresolved" in policy.unresolved_target_ids
    assert apply_rules(policy, [make_label(payload=payload)], []) == []
    known = payload.model_copy(update={"target_id": "cup_01"})
    [d] = apply_rules(policy, [make_label(payload=known)], [])
    assert d.payload.object_id == "cup_01"


def test_policy_digest_tracks_rule_changes(policy: RelationsPolicy) -> None:
    """규칙을 바꾸면 정책 해시가 바뀌고, 같은 파일을 다시 읽으면 같으며, 규칙 ID가 겹치면 검증
    오류인지 본다.
    """
    changed = policy.model_copy(update={"rules": policy.rules[1:]})
    assert changed.digest != policy.digest and policy.digest == load_policy(ROOT).digest
    with pytest.raises(ValueError):
        RelationsPolicy.model_validate(
            {
                **policy.model_dump(mode="json"),
                "rules": [policy.rules[0].model_dump(mode="json")] * 2,
            }
        )


def test_duplicate_working_part_trajectory_gives_one_coverage(
    policy: RelationsPolicy, ontology: Ontology
) -> None:
    """감사 회귀 (4차): 같은 도구 작용부 궤적이 새 ID로 하나 더 있어도(3D 단계 재실행 뒤 이전
    것은 검수돼 남음) 접촉·관계·커버리지가 두 번 나오지 않는다.
    """
    w = generate_wiping_scenario(0)
    base = derive(w.labels, ontology, policy)
    tool = [
        x
        for x in w.labels
        if isinstance(x.payload, Trajectory3DPayload) and x.payload.entity_id == "rag_01"
    ]
    assert tool
    dup = [x.model_copy(update={"label_id": f"{x.label_id}-v2"}) for x in tool]
    d = derive([*w.labels, *dup], ontology, policy)
    assert [c.payload for c in d.coverage] == [c.payload for c in base.coverage]
    assert len(d.contacts) == len(base.contacts)
    assert [r.payload for r in d.relations] == [r.payload for r in base.relations]
