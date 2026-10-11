"""상위 구간 연결 검사의 행동 노드 식별 (F-SCH-01).

양손 행동은 두 손 레코드가 같은 action_id를 쓴다 (ADR 0002 §3, `labels.ActionPayload`).
상위 구간이 그 행동을 자식으로 가리켜도 저장·에피소드 검증이 받아야 하고, 같은 손에 같은
action_id가 겹치는 경우만 ID 중복 위반이다. 정답은 합성 라벨로 정했다.
"""

from __future__ import annotations

from typing import Any

import pytest
import sqlalchemy as sa
from pydantic import ValidationError

from dlp_schema.db.repository import get_labels, insert_labels
from dlp_schema.episode import Entity, EntityKind, EpisodeGraph
from dlp_schema.label_graph import check_segment_links
from dlp_schema.label_validation import validate_session_labels
from dlp_schema.labels import LabelRecord
from dlp_schema.ontology import Ontology
from dlp_schema.testing import action_payload, make_label
from dlp_schema.validation import OntologyViolationError, check_label_set, validate_episode


def _action(label_id: str, hand: str, start: int = 0, end: int = 1000, **kw: Any) -> LabelRecord:
    """action_id a001의 손별 원시 동작 (접촉 시각 없이 구간만 맞춘다)."""
    payload = action_payload(
        hand=hand,
        t_approach_ms=start,
        t_contact_start_ms=None,
        t_contact_end_ms=None,
        t_end_ms=end,
        **kw,
    )
    return make_label(payload, label_id=label_id, t_start_ms=start, t_end_ms=end)


def _skill(children: tuple[str, ...] = ("a001",), ref: str = "sk") -> LabelRecord:
    """0~1000ms 기술 구간."""
    return make_label(
        {
            "kind": "segment",
            "segment_id": ref,
            "level": "skill",
            "ref_id": "wipe",
            "child_ids": children,
        },
        label_id=ref,
    )


def test_bimanual_action_is_one_node_for_storage_and_episode(ontology: Ontology) -> None:
    """두 손 레코드(같은 action_id)와 그 행동을 묶는 상위 구간은 저장·에피소드 모두 유효하다."""
    labels = [_action("r", "right"), _action("l", "left"), _skill()]
    assert check_segment_links(labels) == []
    assert check_label_set(labels, ontology) == []
    validate_session_labels(ontology, [], labels)
    graph = EpisodeGraph(
        episode_id="ep",
        session_id="s001",
        ontology_version="1.0.0",
        t_start_ms=0,
        t_end_ms=1000,
        entities=(Entity(entity_id="rag_01", kind=EntityKind.TOOL, class_id="rag"),),
        labels=tuple(labels),
    )
    validate_episode(graph, ontology)


def test_same_hand_duplicate_action_id_is_still_rejected(ontology: Ontology) -> None:
    """같은 손에 같은 action_id가 겹치면 자식 참조가 모호하므로 위반이다."""
    labels = [_action("r1", "right"), _action("r2", "right"), _skill()]
    problems = check_segment_links(labels)
    assert problems == ["구간·행동 ID 중복: a001 (같은 손 right)"]
    with pytest.raises(OntologyViolationError, match="ID 중복"):
        validate_session_labels(ontology, [], labels)
    # 세 번째 레코드가 같은 손을 다시 쓰면 그 손 하나만 보고한다
    three = [_action("r", "right"), _action("l", "left"), _action("l2", "left"), _skill()]
    assert check_segment_links(three) == ["구간·행동 ID 중복: a001 (같은 손 left)"]


def test_action_and_segment_sharing_an_id_are_rejected() -> None:
    """행동과 상위 구간이 같은 ID를 쓰면 어느 쪽을 자식으로 가리키는지 알 수 없다."""
    labels = [_action("r", "right"), _action("l", "left"), _skill(children=(), ref="a001")]
    assert check_segment_links(labels) == ["구간·행동 ID 중복: a001"]
    segments = [_skill(children=(), ref="sk"), _skill(children=(), ref="sk")]
    assert check_segment_links(segments) == ["구간·행동 ID 중복: sk"]


def test_each_hand_of_a_child_action_must_lie_inside_the_parent() -> None:
    """양손 행동의 어느 한 손이라도 상위 구간 밖이면 위반이다 (손마다 구간이 다를 수 있다)."""
    inside = [_action("r", "right", 100, 900), _action("l", "left", 0, 1000), _skill()]
    assert check_segment_links(inside) == []
    outside = [_action("r", "right", 100, 900), _action("l", "left", 0, 1200), _skill()]
    assert check_segment_links(outside) == ["sk: 자식 a001이 상위 구간 밖에 있습니다"]


def test_episode_still_rejects_same_hand_duplicates_without_segments() -> None:
    """에피소드 계약은 상위 구간이 없어도 노드 검사를 한다: 다른 손은 받고 같은 손은 거부한다."""

    def graph(*labels: LabelRecord) -> EpisodeGraph:
        return EpisodeGraph(
            episode_id="ep",
            session_id="s001",
            ontology_version="1.0.0",
            t_start_ms=0,
            t_end_ms=1000,
            entities=(Entity(entity_id="rag_01", kind=EntityKind.TOOL, class_id="rag"),),
            labels=labels,
        )

    graph(_action("r", "right"), _action("l", "left"))
    with pytest.raises(ValidationError, match="같은 손 right"):
        graph(_action("r1", "right"), _action("r2", "right"))


@pytest.mark.services
def test_storage_accepts_bimanual_action_with_segment(pg: sa.Engine) -> None:
    """DB 저장(`insert_labels`)도 양손 행동과 상위 구간을 한 배치로 받고 같은 손 중복은 거부한다."""
    with pg.begin() as conn:
        insert_labels(conn, [_action("r", "right"), _action("l", "left"), _skill()])
        assert {x.label_id for x in get_labels(conn, "s001")} == {"r", "l", "sk"}
        with pytest.raises(OntologyViolationError, match="같은 손 right"):
            insert_labels(conn, [_action("r2", "right")])
