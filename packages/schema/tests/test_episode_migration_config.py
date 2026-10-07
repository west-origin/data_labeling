from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from dlp_schema.config import load_config
from dlp_schema.dataset import DatasetVersion, Split
from dlp_schema.episode import Entity, EntityKind, EpisodeGraph, current_labels
from dlp_schema.jsonschema import stale_schemas
from dlp_schema.labels import LabelRecord, Verification, VerificationState
from dlp_schema.migration import OntologyMigration, load_migration, migrate_labels
from dlp_schema.testing import FIXED_TIME, action_payload, make_label

ENTITIES = (
    Entity(entity_id="rag_01", kind=EntityKind.TOOL, class_id="rag"),
    Entity(entity_id="sink_01", kind=EntityKind.SURFACE, class_id="sink"),
)


def _state(label_id: str, t: int, value: str) -> LabelRecord:
    payload = {"kind": "object_state", "entity_id": "sink_01", "class_id": "sink",
               "attribute": "cleanliness", "value": value}  # fmt: skip
    return make_label(payload, label_id=label_id, t_start_ms=t, t_end_ms=t + 100)


def _graph(*labels: LabelRecord) -> EpisodeGraph:
    return EpisodeGraph(
        episode_id="e1", session_id="s001", ontology_version="1.0.0",
        t_start_ms=0, t_end_ms=10_000, entities=ENTITIES, labels=labels,
    )  # fmt: skip


def test_episode_graph_layers_and_state_transitions() -> None:
    relation = {
        "kind": "relation",
        "subject_id": "rag_01",
        "predicate": "contact",
        "object_id": "sink_01",
    }
    g = _graph(
        make_label(action_payload(), label_id="a"),
        make_label(relation, label_id="r"),
        _state("s1", 0, "dirty"),
        _state("s2", 3_000, "dirty"),
        _state("s3", 6_000, "clean"),
    )
    assert [x.label_id for x in g.relations] == ["r"]
    assert [x.label_id for x in g.events] == ["a"]
    [transition] = g.state_transitions()
    assert (transition.from_value, transition.to_value, transition.t_ms) == (
        "dirty",
        "clean",
        6_000,
    )


def test_episode_graph_rejects_unknown_entities_and_foreign_labels() -> None:
    with pytest.raises(ValidationError, match="알 수 없는 개체 cup_9"):
        _graph(make_label(action_payload(target_id="cup_9")))
    with pytest.raises(ValidationError, match="다른 세션"):
        _graph(make_label(action_payload(), session_id="s999"))
    with pytest.raises(ValidationError, match="구간 밖"):
        _graph(_state("late", 20_000, "dirty"))


def test_current_labels_follow_correction_history() -> None:
    original = make_label(action_payload(), label_id="v1")
    corrected = make_label(action_payload(verb="lift"), label_id="v2", parent_label_id="v1")
    false_positive = make_label(action_payload(), label_id="fp")
    retraction = make_label(action_payload(), label_id="fp-x", parent_label_id="fp", retracted=True)
    assert [
        x.label_id for x in current_labels([original, corrected, false_positive, retraction])
    ] == ["v2"]


def test_migration_renames_and_flags_removed_ids(tmp_path: Path) -> None:
    path = tmp_path / "1.0.0__1.1.0.yaml"
    path.write_text(
        "from_version: 1.0.0\nto_version: 1.1.0\n"
        "renames: {verbs: {grasp: grip}}\nremoved: {events: [slip]}\n",
        encoding="utf-8",
    )
    migration = load_migration(path)
    reviewed = Verification(
        state=VerificationState.HUMAN_APPROVED, reviewer_id="r1", reviewed_at=FIXED_TIME
    )
    labels = [
        make_label(action_payload(), label_id="a", verification=reviewed),
        make_label({"kind": "event", "event_type": "slip", "severity": 1}, label_id="e"),
        make_label(action_payload(), label_id="old", ontology_version="0.9.0"),
    ]
    result = migrate_labels(labels, migration, FIXED_TIME)
    [migrated] = result.migrated
    assert migrated.label_id == "a:v1.1.0"
    assert migrated.parent_label_id == "a"
    assert migrated.ontology_version == "1.1.0"
    assert migrated.payload.kind == "action" and migrated.payload.verb == "grip"
    assert migrated.verification.state is VerificationState.UNREVIEWED
    assert {lid for lid, _ in result.needs_review} == {"e", "old"}


def test_migration_moves_only_current_labels_of_history() -> None:
    """수정·삭제된 레코드를 다시 이관하면 지운 라벨이 되살아난다 (현재 라벨만 이관)."""
    migration = OntologyMigration(from_version="1.0.0", to_version="1.1.0")
    history = [
        make_label(action_payload(), label_id="v1"),
        make_label(action_payload(verb="lift"), label_id="v2", parent_label_id="v1"),
        make_label(action_payload(), label_id="fp"),
        make_label(action_payload(), label_id="fp-x", parent_label_id="fp", retracted=True),
        make_label(action_payload(), label_id="blind", measurement="blind"),
    ]
    result = migrate_labels(history, migration, FIXED_TIME)
    assert [(x.label_id, x.parent_label_id) for x in result.migrated] == [
        ("v2:v1.1.0", "v2"),
        ("blind:v1.1.0", "blind"),
    ]
    assert result.migrated[1].measurement == "blind"  # 측정 레코드는 이관 후에도 운영 라벨이 아니다
    assert not any(x.retracted for x in result.migrated)
    again = migrate_labels([*history, *result.migrated], migration, FIXED_TIME)
    assert again.migrated == () and again.needs_review == ()


def test_migration_renames_part_fields() -> None:
    migration = OntologyMigration.model_validate(
        {"from_version": "1.0.0", "to_version": "1.1.0",
         "renames": {"parts": {"cloth_face": "cloth_side", "corner_0": "origin"}}}
    )  # fmt: skip
    mask = {"kind": "mask_track", "entity_id": "rag_01", "class_id": "rag", "part": "cloth_face",
            "keyframes": [{"t_ms": 0, "outside": True}]}  # fmt: skip
    traj = {"kind": "trajectory3d", "entity_id": "table_01", "part": "corner_0",
            "frame": "camera", "source_3d": "mono_depth",
            "samples": [{"t_ms": 0, "x": 0, "y": 0, "z": 1}]}  # fmt: skip
    rel = {"kind": "relation", "subject_id": "rag_01", "subject_part": "cloth_face",
           "predicate": "contact", "object_id": "table_01", "object_part": "corner_0"}  # fmt: skip
    labels = [
        make_label(mask, label_id="m", stream_id="bodycam"),
        make_label(traj, label_id="t", stream_id="bodycam"),
        make_label(rel, label_id="r"),
    ]
    out = {x.parent_label_id: x.payload.model_dump() for x in
           migrate_labels(labels, migration, FIXED_TIME).migrated}  # fmt: skip
    assert out["m"]["part"] == "cloth_side"
    assert out["t"]["part"] == "origin"
    assert (out["r"]["subject_part"], out["r"]["object_part"]) == ("cloth_side", "origin")


def test_migration_rejects_unknown_category() -> None:
    with pytest.raises(ValidationError):
        OntologyMigration.model_validate(
            {"from_version": "1.0.0", "to_version": "1.1.0", "renames": {"colors": {"a": "b"}}}
        )


def test_dataset_version_excluded_sessions_cannot_be_split() -> None:
    with pytest.raises(ValidationError, match="제외된 세션"):
        DatasetVersion(
            version_id="d1", ontology_version="1.0.0", created_at=FIXED_TIME,
            snapshot_uri="lakefs://x", splits={"s001": Split.TRAIN}, excluded_sessions=("s001",),
        )  # fmt: skip


def test_defaults_config_loads(repo: Path) -> None:
    cfg = load_config(repo / "config" / "defaults.yaml")
    assert cfg.privacy.blur_hold_ms == 200
    assert cfg.golden_set.split_unit == ("worker_id", "site_id")
    assert cfg.export.include_unreviewed is False


def test_committed_json_schemas_are_current(repo: Path) -> None:
    assert stale_schemas(repo / "schemas") == [], "`dlp schema export`로 다시 생성하세요"
