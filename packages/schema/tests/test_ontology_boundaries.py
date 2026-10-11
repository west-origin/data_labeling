"""온톨로지 저장·계층·이관 경계의 회귀 검사. 실영상 대신 합성 계약을 사용한다."""

from argparse import Namespace

import pytest
import sqlalchemy as sa
from pydantic import ValidationError
from sqlalchemy.exc import DBAPIError, NoResultFound

from dlp_schema.db.repository import (
    get_labels,
    get_ontology,
    get_session_ontology,
    insert_labels,
    insert_session,
    label_to_row,
    register_ontology,
)
from dlp_schema.db.tables import label_records
from dlp_schema.episode import Entity, EntityKind, EpisodeGraph, retractions
from dlp_schema.labels import LabelRecord
from dlp_schema.migration import OntologyMigration, migrate_labels
from dlp_schema.ontology import Ontology
from dlp_schema.testing import FIXED_TIME, action_payload, make_label, make_session
from dlp_schema.validation import OntologyViolationError, validate_episode


def _segment(
    ref: str = "seg", children: tuple[str, ...] = ("a001",), t_end_ms: int = 1000
) -> LabelRecord:
    return make_label(
        {
            "kind": "segment",
            "segment_id": ref,
            "level": "skill",
            "ref_id": "wipe",
            "child_ids": children,
        },
        label_id=ref,
        t_end_ms=t_end_ms,
    )


def _graph(*labels: LabelRecord) -> EpisodeGraph:
    return EpisodeGraph(
        episode_id="ep",
        session_id="s001",
        ontology_version="1.0.0",
        t_start_ms=0,
        t_end_ms=1000,
        entities=(Entity(entity_id="rag_01", kind=EntityKind.TOOL, class_id="rag"),),
        labels=labels,
    )


@pytest.mark.parametrize("children", [("missing",), ("a001", "a001"), ("seg",)])
def test_graph_rejects_broken_segment_links(children: tuple[str, ...]) -> None:
    with pytest.raises(ValidationError):
        _graph(make_label(action_payload()), _segment(children=children))


def test_graph_rejects_child_time_outside_parent():
    with pytest.raises(ValidationError, match="상위 구간 밖"):
        _graph(make_label(action_payload()), _segment(t_end_ms=500))


def test_graph_rejects_cycles_and_duplicate_action_ids():
    with pytest.raises(ValidationError, match="순환"):
        _graph(_segment("s1", ("s2",)), _segment("s2", ("s1",)))
    with pytest.raises(ValidationError, match="ID 중복"):
        _graph(
            make_label(action_payload(), label_id="l1"), make_label(action_payload(), label_id="l2")
        )


def test_graph_accepts_valid_skill_with_actions(ontology: Ontology):
    graph = _graph(make_label(action_payload()), _segment())
    validate_episode(graph, ontology)


def test_episode_checks_entity_part_membership(ontology: Ontology):
    def graph(part: str) -> EpisodeGraph:
        label = make_label(
            {
                "kind": "trajectory3d",
                "entity_id": "mop_01",
                "part": part,
                "frame": "camera",
                "source_3d": "mono_depth",
                "samples": [{"t_ms": 0, "x": 0, "y": 0, "z": 1}],
            },
            stream_id="bodycam",
        )
        return EpisodeGraph(
            episode_id="ep",
            session_id="s001",
            ontology_version="1.0.0",
            t_start_ms=0,
            t_end_ms=1000,
            labels=(label,),
            entities=(Entity(entity_id="mop_01", kind=EntityKind.TOOL, class_id="mop"),),
        )

    validate_episode(graph("mop_head"), ontology)
    with pytest.raises(ValueError, match="없는 부분 bristles"):
        validate_episode(graph("bristles"), ontology)


def test_migration_requires_matching_target_and_rejects_unknown_ids(ontology: Ontology):
    migration = OntologyMigration(
        from_version="1.0.0", to_version="2.0.0", renames={"verbs": {"grasp": "absent_destination"}}
    )
    with pytest.raises(ValueError, match="버전"):
        migrate_labels([], migration, FIXED_TIME, target_ontology=ontology)
    target = Ontology.model_validate(ontology.model_dump() | {"version": "2.0.0"})
    result = migrate_labels(
        [make_label(action_payload())], migration, FIXED_TIME, target_ontology=target
    )
    assert not result.migrated
    assert "absent_destination" in result.needs_review[0][1]


def test_migration_unsupported_state_change_goes_to_review(ontology: Ontology):
    data = ontology.model_dump(mode="json")
    data["version"] = "2.0.0"
    data["state_attributes"]["wetness"]["values"] = ["damp", "dry"]
    target = Ontology.model_validate(data)
    original = make_label(
        {
            "kind": "object_state",
            "entity_id": "rag_01",
            "class_id": "rag",
            "attribute": "wetness",
            "value": "wet",
        }
    )
    result = migrate_labels(
        [original],
        OntologyMigration(from_version="1.0.0", to_version="2.0.0"),
        FIXED_TIME,
        target_ontology=target,
    )
    assert not result.migrated and result.needs_review
    assert original.ontology_version == "1.0.0"


def test_migration_does_not_publish_parent_when_child_needs_review(ontology: Ontology):
    target = Ontology.model_validate(ontology.model_dump() | {"version": "2.0.0"})
    migration = OntologyMigration(
        from_version="1.0.0", to_version="2.0.0", removed={"verbs": ("grasp",)}
    )
    result = migrate_labels(
        [make_label(action_payload()), _segment()], migration, FIXED_TIME, target_ontology=target
    )
    assert not result.migrated
    assert {label_id for label_id, _ in result.needs_review} == {"l001", "seg"}


def test_review_projects_separate_versions_and_draft_additions(ontology: Ontology):
    from dlp_review.ontology_projects import ontology_project

    base = "dlp-labeling-temporal"
    first = ontology_project(base, ontology)
    data = ontology.model_dump(mode="json")
    data["version"] = "2.0.0"
    assert ontology_project(base, Ontology.model_validate(data)) != first
    data["version"] = "1.0.0"
    data["tasks"]["surface_wipe_disinfect"]["substeps"] = {"check_surface": {"ko": "표면 확인"}}
    assert ontology_project(base, Ontology.model_validate(data)) != first
    assert ontology_project(base, Ontology.model_validate(ontology.model_dump())) == first


def test_episode_rejects_substep_from_another_task(ontology: Ontology):
    data = ontology.model_dump(mode="json")
    data["tasks"]["floor_sweep_mop"]["substeps"] = {"sweep_floor": {"ko": "바닥 쓸기"}}
    test_ontology = Ontology.model_validate(data)
    child = make_label(
        {"kind": "segment", "segment_id": "sub", "level": "substep", "ref_id": "sweep_floor"},
        label_id="sub",
    )
    parent = make_label(
        {
            "kind": "segment",
            "segment_id": "task",
            "level": "task",
            "ref_id": "surface_wipe_disinfect",
            "child_ids": ["sub"],
        },
        label_id="task",
    )
    with pytest.raises(ValueError, match="다른 작업"):
        validate_episode(_graph(child, parent), test_ontology)


@pytest.mark.services
def test_storage_rejects_invalid_batch_before_any_insert(pg: sa.Engine):
    with pg.begin() as conn:
        with pytest.raises(OntologyViolationError, match="알 수 없는 동사"):
            insert_labels(
                conn,
                [
                    make_label(action_payload(), label_id="valid"),
                    make_label(action_payload(verb="made_up"), label_id="bad"),
                ],
            )
        assert not get_labels(conn, "s001")


@pytest.mark.services
def test_storage_uses_session_version_and_db_guard(pg: sa.Engine, ontology: Ontology):
    v2 = Ontology.model_validate(ontology.model_dump() | {"version": "2.0.0"})
    wrong = make_label(action_payload(), ontology_version="2.0.0")
    with pg.begin() as conn:
        register_ontology(conn, v2)
        insert_session(conn, make_session("s002", ontology_version="2.0.0"))
        assert get_session_ontology(conn, "s002").version == "2.0.0"
        with pytest.raises(OntologyViolationError, match="버전 불일치"):
            insert_labels(conn, [wrong])
        insert_labels(
            conn, [make_label(action_payload(), session_id="s002", ontology_version="2.0.0")]
        )
    with pytest.raises(DBAPIError), pg.begin() as conn:
        conn.execute(label_records.insert(), label_to_row(wrong))
    with pg.connect() as conn, pytest.raises(NoResultFound):
        get_ontology(conn, "9.9.9")


@pytest.mark.services
def test_missing_session_version_fails_instead_of_using_v1(pg: sa.Engine):
    with pg.begin() as conn:
        insert_session(conn, make_session("no-version", ontology_version=None))
        with pytest.raises(ValueError, match="버전이 없습니다"):
            get_session_ontology(conn, "no-version")


@pytest.mark.services
def test_storage_preserves_seeded_errors_and_legacy_cleanup(pg: sa.Engine):
    error = make_label(action_payload(verb="bad"), label_id="seed", seeded_error=True)
    legacy = make_label(action_payload(verb="old_bad"), label_id="legacy")
    with pg.begin() as conn:
        insert_labels(conn, [error])
        conn.execute(label_records.insert(), label_to_row(legacy))
        gone = retractions([legacy], "cleanup", FIXED_TIME)
        insert_labels(conn, gone)
        forged = gone[0].model_copy(update={"label_id": "forged", "payload": error.payload})
        with pytest.raises(OntologyViolationError, match="부모·페이로드"):
            insert_labels(conn, [forged])


@pytest.mark.services
def test_storage_blocks_dangling_reference_and_preserves_correction(pg: sa.Engine):
    action = make_label(action_payload(), label_id="action")
    with pg.begin() as conn:
        insert_labels(conn, [action, _segment()])
        with pytest.raises(OntologyViolationError, match="알 수 없는 자식"):
            insert_labels(conn, retractions([action], "next", FIXED_TIME))
        fixed = make_label(action_payload(verb="rub"), label_id="fixed", parent_label_id="action")
        insert_labels(conn, [fixed])
        assert len(get_labels(conn, "s001")) == 3


@pytest.mark.services
def test_storage_checks_parts_against_known_entity_class(pg: sa.Engine):
    box = make_label(
        {
            "kind": "box_track",
            "class_id": "mop",
            "entity_id": "mop_01",
            "keyframes": [{"t_ms": 0, "outside": True, "x": 0, "y": 0, "w": 1, "h": 1}],
        },
        stream_id="bodycam",
        label_id="box",
    )

    def trajectory(part: str) -> LabelRecord:
        return make_label(
            {
                "kind": "trajectory3d",
                "entity_id": "mop_01",
                "part": part,
                "frame": "camera",
                "source_3d": "mono_depth",
                "samples": [{"t_ms": 0, "x": 0, "y": 0, "z": 1}],
            },
            stream_id="bodycam",
            label_id="traj",
        )

    with pg.begin() as conn:
        with pytest.raises(OntologyViolationError, match="없는 부분"):
            insert_labels(conn, [box, trajectory("bristles")])
        assert not get_labels(conn, "s001")
        insert_labels(conn, [box, trajectory("mop_head")])


@pytest.mark.services
def test_cli_loads_v2_and_audit_is_read_only(
    pg: sa.Engine,
    ontology: Ontology,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    from dlp_cli import ontology_access, schema_cmds

    url = pg.url.render_as_string(hide_password=False)

    def test_url(_: str | None) -> str:
        return url

    monkeypatch.setattr(ontology_access, "database_url", test_url)
    monkeypatch.setattr(schema_cmds, "database_url", test_url)
    with pg.begin() as conn:
        register_ontology(
            conn, Ontology.model_validate(ontology.model_dump() | {"version": "2.0.0"})
        )
        insert_session(conn, make_session("v2", ontology_version="2.0.0"))
        invalid = make_label(action_payload(verb="legacy_bad"))
        conn.execute(label_records.insert(), label_to_row(invalid))
    assert ontology_access.session_ontology(None, "v2").version == "2.0.0"
    assert schema_cmds.cmd_ontology_audit(Namespace(url=None, session_id="s001")) == 1
    assert "legacy_bad" in capsys.readouterr().out
    with pg.connect() as conn:
        assert get_labels(conn, "s001") == [invalid]
