"""검수 계획 집계의 운영 라벨·부하 규약 회귀 (실제 PostgreSQL)."""

from collections import Counter
from typing import Any

import pytest
import sqlalchemy as sa

from dlp_schema.db import repository as repo
from dlp_schema.db.review_queries import open_assignment_counts, verified_object_classes
from dlp_schema.episode import current_labels, retractions
from dlp_schema.labels import BoxTrackPayload, Provenance, Source, Verification, VerificationState
from dlp_schema.review import AssignmentStatus, ReviewAssignment, ReviewMode
from dlp_schema.testing import FIXED_TIME, make_label, make_session

pytestmark = pytest.mark.services


def test_known_classes_matches_current_operational_history(pg: sa.Engine) -> None:
    """기하 정보 없이 집계해도 수정·철회·표본·측정·오류 삽입 후손 조건은 유지된다."""

    def box(key: str, cls: str, **changes: Any):
        return make_label(
            {
                "kind": "box_track",
                "entity_id": key,
                "class_id": cls,
                "keyframes": [{"t_ms": 0, "x": 1, "y": 1, "w": 10, "h": 10}],
            },
            label_id=key,
            stream_id="bodycam",
            **changes,
        )

    stale = box("old", "mop")
    removed = box("removed", "brush")
    seeded = box("seed", "sponge", seeded_error=True)
    measured = box("measure", "sink", measurement="blind")
    labels = [
        stale,
        box("corrected", "rag", parent_label_id="old"),
        removed,
        *retractions([removed], "m2", FIXED_TIME),
        box(
            "pending",
            "broom",
            confidence=0.9,
            provenance=Provenance(source=Source.MODEL, model_version="m1"),
        ),
        box(
            "sample",
            "dustpan",
            confidence=0.9,
            provenance=Provenance(source=Source.MODEL, model_version="m1"),
            verification=Verification(
                state=VerificationState.SAMPLE_VERIFIED,
                reviewer_id="reviewer",
                reviewed_at=FIXED_TIME,
            ),
        ),
        box(
            "approved",
            "cup",
            confidence=0.9,
            provenance=Provenance(source=Source.MODEL, model_version="m1"),
            verification=Verification(
                state=VerificationState.HUMAN_APPROVED,
                reviewer_id="reviewer",
                reviewed_at=FIXED_TIME,
            ),
        ),
        seeded,
        box("seed-fix", "sponge", parent_label_id="seed"),
        box("seed-grandchild", "sponge", parent_label_id="seed-fix"),
        measured,
        box("measure-fix", "sink", parent_label_id="measure"),
        box("measure-grandchild", "sink", parent_label_id="measure-fix"),
    ]
    with pg.begin() as conn:
        repo.insert_labels(conn, labels)
        repo.insert_session(conn, make_session(session_id="excluded"))
        own = box("own", "door", session_id="excluded")
        repo.insert_labels(conn, [own])
        expected = {
            x.payload.class_id
            for x in current_labels(repo.get_labels(conn, "s001"))
            if isinstance(x.payload, BoxTrackPayload)
            and (
                x.provenance.source is Source.HUMAN
                or x.verification.state
                in (VerificationState.HUMAN_APPROVED, VerificationState.HUMAN_CORRECTED)
            )
        }
        assert expected == {"cup", "rag"}
        assert verified_object_classes(conn, "excluded") == expected
        assert verified_object_classes(conn, "s001") == {"door"}


def test_open_loads_counts_only_candidates_and_open_assignments(pg: sa.Engine) -> None:
    """담당자 없는 배정·완료 배정·후보 밖 담당자는 집계하지 않는다."""
    with pg.begin() as conn:
        for i, (who, status) in enumerate(
            [
                ("a", AssignmentStatus.OPEN),
                ("a", AssignmentStatus.OPEN),
                ("b", AssignmentStatus.OPEN),
                ("a", AssignmentStatus.DONE),
                ("other", AssignmentStatus.OPEN),
                (None, AssignmentStatus.OPEN),
            ]
        ):
            repo.insert_assignment(
                conn,
                ReviewAssignment(
                    assignment_id=f"load-{i}",
                    session_id="s001",
                    stream_id="bodycam",
                    label_kinds=("box_track",),
                    mode=ReviewMode.STANDARD,
                    priority=1,
                    assignee=who,
                    status=status,
                    created_at=FIXED_TIME,
                ),
            )
        expected = Counter(
            a.assignee
            for a in repo.list_assignments(conn, status=AssignmentStatus.OPEN)
            if a.assignee in {"a", "b"}
        )
        assert open_assignment_counts(conn, ["a", "b"]) == expected == {"a": 2, "b": 1}
        assert open_assignment_counts(conn, []) == {}
