"""온톨로지 이관이 실제 오류 삽입 사본을 배정별 묶음으로 검사하는지 (F-SCH-03, 패키지 횡단).

`dlp_schema.migration`은 dlp_review를 import하지 않고 사본 ID 형식(`seed-<배정>-NNNN`)만 읽어
배정 묶음을 정한다. 이 테스트는 `dlp_review.ops.seeding.seed_labels`가 실제로 만든 사본으로 그
형식이 어긋나지 않았는지 지킨다. 형식이 어긋나면 사본마다 따로 묶여 사본 구간이 자기 배정의 사본
행동을 찾지 못하고 재검수로 간다.
"""

from __future__ import annotations

from dlp_review.ops.policy import SeedingPolicy
from dlp_review.ops.seeding import seed_labels
from dlp_schema import repo_root
from dlp_schema.migration import OntologyMigration, migrate_labels
from dlp_schema.ontology import Ontology, load_ontology
from dlp_schema.testing import FIXED_TIME, action_payload, make_label


def test_migration_keeps_operational_labels_and_real_seeded_copies() -> None:
    """운영 행동·구간과 두 배정의 사본(경계 이동 오류 포함)이 모두 이관된다.

    구간(0~5000ms)이 행동(2000~3000ms)을 넉넉히 감싸, 경계 이동(300~800ms)이 어느 쪽에 들어가도
    사본 구간은 사본 행동을 포함한다. 따라서 재검수로 갈 이유가 없다.
    """
    ontology = load_ontology(repo_root() / "config" / "ontology" / "v1")
    target = Ontology.model_validate(ontology.model_dump() | {"version": "1.1.0"})
    action = make_label(
        action_payload(
            t_approach_ms=2000, t_contact_start_ms=2400, t_contact_end_ms=2900, t_end_ms=3000
        ),
        label_id="act",
        t_start_ms=2000,
        t_end_ms=3000,
    )
    segment = make_label(
        {"kind": "segment", "segment_id": "sk", "level": "skill", "ref_id": "wipe",
         "child_ids": ["a001"]},
        label_id="sk",
        t_end_ms=5000,
    )  # fmt: skip
    policy = SeedingPolicy(
        errors_per_task=1,
        types=("boundary_shift",),
        boundary_shift_ms=(300, 800),
        detect_tolerance_ms=200,
        blur_overlap=0.5,
    )
    copies = [
        x
        for assignment in ("s001:temporal:seeded", "s001:temporal:seeded:2")
        for x in seed_labels(
            [action, segment],
            assignment_id=assignment,
            ontology=ontology,
            policy=policy,
            seed=7,
            now=FIXED_TIME,
        ).labels
    ]
    assert len(copies) == 4 and all(x.seeded_error for x in copies)
    labels = [action, segment, *copies]
    result = migrate_labels(
        labels,
        OntologyMigration(from_version="1.0.0", to_version="1.1.0"),
        FIXED_TIME,
        target_ontology=target,
    )
    assert result.needs_review == ()
    assert {x.parent_label_id for x in result.migrated} == {x.label_id for x in labels}
