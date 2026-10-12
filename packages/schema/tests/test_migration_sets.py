"""온톨로지 이관의 집합 검사 묶음 (F-SCH-03).

이관 후 구간·부분 연결 검사(`check_label_set`)는 운영 라벨, 오류 삽입 배정별 사본, 측정 레코드
(방식·검수자별)를 따로 돈다. 오류 삽입 사본은 원래 라벨의 action_id·segment_id를 그대로 쓰므로
(`dlp_review.ops.seeding`: ID만 바꾼 사본) 한 묶음에 섞으면 "ID 중복"이 되어 운영 라벨까지 재검수로
갔다. 정답은 합성 이력으로 정했다: 어느 묶음도 다른 묶음의 결과를 바꾸지 않는다.
"""

from __future__ import annotations

from typing import Any

from dlp_schema.labels import LabelRecord, Provenance, Source, Verification, VerificationState
from dlp_schema.migration import MigrationResult, OntologyMigration, migrate_labels
from dlp_schema.ontology import Ontology
from dlp_schema.testing import FIXED_TIME, action_payload, make_label

MIGRATION = OntologyMigration(from_version="1.0.0", to_version="1.1.0")


def _action(label_id: str, **overrides: Any) -> LabelRecord:
    """오른손 원시 동작 a001 (0~1000ms). overrides는 레코드 필드, verb는 페이로드."""
    verb = overrides.pop("verb", "grasp")
    return make_label(action_payload(verb=verb), label_id=label_id, **overrides)


def _skill(label_id: str, **overrides: Any) -> LabelRecord:
    """a001을 묶는 기술 구간 sk (0~1000ms)."""
    payload = {
        "kind": "segment",
        "segment_id": "sk",
        "level": "skill",
        "ref_id": "wipe",
        "child_ids": ["a001"],
    }
    return make_label(payload, label_id=label_id, **overrides)


def _seeded(assignment: str) -> list[LabelRecord]:
    """오류 삽입 배정 하나의 사본 (seeding.seed_labels 형식: `seed-<배정>-NNNN`, parent 없음)."""
    return [
        _action(f"seed-{assignment}-0000", seeded_error=True),
        _skill(f"seed-{assignment}-0001", seeded_error=True),
    ]


def _parents(result: MigrationResult) -> set[str]:
    """이관된 새 레코드의 원래 라벨 ID."""
    return {x.parent_label_id or "" for x in result.migrated}


def _run(labels: list[LabelRecord], target: Ontology, m: OntologyMigration = MIGRATION):
    return migrate_labels(labels, m, FIXED_TIME, target_ontology=target)


def test_seeded_copies_do_not_block_operational_labels(target_ontology: Ontology) -> None:
    """운영 라벨과 두 배정의 사본(같은 action_id·segment_id)이 모두 이관된다."""
    operational = [_action("act"), _skill("sk")]
    labels = [*operational, *_seeded("as1"), *_seeded("u1:seeded")]
    result = _run(labels, target_ontology)
    assert result.needs_review == ()
    assert _parents(result) == {x.label_id for x in labels}
    # 표시는 그대로라 이관 후에도 운영 라벨이 아니다
    assert {x.parent_label_id for x in result.migrated if x.seeded_error} == {
        "seed-as1-0000",
        "seed-as1-0001",
        "seed-u1:seeded-0000",
        "seed-u1:seeded-0001",
    }


def test_reviewer_fixes_and_new_drawings_stay_in_their_assignment(
    target_ontology: Ontology,
) -> None:
    """사본을 고친 레코드(사본의 후손)와 수집 때 새로 그린 레코드(`seed-<배정>-new-<해시>`)는
    그 배정 묶음에서 검사된다. 고친 구간이 같은 배정의 사본 행동을 가리켜도 통과한다."""
    reviewer = Verification(
        state=VerificationState.HUMAN_CORRECTED, reviewer_id="rev01", reviewed_at=FIXED_TIME
    )
    copies = _seeded("as1")
    fix = _skill(
        "seed-as1-0001:rabc",
        seeded_error=True,
        parent_label_id="seed-as1-0001",
        verification=reviewer,
    )
    drawn = make_label(
        action_payload(action_id="a002", hand="left"),
        label_id="seed-as1-new-0123456789abcdef",
        seeded_error=True,
        verification=reviewer,
    )
    labels = [_action("act"), _skill("sk"), *copies, fix, drawn]
    result = _run(labels, target_ontology)
    assert result.needs_review == ()
    assert _parents(result) == {"act", "sk", "seed-as1-0000", "seed-as1-0001:rabc", drawn.label_id}


def test_broken_seeded_assignment_goes_to_review_alone(target_ontology: Ontology) -> None:
    """한 배정의 사본 행동이 제거된 동사라 빠지면 그 배정의 구간만 재검수로 간다.

    다른 배정과 운영 라벨은 그대로 이관된다 (운영 쪽 행동은 다른 동사라 제거 대상이 아니다).
    """
    removal = OntologyMigration(
        from_version="1.0.0", to_version="1.1.0", removed={"verbs": ("lift",)}
    )
    broken = [
        _action("seed-as1-0000", seeded_error=True, verb="lift"),
        _skill("seed-as1-0001", seeded_error=True),
    ]
    labels = [_action("act"), _skill("sk"), *broken, *_seeded("as2")]
    result = _run(labels, target_ontology, removal)
    assert {label_id for label_id, _ in result.needs_review} == {"seed-as1-0000", "seed-as1-0001"}
    assert _parents(result) == {"act", "sk", "seed-as2-0000", "seed-as2-0001"}


def test_broken_operational_batch_does_not_hold_back_seeded_copies(
    target_ontology: Ontology,
) -> None:
    """운영 묶음이 깨져 재검수로 가도 사본 묶음은 이관된다 (반대 방향도 독립)."""
    removal = OntologyMigration(
        from_version="1.0.0", to_version="1.1.0", removed={"verbs": ("lift",)}
    )
    labels = [_action("act", verb="lift"), _skill("sk"), *_seeded("as1")]
    result = _run(labels, target_ontology, removal)
    assert {label_id for label_id, _ in result.needs_review} == {"act", "sk"}
    assert _parents(result) == {"seed-as1-0000", "seed-as1-0001"}


def test_measurement_records_are_checked_per_mode_and_reviewer(target_ontology: Ontology) -> None:
    """블라인드·이중 측정 레코드는 운영 라벨과 같은 ID를 쓰더라도 (방식, 검수자)별로 검사한다."""
    human = Provenance(source=Source.HUMAN)

    def measured(label_id: str, mode: str, reviewer: str, *, segment: bool) -> LabelRecord:
        verification = Verification(
            state=VerificationState.HUMAN_CORRECTED, reviewer_id=reviewer, reviewed_at=FIXED_TIME
        )
        make = _skill if segment else _action
        return make(label_id, measurement=mode, provenance=human, verification=verification)

    labels = [
        _action("act"),
        _skill("sk"),
        measured("d1-a", "double", "rev02", segment=False),
        measured("d1-s", "double", "rev02", segment=True),
        measured("d2-a", "double", "rev03", segment=False),
        measured("d2-s", "double", "rev03", segment=True),
        measured("b1-a", "blind", "rev02", segment=False),
        measured("b1-s", "blind", "rev02", segment=True),
    ]
    result = _run(labels, target_ontology)
    assert result.needs_review == ()
    assert _parents(result) == {x.label_id for x in labels}


def test_already_migrated_context_is_grouped_the_same_way(target_ontology: Ontology) -> None:
    """이미 새 버전인 현재 라벨(앞선 이관 결과)도 같은 묶음 규칙으로 참조 문맥이 된다.

    행동(운영·사본)만 먼저 이관된 상태에서 구간을 이관하면, 각 구간은 자기 묶음에서 이관된 행동을
    자식으로 찾는다 (두 묶음의 이관된 행동이 같은 action_id라도 중복이 아니다).
    """
    first = _run([_action("act"), _seeded("as1")[0]], target_ontology)
    assert first.needs_review == ()
    labels = [_action("act"), _skill("sk"), *_seeded("as1"), *first.migrated]
    second = _run(labels, target_ontology)
    assert second.needs_review == ()
    assert _parents(second) == {"sk", "seed-as1-0001"}
