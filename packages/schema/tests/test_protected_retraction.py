"""자동 단계가 사람이 확인한 라벨을 지우지 못하는지 (저장 시점 검사).

회귀 (Codex 수정본 검수, 2026-10-11): 프리라벨·행동·관계 단계는 이력을 잠금 없이 읽고 추론한다.
그 사이 검수자가 라벨을 승인하면, 단계는 낡은 이력으로 그 라벨을 검수 전으로 보고 삭제 레코드를
넣었다 (재현됨). 저장(`insert_labels`)은 세션 잠금 아래에서 최신 이력으로
`validate_session_labels`를 부르므로, 여기서 사람 출처가 아닌 삭제 레코드가 보호된 부모를 지우는
것을 거부한다.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest
import sqlalchemy as sa

from dlp_schema.db.repository import get_labels, insert_labels, record_review
from dlp_schema.description_follow import FOLLOW_VERSION
from dlp_schema.episode import current_labels, retractions
from dlp_schema.label_validation import ProtectedLabelRetractionError, validate_session_labels
from dlp_schema.labels import LabelRecord, Provenance, Source, Verification, VerificationState
from dlp_schema.ontology import Ontology
from dlp_schema.testing import FIXED_TIME, action_payload, make_label

MODEL = Provenance(source=Source.MODEL, model_version="actions-v1")


def _model_label(state: VerificationState = VerificationState.UNREVIEWED):
    """모델 출처 행동 라벨 (검수 상태 지정)."""
    reviewed = state is not VerificationState.UNREVIEWED
    verification = Verification(
        state=state,
        reviewer_id="rev01" if reviewed else None,
        reviewed_at=FIXED_TIME + timedelta(minutes=5) if reviewed else None,
    )
    return make_label(
        action_payload(), label_id="a1", provenance=MODEL, confidence=0.9, verification=verification
    )


@pytest.mark.parametrize(
    "state",
    [
        VerificationState.HUMAN_APPROVED,
        VerificationState.HUMAN_CORRECTED,
        VerificationState.SAMPLE_VERIFIED,
    ],
)
def test_model_retraction_of_reviewed_label_is_refused(
    ontology: Ontology, state: VerificationState
) -> None:
    """추론 중에 검수자가 확인한 라벨을 새 모델 버전이 지우려 하면 거부한다."""
    approved = _model_label(state)
    with pytest.raises(ProtectedLabelRetractionError, match="a1"):
        validate_session_labels(
            ontology, [approved], retractions([approved], "actions-v2", FIXED_TIME)
        )


def test_model_retraction_of_unreviewed_label_and_human_deletion_are_allowed(
    ontology: Ontology,
) -> None:
    """검수 전 모델 라벨의 버전 교체 삭제와, 검수자(사람 출처)의 삭제는 그대로 허용한다."""
    stale = _model_label()
    validate_session_labels(ontology, [stale], retractions([stale], "actions-v2", FIXED_TIME))
    approved = _model_label(VerificationState.HUMAN_APPROVED)
    [deletion] = retractions([approved], "review", FIXED_TIME)
    human_deletion = deletion.model_copy(update={"provenance": Provenance(source=Source.HUMAN)})
    validate_session_labels(ontology, [approved], [human_deletion])


# ---------------------------------------------------------------- 조상 사슬 (F-SCH-02)
#
# `current_labels`는 다른 레코드의 parent인 레코드를 뺀다. 승인된 원래 트랙에 모델 자식(3인칭
# 착용자 사본 `<원래>:wearer` 등)이 있으면 원래 트랙은 이미 현재 라벨이 아니고 자식이 그 자리를
# 대신한다. 모델 단계가 그 자식을 지우면 자식도, 원래 트랙도 현재 라벨에서 사라진다 (아래
# `test_model_child_retraction_really_removes_approved_lineage`가 이 조건을 고정한다).

BOX = {
    "kind": "box_track",
    "entity_id": "rag_01",
    "class_id": "rag",
    "keyframes": [{"t_ms": 0, "x": 1, "y": 1, "w": 5, "h": 5}],
}
APPROVED = Verification(
    state=VerificationState.HUMAN_APPROVED,
    reviewer_id="rev01",
    reviewed_at=FIXED_TIME + timedelta(minutes=5),
)


def _box(label_id: str, parent: str | None = None, **overrides: Any) -> LabelRecord:
    """3인칭 스트림 tp1의 모델 박스 트랙 (기본 미검수)."""
    fields: dict[str, Any] = {
        "label_id": label_id,
        "stream_id": "tp1",
        "provenance": MODEL,
        "confidence": 0.9,
        "parent_label_id": parent,
    }
    fields.update(overrides)
    return make_label(BOX, **fields)


def _ids(labels: list[LabelRecord]) -> set[str]:
    return {x.label_id for x in labels}


def test_model_child_retraction_really_removes_approved_lineage() -> None:
    """검사 없이 저장했다면: 승인 트랙 orig도 그 자식 copy도 운영 라벨에 남지 않는다."""
    orig, copy = _box("orig", verification=APPROVED), _box("copy", parent="orig")
    assert _ids(current_labels([orig, copy])) == {"copy"}
    gone = retractions([copy], "p2", FIXED_TIME)
    assert _ids(current_labels([orig, copy, *gone])) == set()


def test_model_retraction_of_last_child_of_approved_label_is_refused(ontology: Ontology) -> None:
    """승인 트랙의 마지막 현재 후손을 모델이 지우면 거부한다 (메시지는 승인 트랙을 가리킨다)."""
    orig, copy = _box("orig", verification=APPROVED), _box("copy", parent="orig")
    with pytest.raises(ProtectedLabelRetractionError, match=r"^copy:retracted: .*라벨 orig\("):
        validate_session_labels(ontology, [orig, copy], retractions([copy], "p2", FIXED_TIME))


def test_protected_ancestor_is_checked_through_the_whole_chain(ontology: Ontology) -> None:
    """보호 상태가 바로 위 부모가 아닌 더 위 조상에 있어도 막는다 (승인 → 모델 자식 → 손자)."""
    for state in (VerificationState.SAMPLE_VERIFIED, VerificationState.HUMAN_CORRECTED):
        verification = APPROVED.model_copy(update={"state": state})
        a = _box("a", verification=verification)
        b, c = _box("b", parent="a"), _box("c", parent="b")
        with pytest.raises(ProtectedLabelRetractionError, match=rf"라벨 a\({state.value}\)"):
            validate_session_labels(ontology, [a, b, c], retractions([c], "p2", FIXED_TIME))


def test_model_child_may_be_replaced_or_dropped_when_lineage_survives(ontology: Ontology) -> None:
    """같은 배치의 새 자식이 대신하거나 다른 현재 후손이 남으면 승인 트랙은 사라지지 않는다."""
    orig, copy = _box("orig", verification=APPROVED), _box("copy", parent="orig")
    # 버전 교체: 이전 사본을 지우고 같은 원래 트랙의 새 사본을 함께 넣는다
    fresh = _box("copy-v2", parent="orig")
    validate_session_labels(ontology, [orig, copy], [*retractions([copy], "p2", FIXED_TIME), fresh])
    # 다른 현재 후손(사람이 원래 트랙을 고친 레코드)이 남는다
    fix = _box("fix", parent="orig", provenance=Provenance(source=Source.HUMAN), confidence=None)
    validate_session_labels(ontology, [orig, copy, fix], retractions([copy], "p2", FIXED_TIME))


def test_unprotected_lineage_and_human_deletion_are_not_blocked(ontology: Ontology) -> None:
    """검수 전 조상(모델·사람이 만든 미검수)의 자식 정리와 검수자의 삭제는 막지 않는다."""
    human = Provenance(source=Source.HUMAN)
    for root in (_box("orig"), _box("orig", provenance=human, confidence=None)):
        copy = _box("copy", parent="orig")
        validate_session_labels(ontology, [root, copy], retractions([copy], "p2", FIXED_TIME))
    orig, copy = _box("orig", verification=APPROVED), _box("copy", parent="orig")
    [deletion] = retractions([copy], "review", FIXED_TIME)
    by_reviewer = deletion.model_copy(update={"provenance": human})
    validate_session_labels(ontology, [orig, copy], [by_reviewer])


def test_non_operational_descendant_does_not_stand_in_for_approved_label(
    ontology: Ontology,
) -> None:
    """측정 레코드(이중 라벨링)가 자식으로 남아도 운영 라벨에서는 승인 트랙이 사라지므로 막는다."""
    orig, copy = _box("orig", verification=APPROVED), _box("copy", parent="orig")
    double = _box(
        "double", parent="orig", provenance=Provenance(source=Source.HUMAN), confidence=None
    ).model_copy(update={"measurement": "double"})
    gone = retractions([copy], "p2", FIXED_TIME)
    assert _ids(current_labels([orig, copy, double, *gone])) == set()
    with pytest.raises(ProtectedLabelRetractionError, match=r"라벨 orig\("):
        validate_session_labels(ontology, [orig, copy, double], gone)


@pytest.mark.services
def test_storage_refuses_stale_child_retraction_after_parent_approval(pg: sa.Engine) -> None:
    """단계가 이력을 읽은 뒤 원래 트랙이 승인되면, 저장(`insert_labels`)이 사본 삭제를 거부한다."""
    orig, copy = _box("orig"), _box("copy", parent="orig")
    with pg.begin() as conn:
        insert_labels(conn, [orig, copy])
        stale = retractions([copy], "p2", FIXED_TIME)  # 검수 전 이력으로 고른 지울 후보
        record_review(conn, "orig", VerificationState.HUMAN_APPROVED, "rev01", FIXED_TIME)
        with pytest.raises(ProtectedLabelRetractionError, match=r"라벨 orig\("):
            insert_labels(conn, stale)
        assert _ids(get_labels(conn, "s001")) == {"orig", "copy"}


# -------------------------------- 설명 따르기의 좁은 예외 (ADR 0043 3절 규칙 6)
#
# 정답 근거: 검수자 r1이 행동 a1(action_id a001)을 승인했고(rev01, FIXED_TIME+5분), 설명 d·den이 그
# 검수를 물려받았다(같은 검수자·시각). 다음 회차에 검수자가 동사를 고친다(사람 레코드 fix). 설명
# 따르기(`FOLLOW_VERSION`)만, 물려받은 검수만 있는 설명만, 계획이 지우는 경우에만 지울 수 있다.


def _description(label_id: str, verification: Verification, **kw: Any) -> LabelRecord:
    payload: dict[str, Any] = {"kind": "description", "segment_id": "a001", "text": "설명"}
    if (source := kw.pop("source", None)) is not None:
        payload |= {"language": "en", "source_label_id": source, "text": "desc"}
    fields: dict[str, Any] = {"provenance": MODEL, "confidence": 0.9, **kw}
    return make_label(payload, label_id=label_id, verification=verification, **fields)


def _reverb(parent: LabelRecord) -> LabelRecord:
    """다음 회차 검수자의 의미 수정 (동사 grasp → lift)."""
    return parent.model_copy(
        update={"label_id": "fix", "parent_label_id": parent.label_id,
                "provenance": Provenance(source=Source.HUMAN), "confidence": None,
                "verification": APPROVED.model_copy(
                    update={"state": VerificationState.HUMAN_CORRECTED}),
                "payload": parent.payload.model_copy(update={"verb": "lift"})}
    )  # fmt: skip


def _inherited_scene() -> tuple[LabelRecord, list[LabelRecord]]:
    """(승인된 행동 a1, 물려받은 설명 d·den)."""
    owner = _model_label(VerificationState.HUMAN_APPROVED)
    inherited = owner.verification
    return owner, [_description("d", inherited), _description("den", inherited, source="d")]


def test_follow_may_retract_inheritance_only_description_after_owner_change(
    ontology: Ontology,
) -> None:
    """주인 의미가 다시 바뀌면 설명 따르기는 물려받은 설명(ko·en)을 지울 수 있다."""
    owner, descs = _inherited_scene()
    history = [owner, _reverb(owner), *descs]
    validate_session_labels(ontology, history, retractions(descs, FOLLOW_VERSION, FIXED_TIME))


def test_follow_exception_is_narrow(ontology: Ontology) -> None:
    """예외 밖은 모두 거부한다: 주인이 그대로인 설명, 다른 단계 버전의 삭제, 직접 검수(lead)·사람
    출처 설명, 설명이 아닌 보호 라벨(행동)."""
    owner, descs = _inherited_scene()
    fix = _reverb(owner)
    refuse = pytest.raises(ProtectedLabelRetractionError)
    with refuse:  # 주인이 그대로다 (계획이 지우지 않는다)
        validate_session_labels(
            ontology, [owner, *descs], retractions(descs, FOLLOW_VERSION, FIXED_TIME)
        )
    with refuse:  # 설명 따르기가 아닌 단계
        validate_session_labels(
            ontology, [owner, fix, *descs], retractions(descs, "translate-x", FIXED_TIME)
        )
    lead = APPROVED.model_copy(update={"reviewer_id": "lead"})
    human = Provenance(source=Source.HUMAN)
    for d in (
        _description("dl", lead),
        _description("dh", owner.verification, provenance=human, confidence=None),
    ):
        with refuse:
            validate_session_labels(
                ontology, [owner, fix, d], retractions([d], FOLLOW_VERSION, FIXED_TIME)
            )
    with refuse:  # 설명이 아닌 보호 라벨
        validate_session_labels(
            ontology, [owner, *descs], retractions([owner], FOLLOW_VERSION, FIXED_TIME)
        )
