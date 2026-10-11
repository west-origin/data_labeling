"""자동 단계가 사람이 확인한 라벨을 지우지 못하는지 (저장 시점 검사).

회귀 (Codex 수정본 검수, 2026-10-11): 프리라벨·행동·관계 단계는 이력을 잠금 없이 읽고 추론한다.
그 사이 검수자가 라벨을 승인하면, 단계는 낡은 이력으로 그 라벨을 검수 전으로 보고 삭제 레코드를
넣었다 (재현됨). 저장(`insert_labels`)은 세션 잠금 아래에서 최신 이력으로
`validate_session_labels`를 부르므로, 여기서 사람 출처가 아닌 삭제 레코드가 보호된 부모를 지우는
것을 거부한다.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from dlp_schema.episode import retractions
from dlp_schema.label_validation import ProtectedLabelRetractionError, validate_session_labels
from dlp_schema.labels import Provenance, Source, Verification, VerificationState
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
