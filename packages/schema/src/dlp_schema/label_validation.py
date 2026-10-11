"""세션 저장 전 순수 검증. 잠금·조회·INSERT는 저장소가 맡는다."""

from dlp_schema.episode import current_labels
from dlp_schema.labels import LabelRecord, Source, VerificationState
from dlp_schema.ontology import Ontology
from dlp_schema.validation import OntologyViolationError, check_label_set, validate_label

# 사람이 확인한 라벨 상태. 이 상태의 라벨은 자동 단계가 지울 수 없다 (ADR 0015, 0019).
# 사람이 만든 미검수 라벨은 여기서 막지 않는다 (기존 정리 경로가 쓴다)
PROTECTED_STATES = (
    VerificationState.HUMAN_APPROVED,
    VerificationState.HUMAN_CORRECTED,
    VerificationState.SAMPLE_VERIFIED,
)


class ProtectedLabelRetractionError(OntologyViolationError):
    """자동 단계(모델·센서 출처)의 삭제 레코드가 사람이 확인한 라벨을 지우려 한다.

    자동 단계는 이력을 읽고 추론하는 동안 세션을 잠그지 않는다. 그 사이 검수자가 라벨을 승인하면
    단계는 낡은 이력으로 그 라벨을 "검수 전"으로 보고 지우려 한다 (Codex 수정본 검수, 2026-10-11
    재현).
    저장 시점에는 세션 잠금 아래에서 최신 검수 상태를 보므로 여기서 거부한다. 단계는 다시 실행하면
    새 이력으로 그 라벨을 보존한다.
    """


def validate_session_labels(
    ontology: Ontology,
    history: list[LabelRecord],
    incoming: list[LabelRecord],
) -> None:
    """새 라벨 의미와 저장 후 현재 구간·부분 참조를 검사한다.

    seeded_error는 시험용 의미 오류를 허용하되 버전은 일치해야 한다. 삭제는 부모 페이로드를
    그대로 복사해야 하므로 기존 오류 정리는 가능하지만 새 오류를 삭제로 위장할 수 없다.
    사람 출처가 아닌 삭제 레코드는 사람이 확인한(`PROTECTED_STATES`) 부모를 지울 수 없다
    (`ProtectedLabelRetractionError`). `history`는 호출자가 세션 잠금 아래에서 읽은 최신 이력이어야
    한다.
    """
    if not incoming:
        return
    parents = {x.label_id: x for x in [*history, *incoming]}
    for label in incoming:
        if not label.retracted and label.ontology_version != ontology.version:
            raise OntologyViolationError(label.label_id, ["세션과 라벨의 온톨로지 버전 불일치"])
        if label.retracted:
            parent = parents.get(label.parent_label_id or "")
            if (
                parent is None
                or parent.payload != label.payload
                or parent.ontology_version != label.ontology_version
            ):
                raise OntologyViolationError(label.label_id, ["삭제 레코드의 부모·페이로드 불일치"])
            if (
                label.provenance.source is not Source.HUMAN
                and parent.verification.state in PROTECTED_STATES
            ):
                raise ProtectedLabelRetractionError(
                    label.label_id,
                    [
                        f"자동 단계가 사람이 확인한 라벨 {parent.label_id}"
                        f"({parent.verification.state.value})을 지우려 한다 — 단계를 다시 실행하면 "
                        "최신 검수 상태로 보존한다"
                    ],
                )
        elif not label.seeded_error:
            validate_label(label, ontology)
    problems = check_label_set(current_labels([*history, *incoming]), ontology)
    if problems:
        raise OntologyViolationError(incoming[0].label_id, problems)
