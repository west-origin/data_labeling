"""세션 저장 전 순수 검증. 잠금·조회·INSERT는 저장소가 맡는다."""

from dlp_schema.episode import current_labels
from dlp_schema.labels import LabelRecord
from dlp_schema.ontology import Ontology
from dlp_schema.validation import OntologyViolationError, check_label_set, validate_label


def validate_session_labels(
    ontology: Ontology,
    history: list[LabelRecord],
    incoming: list[LabelRecord],
) -> None:
    """새 라벨 의미와 저장 후 현재 구간·부분 참조를 검사한다.

    seeded_error는 시험용 의미 오류를 허용하되 버전은 일치해야 한다. 삭제는 부모 페이로드를
    그대로 복사해야 하므로 기존 오류 정리는 가능하지만 새 오류를 삭제로 위장할 수 없다.
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
        elif not label.seeded_error:
            validate_label(label, ontology)
    problems = check_label_set(current_labels([*history, *incoming]), ontology)
    if problems:
        raise OntologyViolationError(incoming[0].label_id, problems)
