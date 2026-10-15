"""세션 저장 전 순수 검증. 잠금·조회·INSERT는 저장소가 맡는다."""

from collections.abc import Sequence

from dlp_schema.description_follow import (
    FOLLOW_VERSION,
    inheritance_only,
    plan_description_follow,
)
from dlp_schema.episode import current_labels, non_operational_ids
from dlp_schema.labels import DescriptionPayload, LabelRecord, Source, VerificationState
from dlp_schema.ontology import Ontology
from dlp_schema.review import DescriptionInheritance
from dlp_schema.validation import OntologyViolationError, check_label_set, validate_label

# 사람이 확인한 라벨 상태. 이 상태의 라벨은 자동 단계가 지울 수 없다 (ADR 0015, 0019).
# 그 라벨을 대신하는 마지막 현재 후손을 지우는 것도 막는다 (F-SCH-02).
# 사람이 만든 미검수 라벨은 여기서 막지 않는다 (기존 정리 경로가 쓴다).
# 예외: 설명 따르기가 물려받은 검수만 있는 설명을 지울 때 (`_FollowExemption`, ADR 0043 3절 규칙 6)
PROTECTED_STATES = (
    VerificationState.HUMAN_APPROVED,
    VerificationState.HUMAN_CORRECTED,
    VerificationState.SAMPLE_VERIFIED,
)


class ProtectedLabelRetractionError(OntologyViolationError):
    """자동 단계(모델·센서 출처)의 삭제 레코드가 사람이 확인한 라벨을 지우려 한다.

    두 경우를 막는다.

    1. 지우려는 레코드 자체가 사람이 확인한 상태다.
    2. 지우려는 레코드의 조상 사슬에 사람이 확인한 라벨이 있고, 이 저장으로 그 조상을 대신할
       현재 후손이 하나도 남지 않는다 (F-SCH-02). `current_labels`는 자식이 있는 레코드를 빼므로,
       승인된 원래 트랙의 모델 자식(3인칭 착용자 사본 등)을 지우면 원래 트랙도 현재 라벨에서
       사라진다.

    자동 단계는 이력을 읽고 추론하는 동안 세션을 잠그지 않는다. 그 사이 검수자가 라벨을 승인하면
    단계는 낡은 이력으로 그 라벨을 "검수 전"으로 보고 지우려 한다 (Codex 수정본 검수, 2026-10-11
    재현).
    저장 시점에는 세션 잠금 아래에서 최신 검수 상태를 보므로 여기서 거부한다. 단계는 다시 실행하면
    새 이력으로 그 라벨을 보존한다 (단계의 지울 후보 선택이 보호 조상을 걸러야 한다).

    좁은 예외 하나 (ADR 0043 3절 규칙 6, `_follow_may_retract`): 설명 따르기(`FOLLOW_VERSION`)가
    물려받은 검수만 있는 설명을, 같은 이력으로 다시 계산한 설명 따르기 계획이 지울 때. 그 밖의
    보호 라벨(사람 출처, 직접 검수, 설명이 아닌 라벨)은 그대로 막는다.
    """


def validate_session_labels(
    ontology: Ontology,
    history: list[LabelRecord],
    incoming: list[LabelRecord],
    inheritance: Sequence[DescriptionInheritance] = (),
) -> None:
    """새 라벨 의미와 저장 후 현재 구간·부분 참조를 검사한다.

    seeded_error는 시험용 의미 오류를 허용하되 버전은 일치해야 한다. 삭제는 부모 페이로드를
    그대로 복사해야 하므로 기존 오류 정리는 가능하지만 새 오류를 삭제로 위장할 수 없다.
    사람 출처가 아닌 삭제 레코드는 사람이 확인한(`PROTECTED_STATES`) 부모를 지울 수 없고,
    사람이 확인한 조상의 마지막 현재 후손도 지울 수 없다 (`ProtectedLabelRetractionError`).
    예외는 설명 따르기가 물려받은 검수만 있는 설명을 지울 때뿐이다 (`_FollowExemption`).
    `history`는 호출자가 세션 잠금 아래에서 읽은 최신 이력이어야 한다. `inheritance`는 같은
    잠금 아래 읽은 세션의 물려받기 기록이다 (예외 판정에만 쓴다. 비면 예외가 없다).
    """
    if not incoming:
        return
    parents = {x.label_id: x for x in [*history, *incoming]}
    follow = _FollowExemption(history, inheritance)
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
                and not follow.allows(label, parent)
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
    _check_protected_lineage(history, incoming)
    problems = check_label_set(current_labels([*history, *incoming]), ontology)
    if problems:
        raise OntologyViolationError(incoming[0].label_id, problems)


class _FollowExemption:
    """보호 라벨 삭제 거부의 좁은 예외 (ADR 0043 3절 규칙 6).

    허용 조건을 모두 만족해야 한다.

    1. 삭제 레코드가 설명 따르기의 것이다: 모델 출처, `model_version == FOLLOW_VERSION`.
    2. 지우는 레코드가 설명 라벨이고 보호가 물려받은 검수뿐이다 (`inheritance_only`: 지금 검수
       상태가 명시 물려받기 기록 `description_inheritance`에서 왔다. 검수자·시각이 우연히 주인과
       같은 직접 검수는 아니다, F-PAR-20).
    3. 저장 전 이력(세션 잠금 아래 최신)으로 다시 계산한 설명 따르기 계획이 그 설명을 지운다
       (주인이 다시 바뀌었거나 지워졌다). 계획은 처음 필요할 때 한 번만 계산한다.
    """

    def __init__(
        self, history: list[LabelRecord], inheritance: Sequence[DescriptionInheritance]
    ) -> None:
        self.history = history
        self.inheritance = inheritance
        self._planned: set[str] | None = None

    def allows(self, label: LabelRecord, parent: LabelRecord) -> bool:
        """label(삭제 레코드)이 보호된 parent를 지워도 되는가."""
        if (
            label.provenance.source is not Source.MODEL
            or label.provenance.model_version != FOLLOW_VERSION
            or not isinstance(parent.payload, DescriptionPayload)
            or not inheritance_only(parent, self.inheritance)
        ):
            return False
        if self._planned is None:
            plan = plan_description_follow(self.history, self.inheritance)
            self._planned = {x.label_id for x in plan.retract}
        return parent.label_id in self._planned


def _protected_ancestors(
    incoming: list[LabelRecord], by_id: dict[str, LabelRecord]
) -> dict[str, str]:
    """자동 단계 삭제 레코드마다 지우는 레코드의 조상 사슬에서 사람이 확인한 라벨을 모은다.

    바로 위 부모(지우는 레코드 자체)는 `validate_session_labels`가 따로 거부하므로 그 위부터 본다.

    Returns:
        보호 조상 label_id → 그 조상을 건드린 (첫) 삭제 레코드 label_id.
    """
    out: dict[str, str] = {}
    for label in incoming:
        if not label.retracted or label.provenance.source is Source.HUMAN:
            continue
        target = by_id.get(label.parent_label_id or "")
        seen: set[str] = set()
        node = by_id.get(target.parent_label_id or "") if target is not None else None
        # 사슬은 보통 짧지만 잘못된 이력의 순환에 대비해 방문한 ID에서 멈춘다
        while node is not None and node.label_id not in seen:
            seen.add(node.label_id)
            if node.verification.state in PROTECTED_STATES:
                out.setdefault(node.label_id, label.label_id)
            node = by_id.get(node.parent_label_id or "")
    return out


def _represented(labels: list[LabelRecord], targets: set[str]) -> set[str]:
    """`labels`를 이력으로 볼 때 자신이나 후손 중 현재 라벨이 남은 대상 ID.

    운영 대상(오류 삽입·측정 계보가 아닌 라벨)은 운영 후손만 센다. 측정·오류 삽입 레코드가
    자식으로 남아도 `current_labels()`(운영 라벨)에서는 사라지기 때문이다.
    """
    current = {x.label_id for x in current_labels(labels, operational=False)}
    excluded = non_operational_ids(labels)
    children: dict[str, list[str]] = {}
    for x in labels:
        if x.parent_label_id:
            children.setdefault(x.parent_label_id, []).append(x.label_id)
    out: set[str] = set()
    for target in targets:
        stack, seen = [target], set[str]()
        while stack:
            node = stack.pop()
            if node in current and (target in excluded or node not in excluded):
                out.add(target)
                break
            seen.add(node)
            stack.extend(c for c in children.get(node, ()) if c not in seen)
    return out


def _check_protected_lineage(history: list[LabelRecord], incoming: list[LabelRecord]) -> None:
    """이 저장으로 사람이 확인한 조상이 현재 라벨에서 사라지면 거부한다 (F-SCH-02).

    저장 전에는 현재 후손이 있었고 저장 후에는 없는 보호 조상만 위반이다 (이미 다른 경로로
    사라진 조상은 이번 저장 탓이 아니다). 같은 배치의 새 자식(버전 교체 사본)이나 남아 있는 다른
    현재 후손(사람의 수정본 등)이 그 조상을 대신하면 통과한다.

    Raises:
        ProtectedLabelRetractionError: 위반한 첫 보호 조상의 삭제 레코드 ID와 함께.
    """
    every = [*history, *incoming]
    by_id = {x.label_id: x for x in every}
    guarded = _protected_ancestors(incoming, by_id)
    if not guarded:
        return
    lost = _represented(history, set(guarded)) - _represented(every, set(guarded))
    if not lost:
        return
    ancestor = min(lost)
    state = by_id[ancestor].verification.state.value
    raise ProtectedLabelRetractionError(
        guarded[ancestor],
        [
            f"자동 단계가 사람이 확인한 라벨 {ancestor}({state})의 마지막 현재 후손을 지워 "
            "그 라벨이 현재 라벨에서 사라진다 — 단계를 다시 실행하면 최신 검수 상태로 보존한다"
        ],
    )
