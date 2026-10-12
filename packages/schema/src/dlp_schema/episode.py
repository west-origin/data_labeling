"""에피소드 그래프: 개체·관계·이벤트·상태 전이 네 층으로 라벨을 묶는다.

역할
    1) `EpisodeGraph`: 세션의 한 구간(에피소드)의 개체 목록과 현재 라벨을 묶고, 라벨이 참조하는
       개체 ID와 상위 구간의 자식·계층·시간 연결을 검증한다. 사전 의미와 부분 소속은
       `validation.validate_episode`가 더 검사한다. 관계·이벤트·커버리지 층과 상태 전이를 꺼내 보는
       도우미를 둔다 (WP1, 기준 문서 "에피소드 그래프").
    2) 라벨 이력 공용 함수: 모든 단계가 다른 단계의 입력을 고를 때 쓰는 `current_labels`, 운영
       라벨이 아닌 레코드를 찾는 `non_operational_ids`, 모델 버전 교체 때 쓰는
       `version_tag`·`retractions`, 검수된 결과 판별 `is_protected`, 같은 대상 트랙이 여럿일 때
       하나를 고르는 `track_rank`·`pick_hand_track_labels` (프리라벨 접촉·3D 단계와 행동 단계가
       같은 기준을 쓴다, F-PAR-04).

주요 이름
    - `EntityKind`, `Entity`, `StateTransition`, `EpisodeGraph`
    - `entity_refs(label)`: 라벨이 참조하는 개체 ID.
    - `non_operational_ids(labels)`: 오류 삽입·측정 레코드와 그 후손의 ID.
    - `current_labels(labels, operational=True)`: 운영 현재 라벨
      (CLAUDE.md: 다른 단계 입력은 이것으로).
    - `version_tag(model_version)`: 모델 출처 라벨 ID에 넣는 8자 해시.
    - `retractions(stale, model_version, now)`: 이전 버전 모델 라벨을 지우는 삭제 레코드.
    - `is_protected(label)`: 사람이 만들었거나 고친, 또는 검수(승인·표본 검증)된 라벨.
    - `track_rank(label)`: 같은 대상 트랙 중 하나를 고르는 정렬 키.
    - `pick_hand_track_labels(labels)`: 손마다 hand21 트랙 하나 (`track_rank` 기준).

멱등·불변 (ADR 0015, 0019)
    라벨은 덮어쓰지 않는다. 각 단계는 다시 돌지 여부를 현재 라벨이 아니라 전체 이력
    (`get_labels`)으로 정하고, 모델 버전이 바뀌면 검수 전인 이전 버전 라벨만 `retractions`로 지운다.
    검수자가 승인·표본 검증한 라벨은 어떤 단계도 지우지 않는다 (그 판단은 호출자 몫이다.
    `retractions`는 받은 것을 모두 지운다).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from datetime import datetime
from enum import StrEnum
from itertools import pairwise

from pydantic import Field, model_validator

from dlp_schema.common import Contract, Identifier, Ms, OntologyId, SemVer, derived_id
from dlp_schema.label_graph import check_segment_links
from dlp_schema.labels import (
    ActionPayload,
    BoxTrackPayload,
    CoveragePayload,
    EventPayload,
    GapPayload,
    Hand,
    HandStatePayload,
    KeypointTrackPayload,
    LabelRecord,
    MaskTrackPayload,
    ObjectStatePayload,
    Provenance,
    RelationPayload,
    SegmentPayload,
    Source,
    Trajectory3DPayload,
    Verification,
    VerificationState,
)


# 에피소드 그래프 개체 종류.
class EntityKind(StrEnum):
    HAND = "hand"  # 작업자의 손 (예: left_hand, right_hand)
    OBJECT = "object"  # 일반 객체
    TOOL = "tool"  # 도구 (작용부·파지부가 있는 객체)
    PERSON = "person"  # 대상자 등 사람
    SURFACE = "surface"  # 처리 대상 표면 (청결·커버리지 기록)
    CAMERA = "camera"  # 카메라 (자세 궤적의 entity_id="camera")


# 개체 하나. entity_id: 세션 안에서 고유한 개체 ID (예: rag_01). class_id: 온톨로지 객체 클래스
# (손·카메라는 None 가능).
class Entity(Contract):
    entity_id: Identifier
    kind: EntityKind
    class_id: OntologyId | None = None


# 객체 상태 전이 (object_state 구간들에서 도출, 저장하지 않는다).
#   entity_id / attribute: 개체와 상태 속성. from_value → to_value: 바뀌기 전·후 값.
#   t_ms: 새 값 구간이 시작한 시각 (마스터 타임라인 ms).
class StateTransition(Contract):
    entity_id: Identifier
    attribute: OntologyId
    from_value: OntologyId
    to_value: OntologyId
    t_ms: Ms


# 에피소드 하나 (세션의 한 구간).
#   episode_id / session_id / ontology_version: 식별과 온톨로지 버전 (모든 라벨이 같은
#     세션·버전이어야 한다).
#   t_start_ms / t_end_ms: 에피소드 구간 (마스터 ms).
#     라벨은 이 구간과 겹쳐야 한다 (걸치는 것은 허용).
#   entities: 개체 목록 (ID 중복 금지). labels: 현재 라벨
#     (삭제 레코드 금지, 개체 참조는 entities 안).
# 주의: 공간 라벨 구간은 스트림 PTS ms라, 바디캠이 아닌 스트림의 공간 라벨을 넣으면 구간 비교가
# 어긋날 수 있다.
class EpisodeGraph(Contract):
    episode_id: Identifier
    session_id: Identifier
    ontology_version: SemVer
    t_start_ms: Ms
    t_end_ms: Ms
    entities: tuple[Entity, ...]
    labels: tuple[LabelRecord, ...] = Field(description="현재 유효한(대체·삭제되지 않은) 라벨")

    @model_validator(mode="after")
    def _check(self) -> EpisodeGraph:
        """개체·라벨 ID 중복, 세션·버전·삭제·구간·개체 참조와 구간 자식 연결을 검사한다.

        Raises:
            ValueError: 개체 ID 중복이면 즉시, 그 밖의 위반은 모두 모아 `; `로 이은 메시지.
        """
        ids = {e.entity_id for e in self.entities}
        if len(ids) != len(self.entities):
            raise ValueError("entity_id가 중복되었습니다")
        problems: list[str] = []
        if len({label.label_id for label in self.labels}) != len(self.labels):
            problems.append("label_id가 중복되었습니다")
        for label in self.labels:
            if label.session_id != self.session_id:
                problems.append(f"{label.label_id}: 다른 세션의 라벨")
            if label.ontology_version != self.ontology_version:
                problems.append(f"{label.label_id}: 온톨로지 버전 불일치")
            if label.retracted:
                problems.append(f"{label.label_id}: 삭제 레코드는 그래프에 넣지 않습니다")
            # 구간이 전혀 겹치지 않을 때만 위반 (끝점이 닿는 것은 겹침으로 본다)
            if label.t_end_ms < self.t_start_ms or label.t_start_ms > self.t_end_ms:
                problems.append(f"{label.label_id}: 에피소드 구간 밖의 라벨")
            problems.extend(
                f"{label.label_id}: 알 수 없는 개체 {ref}"
                for ref in entity_refs(label)
                if ref not in ids
            )
        problems.extend(check_segment_links(self.labels))
        if problems:
            raise ValueError("; ".join(problems))
        return self

    # ------------------------------------------------------------ 층별 보기

    @property
    def relations(self) -> list[LabelRecord]:
        """관계 층: relation 라벨 (입력 순서)."""
        return [x for x in self.labels if isinstance(x.payload, RelationPayload)]

    @property
    def events(self) -> list[LabelRecord]:
        """이벤트 층: 행동·상위 구간·공백·이벤트 라벨을 시작 시각 순으로 (같으면 입력 순서)."""
        kinds = (ActionPayload, SegmentPayload, GapPayload, EventPayload)
        return sorted(
            (x for x in self.labels if isinstance(x.payload, kinds)), key=lambda x: x.t_start_ms
        )

    @property
    def coverage(self) -> list[LabelRecord]:
        """커버리지 라벨 (표면을 도구가 처리한 비율, 입력 순서)."""
        return [x for x in self.labels if isinstance(x.payload, CoveragePayload)]

    def state_transitions(self) -> list[StateTransition]:
        """같은 개체·속성의 상태 구간을 시간순으로 이어 전이를 만든다.

        (개체, 속성)별로 object_state 구간을 시작 시각 순으로 정렬하고, 이웃한 두 구간의 값이 다르면
        뒤 구간의 시작 시각에 전이 하나를 만든다. 값이 같은 연속 구간은 전이가 아니다.
        구간 사이 공백이나 겹침은 따지지 않는다.

        Returns:
            전이 목록 (시각 순).
        """
        series: dict[tuple[str, str], list[tuple[int, str]]] = {}
        for x in self.labels:
            p = x.payload
            if isinstance(p, ObjectStatePayload):
                series.setdefault((p.entity_id, p.attribute), []).append((x.t_start_ms, p.value))
        out: list[StateTransition] = []
        for (entity_id, attribute), points in series.items():
            # (시작 시각, 값) 튜플 정렬: 시각이 같으면 값 문자열 순 (결정적이지만 의미는 없다)
            points.sort()
            for (_, prev), (t, cur) in pairwise(points):
                if prev != cur:
                    out.append(
                        StateTransition(
                            entity_id=entity_id,
                            attribute=attribute,
                            from_value=prev,
                            to_value=cur,
                            t_ms=t,
                        )
                    )
        return sorted(out, key=lambda s: s.t_ms)


def entity_refs(label: LabelRecord) -> list[str]:
    """라벨이 참조하는 개체 ID 목록.

    종류별로 개체를 가리키는 필드(entity_id, target_id, tool_id, surface_id, related_entity_id,
    subject_id/object_id)에서 값이 있는 것만 모은다. 블러·상위
    구간·공백·설명은 개체를 참조하지 않는다.
    """
    p = label.payload
    match p:
        case BoxTrackPayload() | MaskTrackPayload() | KeypointTrackPayload():
            return [p.entity_id]
        case Trajectory3DPayload():
            return [p.entity_id]
        case HandStatePayload():
            return [p.target_id] if p.target_id else []
        case ActionPayload():
            return [r for r in (p.target_id, p.tool_id) if r]
        case ObjectStatePayload():
            return [p.entity_id]
        case CoveragePayload():
            return [r for r in (p.surface_id, p.tool_id) if r]
        case EventPayload():
            return [p.related_entity_id] if p.related_entity_id else []
        case RelationPayload():
            return [p.subject_id, p.object_id]
        case _:
            return []


def non_operational_ids(labels: list[LabelRecord]) -> set[str]:
    """운영 라벨이 아닌 레코드: 오류 삽입 레코드와 그 후손(검수자가 고친 것 포함), 측정용 레코드.

    측정용 레코드(measurement)의 후손도 포함된다. `parent_label_id` 사슬을 아래로 따라가며
    (깊이 우선, 스택) 모두 모은다.

    Args:
        labels: 세션의 라벨 이력.

    Returns:
        제외할 label_id 집합.
    """
    out = {x.label_id for x in labels if x.seeded_error or x.measurement is not None}
    # 부모 → 자식 목록 색인을 만들고 시작 집합에서 아래로 퍼뜨린다
    children: dict[str, list[str]] = {}
    for x in labels:
        if x.parent_label_id:
            children.setdefault(x.parent_label_id, []).append(x.label_id)
    stack = list(out)
    while stack:
        for child in children.get(stack.pop(), []):
            if child not in out:
                out.add(child)
                stack.append(child)
    return out


def current_labels(labels: list[LabelRecord], *, operational: bool = True) -> list[LabelRecord]:
    """수정 이력에서 최신 레코드만 남긴다. 다른 레코드의 parent인 레코드와 삭제 레코드를 뺀다.

    operational이면 오류 삽입·측정용 레코드와 그 후손도 뺀다 (학습·내보내기·다른 단계의 입력).

    Args:
        labels: 라벨 이력 (보통 `get_labels` 결과 전체).
        operational: True(기본)면 운영 라벨만. False는 이관처럼 측정·오류 삽입 레코드도
            다뤄야 할 때.

    Returns:
        입력 순서를 유지한 현재 레코드 목록. 삭제 레코드 자신과, 삭제 레코드가 가리키는 부모(다른
        레코드의 parent이므로)는 모두 빠진다.
    """
    excluded = non_operational_ids(labels) if operational else set[str]()
    superseded = {x.parent_label_id for x in labels if x.parent_label_id}
    retracted = {x.label_id for x in labels if x.retracted}
    return [
        x
        for x in labels
        if x.label_id not in superseded
        and x.label_id not in retracted
        and x.label_id not in excluded
    ]


def version_tag(model_version: str) -> str:
    """모델 버전마다 다른 짧은 ID 조각.

    모델 출처 라벨 ID에 넣어 버전이 바뀌어도 ID가 겹치지 않게 한다.

    Returns:
        `sha256(model_version)`의 16진 앞 8자. 결정적이다.
    """
    return hashlib.sha256(model_version.encode()).hexdigest()[:8]


def retractions(
    stale: Iterable[LabelRecord], model_version: str, now: datetime
) -> list[LabelRecord]:
    """이전 버전의 모델 라벨을 지우는 삭제 레코드 (parent=원래, retracted).

    ID는 `<원래>:retracted`(128자를 넘으면 해시로 줄인 `derived_id`)이고
    출처는 지운 쪽(새 버전)이다.
    한 레코드는 한 번만 지울 수 있다 (지운 뒤에는 현재 라벨이 아니므로 다시 대상이 되지 않는다).
    계약 검증을 거친다 (시간대 없는 now 등은 ValidationError).

    Args:
        stale: 지울 이전 버전 레코드 (호출자가 검수 전인 것만 골라 넘긴다).
        model_version: 지우는 쪽(새 버전) 모델 버전. 삭제 레코드의 출처가 된다.
        now: 삭제 레코드 생성 시각 (시간대 필수).

    Returns:
        삭제 레코드 목록. 페이로드·구간·stream_id(와 seeded_error·measurement 표시)는 원래 것을
        복사하고, 검증 상태는 미검수,
        confidence는 원래 값(없으면 1.0. 모델 출처 계약상 필수라서). DB에 쓰지 않는다.
    """
    return [
        LabelRecord.model_validate(
            {
                **x.model_dump(),
                "label_id": derived_id(x.label_id, "retracted"),
                "parent_label_id": x.label_id,
                "retracted": True,
                "verification": Verification(),
                "provenance": Provenance(source=Source.MODEL, model_version=model_version),
                "confidence": x.confidence if x.confidence is not None else 1.0,
                "created_at": now,
            }
        )
        for x in stale
    ]


def is_protected(x: LabelRecord) -> bool:
    """사람이 만들었거나 고친 라벨(출처 HUMAN), 또는 검수자가 승인·표본 검증한 라벨인가.

    어떤 단계도 이런 라벨을 지우지 않고(ADR 0015), 같은 대상 트랙이 여럿이면 먼저 고른다
    (`track_rank`). 미검수(UNREVIEWED)가 아닌 검증 상태(표본 검증, 승인, 사람 수정)는 모두 검수된
    것으로 본다.
    """
    return (
        x.provenance.source is Source.HUMAN
        or x.verification.state is not VerificationState.UNREVIEWED
    )


def track_rank(x: LabelRecord) -> tuple[bool, datetime, int, str]:
    """같은 대상 트랙이 여럿일 때 하나를 고르는 정렬 키 (튜플이 클수록 앞선다).

    같은 대상 트랙이 함께 현재 라벨일 수 있다 (사람이 새로 그린 트랙과 모델 트랙, 승인된 이전 모델
    버전과 새 버전). 프리라벨 접촉(`dlp_prelabel.runner.pick_hand_tracks`)·3D(`pick_lift_tracks`)
    단계와 행동 단계(`dlp_actions.runner.run_actions`)가 함께 쓴다 (F-PAR-04: 행동 단계만 조회
    순서상 마지막 트랙을 써 기준이 달랐다). 차례로:

    1. 사람이 만들었거나 검수된 트랙 (`is_protected`): 사람이 확인한 위치가 가장 믿을 만하다.
    2. 더 새로 만든 트랙 (`created_at`): 보통 더 새 모델 버전의 출력이다.
    3. 더 긴 구간 (`t_end_ms - t_start_ms`): 놓치는 시간이 짧다.
    4. 라벨 ID (사전순으로 큰 것): 위가 모두 같을 때 결과를 고정하는 마지막 기준.

    검수 상태가 바뀌면(승인은 라벨 ID를 바꾸지 않는다) 고르는 트랙이 바뀔 수 있다. 그래서 고른
    트랙을 쓰는 단계는 입력 해시를 고른 트랙의 라벨 ID로 만든다.
    """
    return (is_protected(x), x.created_at, x.t_end_ms - x.t_start_ms, x.label_id)


def pick_hand_track_labels(labels: Iterable[LabelRecord]) -> dict[Hand, LabelRecord]:
    """손(왼·오른)마다 키포인트 트랙 하나를 고른다 (`track_rank`가 가장 앞서는 것).

    병합하지 않는다: 서로 다른 트랙의 관절을 섞으면 어느 출처의 위치인지 알 수 없게 된다. 결과는
    입력 순서와 상관없다.

    Args:
        labels: 호출자가 스트림(보통 기준 스트림 바디캠)과 골격(hand21)으로 거른 현재 키포인트 트랙.
            손이 정해지지 않은 트랙(`payload.hand`가 None)과 키포인트 트랙이 아닌 라벨은 무시한다.

    Returns:
        손 → 고른 트랙 레코드. 트랙이 없는 손은 키가 없다.
    """
    best: dict[Hand, LabelRecord] = {}
    for x in labels:
        if not isinstance(x.payload, KeypointTrackPayload) or x.payload.hand is None:
            continue
        hand = x.payload.hand
        if hand not in best or track_rank(x) > track_rank(best[hand]):
            best[hand] = x
    return best
