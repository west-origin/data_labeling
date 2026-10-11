"""세션 라벨에서 관계·커버리지를 도출한다 (DB와 무관한 순수 계산).

입력은 수정 이력을 반영한 현재 라벨이다. 이 모듈이 이전에 만든 관계·커버리지는 입력에서 뺀다.
- 개체 클래스: 박스·마스크 트랙의 class_id. 없으면 개체 ID에서 추정한다 (rag_01 → rag).
- 도구 작용부 궤적: 도구 클래스이고 part가 온톨로지의 작용부인 trajectory3d.
- 표면: surface 클래스이고 네 꼭짓점 궤적이 모두 있는 개체. CAMERA는 같은 영상 스트림의
  좌표끼리만 비교하고 WORLD는 공통 좌표로 본다. 외부 파라미터를 추정해 카메라를 섞지 않는다.
- 공간 라벨 PTS는 Session의 오프셋·드리프트·수동 보정으로 마스터 시각에 맞춘다.
- 같은 (개체, 부위, 좌표계·카메라) 궤적은 하나만, 커버리지는 (도구, 표면)마다 하나다.

공개: `derive`(전체 계산), `is_derived`(이 모듈 출력인가), `entity_classes`, `class_of`,
`VERSION_PREFIX`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from dlp_relations.contact import SurfaceContact, Track3D, tool_surface_contacts
from dlp_relations.coverage import coverage_ratio
from dlp_relations.policy import RelationsPolicy
from dlp_relations.rules import RelationDraft, apply_rules
from dlp_schema.episode import current_labels
from dlp_schema.labels import (
    BoxTrackPayload,
    CoordinateFrame,
    CoveragePayload,
    HandStatePayload,
    LabelRecord,
    MaskTrackPayload,
    Source,
    Trajectory3DPayload,
    VerificationState,
)
from dlp_schema.ontology import Ontology
from dlp_schema.session import Session

# 이 모듈 출력의 model_version 접두사 (뒤에 정책 해시가 붙는다)
VERSION_PREFIX = "relations-"

# CAMERA 좌표는 카메라 ID가 있어야 같다. WORLD는 세션 공통 좌표계다.
FrameKey = tuple[str, str]


@dataclass(frozen=True)
class CoverageDraft:
    """커버리지 초안: 페이로드, 구간(접촉들의 처음~끝, ms), 신뢰도."""

    payload: CoveragePayload
    start_ms: int
    end_ms: int
    confidence: float


@dataclass
class Derived:
    """`derive` 결과: 관계 초안, 커버리지 초안, 계산에 쓴 도구-표면 접촉 구간."""

    relations: list[RelationDraft]
    coverage: list[CoverageDraft]
    contacts: list[SurfaceContact]


def is_derived(x: LabelRecord) -> bool:
    """이 모듈(관계 단계)이 만든 모델 레코드인가 (model_version이 `relations-`로 시작)."""
    return x.provenance.source is Source.MODEL and (x.provenance.model_version or "").startswith(
        VERSION_PREFIX
    )


def _preference(x: LabelRecord) -> tuple[bool, float, str]:
    """정렬 키: 검수된·사람 라벨 먼저, 그다음 최신, 같으면 ID 순 (결정적)."""
    reviewed = (
        x.provenance.source is Source.HUMAN
        or x.verification.state is not VerificationState.UNREVIEWED
    )
    return (not reviewed, -x.created_at.timestamp(), x.label_id)


def entity_classes(labels: list[LabelRecord]) -> dict[str, str]:
    """박스·마스크 트랙의 entity_id → class_id (같은 개체가 여럿이면 마지막 것)."""
    out: dict[str, str] = {}
    for x in labels:
        if isinstance(x.payload, BoxTrackPayload | MaskTrackPayload):
            out[x.payload.entity_id] = x.payload.class_id
    return out


def class_of(entity_id: str, classes: dict[str, str]) -> str:
    """개체 클래스. 트랙 클래스가 없으면 ID에서 추정한다: `ov_` 접두사와 `_숫자` 접미사를 뗀다."""
    if entity_id in classes:
        return classes[entity_id]
    return re.sub(r"_\d+$", "", entity_id.removeprefix("ov_"))


def derive(  # noqa: C901
    labels: list[LabelRecord],
    ontology: Ontology,
    policy: RelationsPolicy,
    session: Session | None = None,
) -> Derived:
    """현재 라벨 → 관계·커버리지 초안 (DB 없음, 결정적).

    Args:
        labels: 세션 전체 이력 (여기서 `current_labels`로 현재만 고른다).
        ontology: 도구 작용부(tool.working_parts)·표면(surface) 정의.
        policy: 관계 정책.
        session: 공간 라벨의 스트림 시계를 변환할 세션. 생략은 identity 시계의 bodycam 입력만
            허용한다 (기존 합성 계산용). 운영 러너는 반드시 세션을 넘긴다.

    Returns:
        `Derived`. 관계는 손 상태 규칙 + 도구-표면 규칙, 커버리지는 접촉이 있는 (도구, 표면)마다
            하나.
    """
    # 운영 라벨만 (오류 삽입·측정 레코드 제외). 이전 관계·커버리지 출력은 입력에서 뺀다
    current = [x for x in current_labels(labels) if not is_derived(x)]
    if session is not None and any(x.session_id != session.session_id for x in current):
        raise ValueError("관계 입력 라벨과 세션 ID가 다릅니다")
    classes = entity_classes(current)
    hand_states = [x for x in current if isinstance(x.payload, HandStatePayload)]

    # (도구, 부위, 좌표계·카메라)마다 궤적 하나. 같은 작용부 궤적이 여럿이면(예: 3D 단계가
    # 다시 돌았는데 이전 것은 검수돼 남음) 검수된·사람 궤적을 먼저, 그다음 최신을 쓴다.
    # 같은 접촉·커버리지가 두 번 나와 같은 ID로 겹치지 않게 한다
    chosen: dict[tuple[str, str, FrameKey], LabelRecord] = {}
    corners: dict[tuple[str, FrameKey], dict[str, Track3D]] = {}
    corner_src: dict[tuple[str, str, FrameKey], LabelRecord] = {}

    def track(x: LabelRecord) -> Track3D:
        """저장된 PTS는 그대로 두고 계산용 궤적의 시각만 마스터에 맞춘다."""
        assert isinstance(x.payload, Trajectory3DPayload)
        if session is None:
            if x.stream_id != "bodycam":
                raise ValueError("바디캠 외 궤적의 시각 변환에는 Session이 필요합니다")
            stream = None
        else:
            stream = session.stream(x.stream_id or "")
        return Track3D.from_payload(x.payload, stream)

    for x in sorted(current, key=_preference):
        p = x.payload
        if not isinstance(p, Trajectory3DPayload) or p.part is None:
            continue
        obj = ontology.objects.get(class_of(p.entity_id, classes))
        if obj is None:
            continue
        frame = (p.frame.value, (x.stream_id or "") if p.frame is CoordinateFrame.CAMERA else "")
        key = (p.entity_id, p.part, frame)
        if obj.tool is not None and p.part in obj.tool.working_parts:
            # _preference 순으로 처음 본 것(검수된·사람 먼저, 그다음 최신)만 남긴다
            chosen.setdefault(key, x)
        elif (
            obj.surface
            and p.part in policy.tool_surface.corner_parts
            # 꼭짓점도 (표면, 부위, 좌표계)마다 처음 본 궤적 하나만 쓴다
            and corner_src.setdefault(key, x) is x
        ):
            corners.setdefault((p.entity_id, frame), {})[p.part] = track(x)
    # (도구, 좌표계) → 작용부 궤적들. 커버리지는 (도구, 표면)마다 하나다 (작용부를 합친다)
    tools: dict[tuple[str, FrameKey], list[Track3D]] = {}
    for (entity_id, _, frame), x in sorted(chosen.items()):
        assert isinstance(x.payload, Trajectory3DPayload)
        tool = track(x)
        # 동기화 뒤 세션 밖에 놓인 표본은 시간 구간 라벨로 내지 않는다. 표면 표본은 보간용으로 둔다.
        if session is not None:
            inside = (tool.times >= 0) & (tool.times <= session.duration_ms)
            tool = Track3D(tool.entity_id, tool.part, tool.times[inside], tool.xyz[inside])
        tools.setdefault((entity_id, frame), []).append(tool)

    contacts: list[SurfaceContact] = []
    by_pair: dict[tuple[str, str], list[SurfaceContact]] = {}
    ts = policy.tool_surface
    for (tool_id, frame), parts_tracks in sorted(tools.items()):
        # require_grasp: 이 도구를 대상으로 한 손 상태(tool) 구간
        grasped = [
            (x.t_start_ms, x.t_end_ms)
            for x in hand_states
            if isinstance(x.payload, HandStatePayload)
            and x.payload.contact_target_kind == "tool"
            and x.payload.target_id == tool_id
        ]
        for (surface_id, surface_frame), parts in sorted(corners.items()):
            # 좌표계가 다르거나 꼭짓점 네 개가 다 없으면 이 표면은 건너뛴다
            if surface_frame != frame or set(parts) != set(ts.corner_parts):
                continue
            found = sorted(
                (
                    c
                    for tool in parts_tracks
                    for c in tool_surface_contacts(
                        tool, surface_id, [parts[k] for k in ts.corner_parts], grasped, ts
                    )
                ),
                key=lambda c: (c.start_ms, c.end_ms),
            )
            if not found:
                continue
            contacts += found
            by_pair.setdefault((tool_id, surface_id), []).extend(found)
    # 서로 다른 카메라에서 따로 계산한 접촉은 표면 좌표로 합쳐 커버리지를 하나만 낸다.
    coverage = [
        CoverageDraft(
            CoveragePayload(
                surface_id=surface_id,
                tool_id=tool_id,
                ratio=round(
                    coverage_ratio(
                        found,
                        policy.coverage.footprint(class_of(tool_id, classes)),
                        policy.coverage.grid_m,
                    ),
                    4,
                ),
            ),
            min(c.start_ms for c in found),
            max(c.end_ms for c in found),
            ts.confidence,
        )
        for (tool_id, surface_id), found in sorted(by_pair.items())
    ]
    return Derived(apply_rules(policy, hand_states, contacts), coverage, contacts)
