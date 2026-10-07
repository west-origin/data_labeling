"""우선순위 큐: 검수 단위 안에서 먼저 볼 구간을 찾고 단위 우선순위를 매긴다.

사유:
- low_confidence: 신뢰도가 낮은 모델 라벨
- model_disagreement: 다른 모델(버전)의 라벨이 같은 대상·시간을 다르게 분류
- new_object: 다른 세션에서 사람이 확인한 적 없는 객체 클래스
- contact_mismatch: 장갑 세션에서 장갑과 영상 중 한쪽만 접촉이라고 본 구간 (접촉 융합 신뢰도가 낮다)
단위 우선순위 = Σ 사유 가중치 * 구간 길이(초). 사유가 없으면 routine_priority.

WP12, ADR 0014. 정책은 `config/policies/review.yaml` `priority`·`units` 절.
`ops.runner.plan_session`(`dlp review plan`)이
`units_for` → `flag_unit` → `unit_priority` 순으로 쓴다.

공개 이름:
- `Group`: 단위 묶음 이름 (spatial, temporal, privacy).
- `VIDEO`: 영상 스트림 종류.
- `Unit`: 검수 단위 (세션 * 스트림 * 라벨 종류 묶음).
- `units_for`: 세션의 검수 단위 목록.
- `flag_unit`: 단위 라벨에서 먼저 볼 구간(`FlaggedSpan`)을 찾는다.
- `merge_spans`: 같은 사유의 겹치는 구간을 합친다.
- `unit_priority`: 단위 우선순위 점수.

시간 단위: 구간은 라벨의 `t_start_ms`·`t_end_ms`(정수 ms). 박스 불일치는 두 트랙의 같은 키프레임
시각(스트림 PTS ms)끼리만 비교한다.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from dlp_review.ops.policy import ReviewOpsPolicy
from dlp_schema.labels import (
    ActionPayload,
    BoxTrackPayload,
    HandStatePayload,
    LabelRecord,
    MaskTrackPayload,
    ObjectStatePayload,
    RelationPayload,
    Source,
)
from dlp_schema.review import FlaggedSpan, ReviewReason
from dlp_schema.session import Session, StreamKind

# 검수 단위 묶음: spatial(CVAT 공간 라벨), temporal(Label Studio 시간 라벨), privacy(블러)
Group = Literal["spatial", "temporal", "privacy"]
# 검수 단위를 만드는 영상 스트림 종류
VIDEO = (StreamKind.BODYCAM, StreamKind.THIRD_PERSON)


@dataclass(frozen=True)
class Unit:
    """검수 단위: 세션 * 스트림 * 라벨 종류 묶음. 묶음마다 검수 도구가 다르다."""

    session_id: str
    stream_id: str | None  # 시간 라벨 묶음은 세션 단위(None)
    group: Group
    # 이 단위가 다루는 라벨 종류 (LabelRecord.kind)
    kinds: tuple[str, ...]

    @property
    def unit_id(self) -> str:
        """단위 ID `"<세션>:<스트림 또는 all>:<묶음>"`. 배정 ID와 뽑기 시드의 바탕이다."""
        return f"{self.session_id}:{self.stream_id or 'all'}:{self.group}"

    def select(self, labels: Iterable[LabelRecord]) -> list[LabelRecord]:
        """labels 중 이 단위에 속하는 것 (종류가 맞고, 스트림 단위면 같은 스트림).

        주의: 스트림 단위에서는 stream_id가 정확히 같은 라벨만 고른다. 스트림이 없는(None) 라벨은
        들어가지 않는다 (작업 생성의 `current_for_review`는 None도 넣는 것과 다르다).
        """
        return [
            x
            for x in labels
            if x.kind in self.kinds and (self.stream_id is None or x.stream_id == self.stream_id)
        ]


def units_for(session: Session, policy: ReviewOpsPolicy, *, privacy: bool = False) -> list[Unit]:
    """작업 라벨 단위 (영상 스트림별 공간 묶음 + 세션 시간 묶음). privacy면 블러 단위.

    공간 묶음 종류는 `policy.units.spatial`, 시간 묶음 종류는 `policy.units.temporal`이다.
    블러 단위는 영상 스트림마다 하나 (`blur_track`).
    """
    videos = [s.stream_id for s in session.streams if s.kind in VIDEO]
    if privacy:
        return [Unit(session.session_id, sid, "privacy", ("blur_track",)) for sid in videos]
    units = [Unit(session.session_id, sid, "spatial", policy.units.spatial) for sid in videos]
    units.append(Unit(session.session_id, None, "temporal", policy.units.temporal))
    return units


def _class_key(x: LabelRecord) -> tuple[str, str] | None:
    """(대상 키, 분류). 같은 대상 키끼리 분류가 다르면 불일치다.

    시간 라벨(행동·손 상태·객체 상태·관계)만 다룬다. 그 밖의 종류는 None (불일치 검사 안 함).
    손 상태는 파지 종류를 빼고 접촉 대상만 비교한다 (`measure.as_items`는 파지 종류까지 본다).
    """
    p = x.payload
    match p:
        case ActionPayload():
            return (f"action:{p.hand.value}", p.verb)
        case HandStatePayload():
            return (f"hand:{p.hand.value}", f"{p.contact_target_kind}:{p.target_id}")
        case ObjectStatePayload():
            return (f"state:{p.entity_id}:{p.attribute}", p.value)
        case RelationPayload():
            return (f"rel:{p.subject_id}:{p.object_id}", p.predicate.value)
        case _:
            return None


def _box_at(p: BoxTrackPayload, t: int) -> tuple[float, float, float, float] | None:
    """박스 트랙의 시각 t(ms) 키프레임 (x, y, w, h). 그 시각 키프레임이 없거나 화면 밖이면 None.

    보간하지 않고 정확히 같은 시각의 키프레임만 본다.
    """
    for k in p.keyframes:
        if k.t_ms == t and not k.outside:
            return (k.x, k.y, k.w, k.h)
    return None


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """두 박스 (x, y, w, h)의 IoU. 합집합 넓이가 0이면 0.0."""
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def flag_unit(
    labels: list[LabelRecord],
    policy: ReviewOpsPolicy,
    *,
    known_classes: set[str],
    glove_session: bool,
) -> list[FlaggedSpan]:
    """labels: 단위의 현재 운영 라벨.

    인자:
    - policy: `priority` 절(임계값)을 쓴다.
    - known_classes: 다른 세션에서 사람이 확인한 객체 클래스 (`runner.known_classes`).
    - glove_session: 세션에 장갑 스트림이 있는가 (contact_mismatch 검사 여부).

    반환: 먼저 볼 구간 목록 (`merge_spans`로 합친 뒤 시작 시각 순). 모델 출처 라벨만 본다.
    O(N²) 쌍 비교가 있어 단위 라벨 수가 많으면 느려질 수 있다.
    """
    pp = policy.priority
    model = [x for x in labels if x.provenance.source is Source.MODEL]
    spans: list[FlaggedSpan] = []

    def flag(reason: ReviewReason, start: int, end: int, *ids: str) -> None:
        """사유 구간 하나를 spans에 더한다."""
        spans.append(FlaggedSpan(reason=reason, t_start_ms=start, t_end_ms=end, label_ids=ids))

    # 1) 라벨 하나만 보고 정하는 사유
    for x in model:
        if x.confidence is not None and x.confidence < pp.low_confidence:
            flag(ReviewReason.LOW_CONFIDENCE, x.t_start_ms, x.t_end_ms, x.label_id)
        p = x.payload
        if isinstance(p, BoxTrackPayload | MaskTrackPayload) and p.class_id not in known_classes:
            flag(ReviewReason.NEW_OBJECT, x.t_start_ms, x.t_end_ms, x.label_id)
        # 접촉 융합(장갑·영상)은 두 근거가 어긋나면 신뢰도를 낮춘다.
        # 낮은 신뢰도의 접촉 = 장갑·영상 중 한쪽만 접촉
        if (
            glove_session
            and isinstance(p, HandStatePayload)
            and p.contact_target_kind != "none"
            and x.confidence is not None
            and x.confidence <= pp.contact_mismatch_max_confidence
        ):
            flag(ReviewReason.CONTACT_MISMATCH, x.t_start_ms, x.t_end_ms, x.label_id)

    # 2) 서로 다른 모델 버전의 같은 종류 라벨 쌍 비교 (불일치)
    for i, a in enumerate(model):
        for b in model[i + 1 :]:
            if a.provenance.model_version == b.provenance.model_version or a.kind != b.kind:
                continue
            start, end = max(a.t_start_ms, b.t_start_ms), min(a.t_end_ms, b.t_end_ms)
            if end <= start:
                continue  # 시간이 겹치지 않는다
            pa, pb = a.payload, b.payload
            if isinstance(pa, BoxTrackPayload) and isinstance(pb, BoxTrackPayload):
                if pa.class_id == pb.class_id:
                    continue
                # 두 트랙에 모두 있는 키프레임 시각에서 박스가 충분히 겹치면
                # 같은 물체를 다르게 본 것
                shared = sorted({k.t_ms for k in pa.keyframes} & {k.t_ms for k in pb.keyframes})
                hits = [
                    t
                    for t in shared
                    if (ba := _box_at(pa, t))
                    and (bb := _box_at(pb, t))
                    and _iou(ba, bb) >= pp.disagreement_iou
                ]
                if hits:
                    # 첫·마지막 겹친 키프레임 사이를 구간으로 (하나뿐이면 길이 0 구간)
                    flag(ReviewReason.MODEL_DISAGREEMENT, hits[0], hits[-1], a.label_id, b.label_id)
                continue
            ka, kb = _class_key(a), _class_key(b)
            if ka is None or kb is None or ka[0] != kb[0] or ka[1] == kb[1]:
                continue  # 비교할 수 없거나, 대상이 다르거나, 분류가 같다
            # 겹침 비율 = 겹친 길이 / 짧은 쪽 길이 (짧은 라벨이 긴 라벨 안에 있으면 1)
            shorter = min(a.t_end_ms - a.t_start_ms, b.t_end_ms - b.t_start_ms)
            if shorter > 0 and (end - start) / shorter >= pp.disagreement_overlap:
                flag(ReviewReason.MODEL_DISAGREEMENT, start, end, a.label_id, b.label_id)
    return merge_spans(spans)


def merge_spans(spans: list[FlaggedSpan]) -> list[FlaggedSpan]:
    """같은 사유의 겹치는 구간을 합친다.

    끝과 시작이 맞닿은 구간(s.start == last.end)도 합친다. 라벨 ID는 순서를 유지하며 중복을 뺀다.
    반환: (시작 시각, 사유) 순으로 정렬한 목록.
    """
    out: list[FlaggedSpan] = []
    for s in sorted(spans, key=lambda s: (s.reason.value, s.t_start_ms, s.t_end_ms)):
        last = out[-1] if out else None
        if last is not None and last.reason is s.reason and s.t_start_ms <= last.t_end_ms:
            out[-1] = last.model_copy(
                update={
                    "t_end_ms": max(last.t_end_ms, s.t_end_ms),
                    "label_ids": tuple(dict.fromkeys((*last.label_ids, *s.label_ids))),
                }
            )
        else:
            out.append(s)
    return sorted(out, key=lambda s: (s.t_start_ms, s.reason.value))


def unit_priority(spans: list[FlaggedSpan], policy: ReviewOpsPolicy) -> float:
    """단위 우선순위 = Σ 가중치(사유) * 구간 길이(초). 사유가 없거나 합이 0이면 routine_priority.

    길이 0 구간(한 시각)은 1 ms로 본다. 가중치에 없는 사유(예: sample)는 0이다.
    소수 6자리로 반올림해 같은 입력이면 같은 값(배정 정렬 안정성)을 낸다.
    """
    weights = policy.priority.weights
    total = sum(
        weights.get(s.reason.value, 0.0) * max(s.t_end_ms - s.t_start_ms, 1) / 1000 for s in spans
    )
    return round(total, 6) if total > 0 else policy.priority.routine_priority
