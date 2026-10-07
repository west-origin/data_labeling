"""우선순위 큐: 검수 단위 안에서 먼저 볼 구간을 찾고 단위 우선순위를 매긴다.

사유:
- low_confidence: 신뢰도가 낮은 모델 라벨
- model_disagreement: 다른 모델(버전)의 라벨이 같은 대상·시간을 다르게 분류
- new_object: 다른 세션에서 사람이 확인한 적 없는 객체 클래스
- contact_mismatch: 장갑 세션에서 장갑과 영상 중 한쪽만 접촉이라고 본 구간 (접촉 융합 신뢰도가 낮다)
단위 우선순위 = Σ 사유 가중치 * 구간 길이(초). 사유가 없으면 routine_priority.
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

Group = Literal["spatial", "temporal", "privacy"]
VIDEO = (StreamKind.BODYCAM, StreamKind.THIRD_PERSON)


@dataclass(frozen=True)
class Unit:
    session_id: str
    stream_id: str | None  # 시간 라벨 묶음은 세션 단위(None)
    group: Group
    kinds: tuple[str, ...]

    @property
    def unit_id(self) -> str:
        return f"{self.session_id}:{self.stream_id or 'all'}:{self.group}"

    def select(self, labels: Iterable[LabelRecord]) -> list[LabelRecord]:
        return [
            x
            for x in labels
            if x.kind in self.kinds and (self.stream_id is None or x.stream_id == self.stream_id)
        ]


def units_for(session: Session, policy: ReviewOpsPolicy, *, privacy: bool = False) -> list[Unit]:
    """작업 라벨 단위 (영상 스트림별 공간 묶음 + 세션 시간 묶음). privacy면 블러 단위."""
    videos = [s.stream_id for s in session.streams if s.kind in VIDEO]
    if privacy:
        return [Unit(session.session_id, sid, "privacy", ("blur_track",)) for sid in videos]
    units = [Unit(session.session_id, sid, "spatial", policy.units.spatial) for sid in videos]
    units.append(Unit(session.session_id, None, "temporal", policy.units.temporal))
    return units


def _class_key(x: LabelRecord) -> tuple[str, str] | None:
    """(대상 키, 분류). 같은 대상 키끼리 분류가 다르면 불일치다."""
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
    for k in p.keyframes:
        if k.t_ms == t and not k.outside:
            return (k.x, k.y, k.w, k.h)
    return None


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
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
    """labels: 단위의 현재 운영 라벨."""
    pp = policy.priority
    model = [x for x in labels if x.provenance.source is Source.MODEL]
    spans: list[FlaggedSpan] = []

    def flag(reason: ReviewReason, start: int, end: int, *ids: str) -> None:
        spans.append(FlaggedSpan(reason=reason, t_start_ms=start, t_end_ms=end, label_ids=ids))

    for x in model:
        if x.confidence is not None and x.confidence < pp.low_confidence:
            flag(ReviewReason.LOW_CONFIDENCE, x.t_start_ms, x.t_end_ms, x.label_id)
        p = x.payload
        if isinstance(p, BoxTrackPayload | MaskTrackPayload) and p.class_id not in known_classes:
            flag(ReviewReason.NEW_OBJECT, x.t_start_ms, x.t_end_ms, x.label_id)
        if (
            glove_session
            and isinstance(p, HandStatePayload)
            and p.contact_target_kind != "none"
            and x.confidence is not None
            and x.confidence <= pp.contact_mismatch_max_confidence
        ):
            flag(ReviewReason.CONTACT_MISMATCH, x.t_start_ms, x.t_end_ms, x.label_id)

    for i, a in enumerate(model):
        for b in model[i + 1 :]:
            if a.provenance.model_version == b.provenance.model_version or a.kind != b.kind:
                continue
            start, end = max(a.t_start_ms, b.t_start_ms), min(a.t_end_ms, b.t_end_ms)
            if end <= start:
                continue
            pa, pb = a.payload, b.payload
            if isinstance(pa, BoxTrackPayload) and isinstance(pb, BoxTrackPayload):
                if pa.class_id == pb.class_id:
                    continue
                shared = sorted({k.t_ms for k in pa.keyframes} & {k.t_ms for k in pb.keyframes})
                hits = [
                    t
                    for t in shared
                    if (ba := _box_at(pa, t))
                    and (bb := _box_at(pb, t))
                    and _iou(ba, bb) >= pp.disagreement_iou
                ]
                if hits:
                    flag(ReviewReason.MODEL_DISAGREEMENT, hits[0], hits[-1], a.label_id, b.label_id)
                continue
            ka, kb = _class_key(a), _class_key(b)
            if ka is None or kb is None or ka[0] != kb[0] or ka[1] == kb[1]:
                continue
            shorter = min(a.t_end_ms - a.t_start_ms, b.t_end_ms - b.t_start_ms)
            if shorter > 0 and (end - start) / shorter >= pp.disagreement_overlap:
                flag(ReviewReason.MODEL_DISAGREEMENT, start, end, a.label_id, b.label_id)
    return merge_spans(spans)


def merge_spans(spans: list[FlaggedSpan]) -> list[FlaggedSpan]:
    """같은 사유의 겹치는 구간을 합친다."""
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
    weights = policy.priority.weights
    total = sum(
        weights.get(s.reason.value, 0.0) * max(s.t_end_ms - s.t_start_ms, 1) / 1000 for s in spans
    )
    return round(total, 6) if total > 0 else policy.priority.routine_priority
