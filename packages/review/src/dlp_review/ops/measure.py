"""검수 품질 측정: 이중 라벨링 일치도, 블라인드 과제 대비 프리라벨 편향, 오류 삽입 발견율.

라벨은 비교를 위해 (대상 키, 분류, 시작, 끝) 구간으로 바꾼다. 같은 대상 키끼리 시간 IoU로 맞춘다.
- 일치도: 맞춘 쌍의 분류 카파와 경계 일치 F1, 구간 F1@0.5
- 프리라벨 편향 = F1(표준 검수 결과, 모델 프리라벨) - F1(블라인드 결과, 모델 프리라벨).
  양수면 검수자가 프리라벨에 끌려간다는 뜻이다 (블라인드는 프리라벨을 보지 않았다).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from dlp_eval.metrics.classification import cohen_kappa
from dlp_eval.metrics.temporal import boundary_agreement, interval_iou, segment_f1
from dlp_schema.labels import (
    ActionPayload,
    BlurTrackPayload,
    BoxTrackPayload,
    GapPayload,
    HandStatePayload,
    LabelRecord,
    MaskTrackPayload,
    ObjectStatePayload,
    RelationPayload,
)

Item = tuple[str, str, int, int]  # (대상 키, 분류, 시작, 끝)


def as_items(labels: Iterable[LabelRecord]) -> list[Item]:
    out: list[Item] = []
    for x in labels:
        p = x.payload
        match p:
            case ActionPayload():
                key, cls = f"action:{p.hand.value}", p.verb
            case GapPayload():
                key, cls = f"action:{p.hand.value if p.hand else '-'}", f"gap:{p.gap_type}"
            case HandStatePayload():
                key, cls = (
                    f"hand:{p.hand.value}",
                    f"{p.contact_target_kind}:{p.target_id}:{p.grasp_type}",
                )
            case ObjectStatePayload():
                key, cls = f"state:{p.entity_id}:{p.attribute}", p.value
            case RelationPayload():
                key, cls = f"rel:{p.subject_id}:{p.object_id}", p.predicate.value
            case BoxTrackPayload() | MaskTrackPayload():
                key, cls = f"obj:{x.stream_id}", p.class_id
            case BlurTrackPayload():
                key, cls = f"blur:{x.stream_id}", p.target
            case _:
                continue
        out.append((key, cls, x.t_start_ms, x.t_end_ms))
    return out


@dataclass(frozen=True)
class Agreement:
    pairs: int
    kappa: float
    segment_f1: float
    boundary_f1: float


def agreement(a: Sequence[Item], b: Sequence[Item], tolerance_ms: int, iou: float) -> Agreement:
    """두 라벨 묶음의 일치도. 대상 키별로 시간 IoU가 가장 큰 쌍을 맞춘다."""
    left: list[str] = []
    right: list[str] = []
    used: set[int] = set()
    for key, cls, s, e in a:
        best, best_iou = -1, iou
        for j, (k2, _, s2, e2) in enumerate(b):
            if k2 != key or j in used:
                continue
            v = interval_iou((s, e), (s2, e2))
            if v >= best_iou:
                best, best_iou = j, v
        if best >= 0:
            used.add(best)
            left.append(cls)
            right.append(b[best][1])
    seg = segment_f1(
        [(s, e, f"{k}|{c}") for k, c, s, e in a], [(s, e, f"{k}|{c}") for k, c, s, e in b], iou
    )
    bnd = boundary_agreement(
        [(s, e, k) for k, _, s, e in a], [(s, e, k) for k, _, s, e in b], tolerance_ms
    )
    return Agreement(len(left), cohen_kappa(left, right) if left else 0.0, seg.f1, bnd.f1)


def prelabel_bias(
    model: Sequence[Item],
    standard: Sequence[Item],
    blind: Sequence[Item],
    tolerance_ms: int,
    iou: float,
) -> float:
    """표준 검수 결과가 블라인드 결과보다 모델 프리라벨에 얼마나 더 가까운가 (구간 F1 차)."""
    return (
        agreement(standard, model, tolerance_ms, iou).segment_f1
        - agreement(blind, model, tolerance_ms, iou).segment_f1
    )


@dataclass(frozen=True)
class DetectionRate:
    reviewer: str
    injected: int
    detected: int

    @property
    def rate(self) -> float:
        return self.detected / self.injected if self.injected else 0.0
