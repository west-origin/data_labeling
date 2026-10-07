"""검수 품질 측정: 이중 라벨링 일치도, 블라인드 과제 대비 프리라벨 편향, 오류 삽입 발견율.

라벨은 비교를 위해 (대상 키, 분류, 시작, 끝) 구간으로 바꾼다. 같은 대상 키끼리 시간 IoU로 맞춘다.
- 일치도: 맞춘 쌍의 분류 카파와 경계 일치 F1, 구간 F1@0.5
- 프리라벨 편향 = F1(표준 검수 결과, 모델 프리라벨) - F1(블라인드 결과, 모델 프리라벨).
  양수면 검수자가 프리라벨에 끌려간다는 뜻이다 (블라인드는 프리라벨을 보지 않았다).

WP12, ADR 0014. `ops.runner.quality_report`(`dlp review quality`)가 쓰고, 정책 값은
`config/policies/review.yaml` `measurement` 절(tolerance_ms, match_iou)이다.
지표 구현은 `dlp_eval.metrics`(참조 구현과 일치 테스트가 있는 공용 라이브러리)를 쓴다.

공개 이름: `Item`, `as_items`, `Agreement`, `agreement`, `prelabel_bias`, `DetectionRate`.
시간 단위: 구간의 시작·끝은 LabelRecord의 `t_start_ms`·`t_end_ms` 그대로(정수 ms)다. 시간 라벨은
마스터 타임라인, 공간 라벨은 스트림 PTS 시각이며(ADR 0019) 같은 대상 키 안에서만 비교한다.
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
    """라벨을 비교용 구간 `(대상 키, 분류, 시작 ms, 끝 ms)`로 바꾼다.

    대상 키는 "무엇에 대한 라벨인가"(손, 개체·속성, 관계 쌍, 스트림), 분류는 "무엇이라고 했나"다.
    - 행동·사이 구간: 키 `action:<손>`(사이 구간은 손이 없으면 `-`), 분류 동사 / `gap:<종류>`.
      같은 손 트랙에서 행동과 사이 구간이 서로 맞춰질 수 있게 같은 키를 쓴다.
    - 손 상태: 키 `hand:<손>`, 분류 `<접촉 대상 종류>:<대상 ID>:<파지 종류>`.
    - 객체 상태: 키 `state:<개체>:<속성>`, 분류 값.
    - 관계: 키 `rel:<주어>:<목적어>`, 분류 술어.
    - 박스·마스크: 키 `obj:<스트림>`, 분류 클래스. 블러: 키 `blur:<스트림>`, 분류 대상 종류.
    그 밖의 종류(상위 구간·이벤트·키포인트 등)는 건너뛴다.
    """
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
    """두 라벨 묶음의 일치도."""

    # 시간 IoU로 짝지은 쌍 수
    pairs: int
    # 짝지은 쌍의 분류 Cohen 카파 (쌍이 없으면 0.0)
    kappa: float
    # 대상 키·분류까지 같아야 맞는 구간 F1 (IoU ≥ match_iou)
    segment_f1: float
    # 대상 키별 경계(시작·끝) 일치 F1 (허용 오차 tolerance_ms)
    boundary_f1: float


def agreement(a: Sequence[Item], b: Sequence[Item], tolerance_ms: int, iou: float) -> Agreement:
    """두 라벨 묶음의 일치도. 대상 키별로 시간 IoU가 가장 큰 쌍을 맞춘다.

    인자:
    - a, b: `as_items` 결과. 순서가 짝짓기 결과에 영향을 준다(a 순서대로 탐욕적으로 맞춘다).
    - tolerance_ms: 경계 일치 허용 오차 (ms).
    - iou: 짝지을 최소 시간 IoU (이 값 이상이어야 짝).

    반환: `Agreement`. 카파는 짝지은 쌍의 분류로만 계산한다.
    """
    left: list[str] = []
    right: list[str] = []
    used: set[int] = set()
    # 탐욕적 짝짓기: a의 각 구간에 대해 같은 대상 키의 아직 안 쓴 b 구간 중 IoU가 가장 큰 것
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
    # 구간 F1: 키와 분류를 한 라벨 문자열로 묶어 "같은 대상·같은 분류·겹침"이어야 맞다
    seg = segment_f1(
        [(s, e, f"{k}|{c}") for k, c, s, e in a], [(s, e, f"{k}|{c}") for k, c, s, e in b], iou
    )
    # 경계 F1: 분류는 무시하고 대상 키만 같으면 경계 시각을 비교한다
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
    """표준 검수 결과가 블라인드 결과보다 모델 프리라벨에 얼마나 더 가까운가 (구간 F1 차).

    인자: model(모델 프리라벨), standard(표준 검수 뒤 운영 라벨), blind(블라인드 측정 레코드).
    반환: F1(standard, model) - F1(blind, model). 범위 [-1, 1], 양수면 프리라벨 편향.
    """
    return (
        agreement(standard, model, tolerance_ms, iou).segment_f1
        - agreement(blind, model, tolerance_ms, iou).segment_f1
    )


@dataclass(frozen=True)
class DetectionRate:
    """검수자별 오류 삽입 발견율."""

    # 검수자 ID (담당자가 없던 배정은 "unassigned")
    reviewer: str
    # 넣은 오류 수
    injected: int
    # 발견한 오류 수 (`ops.seeding.detected`)
    detected: int

    @property
    def rate(self) -> float:
        """발견율 = detected / injected (넣은 오류가 없으면 0.0)."""
        return self.detected / self.injected if self.injected else 0.0
