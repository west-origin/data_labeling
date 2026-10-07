"""손으로 계산한 작은 예제와의 대조.

WP11 완료 기준("각 지표를 손으로 계산한 작은 예제와 대조")의 단위 테스트. 각 테스트의 정답은 주석에
적은 손 계산이다 (합성 픽스처 없이 작은 수치 예제를 직접 쓴다). 참조 라이브러리 일치 검사는
test_metrics_reference.py·test_metrics_temporal_reference.py가 따로 한다.
"""

from __future__ import annotations

import numpy as np
import pytest

from dlp_eval.metrics.classification import ece, under_sampled
from dlp_eval.metrics.detection import box_iou
from dlp_eval.metrics.keypoints import pck
from dlp_eval.metrics.states import transition_accuracy, transitions
from dlp_eval.metrics.temporal import (
    boundary_agreement,
    interval_iou,
    match_events,
    segment_f1,
    temporal_map,
)


def test_box_and_interval_iou() -> None:
    """박스 IoU와 구간 IoU를 손 계산 값과 대조한다 (겹침 있음·없음)."""
    # 10x10 두 박스가 5만큼 겹침: 교집합 50, 합집합 150
    assert box_iou([(0, 0, 10, 10)], [(5, 0, 10, 10)])[0, 0] == pytest.approx(1 / 3)
    assert interval_iou((0, 100), (50, 150)) == pytest.approx(50 / 150)
    assert interval_iou((0, 100), (200, 300)) == 0.0


def test_event_matching_with_tolerance() -> None:
    """시점 사건 매칭: 허용 오차 밖 예측은 FP, 짝 없는 정답은 FN, 오차는 맞춘 쌍만 센다."""
    # 정답 100, 500, 900 / 예측 130, 560(허용 50 밖), 905, 2000
    r = match_events([100, 500, 900], [130, 560, 905, 2000], tolerance_ms=50)
    assert (r.tp, r.fp, r.fn) == (2, 2, 1)
    assert r.errors_ms == (30, 5) and r.mean_error_ms == 17.5
    assert r.precision == 0.5 and r.recall == pytest.approx(2 / 3)
    assert r.f1 == pytest.approx(2 * 0.5 * (2 / 3) / (0.5 + 2 / 3))


def test_segment_f1_and_boundaries() -> None:
    """구간 F1@IoU(이미 쓴 정답과 겹친 예측은 FP)와 경계 일치(양 끝 포함)를 손 계산과 대조한다."""
    truth = [(0, 100, "rub"), (100, 200, "push"), (200, 300, "rub")]
    pred = [(0, 90, "rub"), (90, 210, "push"), (210, 260, "rub"), (260, 300, "rub")]
    # rub(0,90): IoU .9 TP / push: 100/120=.83 TP / rub(210,260): .5 TP@.5
    # rub(260,300): 같은 정답이 이미 쓰여 FP
    r = segment_f1(truth, pred, 0.5)
    assert (r.tp, r.fp, r.fn) == (3, 1, 0) and r.f1 == pytest.approx(2 * 0.75 * 1 / 1.75)
    assert segment_f1(truth, pred, 0.75).tp == 2
    b = boundary_agreement(truth, pred, tolerance_ms=10)
    # 정답 경계 0,100,200,300 / 예측 0,90,210,260,300 → 0, 90(10), 210(10), 300 맞음
    assert (b.tp, b.fp, b.fn) == (4, 1, 0)


def test_boundary_agreement_excludes_timeline_extremes() -> None:
    """exclude_extremes: 정답·예측 합집합 타임라인의 양 끝 경계를 빼고 센다 (공짜 일치 제거)."""
    # 공백 없이 채운 타임라인은 양 끝(0, 300)이 항상 같아 공짜로 맞는다 → 빼고 센다
    truth = [(0, 100, "rub"), (100, 300, "push")]
    pred = [(0, 200, "rub"), (200, 300, "push")]
    b = boundary_agreement(truth, pred, tolerance_ms=10)
    assert (b.tp, b.fp, b.fn) == (2, 1, 1)  # 0, 300 공짜
    b = boundary_agreement(truth, pred, tolerance_ms=10, exclude_extremes=True)
    assert (b.tp, b.fp, b.fn) == (0, 1, 1) and b.f1 == 0.0
    # 예측이 늦게 시작하면 그 시작은 경계로 남아 오탐이다 (양 끝은 정답·예측을 합친 타임라인 기준)
    b = boundary_agreement(truth, [(50, 100, "rub"), (100, 300, "push")], 10, exclude_extremes=True)
    assert (b.tp, b.fp, b.fn) == (1, 1, 0)


def test_temporal_map_hand_example() -> None:
    """temporal mAP(tIoU 0.5 하나): 정밀도 포락선 넓이를 손 계산하고, 다른 묶음 정답과는 맞추지
    않는다."""
    truth = [("v1", 0, 100, "rub"), ("v1", 200, 300, "rub")]
    pred = [("v1", 0, 100, "rub", 0.9), ("v1", 500, 600, "rub", 0.8), ("v1", 200, 290, "rub", 0.7)]
    # tIoU 0.5: TP, FP, TP → 정밀도 1, .5, .67 / 재현율 .5, .5, 1 → AP = .5*1 + .5*.67 = .8333
    m, _ = temporal_map(truth, pred, thresholds=(0.5,))
    assert m == pytest.approx(0.5 + 0.5 * 2 / 3)
    # 다른 묶음(v2)의 정답과는 맞추지 않는다
    m2, _ = temporal_map(truth, [("v2", 0, 100, "rub", 0.9)], thresholds=(0.5,))
    assert m2 == 0.0


def test_pck_counts_visible_joints_only() -> None:
    """PCK: 보이는 정답 관절만 분모에 넣고, 예측 없음(None)은 그 관절을 모두 틀린 것으로 센다."""
    gt = np.array([[0, 0, 2], [100, 0, 2], [50, 50, 0]], dtype=float)  # 기준 길이 100
    pred = np.array([[5, 0], [100, 30], [0, 0]], dtype=float)  # 거리 5, 30
    r = pck([(gt, pred), (gt, None)], alpha=0.1)
    assert (r.correct, r.total) == (1, 4) and r.pck == 0.25


def test_ece_hand_example() -> None:
    """ECE 구간별 가중 평균을 손 계산과 대조하고, 표본 부족 클래스 표시를 확인한다."""
    # 구간 (0.8,0.867]: 신뢰도 .85 두 개, 정답 1개 → |.5-.85| = .35, 가중치 2/4
    # 구간 (0.267,0.333]: 신뢰도 .3 두 개, 정답 0개 → |0-.3| = .3, 가중치 2/4
    assert ece([0.85, 0.85, 0.3, 0.3], [True, False, False, False], bins=15) == pytest.approx(0.325)
    assert ece([1.0, 1.0], [True, True]) == 0.0
    assert under_sampled({"mop": 3, "cup": 40}, 10) == ["mop"]


def test_state_transitions() -> None:
    """상태 전이 추출과 정확도: 허용 오차 안의 늦은 전이는 맞고, 예측이 놓친 전이(걸레 젖음)는
    틀린다."""
    sink, rag = ("sink_01", "cleanliness"), ("rag_01", "wetness")
    truth = [
        (*sink, 0, 1000, "dirty"), (*sink, 1000, 3000, "clean"),
        (*rag, 0, 500, "dry"), (*rag, 500, 3000, "wet"),
    ]  # fmt: skip
    pred = [(*sink, 0, 1100, "dirty"), (*sink, 1100, 3000, "clean"), (*rag, 0, 3000, "dry")]
    assert transitions(truth) == [(*sink, "dirty", "clean", 1000), (*rag, "dry", "wet", 500)]
    r = transition_accuracy(truth, pred, tolerance_ms=200)
    assert (r.matched, r.truth, r.pred) == (1, 2, 1) and r.accuracy == 0.5
