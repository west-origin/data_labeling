"""객체 검출 AP (COCO 방식, pycocotools와 같은 결과).

- 이미지(=세션·스트림·시각)·클래스마다 점수 내림차순으로 예측을 보고, IoU가 문턱 이상인 아직 안
  쓴 정답 중 IoU가 가장 큰 것과 맞춘다. 이미지당 점수 상위 max_dets개만 쓴다.
- 클래스마다 모든 이미지의 예측을 점수순(안정 정렬)으로 모아 누적 TP·FP로 정밀도-재현율을 만들고,
  정밀도를 단조 감소로 만든 뒤 재현율 0, 0.01, …, 1 101점에서 읽어 평균한다.
- mAP는 정답이 있는 클래스의 AP 평균, IoU 문턱 0.50:0.05:0.95 평균이다 (AP50, AP75도 낸다).
군중(crowd)·면적 구간은 쓰지 않는다 (면적 "all").

참조: pycocotools `COCOeval.evaluateImg`·`accumulate`·`summarize`
(bbox, areaRng "all", maxDets 100). test_metrics_reference.py가 무작위 입력에서
stats[0..2](mAP, AP50, AP75)와 1e-9 안에서 같은지 본다.

pycocotools와 다른 점:
- iscrowd·무시(ignore) 정답이 없다 (모든 정답이 일반 정답).
- 면적 구간 "all"의 상한(1e10 px²)을 두지 않는다 (실제 박스에서는 차이가 없다).
- 정답이 없는 이미지의 예측도 그 클래스의 FP로 센다. pycocotools도 GT 이미지 목록에 있는 이미지면
  같지만, 여기서는 "이미지"가 하네스의 (세션, 스트림, 정답 시각)이라 예측만 있는 시각은 애초에
  만들지 않는다 (`harness.eval_objects`는 정답 키프레임 시각에서만 비교한다).
- 클래스 목록은 정답에 나온 클래스만이다. 예측에만 나온 클래스는 AP 평균에 들지 않는다
  (pycocotools도 정답이 없는 클래스는 -1로 빼므로 결과는 같다).

박스 좌표는 픽셀 (x, y, w, h), 왼쪽 위 기준이다 (라벨 계약 `BoxKeyframe`와 같다).
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Box = tuple[float, float, float, float]  # x, y, w, h
# COCO IoU 문턱 0.50, 0.55, …, 0.95 (10개). 인덱스 0 = AP50, 인덱스 5 = AP75
IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
# COCO 보간 재현율 101점 (0.00, 0.01, …, 1.00)
RECALL_POINTS = np.linspace(0.0, 1.0, 101)


@dataclass(frozen=True)
class GtBox:
    """정답 박스 하나."""

    image: Hashable  # 이미지 키. 하네스는 (세션, 스트림, 시각 ms)를 쓴다
    category: str  # 클래스 (온톨로지 class_id)
    box: Box  # 픽셀 (x, y, w, h)


@dataclass(frozen=True)
class DetBox:
    """예측 박스 하나."""

    image: Hashable  # 정답과 같은 이미지 키
    category: str  # 예측 클래스
    box: Box  # 픽셀 (x, y, w, h)
    score: float  # 신뢰도 (정렬에만 쓴다. 범위 제한 없음)


def box_iou(a: Sequence[Box], b: Sequence[Box]) -> NDArray[np.float64]:
    """(len(a), len(b)) IoU 행렬.

    박스는 (x, y, w, h). 합집합 넓이가 0인 짝(넓이 0인 두 박스)은 IoU 0이다.
    한쪽이 비어 있으면 (len(a), len(b)) 모양의 0 행렬을 돌려준다.
    pycocotools `maskUtils.iou`(iscrowd 0)와 같은 연속 좌표 계산이다 (+1 픽셀 보정 없음).
    """
    if not a or not b:
        return np.zeros((len(a), len(b)))
    aa, bb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    # 오른쪽 아래 모서리 (x2, y2)
    ax2, ay2 = aa[:, 0] + aa[:, 2], aa[:, 1] + aa[:, 3]
    bx2, by2 = bb[:, 0] + bb[:, 2], bb[:, 1] + bb[:, 3]
    # 교집합 너비·높이 (겹치지 않으면 0으로 자른다), 브로드캐스트로 (len(a), len(b))
    iw = np.clip(
        np.minimum(ax2[:, None], bx2[None]) - np.maximum(aa[:, 0, None], bb[None, :, 0]), 0, None
    )
    ih = np.clip(
        np.minimum(ay2[:, None], by2[None]) - np.maximum(aa[:, 1, None], bb[None, :, 1]), 0, None
    )
    inter = iw * ih
    union = (aa[:, 2] * aa[:, 3])[:, None] + (bb[:, 2] * bb[:, 3])[None] - inter
    # union이 0인 칸은 나눗셈 경고를 피하려 분모를 1로 바꾸고 결과는 0으로 둔다
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


@dataclass(frozen=True)
class ApResult:
    """COCO AP 결과. 값이 -1.0이면 정의되지 않음 (정답 클래스가 하나도 없음, pycocotools 규약)."""

    map: float  # 0.50:0.95
    ap50: float
    ap75: float
    per_class: dict[str, float]  # 0.50:0.95
    gt_counts: dict[str, int]  # 클래스 → 정답 박스 수


def _match(
    gts: list[Box], dets: list[DetBox], thresholds: NDArray[np.float64]
) -> NDArray[np.bool_]:
    """(T, len(dets)) TP 여부. dets는 점수 내림차순.

    한 이미지·한 클래스 안의 탐욕 매칭 (pycocotools `evaluateImg`와 같다): IoU 문턱마다 점수가 높은
    예측부터, 아직 안 쓴 정답 중 IoU가 가장 큰 것(문턱 이상)을 가져간다. 동점이면 정답 순서가 뒤인
    것이 이긴다 (`<`로 비교하므로 같은 값이면 갱신) — pycocotools와 같은 동작이다.
    """
    tp = np.zeros((len(thresholds), len(dets)), dtype=bool)
    if not gts or not dets:
        return tp
    ious = box_iou([d.box for d in dets], gts)
    for ti, thr in enumerate(thresholds):
        used = np.zeros(len(gts), dtype=bool)
        for di in range(len(dets)):
            # pycocotools: iou = min(t, 1 - 1e-10). 문턱 1.0에서도 IoU 1인 짝을 맞출 수 있게 한다
            best, best_iou = -1, min(thr, 1 - 1e-10)
            for gi in range(len(gts)):
                if used[gi] or ious[di, gi] < best_iou:
                    continue
                best, best_iou = gi, ious[di, gi]
            if best >= 0:
                used[best] = True
                tp[ti, di] = True
    return tp


def average_precision(
    gts: Sequence[GtBox], dets: Sequence[DetBox], max_dets: int = 100
) -> ApResult:
    """COCO 방식 AP·mAP를 계산한다.

    Args:
        gts: 정답 박스 (모든 이미지).
        dets: 예측 박스 (모든 이미지). 이미지 키가 정답과 같아야 맞출 수 있다.
        max_dets: 이미지·클래스마다 점수 상위 몇 개만 쓸지 (COCO 기본 100).

    Returns:
        `ApResult`. 정답이 있지만 예측이 하나도 없는 클래스는 AP 0이다. 정답 클래스가 하나도 없으면
        모든 값이 -1.0이다 (호출자인 하네스는 정답이 없으면 이 함수를 부르지 않는다).

    계산 비용: 클래스 x 이미지마다 목록을 다시 거른다 (O(C·I·N)). 골든셋 규모에서는 문제가 없지만
    정답·예측이 아주 많으면 미리 묶어 두는 편이 낫다.
    """
    categories = sorted({g.category for g in gts})
    images = sorted({g.image for g in gts} | {d.image for d in dets}, key=repr)
    # precision[T, R, K]: IoU 문턱 x 재현율 점 x 클래스. -1은 "정의되지 않음"(정답 없는 클래스)
    precision = np.full((len(IOU_THRESHOLDS), len(RECALL_POINTS), len(categories)), -1.0)
    gt_counts: dict[str, int] = {}
    for ci, cat in enumerate(categories):
        scores: list[float] = []
        tps: list[NDArray[np.bool_]] = []
        n_gt = 0
        for img in images:
            g = [x.box for x in gts if x.image == img and x.category == cat]
            d: list[DetBox] = [x for x in dets if x.image == img and x.category == cat]
            # 이미지 안에서 점수 내림차순(안정 정렬) 후 상위 max_dets개 (pycocotools evaluateImg)
            order = np.argsort([-x.score for x in d], kind="mergesort")[:max_dets]
            d = [d[i] for i in order]
            n_gt += len(g)
            if d:
                tps.append(_match(g, d, IOU_THRESHOLDS))
                scores += [x.score for x in d]
        gt_counts[cat] = n_gt
        if n_gt == 0:
            continue
        if not scores:
            # 정답은 있는데 예측이 없음 → 모든 재현율 점에서 정밀도 0 (pycocotools와 같다)
            precision[:, :, ci] = 0.0
            continue
        # 모든 이미지의 예측을 점수순으로 합친다 (pycocotools accumulate: mergesort)
        order = np.argsort(-np.asarray(scores), kind="mergesort")
        tp = np.concatenate(tps, axis=1)[:, order]
        tp_sum = np.cumsum(tp, axis=1, dtype=np.float64)
        fp_sum = np.cumsum(~tp, axis=1, dtype=np.float64)
        for ti in range(len(IOU_THRESHOLDS)):
            rc = tp_sum[ti] / n_gt
            pr = tp_sum[ti] / (fp_sum[ti] + tp_sum[ti] + np.spacing(1))
            # 정밀도 포락선: 뒤에서부터 누적 최댓값 (단조 감소)
            pr = np.maximum.accumulate(pr[::-1])[::-1]
            # 재현율 점마다 그 재현율 이상을 처음 달성한 위치의 정밀도. 도달하지 못한 점은 0
            idx = np.searchsorted(rc, RECALL_POINTS, side="left")
            q = np.zeros(len(RECALL_POINTS))
            valid = idx < len(pr)
            q[valid] = pr[idx[valid]]
            precision[ti, :, ci] = q

    def mean(p: NDArray[np.float64]) -> float:
        """-1(정의되지 않음)을 뺀 평균. 모두 -1이면 -1.0 (pycocotools summarize와 같다)."""
        v = p[p > -1]
        return float(v.mean()) if v.size else -1.0

    per_class = {c: mean(precision[:, :, i]) for i, c in enumerate(categories)}
    return ApResult(
        map=mean(precision),
        ap50=mean(precision[0]),
        ap75=mean(precision[5]),
        per_class=per_class,
        gt_counts=gt_counts,
    )
