"""객체 검출 AP (COCO 방식, pycocotools와 같은 결과).

- 이미지(=세션·시각)·클래스마다 점수 내림차순으로 예측을 보고, IoU가 문턱 이상인 아직 안 쓴 정답 중
  IoU가 가장 큰 것과 맞춘다. 이미지당 점수 상위 max_dets개만 쓴다.
- 클래스마다 모든 이미지의 예측을 점수순(안정 정렬)으로 모아 누적 TP·FP로 정밀도-재현율을 만들고,
  정밀도를 단조 감소로 만든 뒤 재현율 0, 0.01, …, 1 101점에서 읽어 평균한다.
- mAP는 정답이 있는 클래스의 AP 평균, IoU 문턱 0.50:0.05:0.95 평균이다 (AP50, AP75도 낸다).
군중(crowd)·면적 구간은 쓰지 않는다 (면적 "all").
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Box = tuple[float, float, float, float]  # x, y, w, h
IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
RECALL_POINTS = np.linspace(0.0, 1.0, 101)


@dataclass(frozen=True)
class GtBox:
    image: Hashable
    category: str
    box: Box


@dataclass(frozen=True)
class DetBox:
    image: Hashable
    category: str
    box: Box
    score: float


def box_iou(a: Sequence[Box], b: Sequence[Box]) -> NDArray[np.float64]:
    """(len(a), len(b)) IoU 행렬."""
    if not a or not b:
        return np.zeros((len(a), len(b)))
    aa, bb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    ax2, ay2 = aa[:, 0] + aa[:, 2], aa[:, 1] + aa[:, 3]
    bx2, by2 = bb[:, 0] + bb[:, 2], bb[:, 1] + bb[:, 3]
    iw = np.clip(
        np.minimum(ax2[:, None], bx2[None]) - np.maximum(aa[:, 0, None], bb[None, :, 0]), 0, None
    )
    ih = np.clip(
        np.minimum(ay2[:, None], by2[None]) - np.maximum(aa[:, 1, None], bb[None, :, 1]), 0, None
    )
    inter = iw * ih
    union = (aa[:, 2] * aa[:, 3])[:, None] + (bb[:, 2] * bb[:, 3])[None] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


@dataclass(frozen=True)
class ApResult:
    map: float  # 0.50:0.95
    ap50: float
    ap75: float
    per_class: dict[str, float]  # 0.50:0.95
    gt_counts: dict[str, int]


def _match(
    gts: list[Box], dets: list[DetBox], thresholds: NDArray[np.float64]
) -> NDArray[np.bool_]:
    """(T, len(dets)) TP 여부. dets는 점수 내림차순."""
    tp = np.zeros((len(thresholds), len(dets)), dtype=bool)
    if not gts or not dets:
        return tp
    ious = box_iou([d.box for d in dets], gts)
    for ti, thr in enumerate(thresholds):
        used = np.zeros(len(gts), dtype=bool)
        for di in range(len(dets)):
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
    categories = sorted({g.category for g in gts})
    images = sorted({g.image for g in gts} | {d.image for d in dets}, key=repr)
    precision = np.full((len(IOU_THRESHOLDS), len(RECALL_POINTS), len(categories)), -1.0)
    gt_counts: dict[str, int] = {}
    for ci, cat in enumerate(categories):
        scores: list[float] = []
        tps: list[NDArray[np.bool_]] = []
        n_gt = 0
        for img in images:
            g = [x.box for x in gts if x.image == img and x.category == cat]
            d: list[DetBox] = [x for x in dets if x.image == img and x.category == cat]
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
            precision[:, :, ci] = 0.0
            continue
        order = np.argsort(-np.asarray(scores), kind="mergesort")
        tp = np.concatenate(tps, axis=1)[:, order]
        tp_sum = np.cumsum(tp, axis=1, dtype=np.float64)
        fp_sum = np.cumsum(~tp, axis=1, dtype=np.float64)
        for ti in range(len(IOU_THRESHOLDS)):
            rc = tp_sum[ti] / n_gt
            pr = tp_sum[ti] / (fp_sum[ti] + tp_sum[ti] + np.spacing(1))
            pr = np.maximum.accumulate(pr[::-1])[::-1]
            idx = np.searchsorted(rc, RECALL_POINTS, side="left")
            q = np.zeros(len(RECALL_POINTS))
            valid = idx < len(pr)
            q[valid] = pr[idx[valid]]
            precision[ti, :, ci] = q

    def mean(p: NDArray[np.float64]) -> float:
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
