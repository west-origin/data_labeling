"""시간 구간 지표: 구간 IoU, 시점 사건 매칭(접촉 시작·종료 오차·F1), 구간 F1@IoU, temporal mAP,
경계 일치율.

구간은 (시작 ms, 끝 ms, 클래스) 튜플이다. 시간은 정수 ms다.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np

from dlp_eval.metrics.assign import assign

Interval = tuple[int, int, str]


def interval_iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else float(a == b)


@dataclass(frozen=True)
class EventResult:
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    errors_ms: tuple[int, ...]  # 맞춘 쌍의 |예측 - 정답|

    @property
    def mean_error_ms(self) -> float:
        return float(np.mean(self.errors_ms)) if self.errors_ms else float("nan")

    @property
    def median_error_ms(self) -> float:
        return float(np.median(self.errors_ms)) if self.errors_ms else float("nan")


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def match_events(truth: Sequence[int], pred: Sequence[int], tolerance_ms: int) -> EventResult:
    """시점 사건(접촉 시작 등)을 허용 오차 안에서 일대일로 맞춘다 (오차 합 최소)."""
    t, p = np.asarray(truth, dtype=np.int64), np.asarray(pred, dtype=np.int64)
    errors: list[int] = []
    if len(t) and len(p):
        diff = np.abs(t[:, None] - p[None, :]).astype(np.float64)
        cost = np.where(diff <= tolerance_ms, diff, 1e9)
        rows, cols = assign(cost)
        errors = [
            int(diff[r, c]) for r, c in zip(rows, cols, strict=True) if diff[r, c] <= tolerance_ms
        ]
    tp = len(errors)
    fp, fn = len(p) - tp, len(t) - tp
    return EventResult(tp, fp, fn, *_prf(tp, fp, fn), tuple(errors))


@dataclass(frozen=True)
class SegmentF1:
    threshold: float
    tp: int
    fp: int
    fn: int
    f1: float


def segment_f1(truth: Sequence[Interval], pred: Sequence[Interval], threshold: float) -> SegmentF1:
    """구간 F1@IoU (MS-TCN 방식).

    예측을 시작 순서로 보며, 같은 클래스 정답 중 IoU가 가장 큰 것이 문턱 이상이고 아직 안 쓰였으면
    TP다.
    """
    used = [False] * len(truth)
    tp = fp = 0
    for ps, pe, pc in sorted(pred):
        best, best_iou = -1, 0.0
        for i, (ts, te, tc) in enumerate(truth):
            if tc != pc:
                continue
            v = interval_iou((ps, pe), (ts, te))
            if v > best_iou:
                best, best_iou = i, v
        if best >= 0 and best_iou >= threshold and not used[best]:
            used[best] = True
            tp += 1
        else:
            fp += 1
    fn = len(truth) - tp
    return SegmentF1(threshold, tp, fp, fn, _prf(tp, fp, fn)[2])


GroupedInterval = tuple[Hashable, int, int, str]  # (영상·손 등 묶음, 시작, 끝, 클래스)


def temporal_map(
    truth: Sequence[GroupedInterval],
    pred: Sequence[tuple[Hashable, int, int, str, float]],
    thresholds: Sequence[float] = tuple(np.round(np.arange(0.5, 0.951, 0.05), 2)),
) -> tuple[float, dict[str, float]]:
    """ActivityNet 방식 temporal mAP. (mAP, 클래스별 AP).

    클래스별로 모든 묶음의 예측을 점수순으로 보고, tIoU 문턱마다 같은 묶음의 정답 중 IoU가 큰 순서로
    아직 안 쓴 것을 가져간다. AP는 정밀도 포락선 아래 넓이다. 정답이 있는 클래스만 평균한다.
    """
    classes = sorted({c for _, _, _, c in truth})
    per_class: dict[str, float] = {}
    for cls in classes:
        gts = [(g, s, e) for g, s, e, c in truth if c == cls]
        dets = sorted(((g, s, e, sc) for g, s, e, c, sc in pred if c == cls), key=lambda d: -d[3])
        aps: list[float] = []
        for thr in thresholds:
            used = [False] * len(gts)
            tp = np.zeros(len(dets))
            for di, (group, s, e, _) in enumerate(dets):
                ious = np.array(
                    [interval_iou((s, e), (gs, ge)) if gg == group else -1.0 for gg, gs, ge in gts]
                )
                for gi in np.argsort(ious, kind="stable")[::-1]:
                    if ious[gi] < thr:
                        break
                    if not used[gi]:
                        used[gi] = True
                        tp[di] = 1
                        break
            ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
            rec = ctp / len(gts)
            prec = ctp / np.maximum(ctp + cfp, np.finfo(np.float64).eps)
            mprec = np.concatenate([[0.0], prec, [0.0]])
            mrec = np.concatenate([[0.0], rec, [1.0]])
            for i in range(len(mprec) - 2, -1, -1):
                mprec[i] = max(mprec[i], mprec[i + 1])
            idx = np.where(mrec[1:] != mrec[:-1])[0] + 1
            aps.append(float(np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])))
        per_class[cls] = float(np.mean(aps))
    return (float(np.mean(list(per_class.values()))) if per_class else 0.0), per_class


def boundaries(intervals: Sequence[Interval]) -> list[int]:
    """구간들의 경계 시각 (중복 제거, 정렬)."""
    return sorted({t for s, e, _ in intervals for t in (s, e)})


def boundary_agreement(
    truth: Sequence[Interval],
    pred: Sequence[Interval],
    tolerance_ms: int,
    *,
    exclude_extremes: bool = False,
) -> EventResult:
    """경계 일치율: 경계 시각을 허용 오차 안에서 일대일로 맞춘 F1.

    exclude_extremes: 정답·예측을 합친 타임라인의 맨 앞 시작과 맨 뒤 끝 시각을 경계에서 뺀다.
    공백 없이 채운 타임라인(행동 구간)은 양 끝이 항상 세션 처음·끝이라 공짜로 맞기 때문이다.
    """
    t, p = boundaries(truth), boundaries(pred)
    if exclude_extremes and (truth or pred):
        lo = min(s for s, _, _ in (*truth, *pred))
        hi = max(e for _, e, _ in (*truth, *pred))
        t = [x for x in t if x not in (lo, hi)]
        p = [x for x in p if x not in (lo, hi)]
    return match_events(t, p, tolerance_ms)
