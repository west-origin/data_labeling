"""구간 지표를 공개 참조 구현의 계산 방식과 대조한다.

참조 구현은 런타임·개발 의존성이 아니라서 (MS-TCN 저장소의 eval.py, ActivityNet Evaluation의
eval_detection.py) 계산 부분을 이 파일에 그대로 옮겨 두고, 무작위 입력에서 결과가 같은지 본다.
- MS-TCN `f_score`: 프레임 라벨 열에서 구간을 뽑아 예측을 시간 순서로 맞춘다.
- ActivityNet `compute_average_precision_detection` + `interpolated_prec_rec`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from dlp_eval.metrics.temporal import segment_f1, temporal_map

# ---------------------------------------------------------------- MS-TCN eval.py (옮김)


def _mstcn_segments(frame_labels: Sequence[str]) -> tuple[list[str], list[int], list[int]]:
    """get_labels_start_end_time (배경 클래스 없음)."""
    labels: list[str] = [frame_labels[0]]
    starts: list[int] = [0]
    ends: list[int] = []
    last = frame_labels[0]
    for i in range(len(frame_labels)):
        if frame_labels[i] != last:
            labels.append(frame_labels[i])
            starts.append(i)
            ends.append(i)
            last = frame_labels[i]
    ends.append(len(frame_labels))
    return labels, starts, ends


def _mstcn_f_score(
    recognized: Sequence[str], ground_truth: Sequence[str], overlap: float
) -> tuple[int, int, int]:
    """MS-TCN eval.py `f_score` (옮김). 프레임 라벨 열 두 개 → (TP, FP, FN)."""
    p_label, p_start, p_end = _mstcn_segments(recognized)
    y_label, y_start, y_end = _mstcn_segments(ground_truth)
    tp = fp = 0
    hits = np.zeros(len(y_label))
    ys, ye = np.array(y_start), np.array(y_end)
    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], ye) - np.maximum(p_start[j], ys)
        union = np.maximum(p_end[j], ye) - np.minimum(p_start[j], ys)
        iou = (1.0 * intersection / union) * np.array([p_label[j] == y for y in y_label])
        idx = int(np.array(iou).argmax())
        if iou[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - int(hits.sum())
    return tp, fp, fn


def _frames(rng: np.random.Generator, n: int, classes: Sequence[str]) -> list[str]:
    """무작위 길이 구간으로 채운 프레임 라벨 열 (이웃 구간은 클래스가 다르다)."""
    out: list[str] = []
    last = ""
    while len(out) < n:
        c = str(rng.choice([x for x in classes if x != last]))
        out += [c] * int(rng.integers(3, 40))
        last = c
    return out[:n]


def _intervals(frame_labels: Sequence[str]) -> list[tuple[int, int, str]]:
    """프레임 라벨 열 → 우리 구현 입력 구간 (프레임 하나 = 10 ms로 본다)."""
    labels, starts, ends = _mstcn_segments(frame_labels)
    return [(s * 10, e * 10, c) for c, s, e in zip(labels, starts, ends, strict=True)]  # ms


@pytest.mark.parametrize("seed", range(20))
def test_segment_f1_matches_mstcn(seed: int) -> None:
    """무작위 프레임 라벨 열에서 구간 F1@{0.1, 0.25, 0.5}의 TP·FP·FN이 MS-TCN과 같다."""
    rng = np.random.default_rng(seed)
    classes = ["rub", "push", "spray", "wring"]
    truth = _frames(rng, 400, classes)
    pred = _frames(rng, 400, classes)
    for overlap in (0.1, 0.25, 0.5):
        ref = _mstcn_f_score(pred, truth, overlap)
        r = segment_f1(_intervals(truth), _intervals(pred), overlap)
        assert (r.tp, r.fp, r.fn) == ref


def test_segment_f1_mstcn_hand_example() -> None:
    """손 계산 예제에서 MS-TCN 참조와 우리 구현이 같은 답(문턱별 TP·FP·FN)을 낸다."""
    # 정답 rub 0~50, push 50~100 / 예측 rub 0~30, push 30~100
    # rub IoU .6, push IoU 50/70=.714 → @0.5 둘 다 TP, @0.65 push만 TP
    truth = ["rub"] * 50 + ["push"] * 50
    pred = ["rub"] * 30 + ["push"] * 70
    assert _mstcn_f_score(pred, truth, 0.5) == (2, 0, 0)
    assert _mstcn_f_score(pred, truth, 0.65) == (1, 1, 1)
    r = segment_f1(_intervals(truth), _intervals(pred), 0.65)
    assert (r.tp, r.fp, r.fn) == (1, 1, 1) and r.f1 == pytest.approx(0.5)


# ---------------------------------------------------------------- ActivityNet (옮김)


def _segment_iou(target: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """ActivityNet `segment_iou` (옮김): 구간 하나와 후보 구간들의 tIoU."""
    tt1 = np.maximum(target[0], candidates[:, 0])
    tt2 = np.minimum(target[1], candidates[:, 1])
    inter = (tt2 - tt1).clip(0)
    union = (candidates[:, 1] - candidates[:, 0]) + (target[1] - target[0]) - inter
    return inter.astype(float) / union


def _interpolated_prec_rec(prec: np.ndarray, rec: np.ndarray) -> float:
    """ActivityNet `interpolated_prec_rec` (옮김): 정밀도 포락선 아래 넓이 AP."""
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    return float(np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx]))


def _anet_ap(
    gts: list[tuple[str, int, int]],
    preds: list[tuple[str, int, int, float]],
    thresholds: np.ndarray,
) -> np.ndarray:
    """ActivityNet `compute_average_precision_detection` (옮김, 한 클래스). 문턱별 AP 배열."""
    ap = np.zeros(len(thresholds))
    if not preds:
        return ap
    npos = float(len(gts))
    lock_gt = np.ones((len(thresholds), len(gts))) * -1
    order = np.argsort([-p[3] for p in preds], kind="stable")
    preds = [preds[i] for i in order]
    tp = np.zeros((len(thresholds), len(preds)))
    fp = np.zeros((len(thresholds), len(preds)))
    by_video: dict[str, list[int]] = {}
    for i, (v, _, _) in enumerate(gts):
        by_video.setdefault(v, []).append(i)
    for idx, (video, s, e, _) in enumerate(preds):
        gidx = by_video.get(video)
        if gidx is None:
            fp[:, idx] = 1
            continue
        cand = np.array([[gts[i][1], gts[i][2]] for i in gidx])
        tiou_arr = _segment_iou(np.array([s, e]), cand)
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for tidx, tiou_thr in enumerate(thresholds):
            for jdx in tiou_sorted_idx:
                if tiou_arr[jdx] < tiou_thr:
                    fp[tidx, idx] = 1
                    break
                if lock_gt[tidx, gidx[jdx]] >= 0:
                    continue
                tp[tidx, idx] = 1
                lock_gt[tidx, gidx[jdx]] = idx
                break
            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1
    tp_cumsum = np.cumsum(tp, axis=1).astype(float)
    fp_cumsum = np.cumsum(fp, axis=1).astype(float)
    recall = tp_cumsum / npos
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    for tidx in range(len(thresholds)):
        ap[tidx] = _interpolated_prec_rec(precision[tidx, :], recall[tidx, :])
    return ap


@pytest.mark.parametrize("seed", range(20))
def test_temporal_map_matches_activitynet(seed: int) -> None:
    """영상 3개의 무작위 정답·흔든 예측·오탐에서 클래스별 AP와 mAP가 ActivityNet과 같다."""
    rng = np.random.default_rng(100 + seed)
    classes = ["rub", "push", "spray"]
    videos = ["v1", "v2", "v3"]
    truth: list[tuple[str, int, int, str]] = []
    pred: list[tuple[str, int, int, str, float]] = []
    for v in videos:
        for _ in range(int(rng.integers(2, 6))):
            s = int(rng.integers(0, 10_000))
            e = s + int(rng.integers(200, 3000))
            c = str(rng.choice(classes))
            truth.append((v, s, e, c))
            # 정답을 흔든 예측과 엉뚱한 예측
            js, je = int(rng.integers(-400, 400)), int(rng.integers(-400, 400))
            pred.append((v, s + js, max(s + js + 50, e + je), c, float(rng.random())))
        for _ in range(int(rng.integers(0, 4))):
            s = int(rng.integers(0, 10_000))
            pred.append((v, s, s + int(rng.integers(100, 2000)), str(rng.choice(classes)),
                         float(rng.random())))  # fmt: skip
    thresholds = np.round(np.arange(0.5, 0.951, 0.05), 2)
    ours, per_class = temporal_map(truth, pred, tuple(thresholds))
    ref: dict[str, float] = {}
    for c in sorted({t[3] for t in truth}):
        gts = [(v, s, e) for v, s, e, cc in truth if cc == c]
        ps = [(v, s, e, sc) for v, s, e, cc, sc in pred if cc == c]
        ref[c] = float(np.mean(_anet_ap(gts, ps, thresholds)))
    assert per_class == pytest.approx(ref, abs=1e-12)
    assert ours == pytest.approx(float(np.mean(list(ref.values()))), abs=1e-12)
