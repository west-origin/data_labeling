"""다중 객체 추적 지표: HOTA, IDF1, CLEAR(MOTA). TrackEval과 같은 계산.

입력은 시각마다 (정답 ID 배열, 예측 ID 배열, 유사도 행렬)이다. ID는 0부터 이어진 정수로 바꿔 넣는다.
유사도는 박스 IoU 등 0~1 값이다.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from dlp_eval.metrics.assign import assign

EPS = float(np.finfo("float").eps)
ALPHAS = np.arange(0.05, 0.99, 0.05)

Ids = NDArray[np.int64]
Sim = NDArray[np.float64]


@dataclass(frozen=True)
class TrackingData:
    num_gt_ids: int
    num_tracker_ids: int
    gt_ids: list[Ids]
    tracker_ids: list[Ids]
    similarity: list[Sim]  # (len(gt_ids[t]), len(tracker_ids[t]))

    @property
    def num_gt_dets(self) -> int:
        return sum(len(x) for x in self.gt_ids)

    @property
    def num_tracker_dets(self) -> int:
        return sum(len(x) for x in self.tracker_ids)

    @classmethod
    def from_frames(
        cls,
        frames: Sequence[tuple[Sequence[Hashable], Sequence[Hashable], Sim]],
    ) -> TrackingData:
        """시각별 (정답 개체 키, 예측 개체 키, 유사도)에서 만든다. 키는 무엇이든 된다."""
        gt_map: dict[Hashable, int] = {}
        tr_map: dict[Hashable, int] = {}
        gt_ids: list[Ids] = []
        tr_ids: list[Ids] = []
        sims: list[Sim] = []
        for g, p, s in frames:
            gt_ids.append(np.array([gt_map.setdefault(k, len(gt_map)) for k in g], dtype=np.int64))
            tr_ids.append(np.array([tr_map.setdefault(k, len(tr_map)) for k in p], dtype=np.int64))
            sims.append(np.asarray(s, dtype=np.float64).reshape(len(g), len(p)))
        return cls(len(gt_map), len(tr_map), gt_ids, tr_ids, sims)


@dataclass(frozen=True)
class HotaResult:
    hota: float
    det_a: float
    ass_a: float
    det_re: float
    det_pr: float
    loc_a: float


def hota(data: TrackingData) -> HotaResult:
    n_a = len(ALPHAS)
    tp, fn, fp = np.zeros(n_a), np.zeros(n_a), np.zeros(n_a)
    loc = np.zeros(n_a)
    if data.num_tracker_dets == 0 or data.num_gt_dets == 0:
        return HotaResult(0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

    potential = np.zeros((data.num_gt_ids, data.num_tracker_ids))
    gt_count = np.zeros((data.num_gt_ids, 1))
    tr_count = np.zeros((1, data.num_tracker_ids))
    for g, p, s in zip(data.gt_ids, data.tracker_ids, data.similarity, strict=True):
        denom = s.sum(0)[np.newaxis, :] + s.sum(1)[:, np.newaxis] - s
        sim_iou = np.zeros_like(s)
        mask = denom > 0 + EPS
        sim_iou[mask] = s[mask] / denom[mask]
        potential[g[:, np.newaxis], p[np.newaxis, :]] += sim_iou
        gt_count[g] += 1
        tr_count[0, p] += 1
    global_score = potential / (gt_count + tr_count - potential)

    matches = [np.zeros_like(potential) for _ in ALPHAS]
    for g, p, s in zip(data.gt_ids, data.tracker_ids, data.similarity, strict=True):
        if len(g) == 0:
            fp += len(p)
            continue
        if len(p) == 0:
            fn += len(g)
            continue
        score = global_score[g[:, np.newaxis], p[np.newaxis, :]] * s
        rows, cols = assign(-score)
        for a, alpha in enumerate(ALPHAS):
            ok = s[rows, cols] >= alpha - EPS
            r, c = rows[ok], cols[ok]
            n = len(r)
            tp[a] += n
            fn[a] += len(g) - n
            fp[a] += len(p) - n
            if n > 0:
                loc[a] += float(s[r, c].sum())
                matches[a][g[r], p[c]] += 1

    ass_a = np.zeros(n_a)
    for a in range(n_a):
        m = matches[a]
        ass = m / np.maximum(1, gt_count + tr_count - m)
        ass_a[a] = float(np.sum(m * ass) / np.maximum(1, tp[a]))
    loc_a = np.maximum(1e-10, loc) / np.maximum(1e-10, tp)
    det_re = tp / np.maximum(1, tp + fn)
    det_pr = tp / np.maximum(1, tp + fp)
    det_a = tp / np.maximum(1, tp + fn + fp)
    h = np.sqrt(det_a * ass_a)
    return HotaResult(
        float(h.mean()), float(det_a.mean()), float(ass_a.mean()),
        float(det_re.mean()), float(det_pr.mean()), float(loc_a.mean()),
    )  # fmt: skip


@dataclass(frozen=True)
class IdentityResult:
    idf1: float
    idtp: int
    idfp: int
    idfn: int


def identity(data: TrackingData, threshold: float = 0.5) -> IdentityResult:
    if data.num_tracker_dets == 0:
        return IdentityResult(0.0, 0, 0, data.num_gt_dets)
    if data.num_gt_dets == 0:
        return IdentityResult(0.0, 0, data.num_tracker_dets, 0)
    ng, nt = data.num_gt_ids, data.num_tracker_ids
    potential = np.zeros((ng, nt))
    gt_count, tr_count = np.zeros(ng), np.zeros(nt)
    for g, p, s in zip(data.gt_ids, data.tracker_ids, data.similarity, strict=True):
        mg, mp = np.nonzero(s >= threshold)
        potential[g[mg], p[mp]] += 1
        gt_count[g] += 1
        tr_count[p] += 1
    fp_mat = np.zeros((ng + nt, ng + nt))
    fn_mat = np.zeros((ng + nt, ng + nt))
    fp_mat[ng:, :nt] = 1e10
    fn_mat[:ng, nt:] = 1e10
    for i in range(ng):
        fn_mat[i, :nt] = gt_count[i]
        fn_mat[i, nt + i] = gt_count[i]
    for j in range(nt):
        fp_mat[:ng, j] = tr_count[j]
        fp_mat[j + ng, j] = tr_count[j]
    fn_mat[:ng, :nt] -= potential
    fp_mat[:ng, :nt] -= potential
    rows, cols = assign(fn_mat + fp_mat)
    idfn = int(fn_mat[rows, cols].sum())
    idfp = int(fp_mat[rows, cols].sum())
    idtp = int(gt_count.sum() - idfn)
    return IdentityResult(idtp / max(1.0, idtp + 0.5 * idfp + 0.5 * idfn), idtp, idfp, idfn)


@dataclass(frozen=True)
class ClearResult:
    mota: float
    motp: float
    tp: int
    fp: int
    fn: int
    idsw: int


def clear(data: TrackingData, threshold: float = 0.5) -> ClearResult:
    if data.num_tracker_dets == 0:
        return ClearResult(0.0, 0.0, 0, 0, data.num_gt_dets, 0)
    if data.num_gt_dets == 0:
        return ClearResult(-float(data.num_tracker_dets), 0.0, 0, data.num_tracker_dets, 0, 0)
    tp = fp = fn = idsw = 0
    motp_sum = 0.0
    prev_tr = np.full(data.num_gt_ids, np.nan)  # 마지막으로 맞은 예측 ID (ID 전환 판정)
    prev_step = np.full(data.num_gt_ids, np.nan)  # 바로 앞 시각에 맞은 예측 ID (매칭 우선)
    for g, p, s in zip(data.gt_ids, data.tracker_ids, data.similarity, strict=True):
        if len(g) == 0:
            fp += len(p)
            continue
        if len(p) == 0:
            fn += len(g)
            continue
        score = 1000 * (p[np.newaxis, :] == prev_step[g[:, np.newaxis]]) + s
        score[s < threshold - EPS] = 0
        rows, cols = assign(-score)
        ok = score[rows, cols] > 0 + EPS
        rows, cols = rows[ok], cols[ok]
        mg, mp = g[rows], p[cols]
        before: NDArray[np.float64] = prev_tr[mg]
        switched = ~np.isnan(before) & (mp.astype(np.float64) != before)
        idsw += int(np.count_nonzero(switched))
        prev_tr[mg] = mp
        prev_step[:] = np.nan
        prev_step[mg] = mp
        n = len(mg)
        tp += n
        fn += len(g) - n
        fp += len(p) - n
        if n:
            motp_sum += float(s[rows, cols].sum())
    mota = (tp - fp - idsw) / max(1.0, tp + fn)
    return ClearResult(mota, motp_sum / max(1.0, tp), tp, fp, fn, idsw)
