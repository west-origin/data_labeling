"""다중 객체 추적 지표: HOTA, IDF1, CLEAR(MOTA). TrackEval과 같은 계산.

입력은 시각마다 (정답 ID 배열, 예측 ID 배열, 유사도 행렬)이다. ID는 0부터 이어진 정수로 바꿔 넣는다.
유사도는 박스 IoU 등 0~1 값이다.

참조:
- HOTA: Luiten et al. IJCV 2021, TrackEval `trackeval/metrics/hota.py` `eval_sequence`.
  alpha 0.05:0.05:0.95(19개) 평균. DetA·AssA·DetRe·DetPr·LocA도 낸다.
- IDF1: Ristani et al. ECCV 2016, TrackEval `identity.py` (전역 ID 일대일 할당, 유사도 문턱).
- CLEAR MOT: Bernardin & Stiefelhagen 2008, TrackEval `clear.py` (MOTA·MOTP·ID 전환).
  MT/ML/Frag 등 나머지 CLEAR 필드는 내지 않는다.
test_metrics_reference.py가 무작위 시퀀스에서 TrackEval과 1e-12 안에서 같은지 본다.

하네스는 여러 세션·스트림을 한 시퀀스로 이어 붙여 넣는다. 개체 키에 세션·스트림을 붙여 ID가
시퀀스끼리 겹치지 않고 한 "프레임"에는 한 (세션, 스트림)만 들어가므로, 결과는 TrackEval이 시퀀스별로
계산해 합친 값(COMBINED_SEQ: 정수 필드 합, AssA·LocA는 TP 가중 평균)과 같다. 시퀀스별 HOTA의 단순
평균은 아니다. 정답 키프레임 시각만 "프레임"으로 쓴다 (`harness.eval_objects`).
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from dlp_eval.metrics.assign import assign

# TrackEval과 같은 부동소수 허용치 (문턱 비교에서 반올림 오차를 흡수)
EPS = float(np.finfo("float").eps)
# HOTA 위치 정확도 문턱 alpha = 0.05, 0.10, …, 0.95 (19개, TrackEval과 같다)
ALPHAS = np.arange(0.05, 0.99, 0.05)

Ids = NDArray[np.int64]
Sim = NDArray[np.float64]


@dataclass(frozen=True)
class TrackingData:
    """TrackEval `eval_sequence` 입력과 같은 구조 (한 시퀀스)."""

    num_gt_ids: int  # 정답 개체 수 (ID는 0..num_gt_ids-1)
    num_tracker_ids: int  # 예측 트랙 수 (ID는 0..num_tracker_ids-1)
    gt_ids: list[Ids]  # 시각마다 그 시각에 있는 정답 ID
    tracker_ids: list[Ids]  # 시각마다 그 시각에 있는 예측 ID
    similarity: list[Sim]  # (len(gt_ids[t]), len(tracker_ids[t]))

    @property
    def num_gt_dets(self) -> int:
        """모든 시각의 정답 검출 수."""
        return sum(len(x) for x in self.gt_ids)

    @property
    def num_tracker_dets(self) -> int:
        """모든 시각의 예측 검출 수."""
        return sum(len(x) for x in self.tracker_ids)

    @classmethod
    def from_frames(
        cls,
        frames: Sequence[tuple[Sequence[Hashable], Sequence[Hashable], Sim]],
    ) -> TrackingData:
        """시각별 (정답 개체 키, 예측 개체 키, 유사도)에서 만든다. 키는 무엇이든 된다.

        키는 처음 나온 순서대로 0부터 정수 ID로 바꾼다. 유사도는 (정답 수, 예측 수) 모양으로 바꾼다
        (빈 시각도 (0, n)·(n, 0) 모양이 되게). 한 시각에 같은 키가 두 번 나오면 같은 ID가 되므로
        호출자가 키를 유일하게 만들어야 한다.
        """
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
    """HOTA와 하위 지표 (모두 alpha 19개의 평균)."""

    hota: float  # sqrt(DetA * AssA)
    det_a: float  # 검출 정확도 TP / (TP + FN + FP)
    ass_a: float  # 연결 정확도
    det_re: float  # 검출 재현율
    det_pr: float  # 검출 정밀도
    loc_a: float  # 위치 정확도 (맞춘 쌍 평균 유사도)


def hota(data: TrackingData) -> HotaResult:
    """HOTA (TrackEval `HOTA.eval_sequence`와 같은 계산).

    단계:
    1. 전역 정렬 점수: 정답 i·예측 j가 함께 나온 시각에서 유사도를 정규화해 누적한 잠재 매칭 수를
       |i| + |j| - 잠재 매칭 수로 나눈다 (트랙 수준 Jaccard).
    2. 시각마다 "전역 점수 x 유사도"를 최대로 하는 헝가리안 매칭을 하고, alpha마다 유사도 >= alpha인
       짝만 TP로 센다.
    3. alpha마다 AssA = TP 가중 평균 연결 IoU, DetA = TP/(TP+FN+FP), HOTA = sqrt(DetA·AssA).

    정답이나 예측 검출이 하나도 없으면 모두 0, LocA 1.0 (TrackEval과 같다).
    """
    n_a = len(ALPHAS)
    tp, fn, fp = np.zeros(n_a), np.zeros(n_a), np.zeros(n_a)
    loc = np.zeros(n_a)
    if data.num_tracker_dets == 0 or data.num_gt_dets == 0:
        return HotaResult(0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

    # 1) 전역 정렬 점수 (트랙 쌍마다)
    potential = np.zeros((data.num_gt_ids, data.num_tracker_ids))
    gt_count = np.zeros((data.num_gt_ids, 1))
    tr_count = np.zeros((1, data.num_tracker_ids))
    for g, p, s in zip(data.gt_ids, data.tracker_ids, data.similarity, strict=True):
        # 유사도를 행·열 합으로 정규화 (한 검출이 여러 짝과 겹칠 때 몫을 나눈다)
        denom = s.sum(0)[np.newaxis, :] + s.sum(1)[:, np.newaxis] - s
        sim_iou = np.zeros_like(s)
        mask = denom > 0 + EPS
        sim_iou[mask] = s[mask] / denom[mask]
        potential[g[:, np.newaxis], p[np.newaxis, :]] += sim_iou
        gt_count[g] += 1
        tr_count[0, p] += 1
    global_score = potential / (gt_count + tr_count - potential)

    # 2) 시각별 매칭과 alpha별 TP·FN·FP
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

    # 3) alpha별 최종 지표
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
    """IDF1 결과."""

    idf1: float  # IDTP / (IDTP + 0.5·IDFP + 0.5·IDFN)
    idtp: int
    idfp: int
    idfn: int


def identity(data: TrackingData, threshold: float = 0.5) -> IdentityResult:
    """IDF1 (TrackEval `Identity.eval_sequence`와 같은 계산).

    정답 트랙과 예측 트랙을 시퀀스 전체에서 일대일로 묶는다 (한 정답 트랙은 한 예측 트랙에만).
    묶인 쌍이 같은 시각에 유사도 >= threshold이면 IDTP다. 묶이지 않는 경우를 표현하려고 가상 정답·
    가상 예측을 붙인 (ng+nt) 정방 행렬에서 IDFN + IDFP 합을 최소로 하는 헝가리안 할당을 한다.

    Args:
        threshold: 유사도 문턱 (하네스는 `evaluation.yaml track_iou`).

    Returns:
        `IdentityResult`. 예측이 없으면 IDF1 0 (IDFN = 정답 수), 정답이 없으면 IDF1 0 (IDFP = 예측
        수).
    """
    if data.num_tracker_dets == 0:
        return IdentityResult(0.0, 0, 0, data.num_gt_dets)
    if data.num_gt_dets == 0:
        return IdentityResult(0.0, 0, data.num_tracker_dets, 0)
    ng, nt = data.num_gt_ids, data.num_tracker_ids
    # potential[i, j]: 정답 i·예측 j가 문턱 이상으로 겹친 시각 수
    potential = np.zeros((ng, nt))
    gt_count, tr_count = np.zeros(ng), np.zeros(nt)
    for g, p, s in zip(data.gt_ids, data.tracker_ids, data.similarity, strict=True):
        mg, mp = np.nonzero(s >= threshold)
        potential[g[mg], p[mp]] += 1
        gt_count[g] += 1
        tr_count[p] += 1
    # 비용 행렬 구성 (TrackEval과 같다): 왼쪽 위 = 실제 쌍, 오른쪽 위 = 정답 i를 짝 없이 둠(대각만
    # 허용), 왼쪽 아래 = 예측 j를 짝 없이 둠(대각만 허용), 오른쪽 아래 = 가상끼리 (비용 0). 1e10은
    # 금지 칸
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
    """CLEAR MOT 결과 (일부 필드)."""

    mota: float  # (TP - FP - IDSW) / 정답 수. 음수가 될 수 있다
    motp: float  # 맞춘 쌍 평균 유사도
    tp: int
    fp: int
    fn: int
    idsw: int  # ID 전환 수


def clear(data: TrackingData, threshold: float = 0.5) -> ClearResult:
    """CLEAR MOT (TrackEval `CLEAR.eval_sequence`와 같은 계산).

    시각마다 유사도 >= threshold인 짝 중에서 헝가리안 매칭을 하되, 바로 앞 시각에 맞았던 (정답,
    예측) 짝을 1000점 가산으로 우선한다 (트랙 유지). 정답이 마지막으로 맞았던 예측 ID와 다른 예측에
    맞으면 ID 전환이다.

    Args:
        threshold: 유사도 문턱 (하네스는 `evaluation.yaml track_iou`).

    Returns:
        `ClearResult`. 예측이 없으면 MOTA 0, 정답이 없으면 MOTA = -예측 수 (TrackEval과 같다).
    """
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
        # 문턱 미만 짝은 점수 0 → 아래에서 맞춘 것으로 세지 않는다
        score[s < threshold - EPS] = 0
        rows, cols = assign(-score)
        ok = score[rows, cols] > 0 + EPS
        rows, cols = rows[ok], cols[ok]
        mg, mp = g[rows], p[cols]
        before: NDArray[np.float64] = prev_tr[mg]
        switched = ~np.isnan(before) & (mp.astype(np.float64) != before)
        idsw += int(np.count_nonzero(switched))
        prev_tr[mg] = mp
        # 앞 시각 우선 정보는 이번 시각에 맞은 짝만 남긴다 (끊기면 우선이 사라진다)
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
