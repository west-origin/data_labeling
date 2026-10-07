"""시간 구간 지표: 구간 IoU, 시점 사건 매칭(접촉 시작·종료 오차·F1), 구간 F1@IoU, temporal mAP,
경계 일치율.

구간은 (시작 ms, 끝 ms, 클래스) 튜플이다. 시간은 정수 ms다.

참조 정의:
- `segment_f1`: MS-TCN(Farha & Gall, CVPR 2019) 저장소 eval.py의 `f_score` (구간 F1@k).
  test_metrics_temporal_reference.py가 계산 부분을 옮겨 두고 무작위 입력에서 TP·FP·FN이 같은지 본다.
  차이: MS-TCN은 프레임 라벨 열에서 구간을 뽑고 배경 클래스를 빼지만, 여기서는 구간 목록을 바로
  받고 배경 개념이 없다. IoU는 연속 ms 구간으로 계산한다 (프레임 끝을 배타로 보는 MS-TCN과 같은 식).
- `temporal_map`: ActivityNet Evaluation `eval_detection.py`의
  `compute_average_precision_detection`과
  `interpolated_prec_rec` (같은 테스트 파일에서 일치 검사). 기본 tIoU 문턱 0.50:0.05:0.95.
- `match_events`·`boundary_agreement`: 이 프로젝트 정의. 허용 오차 안에서 오차 합 최소 일대일
  매칭(헝가리안) 후 정밀도·재현율·F1. 행동 분할 문헌의 경계 F1(허용 창 안 경계 일치)과 같은
  생각이지만 탐욕 대신 최적 매칭을 쓴다.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np

from dlp_eval.metrics.assign import assign

Interval = tuple[int, int, str]  # (시작 ms, 끝 ms, 클래스). 끝은 시작보다 크거나 같다


def interval_iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    """두 구간 (시작, 끝)의 시간 IoU.

    교집합 길이 / 합집합 길이. 합집합은 두 구간을 덮는 최소 구간 길이(max 끝 - min 시작)로 잰다 —
    떨어진 두 구간이면 교집합이 0이므로 IoU 0이고 결과에 영향이 없다. 두 구간 모두 길이 0이면
    (합집합 0) 같은 점일 때만 1.0, 아니면 0.0이다.
    """
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else float(a == b)


@dataclass(frozen=True)
class EventResult:
    """시점 사건 매칭 결과."""

    tp: int  # 허용 오차 안에서 맞춘 쌍 수
    fp: int  # 맞추지 못한 예측 수
    fn: int  # 맞추지 못한 정답 수
    precision: float  # tp / (tp + fp), 분모 0이면 0.0
    recall: float  # tp / (tp + fn), 분모 0이면 0.0
    f1: float  # 정밀도·재현율 조화 평균, 둘 다 0이면 0.0
    errors_ms: tuple[int, ...]  # 맞춘 쌍의 |예측 - 정답|

    @property
    def mean_error_ms(self) -> float:
        """맞춘 쌍의 평균 절대 오차 (ms). 맞춘 쌍이 없으면 NaN (게이트가 NaN 규칙으로 다룬다)."""
        return float(np.mean(self.errors_ms)) if self.errors_ms else float("nan")

    @property
    def median_error_ms(self) -> float:
        """맞춘 쌍의 절대 오차 중앙값 (ms). 맞춘 쌍이 없으면 NaN."""
        return float(np.median(self.errors_ms)) if self.errors_ms else float("nan")


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """(정밀도, 재현율, F1). 분모가 0이면 그 값은 0.0이다 (정답·예측이 모두 없으면 F1 0)."""
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def match_events(truth: Sequence[int], pred: Sequence[int], tolerance_ms: int) -> EventResult:
    """시점 사건(접촉 시작 등)을 허용 오차 안에서 일대일로 맞춘다 (오차 합 최소).

    Args:
        truth, pred: 사건 시각 (ms). 순서·중복 상관없다.
        tolerance_ms: |예측 - 정답| <= tolerance_ms 이면 맞출 수 있다 (양 끝 포함).

    Returns:
        `EventResult`. 허용 오차를 넘는 짝은 할당되더라도 맞춘 것으로 세지 않는다.

    방법: 허용 밖 칸에 큰 비용(1e9)을 넣은 |차이| 행렬에 헝가리안 할당을 하고, 허용 안의 짝만 TP로
    센다. 맞춘 쌍 수를 먼저 최대로 하는 것이 아니라 비용 합을 최소로 하므로, 허용 밖 칸 비용이
    충분히 커서 허용 안 짝 수가 최대인 할당이 고른다 (1e9 > 가능한 오차 합).
    """
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
    """구간 F1@IoU 결과 (문턱 하나)."""

    threshold: float  # IoU 문턱 (0~1)
    tp: int
    fp: int
    fn: int
    f1: float  # 분모 0이면 0.0


def segment_f1(truth: Sequence[Interval], pred: Sequence[Interval], threshold: float) -> SegmentF1:
    """구간 F1@IoU (MS-TCN 방식).

    예측을 시작 순서로 보며, 같은 클래스 정답 중 IoU가 가장 큰 것이 문턱 이상이고 아직 안 쓰였으면
    TP다.

    MS-TCN `f_score`와 같은 규칙: 가장 IoU가 큰 정답이 이미 쓰였으면 다른 정답을 찾지 않고 FP다
    (차선 정답으로 넘어가지 않는다). 동점이면 앞 정답이 이긴다 (`>`로 비교, numpy argmax와 같다).
    예측은 (시작, 끝, 클래스) 튜플 순서로 정렬해 본다 (MS-TCN은 프레임 열이라 시작 순서가 자연히
    정해진다).

    Args:
        truth, pred: 같은 묶음(세션·손 등) 안의 구간. 묶음이 여럿이면 호출자가 묶음마다 부르고
        TP·FP·FN을
            더한다 (`harness.eval_actions`).
        threshold: IoU 문턱 (`evaluation.yaml segment_iou`·`relation_iou`).

    Returns:
        `SegmentF1`. F1 = 2PR/(P+R).
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

    Args:
        truth: (묶음, 시작 ms, 끝 ms, 클래스). 묶음은 ActivityNet의 video-id에 해당한다
            (하네스는 "세션/손").
        pred: (묶음, 시작 ms, 끝 ms, 클래스, 점수).
        thresholds: tIoU 문턱 목록 (기본 0.50:0.05:0.95, ActivityNet과 같다).

    Returns:
        (클래스별 AP의 평균(문턱 평균), 클래스 → 문턱 평균 AP). 정답 클래스가 없으면 (0.0, {}).

    ActivityNet과의 차이: 점수 정렬은 Python 안정 정렬(ActivityNet은 numpy argsort 기본 quicksort라
    동점 순서가 다를 수 있다). 다른 묶음의 정답은 IoU -1로 두어 어떤 문턱에서도 맞지 않게 한다.
    예측에만 있는 클래스는 평균에 들지 않는다.
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
                # IoU 내림차순으로 보며 문턱 미만이면 멈추고(FP), 이미 쓴 정답은 건너뛴다
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
            # interpolated_prec_rec: 양 끝에 (0, 0)·(1, 0)을 붙이고 정밀도 포락선 아래 넓이
            mprec = np.concatenate([[0.0], prec, [0.0]])
            mrec = np.concatenate([[0.0], rec, [1.0]])
            for i in range(len(mprec) - 2, -1, -1):
                mprec[i] = max(mprec[i], mprec[i + 1])
            idx = np.where(mrec[1:] != mrec[:-1])[0] + 1
            aps.append(float(np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])))
        per_class[cls] = float(np.mean(aps))
    return (float(np.mean(list(per_class.values()))) if per_class else 0.0), per_class


def boundaries(intervals: Sequence[Interval]) -> list[int]:
    """구간들의 경계 시각 (중복 제거, 정렬).

    이웃 구간이 맞닿으면(앞 구간 끝 = 뒤 구간 시작) 경계 하나로 센다.
    """
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

    클래스는 보지 않는다 (경계 위치만 비교). 허용 오차는 `evaluation.yaml tolerance_ms.boundary`.
    양 끝은 정답·예측 합집합 기준이므로, 예측이 정답보다 늦게 시작하면 그 시작은 경계로 남아 FP다.
    """
    t, p = boundaries(truth), boundaries(pred)
    if exclude_extremes and (truth or pred):
        lo = min(s for s, _, _ in (*truth, *pred))
        hi = max(e for _, e, _ in (*truth, *pred))
        t = [x for x in t if x not in (lo, hi)]
        p = [x for x in p if x not in (lo, hi)]
    return match_events(t, p, tolerance_ms)
