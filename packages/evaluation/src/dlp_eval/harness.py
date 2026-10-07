"""라벨 레코드 → 과제별 지표. 골든셋 세션들의 정답(사람)과 예측(모델 버전)을 비교한다.

과제와 라벨 종류:
- objects: box_track → mAP(0.50:0.95), AP50, HOTA, IDF1, MOTA, ECE.
  정답 키프레임 시각(모든 정답 트랙의 합집합)에서 비교한다. 사람 정답은 키프레임이 성기다 (CVAT가
  사이를 보간한다). 그래서 정답 트랙은 자기 키프레임 사이를 간격 제한 없이 선형 보간하고 (화면 밖
  키프레임이 끼면 그 사이는 없음), 예측은 과제별 max_interp_ms 이내에서만 보간한다 (프리라벨
  트래커가 한 트랙으로 잇는 끊김 이상, evaluation.yaml).
- hands / body: keypoint_track(hand21 / coco17) → PCK. 정답 시각마다 손은 같은 스트림의 같은 쪽
  손끼리 관절 거리로, 전신은 같은 스트림에서 보이는 관절 박스 IoU로 일대일 맞춘다 (헝가리안).
- contact: hand_state(접촉 대상 있음) → 접촉 시작·종료 F1과 오차(ms), 파지 유형 macro F1.
  장갑 세션은 contact_glove 허용 오차를 쓴다.
- actions: action → 구간 F1@IoU, temporal mAP, 경계 일치 F1(타임라인 양 끝 제외), 동사 macro F1.
- relations: relation → 같은 (주어, 술어, 목적어, 부분) 구간 F1@relation_iou.
- states: object_state → 상태 전이 정확도.
- coverage: coverage → 같은 (표면, 도구) 쌍의 비율 절대 오차 평균.
- privacy: blur_track → 블러 재현(정답 박스가 예측 블러로 충분히 덮인 비율)과 정밀.
공간 라벨의 키프레임 시각은 그 스트림의 PTS 시각이다 (ADR 0019). 그래서 공간 과제(objects·hands·
body·privacy)는 정답과 같은 스트림의 예측만 비교하고, 영상·개체 ID에 세션과 스트림 ID를 붙여
구분한다 (추적 지표는 여러 시퀀스를 이어 붙인 것과 같다).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from dlp_eval.metrics.assign import assign
from dlp_eval.metrics.classification import ece, macro_f1, under_sampled
from dlp_eval.metrics.detection import Box, DetBox, GtBox, average_precision, box_iou
from dlp_eval.metrics.keypoints import pck
from dlp_eval.metrics.states import StateSpan, transition_accuracy
from dlp_eval.metrics.temporal import (
    GroupedInterval,
    Interval,
    boundary_agreement,
    interval_iou,
    match_events,
    segment_f1,
    temporal_map,
)
from dlp_eval.metrics.tracking import TrackingData, clear, hota, identity
from dlp_eval.policy import TASKS, EvaluationPolicy, Task
from dlp_schema.labels import (
    ActionPayload,
    BlurTrackPayload,
    BoxTrackPayload,
    CoveragePayload,
    Hand,
    HandStatePayload,
    KeypointTrackPayload,
    LabelRecord,
    ObjectStatePayload,
    RelationPayload,
)

LOWER_IS_BETTER = frozenset(
    {"ece", "contact_start_error_ms", "contact_end_error_ms", "coverage_abs_error"}
)


@dataclass
class SessionData:
    session_id: str
    truth: list[LabelRecord]
    pred: list[LabelRecord]
    groups: dict[str, str] = field(default_factory=dict[str, str])  # 하위 집단 이름 → 값

    @property
    def glove(self) -> bool:
        return self.groups.get("glove") == "glove"


@dataclass
class TaskReport:
    metrics: dict[str, float]
    class_counts: dict[str, int]
    under_sampled: list[str]
    sessions: int


def _payloads[T](labels: list[LabelRecord], cls: type[T]) -> list[tuple[LabelRecord, T]]:
    return [(x, x.payload) for x in labels if isinstance(x.payload, cls) and not x.retracted]


TRUTH_GAP = math.inf  # 정답 트랙은 키프레임 간격과 상관없이 보간한다 (CVAT 보간과 같다)


def _interp(
    keyframes: Sequence[tuple[int, NDArray[np.float64] | None]], t: int, max_gap: float
) -> NDArray[np.float64] | None:
    """키프레임 (시각, 값 또는 화면 밖 None)에서 시각 t의 값.

    두 키프레임 사이는 선형 보간한다. 어느 한쪽이 화면 밖이거나 간격이 max_gap보다 크면 None이고,
    첫 키프레임 앞·마지막 키프레임 뒤는 None이다.
    """
    times = [k[0] for k in keyframes]
    i = int(np.searchsorted(times, t))
    if i < len(times) and times[i] == t:
        return keyframes[i][1]
    if i == 0 or i == len(times):
        return None
    (t0, v0), (t1, v1) = keyframes[i - 1], keyframes[i]
    if v0 is None or v1 is None or t1 - t0 > max_gap:
        return None
    w = (t - t0) / (t1 - t0)
    return v0 * (1 - w) + v1 * w


def _box_frames(p: BoxTrackPayload) -> list[tuple[int, NDArray[np.float64] | None]]:
    return [
        (k.t_ms, None if k.outside else np.array([k.x, k.y, k.w, k.h], dtype=np.float64))
        for k in sorted(p.keyframes, key=lambda k: k.t_ms)
    ]


def _kp_frames(p: KeypointTrackPayload) -> list[tuple[int, NDArray[np.float64] | None]]:
    return [
        (f.t_ms, np.array([[q.x, q.y, q.visibility] for q in f.points], dtype=np.float64))
        for f in sorted(p.keyframes, key=lambda f: f.t_ms)
    ]


def _report(
    metrics: dict[str, float], counts: Counter[str], sessions: int, minimum: int
) -> TaskReport:
    return TaskReport(metrics, dict(counts), under_sampled(dict(counts), minimum), sessions)


# ---------------------------------------------------------------- 객체 검출·추적


def _as_box(v: NDArray[np.float64]) -> Box:
    return (float(v[0]), float(v[1]), float(v[2]), float(v[3]))


def eval_objects(data: list[SessionData], policy: EvaluationPolicy) -> TaskReport | None:
    gts: list[GtBox] = []
    dets: list[DetBox] = []
    frames: list[tuple[list[tuple[str, str]], list[tuple[str, str]], NDArray[np.float64]]] = []
    conf: list[float] = []
    correct: list[bool] = []
    counts: Counter[str] = Counter()
    gap = policy.max_interp_ms.for_task("objects")
    for s in data:
        truth = [
            (x.stream_id, p.entity_id, p.class_id, _box_frames(p))
            for x, p in _payloads(s.truth, BoxTrackPayload)
        ]
        pred = [
            (
                x.stream_id,
                p.entity_id,
                p.class_id,
                _box_frames(p),
                x.confidence if x.confidence is not None else 1.0,
            )
            for x, p in _payloads(s.pred, BoxTrackPayload)
        ]
        times = sorted({(st, t) for st, _, _, kf in truth for t, v in kf if v is not None})
        for stream, t in times:
            g = [
                (e, c, v)
                for st, e, c, kf in truth
                if st == stream and (v := _interp(kf, t, TRUTH_GAP)) is not None
            ]
            d = [
                (e, c, v, sc)
                for st, e, c, kf, sc in pred
                if st == stream and (v := _interp(kf, t, gap)) is not None
            ]
            image = (s.session_id, stream, t)
            gb = [_as_box(v) for _, _, v in g]
            db = [_as_box(v) for _, _, v, _ in d]
            for (_, c, _), box in zip(g, gb, strict=True):
                gts.append(GtBox(image, c, box))
                counts[c] += 1
            for (_, c, _, sc), box in zip(d, db, strict=True):
                dets.append(DetBox(image, c, box, sc))
            iou = box_iou(gb, db)
            same = np.array(
                [[gc == dc for _, dc, _, _ in d] for _, gc, _ in g], dtype=bool
            ).reshape(iou.shape)
            sim = iou * same
            frames.append(
                (
                    [(s.session_id, f"{stream}/{e}") for e, _, _ in g],
                    [(s.session_id, f"{stream}/{e}") for e, _, _, _ in d],
                    sim,
                )
            )
            used: set[int] = set()
            scores = np.array([x[3] for x in d], dtype=np.float64)
            for j in (int(k) for k in np.argsort(-scores, kind="mergesort")):
                hit: int | None = None
                for i in (int(k) for k in np.argsort(-sim[:, j])):
                    if i not in used and sim[i, j] >= policy.track_iou:
                        hit = i
                        break
                if hit is not None:
                    used.add(hit)
                conf.append(d[j][3])
                correct.append(hit is not None)
    if not gts:
        return None
    ap = average_precision(gts, dets)
    tracking = TrackingData.from_frames(frames)
    metrics = {
        "map": ap.map, "ap50": ap.ap50, "ap75": ap.ap75,
        "hota": hota(tracking).hota,
        "idf1": identity(tracking, policy.track_iou).idf1,
        "mota": clear(tracking, policy.track_iou).mota,
        "ece": ece(conf, correct, policy.ece_bins),
    }  # fmt: skip
    metrics |= {f"ap/{c}": v for c, v in ap.per_class.items()}
    return _report(metrics, counts, len(data), policy.min_samples_per_class)


# ---------------------------------------------------------------- 키포인트


def eval_keypoints(
    data: list[SessionData], policy: EvaluationPolicy, skeleton: str
) -> TaskReport | None:
    """정답 키프레임 시각마다 같은 스트림(손은 같은 쪽 손)의 정답·예측을 일대일로 맞춘다.

    한 시각에 정답이 여럿일 수 있다 (바디캠에 착용자와 돌봄 대상의 왼손, 3인칭에 여러 사람).
    예측 트랙은 시각마다 한 번만 쓴다. 손은 관절 평균 거리 합이 가장 작게, 전신은 보이는 관절
    박스 IoU 합이 가장 크게 (IoU 0인 짝은 맞추지 않는다) 헝가리안 할당으로 맞춘다.
    맞출 예측이 없는 정답은 예측 없음(모든 관절 틀림)으로 센다.
    """
    task: Task = "hands" if skeleton == "hand21" else "body"
    gap = policy.max_interp_ms.for_task(task)
    pairs: list[tuple[NDArray[np.float64], NDArray[np.float64] | None]] = []
    counts: Counter[str] = Counter()
    for s in data:
        truth = [
            (x.stream_id, p)
            for x, p in _payloads(s.truth, KeypointTrackPayload)
            if p.skeleton == skeleton
        ]
        pred = [
            (x.stream_id, p.hand, _kp_frames(p))
            for x, p in _payloads(s.pred, KeypointTrackPayload)
            if p.skeleton == skeleton
        ]
        # (스트림, 손, 시각) → 그 시각의 정답들. 전신은 손 구분 없이 (hand=None)
        groups: dict[tuple[str | None, Hand | None, int], list[NDArray[np.float64]]] = {}
        for stream, p in truth:
            side = p.hand if skeleton == "hand21" else None
            for t, g in _kp_frames(p):
                assert g is not None
                groups.setdefault((stream, side, t), []).append(g)
        for (stream, side, t), gts in sorted(
            groups.items(), key=lambda kv: (kv[0][0] or "", str(kv[0][1]), kv[0][2])
        ):
            cands = [
                v
                for st, hand, kf in pred
                if st == stream
                and (skeleton != "hand21" or hand is side)
                and (v := _interp(kf, t, gap)) is not None
            ]
            match = _match_hands(gts, cands) if skeleton == "hand21" else _match_people(gts, cands)
            for g, j in zip(gts, match, strict=True):
                pairs.append((g, cands[j][:, :2] if j is not None else None))
                counts[(side.value if side else "hand") if skeleton == "hand21" else "person"] += 1
    if not pairs:
        return None
    r = pck(pairs, policy.pck_alpha)
    return _report({"pck": r.pck}, counts, len(data), policy.min_samples_per_class)


def _assign_pairs(cost: NDArray[np.float64], allowed: NDArray[np.bool_]) -> list[int | None]:
    """행(정답)마다 맞춘 열(예측) 번호. 허용되지 않은 짝은 맞추지 않는다."""
    out: list[int | None] = [None] * cost.shape[0]
    if not cost.size or not allowed.any():
        return out
    big = float(np.abs(cost[allowed]).max()) * 2 + 1.0
    rows, cols = assign(np.where(allowed, cost, big))
    for i, j in zip(rows.tolist(), cols.tolist(), strict=True):
        if allowed[i, j]:
            out[i] = j
    return out


def _match_hands(
    gts: list[NDArray[np.float64]], cands: list[NDArray[np.float64]]
) -> list[int | None]:
    """같은 쪽 손이 여럿이면 관절 평균 거리(정답에서 보이는 관절)가 가까운 예측끼리 맞춘다."""
    cost = np.zeros((len(gts), len(cands)))
    allowed = np.zeros((len(gts), len(cands)), dtype=bool)
    for i, g in enumerate(gts):
        visible = g[:, 2] > 0
        if not visible.any():
            continue
        for j, v in enumerate(cands):
            d = np.hypot(v[visible, 0] - g[visible, 0], v[visible, 1] - g[visible, 1])
            cost[i, j] = float(d.mean())
            allowed[i, j] = True
    return _assign_pairs(cost, allowed)


def _match_people(
    gts: list[NDArray[np.float64]], cands: list[NDArray[np.float64]]
) -> list[int | None]:
    """보이는 관절 박스의 IoU 합이 가장 큰 일대일 할당 (IoU 0인 짝은 맞추지 않는다)."""
    iou = box_iou([_kp_box(g) for g in gts], [_kp_box(v) for v in cands])
    return _assign_pairs(-iou, iou > 0)


def _kp_box(points: NDArray[np.float64]) -> Box:
    """보이는(visibility>0) 관절의 박스. 표시하지 않은 관절(보통 (0, 0))은 넣지 않는다."""
    visible = points[points[:, 2] > 0]
    if not len(visible):
        return (0.0, 0.0, 0.0, 0.0)
    xs, ys = visible[:, 0], visible[:, 1]
    return (
        float(xs.min()),
        float(ys.min()),
        float(xs.max() - xs.min()),
        float(ys.max() - ys.min()),
    )


# ---------------------------------------------------------------- 접촉


def _contacts(labels: list[LabelRecord]) -> dict[str, list[tuple[int, int, str]]]:
    out: dict[str, list[tuple[int, int, str]]] = {}
    for x, p in _payloads(labels, HandStatePayload):
        if p.contact_target_kind != "none":
            out.setdefault(p.hand.value, []).append(
                (x.t_start_ms, x.t_end_ms, p.grasp_type or "none")
            )
    return out


def eval_contact(data: list[SessionData], policy: EvaluationPolicy) -> TaskReport | None:
    totals = {"start": [0, 0, 0], "end": [0, 0, 0]}
    errors: dict[str, list[int]] = {"start": [], "end": []}
    grasp_truth: list[str] = []
    grasp_pred: list[str] = []
    counts: Counter[str] = Counter()
    for s in data:
        tol = policy.tolerance_ms.contact_glove if s.glove else policy.tolerance_ms.contact
        truth, pred = _contacts(s.truth), _contacts(s.pred)
        for hand in set(truth) | set(pred):
            t, p = truth.get(hand, []), pred.get(hand, [])
            edges = {
                "start": ([x[0] for x in t], [x[0] for x in p]),
                "end": ([x[1] for x in t], [x[1] for x in p]),
            }
            for edge, (t_times, p_times) in edges.items():
                r = match_events(t_times, p_times, tol)
                totals[edge][0] += r.tp
                totals[edge][1] += r.fp
                totals[edge][2] += r.fn
                errors[edge] += list(r.errors_ms)
            for ts, te, grasp in t:
                counts[grasp] += 1
                best = max(p, key=lambda q: interval_iou((ts, te), (q[0], q[1])), default=None)
                ok = (
                    best is not None
                    and interval_iou((ts, te), (best[0], best[1])) >= policy.match_iou
                )
                grasp_truth.append(grasp)
                grasp_pred.append(best[2] if ok and best else "missing")
    if not grasp_truth and not any(sum(v) for v in totals.values()):
        return None
    metrics: dict[str, float] = {}
    for edge in ("start", "end"):
        tp, fp, fn = totals[edge]
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        metrics[f"contact_{edge}_f1"] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        metrics[f"contact_{edge}_error_ms"] = (
            float(np.mean(errors[edge])) if errors[edge] else float("nan")
        )
    metrics["grasp_macro_f1"] = macro_f1(grasp_truth, grasp_pred)
    return _report(metrics, counts, len(data), policy.min_samples_per_class)


# ---------------------------------------------------------------- 행동 구간


def eval_actions(data: list[SessionData], policy: EvaluationPolicy) -> TaskReport | None:
    seg = {thr: [0, 0, 0] for thr in policy.segment_iou}
    bnd = [0, 0, 0]
    t_grouped: list[GroupedInterval] = []
    p_grouped: list[tuple[str, int, int, str, float]] = []
    verbs_truth: list[str] = []
    verbs_pred: list[str] = []
    counts: Counter[str] = Counter()
    for s in data:
        by_hand: dict[str, tuple[list[Interval], list[tuple[int, int, str, float]]]] = {}
        for _, p in _payloads(s.truth, ActionPayload):
            by_hand.setdefault(p.hand.value, ([], []))[0].append(
                (p.t_approach_ms, p.t_end_ms, p.verb)
            )
        for x, p in _payloads(s.pred, ActionPayload):
            conf = x.confidence if x.confidence is not None else 1.0
            by_hand.setdefault(p.hand.value, ([], []))[1].append(
                (p.t_approach_ms, p.t_end_ms, p.verb, conf)
            )
        for hand, (truth, pred) in by_hand.items():
            group = f"{s.session_id}/{hand}"
            plain = [(a, b, c) for a, b, c, _ in pred]
            for thr in policy.segment_iou:
                r = segment_f1(truth, plain, thr)
                seg[thr][0] += r.tp
                seg[thr][1] += r.fp
                seg[thr][2] += r.fn
            b = boundary_agreement(
                truth, plain, policy.tolerance_ms.boundary, exclude_extremes=True
            )
            bnd[0] += b.tp
            bnd[1] += b.fp
            bnd[2] += b.fn
            t_grouped += [(group, a, e, c) for a, e, c in truth]
            p_grouped += [(group, a, e, c, sc) for a, e, c, sc in pred]
            for a, e, verb in truth:
                counts[verb] += 1
                best = max(plain, key=lambda q: interval_iou((a, e), (q[0], q[1])), default=None)
                ok = (
                    best is not None
                    and interval_iou((a, e), (best[0], best[1])) >= policy.match_iou
                )
                verbs_truth.append(verb)
                verbs_pred.append(best[2] if ok and best else "missing")
    if not t_grouped:
        return None

    def f1(tp: int, fp: int, fn: int) -> float:
        return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0

    metrics = {f"segment_f1_{thr}": f1(*seg[thr]) for thr in policy.segment_iou}
    metrics["temporal_map"] = temporal_map(t_grouped, p_grouped)[0]
    # 양 끝을 빼고 나면 경계가 하나도 없을 수 있다 (구간이 하나뿐) → 정의되지 않음
    metrics["boundary_f1"] = f1(*bnd) if sum(bnd) else math.nan
    metrics["verb_macro_f1"] = macro_f1(verbs_truth, verbs_pred)
    return _report(metrics, counts, len(data), policy.min_samples_per_class)


# ---------------------------------------------------------------- 관계·상태·커버리지


def _relation_key(p: RelationPayload) -> str:
    return "|".join(
        str(v)
        for v in (p.subject_id, p.subject_part, p.predicate.value, p.object_id, p.object_part)
    )


def eval_relations(data: list[SessionData], policy: EvaluationPolicy) -> TaskReport | None:
    tp = fp = fn = 0
    counts: Counter[str] = Counter()
    for s in data:
        truth = [
            (x.t_start_ms, x.t_end_ms, _relation_key(p))
            for x, p in _payloads(s.truth, RelationPayload)
        ]
        pred = [
            (x.t_start_ms, x.t_end_ms, _relation_key(p))
            for x, p in _payloads(s.pred, RelationPayload)
        ]
        counts.update(p.predicate.value for _, p in _payloads(s.truth, RelationPayload))
        r = segment_f1(truth, pred, policy.relation_iou)
        tp, fp, fn = tp + r.tp, fp + r.fp, fn + r.fn
    if not counts:
        return None
    return _report(
        {"relation_f1": 2 * tp / max(1, 2 * tp + fp + fn)},
        counts,
        len(data),
        policy.min_samples_per_class,
    )


def eval_states(data: list[SessionData], policy: EvaluationPolicy) -> TaskReport | None:
    matched = n_truth = n_pred = 0
    counts: Counter[str] = Counter()

    def spans(labels: list[LabelRecord]) -> list[StateSpan]:
        return [
            (p.entity_id, p.attribute, x.t_start_ms, x.t_end_ms, p.value)
            for x, p in _payloads(labels, ObjectStatePayload)
        ]

    for s in data:
        r = transition_accuracy(spans(s.truth), spans(s.pred), policy.tolerance_ms.state)
        matched, n_truth, n_pred = matched + r.matched, n_truth + r.truth, n_pred + r.pred
        counts.update(p.attribute for _, p in _payloads(s.truth, ObjectStatePayload))
    if not counts:
        return None
    metrics = {
        "transition_accuracy": matched / n_truth if n_truth else 1.0,
        "transition_precision": matched / n_pred if n_pred else 1.0,
    }
    return _report(metrics, counts, len(data), policy.min_samples_per_class)


def eval_coverage(data: list[SessionData], policy: EvaluationPolicy) -> TaskReport | None:
    errors: list[float] = []
    counts: Counter[str] = Counter()
    for s in data:
        pred = {(p.surface_id, p.tool_id): p.ratio for _, p in _payloads(s.pred, CoveragePayload)}
        for _, p in _payloads(s.truth, CoveragePayload):
            counts[p.surface_id] += 1
            errors.append(
                abs(pred.get((p.surface_id, p.tool_id), 0.0) - p.ratio)
            )  # 예측이 없으면 0으로 본다
    if not errors:
        return None
    return _report(
        {"coverage_abs_error": float(np.mean(errors))},
        counts,
        len(data),
        policy.min_samples_per_class,
    )


# ---------------------------------------------------------------- 블러 (프라이버시)


def _covered(target: Box, covers: Sequence[Box]) -> float:
    """target 면적 중 covers의 합집합이 덮는 비율 (1px 격자로 센다)."""
    x0, y0 = int(np.floor(target[0])), int(np.floor(target[1]))
    w, h = (
        max(int(np.ceil(target[0] + target[2])) - x0, 1),
        max(int(np.ceil(target[1] + target[3])) - y0, 1),
    )
    mask = np.zeros((h, w), dtype=bool)
    for bx, by, bw, bh in covers:
        c0, r0 = max(int(np.floor(bx)) - x0, 0), max(int(np.floor(by)) - y0, 0)
        c1, r1 = min(int(np.ceil(bx + bw)) - x0, w), min(int(np.ceil(by + bh)) - y0, h)
        if c1 > c0 and r1 > r0:
            mask[r0:r1, c0:c1] = True
    return float(mask.mean())


def eval_privacy(data: list[SessionData], policy: EvaluationPolicy) -> TaskReport | None:
    """스트림별 정답 키프레임 시각마다 비교한다. 정답 트랙은 자기 키프레임 사이를 보간하고 (사람
    키프레임은 성기다), 예측은 같은 스트림만 max_interp_ms(privacy) 안에서 보간한다.

    재현: 정답 박스 면적의 coverage 이상이 예측 블러들로 덮였는가 (대상 종류별로도 낸다).
    정밀: 예측 박스 면적의 precision_overlap 이상이 정답 박스들 위에 있는가.
    """
    hits: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    pred_total = pred_ok = 0
    gap = policy.max_interp_ms.for_task("privacy")
    for s in data:
        truth = [
            (x.stream_id, p.target, _box_frames(_as_track(p)))
            for x, p in _payloads(s.truth, BlurTrackPayload)
        ]
        pred = [
            (x.stream_id, _box_frames(_as_track(p))) for x, p in _payloads(s.pred, BlurTrackPayload)
        ]
        times = sorted({(st, t) for st, _, kf in truth for t, v in kf if v is not None})
        for stream, t in times:
            g = [
                (target, _as_box(v))
                for st, target, kf in truth
                if st == stream and (v := _interp(kf, t, TRUTH_GAP)) is not None
            ]
            d = [
                _as_box(v)
                for st, kf in pred
                if st == stream and (v := _interp(kf, t, gap)) is not None
            ]
            for target, box in g:
                counts[target] += 1
                if _covered(box, d) >= policy.privacy.coverage:
                    hits[target] += 1
            gb = [b for _, b in g]
            for box in d:
                pred_total += 1
                if _covered(box, gb) >= policy.privacy.precision_overlap:
                    pred_ok += 1
    total = sum(counts.values())
    if not total:
        return None
    metrics = {
        "blur_recall": sum(hits.values()) / total,
        "blur_precision": pred_ok / pred_total if pred_total else math.nan,
    }
    metrics |= {f"blur_recall/{c}": hits[c] / n for c, n in sorted(counts.items())}
    return _report(metrics, counts, len(data), policy.min_samples_per_class)


def _as_track(p: BlurTrackPayload) -> BoxTrackPayload:
    return BoxTrackPayload(entity_id="blur", class_id=p.target, keyframes=p.keyframes)


EVALUATORS: dict[Task, Callable[[list[SessionData], EvaluationPolicy], TaskReport | None]] = {
    "objects": eval_objects,
    "hands": lambda d, p: eval_keypoints(d, p, "hand21"),
    "body": lambda d, p: eval_keypoints(d, p, "coco17"),
    "contact": eval_contact,
    "actions": eval_actions,
    "relations": eval_relations,
    "states": eval_states,
    "coverage": eval_coverage,
    "privacy": eval_privacy,
}


@dataclass
class EvalReport:
    golden_version: str
    model_versions: dict[str, str]  # 과제 → 모델 버전
    overall: dict[str, TaskReport]
    subgroups: dict[str, dict[str, TaskReport]]  # "glove=bare" → 과제 → 리포트


def evaluate(
    data_by_task: dict[Task, list[SessionData]],
    policy: EvaluationPolicy,
    *,
    golden_version: str,
    model_versions: dict[str, str],
) -> EvalReport:
    overall: dict[str, TaskReport] = {}
    subgroups: dict[str, dict[str, TaskReport]] = {}
    for task in TASKS:
        data = data_by_task.get(task)
        if not data:
            continue
        report = EVALUATORS[task](data, policy)
        if report is None:
            continue
        overall[task] = report
        for name in policy.subgroups:
            for value in sorted({s.groups.get(name, "unknown") for s in data}):
                subset = [s for s in data if s.groups.get(name, "unknown") == value]
                sub = EVALUATORS[task](subset, policy)
                if sub is not None:
                    subgroups.setdefault(f"{name}={value}", {})[task] = sub
    return EvalReport(golden_version, model_versions, overall, subgroups)
