"""라벨 레코드 → 과제별 지표. 골든셋 세션들의 정답(사람)과 예측(모델 버전)을 비교한다.

과제와 라벨 종류:
- objects: box_track → mAP(0.50:0.95), AP50, HOTA, IDF1, MOTA, ECE.
  정답 키프레임 시각에서 비교하고 예측은 키프레임 사이를 보간한다 (max_interp_ms 이내).
- hands / body: keypoint_track(hand21 / coco17) → PCK. 손은 왼손·오른손끼리, 전신은 시각마다
  키포인트 박스 IoU로 사람을 맞춘다.
- contact: hand_state(접촉 대상 있음) → 접촉 시작·종료 F1과 오차(ms), 파지 유형 macro F1.
  장갑 세션은 contact_glove 허용 오차를 쓴다.
- actions: action → 구간 F1@IoU, temporal mAP, 경계 일치 F1, 동사 macro F1.
- relations: relation → 같은 (주어, 술어, 목적어, 부분) 구간 F1@relation_iou.
- states: object_state → 상태 전이 정확도.
- coverage: coverage → 같은 (표면, 도구) 쌍의 비율 절대 오차 평균.
세션 사이의 개체 ID는 세션 ID를 붙여 구분한다 (추적 지표는 여러 시퀀스를 이어 붙인 것과 같다).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

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
    BoxTrackPayload,
    CoveragePayload,
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


def _interp(
    keyframes: Sequence[tuple[int, NDArray[np.float64] | None]], t: int, max_gap: int
) -> NDArray[np.float64] | None:
    """키프레임 (시각, 값 또는 화면 밖 None)에서 시각 t의 값."""
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
    for s in data:
        truth = [
            (p.entity_id, p.class_id, _box_frames(p))
            for _, p in _payloads(s.truth, BoxTrackPayload)
        ]
        pred = [
            (
                p.entity_id,
                p.class_id,
                _box_frames(p),
                x.confidence if x.confidence is not None else 1.0,
            )
            for x, p in _payloads(s.pred, BoxTrackPayload)
        ]
        times = sorted({t for _, _, kf in truth for t, v in kf if v is not None})
        for t in times:
            g = [(e, c, v) for e, c, kf in truth if (v := _interp(kf, t, 0)) is not None]
            d = [
                (e, c, v, sc)
                for e, c, kf, sc in pred
                if (v := _interp(kf, t, policy.max_interp_ms)) is not None
            ]
            gb = [_as_box(v) for _, _, v in g]
            db = [_as_box(v) for _, _, v, _ in d]
            for (_, c, _), box in zip(g, gb, strict=True):
                gts.append(GtBox((s.session_id, t), c, box))
                counts[c] += 1
            for (_, c, _, sc), box in zip(d, db, strict=True):
                dets.append(DetBox((s.session_id, t), c, box, sc))
            iou = box_iou(gb, db)
            same = np.array(
                [[gc == dc for _, dc, _, _ in d] for _, gc, _ in g], dtype=bool
            ).reshape(iou.shape)
            sim = iou * same
            frames.append(
                ([(s.session_id, e) for e, _, _ in g], [(s.session_id, e) for e, _, _, _ in d], sim)
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
    pairs: list[tuple[NDArray[np.float64], NDArray[np.float64] | None]] = []
    counts: Counter[str] = Counter()
    for s in data:
        truth = [p for _, p in _payloads(s.truth, KeypointTrackPayload) if p.skeleton == skeleton]
        pred = [p for _, p in _payloads(s.pred, KeypointTrackPayload) if p.skeleton == skeleton]
        if skeleton == "hand21":
            for t_track in truth:
                match = next((p for p in pred if p.hand is t_track.hand), None)
                kf = _kp_frames(match) if match else []
                for t, gt in _kp_frames(t_track):
                    assert gt is not None
                    v = _interp(kf, t, policy.max_interp_ms) if kf else None
                    pairs.append((gt, v[:, :2] if v is not None else None))
                    counts[t_track.hand.value if t_track.hand else "hand"] += 1
            continue
        pred_frames = [_kp_frames(p) for p in pred]
        times = sorted({f.t_ms for p in truth for f in p.keyframes})
        for t in times:
            gt_now = [g for p in truth for tt, g in _kp_frames(p) if tt == t and g is not None]
            pr_now = [
                v for kf in pred_frames if (v := _interp(kf, t, policy.max_interp_ms)) is not None
            ]
            iou = box_iou([_kp_box(g) for g in gt_now], [_kp_box(v) for v in pr_now])
            used: set[int] = set()
            for i, g in enumerate(gt_now):
                j = (
                    next(
                        (
                            int(j)
                            for j in np.argsort(-iou[i])
                            if int(j) not in used and iou[i, j] > 0
                        ),
                        None,
                    )
                    if len(pr_now)
                    else None
                )
                if j is not None:
                    used.add(j)
                pairs.append((g, pr_now[j][:, :2] if j is not None else None))
                counts["person"] += 1
    if not pairs:
        return None
    r = pck(pairs, policy.pck_alpha)
    return _report({"pck": r.pck}, counts, len(data), policy.min_samples_per_class)


def _kp_box(points: NDArray[np.float64]) -> Box:
    xs, ys = points[:, 0], points[:, 1]
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
            b = boundary_agreement(truth, plain, policy.tolerance_ms.boundary)
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
    metrics["boundary_f1"] = f1(*bnd)
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


EVALUATORS: dict[Task, Callable[[list[SessionData], EvaluationPolicy], TaskReport | None]] = {
    "objects": eval_objects,
    "hands": lambda d, p: eval_keypoints(d, p, "hand21"),
    "body": lambda d, p: eval_keypoints(d, p, "coco17"),
    "contact": eval_contact,
    "actions": eval_actions,
    "relations": eval_relations,
    "states": eval_states,
    "coverage": eval_coverage,
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
