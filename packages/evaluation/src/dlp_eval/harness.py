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

WP11, ADR 0013·0019·0025. 입력 `SessionData`는 `dlp_eval.runner`(DB) 또는 재학습 루프
(`dlp_train.loop.predict_golden`, 메모리 예측)가 만든다. 이 모듈은 DB·저장소에 접근하지 않는다.

공개 이름:
- `SessionData` — 세션 하나의 정답·예측·하위 집단 값.
- `TaskReport`, `EvalReport` — 과제별 지표와 전체·하위 집단 리포트.
- `eval_objects`·`eval_keypoints`·`eval_contact`·`eval_actions`·`eval_relations`·`eval_states`·
  `eval_coverage`·`eval_privacy` — 과제별 평가기. 평가할 정답이 없으면 None.
- `EVALUATORS` — 과제 → 평가기. `evaluate` — 모든 과제와 하위 집단을 평가한다.
- `LOWER_IS_BETTER` — 낮을수록 좋은 지표 이름 (게이트·리포트가 쓴다).

공통 규칙: 삭제(retracted) 레코드는 정답·예측 어디에도 쓰지 않는다 (`_payloads`). 정답·예측 필터
(사람 정답, 모델 버전, 비운영 레코드 제외)는 호출자가 이미 했다고 가정한다.
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

# 낮을수록 좋은 지표. 게이트(`gate._gain`)가 방향을 뒤집고, 리포트 JSON에 `lower_is_better`로
# 남긴다. 새 오차형 지표를 추가하면 여기에도 넣어야 게이트가 올바른 방향으로 판정한다.
LOWER_IS_BETTER = frozenset(
    {"ece", "contact_start_error_ms", "contact_end_error_ms", "coverage_abs_error"}
)


@dataclass
class SessionData:
    """세션 하나의 평가 입력 (과제 하나 분량)."""

    session_id: str
    truth: list[LabelRecord]  # 정답 라벨 (사람이 만들거나 승인·수정한 운영 라벨)
    pred: list[LabelRecord]  # 평가할 모델 버전의 예측 라벨 (원래 모델 출력)
    groups: dict[str, str] = field(default_factory=dict[str, str])  # 하위 집단 이름 → 값

    @property
    def glove(self) -> bool:
        """장갑 스트림이 있는 세션인가 (`groups["glove"] == "glove"`). 접촉 허용 오차를 고른다."""
        return self.groups.get("glove") == "glove"


@dataclass
class TaskReport:
    """과제 하나의 평가 결과 (전체 또는 하위 집단 하나)."""

    metrics: dict[str, float]  # 지표 이름 → 값. 정의되지 않은 값은 NaN (JSON에는 null)
    class_counts: dict[str, int]  # 클래스 → 정답 표본 수 (과제마다 세는 단위가 다르다)
    under_sampled: list[str]  # min_samples_per_class 미만 클래스
    sessions: int  # 평가에 넣은 세션 수 (정답이 없는 세션 포함)


def _payloads[T](labels: list[LabelRecord], cls: type[T]) -> list[tuple[LabelRecord, T]]:
    """`cls` 종류 페이로드를 가진 삭제되지 않은 레코드와 그 페이로드 쌍."""
    return [(x, x.payload) for x in labels if isinstance(x.payload, cls) and not x.retracted]


TRUTH_GAP = math.inf  # 정답 트랙은 키프레임 간격과 상관없이 보간한다 (CVAT 보간과 같다)


def _interp(
    keyframes: Sequence[tuple[int, NDArray[np.float64] | None]], t: int, max_gap: float
) -> NDArray[np.float64] | None:
    """키프레임 (시각, 값 또는 화면 밖 None)에서 시각 t의 값.

    두 키프레임 사이는 선형 보간한다. 어느 한쪽이 화면 밖이거나 간격이 max_gap보다 크면 None이고,
    첫 키프레임 앞·마지막 키프레임 뒤는 None이다.

    Args:
        keyframes: 시각 오름차순 (스트림 PTS ms, 값). 값은 박스 [x, y, w, h] 또는 관절 (K, 3) 배열.
        t: 구할 시각 (ms, 같은 스트림 PTS).
        max_gap: 보간할 최대 키프레임 간격 (ms). `TRUTH_GAP`(무한)이면 제한 없음.

    Returns:
        t가 키프레임 시각과 같으면 그 값(화면 밖이면 None), 아니면 보간 값 또는 None.
        키포인트의 visibility 열도 선형 보간된다 (0과 2 사이면 1 같은 중간값 — `> 0` 판정만 쓴다).
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
    """박스 트랙 → 시각순 (t_ms, [x, y, w, h] 또는 화면 밖이면 None)."""
    return [
        (k.t_ms, None if k.outside else np.array([k.x, k.y, k.w, k.h], dtype=np.float64))
        for k in sorted(p.keyframes, key=lambda k: k.t_ms)
    ]


def _kp_frames(p: KeypointTrackPayload) -> list[tuple[int, NDArray[np.float64] | None]]:
    """키포인트 트랙 → 시각순 (t_ms, (K, 3) [x, y, visibility]). 화면 밖 개념이 없어 None은 없다."""
    return [
        (f.t_ms, np.array([[q.x, q.y, q.visibility] for q in f.points], dtype=np.float64))
        for f in sorted(p.keyframes, key=lambda f: f.t_ms)
    ]


def _report(
    metrics: dict[str, float], counts: Counter[str], sessions: int, minimum: int
) -> TaskReport:
    """지표·클래스 수로 `TaskReport`를 만든다 (표본 부족 클래스 계산 포함)."""
    return TaskReport(metrics, dict(counts), under_sampled(dict(counts), minimum), sessions)


# ---------------------------------------------------------------- 객체 검출·추적


def _as_box(v: NDArray[np.float64]) -> Box:
    """[x, y, w, h] 배열 → `Box` 튜플."""
    return (float(v[0]), float(v[1]), float(v[2]), float(v[3]))


def eval_objects(data: list[SessionData], policy: EvaluationPolicy) -> TaskReport | None:
    """객체 박스 트랙(box_track) 평가: COCO mAP·AP50·AP75, HOTA, IDF1, MOTA, ECE, 클래스별 AP.

    (세션, 스트림)마다 정답 키프레임 시각(모든 정답 트랙의 화면 안 키프레임 시각 합집합)을
    "이미지"로 삼는다. 그 시각의 정답 박스는 정답 트랙 보간(간격 제한 없음), 예측 박스는 같은
    스트림 예측 트랙을 `max_interp_ms.for_task("objects")` 안에서 보간한 값이다. 예측만 있는 시각은
    비교하지 않는다 (ADR 0013 한계: 정답 키프레임 사이의 오탐은 세지 못한다).

    - 검출(AP): 이미지 키 (세션, 스트림, 시각), `average_precision` 기본 max_dets 100.
    - 추적(HOTA·IDF1·MOTA): 유사도 = IoU x 같은 클래스 여부. 개체 키에 세션·스트림을 붙여 한
      시퀀스로 이어 붙인다. IDF1·MOTA 문턱은 `track_iou`.
    - ECE: 이미지마다 예측을 점수순으로, 같은 클래스·IoU >= track_iou인 아직 안 쓴 정답 중 유사도가
      가장 큰 것과 탐욕 매칭해 맞음/틀림을 정한다. 신뢰도 없는 예측은 1.0으로 본다.
    - 클래스 수(class_counts): 정답 시각별 박스 수 (트랙 수가 아니다).

    Returns:
        정답 박스가 하나도 없으면 None.
    """
    gts: list[GtBox] = []
    dets: list[DetBox] = []
    frames: list[tuple[list[tuple[str, str]], list[tuple[str, str]], NDArray[np.float64]]] = []
    conf: list[float] = []
    correct: list[bool] = []
    counts: Counter[str] = Counter()
    gap = policy.max_interp_ms.for_task("objects")
    for s in data:
        # (스트림, 개체, 클래스, 키프레임)
        truth = [
            (x.stream_id, p.entity_id, p.class_id, _box_frames(p))
            for x, p in _payloads(s.truth, BoxTrackPayload)
        ]
        # (스트림, 개체, 클래스, 키프레임, 신뢰도). 신뢰도가 없으면 1.0
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
        # 비교 시각: 정답 트랙의 화면 안 키프레임 시각 (스트림별)
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
            # 추적 유사도: 클래스가 다르면 0 (다른 클래스끼리는 같은 개체로 보지 않는다)
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
            # ECE용 맞음/틀림: 점수 내림차순(안정 정렬) 탐욕 매칭
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

    Args:
        skeleton: "hand21"(과제 hands) 또는 "coco17"(과제 body). 다른 골격 라벨은 무시한다.

    Returns:
        PCK 하나("pck")를 담은 리포트. 클래스 수는 손이면 "left"/"right", 전신이면 "person" 단위의
        정답 (시각, 개체) 수다. 정답이 없으면 None.

    주의: 정답 트랙은 보간하지 않고 자기 키프레임 시각에서만 비교한다 (객체·블러와 다르다). 예측은
    `max_interp_ms.for_task(hands|body)` 안에서 보간한다. 시각 간 ID 일관성은 보지 않는다 (ADR
    0013).
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
        # 결정적 순서로 돈다 (결과는 순서와 무관하지만 디버깅 편의)
        for (stream, side, t), gts in sorted(
            groups.items(), key=lambda kv: (kv[0][0] or "", str(kv[0][1]), kv[0][2])
        ):
            # 후보: 같은 스트림(손은 같은 쪽)이고 그 시각에 보간 가능한 예측 트랙
            cands = [
                v
                for st, hand, kf in pred
                if st == stream
                and (skeleton != "hand21" or hand is side)
                and (v := _interp(kf, t, gap)) is not None
            ]
            match = _match_hands(gts, cands) if skeleton == "hand21" else _match_people(gts, cands)
            for g, j in zip(gts, match, strict=True):
                # 예측은 좌표 두 열만 PCK에 넘긴다 (visibility 열 제외)
                pairs.append((g, cands[j][:, :2] if j is not None else None))
                counts[(side.value if side else "hand") if skeleton == "hand21" else "person"] += 1
    if not pairs:
        return None
    r = pck(pairs, policy.pck_alpha)
    return _report({"pck": r.pck}, counts, len(data), policy.min_samples_per_class)


def _assign_pairs(cost: NDArray[np.float64], allowed: NDArray[np.bool_]) -> list[int | None]:
    """행(정답)마다 맞춘 열(예측) 번호. 허용되지 않은 짝은 맞추지 않는다.

    허용되지 않은 칸에는 허용 칸 비용 절댓값 최대의 2배 + 1을 넣어 헝가리안 할당이 피하게 하고,
    할당 뒤에도 허용되지 않은 짝은 None으로 버린다. 허용 칸이 하나도 없으면 모두 None.
    """
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
    """같은 쪽 손이 여럿이면 관절 평균 거리(정답에서 보이는 관절)가 가까운 예측끼리 맞춘다.

    거리 문턱은 없다: 후보가 있으면 아무리 멀어도 맞춘다 (틀린 관절은 PCK가 센다). 보이는 관절이
    없는 정답은 맞추지 않는다 (PCK에서도 세지 않는다).
    """
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
    """보이는(visibility>0) 관절의 박스. 표시하지 않은 관절(보통 (0, 0))은 넣지 않는다.

    보이는 관절이 없으면 넓이 0 박스 (0, 0, 0, 0) → 어떤 박스와도 IoU 0이라 맞추지 않는다.
    """
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
    """접촉 구간을 손별로 모은다: 손("left"/"right") → [(시작 ms, 끝 ms, 파지 유형)].

    `contact_target_kind != "none"`인 hand_state만 접촉이다 (학습 예제
    `dlp_train.extract.matches`와 같은
    기준). 파지 유형이 없으면 "none". 시각은 레코드의 마스터 타임라인 t_start_ms·t_end_ms.
    """
    out: dict[str, list[tuple[int, int, str]]] = {}
    for x, p in _payloads(labels, HandStatePayload):
        if p.contact_target_kind != "none":
            out.setdefault(p.hand.value, []).append(
                (x.t_start_ms, x.t_end_ms, p.grasp_type or "none")
            )
    return out


def eval_contact(data: list[SessionData], policy: EvaluationPolicy) -> TaskReport | None:
    """접촉 평가: 시작·종료 시점 F1과 평균 오차(ms), 파지 유형 macro F1.

    세션·손마다 접촉 시작 시각끼리, 종료 시각끼리 `match_events`로 맞춘다. 허용 오차는 장갑 세션이면
    `tolerance_ms.contact_glove`, 아니면 `tolerance_ms.contact`. TP·FP·FN은 모든 세션·손을 더해
    F1을 낸다 (미시 평균). 오차는 맞춘 쌍 전체의 평균이며, 맞춘 쌍이 없으면 NaN.

    파지 유형: 정답 접촉마다 같은 손의 예측 중 IoU가 가장 큰 구간이 `match_iou` 이상이면 그 예측의
    파지 유형, 아니면 "missing"을 예측으로 놓고 macro F1을 낸다 (예측 쪽 오탐은 파지 F1에 들지
    않는다). 클래스 수는 정답 파지 유형별 접촉 수.

    Returns:
        정답 접촉도 없고 맞출 사건(TP·FP·FN)도 없으면 None. 예측만 있으면 F1 0인 리포트가 나온다.
    """
    # 가장자리별 [TP, FP, FN] 누적
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
    """행동 구간 평가: 구간 F1@IoU(문턱마다), temporal mAP, 경계 F1, 동사 macro F1.

    구간은 (t_approach_ms, t_end_ms, 동사)다 (접근 단계부터 끝까지, 마스터 타임라인 ms). 세션·손마다
    묶어 비교하고, TP·FP·FN은 모든 묶음을 더해 F1을 낸다 (미시 평균).
    - `segment_f1_<문턱>`: `segment_f1`, 문턱은 `segment_iou` 목록.
    - `temporal_map`: 묶음 = "세션/손", 점수 = 예측 신뢰도(없으면 1.0), tIoU 0.50:0.95.
    - `boundary_f1`: `boundary_agreement(exclude_extremes=True)`, 허용 오차 `tolerance_ms.boundary`.
      양 끝을 뺀 뒤 경계가 하나도 없으면 NaN.
    - `verb_macro_f1`: 정답 구간마다 IoU 최대 예측(`match_iou` 이상)의 동사, 아니면 "missing".

    Returns:
        정답 행동이 하나도 없으면 None.
    """
    seg = {thr: [0, 0, 0] for thr in policy.segment_iou}
    bnd = [0, 0, 0]
    t_grouped: list[GroupedInterval] = []
    p_grouped: list[tuple[str, int, int, str, float]] = []
    verbs_truth: list[str] = []
    verbs_pred: list[str] = []
    counts: Counter[str] = Counter()
    for s in data:
        # 손 → (정답 구간, 예측 구간+신뢰도)
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
        """2TP / (2TP + FP + FN). 모두 0이면 0.0."""
        return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0

    metrics = {f"segment_f1_{thr}": f1(*seg[thr]) for thr in policy.segment_iou}
    metrics["temporal_map"] = temporal_map(t_grouped, p_grouped)[0]
    # 양 끝을 빼고 나면 경계가 하나도 없을 수 있다 (구간이 하나뿐) → 정의되지 않음
    metrics["boundary_f1"] = f1(*bnd) if sum(bnd) else math.nan
    metrics["verb_macro_f1"] = macro_f1(verbs_truth, verbs_pred)
    return _report(metrics, counts, len(data), policy.min_samples_per_class)


# ---------------------------------------------------------------- 관계·상태·커버리지


def _relation_key(p: RelationPayload) -> str:
    """관계를 비교할 키: "주어|주어 부분|술어|목적어|목적어 부분" (부분이 없으면 "None")."""
    return "|".join(
        str(v)
        for v in (p.subject_id, p.subject_part, p.predicate.value, p.object_id, p.object_part)
    )


def eval_relations(data: list[SessionData], policy: EvaluationPolicy) -> TaskReport | None:
    """관계 평가: 같은 관계 키의 구간끼리 `segment_f1`(문턱 `relation_iou`)으로 맞춘 F1.

    세션마다 TP·FP·FN을 더한다. 시각은 레코드의 마스터 타임라인 t_start_ms·t_end_ms. 클래스 수는
    정답 술어별 관계 수.

    Returns:
        정답 관계가 없으면 None (예측만 있는 세션의 오탐은 다른 세션에 정답이 있을 때만 FP로
        들어간다).
    """
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
    """상태 평가: 상태 전이 정확도(재현)·정밀도 (`transition_accuracy`, 허용 오차
    `tolerance_ms.state`).

    세션마다 전이를 맞추고 맞춘 수·정답 수·예측 수를 더한다. 분모가 0이면 1.0. 클래스 수는 정답
    속성별 상태 구간 수.

    Returns:
        정답 상태 라벨이 없으면 None.
    """
    matched = n_truth = n_pred = 0
    counts: Counter[str] = Counter()

    def spans(labels: list[LabelRecord]) -> list[StateSpan]:
        """object_state 레코드 → (개체, 속성, 시작 ms, 끝 ms, 값)."""
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
    """표면 커버리지 평가: 정답 (표면, 도구) 쌍마다 |예측 비율 - 정답 비율|의 평균.

    예측이 없는 쌍은 비율 0으로 본다. 정답에 없는 예측 쌍은 세지 않는다 (오탐 벌점 없음).
    같은 세션에 같은 (표면, 도구) 예측이 여럿이면 마지막 것이 쓰인다 (dict). 클래스 수는 정답
    표면별 수.

    Returns:
        정답 커버리지 라벨이 없으면 None.
    """
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
    """target 면적 중 covers의 합집합이 덮는 비율 (1px 격자로 센다).

    target을 바깥쪽으로 정수 픽셀에 맞춘 격자(최소 1x1)를 만들고, 각 cover도 바깥쪽으로 반올림해
    칠한 뒤 칠해진 칸 비율을 돌려준다. 겹치는 cover는 한 번만 센다. 0~1.
    격자 크기는 target 박스 크기(px²)에 비례한다.
    """
    x0, y0 = int(np.floor(target[0])), int(np.floor(target[1]))
    w, h = (
        max(int(np.ceil(target[0] + target[2])) - x0, 1),
        max(int(np.ceil(target[1] + target[3])) - y0, 1),
    )
    mask = np.zeros((h, w), dtype=bool)
    for bx, by, bw, bh in covers:
        # cover 박스를 target 격자 좌표로 옮기고 격자 밖은 자른다
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

    지표: `blur_recall`(주 지표, 놓친 대상은 노출), `blur_precision`(정답 시각의 예측 박스가 없으면
    NaN), `blur_recall/<대상>`. 블러 대상 종류(target)는 재현을 나눌 때만 쓰고, 덮는지 판정할 때는
    종류를 보지 않는다 (얼굴을 문서 블러가 덮어도 덮인 것이다). 정답 시각에만 비교하므로 정답이
    없는 시각의 과잉 블러는 정밀에 들지 않는다. 클래스 수는 정답 대상 종류별 (시각, 박스) 수.

    Returns:
        정답 블러 박스가 하나도 없으면 None.
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
    """블러 트랙을 박스 트랙 모양으로 바꾼다 (키프레임 처리 `_box_frames`를 같이 쓰려고)."""
    return BoxTrackPayload(entity_id="blur", class_id=p.target, keyframes=p.keyframes)


# 과제 → 평가기. 손·전신은 같은 평가기를 골격만 바꿔 쓴다
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
    """골든셋 평가 리포트 (한 모델 조합)."""

    golden_version: str
    model_versions: dict[str, str]  # 과제 → 모델 버전
    overall: dict[str, TaskReport]  # 과제 → 전체 리포트 (평가할 정답이 있는 과제만)
    subgroups: dict[str, dict[str, TaskReport]]  # "glove=bare" → 과제 → 리포트


def evaluate(
    data_by_task: dict[Task, list[SessionData]],
    policy: EvaluationPolicy,
    *,
    golden_version: str,
    model_versions: dict[str, str],
) -> EvalReport:
    """과제마다 전체와 하위 집단 리포트를 만든다.

    Args:
        data_by_task: 과제 → 세션별 정답·예측. 비어 있거나 없는 과제는 건너뛴다.
        policy: 평가 정책.
        golden_version: 리포트에 남길 골든셋 버전.
        model_versions: 리포트에 남길 과제 → 모델 버전 (평가에는 쓰지 않는다).

    Returns:
        `EvalReport`. 하위 집단은 `policy.subgroups`의 축마다 세션의 값(없으면 "unknown")으로
        세션을 나눠 같은 평가기를 다시 돌린 결과다 ("glove=glove", "site=site0" 등). 정답이 없는
        하위 집단은 빠진다.
    """
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
