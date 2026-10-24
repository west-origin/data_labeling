"""정답 없는 품질 대리 지표 (proxy metrics, `dlp ops quality-proxy`).

골든셋이 없는 실영상 세션에서 프리라벨 품질의 경향을 잰다. 정확도가 아니라 "이상해 보이는 정도"다:
짧은 트랙·깜박임(끊김)·프레임당 탐지 수, 모델 사이 불일치(MediaPipe 손목 vs RTMPose 손목, 얼굴 블러
vs 사람·머리 위치), 손 좌우와 화면 위치, 손 3D의 물리 위반(팔 길이 밖 거리, 속도·가속도, 손목→손끝
거리 흔들림)과 3D 맞춤 방식·신뢰도, 착용자 팔(팔꿈치 보임·팔 모델 손목 일치·아래팔 길이 흔들림)과
착용자처럼 보이는 사람 박스(ADR 0058). 같은 기준(`evaluation.yaml proxy`)으로 잰 두 결과를 나란히
두어 필터·모델 변경 전후를 비교한다.

입력은 한 세션 한 스트림의 운영 라벨(`current_labels`)이다. DB에 쓰지 않는다. 결과는 JSON으로
바꿀 수 있는 dict다 (키 이름은 보고서·비교 스크립트가 쓰므로 바꾸지 않는다).

좌표·시각: 2D는 스트림 픽셀, 키프레임 시각은 스트림 PTS ms다 (ADR 0019). 3D 궤적은 카메라 좌표(m)만
본다 (월드 좌표는 카메라 자세 오차가 섞인다).
"""

from __future__ import annotations

import bisect
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from itertools import pairwise
from typing import Any

import numpy as np

from dlp_eval.policy import ProxyPolicy
from dlp_schema.labels import (
    BlurTrackPayload,
    BoxKeyframe,
    BoxTrackPayload,
    CoordinateFrame,
    Hand,
    KeypointFrame,
    KeypointTrackPayload,
    LabelRecord,
    Trajectory3DPayload,
    Trajectory3DSample,
)
from dlp_schema.wearer_body import WearerEvidence, wearer_box_frame

# COCO 17점 관절 번호 (RTMPose): 머리 다섯 점(코·눈·귀)과 손목
COCO_HEAD = (0, 1, 2, 3, 4)
COCO_WRIST = {Hand.LEFT: 9, Hand.RIGHT: 10}
# MediaPipe hand21 관절 번호: 손목, 중지 뿌리 (손 크기 = 둘 사이 픽셀 거리)
HAND_WRIST, HAND_MIDDLE_MCP = 0, 9
# 손 3D 궤적 부위 (lift3d가 내는 손목·다섯 손끝)
TIPS = ("thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip")

Report = dict[str, Any]


def dist(values: Iterable[float]) -> dict[str, float | int | None]:
    """값 목록의 요약 (n, 평균, 10·50·90·95 백분위, 최댓값). 빈 목록이면 n=0과 None."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {
            "n": 0,
            "mean": None,
            "p10": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "max": None,
        }
    q = np.percentile(arr, [10, 50, 90, 95])
    return {
        "n": int(arr.size),
        "mean": _r(float(arr.mean())),
        "p10": _r(float(q[0])),
        "p50": _r(float(q[1])),
        "p90": _r(float(q[2])),
        "p95": _r(float(q[3])),
        "max": _r(float(arr.max())),
    }


def _r(v: float) -> float:
    """보고용 반올림 (유효 숫자 4자리 정도)."""
    return round(v, 4)


def _frac(num: int, den: int) -> float | None:
    """비율. 분모가 0이면 None."""
    return _r(num / den) if den else None


def family(label: LabelRecord) -> str:
    """모델 계열 이름 (model_version 첫 토큰, 예: mediapipe·owlv2·rtmpose·iou). 사람 라벨은 출처."""
    mv = label.provenance.model_version
    return mv.split("-", 1)[0].split("+", 1)[0] if mv else label.provenance.source.value


def group_key(label: LabelRecord) -> str | None:
    """트랙 묶음 이름. 박스·블러·키포인트 트랙만 묶고 나머지(3D·구간 등)는 None.

    box:<계열>:<클래스>, blur:<대상>, kp:<계열>:<골격>[:<손>].
    """
    p = label.payload
    if isinstance(p, BoxTrackPayload):
        return f"box:{family(label)}:{p.class_id}"
    if isinstance(p, BlurTrackPayload):
        return f"blur:{p.target}"
    if isinstance(p, KeypointTrackPayload):
        hand = f":{p.hand.value}" if p.hand is not None else ""
        return f"kp:{family(label)}:{p.skeleton}{hand}"
    return None


def keyframe_times(label: LabelRecord) -> list[int]:
    """트랙의 키프레임 시각 (박스·블러는 outside가 아닌 것만). 트랙이 아니면 빈 목록."""
    p = label.payload
    if isinstance(p, BoxTrackPayload | BlurTrackPayload):
        return [k.t_ms for k in p.keyframes if not k.outside]
    if isinstance(p, KeypointTrackPayload):
        return [k.t_ms for k in p.keyframes]
    return []


def _median_step(times: Sequence[int]) -> float | None:
    """정렬된 시각들의 양의 간격 중앙값. 간격이 없으면 None."""
    steps = [b - a for a, b in pairwise(times) if b > a]
    return float(np.median(steps)) if steps else None


def runs(times: Sequence[int], gap_factor: float) -> list[tuple[int, int]]:
    """정렬된 고유 시각을 연속 구간 [(시작, 끝)]으로 나눈다.

    간격이 중앙 간격 x gap_factor보다 크면 끊김이다 (탐지 간격이 다른 모델도 같은 기준). 시각이
    하나면 구간 하나.
    """
    if not times:
        return []
    step = _median_step(times)
    limit = step * gap_factor if step is not None else math.inf
    out: list[tuple[int, int]] = []
    start = prev = times[0]
    for t in times[1:]:
        if t - prev > limit:
            out.append((start, prev))
            start = t
        prev = t
    out.append((start, prev))
    return out


def _groups(labels: Iterable[LabelRecord]) -> dict[str, list[LabelRecord]]:
    """묶음 이름 → 트랙 목록 (`group_key`가 None인 라벨은 뺀다)."""
    out: dict[str, list[LabelRecord]] = defaultdict(list)
    for x in labels:
        key = group_key(x)
        if key is not None and keyframe_times(x):
            out[key].append(x)
    return dict(sorted(out.items()))


def track_stats(tracks: Sequence[LabelRecord], policy: ProxyPolicy, minutes: float) -> Report:
    """한 묶음의 트랙 통계: 수, 분당 수, 길이(ms)·키프레임 수 분포, 짧은·키프레임 하나 트랙 비율."""
    durations = [max(keyframe_times(x)) - min(keyframe_times(x)) for x in tracks]
    counts = [len(keyframe_times(x)) for x in tracks]
    confidences = [x.confidence for x in tracks if x.confidence is not None]
    return {
        "tracks": len(tracks),
        "tracks_per_min": _r(len(tracks) / minutes) if minutes > 0 else None,
        "duration_ms": dist(durations),
        "keyframes": dist(counts),
        "short_frac": _frac(sum(d < policy.short_track_ms for d in durations), len(tracks)),
        "single_keyframe_frac": _frac(sum(c == 1 for c in counts), len(tracks)),
        "confidence": dist(confidences),
    }


def presence_stats(tracks: Sequence[LabelRecord], policy: ProxyPolicy, minutes: float) -> Report:
    """한 묶음의 시각별 존재: 프레임당 탐지 수, 존재 구간의 끊김(깜박임)과 짧은 구간 비율.

    프레임 = 그 묶음 키프레임 시각 하나. 깜박임은 존재 구간 수(끊김 수 + 1)를 분당으로 센다.
    """
    per_t: Counter[int] = Counter()
    for x in tracks:
        per_t.update(set(keyframe_times(x)))
    times = sorted(per_t)
    segs = runs(times, policy.gap_factor)
    lengths = [b - a for a, b in segs]
    return {
        "active_frames": len(times),
        "per_frame": dist(per_t.values()),
        "runs": len(segs),
        "runs_per_min": _r(len(segs) / minutes) if minutes > 0 else None,
        "short_run_frac": _frac(sum(n < policy.short_track_ms for n in lengths), len(segs)),
        "median_step_ms": _median_step(times),
    }


def _hand_frames(labels: Iterable[LabelRecord]) -> dict[Hand, dict[int, KeypointFrame]]:
    """손 → {시각: hand21 키프레임}. 같은 손 트랙이 여럿이면 먼저 본 키프레임을 쓴다."""
    out: dict[Hand, dict[int, KeypointFrame]] = {Hand.LEFT: {}, Hand.RIGHT: {}}
    for x in labels:
        p = x.payload
        if isinstance(p, KeypointTrackPayload) and p.skeleton == "hand21" and p.hand is not None:
            for k in p.keyframes:
                out[p.hand].setdefault(k.t_ms, k)
    return out


def _hand_scale(frame: KeypointFrame) -> float | None:
    """손 크기 px (손목→중지 뿌리). 점이 모자라거나 0이면 None."""
    if len(frame.points) <= HAND_MIDDLE_MCP:
        return None
    a, b = frame.points[HAND_WRIST], frame.points[HAND_MIDDLE_MCP]
    d = math.hypot(a.x - b.x, a.y - b.y)
    return d if d > 0 else None


def wrist_jitter(frames: dict[int, KeypointFrame], max_dt_ms: int) -> list[float]:
    """손목 2D 2차 차분(가속도 성분)을 손 크기로 나눈 값들. 이웃 간격이 max_dt_ms 넘으면 뺀다.

    정지·등속 운동이면 0에 가깝고, 프레임마다 튀는 떨림이 클수록 커진다 (평활 전후 비교용).
    """
    times = sorted(frames)
    out: list[float] = []
    for a, b, c in zip(times, times[1:], times[2:], strict=False):
        if b - a > max_dt_ms or c - b > max_dt_ms:
            continue
        scale = _hand_scale(frames[b])
        if scale is None:
            continue
        pa, pb, pc = (frames[t].points[HAND_WRIST] for t in (a, b, c))
        out.append(math.hypot(pa.x - 2 * pb.x + pc.x, pa.y - 2 * pb.y + pc.y) / scale)
    return out


def hands_2d(labels: Sequence[LabelRecord], policy: ProxyPolicy, duration_ms: int | None) -> Report:
    """손 2D: 손마다 존재 비율 추정·떨림, 두 손 프레임에서 왼손이 화면 왼쪽인 비율 (1인칭 기대 1).

    존재 비율 추정 = 손이 있는 프레임 수 / (세션 길이 / 손 키프레임 중앙 간격). 영상 프레임 수를
    라벨만으로 알 수 없어 추정이다. 보간해 채운 키프레임(`interpolated`, ADR 0054)은
    `filled_frames`로 따로 세고, `*_observed`는 관측 키프레임만 본 값이다.
    """
    frames = _hand_frames(labels)
    out: Report = {}
    for hand, by_t in frames.items():
        times = sorted(by_t)
        step = _median_step(times)
        expected = duration_ms / step if duration_ms and step else None
        out[hand.value] = {
            "frames": len(times),
            "filled_frames": sum(bool(f.interpolated) for f in by_t.values()),
            "present_frac_est": _r(min(1.0, len(times) / expected)) if expected else None,
            "runs": len(runs(times, policy.gap_factor)),
            "wrist_jitter": dist(wrist_jitter(by_t, policy.max_dt_ms)),
        }
    both = sorted(set(frames[Hand.LEFT]) & set(frames[Hand.RIGHT]))
    left_on_left = sum(
        frames[Hand.LEFT][t].points[HAND_WRIST].x < frames[Hand.RIGHT][t].points[HAND_WRIST].x
        for t in both
    )
    observed = [
        t
        for t in both
        if not frames[Hand.LEFT][t].interpolated and not frames[Hand.RIGHT][t].interpolated
    ]
    left_obs = sum(
        frames[Hand.LEFT][t].points[HAND_WRIST].x < frames[Hand.RIGHT][t].points[HAND_WRIST].x
        for t in observed
    )
    out["two_hand_frames"] = len(both)
    out["left_on_screen_left_frac"] = _frac(left_on_left, len(both))
    out["two_hand_frames_observed"] = len(observed)
    out["left_on_screen_left_frac_observed"] = _frac(left_obs, len(observed))
    return out


class _TimeIndex[T]:
    """시각 → 항목 목록. 허용 오차 안의 항목을 찾는다."""

    def __init__(self) -> None:
        """빈 색인."""
        self.items: dict[int, list[T]] = defaultdict(list)
        self.keys: list[int] = []

    def add(self, t: int, item: T) -> None:
        """항목을 넣는다 (`near` 전에 `freeze`)."""
        self.items[t].append(item)

    def freeze(self) -> _TimeIndex[T]:
        """찾기용 정렬 키를 만든다."""
        self.keys = sorted(self.items)
        return self

    def near(self, t: int, tol: int) -> list[T]:
        """|시각 - t| <= tol인 항목 전부."""
        lo = bisect.bisect_left(self.keys, t - tol)
        hi = bisect.bisect_right(self.keys, t + tol)
        return [item for k in self.keys[lo:hi] for item in self.items[k]]


def _coco_frames(labels: Iterable[LabelRecord]) -> _TimeIndex[KeypointFrame]:
    """COCO 17점(RTMPose 등) 키프레임 색인."""
    index: _TimeIndex[KeypointFrame] = _TimeIndex()
    for x in labels:
        p = x.payload
        if isinstance(p, KeypointTrackPayload) and p.skeleton == "coco17":
            for k in p.keyframes:
                index.add(k.t_ms, k)
    return index.freeze()


def _nearest_wrist(
    frame: KeypointFrame, others: Sequence[KeypointFrame]
) -> tuple[float, Hand] | None:
    """MediaPipe 손목과 가장 가까운 보이는 COCO 손목 (거리 / 손 크기, 그 손목의 좌우)."""
    scale = _hand_scale(frame)
    if scale is None:
        return None
    w = frame.points[HAND_WRIST]
    best: tuple[float, Hand] | None = None
    for other in others:
        for hand, idx in COCO_WRIST.items():
            if idx >= len(other.points) or other.points[idx].visibility != 2:
                continue
            q = other.points[idx]
            d = math.hypot(w.x - q.x, w.y - q.y) / scale
            if best is None or d < best[0]:
                best = (d, hand)
    return best


def wrist_agreement(labels: Sequence[LabelRecord], policy: ProxyPolicy) -> Report:
    """MediaPipe 손목 vs RTMPose(COCO 17점) 손목 일치.

    같은 시각(허용 오차 안)에 보이는 COCO 손목이 있는 손 프레임만 센다. agree_frac: 가장 가까운
    COCO 손목이 손 크기 x wrist_agree_scale 안인 비율. side_match_frac: 일치한 쌍 중 좌우가 같은
    비율 (1인칭에서 RTMPose가 착용자 자신의 팔을 사람으로 볼 때 좌우가 틀릴 수 있어 참고값).
    """
    coco = _coco_frames(labels)
    pairs = agree = side = 0
    for hand, by_t in _hand_frames(labels).items():
        for t, frame in by_t.items():
            found = _nearest_wrist(frame, coco.near(t, policy.time_tolerance_ms))
            if found is None:
                continue
            pairs += 1
            if found[0] <= policy.wrist_agree_scale:
                agree += 1
                side += found[1] is hand
    return {
        "pairs": pairs,
        "agree_frac": _frac(agree, pairs),
        "side_match_frac": _frac(side, agree),
    }


def _box_contains(box: BoxKeyframe, x: float, y: float, margin: float) -> bool:
    """점이 각 변을 margin 비율만큼 넓힌 박스 안인가."""
    mx, my = box.w * margin, box.h * margin
    return box.x - mx <= x <= box.x + box.w + mx and box.y - my <= y <= box.y + box.h + my


def _people(labels: Iterable[LabelRecord]) -> _TimeIndex[BoxKeyframe]:
    """사람 박스 색인: 객체 박스 트랙의 person과 COCO 17점의 보이는 점 외접 박스."""
    index: _TimeIndex[BoxKeyframe] = _TimeIndex()
    for x in labels:
        p = x.payload
        if isinstance(p, BoxTrackPayload) and p.class_id == "person":
            for k in p.keyframes:
                if not k.outside:
                    index.add(k.t_ms, k)
        elif isinstance(p, KeypointTrackPayload) and p.skeleton == "coco17":
            for k in p.keyframes:
                pts = [q for q in k.points if q.visibility == 2]
                if pts:
                    x0, y0 = min(q.x for q in pts), min(q.y for q in pts)
                    w, h = max(q.x for q in pts) - x0, max(q.y for q in pts) - y0
                    index.add(k.t_ms, BoxKeyframe(t_ms=k.t_ms, x=x0, y=y0, w=w, h=h))
    return index.freeze()


def face_support(labels: Sequence[LabelRecord], policy: ProxyPolicy) -> Report:
    """얼굴 블러 키프레임 중 사람 박스 근처·머리 관절 근처인 비율.

    1인칭 영상에서 사람 박스·머리 관절과 떨어진 얼굴 박스는 오탐일 가능성이 높다 (싱크·손에 붙은
    얼굴). 블러 판정에는 쓰지 않는 측정값이다 (재현 우선, 프라이버시 탐지는 지우지 않는다).
    """
    people = _people(labels)
    heads = _coco_frames(labels)
    faces = near_person = near_head = 0
    for x in labels:
        p = x.payload
        if not (isinstance(p, BlurTrackPayload) and p.target == "face"):
            continue
        for k in p.keyframes:
            if k.outside:
                continue
            faces += 1
            cx, cy = k.x + k.w / 2, k.y + k.h / 2
            tol = policy.time_tolerance_ms
            if any(_box_contains(b, cx, cy, policy.face_margin) for b in people.near(k.t_ms, tol)):
                near_person += 1
            if any(
                i < len(f.points)
                and f.points[i].visibility == 2
                and _box_contains(k, f.points[i].x, f.points[i].y, policy.face_margin)
                for f in heads.near(k.t_ms, tol)
                for i in COCO_HEAD
            ):
                near_head += 1
    return {
        "face_keyframes": faces,
        "near_person_frac": _frac(near_person, faces),
        "near_head_frac": _frac(near_head, faces),
    }


def _trajectories(labels: Iterable[LabelRecord]) -> dict[tuple[str, str], list[Trajectory3DSample]]:
    """(손 개체, 부위) → 카메라 좌표 3D 표본(시각순). 손 개체(`*_hand`)만."""
    out: dict[tuple[str, str], list[Trajectory3DSample]] = defaultdict(list)
    for x in labels:
        p = x.payload
        if (
            isinstance(p, Trajectory3DPayload)
            and p.frame is CoordinateFrame.CAMERA
            and p.entity_id.endswith("_hand")
            and p.part is not None
        ):
            out[(p.entity_id, p.part)].extend(p.samples)
    return {k: sorted(v, key=lambda s: s.t_ms) for k, v in out.items()}


def _norm(s: Trajectory3DSample) -> float:
    """카메라 원점에서의 거리 (m)."""
    return math.sqrt(s.x * s.x + s.y * s.y + s.z * s.z)


def _vel(a: Trajectory3DSample, b: Trajectory3DSample) -> tuple[float, float, float]:
    """a→b 속도 벡터 (m/s)."""
    dt = (b.t_ms - a.t_ms) / 1000
    return ((b.x - a.x) / dt, (b.y - a.y) / dt, (b.z - a.z) / dt)


def motion(
    samples: Sequence[Trajectory3DSample], max_dt_ms: int
) -> tuple[list[float], list[float]]:
    """이웃 표본(간격 0 < dt <= max_dt_ms)으로 낸 속도(m/s)와 가속도(m/s²) 크기 목록."""
    speeds: list[float] = []
    accels: list[float] = []
    prev: tuple[tuple[float, float, float], int] | None = None
    for a, b in pairwise(samples):
        dt = b.t_ms - a.t_ms
        if not 0 < dt <= max_dt_ms:
            prev = None
            continue
        v = _vel(a, b)
        speeds.append(math.sqrt(sum(c * c for c in v)))
        mid = (a.t_ms + b.t_ms) / 2
        if prev is not None:
            dtv = (mid - prev[1]) / 1000
            accels.append(
                math.sqrt(sum((c - d) ** 2 for c, d in zip(v, prev[0], strict=True))) / dtv
            )
        prev = (v, round(mid))
    return speeds, accels


def _wrist_physics(wrist: Sequence[Trajectory3DSample], policy: ProxyPolicy) -> Report:
    """손목 하나의 거리 범위·속도·가속도 위반과 방식·신뢰도."""
    dists = [_norm(s) for s in wrist]
    low, high = policy.reach_m
    speeds, accels = motion(wrist, policy.max_dt_ms)
    methods = Counter(s.method.value if s.method else "none" for s in wrist)
    return {
        "samples": len(wrist),
        "distance_m": dist(dists),
        "out_of_reach_frac": _frac(sum(not low <= d <= high for d in dists), len(dists)),
        "speed_mps": dist(speeds),
        "over_speed_frac": _frac(sum(v > policy.max_speed_mps for v in speeds), len(speeds)),
        "over_accel_frac": _frac(sum(a > policy.max_accel_mps2 for a in accels), len(accels)),
        "methods": dict(sorted(methods.items())),
        "confidence": dist(s.confidence for s in wrist if s.confidence is not None),
    }


def _bones(
    wrist: Sequence[Trajectory3DSample],
    tips: dict[str, list[Trajectory3DSample]],
    policy: ProxyPolicy,
) -> Report:
    """손목→손끝 거리: 손끝마다 변동계수(표준편차/평균), 타당 범위 밖 비율, 손끝 표본 비율."""
    at = {s.t_ms: s for s in wrist}
    low, high = policy.wrist_tip_m
    cvs: dict[str, float | None] = {}
    total = bad = 0
    tip_times: set[int] = set()
    for name in TIPS:
        lengths: list[float] = []
        for s in tips.get(name, []):
            w = at.get(s.t_ms)
            if w is None:
                continue
            tip_times.add(s.t_ms)
            lengths.append(math.dist((s.x, s.y, s.z), (w.x, w.y, w.z)))
        total += len(lengths)
        bad += sum(not low <= d <= high for d in lengths)
        mean = float(np.mean(lengths)) if lengths else 0.0
        cvs[name] = _r(float(np.std(lengths)) / mean) if len(lengths) > 1 and mean > 0 else None
    return {
        "tip_coverage_frac": _frac(len(tip_times), len(wrist)),
        "length_cv": cvs,
        "length_out_of_range_frac": _frac(bad, total),
    }


def hand_3d(labels: Sequence[LabelRecord], policy: ProxyPolicy) -> Report:
    """손 3D(카메라 좌표) 물리 위반: 손마다 손목 거리·속도·가속도·손목→손끝 거리."""
    tracks = _trajectories(labels)
    out: Report = {}
    for entity in sorted({e for e, _ in tracks}):
        wrist = tracks.get((entity, "wrist"), [])
        tips = {part: s for (e, part), s in tracks.items() if e == entity and part in TIPS}
        out[entity] = {**_wrist_physics(wrist, policy), **_bones(wrist, tips, policy)}
    return out


def _arm_frames(labels: Iterable[LabelRecord]) -> dict[Hand, list[KeypointFrame]]:
    """손 → arm3 키프레임 (착용자 팔, ADR 0058)."""
    out: dict[Hand, list[KeypointFrame]] = defaultdict(list)
    for x in labels:
        p = x.payload
        if isinstance(p, KeypointTrackPayload) and p.skeleton == "arm3" and p.hand is not None:
            out[p.hand].extend(p.keyframes)
    return out


def _forearm(labels: Sequence[LabelRecord], hand: Hand, policy: ProxyPolicy) -> Report:
    """손목(손 3D)→팔꿈치(팔 3D) 거리: 분포, 변동계수, 범위 밖 비율 (카메라 좌표)."""
    wrist = {s.t_ms: s for s in _trajectories(labels).get((f"{hand.value}_hand", "wrist"), [])}
    lengths = [
        math.dist((s.x, s.y, s.z), (w.x, w.y, w.z))
        for x in labels
        if isinstance(p := x.payload, Trajectory3DPayload)
        and p.frame is CoordinateFrame.CAMERA
        and p.entity_id == f"{hand.value}_arm"
        and p.part == "elbow"
        for s in p.samples
        if (w := wrist.get(s.t_ms)) is not None
    ]
    low, high = policy.forearm_m
    mean = float(np.mean(lengths)) if lengths else 0.0
    return {
        "length_m": dist(lengths),
        "length_cv": _r(float(np.std(lengths)) / mean) if len(lengths) > 1 and mean > 0 else None,
        "out_of_range_frac": _frac(sum(not low <= d <= high for d in lengths), len(lengths)),
    }


def wearer_arm(labels: Sequence[LabelRecord], policy: ProxyPolicy) -> Report:
    """착용자 팔 (ADR 0058): 손마다 팔꿈치 보임 비율, 팔 모델 손목 일치, 아래팔 3D 길이 흔들림.

    elbow_visible_frac: arm3 키프레임 중 팔꿈치가 보임(2)인 비율. chain_frac: 팔 모델 사슬이 hand21
    손목과 맞은 키프레임 비율 (손목 점 신뢰도가 있음). wrist_agreement: 그 일치 정도(0~1) 분포.
    forearm: 손목→팔꿈치 3D 거리 (`_forearm`).
    """
    out: Report = {}
    for hand, frames in sorted(_arm_frames(labels).items(), key=lambda kv: kv[0].value):
        elbows = sum(f.points[1].visibility == 2 for f in frames)
        agreement = [f.points[0].confidence for f in frames if f.points[0].confidence is not None]
        out[hand.value] = {
            "keyframes": len(frames),
            "elbow_visible_frac": _frac(elbows, len(frames)),
            "chain_frac": _frac(len(agreement), len(frames)),
            "wrist_agreement": dist(agreement),
            "forearm": _forearm(labels, hand, policy),
        }
    return out


def wearer_like_people(labels: Sequence[LabelRecord], policy: ProxyPolicy) -> Report:
    """사람 박스 트랙 수와 그중 착용자처럼 보이는 트랙 수 (`dlp_schema.wearer_body`, ADR 0058).

    키프레임 중 박스 안에 착용자 손목이 있고 다른 사람 머리 관절이 없는 비율이 wearer_share
    이상이면 착용자처럼 보인다 (프레임 높이를 몰라 아래 가장자리 조건은 보지 않는다). 프리라벨
    억제 뒤 남은 착용자 박스의 대리 지표다.
    """
    evidence = WearerEvidence.from_labels(labels, labels[0].stream_id or "") if labels else None
    tracks = like = 0
    tol, m = policy.time_tolerance_ms, policy.wearer_margin
    for x in labels:
        p = x.payload
        if not (isinstance(p, BoxTrackPayload) and p.class_id == "person") or evidence is None:
            continue
        frames = [k for k in p.keyframes if not k.outside]
        if not frames:
            continue
        tracks += 1
        hits = sum(
            wearer_box_frame(
                (k.x, k.y, k.w, k.h), evidence.hands_near(k.t_ms, tol),
                evidence.heads_near(k.t_ms, tol), margin=m, frame_h=None, bottom_band=1.0,
            )
            for k in frames
        )  # fmt: skip
        like += hits / len(frames) >= policy.wearer_share
    return {"person_tracks": tracks, "wearer_like_tracks": like}


def _span_ms(labels: Sequence[LabelRecord]) -> int:
    """라벨이 덮는 시간 길이 (세션 길이를 모를 때)."""
    if not labels:
        return 0
    return max(x.t_end_ms for x in labels) - min(x.t_start_ms for x in labels)


def proxy_report(
    labels: Sequence[LabelRecord],
    policy: ProxyPolicy,
    *,
    stream_id: str | None = None,
    duration_ms: int | None = None,
) -> Report:
    """세션 한 스트림의 대리 지표 전체.

    Args:
        labels: 세션의 운영 라벨 (`current_labels`). 다른 스트림 라벨은 stream_id로 거른다.
        policy: `evaluation.yaml proxy`.
        stream_id: 볼 스트림 (None이면 거르지 않는다).
        duration_ms: 세션(스트림) 길이. None이면 라벨이 덮는 길이.

    Returns:
        {"duration_ms", "groups": {묶음: {track, presence}}, "hands_2d", "agreement": {wrist, face},
        "hand_3d", "wearer": {arm, people}} (wearer는 ADR 0058).
    """
    own = [x for x in labels if stream_id is None or x.stream_id == stream_id]
    span = duration_ms if duration_ms else _span_ms(own)
    minutes = span / 60000
    groups = {
        key: {
            "track": track_stats(tracks, policy, minutes),
            "presence": presence_stats(tracks, policy, minutes),
        }
        for key, tracks in _groups(own).items()
    }
    return {
        "duration_ms": span,
        "groups": groups,
        "hands_2d": hands_2d(own, policy, span),
        "agreement": {"wrist": wrist_agreement(own, policy), "face": face_support(own, policy)},
        "hand_3d": hand_3d(own, policy),
        "wearer": {"arm": wearer_arm(own, policy), "people": wearer_like_people(own, policy)},
    }


def _fmt(v: object) -> str:
    """표 칸 문자열 (None은 '-')."""
    return "-" if v is None else str(v)


# 요약 표 열: (제목, 묶음 결과 안 경로)
_TABLE = (
    ("트랙", ("track", "tracks")),
    ("분당 트랙", ("track", "tracks_per_min")),
    ("길이 p50 ms", ("track", "duration_ms", "p50")),
    ("짧은 트랙", ("track", "short_frac")),
    ("프레임당 평균", ("presence", "per_frame", "mean")),
    ("분당 구간", ("presence", "runs_per_min")),
    ("짧은 구간", ("presence", "short_run_frac")),
)


def _get(data: Report, path: Sequence[str]) -> object:
    """중첩 dict에서 경로의 값."""
    value: Any = data
    for key in path:
        value = value[key]
    return value


def _summary_lines(report: Report) -> list[str]:
    """손·일치·3D 요약 줄."""
    h2, agr = report["hands_2d"], report["agreement"]
    wrist, face = agr["wrist"], agr["face"]
    lines = [
        f"- 손: 두 손 프레임 {h2['two_hand_frames']}, "
        f"왼손이 화면 왼쪽 {_fmt(h2['left_on_screen_left_frac'])}",
        f"- 손목 일치(MediaPipe vs RTMPose): 쌍 {wrist['pairs']}, 일치 {_fmt(wrist['agree_frac'])}",
        f"- 얼굴 블러: 키프레임 {face['face_keyframes']}, 사람 근처 "
        f"{_fmt(face['near_person_frac'])}, 머리 관절 근처 {_fmt(face['near_head_frac'])}",
    ]
    for hand in (Hand.LEFT.value, Hand.RIGHT.value):
        if hand in h2:
            lines.append(
                f"- {hand} 손 2D: 존재 추정 {_fmt(h2[hand]['present_frac_est'])}, "
                f"떨림 p50 {_fmt(h2[hand]['wrist_jitter']['p50'])}"
            )
    for entity, h in report["hand_3d"].items():
        lines.append(
            f"- {entity} 3D: 손목 표본 {h['samples']}, 거리 p50 {_fmt(h['distance_m']['p50'])} m, "
            f"범위 밖 {_fmt(h['out_of_reach_frac'])}, 속도 초과 {_fmt(h['over_speed_frac'])}, "
            f"가속도 초과 {_fmt(h['over_accel_frac'])}, "
            f"손끝 거리 범위 밖 {_fmt(h['length_out_of_range_frac'])}"
        )
    wearer = report.get("wearer", {})
    for hand, a in wearer.get("arm", {}).items():
        fa = a["forearm"]
        lines.append(
            f"- {hand} 팔: 키프레임 {a['keyframes']}, 팔꿈치 보임 {_fmt(a['elbow_visible_frac'])}, "
            f"사슬 손목 일치 {_fmt(a['chain_frac'])} "
            f"(정도 p50 {_fmt(a['wrist_agreement']['p50'])}), "
            f"아래팔 p50 {_fmt(fa['length_m']['p50'])} m, 변동계수 {_fmt(fa['length_cv'])}, "
            f"범위 밖 {_fmt(fa['out_of_range_frac'])}"
        )
    if "people" in wearer:
        pp = wearer["people"]
        lines.append(
            f"- 사람 박스 트랙 {pp['person_tracks']}, 착용자처럼 보임 {pp['wearer_like_tracks']}"
        )
    return lines


def markdown(report: Report, title: str) -> str:
    """사람용 요약 표 (Markdown). 전체 값은 JSON에 있다."""
    lines = [f"# {title}", "", f"길이 {report['duration_ms']} ms", ""]
    lines.append("| 묶음 | " + " | ".join(name for name, _ in _TABLE) + " |")
    lines.append("| --- |" + " --- |" * len(_TABLE))
    for key, g in report["groups"].items():
        cells = " | ".join(_fmt(_get(g, path)) for _, path in _TABLE)
        lines.append(f"| {key} | {cells} |")
    return "\n".join([*lines, "", *_summary_lines(report)]) + "\n"
