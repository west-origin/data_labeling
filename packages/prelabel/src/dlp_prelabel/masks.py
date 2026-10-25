"""객체·손 모양 마스크 단계 (ADR 0059).

바디캠의 현재 박스 트랙(객체·도구, 트랙 거르기 ADR 0054·장면 질의 ADR 0055·착용자 억제 ADR 0058을
지난 것)과 hand21 손 트랙을 프롬프트로 SAM 2.1 tiny(이미지 모드, `dlp_models.sam2`)를
`masks.frame_stride_ms` 간격 프레임마다 돌려 mask_track 라벨을 만든다. 영상 전파는 쓰지 않는다.

- 박스 프롬프트: 그 프레임 시각의 박스 키프레임, 없으면 앞뒤 키프레임(둘 다 보이고 간격이
  `box_max_gap_ms` 이하)을 선형 보간한 박스. 착용자 몸 판정(`wearer_suppress.box_frame_is_wearer`,
  `wearer_suppression` 값)에 걸린 박스 키프레임과, 아래 변이 화면 아래 띠에 닿는 사람 박스
  (`person_bottom_band`, 손이 안 보이는 시각의 착용자 몸통)는 쓰지 않는다 (착용자 팔·몸통은 객체
  마스크가 되지 않는다, ADR 0058).
- 손 프롬프트: 그 시각(허용 오차 안)의 관측 키프레임(보간 키프레임은 관측이 아니라 뺀다, ADR 0054)
  에서 보이는 관절 외접 박스를 넓힌 것과 손 안쪽 양성 점(`hand_point_joints`).
- 검사: 예측 IoU < min_score, 넓이 / 박스 넓이 < min_area_ratio, 넓힌 박스 밖 화소 비율 >
  max_outside_ratio(새어 나감)이면 그 키프레임을 버리고 사유를 센다. 객체 마스크에서는 같은
  프레임 손 마스크를 뺀다 (`subtract_hands`). 마스크 외접 박스와 프롬프트 박스의 IoU를 요약에
  남긴다.
- 출력: 프롬프트 트랙마다 mask_track 하나 (entity·class는 프롬프트 트랙, 손은 `hand_class`),
  `derived_from` = 프롬프트 트랙 라벨 ID, 키프레임마다 COCO RLE와 예측 IoU(`score`), 라벨 신뢰도는
  키프레임 점수 평균. 라벨 ID `<세션>-<스트림>-masks-<version_tag>-<개체>`.

파이프라인 위치: `runner._masks`가 입력 트랙을 골라(`pick_mask_tracks`) 부른다. 프라이버시 블러와
무관하다 (블러는 박스 그대로이고 마스크로 바꾸지 않는다). 시간: 키프레임 시각은 바디캠 PTS ms다.

CPU stub: `BoxEllipseSegmenter` (프롬프트 박스에 내접한 타원, CI·합성 시험용).
"""

from __future__ import annotations

import bisect
import hashlib
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from dlp_media.rle import encode_rle
from dlp_models.registry import resolve
from dlp_models.sam2 import Box, MaskResult, Prompt
from dlp_prelabel.common import Image, iter_frames, model_label, strictly_increasing
from dlp_prelabel.policy import MasksPolicy, PrelabelPolicy, WearerSuppressionPolicy
from dlp_prelabel.wearer_suppress import box_frame_is_wearer
from dlp_schema.episode import version_tag
from dlp_schema.labels import (
    BoxKeyframe,
    BoxTrackPayload,
    KeypointFrame,
    KeypointTrackPayload,
    LabelRecord,
    MaskKeyframe,
    MaskTrackPayload,
)
from dlp_schema.labels import Source as LabelSource
from dlp_schema.wearer_body import WearerEvidence

# 마스크 알고리즘 판 (프롬프트·검사 규칙을 바꾸면 올린다)
MASK_REVISION = "mk1"  # 규칙 판 (정책 값은 masks 절 해시로 버전에 들어간다)
# 모델 역할 (prelabel.yaml models): 그래프와 외부 가중치 파일
MASK_ROLES = ("mask_encoder", "mask_encoder_data", "mask_decoder", "mask_decoder_data")


class Segmenter(Protocol):
    """마스크 모델 인터페이스 (`dlp_models.sam2.Sam2Segmenter`, stub `BoxEllipseSegmenter`)."""

    def predict(self, image: Image, prompts: Sequence[Prompt]) -> list[MaskResult | None]:
        """(H, W, 3) RGB 프레임 한 장 → 프롬프트마다 마스크 (대상 없음이면 None)."""
        ...


class BoxEllipseSegmenter:
    """CPU stub: 프롬프트 박스에 내접한 타원 마스크 (점수 `score`). 영상 내용은 보지 않는다."""

    def __init__(self, score: float = 0.9) -> None:
        """score: 모든 마스크의 예측 IoU 점수."""
        self.score = score

    def predict(self, image: Image, prompts: Sequence[Prompt]) -> list[MaskResult | None]:
        """프롬프트 박스마다 내접 타원."""
        h, w = image.shape[:2]
        out: list[MaskResult | None] = []
        for p in prompts:
            x, y, bw, bh = p.box
            m = np.zeros((h, w), np.uint8)
            center = (round(x + bw / 2), round(y + bh / 2))
            cv2.ellipse(m, center, (max(1, round(bw / 2)), max(1, round(bh / 2))), 0, 0, 360, 1, -1)
            out.append(MaskResult(m.astype(bool), self.score))
        return out


@dataclass(frozen=True)
class PromptTrack:
    """프롬프트 트랙 하나 (박스 트랙 또는 hand21 손 트랙)."""

    record: LabelRecord
    entity_id: str
    class_id: str
    is_hand: bool
    times: tuple[int, ...]  # 키프레임 시각 (오름차순)

    @property
    def rank(self) -> float:
        """한 프레임 프롬프트 상한을 넘을 때 앞세울 순서 (손 먼저, 그다음 트랙 신뢰도)."""
        return 2.0 if self.is_hand else (self.record.confidence or 1.0)


@dataclass
class MaskReport:
    """마스크 단계 요약 (CLI·파일럿 보고용, 라벨 아님)."""

    frames: int = 0  # 모델을 돌린 프레임 수
    prompts: int = 0
    masks: int = 0  # 남긴 마스크 키프레임 수
    hand_masks: int = 0
    tracks: int = 0  # 낸 mask_track 수
    wearer_skipped: int = 0  # 착용자 몸 판정으로 쓰지 않은 박스 프롬프트
    dropped: Counter[str] = field(default_factory=Counter[str])  # 사유 → 버린 키프레임 수
    # 마스크 외접 박스 ↔ 프롬프트 박스 IoU (객체, 손 따로: 손 프롬프트는 넓힌 관절 박스다)
    box_iou: list[float] = field(default_factory=list[float])
    hand_box_iou: list[float] = field(default_factory=list[float])
    seconds: float = 0.0  # 모델 시간 합 (초)

    def text(self) -> str:
        """한 줄 요약 (CLI 출력)."""
        per = self.seconds / self.frames if self.frames else 0.0
        drops = ", ".join(f"{k} {v}" for k, v in sorted(self.dropped.items())) or "없음"
        return (
            f"프레임 {self.frames}(프레임당 {per:.2f}초), 프롬프트 {self.prompts}, "
            f"마스크 {self.masks}(손 {self.hand_masks}), 트랙 {self.tracks}, 버림 {drops}, "
            f"착용자 박스 건너뜀 {self.wearer_skipped}, 박스 IoU p10/50/90 객체 "
            f"{_quantiles(self.box_iou)} 손 {_quantiles(self.hand_box_iou)}"
        )


def _quantiles(values: Sequence[float]) -> list[float]:
    """10·50·90 백분위 (소수 둘째 자리, 비면 빈 목록)."""
    if not values:
        return []
    q: NDArray[np.float64] = np.percentile(np.asarray(values, np.float64), [10, 50, 90])
    return [round(float(v), 2) for v in q.tolist()]


def mask_source(x: LabelRecord, policy: MasksPolicy) -> PromptTrack | None:
    """현재 라벨 하나 → 프롬프트 트랙 (쓰지 않는 라벨이면 None).

    박스 트랙: 제외 클래스가 아니고, 모델 출처면 신뢰도가 min_track_confidence 이상.
    hand21 손 트랙: 정책 hands가 켜져 있을 때 (entity는 손 트랙 그대로, class는 hand_class).
    """
    p = x.payload
    if isinstance(p, BoxTrackPayload):
        if p.class_id in policy.exclude_classes:
            return None
        model = x.provenance.source is LabelSource.MODEL
        if model and (x.confidence or 0.0) < policy.min_track_confidence:
            return None
        times = tuple(k.t_ms for k in p.keyframes)
        return PromptTrack(x, p.entity_id, p.class_id, False, times)
    if isinstance(p, KeypointTrackPayload) and p.skeleton == "hand21" and policy.hands:
        times = tuple(f.t_ms for f in p.keyframes)
        return PromptTrack(x, p.entity_id, policy.hand_class, True, times)
    return None


def _lerp(a: BoxKeyframe, b: BoxKeyframe, t: int) -> Box:
    """두 박스 키프레임 사이 시각 t의 선형 보간 박스."""
    r = (t - a.t_ms) / (b.t_ms - a.t_ms)
    return (
        a.x + (b.x - a.x) * r,
        a.y + (b.y - a.y) * r,
        a.w + (b.w - a.w) * r,
        a.h + (b.h - a.h) * r,
    )


def box_at(p: BoxTrackPayload, times: Sequence[int], t: int, max_gap_ms: int) -> Box | None:
    """박스 트랙의 시각 t 박스 (키프레임 그대로 또는 앞뒤 보간). 안 보이거나 간격이 길면 None."""
    i = bisect.bisect_left(times, t)
    if i < len(times) and times[i] == t:
        k = p.keyframes[i]
        return None if k.outside else (k.x, k.y, k.w, k.h)
    if i == 0 or i == len(times):
        return None
    a, b = p.keyframes[i - 1], p.keyframes[i]
    if a.outside or b.outside or b.t_ms - a.t_ms > max_gap_ms:
        return None
    return _lerp(a, b, t)


def _hand_frame(p: KeypointTrackPayload, times: Sequence[int], t: int, tol: int) -> KeypointFrame:
    """시각 t에 가장 가까운 손 키프레임 (호출자가 times가 비지 않음을 보장)."""
    i = bisect.bisect_left(times, t)
    near = [j for j in (i - 1, i) if 0 <= j < len(times) and abs(times[j] - t) <= tol]
    return p.keyframes[min(near, key=lambda j: abs(times[j] - t))] if near else p.keyframes[-1]


def hand_prompt_at(
    p: KeypointTrackPayload, times: Sequence[int], t: int, policy: MasksPolicy
) -> Prompt | None:
    """손 트랙의 시각 t 프롬프트 (보이는 관절 외접 박스를 넓힌 것 + 손 안쪽 양성 점)."""
    if not times:
        return None
    f = _hand_frame(p, times, t, policy.hand_time_tolerance_ms)
    if abs(f.t_ms - t) > policy.hand_time_tolerance_ms or f.interpolated:
        return None
    seen = [q for q in f.points if q.visibility == 2]
    if len(seen) < policy.hand_min_points:
        return None
    xs, ys = [q.x for q in seen], [q.y for q in seen]
    pad = policy.hand_box_margin * max(max(xs) - min(xs), max(ys) - min(ys))
    box = (min(xs) - pad, min(ys) - pad, max(xs) - min(xs) + 2 * pad, max(ys) - min(ys) + 2 * pad)
    points = tuple(
        (f.points[j].x, f.points[j].y) for j in policy.hand_point_joints if f.points[j].visibility
    )
    return Prompt(box, points)


def clip_box(box: Box, size: tuple[int, int]) -> Box:
    """박스를 프레임 (너비, 높이) 안으로 자른다 (넓이 0이 될 수 있다)."""
    x0, y0 = max(0.0, box[0]), max(0.0, box[1])
    x1, y1 = min(float(size[0]), box[0] + box[2]), min(float(size[1]), box[1] + box[3])
    return (x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0))


def box_iou(a: Box, b: Box) -> float:
    """두 (x, y, w, h) 박스의 IoU."""
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def mask_bbox(mask: NDArray[np.bool_]) -> Box:
    """마스크 외접 박스 (x, y, w, h) px. 빈 마스크면 (0, 0, 0, 0)."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return (0.0, 0.0, 0.0, 0.0)
    return (float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1),
            float(ys.max() - ys.min() + 1))  # fmt: skip


def _largest(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """가장 큰 8-연결 성분만 남긴 마스크."""
    n, comp, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 2:
        return mask
    areas: NDArray[np.int32] = np.asarray(stats[1:, cv2.CC_STAT_AREA], np.int32)
    labels: NDArray[np.int32] = np.asarray(comp, np.int32)
    return labels == 1 + int(np.argmax(areas))


def _outside_ratio(mask: NDArray[np.bool_], box: Box, margin: float) -> float:
    """마스크 화소 중 (각 변을 긴 변 x margin만큼 넓힌) 박스 밖 비율."""
    area = int(mask.sum())
    if area == 0:
        return 0.0
    pad = margin * max(box[2], box[3])
    h, w = mask.shape
    x0, y0 = max(0, int(box[0] - pad)), max(0, int(box[1] - pad))
    x1 = min(w, int(np.ceil(box[0] + box[2] + pad)))
    y1 = min(h, int(np.ceil(box[1] + box[3] + pad)))
    inside = int(mask[y0:y1, x0:x1].sum())
    return (area - inside) / area


def check_mask(
    result: MaskResult | None,
    box: Box,
    policy: MasksPolicy,
    hands: NDArray[np.bool_] | None = None,
) -> tuple[NDArray[np.bool_] | None, str | None]:
    """모델 마스크 → (남길 마스크, None) 또는 (None, 버린 사유).

    사유: no_object(대상 없음), low_score, leak(넓힌 박스 밖 비율 초과), small(넓이 / 박스 넓이
    하한 미만, 손 마스크를 뺀 뒤 포함). hands가 있으면(객체 마스크) 그 화소를 뺀다.
    """
    if result is None:
        return None, "no_object"
    if result.score < policy.min_score:
        return None, "low_score"
    mask = result.mask
    if policy.largest_component:
        mask = _largest(mask)
    if _outside_ratio(mask, box, policy.outside_margin) > policy.max_outside_ratio:
        return None, "leak"
    if hands is not None:
        mask = mask & ~hands
    if mask.sum() < policy.min_area_ratio * max(box[2] * box[3], 1.0):
        return None, "small"
    return mask, None


@dataclass
class _Frame:
    """한 처리 프레임의 프롬프트 (손 먼저)."""

    sources: list[PromptTrack]
    prompts: list[Prompt]


def _is_wearer_box(
    box: Box,
    class_id: str,
    t: int,
    size: tuple[int, int],
    policy: MasksPolicy,
    wearer: tuple[WearerEvidence, WearerSuppressionPolicy] | None,
) -> bool:
    """박스 프롬프트가 착용자 몸으로 보이는가.

    1) 착용자 증거가 있으면 객체 단계와 같은 판정(`box_frame_is_wearer`, ADR 0058).
    2) 손이 안 보여 증거가 없는 시각에도, 사람 클래스(`person_classes`) 박스의 아래 변이 화면 아래
       `person_bottom_band` 띠 안이면 착용자 몸통으로 본다 (1인칭 화면의 몸통은 아래 가장자리에서
       들어온다. 파일럿에서 손 없는 시각의 착용자 셔츠가 사람 마스크가 됐다, ADR 0059).
    """
    if wearer is not None:
        evidence, wpolicy = wearer
        if box_frame_is_wearer(box, class_id, t, evidence, wpolicy, size):
            return True
    band = policy.person_bottom_band
    if band is None or class_id not in policy.person_classes:
        return False
    return box[1] + box[3] >= (1.0 - band) * size[1]


def frame_prompts(
    t: int,
    sources: Sequence[PromptTrack],
    policy: MasksPolicy,
    size: tuple[int, int],
    wearer: tuple[WearerEvidence, WearerSuppressionPolicy] | None,
    report: MaskReport,
) -> _Frame:
    """시각 t의 프롬프트 (작은 박스·착용자 몸 박스 제외, 손 → 신뢰도 순, 상한까지)."""
    picked: list[tuple[PromptTrack, Prompt]] = []
    for s in sources:
        p = s.record.payload
        if s.is_hand:
            assert isinstance(p, KeypointTrackPayload)
            prompt = hand_prompt_at(p, s.times, t, policy)
        else:
            assert isinstance(p, BoxTrackPayload)
            box = box_at(p, s.times, t, policy.box_max_gap_ms)
            if box is not None and _is_wearer_box(box, s.class_id, t, size, policy, wearer):
                report.wearer_skipped += 1
                box = None
            prompt = None if box is None else Prompt(box)
        if prompt is None:
            continue
        box = clip_box(prompt.box, size)
        if min(box[2], box[3]) < policy.min_box_px:
            continue
        picked.append((s, Prompt(box, prompt.points)))
    picked.sort(key=lambda sp: -sp[0].rank)
    picked = picked[: policy.max_prompts_per_frame]
    return _Frame([s for s, _ in picked], [p for _, p in picked])


def _span(sources: Sequence[PromptTrack]) -> tuple[int, int] | None:
    """프롬프트 트랙 전체의 (처음, 끝) 키프레임 시각."""
    spans = [(s.times[0], s.times[-1]) for s in sources if s.times]
    return (min(a for a, _ in spans), max(b for _, b in spans)) if spans else None


@dataclass
class MaskRun:
    """`make_masks` 결과: 개체 → 마스크 키프레임 목록, 요약."""

    keyframes: dict[str, list[MaskKeyframe]]
    report: MaskReport


def make_masks(
    frames: Iterable[tuple[int, Image]],
    sources: Sequence[PromptTrack],
    segmenter: Segmenter,
    policy: MasksPolicy,
    wearer: tuple[WearerEvidence, WearerSuppressionPolicy] | None = None,
) -> MaskRun:
    """영상 프레임을 돌며 frame_stride_ms마다 프롬프트가 있는 프레임에서 마스크를 만든다.

    프롬프트가 없는 프레임은 모델을 돌리지 않고 다음 프레임을 본다 (간격 시계는 모델을 돌린
    프레임에서만 간다).
    """
    report = MaskReport()
    out: dict[str, list[MaskKeyframe]] = {}
    span = _span(sources)
    due: int | None = None
    for t, image in strictly_increasing(frames):
        if span is None or t > span[1]:
            break
        if t < span[0] or (due is not None and t < due):
            continue
        size = (image.shape[1], image.shape[0])
        frame = frame_prompts(t, sources, policy, size, wearer, report)
        if not frame.prompts:
            continue
        due = t + policy.frame_stride_ms
        start = time.perf_counter()
        results = segmenter.predict(image, frame.prompts)
        report.seconds += time.perf_counter() - start
        report.frames += 1
        report.prompts += len(frame.prompts)
        _keep(t, frame, results, policy, out, report)
    report.tracks = len(out)
    return MaskRun(out, report)


def _keep(
    t: int,
    frame: _Frame,
    results: Sequence[MaskResult | None],
    policy: MasksPolicy,
    out: dict[str, list[MaskKeyframe]],
    report: MaskReport,
) -> None:
    """한 프레임 결과를 검사해 개체별 키프레임에 더한다 (손을 먼저 처리해 객체에서 뺀다)."""
    hands: NDArray[np.bool_] | None = None
    order = sorted(range(len(frame.sources)), key=lambda i: not frame.sources[i].is_hand)
    for i in order:
        s, prompt, r = frame.sources[i], frame.prompts[i], results[i]
        subtract = None if s.is_hand or not policy.subtract_hands else hands
        mask, reason = check_mask(r, prompt.box, policy, subtract)
        if mask is None or r is None:
            report.dropped[reason or "no_object"] += 1
            continue
        if s.is_hand:
            hands = mask if hands is None else hands | mask
            report.hand_masks += 1
        report.masks += 1
        ious = report.hand_box_iou if s.is_hand else report.box_iou
        ious.append(round(box_iou(mask_bbox(mask), prompt.box), 4))
        out.setdefault(s.entity_id, []).append(
            MaskKeyframe(t_ms=t, rle=encode_rle(mask), score=round(r.score, 4))
        )


def mask_labels(
    run: MaskRun,
    sources: Sequence[PromptTrack],
    *,
    session_id: str,
    stream_id: str,
    ontology_version: str,
    version: str,
    now: datetime,
) -> list[LabelRecord]:
    """개체별 키프레임 → mask_track 라벨 (개체 순). 신뢰도는 키프레임 점수 평균."""
    by_entity = {s.entity_id: s for s in sources}
    tag = version_tag(version)
    labels: list[LabelRecord] = []
    for entity, kfs in sorted(run.keyframes.items()):
        s = by_entity[entity]
        scores = [k.score for k in kfs if k.score is not None]
        payload = MaskTrackPayload(
            entity_id=entity, class_id=s.class_id, keyframes=tuple(kfs),
            derived_from=(s.record.label_id,),
        )  # fmt: skip
        labels.append(
            model_label(
                label_id=f"{session_id}-{stream_id}-masks-{tag}-{entity}",
                session_id=session_id,
                stream_id=stream_id,
                t_start_ms=kfs[0].t_ms,
                t_end_ms=kfs[-1].t_ms,
                ontology_version=ontology_version,
                model_version=version,
                confidence=float(np.mean(scores)) if scores else 0.0,
                payload=payload,
                now=now,
            )
        )
    return labels


def weights_version(root: Path, policy: PrelabelPolicy) -> str:
    """네 가중치 파일(그래프·외부 데이터)의 실제 해시를 묶은 짧은 판 (`sam2t-<8자>`).

    Raises:
        ModelUnavailableError: 파일이 없거나 해시가 등록값과 다를 때 (`make models`).
    """
    versions = [resolve(root, getattr(policy.models, role))[1] for role in MASK_ROLES]
    return "sam2t-" + hashlib.sha256("|".join(versions).encode()).hexdigest()[:8]


class MaskMaker:
    """마스크 단계. 러너가 Predictor 목록과 따로 부른다 (입력이 영상 + 다른 라벨이다).

    name: "masks". version: `masks-<가중치 판>+mk1+p<masks 절 해시>+w<wearer_suppression 절 해시>`
    (러너가 입력 해시 `+i<...>`를 붙인다). 가중치 판은 네 파일의 실제 해시로 정한다.
    """

    name = "masks"

    def __init__(
        self, policy: PrelabelPolicy, segmenter: Segmenter, weights: str, *, now: datetime
    ) -> None:
        """policy: 프리라벨 정책. segmenter: 마스크 모델 (실제 `Sam2Segmenter` 또는 stub).

        weights: 가중치 판 (`weights_version`, stub이면 "stub"). now: 새 라벨 created_at.
        """
        self.policy, self.segmenter, self.now = policy, segmenter, now
        digest = policy.digest("masks")
        wearer = policy.digest("wearer_suppression")
        self.version = f"masks-{weights}+{MASK_REVISION}+p{digest}+w{wearer}"
        self.last_report: MaskReport | None = None

    @classmethod
    def from_registry(cls, root: Path, policy: PrelabelPolicy, *, now: datetime) -> MaskMaker:
        """등록 가중치로 실제 SAM 2.1 분할기를 만든다.

        외부 데이터 파일은 그래프와 같은 디렉터리에 등록 이름 그대로 있어야 한다 (ONNX가 그 이름으로
        읽는다). 세션은 여기서 연다 (수 초).

        Raises:
            ModelUnavailableError: 가중치가 없거나 해시가 다르거나 데이터 파일 위치가 맞지 않을 때.
        """
        from dlp_models.sam2 import Sam2Segmenter  # onnxruntime 세션은 실제 실행 때만
        from dlp_schema.predictor import ModelUnavailableError

        weights = weights_version(root, policy)
        paths = {role: resolve(root, getattr(policy.models, role))[0] for role in MASK_ROLES}
        for graph in ("mask_encoder", "mask_decoder"):
            if paths[f"{graph}_data"] != paths[graph].with_name(paths[graph].name + "_data"):
                raise ModelUnavailableError(
                    f"{graph}: 외부 가중치 파일은 그래프 옆 {paths[graph].name}_data여야 합니다"
                )
        segmenter = Sam2Segmenter(paths["mask_encoder"], paths["mask_decoder"])
        return cls(policy, segmenter, weights, now=now)

    def run(
        self,
        video: Path,
        *,
        session_id: str,
        stream_id: str,
        tracks: Sequence[LabelRecord],
        ontology_version: str,
        version: str | None = None,
        wearer: WearerEvidence | None = None,
        frames: Iterable[tuple[int, Image]] | None = None,
    ) -> list[LabelRecord]:
        """바디캠 영상에서 마스크 라벨을 만든다 (DB에 쓰지 않는다, 러너가 쓴다).

        Args:
            video: 바디캠 영상 로컬 경로.
            tracks: 프롬프트 트랙 (`pick_mask_tracks`, 개체마다 하나).
            version: 라벨 모델 버전 (러너가 입력 해시를 붙여 넘긴다, 기본 `self.version`).
            wearer: 착용자 증거 (있으면 박스 프롬프트의 착용자 몸 판정에 쓴다).
            frames: 시험·파일럿용 프레임 순회 (기본 `iter_frames(video)`).
        """
        sources = [s for x in tracks if (s := mask_source(x, self.policy.masks)) is not None]
        wpolicy = self.policy.wearer_suppression
        use = wearer is not None and not wearer.empty and wpolicy.enabled
        guard = (wearer, wpolicy) if use and wearer is not None else None
        run = make_masks(
            frames if frames is not None else iter_frames(video),
            sources, self.segmenter, self.policy.masks, guard,
        )  # fmt: skip
        self.last_report = run.report
        return mask_labels(
            run, sources, session_id=session_id, stream_id=stream_id,
            ontology_version=ontology_version, version=version or self.version, now=self.now,
        )  # fmt: skip
