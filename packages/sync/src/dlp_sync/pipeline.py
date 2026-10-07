"""세션 동기화.

스트림 종류별로 정책의 방법 목록을 순서대로 시도하고, 신뢰도가 min_confidence 이상인 첫 결과를 쓴다.
기준 스트림(바디캠)과 같은 시계인 스트림(shared_clock)은 건드리지 않는다. 사람이 넣은
manual_adjustment_ms는 다시 동기화해도 유지한다.

- 사람이 직접 맞춘 스트림(manual)은 다시 동기화해도 건드리지 않는다.
- 다시 돌렸는데 이번에는 맞추지 못하면 이전의 자동 결과를 그대로 둔다 (unsynced로 내리지 않는다).
- 자동으로 맞추지 못한(unsynced) 스트림에 사람이 조정값을 넣으면 manual이 된다
  (오프셋 0, 배율 1, 조정값 = 사람이 정한 오프셋). unsynced 스트림은 다음 단계가 쓰지 않는다.

신뢰도
- qr_slate: 슬레이트 2개 이상 confidence_many, 1개 confidence_one에
  exp(-max(0, 오차 상한 - 양자화) / slate.residual_scale_ms)를 곱한다.
  오차 상한 = 최대 앵커 잔차 + 드리프트가 쌓일 수 있는 양(드리프트를 맞췄으면 앵커 밖 외삽분,
  아니면 max_drift_ppm·앵커에서 가장 먼 거리). 양자화 = 앵커의 프레임 간격(gap_ms) 중 최대.
  긴 녹화를 오프셋만으로 맞추면 신뢰도가 떨어져 다음 방법으로 넘어간다.
  두 영상에 오디오가 있으면(refine_with_audio) 슬레이트 맞춤을 출발점으로 오디오 상관 창을 맞춰
  드리프트까지 다듬고, 그 결과가 슬레이트 앵커와 양자화 안에서 맞으면 쓴다
  (신뢰도는 슬레이트 기본값과 상관 신뢰도 중 큰 값).
- tap_event: (짝지은 두 번 두드림 수 / 2, 최대 1) * exp(-잔차 RMS / residual_scale_ms)
- audio_xcorr, motion_xcorr: 1 - min_psr / PSR (PSR이 min_psr 이하면 0). 앵커가 2개 이상이면
  exp(-잔차 RMS / residual_scale_ms)를 곱한다
"""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from dlp_schema.session import Session, Stream, StreamKind, SyncMethod
from dlp_sync.anchors import Anchor, ClockFit, FitError, fit_clock
from dlp_sync.policy import MethodName, SyncPolicy
from dlp_sync.signals import Audio, Series
from dlp_sync.slate import SlateScan, scan_slates
from dlp_sync.taps import DoubleTap, detect_double_taps, match_taps
from dlp_sync.xcorr import LagPrior, XcorrResult, audio_anchors, motion_anchors

METHOD_ENUM: dict[MethodName, SyncMethod] = {
    "qr_slate": SyncMethod.QR_SLATE,
    "tap_event": SyncMethod.TAP_EVENT,
    "audio_xcorr": SyncMethod.AUDIO_XCORR,
    "motion_xcorr": SyncMethod.MOTION_XCORR,
}


# 다시 동기화해도 건드리지 않는 스트림: 기준, 같은 시계, 사람이 맞춘 스트림
KEEP_METHODS = frozenset({SyncMethod.REFERENCE, SyncMethod.SHARED_CLOCK, SyncMethod.MANUAL})
AUTO_METHODS = frozenset(METHOD_ENUM.values())


@dataclass
class StreamMedia:
    """한 스트림의 동기화 입력. 없는 항목은 None."""

    video: Path | None = None
    audio: Audio | None = None
    series: Series | None = None  # 장갑 압력 합, IMU 가속도 크기 등


@dataclass
class Attempt:
    method: MethodName
    confidence: float
    reason: str
    fit: ClockFit | None = None
    anchors: list[Anchor] = field(default_factory=list[Anchor])


@dataclass
class StreamReport:
    stream_id: str
    chosen: MethodName | None
    attempts: list[Attempt]


@dataclass
class SyncReport:
    session_id: str
    streams: list[StreamReport]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class _Reference:
    """기준 스트림에서 한 번만 계산해 두는 값."""

    def __init__(self, media: StreamMedia, imu: Series | None, policy: SyncPolicy) -> None:
        self.media = media
        self.policy = policy
        self.imu = imu
        self._slates: SlateScan | None = None
        self._audio_taps: list[DoubleTap] | None = None

    @property
    def slates(self) -> SlateScan:
        if self._slates is None:
            v = self.media.video
            self._slates = scan_slates(v, self.policy.slate) if v else SlateScan([], None)
        return self._slates

    @property
    def audio_taps(self) -> list[DoubleTap]:
        if self._audio_taps is None:
            self._audio_taps = _audio_taps(self.media.audio, self.policy)
        return self._audio_taps


def _audio_taps(audio: Audio | None, policy: SyncPolicy) -> list[DoubleTap]:
    if audio is None:
        return []
    t = audio.start_ms + np.arange(audio.samples.size) * 1000 / audio.rate
    return detect_double_taps(t, audio.samples.astype(np.float64), policy.tap)


def synchronize(
    session: Session, media: dict[str, StreamMedia], policy: SyncPolicy
) -> tuple[Session, SyncReport]:
    ref_stream = session.reference_stream
    shared_imu = next(
        (
            media[s.stream_id].series
            for s in session.streams
            if s.kind is StreamKind.IMU
            and s.sync_method is SyncMethod.SHARED_CLOCK
            and s.stream_id in media
        ),
        None,
    )
    ref = _Reference(media.get(ref_stream.stream_id, StreamMedia()), shared_imu, policy)

    updated: list[Stream] = []
    reports: list[StreamReport] = []
    for stream in session.streams:
        if stream.sync_method in KEEP_METHODS:
            updated.append(stream)
            continue
        methods = policy.methods.get(stream.kind, ())
        target = media.get(stream.stream_id, StreamMedia())
        attempts: list[Attempt] = []
        chosen: Attempt | None = None
        for method in methods:
            attempt = _try(method, ref, target, policy)
            attempts.append(attempt)
            if attempt.fit is not None and attempt.confidence >= policy.min_confidence:
                chosen = attempt
                break
        reports.append(StreamReport(stream.stream_id, chosen.method if chosen else None, attempts))
        if chosen is None or chosen.fit is None:
            if stream.sync_method in AUTO_METHODS:
                updated.append(stream)  # 이전의 자동 결과를 유지한다
                continue
            updated.append(
                stream.model_copy(
                    update={
                        "offset_ms": 0.0,
                        "clock_scale": 1.0,
                        "sync_method": SyncMethod.UNSYNCED,
                        "sync_confidence": None,
                    }
                )
            )
        else:
            updated.append(
                stream.model_copy(
                    update={
                        "offset_ms": chosen.fit.offset_ms,
                        "clock_scale": chosen.fit.clock_scale,
                        "sync_method": METHOD_ENUM[chosen.method],
                        "sync_confidence": round(chosen.confidence, 4),
                    }
                )
            )
    new_session = Session.model_validate(
        {**session.model_dump(), "streams": [s.model_dump() for s in updated]}
    )
    return new_session, SyncReport(session.session_id, reports)


def _fit(anchors: list[Anchor], policy: SyncPolicy) -> ClockFit:
    return fit_clock(
        anchors, min_drift_span_ms=policy.min_drift_span_ms, max_drift_ppm=policy.max_drift_ppm
    )


def _try(method: MethodName, ref: _Reference, target: StreamMedia, policy: SyncPolicy) -> Attempt:
    try:
        if method == "qr_slate":
            return _slate(ref, target, policy)
        if method == "tap_event":
            return _tap(ref, target, policy)
        if method == "audio_xcorr":
            if ref.media.audio is None or target.audio is None:
                return Attempt(method, 0.0, "오디오가 없습니다")
            res = audio_anchors(
                ref.media.audio,
                target.audio,
                policy.audio_xcorr,
                policy.max_offset_ms,
                max_drift_ppm=policy.max_drift_ppm,
            )
            ax = policy.audio_xcorr
            return _from_xcorr(method, res, ax.min_psr, ax.residual_scale_ms, policy)
        if ref.imu is None or target.series is None:
            return Attempt(method, 0.0, "기준 IMU나 대상 시계열이 없습니다")
        res = motion_anchors(
            ref.imu,
            target.series,
            policy.motion_xcorr,
            policy.max_offset_ms,
            max_drift_ppm=policy.max_drift_ppm,
        )
        mx = policy.motion_xcorr
        return _from_xcorr(method, res, mx.min_psr, mx.residual_scale_ms, policy)
    except FitError as exc:
        return Attempt(method, 0.0, str(exc))


def _slate(ref: _Reference, target: StreamMedia, policy: SyncPolicy) -> Attempt:
    if target.video is None:
        return Attempt("qr_slate", 0.0, "영상이 없습니다")
    sp = policy.slate
    ref_by_payload = {s.payload: s for s in ref.slates.sightings}
    scan = scan_slates(target.video, sp)
    matched = [
        (ref_by_payload[s.payload], s) for s in scan.sightings if s.payload in ref_by_payload
    ]
    if not matched:
        return Attempt("qr_slate", 0.0, "두 영상에서 같은 슬레이트를 찾지 못했습니다")
    anchors = [Anchor(r.stream_ms, t.stream_ms, t.payload) for r, t in matched]
    # 앵커 오차는 (-대상 간격, +기준 간격) 안이다
    quant_ms = max(max(r.gap_ms, t.gap_ms) for r, t in matched)
    stream_s = [a.stream_ms for a in anchors]
    span = max(stream_s) - min(stream_s)
    drift = sp.estimate_drift and len(anchors) >= 2 and span >= sp.min_drift_span_ms
    fit = fit_clock(
        anchors,
        min_drift_span_ms=sp.min_drift_span_ms if drift else math.inf,
        max_drift_ppm=policy.max_drift_ppm,
    )
    extent = _stream_extent(target, scan, stream_s)
    residual = max(abs(a.master_ms - fit.to_master(a.stream_ms)) for a in anchors)
    if fit.drift_estimated:
        # 앵커 밖으로 외삽하는 구간: 드리프트 오차 상한 (양 끝 앵커 오차 합 / 앵커 간격)
        outside = max(min(stream_s) - extent[0], extent[1] - max(stream_s), 0.0)
        drift_bound = 2 * quant_ms * outside / span
    else:
        drift_bound = policy.max_drift_ppm * 1e-6 * _farthest(stream_s, extent)
    bound = residual + drift_bound
    base = sp.confidence_many if len(anchors) >= 2 else sp.confidence_one
    confidence = base * math.exp(-max(0.0, bound - quant_ms) / sp.residual_scale_ms)
    reason = f"슬레이트 {len(anchors)}개, 오차 상한 {bound:.1f} ms (양자화 {quant_ms:.1f} ms)"
    attempt = Attempt("qr_slate", confidence, reason, fit, anchors)
    if sp.refine_with_audio and ref.media.audio is not None and target.audio is not None:
        refined = _refine_slate(attempt, ref.media.audio, target.audio, quant_ms, policy)
        if refined is not None:
            return refined
        attempt.reason += ", 오디오 정밀화 실패"
    return attempt


def _stream_extent(
    target: StreamMedia, scan: SlateScan, stream_s: list[float]
) -> tuple[float, float]:
    """대상 스트림 시각의 범위. 길이를 모르면 앵커 범위."""
    end = scan.duration_ms
    if end is None and target.audio is not None:
        end = target.audio.start_ms + target.audio.duration_ms
    return (0.0, end if end is not None else max(stream_s))


def _farthest(points: list[float], extent: tuple[float, float]) -> float:
    """범위 안의 점에서 가장 가까운 앵커까지 거리의 최댓값."""
    pts = sorted(points)
    gaps = [(b - a) / 2 for a, b in itertools.pairwise(pts)]
    return max([pts[0] - extent[0], extent[1] - pts[-1], *gaps, 0.0])


def _refine_slate(
    slate: Attempt, ref_audio: Audio, target_audio: Audio, quant_ms: float, policy: SyncPolicy
) -> Attempt | None:
    """슬레이트 맞춤을 출발점으로 오디오 상관 창을 맞춘다.

    결과가 슬레이트 앵커와 양자화(+ refine_ms) 안에서 맞을 때만 쓴다.
    """
    assert slate.fit is not None
    fit = slate.fit
    ax = policy.audio_xcorr
    # 드리프트를 맞췄으면 그 추정 오차(앵커 오차 합 / 앵커 간격), 아니면 드리프트 상한으로
    # 앵커에서 멀수록 넓게 찾는다
    drift_ppm = 2 * quant_ms / fit.span_ms * 1e6 if fit.drift_estimated else policy.max_drift_ppm
    prior = LagPrior(
        fit.offset_ms,
        fit.clock_scale,
        tuple(a.master_ms for a in slate.anchors),
        2 * quant_ms + ax.refine_ms,
        drift_ppm,
    )
    res = audio_anchors(
        ref_audio, target_audio, ax, policy.max_offset_ms,
        max_drift_ppm=policy.max_drift_ppm, prior=prior,
    )  # fmt: skip
    if res is None or not res.anchors:
        return None
    xc = _from_xcorr("audio_xcorr", res, ax.min_psr, ax.residual_scale_ms, policy)
    if xc.fit is None or xc.confidence < policy.min_confidence:
        return None
    worst = max(abs(a.master_ms - xc.fit.to_master(a.stream_ms)) for a in slate.anchors)
    if worst > quant_ms + ax.refine_ms:
        return None
    sp = policy.slate
    base = sp.confidence_many if len(slate.anchors) >= 2 else sp.confidence_one
    return Attempt(
        "qr_slate",
        max(base, xc.confidence),
        f"{slate.reason}, 오디오 정밀화 {xc.reason} 창 {len(xc.anchors)}개",
        xc.fit,
        [*slate.anchors, *xc.anchors],
    )


def _tap(ref: _Reference, target: StreamMedia, policy: SyncPolicy) -> Attempt:
    if not ref.audio_taps:
        return Attempt("tap_event", 0.0, "기준 오디오에서 두드림을 찾지 못했습니다")
    if target.audio is not None:
        target_taps = _audio_taps(target.audio, policy)
    elif target.series is not None:
        target_taps = detect_double_taps(target.series.t_ms, target.series.values, policy.tap)
    else:
        return Attempt("tap_event", 0.0, "대상 신호가 없습니다")
    anchors = match_taps(
        ref.audio_taps, target_taps, policy.tap, max_drift_ppm=policy.max_drift_ppm
    )
    if not anchors:
        return Attempt("tap_event", 0.0, "두드림을 짝짓지 못했습니다")
    fit = _fit(anchors, policy)
    pairs = len(anchors) // 2
    confidence = min(1.0, pairs / 2) * math.exp(-fit.residual_rms_ms / policy.tap.residual_scale_ms)
    return Attempt("tap_event", confidence, f"두 번 두드림 {pairs}쌍", fit, anchors)


def _from_xcorr(
    method: MethodName,
    res: XcorrResult | None,
    min_psr: float,
    residual_scale_ms: float,
    policy: SyncPolicy,
) -> Attempt:
    if res is None:
        return Attempt(method, 0.0, "상관 탐색 범위가 겹치지 않습니다")
    if not res.anchors:
        return Attempt(method, 0.0, f"PSR {res.psr:.1f} (모든 창이 min_psr 미만)")
    fit = _fit(res.anchors, policy)
    confidence = max(0.0, 1 - min_psr / res.psr) if res.psr > 0 else 0.0
    if fit.n_anchors >= 2:
        confidence *= math.exp(-fit.residual_rms_ms / residual_scale_ms)
    return Attempt(method, confidence, f"PSR {res.psr:.1f}", fit, res.anchors)


def apply_manual_adjustment(session: Session, stream_id: str, adjustment_ms: float) -> Session:
    """사람이 검수 화면에서 정한 미세 조정값(ms)을 기록한다. 자동 결과는 그대로 둔다.

    자동으로 맞추지 못한(unsynced) 스트림이면 사람이 오프셋 전체를 정한 것이므로 manual로 바꾼다
    (오프셋 0, 배율 1, 조정값 = 사람이 정한 오프셋). 이후 다시 동기화해도 덮어쓰지 않는다.
    """
    stream = session.stream(stream_id)
    if stream.sync_method is SyncMethod.REFERENCE:
        raise ValueError("기준 스트림은 조정할 수 없습니다")
    update: dict[str, object] = {"manual_adjustment_ms": adjustment_ms}
    if stream.sync_method is SyncMethod.UNSYNCED:
        update.update(
            offset_ms=0.0, clock_scale=1.0, sync_method=SyncMethod.MANUAL, sync_confidence=None
        )
    streams = [
        s.model_copy(update=update) if s.stream_id == stream_id else s for s in session.streams
    ]
    return Session.model_validate(
        {**session.model_dump(), "streams": [s.model_dump() for s in streams]}
    )
