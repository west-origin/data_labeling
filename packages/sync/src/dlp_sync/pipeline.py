"""세션 동기화.

스트림 종류별로 정책의 방법 목록을 순서대로 시도하고, 신뢰도가 min_confidence 이상인 첫 결과를 쓴다.
기준 스트림(바디캠)과 같은 시계인 스트림(shared_clock)은 건드리지 않는다. 사람이 넣은
manual_adjustment_ms는 다시 동기화해도 유지한다.

신뢰도
- qr_slate: 슬레이트 2개 이상 0.95, 1개 0.8 (프레임 간격만큼의 양자화 오차가 있다)
- tap_event: (짝지은 두 번 두드림 수 / 2, 최대 1) * exp(-잔차 RMS / residual_scale_ms)
- audio_xcorr, motion_xcorr: 1 - min_psr / PSR (PSR이 min_psr 이하면 0), 앵커 잔차가 크면 낮춘다
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from dlp_schema.session import Session, Stream, StreamKind, SyncMethod
from dlp_sync.anchors import Anchor, ClockFit, FitError, fit_clock
from dlp_sync.policy import MethodName, SyncPolicy
from dlp_sync.signals import Audio, Series
from dlp_sync.slate import SlateSighting, detect_slates
from dlp_sync.taps import DoubleTap, detect_double_taps, match_taps
from dlp_sync.xcorr import XcorrResult, audio_anchors, motion_anchors

METHOD_ENUM: dict[MethodName, SyncMethod] = {
    "qr_slate": SyncMethod.QR_SLATE,
    "tap_event": SyncMethod.TAP_EVENT,
    "audio_xcorr": SyncMethod.AUDIO_XCORR,
    "motion_xcorr": SyncMethod.MOTION_XCORR,
}


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
        self._slates: list[SlateSighting] | None = None
        self._audio_taps: list[DoubleTap] | None = None

    @property
    def slates(self) -> list[SlateSighting]:
        if self._slates is None:
            v = self.media.video
            self._slates = detect_slates(v, self.policy.slate) if v else []
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
        if stream.sync_method in (SyncMethod.REFERENCE, SyncMethod.SHARED_CLOCK):
            updated.append(stream)
            continue
        methods = policy.methods.get(stream.kind.value, ())
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
                ref.media.audio, target.audio, policy.audio_xcorr, policy.max_offset_ms
            )
            return _from_xcorr(method, res, policy.audio_xcorr.min_psr, policy)
        if ref.imu is None or target.series is None:
            return Attempt(method, 0.0, "기준 IMU나 대상 시계열이 없습니다")
        res = motion_anchors(ref.imu, target.series, policy.motion_xcorr, policy.max_offset_ms)
        return _from_xcorr(method, res, policy.motion_xcorr.min_psr, policy)
    except FitError as exc:
        return Attempt(method, 0.0, str(exc))


def _slate(ref: _Reference, target: StreamMedia, policy: SyncPolicy) -> Attempt:
    if target.video is None:
        return Attempt("qr_slate", 0.0, "영상이 없습니다")
    ref_by_payload = {s.payload: s.stream_ms for s in ref.slates}
    anchors = [
        Anchor(ref_by_payload[s.payload], s.stream_ms, s.payload)
        for s in detect_slates(target.video, policy.slate)
        if s.payload in ref_by_payload
    ]
    if not anchors:
        return Attempt("qr_slate", 0.0, "두 영상에서 같은 슬레이트를 찾지 못했습니다")
    if policy.slate.estimate_drift:
        fit = _fit(anchors, policy)
    else:
        fit = fit_clock(anchors, min_drift_span_ms=math.inf, max_drift_ppm=policy.max_drift_ppm)
    confidence = 0.95 if len(anchors) >= 2 else 0.8
    return Attempt("qr_slate", confidence, f"슬레이트 {len(anchors)}개", fit, anchors)


def _tap(ref: _Reference, target: StreamMedia, policy: SyncPolicy) -> Attempt:
    if not ref.audio_taps:
        return Attempt("tap_event", 0.0, "기준 오디오에서 두드림을 찾지 못했습니다")
    if target.audio is not None:
        target_taps = _audio_taps(target.audio, policy)
    elif target.series is not None:
        target_taps = detect_double_taps(target.series.t_ms, target.series.values, policy.tap)
    else:
        return Attempt("tap_event", 0.0, "대상 신호가 없습니다")
    anchors = match_taps(ref.audio_taps, target_taps, policy.tap)
    if not anchors:
        return Attempt("tap_event", 0.0, "두드림을 짝짓지 못했습니다")
    fit = _fit(anchors, policy)
    pairs = len(anchors) // 2
    confidence = min(1.0, pairs / 2) * math.exp(-fit.residual_rms_ms / policy.tap.residual_scale_ms)
    return Attempt("tap_event", confidence, f"두 번 두드림 {pairs}쌍", fit, anchors)


def _from_xcorr(
    method: MethodName, res: XcorrResult | None, min_psr: float, policy: SyncPolicy
) -> Attempt:
    if res is None:
        return Attempt(method, 0.0, "상관 탐색 범위가 겹치지 않습니다")
    fit = _fit(res.anchors, policy)
    confidence = max(0.0, 1 - min_psr / res.psr) if res.psr > 0 else 0.0
    if fit.n_anchors >= 2:
        confidence *= math.exp(-fit.residual_rms_ms / 5.0)
    return Attempt(method, confidence, f"PSR {res.psr:.1f}", fit, res.anchors)


def apply_manual_adjustment(session: Session, stream_id: str, adjustment_ms: float) -> Session:
    """사람이 검수 화면에서 정한 미세 조정값(ms)을 기록한다. 자동 결과는 그대로 둔다."""
    stream = session.stream(stream_id)
    if stream.sync_method is SyncMethod.REFERENCE:
        raise ValueError("기준 스트림은 조정할 수 없습니다")
    streams = [
        s.model_copy(update={"manual_adjustment_ms": adjustment_ms})
        if s.stream_id == stream_id
        else s
        for s in session.streams
    ]
    return Session.model_validate(
        {**session.model_dump(), "streams": [s.model_dump() for s in streams]}
    )
