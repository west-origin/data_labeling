"""세션 동기화 (WP4, ADR 0004·0028). `dlp sync run`의 계산 본체.

입력: 세션(스트림 목록과 현재 동기화 상태) + 스트림별 신호(`StreamMedia`) + 정책.
출력: 스트림의 `offset_ms`·`clock_scale`·`sync_method`·`sync_confidence`를 갱신한 새 세션 + 보고서.
DB·저장소에 접근하지 않는 순수 함수다 (`runner.run_sync`가 저장한다). 같은 입력이면 같은 결과(멱등).

시계 모델: master_ms = offset_ms + manual_adjustment_ms + stream_ms * clock_scale (계약
`Stream.to_master_ms`). 마스터 타임라인 = 바디캠(기준 스트림) 시계. 앵커와 맞춤은 `anchors.py`.

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

대체 순서 (기본 `sync.yaml methods`)
- 3인칭 영상: qr_slate → tap_event → audio_xcorr
- 장갑(좌·우): tap_event → motion_xcorr (바디캠 shared_clock IMU가 있어야 운동 상관 가능)
- 외부 IMU: tap_event / 외부 오디오: tap_event → audio_xcorr
어떤 방법이든 예외(`FitError`)나 입력 부족은 신뢰도 0의 시도로 기록하고 다음 방법으로 넘어간다.
두드림 기준은 언제나 바디캠 오디오의 두드림이다.

주요 공개 항목: `synchronize`, `apply_manual_adjustment`, `StreamMedia`, `SyncReport`
(`StreamReport`, `Attempt`), `METHOD_ENUM`.
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

# 정책의 방법 이름 → 계약의 `SyncMethod` (DB `streams.sync_method`에 저장되는 값)
METHOD_ENUM: dict[MethodName, SyncMethod] = {
    "qr_slate": SyncMethod.QR_SLATE,
    "tap_event": SyncMethod.TAP_EVENT,
    "audio_xcorr": SyncMethod.AUDIO_XCORR,
    "motion_xcorr": SyncMethod.MOTION_XCORR,
}


# 다시 동기화해도 건드리지 않는 스트림: 기준, 같은 시계, 사람이 맞춘 스트림
KEEP_METHODS = frozenset({SyncMethod.REFERENCE, SyncMethod.SHARED_CLOCK, SyncMethod.MANUAL})
# 자동 동기화가 남기는 방법. 다시 돌려 실패해도 이 결과는 unsynced로 내리지 않는다
AUTO_METHODS = frozenset(METHOD_ENUM.values())


@dataclass
class StreamMedia:
    """한 스트림의 동기화 입력. 없는 항목은 None.

    `loader.load_session_media`가 스트림 종류에 따라 채운다. 시각은 모두 그 스트림 시계 ms.
    """

    video: Path | None = None  # 영상 파일 로컬 경로 (QR 슬레이트 검출)
    audio: Audio | None = None  # 오디오 (두드림 검출, 오디오 상관)
    series: Series | None = None  # 장갑 압력 합, IMU 가속도 크기 등


@dataclass
class Attempt:
    """방법 하나를 시도한 결과 (보고서에 그대로 남는다).

    Attributes:
        method: 정책의 방법 이름.
        confidence: 0~1 신뢰도 (산식은 모듈 docstring). 실패면 0.
        reason: 사람이 읽을 설명 (실패 사유, 앵커 수, PSR 등).
        fit: 맞춘 시계. 실패면 None (신뢰도가 높아도 fit이 없으면 채택하지 않는다).
        anchors: 사용한 앵커 (보고서·디버깅용).
    """

    method: MethodName
    confidence: float
    reason: str
    fit: ClockFit | None = None
    anchors: list[Anchor] = field(default_factory=list[Anchor])


@dataclass
class StreamReport:
    """스트림 하나의 동기화 보고: 채택한 방법(없으면 None)과 시도 전체 (순서대로)."""

    stream_id: str
    chosen: MethodName | None
    attempts: list[Attempt]


@dataclass
class SyncReport:
    """세션 동기화 보고서. `sessions/<세션>/derived/sync_report.json`으로 저장된다 (덮어씀).

    기준·shared_clock·manual 스트림은 시도하지 않으므로 `streams`에 없다.
    """

    session_id: str
    streams: list[StreamReport]

    def to_dict(self) -> dict[str, object]:
        """JSON으로 쓸 수 있는 dict (dataclass 재귀 변환)."""
        return asdict(self)


class _Reference:
    """기준 스트림에서 한 번만 계산해 두는 값.

    대상 스트림이 여럿이어도 기준 영상의 슬레이트 검출과 기준 오디오 두드림 검출은 한 번만 한다
    (처음 필요할 때 계산해 캐시). `imu`는 바디캠과 같은 시계(shared_clock)의 IMU 가속도 크기로,
    운동 상관의 기준 신호다 (없으면 None → motion_xcorr 불가).
    """

    def __init__(self, media: StreamMedia, imu: Series | None, policy: SyncPolicy) -> None:
        """기준 입력을 보관한다 (검출은 처음 쓸 때 한다).

        Args:
            media: 기준 스트림(바디캠)의 입력 (영상 경로, 오디오).
            imu: 바디캠 shared_clock IMU 신호 (없으면 None).
            policy: 동기화 정책.
        """
        self.media = media
        self.policy = policy
        self.imu = imu
        self._slates: SlateScan | None = None
        self._audio_taps: list[DoubleTap] | None = None

    @property
    def slates(self) -> SlateScan:
        """기준 영상의 슬레이트 검출 결과 (영상이 없으면 빈 결과)."""
        if self._slates is None:
            v = self.media.video
            self._slates = scan_slates(v, self.policy.slate) if v else SlateScan([], None)
        return self._slates

    @property
    def audio_taps(self) -> list[DoubleTap]:
        """기준 오디오의 두 번 두드림 (오디오가 없으면 빈 목록)."""
        if self._audio_taps is None:
            self._audio_taps = _audio_taps(self.media.audio, self.policy)
        return self._audio_taps


def _audio_taps(audio: Audio | None, policy: SyncPolicy) -> list[DoubleTap]:
    """오디오 샘플을 그대로 포락선으로 써서 두 번 두드림을 찾는다 (오디오 없으면 빈 목록)."""
    if audio is None:
        return []
    # 샘플 시각 ms = 시작 + 인덱스 / 레이트
    t = audio.start_ms + np.arange(audio.samples.size) * 1000 / audio.rate
    return detect_double_taps(t, audio.samples.astype(np.float64), policy.tap)


def synchronize(
    session: Session, media: dict[str, StreamMedia], policy: SyncPolicy
) -> tuple[Session, SyncReport]:
    """세션의 동기화 대상 스트림마다 정책 순서대로 방법을 시도해 시계를 맞춘다.

    Args:
        session: 현재 세션 (기준 스트림 포함, 이전 동기화 결과 포함 가능).
        media: `stream_id` → 동기화 입력. 없는 스트림은 빈 입력으로 다룬다.
        policy: `sync.yaml`.

    Returns:
        (새 세션, 보고서). 스트림 처리 규칙:
        - 기준·shared_clock·manual 스트림: 그대로 (보고서에도 없음).
        - 채택된 방법이 있으면: 오프셋·배율·방법·신뢰도(소수 4자리)를 바꾼다.
          `manual_adjustment_ms`는 건드리지 않는다.
        - 채택이 없고 이전 결과가 자동 방법이면: 이전 결과 유지.
        - 채택이 없고 그 밖(unsynced 등)이면: unsynced (오프셋 0, 배율 1, 신뢰도 None).

    Raises:
        pydantic.ValidationError: 새 세션이 계약을 어길 때 (정상 경로에서는 없다).
    """
    ref_stream = session.reference_stream
    # 바디캠과 같은 시계인 IMU(내장 IMU)의 신호: 운동 상관(motion_xcorr)의 기준
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
            # 앞에서부터 처음으로 기준을 넘은 방법을 쓰고 나머지는 시도하지 않는다
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
    # 계약 검증(기준 스트림 규칙 등)을 다시 거치도록 dict에서 새로 만든다
    new_session = Session.model_validate(
        {**session.model_dump(), "streams": [s.model_dump() for s in updated]}
    )
    return new_session, SyncReport(session.session_id, reports)


def _fit(anchors: list[Anchor], policy: SyncPolicy) -> ClockFit:
    """공통 드리프트 기준(`min_drift_span_ms`, `max_drift_ppm`)으로 `fit_clock`.

    Raises:
        FitError: `fit_clock` 참고.
    """
    return fit_clock(
        anchors, min_drift_span_ms=policy.min_drift_span_ms, max_drift_ppm=policy.max_drift_ppm
    )


def _try(method: MethodName, ref: _Reference, target: StreamMedia, policy: SyncPolicy) -> Attempt:
    """방법 하나를 시도한다. 입력 부족·`FitError`는 신뢰도 0의 `Attempt`로 바꾼다.

    `FitError` 외의 예외(파일 읽기 오류 등)는 그대로 올라간다.

    `method`가 앞의 세 이름이 아니면 `motion_xcorr`로 처리한다 (`MethodName`이 넷뿐이라 안전).
    """
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
    """QR 슬레이트로 맞춘다 (ADR 0028 결정 1~3).

    1. 대상 영상의 슬레이트를 찾아 기준 영상의 같은 payload와 짝지어 앵커로.
    2. 슬레이트 둘 이상이 `slate.min_drift_span_ms` 이상 떨어져 있으면(`estimate_drift`)
       드리프트까지, 아니면 오프셋만 맞춘다.
    3. 오차 상한(최대 앵커 잔차 + 드리프트 누적 가능량)으로 신뢰도를 깎는다.
    4. `refine_with_audio`이고 두 영상에 오디오가 있으면 오디오 상관으로 다듬는다 (`_refine_slate`).
       실패하면(결과 없음·어긋남, 또는 `FitError`) 슬레이트 결과를 그대로 쓰고 사유에 적는다
       (`FitError` 처리는 ADR 0031 결정 7).

    Raises:
        FitError: 슬레이트 앵커의 드리프트 추정이 상한을 넘을 때 (`_try`가 잡는다). 오디오
            정밀화의 `FitError`는 여기서 잡아 슬레이트 결과로 물러나므로 올라가지 않는다.
    """
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
        try:
            refined = _refine_slate(attempt, ref.media.audio, target.audio, quant_ms, policy)
        except FitError as exc:
            # 오디오 앵커의 드리프트가 상한을 넘는 등 정밀화만 실패했다. 예전에는 이 예외가
            # `_try`까지 올라가 이미 맞춘 슬레이트 결과까지 신뢰도 0으로 버렸다. 정밀화는 선택
            # 단계이므로 슬레이트 맞춤으로 물러나고 사유를 보고서에 남긴다.
            attempt.reason += f", 오디오 정밀화 실패 ({exc})"
            return attempt
        if refined is not None:
            return refined
        attempt.reason += ", 오디오 정밀화 실패"
    return attempt


def _stream_extent(
    target: StreamMedia, scan: SlateScan, stream_s: list[float]
) -> tuple[float, float]:
    """대상 스트림 시각의 범위. 길이를 모르면 앵커 범위.

    (0, 끝) ms. 끝은 컨테이너 길이 → 오디오 끝 → 마지막 앵커 순으로 정한다.
    """
    end = scan.duration_ms
    if end is None and target.audio is not None:
        end = target.audio.start_ms + target.audio.duration_ms
    return (0.0, end if end is not None else max(stream_s))


def _farthest(points: list[float], extent: tuple[float, float]) -> float:
    """범위 안의 점에서 가장 가까운 앵커까지 거리의 최댓값.

    앵커 바깥 양 끝까지의 거리와 앵커 사이 간격의 절반 중 가장 큰 값 ms. 오프셋만 맞췄을 때
    드리프트가 쌓일 수 있는 최대 거리다.
    """
    pts = sorted(points)
    gaps = [(b - a) / 2 for a, b in itertools.pairwise(pts)]
    return max([pts[0] - extent[0], extent[1] - pts[-1], *gaps, 0.0])


def _refine_slate(
    slate: Attempt, ref_audio: Audio, target_audio: Audio, quant_ms: float, policy: SyncPolicy
) -> Attempt | None:
    """슬레이트 맞춤을 출발점으로 오디오 상관 창을 맞춘다.

    결과가 슬레이트 앵커와 양자화(+ refine_ms) 안에서 맞을 때만 쓴다.

    탐색 반폭 = 2·양자화 + `audio_xcorr.refine_ms` + 드리프트 불확실성·(슬레이트 앵커까지 거리).

    Returns:
        정밀화한 `qr_slate` 시도 (fit은 오디오 창 앵커로 맞춘 것, 앵커는 슬레이트 + 오디오 창,
        신뢰도는 슬레이트 기본값과 상관 신뢰도 중 큰 값). 오디오 결과가 없거나, 신뢰도가
        `min_confidence` 미만이거나, 슬레이트 앵커와 어긋나면 None.

    Raises:
        FitError: 오디오 앵커의 드리프트가 상한을 넘을 때. 호출자 `_slate`가 잡아 슬레이트
            맞춤으로 물러난다 (슬레이트 결과는 버리지 않는다).
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
    # 오디오 맞춤이 슬레이트 앵커를 얼마나 설명하는지: 가장 크게 벗어난 슬레이트 앵커
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
    """두 번 두드림으로 맞춘다.

    기준은 바디캠 오디오의 두드림. 대상은 오디오가 있으면 오디오, 없으면 시계열(장갑 압력·IMU).
    신뢰도 = min(1, 짝 수 / 2) · exp(-잔차 RMS / `tap.residual_scale_ms`). 짝 하나면 최대 0.5라
    기본 `min_confidence` 0.6을 넘지 못한다 (ADR 0004 결정 4: 한 쌍만 맞으면 믿지 않는다).

    Raises:
        FitError: 드리프트 추정이 상한을 넘을 때 (`_try`가 잡는다).
    """
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
    """상관 결과 → `Attempt`. 신뢰도 = max(0, 1 - min_psr / PSR), 앵커 2개 이상이면 잔차로 깎는다.

    Raises:
        FitError: 드리프트 추정이 상한을 넘을 때.
    """
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

    Args:
        session: 현재 세션.
        stream_id: 조정할 스트림.
        adjustment_ms: 새 `manual_adjustment_ms` ms (기존 값에 더하지 않고 바꾼다).

    Returns:
        새 세션 (DB 쓰기는 호출자 `runner.adjust`).

    Raises:
        ValueError: 기준 스트림일 때.
        KeyError: 세션에 `stream_id` 스트림이 없을 때 (`Session.stream`).
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
