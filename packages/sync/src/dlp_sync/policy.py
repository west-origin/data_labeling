"""동기화 정책 (config/policies/sync.yaml)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, field_validator

from dlp_schema.common import Contract
from dlp_schema.session import StreamKind

MethodName = Literal["qr_slate", "tap_event", "audio_xcorr", "motion_xcorr"]


class SlatePolicy(Contract):
    payload_prefix: str
    search_window_ms: float = Field(gt=0)
    frame_stride: int = Field(ge=1)
    estimate_drift: bool
    min_drift_span_ms: float = Field(
        gt=0, description="슬레이트 앵커 사이가 이보다 길 때만 드리프트를 추정한다 (프레임 양자화)"
    )
    confidence_one: float = Field(ge=0, le=1)
    confidence_many: float = Field(ge=0, le=1)
    residual_scale_ms: float = Field(
        gt=0, description="신뢰도 *= exp(-max(0, 오차 상한 - 양자화) / 이 값)"
    )
    refine_with_audio: bool = Field(
        description="두 영상에 오디오가 있으면 슬레이트 맞춤을 출발점으로 오디오 상관으로 다듬는다"
    )


class TapPolicy(Contract):
    double_tap_gap_ms: tuple[float, float]
    max_pulse_ms: float = Field(gt=0)
    merge_gap_ms: float = Field(ge=0)
    threshold_mad: float = Field(gt=0)
    match_tolerance_ms: float = Field(
        gt=0,
        description="기준 두드림에서 |Δt|만큼 떨어진 두드림은 max_drift_ppm·|Δt|를 더 허용한다",
    )
    residual_scale_ms: float = Field(gt=0)


class AudioXcorrPolicy(Contract):
    analysis_rate_hz: int = Field(gt=0)
    coarse_segment_ms: float = Field(gt=0)
    window_ms: float = Field(gt=0)
    windows: int = Field(ge=1)
    refine_ms: float = Field(
        gt=0,
        description=(
            "정밀 탐색 범위. 거친 추정 지점에서 |Δt|만큼 떨어진 창은 max_drift_ppm·|Δt|를 더 본다"
        ),
    )
    min_psr: float = Field(gt=0)
    residual_scale_ms: float = Field(
        gt=0, description="앵커가 2개 이상일 때 신뢰도에 곱하는 exp(-잔차 RMS / 이 값)"
    )


class MotionXcorrPolicy(Contract):
    rate_hz: float = Field(gt=0)
    window_ms: float = Field(gt=0, description="창별 정밀 탐색의 창 길이")
    refine_ms: float = Field(
        gt=0, description="창별 정밀 탐색 범위 (전체 상관 추정 ± 이 값 + max_drift_ppm·녹화 길이)"
    )
    window_min_psr: float = Field(gt=0, description="창별 정밀 탐색에서 앵커로 쓸 창의 PSR 하한")
    min_psr: float = Field(gt=0)
    residual_scale_ms: float = Field(
        gt=0, description="앵커가 2개 이상일 때 신뢰도에 곱하는 exp(-잔차 RMS / 이 값)"
    )


class GloveSignalPolicy(Contract):
    """장갑 Parquet에서 동기화 신호(압력 합)로 쓸 채널. 시각 열은 이름과 무관하게 뺀다."""

    pressure_prefixes: tuple[str, ...] = Field(min_length=1)


class SyncPolicy(Contract):
    version: int
    min_confidence: float = Field(ge=0, le=1)
    max_offset_ms: float = Field(gt=0)
    min_drift_span_ms: float = Field(gt=0)
    max_drift_ppm: float = Field(gt=0)
    methods: dict[StreamKind, tuple[MethodName, ...]]
    glove: GloveSignalPolicy
    slate: SlatePolicy
    tap: TapPolicy
    audio_xcorr: AudioXcorrPolicy
    motion_xcorr: MotionXcorrPolicy

    @field_validator("methods")
    @classmethod
    def _no_reference(
        cls, value: dict[StreamKind, tuple[MethodName, ...]]
    ) -> dict[StreamKind, tuple[MethodName, ...]]:
        if StreamKind.BODYCAM in value:
            raise ValueError("기준 스트림(bodycam)은 동기화 방법을 가질 수 없습니다")
        return value


def load_policy(path: Path) -> SyncPolicy:
    data: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    return SyncPolicy.model_validate(data)
