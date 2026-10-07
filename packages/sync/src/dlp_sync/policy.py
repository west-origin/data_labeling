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
    confidence_one: float = Field(ge=0, le=1)
    confidence_many: float = Field(ge=0, le=1)


class TapPolicy(Contract):
    double_tap_gap_ms: tuple[float, float]
    max_pulse_ms: float = Field(gt=0)
    merge_gap_ms: float = Field(ge=0)
    threshold_mad: float = Field(gt=0)
    match_tolerance_ms: float = Field(gt=0)
    residual_scale_ms: float = Field(gt=0)


class AudioXcorrPolicy(Contract):
    analysis_rate_hz: int = Field(gt=0)
    coarse_segment_ms: float = Field(gt=0)
    window_ms: float = Field(gt=0)
    windows: int = Field(ge=1)
    refine_ms: float = Field(gt=0)
    min_psr: float = Field(gt=0)
    residual_scale_ms: float = Field(
        gt=0, description="앵커가 2개 이상일 때 신뢰도에 곱하는 exp(-잔차 RMS / 이 값)"
    )


class MotionXcorrPolicy(Contract):
    rate_hz: float = Field(gt=0)
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
