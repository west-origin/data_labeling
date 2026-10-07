"""동기화 정책 (config/policies/sync.yaml)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field

from dlp_schema.common import Contract

MethodName = Literal["qr_slate", "tap_event", "audio_xcorr", "motion_xcorr"]


class SlatePolicy(Contract):
    payload_prefix: str
    search_window_ms: float = Field(gt=0)
    frame_stride: int = Field(ge=1)
    estimate_drift: bool


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


class MotionXcorrPolicy(Contract):
    rate_hz: float = Field(gt=0)
    min_psr: float = Field(gt=0)


class SyncPolicy(Contract):
    version: int
    min_confidence: float = Field(ge=0, le=1)
    max_offset_ms: float = Field(gt=0)
    min_drift_span_ms: float = Field(gt=0)
    max_drift_ppm: float = Field(gt=0)
    methods: dict[str, tuple[MethodName, ...]]
    slate: SlatePolicy
    tap: TapPolicy
    audio_xcorr: AudioXcorrPolicy
    motion_xcorr: MotionXcorrPolicy


def load_policy(path: Path) -> SyncPolicy:
    data: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    return SyncPolicy.model_validate(data)
