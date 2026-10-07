"""동기화 정책 (config/policies/sync.yaml, WP4, ADR 0004·0028).

YAML을 Pydantic 모델로 검증해 읽는다. 각 절의 의미·단위는 `sync.yaml`의 주석에 있다.
이 정책 전체의 해시는 모델 버전에 넣지 않는다 (동기화 결과는 라벨이 아니라 DB 스트림 필드에
쓴다). 단 `glove.pressure_prefixes` 값은 접촉 프리라벨의 모델 버전에 들어간다.

- `SyncPolicy`: 최상위 (공통 허용치 + 스트림 종류별 방법 순서 + 방법별 절)
- `SlatePolicy`, `TapPolicy`, `AudioXcorrPolicy`, `MotionXcorrPolicy`: 방법별 절
- `GloveSignalPolicy`: 장갑 압력 채널 선택
- `load_policy`: 파일 → `SyncPolicy`

`Contract`는 알 수 없는 키를 거부하므로(extra="forbid") YAML 키 오타는 적재 시 오류가 된다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, field_validator

from dlp_schema.common import Contract
from dlp_schema.session import StreamKind

# 정책 `methods`에 쓸 수 있는 방법 이름. `pipeline.METHOD_ENUM`이 계약의 `SyncMethod`로 바꾼다
MethodName = Literal["qr_slate", "tap_event", "audio_xcorr", "motion_xcorr"]


class SlatePolicy(Contract):
    """`sync.yaml slate`: QR 슬레이트 검출과 신뢰도."""

    # QR 내용이 이 접두사로 시작해야 슬레이트로 본다 (다른 QR·바코드 무시)
    payload_prefix: str
    # 영상 앞뒤 이 길이(ms) 안에서만 찾는다
    search_window_ms: float = Field(gt=0)
    # 이 간격(프레임)마다 QR 디코딩. 찾으면 건너뛴 프레임을 되짚는다
    frame_stride: int = Field(ge=1)
    # False면 슬레이트로는 늘 오프셋만 맞춘다
    estimate_drift: bool
    min_drift_span_ms: float = Field(
        gt=0, description="슬레이트 앵커 사이가 이보다 길 때만 드리프트를 추정한다 (프레임 양자화)"
    )
    # 슬레이트 1개 / 2개 이상일 때 기본 신뢰도 (오차 상한이 양자화 안일 때)
    confidence_one: float = Field(ge=0, le=1)
    confidence_many: float = Field(ge=0, le=1)
    residual_scale_ms: float = Field(
        gt=0, description="신뢰도 *= exp(-max(0, 오차 상한 - 양자화) / 이 값)"
    )
    refine_with_audio: bool = Field(
        description="두 영상에 오디오가 있으면 슬레이트 맞춤을 출발점으로 오디오 상관으로 다듬는다"
    )


class TapPolicy(Contract):
    """`sync.yaml tap`: 두 번 두드림 검출과 짝짓기."""

    # 두 번 두드림 사이 간격의 허용 범위 [하한, 상한] ms
    double_tap_gap_ms: tuple[float, float]
    # 이보다 긴 사건(ms)은 두드림이 아니다
    max_pulse_ms: float = Field(gt=0)
    # 이 간격(ms) 안의 문턱 초과 샘플은 한 사건
    merge_gap_ms: float = Field(ge=0)
    # 문턱 = 중앙값 + threshold_mad * MAD
    threshold_mad: float = Field(gt=0)
    match_tolerance_ms: float = Field(
        gt=0,
        description="기준 두드림에서 |Δt|만큼 떨어진 두드림은 max_drift_ppm·|Δt|를 더 허용한다",
    )
    # 신뢰도 *= exp(-잔차 RMS / 이 값)
    residual_scale_ms: float = Field(gt=0)


class AudioXcorrPolicy(Contract):
    """`sync.yaml audio_xcorr`: 오디오 상호상관 (거친 탐색 + 창별 정밀 탐색)."""

    # 거친 탐색 샘플레이트 Hz (16 kHz 오디오를 정수배로 줄인다)
    analysis_rate_hz: int = Field(gt=0)
    # 거친 탐색에 쓰는 기준 오디오 가운데 구간 길이 ms
    coarse_segment_ms: float = Field(gt=0)
    # 정밀 탐색 창 길이 ms
    window_ms: float = Field(gt=0)
    # 정밀 탐색 창 개수 (= 최대 앵커 수)
    windows: int = Field(ge=1)
    refine_ms: float = Field(
        gt=0,
        description=(
            "정밀 탐색 범위. 거친 추정 지점에서 |Δt|만큼 떨어진 창은 max_drift_ppm·|Δt|를 더 본다"
        ),
    )
    # 창 PSR 하한. 미만인 창은 앵커로 쓰지 않는다. 신뢰도 = 1 - min_psr / PSR 중앙값
    min_psr: float = Field(gt=0)
    residual_scale_ms: float = Field(
        gt=0, description="앵커가 2개 이상일 때 신뢰도에 곱하는 exp(-잔차 RMS / 이 값)"
    )


class MotionXcorrPolicy(Contract):
    """`sync.yaml motion_xcorr`: 바디캠 IMU 가속도와 장갑 압력의 상호상관."""

    # 두 신호를 리샘플할 공통 격자 레이트 Hz
    rate_hz: float = Field(gt=0)
    window_ms: float = Field(gt=0, description="창별 정밀 탐색의 창 길이")
    refine_ms: float = Field(
        gt=0, description="창별 정밀 탐색 범위 (전체 상관 추정 ± 이 값 + max_drift_ppm·녹화 길이)"
    )
    window_min_psr: float = Field(gt=0, description="창별 정밀 탐색에서 앵커로 쓸 창의 PSR 하한")
    # 전체 상관 PSR 하한 (신뢰도 = 1 - min_psr / PSR)
    min_psr: float = Field(gt=0)
    residual_scale_ms: float = Field(
        gt=0, description="앵커가 2개 이상일 때 신뢰도에 곱하는 exp(-잔차 RMS / 이 값)"
    )


class GloveSignalPolicy(Contract):
    """장갑 Parquet에서 동기화 신호(압력 합)로 쓸 채널. 시각 열은 이름과 무관하게 뺀다."""

    # 열 이름 접두사 목록 (하나 이상). 접촉 프리라벨·검수 동기 재생도 같은 값을 쓴다
    pressure_prefixes: tuple[str, ...] = Field(min_length=1)


class SyncPolicy(Contract):
    """`config/policies/sync.yaml` 전체."""

    # 정책 형식 버전
    version: int
    # 이 신뢰도 이상인 첫 방법을 채택. 모두 미달이면 unsynced(또는 이전 자동 결과 유지)
    min_confidence: float = Field(ge=0, le=1)
    # 탐색할 최대 오프셋 |ms| (양방향). 넘는 추정은 버린다
    max_offset_ms: float = Field(gt=0)
    # 슬레이트 외 방법에서 드리프트를 추정할 최소 앵커 범위 ms
    min_drift_span_ms: float = Field(gt=0)
    # 드리프트 상한 ppm. 넘으면 FitError. 두드림 허용 오차·상관 탐색 폭 확장에도 쓴다
    max_drift_ppm: float = Field(gt=0)
    # 스트림 종류 → 시도할 방법 순서 (대체 순서). 기준(bodycam)은 넣을 수 없다
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
        """기준 스트림(bodycam)에 방법을 주면 거부한다 (기준은 동기화 대상이 아니다).

        Raises:
            ValueError: `methods`에 `bodycam` 키가 있을 때.
        """
        if StreamKind.BODYCAM in value:
            raise ValueError("기준 스트림(bodycam)은 동기화 방법을 가질 수 없습니다")
        return value


def load_policy(path: Path) -> SyncPolicy:
    """YAML 파일을 읽어 `SyncPolicy`로 검증한다.

    Args:
        path: 보통 `<저장소>/config/policies/sync.yaml`.

    Raises:
        pydantic.ValidationError: 키가 없거나, 모르는 키가 있거나, 값이 범위를 벗어날 때.
    """
    data: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    return SyncPolicy.model_validate(data)
