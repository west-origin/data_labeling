"""프라이버시 정책 (config/policies/privacy.yaml + config/defaults.yaml privacy).

WP5. `dlp privacy detect|approve|render`와 블러본을 확인하는 모든 소비자(검수 작업, 내보내기, 행동
VLM, 큐레이션)가 `load_policy`로 읽는다.

- `PrivacyPolicy`: 정책 전체. privacy.yaml 값에 defaults.yaml의 `privacy` 절을
  `platform`으로 붙인다.
- `TargetPolicy` / `TrackerPolicy` / `RenderPolicy` / `DetectorSpec`: 각 절의 모델.
- `ReviewReason`: 검수 우선 구간 종류 (privacy.yaml `review_priority`의 원소).

주의:
- 탐지 결과를 정하는 값(문턱·대상·트래커·탐지기 설정·`platform.blur_hold_ms` 등)은
  `pipeline.detection_policy_digest`로 탐지 모델 버전에 들어간다 (ADR 0024). 바꾸면 다시 탐지한다.
  해시는 **파싱된 값**(`model_dump`)으로 계산하므로 YAML 주석·서식 변경은 버전을 바꾸지 않는다.
- 렌더 값(`render`, `platform.render_mode`)은 `runner.render_hash`로 블러본 해시에 들어간다.
- 이 모델들은 dlp_schema 계약이 아니라 JSON Schema(`schemas/`)에 나오지 않는다. 그래도 Field
  description을 바꿀 이유는 없으므로 설명은 `#` 주석으로 보탠다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field

from dlp_schema.common import Contract, OntologyId
from dlp_schema.config import PrivacyConfig, load_config

# 검수 우선 구간 종류. 우선순위(앞이 먼저)는 privacy.yaml review_priority 순서가 정한다.
# - no_detector: 대상에 쓸 수 있는 탐지기가 없어 영상 전체를 사람이 본다.
# - track_gap: 보간·유지(held)한 프레임, 또는 나뉜 트랙 사이 블러 없는 틈(detail="split").
# - disagreement: 대상에 탐지기가 둘 이상인데 일부만 찾은 관측 프레임.
# - low_confidence: 관측 점수가 review_score 미만인 프레임.
# - reflection: 반사면 속 얼굴 블러가 있는 프레임 (항상 확인).
# - trained_model: 배포된 재학습 블러 모델이 낸 구간 (detail=모델 버전).
ReviewReason = Literal[
    "no_detector", "track_gap", "disagreement", "low_confidence", "reflection", "trained_model"
]


class TargetPolicy(Contract):
    """블러 대상 하나(privacy.yaml `targets.<대상>`)의 정책."""

    # 박스 여유: 박스 가로·세로에 각각 (크기 x margin)을 더한다 (양쪽에 절반씩). 0.2면 20% 크게.
    margin: float = Field(ge=0)
    # 이 대상을 찾는 탐지기 이름 (privacy.yaml detectors의 키). 여러 개면 결과를 합친다.
    # 하나도 쓸 수 없으면 영상 전체가 no_detector 검수 구간이 된다.
    detectors: tuple[str, ...]


class TrackerPolicy(Contract):
    """트래커 정책 (privacy.yaml `tracker`). 단위는 ms."""

    # 이전 프레임 박스와 IoU가 이 이상이어야 같은 트랙으로 잇는다 (0 초과 1 이하).
    iou_match: float = Field(gt=0, le=1)
    # 관측이 이 시간(ms) 이하로 끊기면 같은 트랙으로 보고 선형 보간한다. 넘으면 새 트랙.
    max_gap_ms: float = Field(ge=0)
    # 나뉜 트랙 사이(블러 없음) 틈의 앞뒤 블러 간격이 이 이하이면 track_gap(split) 검수 구간.
    split_review_ms: float = Field(
        ge=0, description="같은 대상 트랙 사이 블러 없는 틈이 이 이하이면 검수 우선 구간"
    )


class RenderPolicy(Contract):
    """블러본 렌더 정책 (privacy.yaml `render`). 바꾸면 render_hash가 바뀌어 다시 렌더한다."""

    # 모자이크 블록 한 변의 최소 픽셀. 작은 박스도 블록이 이보다 작아지지 않는다.
    min_block_px: int = Field(ge=1)
    # 박스 짧은 변을 이 개수 이하의 블록으로 나눈다 (클수록 블록이 작아져 덜 가려진다).
    blocks_per_box: int = Field(ge=1)
    # libx264 명목 프레임레이트. 실제 프레임 PTS는 원본을 그대로 쓰므로 VFR도 유지된다.
    encoder_rate: int = Field(gt=0, description="인코더 명목 프레임레이트 (PTS는 원본 그대로)")
    # libx264 CRF (0~51, 낮을수록 화질이 좋고 파일이 크다).
    crf: int = Field(ge=0, le=51)


class DetectorSpec(Contract):
    """탐지기 하나의 설정 (privacy.yaml `detectors.<이름>`).

    `kind`에 따라 쓰는 필드가 다르다. 필요한 값이 비어 있으면 `detectors._required`가 오류를 낸다
    (코드 기본값으로 채우지 않는다). 이 설정은 탐지 모델 버전 해시에 들어간다.
    """

    # 구현 종류. "unavailable"은 이 환경에서 쓸 수 없음을 명시할 때 쓴다 (build_detectors가
    # missing으로 보고). "oracle"은 정책 파일에서 만들지 않고 테스트가 extra로 끼운다.
    kind: Literal["yunet", "opencv_codes", "reflection", "open_vocab", "unavailable", "oracle"]
    # 가중치 이름 (config/models.yaml 키, dlp_models.registry.resolve로 경로·버전을 얻는다).
    model: str | None = Field(default=None, description="config/models.yaml 이름")
    # open_vocab 전용: 토크나이저 이름 (config/models.yaml 키).
    tokenizer: str | None = None
    # reflection 전용: 반사면 영역을 찾는 탐지기 이름과 그 안에서 얼굴을 찾는 탐지기 이름.
    region_detector: str | None = None
    face_detector: str | None = None
    # reflection 전용: 반사면 안 얼굴 탐지 문턱 = detection_threshold x threshold_scale.
    threshold_scale: float | None = None
    # yunet 전용
    # 겹친 얼굴 박스를 지우는 NMS IoU 문턱, NMS 전 후보 수 상한.
    nms_threshold: float | None = Field(default=None, ge=0, le=1)
    top_k: int | None = Field(default=None, ge=1)
    # opencv_codes 전용: 코드는 찾거나 못 찾거나이므로 모든 탐지에 이 고정 신뢰도를 준다.
    score: float | None = Field(default=None, ge=0, le=1, description="고정 신뢰도 (opencv_codes)")
    # open_vocab 전용
    # 추론 간격(ms). 0이면 매 프레임 추론. 사이 프레임은 마지막 결과를 그대로 쓴다.
    frame_stride_ms: int = Field(default=0, ge=0)
    # OWLv2 원점수 문턱 (모델 내부 필터). 이보다 낮은 원점수는 버린다.
    score_threshold: float | None = Field(default=None, ge=0, le=1)
    # 신뢰도 정규화: 신뢰도 = min(1, 원점수 / score_full). 원점수가 낮은 모델이라 늘려 쓴다.
    score_full: float | None = Field(default=None, gt=0, le=1)
    # 질의 문장 → 대상 ID. 같은 대상에 여러 문장을 둘 수 있다. "reflective_surface"는 대상이
    # 아니라 reflection 탐지기용 중간 결과다 (파이프라인이 대상 밖 탐지를 버린다).
    queries: dict[str, str] = Field(default_factory=dict[str, str])


class PrivacyPolicy(Contract):
    """프라이버시 정책 전체. `load_policy`로 만든다 (테스트는 model_copy로 일부를 바꾼다)."""

    # privacy.yaml 형식 버전 (해시에는 들어가지 않는다).
    version: int
    # 블러 대상 ID(온톨로지 프라이버시 사전) → 대상 정책.
    targets: dict[OntologyId, TargetPolicy]
    # 탐지기에 넘기는 신뢰도 문턱 (0~1). 재현율 우선으로 낮게 두고 오탐은 사람이 지운다.
    detection_threshold: float = Field(ge=0, le=1)
    # 관측 점수가 이 미만인 프레임은 low_confidence 검수 구간.
    review_score: float = Field(ge=0, le=1)
    tracker: TrackerPolicy
    render: RenderPolicy
    # 검수 우선 구간 종류의 우선순위 (앞이 먼저, ReviewSegment.priority = 이 목록의 위치).
    review_priority: tuple[ReviewReason, ...]
    # 탐지기 이름 → 설정. targets.*.detectors와 reflection의 region/face_detector가 가리킨다.
    detectors: dict[str, DetectorSpec]
    # config/defaults.yaml에서 채운다
    # (blur_hold_ms: 소실 전후 블러 유지 ms, render_mode: mosaic|solid, strip_audio_in_release,
    #  full_review_exit: 전수 검수 종료 판정 값 등)
    platform: PrivacyConfig


def load_policy(root: Path) -> PrivacyPolicy:
    """저장소 루트에서 privacy.yaml과 defaults.yaml을 함께 읽는다.

    Args:
        root: 저장소 루트 (`config/`가 있는 디렉터리).

    Returns:
        검증된 `PrivacyPolicy`. `platform`에는 defaults.yaml `privacy` 절이 들어간다.

    Raises:
        FileNotFoundError: 설정 파일이 없을 때.
        pydantic.ValidationError: 값이 범위를 벗어나거나 키가 틀렸을 때 (Contract는 모르는 키를
            허용하지 않는다).
    """
    data: Any = yaml.safe_load((root / "config" / "policies" / "privacy.yaml").read_text("utf-8"))
    # defaults.yaml을 먼저 자체 모델로 검증한 뒤 dict로 붙인다 (PrivacyPolicy에서 다시 검증된다)
    data["platform"] = load_config(root / "config" / "defaults.yaml").privacy.model_dump()
    return PrivacyPolicy.model_validate(data)
