"""검수 운영 정책 (config/policies/review.yaml + config/defaults.yaml review 비율).

WP12, ADR 0014. `dlp review plan|assign|queue|qa|quality|create|collect`와 운영 지표
(`dlp_ops.metrics`)가 이 로더를 쓴다.

공개 이름:
- `ErrorType`: 오류 삽입 종류 (boundary_shift, class_swap, blur_deletion).
- `CVAT_KINDS`: CVAT 공간 검수 화면이 보여 주는 작업 라벨 종류 (블러 제외).
- `UnitsPolicy`·`PriorityPolicy`·`SamplingPolicy`·`SeedingPolicy`·`MeasurementPolicy`·
  `ReviewersPolicy`·`MediaPolicy`·`CvatPolicy`: review.yaml의 절별 모델.
- `ReviewOpsPolicy`: 전체 정책 (review.yaml + `ratios` = defaults.yaml `review`).
- `load_policy`: 저장소 루트에서 정책을 읽어 검증한다.

주의점:
- 정책은 YAML을 파싱한 값으로 검증한다. 텍스트 해시를 만들지 않으므로
  YAML 주석은 동작에 영향이 없다.
- 각 모델은 `dlp_schema.common.Contract`(알 수 없는 키 금지, 생성 후 변경 금지)를 따른다. 키를
  바꾸면 YAML과 이 모델을 함께 바꿔야 한다.
- `Field(description=...)` 문자열은 이 패키지 정책 모델이며 `schemas/`에 내보내지 않는다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, model_validator

from dlp_schema.common import Contract
from dlp_schema.config import ReviewConfig, load_config

# 오류 삽입 과제에 넣는 오류 종류 (dlp_review.ops.seeding 참조)
ErrorType = Literal["boundary_shift", "class_swap", "blur_deletion"]


# CVAT 작업 라벨(공간) 검수 화면이 보여 주는 라벨 종류. 블러(blur_track)는 프라이버시 검수 단위라
# 여기에 넣지 않는다. dlp_review.cvat.CVAT_KINDS(블러 포함, 변환기가 다루는 종류)와 다르다.
CVAT_KINDS = ("box_track", "keypoint_track")


class UnitsPolicy(Contract):
    """검수 단위에 넣을 라벨 종류 (review.yaml `units`).

    검수 도구가 화면에 보여 주는 종류만 허용한다. 보여 주지 않는 종류를 표본으로 뽑으면 검수자가
    판정할 수 없어 표본 판정이 끝나지 않는다.
    """

    # 영상 스트림별 공간 단위(CVAT)에 넣을 종류
    spatial: tuple[str, ...]
    # 세션 단위 시간 단위(Label Studio)에 넣을 종류
    temporal: tuple[str, ...]

    @model_validator(mode="after")
    def _shown_by_tools(self) -> UnitsPolicy:
        """spatial ⊆ `CVAT_KINDS`, temporal ⊆ `LS_KINDS`인지 검사한다. 아니면 ValueError."""
        # 함수 안에서 import한다 (정책 모듈을 읽을 때 변환기 모듈까지 끌어오지 않는다)
        from dlp_review.labelstudio import LS_KINDS

        extra = (set(self.spatial) - set(CVAT_KINDS)) | (set(self.temporal) - set(LS_KINDS))
        if extra:
            raise ValueError(f"검수 도구가 보여 주지 않는 라벨 종류: {sorted(extra)}")
        return self


class PriorityPolicy(Contract):
    """우선순위 큐 (review.yaml `priority`). 의미는 `dlp_review.ops.priority` 참조."""

    # 이보다 신뢰도가 낮은 모델 라벨은 low_confidence 사유 (0~1)
    low_confidence: float = Field(ge=0, le=1)
    # 서로 다른 모델 버전의 박스가 같은 키프레임 시각에 이 IoU 이상 겹치는데 클래스가 다르면 불일치
    disagreement_iou: float = Field(gt=0, le=1)
    # 서로 다른 모델 버전의 시간 구간이 짧은 쪽 길이 대비 이 비율 이상 겹치는데 분류가 다르면 불일치
    disagreement_overlap: float = Field(gt=0, le=1)
    # 장갑 세션에서 접촉 손 상태 라벨의 신뢰도가 이 값 이하이면 contact_mismatch 사유
    contact_mismatch_max_confidence: float = Field(ge=0, le=1)
    # 사유(ReviewReason 값) → 가중치. 없는 사유는 0으로 본다
    weights: dict[str, float]
    # 사유가 하나도 없는 단위의 기본 우선순위
    routine_priority: float = Field(ge=0)


class SamplingPolicy(Contract):
    """높은 신뢰도 라벨 표본 검수 (review.yaml `sampling`). `dlp_review.ops.sampling` 참조."""

    # 이 신뢰도 이상인 미검수 모델 라벨만 묶음(lot)에 넣는다
    high_confidence: float = Field(ge=0, le=1)
    # 묶음 크기 대비 표본 비율 (올림)
    ratio: float = Field(gt=0, le=1)
    # 묶음당 최소 표본 수 (묶음이 더 작으면 전수)
    min_sample: int = Field(ge=1)
    # 표본 중 사람이 고치거나 지운 비율이 이 값 이하이면 합격
    max_defect_ratio: float = Field(ge=0, le=1)


class SeedingPolicy(Contract):
    """오류 삽입 과제 (review.yaml `seeding`). `dlp_review.ops.seeding` 참조."""

    # 과제 하나에 넣을 오류 수 (후보가 모자라면 덜 넣는다)
    errors_per_task: int = Field(ge=1)
    # 넣을 오류 종류. 순서대로 돌아가며 넣는다
    types: tuple[ErrorType, ...] = Field(min_length=1)
    # boundary_shift가 경계를 옮기는 거리 범위 [최소, 최대] ms (양끝 포함)
    boundary_shift_ms: tuple[int, int]
    # 검수자가 경계를 원래 값에서 이 ms 안으로 되돌리면 발견
    detect_tolerance_ms: int = Field(ge=0)
    # 다시 그린 블러가 뺀 블러 구간과 이 비율 이상 겹치면 발견 (뺀 블러 길이 기준)
    blur_overlap: float = Field(gt=0, le=1)


class MeasurementPolicy(Contract):
    """검수 품질 측정 (review.yaml `measurement`). `dlp_review.ops.measure` 참조."""

    # 경계 일치 허용 오차 (ms)
    tolerance_ms: int = Field(ge=0)
    # 두 라벨 묶음의 구간을 짝지을 때 쓰는 최소 시간 IoU
    match_iou: float = Field(gt=0, le=1)


class ReviewersPolicy(Contract):
    """검수자 역할 (review.yaml `reviewers`)."""

    # 선임 검수자 (QA 배정 대상). 비어 있으면 QA 배정 담당자가 None이 된다
    senior: tuple[str, ...] = ()
    privacy: tuple[str, ...] = Field(default=(), description="원본 접근 권한자 (블러 검수)")


class MediaPolicy(Contract):
    """검수 화면용 매체 (라벨러 워터마크 영상, 시계열 CSV)."""

    # 워터마크 글씨 불투명도 (0, 1]
    watermark_opacity: float = Field(gt=0, le=1)
    # 워터마크 영상 libx264 CRF (0~51, 낮을수록 고화질·큰 파일)
    watermark_crf: int = Field(ge=0, le=51)
    encoder_rate: int = Field(gt=0, description="인코더 명목 프레임레이트 (PTS는 원본 그대로)")
    # Label Studio 시계열 CSV 격자 주파수 (Hz, 마스터 타임라인)
    timeseries_rate_hz: float = Field(gt=0)


class CvatPolicy(Contract):
    """CVAT 계정 연결과 블러 검수 화면 안내."""

    users: dict[str, str] = Field(
        default_factory=dict[str, str],
        description="dlp 검수자 ID → CVAT 사용자 이름. 블러 검수 담당자는 반드시 있어야 한다",
    )
    privacy_issue_limit: int = Field(
        ge=0, description="블러 검수 작업에 CVAT 이슈로 남기는 검수 우선 구간 최대 수"
    )

    @model_validator(mode="after")
    def _one_to_one(self) -> CvatPolicy:
        """한 CVAT 계정을 두 검수자에게 줄 수 없다 (웹훅의 job 담당자로 검수자를 정하므로)."""
        names = list(self.users.values())
        if len(names) != len(set(names)):
            raise ValueError("cvat.users: 한 CVAT 사용자를 두 검수자에게 줄 수 없습니다")
        return self


class ReviewOpsPolicy(Contract):
    """검수 운영 정책 전체. `load_policy`가 만든다."""

    # 정책 파일 형식 버전
    version: int
    media: MediaPolicy
    units: UnitsPolicy
    priority: PriorityPolicy
    sampling: SamplingPolicy
    measurement: MeasurementPolicy
    seeding: SeedingPolicy
    reviewers: ReviewersPolicy
    cvat: CvatPolicy
    ratios: ReviewConfig  # config/defaults.yaml review


def load_policy(root: Path) -> ReviewOpsPolicy:
    """저장소 루트(root)에서 검수 운영 정책을 읽는다.

    `config/policies/review.yaml`을 읽고, 배정 비율은 `config/defaults.yaml`의 `review` 절을
    `ratios` 키로 끼워 넣어 함께 검증한다.

    예외: 파일이 없으면 OSError, 값이 범위를 벗어나거나 키가 맞지 않으면 pydantic ValidationError.
    부작용 없음 (파일 읽기만).
    """
    data: Any = yaml.safe_load((root / "config" / "policies" / "review.yaml").read_text("utf-8"))
    data["ratios"] = load_config(root / "config" / "defaults.yaml").review.model_dump()
    return ReviewOpsPolicy.model_validate(data)
