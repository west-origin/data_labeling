"""검수 운영 정책 (config/policies/review.yaml + config/defaults.yaml review 비율).

WP12, ADR 0014. `dlp review plan|assign|qa|quality|create|collect|serve`와 운영 지표
(`dlp ops weekly`, `dlp ops audit-report`의 `reviewers.privacy`)가 이 정책을 쓴다.
라이브러리 로더의 기본 파일은 review.yaml이다. CLI는 `dlp_cli.review_config`에서
`DLP_REVIEW_POLICY_FILE`을 선택한 뒤 이 로더에 명시적으로 넘기고, 읽은 정책을 라이브러리
함수와 `tasks.ReviewSetup`(`media`·`cvat_config`)에 넘긴다(매체 회수 전용 설정은 정책 값을 쓰지
않아 읽지 않는다). 정책을 받지 않은 라이브러리 경로
(`ReviewSetup`의 지연 로더, `ops.results.finish_assignment`의 기본값)는 환경 변수를 보지 않고
기본 파일을 읽는다.

공개 이름:
- `ErrorType`: 오류 삽입 종류 (boundary_shift, class_swap, blur_deletion).
- `CVAT_KINDS`: CVAT 공간 검수 화면이 보여 주는 작업 라벨 종류 (블러 제외).
- `SIGV4_MAX_EXPIRES_S`: 서명 URL 유효 시간 상한 (`media.signed_url_ttl_s` 검증).
- `UnitsPolicy`·`PriorityPolicy`·`SamplingPolicy`·`SeedingPolicy`·`MeasurementPolicy`·
  `ReviewersPolicy`·`MediaPolicy`·`CvatPolicy`·`LabelStudioPolicy`·`WebhookPolicy`·`OrphanPolicy`·
  `AiConfidenceFloor`·`AiReviewDisplayPolicy`(AI 1차 검수 판정 사용 방식, WP17.6)·
  `AiCalibrationPolicy`·`AiCalibrationFit`·`AiCalibrationRef`(판정 표시 보정, ADR 0062)·
  `VerifyPolicy`(완료 판정에서 뺄 자동 라벨, ADR 0050): review.yaml의 절별 모델.
- `AutomaticLabel`: 사람 검수 대상이 아닌 자동 라벨 분류 이름 (camera_pose, relation, coverage).
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

from dlp_media.geometry import TtaSpec
from dlp_schema.aireview import AiVersion
from dlp_schema.common import Contract
from dlp_schema.config import ReviewConfig, load_config

# 오류 삽입 과제에 넣는 오류 종류 (dlp_review.ops.seeding 참조)
ErrorType = Literal["boundary_shift", "class_swap", "blur_deletion"]

# 사람 검수 대상이 아닌 자동 라벨 분류 (ADR 0050). camera_pose: 모델 출처 카메라 자세 궤적
# (`dlp_schema.trajectory_follow.is_camera_pose`, entity camera). relation·coverage: 규칙 엔진
# (`dlp relations run`)이 손 상태·3D 궤적에서 도출하는 모델 출처 관계·커버리지 (ADR 0011 개정)
AutomaticLabel = Literal["camera_pose", "relation", "coverage"]


# CVAT 작업 라벨(공간) 검수 화면이 보여 주는 라벨 종류. 블러(blur_track)는 프라이버시 검수 단위라
# 여기에 넣지 않는다. dlp_review.cvat.CVAT_KINDS(블러 포함, 변환기가 다루는 종류)와 다르다.
# mask_track은 모양 모드 마스크로 보인다 (ADR 0059)
CVAT_KINDS = ("box_track", "keypoint_track", "mask_track")

# 서명 URL 유효 시간의 상한(초, 7일). 정책 값이 아니라 S3 SigV4 서명 규격의 상한이다 (boto3 기본
# 설정의 개발 SeaweedFS는 만료 시각을 담는 SigV2 URL을 만들어 상한이 없지만 같은 상한을 둔다)
SIGV4_MAX_EXPIRES_S = 7 * 24 * 3600


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

    # 선임 검수자 (QA 배정 대상). 비어 있거나 원래 담당자뿐이면 QA 배정 담당자가 None이 된다.
    # 담당자 없는 배정은 작업을 만들지 않는다. 여기에 원래 담당자가 아닌 선임을 넣은 뒤
    # `dlp review assign <배정> --assignee <그 선임>`으로 정한다 (F-REV-07, QA 담당자는 이 목록에
    # 있고 짝 배정 담당자가 아니어야 해서, 목록을 고치지 않으면 assign이 거부된다)
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
    # Label Studio 작업에 박는 영상·시계열 서명 URL의 유효 시간 (초, F-REV-08). S3 SigV4 서명의
    # 상한이 7일(604800초)이다. 열린 작업은 `dlp review refresh-urls`로 이보다 자주 다시 서명한다
    signed_url_ttl_s: int = Field(gt=0, le=SIGV4_MAX_EXPIRES_S)


class CvatPolicy(Contract):
    """CVAT 계정 연결과 블러 검수 화면 안내."""

    users: dict[str, str] = Field(
        default_factory=dict[str, str],
        description="dlp 검수자 ID → CVAT 사용자 이름. 블러 검수 담당자는 반드시 있어야 한다",
    )
    privacy_issue_limit: int = Field(
        ge=0, description="블러 검수 작업에 CVAT 이슈로 남기는 검수 우선 구간 최대 수"
    )
    # 오탐 의심 힌트 이슈 최대 수 (블러·라벨링 작업 각각, ADR 0055). 절이 없으면 0 (달지 않음)
    hint_issue_limit: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _one_to_one(self) -> CvatPolicy:
        """한 CVAT 계정을 두 검수자에게 줄 수 없다 (웹훅의 job 담당자로 검수자를 정하므로)."""
        names = list(self.users.values())
        if len(names) != len(set(names)):
            raise ValueError("cvat.users: 한 CVAT 사용자를 두 검수자에게 줄 수 없습니다")
        return self


class LabelStudioPolicy(Contract):
    """Label Studio 계정 연결 (review.yaml `label_studio`, F-REV-06).

    수집·웹훅은 작업 담당자의 Label Studio 계정이 만들고 마지막으로 고친 주석만 받는다. 연결이
    없는 담당자에게는 Label Studio 시간 검수 작업을 만들지 않는다
    (`dlp_review.tasks.require_label_studio_user`). 절을 빼면 연결이 비어 있는 것으로 본다. 이 절만
    선택이다: 이전에 복사한 정책 파일은 그 뒤 더한 필수 키(`orphans.min_age_s`,
    `media.signed_url_ttl_s`)가 없으면 정책을 읽는 모든 명령에서 실패하므로 기본 review.yaml 값을
    옮겨 적는다.
    """

    users: dict[str, str] = Field(
        default_factory=dict[str, str],
        description="dlp 검수자 ID → Label Studio 사용자 이메일. 시간 검수 담당자는 반드시 둔다",
    )

    @model_validator(mode="after")
    def _one_to_one(self) -> LabelStudioPolicy:
        """값은 이메일이고, 한 계정을 두 검수자에게 줄 수 없다 (이메일 대소문자 무시)."""
        bad = sorted(k for k, v in self.users.items() if "@" not in v)
        if bad:
            raise ValueError(f"label_studio.users: 이메일이 아닌 값 ({bad})")
        emails = [v.casefold() for v in self.users.values()]
        if len(emails) != len(set(emails)):
            raise ValueError("label_studio.users: 한 계정을 두 검수자에게 줄 수 없습니다")
        return self


class WebhookPolicy(Contract):
    """웹훅 HTTP 경계. 정책값으로 메모리·연결·동시 처리량을 제한한다."""

    bind_host: str = Field(min_length=1)
    max_body_bytes: int = Field(gt=0)
    request_timeout_s: float = Field(gt=0)
    max_workers: int = Field(gt=0)
    retry_after_s: int = Field(gt=0)


class OrphanPolicy(Contract):
    """DB 기록 없이 남은 도구 작업·라벨링 버킷 사본(고아) 정리 (review.yaml `orphans`, F-REV-15).

    `dlp review revoke --orphans`가 쓴다. 작업 생성은 도구 작업·사본을 먼저 만들고 DB에 나중에
    커밋하므로, 막 만든 것은 아직 DB에 없다. 이 나이보다 어린 것은 고아로 보지 않는다.
    """

    # 도구·저장소가 기록한 생성 시각에서 이만큼(초) 지나야 고아 후보다 (시각을 모르면 후보 아님)
    min_age_s: int = Field(ge=0)


class AiConfidenceFloor(Contract):
    """판정 값별 최소 확신도. 이보다 낮은(또는 확신도가 없는데 0보다 큰) 판정은 화면에 따로
    표시하지 않는다 (노출 기록 `hidden`). ok는 구간으로 표시하지 않으므로 없다."""

    wrong: float = Field(default=0.0, ge=0, le=1)
    suspect: float = Field(default=0.0, ge=0, le=1)
    unsure: float = Field(default=0.0, ge=0, le=1)


class AiCalibrationRef(Contract):
    """쓸 보정 산출물 (`ai_review.calibration.artifact`).

    path: 저장소 루트 기준 파일 경로. sha256: 그 파일의 sha256 (다르면 검수 CLI가 멈춘다, 산출물
    고정).
    """

    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class AiCalibrationFit(Contract):
    """보정 학습 설정 (`ai_review.calibration.fit`, `dlp aireview calibrate`, ADR 0062).

    - verdicts: 보정하는 판정 값 (wrong·suspect만. ok·unsure는 보정하지 않는다).
    - min_effective_samples: 칸(판정 / 종류 / 클래스)의 대조군 유효 표본 수가 이보다 적으면 그 칸을
      만들지 않고 더 넓은 칸이나 기본 동작(보정 없음)을 쓴다.
    - hide_below_precision: 지적 정밀도의 한쪽 95% 상한이 이보다 낮으면 (확실히 쓸모없는 지적이면)
      그 칸의 지적을 화면에 따로 표시하지 않는다.
    - prior_strength: 표시 순서 가중(수축 정밀도)의 사전 표본 수.
    - window_days: `--since`가 없을 때 기간 (일).
    """

    verdicts: tuple[Literal["wrong", "suspect"], ...] = ("suspect", "wrong")
    min_effective_samples: float = Field(default=30.0, ge=1)
    hide_below_precision: float = Field(default=0.15, ge=0, le=1)
    prior_strength: float = Field(default=10.0, ge=0)
    window_days: int = Field(default=90, ge=1)


class AiCalibrationPolicy(Contract):
    """AI 판정 표시 보정 (`ai_review.calibration`, ADR 0062). artifact가 없으면 보정하지 않는다."""

    artifact: AiCalibrationRef | None = None
    fit: AiCalibrationFit = Field(default_factory=AiCalibrationFit)


class AiReviewDisplayPolicy(Contract):
    """AI 1차 검수 판정을 검수 운영에 쓰는 방식 (review.yaml `ai_review`, WP17.6, ADR 0040).

    판정은 참고용이고 사람 검수를 늘리는 쪽(우선순위, 표본 제외, 늦은 지적의 재검수)으로만 쓴다.
    `dlp_review.ai_display`가 읽는다. 절이 없으면 끔(off)이다 (코드 기본값. 저장소 설정은 장비 도착
    전 advisory, 파일럿 WP17.9 뒤 required로 바꾼다).
    """

    # off: AI 표를 읽지 않는다 (검수 화면의 AI 컨트롤은 그대로 두고 채우지 않는다).
    # advisory: 작업을 만들 때의 유효한 판정만 얹는다. required: advisory + `dlp review plan`이 지금
    # 버전의 완료된 판정 실행이 없는 세션의 배정을 미룬다 (건너뜀 기록·max_wait_h 뒤 진행).
    mode: Literal["off", "advisory", "required"] = "off"
    # required: prelabeled 전이 뒤 이 시간(시)이 지나면 waived:timeout을 남기고 배정한다
    max_wait_h: float = Field(default=24.0, gt=0)
    # 표시·증가에 쓸 판정 버전 (`aireview-<digest>+vlm-<model>`). 비면 aireview.yaml 지금 버전만
    # (CLI가 계산해 넘긴다, 검수 패키지는 판정 정책을 읽지 않는다)
    versions: tuple[AiVersion, ...] = ()
    # 판정을 보일 배정 방식. 블라인드·오류 삽입·QA는 넣을 수 없다 (측정 편향, ADR 0040 6절).
    # double은 짝 표준 배정의 대조군 결정을 따른다
    show_modes: tuple[Literal["standard", "double"], ...] = ("standard", "double")
    # 대조군(holdout) 비율: 판정을 숨겨 편향 없는 품질 측정에 쓴다 (결정 0.1)
    holdout_ratio: float = Field(default=0.1, ge=0, le=1)
    holdout_seed: int = 0
    show_min_confidence: AiConfidenceFloor = Field(default_factory=AiConfidenceFloor)
    # 같은 10분(라벨 시각 기준) 안에 따로 표시하는 판정 수 상한 (잡음 상한)
    max_flags_per_10min: int = Field(default=30, ge=0)
    # 긴 라벨의 표시·우선순위 구간 상한 (ms): 이보다 긴 라벨은 근거 시각 주변만 표시한다
    span_cap_ms: int = Field(default=10_000, gt=0)
    # 긴 라벨의 Label Studio AI 구간 폭 (ms, 근거 시각 ±절반)
    evidence_window_ms: int = Field(default=2_000, gt=0)
    # CVAT 작업 하나에 다는 AI 이슈 상한 (나머지는 노출 기록 hidden)
    cvat_issue_limit: int = Field(default=200, ge=0)
    # 표본 검수 묶음에서 빼고 반드시 검수에 넣을 판정 (마무리 보호도 같은 값). ok는 넣을 수 없다
    # (AI의 ok는 아무것도 줄이지 않는다, I9)
    sampling_exclude: tuple[Literal["wrong", "suspect", "unsure"], ...] = (
        "wrong",
        "suspect",
        "unsure",
    )
    # 맥락 라벨까지 지금도 현재여야 판정을 유효로 본다 (결정: false). ID가 현재인지만 보므로 맥락
    # 라벨의 승인(같은 ID의 검수 상태 변화)은 true여도 무효로 만들지 않는다
    strict_context: bool = False
    # 판정 표시 보정 (ADR 0062): 사람 판정 기록(대조군)으로 학습한 산출물이 의심·틀림 표시만 거른다.
    # 증가 프로필(우선순위·표본 제외·마무리 보호)과 ok는 바꾸지 않는다 (I9)
    calibration: AiCalibrationPolicy = Field(default_factory=AiCalibrationPolicy)


class ObjectHintsPolicy(Contract):
    """객체·도구 오탐 의심 힌트 (review.yaml `object_hints`, ADR 0055, `dlp_review.object_hints`).

    라벨을 바꾸지 않고 검수 화면에 표시만 한다. time_tolerance_ms: 손·머리 키프레임 시각 허용 오차.
    margin: 박스 여유 비율. wearer_share: 사람 박스 키프레임 중 착용자 손목을 담고 머리 관절이 없는
    비율이 이 이상이면 wearer_body. tta_min_support / tta: 도구 TTA (OWLv2를 변환 화면에 다시
    돌린다).
    """

    time_tolerance_ms: int = Field(ge=0)
    margin: float = Field(ge=0)
    wearer_share: float = Field(gt=0, le=1)
    tta_min_support: float = Field(ge=0, le=1)
    tta: TtaSpec


class AiCropsPolicy(Contract):
    """잘라 낸 박스 AI 확인 (review.yaml `ai_crops`, ADR 0056, `dlp_aireview.crops`). 힌트만 낸다.

    margin: 박스 여유 비율 (블러 박스는 주변 맥락이 필요해 크게). max_side: 보낼 조각 긴 변(px).
    jpeg_quality: JPEG 품질. min_confidence: no 답의 신뢰도 하한. frames_per_track: 트랙마다 물을
    키프레임 수. max_tracks: 스트림마다 물을 트랙 수 상한 (블러·도구 각각, 신뢰도 낮은 것부터).
    targets: 물을 블러 대상 ID (빈 목록이면 블러는 묻지 않는다). tools: 도구 박스도 물을지.
    """

    margin: float = Field(ge=0)
    max_side: int = Field(ge=32)
    jpeg_quality: int = Field(ge=10, le=100)
    min_confidence: float = Field(ge=0, le=1)
    frames_per_track: int = Field(ge=1)
    max_tracks: int = Field(ge=0)
    targets: tuple[str, ...] = ()
    tools: bool = True


class VerifyPolicy(Contract):
    """`dlp review verify` 완료 판정 (ADR 0050).

    automatic: 사람 검수 대상이 아닌 자동 라벨 분류. 검수 전이어도 완료를 막지 않는다 (검수 상태를
    바꾸지 않는다. 기본 내보내기에는 여전히 들어가지 않고, 그 값에서 계산한 라벨을 내보내면
    manifest가 automatic_unreviewed로 밝힌다).
    require_current_3d: 참이면 출처 2D가 바뀌어 지운(또는 낡은) 3D 궤적을 고친 2D로 다시 만들기
        전에는 완료를 막는다 (`dlp prelabel run`을 안내한다, F-COE-30). 깊이 모델이 없어 다시 만들
        수 없는 운영에서만 끈다.
    """

    automatic: tuple[AutomaticLabel, ...] = ("camera_pose", "relation", "coverage")
    require_current_3d: bool = True


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
    # 절이 없으면 연결이 비어 있다 (Label Studio 작업 생성·수집이 계정 연결 없음으로 거부된다)
    label_studio: LabelStudioPolicy = Field(default_factory=LabelStudioPolicy)
    webhook: WebhookPolicy
    orphans: OrphanPolicy
    # 절이 없으면 끔 (AI 1차 검수 판정을 읽지 않는다, WP17.6)
    ai_review: AiReviewDisplayPolicy = Field(default_factory=AiReviewDisplayPolicy)
    # 절이 없으면 카메라 자세를 자동 라벨로 본다 (ADR 0050)
    verify: VerifyPolicy = Field(default_factory=VerifyPolicy)
    # 절이 없으면 힌트를 계산하지 않는다 (`dlp review hints`가 거부, ADR 0055)
    object_hints: ObjectHintsPolicy | None = None
    # 절이 없으면 `dlp review hints --ai`가 거부한다 (ADR 0056)
    ai_crops: AiCropsPolicy | None = None
    ratios: ReviewConfig  # 배정 비율 (config/defaults.yaml `review` 절)


def load_policy(root: Path, path: Path | None = None) -> ReviewOpsPolicy:
    """저장소 루트(root)에서 검수 운영 정책을 읽는다.

    path가 없으면 `config/policies/review.yaml`을 읽고, 배정 비율은 defaults.yaml의 `review` 절을
    `ratios` 키로 끼워 넣어 함께 검증한다.

    예외: 파일이 없으면 OSError, 값이 범위를 벗어나거나 키가 맞지 않으면 pydantic ValidationError.
    부작용 없음 (파일 읽기만).
    """
    source = path if path is not None else root / "config" / "policies" / "review.yaml"
    data: Any = yaml.safe_load(source.read_text("utf-8"))
    data["ratios"] = load_config(root / "config" / "defaults.yaml").review.model_dump()
    return ReviewOpsPolicy.model_validate(data)
