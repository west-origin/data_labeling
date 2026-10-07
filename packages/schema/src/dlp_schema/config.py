"""config/defaults.yaml 로더. 코드는 정책 값을 하드코딩하지 않고 이 설정을 읽는다.

역할
    플랫폼 전역 기본값(인프라 선택, 버킷 이름, 프록시 인코딩, 프라이버시·검수 비율, 골든셋 분할
    단위, 내보내기 기본, 원본 보관 기간, 성공 기준)을 타입이 있는 계약으로 읽는다 (WP1).
    단계별 세부 정책은 `config/policies/<단계>.yaml`에 따로 있고 각 패키지의 정책 로더가 읽는다.

주요 이름
    - `PlatformConfig`와 하위 절 타입 (`InfrastructureConfig`,
      `BucketsConfig`, ...): YAML 구조 그대로.
    - `load_config(path)`: YAML 파일을 읽어 검증한다.
    - `repo_root(start)`: `config/defaults.yaml`이 있는 저장소 루트를 위로 올라가며 찾는다.

주의
    - 모든 계약 타입이 `extra="forbid"`라 YAML에 모르는 키가 있으면 검증 오류다 (오타 방지).
      YAML에 키를 추가하면 여기 타입에도 추가하고 `make schemas`(platform_config.schema.json)를
      다시 한다.
    - YAML 주석은 파싱 후 사라지므로 값의 의미 설명은 `config/defaults.yaml` 주석을 본다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field

from dlp_schema.common import Contract, OntologyId


# defaults.yaml `infrastructure`: 인프라 구성 요소 선택 (기준 문서 "미결정 사항"의 기본값).
#   object_store: 객체 저장소 구현. data_versioning: 데이터셋 버전 도구.
#   annotation_tools: hybrid(CVAT+Label Studio) | cvat_only. orchestrator: 워크플로 실행기.
#   experiment_tracking: 실험 기록 (현재 mlflow만).
# 현재 코드는 이 절을 검증만 하고 분기에 쓰지 않는다 (구현은
# seaweedfs·lakefs·hybrid·prefect·mlflow 기준).
class InfrastructureConfig(Contract):
    object_store: Literal["seaweedfs", "minio"]
    data_versioning: Literal["lakefs", "dvc"]
    annotation_tools: Literal["hybrid", "cvat_only"]
    orchestrator: Literal["prefect", "airflow"]
    experiment_tracking: Literal["mlflow"]


# defaults.yaml `buckets`: 객체 저장소 버킷 이름.
#   raw: 원본(블러 전). 원본 권한자·학습 서버만 접근, 모든 접근은 감사 기록 (ADR 0020).
#   labeling: 블러본·프록시 (일반 라벨러 읽기 전용).
#   datasets: 데이터셋 버전·내보내기. mlflow: 학습 산출물.
class BucketsConfig(Contract):
    raw: str
    labeling: str
    datasets: str
    mlflow: str


class ProxyConfig(Contract):
    """검수 화면용 프록시 영상 인코딩 설정."""

    # max_height: 출력 높이 상한(px, 짝수). 원본이 더 높으면 이 높이로 줄인다.
    # crf: libx264 CRF (0~51, 낮을수록 고화질·큰 파일).
    # keyframe_ms: 키프레임 간격(ms, 0보다 큼). 짧을수록 스크러빙이 빠르고 파일이 커진다.
    max_height: int = Field(
        ge=2,
        multiple_of=2,
        description="높이 상한 (짝수: libx264 yuv420p는 가로·세로가 짝수여야 한다)",
    )
    crf: int = Field(ge=0, le=51)
    keyframe_ms: int = Field(gt=0)


# defaults.yaml `media`: 수집 단계(`dlp ingest`) 설정.
class MediaConfig(Contract):
    proxy: ProxyConfig


# defaults.yaml `privacy.full_review_exit`: 블러 전수 검수를 끝내고 표본 감사로 넘어가는 조건.
#   weeks_below_target: 잔여 누락률이 목표 아래로 유지되어야 하는 연속 주 수 (양수).
#   audit_sample_ratio: 전수 검수를 끝낸 뒤 감사할 표본 비율 (0 초과 1 이하).
class FullReviewExit(Contract):
    weeks_below_target: int = Field(gt=0)
    audit_sample_ratio: float = Field(gt=0, le=1)


# defaults.yaml `privacy`: 프라이버시 게이트 공통 기본값 (세부는 config/policies/privacy.yaml).
#   blur_hold_ms: 탐지가 끊겨도 블러를 유지하는 시간(ms, 0 이상).
#   render_mode: 렌더 방식 (mosaic | solid).
#   strip_audio_in_release: 블러본(공개본)에서 오디오를 뺄지.
#   extra_targets_v1: 필수 외에 활성화할 확장 블러 대상 (온톨로지 privacy_targets의
#     required: false 키). 현재 프라이버시 패키지는 이 값을 읽지 않는다 (탐지 대상은
#     config/policies/privacy.yaml이 정한다).
#   full_review_exit: 전수 검수 종료 조건.
class PrivacyConfig(Contract):
    blur_hold_ms: int = Field(ge=0)
    render_mode: Literal["mosaic", "solid"]
    strip_audio_in_release: bool
    extra_targets_v1: tuple[OntologyId, ...]
    full_review_exit: FullReviewExit


# defaults.yaml `review`: 검수 운영 비율 (0~1). `dlp_review.ops.policy`가 review.yaml과 합쳐 쓴다.
#   qa_sample_ratio: 선임 QA 재검수 표본 비율.
#   double_annotation_ratio: 이중 라벨링(일치도 측정) 비율.
#   blind_task_ratio: 프리라벨 없이 처음부터 하는 블라인드 과제 비율 (프리라벨 편향 측정).
#   seeded_error_task_ratio: 오류 삽입 과제 비율 (검수자 발견율 측정).
class ReviewConfig(Contract):
    qa_sample_ratio: float = Field(ge=0, le=1)
    double_annotation_ratio: float = Field(ge=0, le=1)
    blind_task_ratio: float = Field(ge=0, le=1)
    seeded_error_task_ratio: float = Field(ge=0, le=1)


# defaults.yaml `golden_set`: 골든셋 구성 기본값.
#   min_instances_per_class: 클래스마다 골든셋에 있어야 하는 최소 인스턴스 수 (양수).
#   split_unit: 분할 누수 방지 단위. 여기 든 속성(worker_id, site_id)이 같은 세션은 같은
#     분할로 간다.
# 주의: 현재 코드(`dlp_datasets.splitter` 등)는 이 절을 읽지 않는다. 분할기는 작업자·장소 둘 다를
# 항상 기준으로 쓰고, 골든셋 클래스 최소 수도 이 값으로 검사하지 않는다.
class GoldenSetConfig(Contract):
    min_instances_per_class: int = Field(gt=0)
    split_unit: tuple[Literal["worker_id", "site_id"], ...] = Field(min_length=1)


# defaults.yaml `export`: 내보내기 기본. include_unreviewed가
# false면 미검수 모델 라벨을 빼고 내보낸다.
class ExportConfig(Contract):
    include_unreviewed: bool


# defaults.yaml `retention`: 원본 보관 기간(일, 양수). None이면 미정이라 만료 알림을 내지 않는다.
class RetentionConfig(Contract):
    raw_retention_days: int | None = Field(default=None, gt=0)


# defaults.yaml `success_criteria`: 운영 성공 기준 (None = 아직 미정, 경고를 내지 않는다).
#   review_minutes_per_video_hour: 영상 1시간당 검수 분 목표.
#   residual_blur_miss_per_hour_max: 영상 1시간당 잔여 블러 누락 허용 상한.
#   boundary_agreement_min: 행동 경계 일치율 하한 (0~1로 쓰는 것을 전제).
# 현재 코드에서 읽는 것은 residual_blur_miss_per_hour_max뿐이다
# (`dlp privacy audit-sample`의 전수 검수 종료 판정).
class SuccessCriteria(Contract):
    review_minutes_per_video_hour: float | None = None
    residual_blur_miss_per_hour_max: float | None = None
    boundary_agreement_min: float | None = None


# config/defaults.yaml 전체. version은 파일 형식 버전(현재 1)이다.
class PlatformConfig(Contract):
    version: int
    infrastructure: InfrastructureConfig
    buckets: BucketsConfig
    media: MediaConfig
    privacy: PrivacyConfig
    review: ReviewConfig
    golden_set: GoldenSetConfig
    export: ExportConfig
    retention: RetentionConfig
    success_criteria: SuccessCriteria


def load_config(path: Path) -> PlatformConfig:
    """`config/defaults.yaml`을 읽어 검증한다.

    Args:
        path: YAML 파일 경로 (보통 `repo_root() / "config" / "defaults.yaml"`).

    Returns:
        검증된 `PlatformConfig`.

    Raises:
        FileNotFoundError: 파일이 없을 때.
        yaml.YAMLError: YAML 문법 오류.
        pydantic.ValidationError: 키 누락·모르는 키·범위 위반.
    """
    data: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    return PlatformConfig.model_validate(data)


def repo_root(start: Path | None = None) -> Path:
    """config/defaults.yaml이 있는 가장 가까운 상위 디렉터리.

    Args:
        start: 탐색 시작 경로. None이면 현재 작업 디렉터리. 자기 자신부터 부모 방향으로 찾는다.

    Raises:
        FileNotFoundError: 파일 시스템 루트까지 못 찾았을 때.
    """
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "config" / "defaults.yaml").is_file():
            return candidate
    raise FileNotFoundError("config/defaults.yaml을 찾을 수 없습니다")
