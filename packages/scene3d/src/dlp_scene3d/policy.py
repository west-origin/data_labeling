"""청소 전후 3D 비교와 장면 안 바디캠 위치 추정 정책 (config/policies/scene3d.yaml).

CLI(`dlp_cli.scene3d_cmds`)가 `load_policy`로 읽어 각 단계에 절을 넘긴다. 정책 값은 코드에 두지
않는다. 절마다 `section_digest`로 해시를 만들어 그 단계 결과의 모델 버전에 넣는다 (절을 바꾸면 그
단계부터 다시 계산하고, 다른 절을 바꿔도 앞 단계는 그대로 쓴다). `interpolation` 절만 예외다:
위치 기록을 읽을 때만 쓰므로 어느 버전에도 들어가지 않는다 (ADR 0067).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field

from dlp_schema.common import Contract, OntologyId


class CapturePolicy(Contract):
    """`capture` 절: 받는 확장자와 사진 수 범위."""

    extensions: tuple[Literal[".jpg", ".jpeg", ".png"], ...]
    min_photos: int = Field(ge=2)
    max_photos: int = Field(ge=2)


class MarkerPolicy(Contract):
    """`reconstruction.marker` 절: 크기를 아는 ArUco 마커로 미터 척도를 정한다 (ADR 0067).

    값의 뜻과 단위는 YAML 주석. dictionary는 `cv2.aruco`의 사전 이름(`DICT_…`)이다.
    """

    enabled: bool
    dictionary: str = Field(pattern=r"^DICT_[A-Za-z0-9_]+$")
    side_m: float = Field(gt=0)
    ids: tuple[int, ...]
    corner_refinement: Literal["apriltag", "subpix"]
    min_photos: int = Field(ge=2)
    max_corner_error: float = Field(gt=0)
    min_parallax_deg: float = Field(ge=0, lt=180)
    max_side_spread: float = Field(gt=0)
    metric_check_tolerance: float = Field(gt=0)


class ReconstructionPolicy(Contract):
    """`reconstruction` 절: SfM·스플랫 백엔드, 실패 판정, 마커 척도."""

    sfm: Literal["pycolmap", "unavailable"]
    splat: Literal["gsplat", "init_points"]
    sfm_tool: str
    splat_tool: str
    min_registered_ratio: float = Field(gt=0, le=1)
    splat_iterations: int = Field(ge=1)
    init_scale: float = Field(gt=0)
    marker: MarkerPolicy


class AlignmentPolicy(Contract):
    """`alignment` 절: 점 대응 RANSAC과 거부 문턱."""

    descriptor_ratio: float = Field(gt=0, le=1)
    inlier_threshold_m: float = Field(gt=0)
    ransac_iterations: int = Field(ge=1)
    seed: int
    min_inliers: int = Field(ge=3)
    min_inlier_ratio: float = Field(gt=0, le=1)
    max_rms_m: float = Field(gt=0)


class ObjectsPolicy(Contract):
    """`objects` 절: 탐지기, 질의, 외형 특징, 다시점 연결."""

    detector: Literal["owlv2", "unavailable"]
    model: str
    tokenizer: str
    score_threshold: float = Field(ge=0, le=1)
    nms_iou: float = Field(gt=0, le=1)
    queries: dict[OntologyId, str] = Field(min_length=1)
    appearance_bins: tuple[int, int]
    appearance_crop: float = Field(gt=0, le=1)
    center_tolerance: float = Field(gt=0)
    min_views: int = Field(ge=2)
    full_views: int = Field(ge=2)


class ChangeWeights(Contract):
    """대응 비용 가중치."""

    appearance: float = Field(ge=0)
    size: float = Field(ge=0)
    distance: float = Field(ge=0)


class ChangePolicy(Contract):
    """`change` 절: 대응 관문과 이동 판정."""

    require_metric: bool
    unchanged_tolerance_m: float = Field(gt=0)
    max_appearance_distance: float = Field(gt=0, le=1)
    max_size_ratio: float = Field(ge=1)
    distance_scale_m: float = Field(gt=0)
    weights: ChangeWeights


class LocalizationPolicy(Contract):
    """`localization` 절: 장면 안 바디캠 위치 추정 (프레임 추출, 대응, PnP, 거부 문턱, ADR 0067).

    값의 뜻과 단위는 YAML 주석. 문턱을 못 넘은 프레임은 기록하지 않는다.
    """

    features: Literal["pycolmap", "unavailable"]
    features_tool: str
    max_features: int = Field(ge=1)
    frame_interval_ms: int = Field(ge=1)
    descriptor_ratio: float = Field(gt=0, le=1)
    ransac_reproj_ratio: float = Field(gt=0, lt=1)
    ransac_iterations: int = Field(ge=1)
    ransac_confidence: float = Field(gt=0, lt=1)
    min_inliers: int = Field(ge=4)
    max_reproj_ratio: float = Field(gt=0, lt=1)
    min_inlier_spread: float = Field(ge=0, le=1)
    max_intrinsics_aspect_mismatch: float = Field(ge=0)


class InterpolationPolicy(Contract):
    """`interpolation` 절: 위치 기록을 읽어 임의 시각의 자세를 보간할 때의 한계 (ADR 0067).

    계산이 아니라 읽기에만 쓰므로 어떤 단계의 모델 버전에도 들어가지 않는다 (바꿔도 다시 계산하지
    않는다).
    """

    max_gap_ms: int = Field(ge=1)


class Scene3dPolicy(Contract):
    """정책 파일 전체."""

    version: int
    capture: CapturePolicy
    reconstruction: ReconstructionPolicy
    alignment: AlignmentPolicy
    objects: ObjectsPolicy
    change: ChangePolicy
    localization: LocalizationPolicy
    interpolation: InterpolationPolicy


def section_digest(section: Contract) -> str:
    """절 하나의 짧은 해시 (파싱·검증된 값 기준이라 YAML 주석·키 순서와 무관하다). 앞 10자."""
    raw = json.dumps(section.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()[:10]


def load_policy(root: Path) -> Scene3dPolicy:
    """저장소 루트에서 `config/policies/scene3d.yaml`을 읽어 검증한다.

    Raises:
        FileNotFoundError: 파일이 없을 때.
        pydantic.ValidationError: 값이 범위를 벗어나거나 키가 틀렸을 때.
    """
    data: Any = yaml.safe_load((root / "config" / "policies" / "scene3d.yaml").read_text("utf-8"))
    return Scene3dPolicy.model_validate(data)
