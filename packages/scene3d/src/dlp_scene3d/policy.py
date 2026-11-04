"""청소 전후 3D 비교 정책 (config/policies/scene3d.yaml).

CLI(`dlp_cli.scene3d_cmds`)가 `load_policy`로 읽어 각 단계에 절을 넘긴다. 정책 값은 코드에 두지
않는다. 절마다 `section_digest`로 해시를 만들어 그 단계 결과의 모델 버전에 넣는다 (절을 바꾸면 그
단계부터 다시 계산하고, 다른 절을 바꿔도 앞 단계는 그대로 쓴다).
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


class ReconstructionPolicy(Contract):
    """`reconstruction` 절: SfM·스플랫 백엔드와 실패 판정."""

    sfm: Literal["pycolmap", "unavailable"]
    splat: Literal["gsplat", "init_points"]
    sfm_tool: str
    splat_tool: str
    min_registered_ratio: float = Field(gt=0, le=1)
    splat_iterations: int = Field(ge=1)
    init_scale: float = Field(gt=0)


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


class Scene3dPolicy(Contract):
    """정책 파일 전체."""

    version: int
    capture: CapturePolicy
    reconstruction: ReconstructionPolicy
    alignment: AlignmentPolicy
    objects: ObjectsPolicy
    change: ChangePolicy


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
