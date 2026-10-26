"""공개 데이터셋 목록과 1차 학습 설정 검사 (config/external_datasets.yaml, ADR 0060).

공개 데이터셋은 파이프라인 밖에서 프리라벨 가중치를 1차 학습하는 데만 쓴다. 라벨 이력·데이터셋 버전·
골든셋·내보내기에는 들어가지 않는다 (ADR 0060 1절). 이 모듈은 그 경계를 설정으로 고정한다.

- `ExternalDataset` / `DatasetRegistry`: 데이터셋마다 판, 받는 곳, 주석·매체 라이선스, 약관 요지,
  접근 절차, 상업 사용 분류(ADR 0010과 같은 allowed·review·forbidden), 원문 확인 근거.
- `PretrainObjects`: 객체 탐지기 1차 학습 설정 (`config/pretrain/objects.yaml`). 클래스 대응은
  온톨로지 ID → 데이터셋별 원래 클래스 이름 목록이다.
- `dataset_violations`: `models.yaml`에서 `training_datasets`를 적은 가중치의 분류가 데이터셋(과
  출발 가중치) 중 가장 엄한 분류보다 느슨하면 위반이다. forbidden 데이터셋은 학습에 쓰면 그 자체로
  위반이다.
  `make licenses`(`dlp models licenses`)가 `ModelRegistry.violations`와 함께 부른다.

분류 순서: allowed < review < forbidden (`SEVERITY`).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, model_validator

from dlp_models.registry import Commercial, ModelRegistry
from dlp_schema.common import Contract

# 상업 사용 분류의 엄한 정도 (클수록 엄하다)
SEVERITY: dict[Commercial, int] = {"allowed": 0, "review": 1, "forbidden": 2}


class ExternalDataset(Contract):
    """`external_datasets.yaml datasets.<이름>` 항목.

    필드:
    - version: 쓰는 판 (예: "v7", "v1").
    - homepage: 공식 안내 주소. download: 주석·매체를 받는 공식 주소 목록.
    - annotation_license / media_license: 주석과 이미지·영상의 라이선스 (원문 표기 그대로).
    - terms: 약관 요지 (모델 학습·상업 사용·재배포 조건, 한국어 한두 문장).
    - access: open(바로 받음) | request(신청·승인) | signed(서명한 약정 필요).
    - commercial: 이 데이터로 학습한 가중치의 분류 (ADR 0010). 학습 자체나 파생 모델 사용을 막으면
      forbidden(학습에 쓰지 않는다).
    - tasks: 쓰는 과제 (예: objects). verified: 라이선스 원문을 확인한 날 (없으면 미확인).
    - sources: 확인 근거 주소 (라이선스 원문·데이터 카드).
    """

    version: str
    homepage: str
    download: tuple[str, ...] = ()
    annotation_license: str
    media_license: str
    terms: str
    access: Literal["open", "request", "signed"]
    commercial: Commercial
    tasks: tuple[str, ...] = ()
    verified: date | None = None
    sources: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _verified_needs_sources(self) -> ExternalDataset:
        """원문을 확인했다고 적었으면 근거 주소가 있어야 한다."""
        if self.verified is not None and not self.sources:
            raise ValueError("verified가 있으면 sources(근거 주소)가 필요합니다")
        return self


class DatasetRegistry(Contract):
    """`config/external_datasets.yaml` 전체 (version: 형식 버전, datasets: 이름 → 항목)."""

    version: int
    datasets: dict[str, ExternalDataset]


class PretrainBuild(Contract):
    """학습 데이터 만들기 설정 (`build` 절). 단위: 이미지 장 수, 비율, px."""

    max_images_per_class: int = Field(ge=1)
    val_fraction: float = Field(gt=0, lt=1)
    seed: int = Field(ge=0)
    min_box_px: int = Field(ge=1)


class PretrainTrain(Contract):
    """학습 설정 (`train` 절).

    yolox_commit: Megvii YOLOX 고정 커밋(40자 16진). model: 출발 모델 이름.
    """

    yolox_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    model: str
    input_size: int = Field(ge=32)
    epochs: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    fp16: bool


class PretrainObjects(Contract):
    """객체 탐지기 1차 학습 설정 (`config/pretrain/objects.yaml`).

    datasets: 쓰는 데이터셋 이름 (`external_datasets.yaml`에 있어야 한다). classes: 온톨로지 객체
    ID →
    {데이터셋 이름: 원래 클래스 이름 목록}. 클래스 순서(키 순서)가 탐지기 출력 클래스 번호다.
    """

    version: int
    datasets: tuple[str, ...] = Field(min_length=1)
    classes: dict[str, dict[str, tuple[str, ...]]] = Field(min_length=1)
    build: PretrainBuild
    train: PretrainTrain

    @model_validator(mode="after")
    def _classes_use_listed_datasets(self) -> PretrainObjects:
        """클래스 대응은 datasets에 든 데이터셋만 쓰고, 객체마다 원래 클래스가 하나 이상이다."""
        for obj, by_ds in self.classes.items():
            unknown = set(by_ds) - set(self.datasets)
            if unknown:
                raise ValueError(f"classes.{obj}: datasets에 없는 데이터셋 {sorted(unknown)}")
            if not any(by_ds.values()):
                raise ValueError(f"classes.{obj}: 원래 클래스가 하나도 없습니다")
        return self


def load_datasets(root: Path) -> DatasetRegistry:
    """`<root>/config/external_datasets.yaml`을 읽어 검증한다.

    Raises:
        FileNotFoundError: 파일이 없을 때. pydantic.ValidationError: 형식이 맞지 않을 때.
    """
    data: Any = yaml.safe_load(
        (root / "config" / "external_datasets.yaml").read_text(encoding="utf-8")
    )
    return DatasetRegistry.model_validate(data)


def load_pretrain_objects(root: Path) -> PretrainObjects:
    """`<root>/config/pretrain/objects.yaml`을 읽어 검증한다."""
    data: Any = yaml.safe_load(
        (root / "config" / "pretrain" / "objects.yaml").read_text(encoding="utf-8")
    )
    return PretrainObjects.model_validate(data)


def dataset_violations(models: ModelRegistry, datasets: DatasetRegistry) -> list[str]:
    """`training_datasets`를 적은 가중치와 1차 학습 설정의 데이터셋 분류 위반 (사람이 읽는 메시지).

    - 목록에 없는 데이터셋 이름, forbidden 데이터셋으로 학습한 가중치.
    - 가중치 `commercial`이 데이터셋·출발 가중치(`base_model`) 분류 중 가장 엄한 것보다 느슨함.
    - 출발 가중치 이름이 `models.yaml`에 없음.
    """
    out: list[str] = []
    for name, spec in sorted(models.models.items()):
        if not spec.training_datasets and spec.base_model is None:
            continue
        floor: list[Commercial] = []
        for ds in spec.training_datasets:
            entry = datasets.datasets.get(ds)
            if entry is None:
                out.append(f"{name}: 공개 데이터셋 목록에 없는 {ds}")
                continue
            if entry.commercial == "forbidden":
                out.append(f"{name}: forbidden 데이터셋 {ds}로 학습했습니다")
            floor.append(entry.commercial)
        if spec.base_model is not None:
            base = models.models.get(spec.base_model)
            if base is None:
                out.append(f"{name}: 출발 가중치 {spec.base_model}이 모델 목록에 없습니다")
            else:
                floor.append(base.commercial)
        worst: Commercial = max(floor, key=SEVERITY.__getitem__) if floor else "allowed"
        if SEVERITY[spec.commercial] < SEVERITY[worst]:
            out.append(
                f"{name}: 분류 {spec.commercial}이 학습 데이터·출발 가중치({worst})보다 느슨합니다"
            )
    return out


def pretrain_violations(config: PretrainObjects, datasets: DatasetRegistry) -> list[str]:
    """1차 학습 설정이 목록에 없거나 forbidden인 데이터셋을 쓰면 위반."""
    out: list[str] = []
    for ds in config.datasets:
        entry = datasets.datasets.get(ds)
        if entry is None:
            out.append(f"pretrain objects: 공개 데이터셋 목록에 없는 {ds}")
        elif entry.commercial == "forbidden":
            out.append(f"pretrain objects: forbidden 데이터셋 {ds}")
    return out
