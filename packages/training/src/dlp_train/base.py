"""학습기·로더 인터페이스 (`Trainer`, `ModelLoader`, `TrainOutput`, `LoadContext`, `TrainingError`).

`dlp_train.trainers`가 다시 내보내고 등록부(`TRAINERS`·`LOADERS`)를 둔다. 학습기 구현 모듈
(`trainers`의 oracle-stub, `yolox`)이 서로를 import하지 않고 이 모듈만 보게 따로 둔다
(ADR 0016·0062).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from dlp_schema.labels import LabelRecord
from dlp_schema.predictor import Predictor
from dlp_train.extract import TrainingData


class TrainingError(RuntimeError):
    """재학습·배포를 진행할 수 없음 (정책·학습기 없음, 골든셋 누출, 해시 불일치, 승인 조건 위반
    등). `dlp_train.loop`이 다시 내보낸다 (학습기 모듈이 루프를 import하지 않고 쓰게 여기 둔다)."""


@dataclass(frozen=True)
class TrainOutput:
    """학습 결과."""

    artifact: Path  # 산출물 파일 (루프가 sha256을 재고 저장소·MLflow에 올린다)
    # 학습 지표 (루프가 MLflow에 train/ 접두로 기록)
    metrics: dict[str, float] = field(default_factory=dict[str, float])


class Trainer(Protocol):
    """학습기 인터페이스."""

    name: str  # 등록 이름 (정책 trainer, 레지스트리 model_versions.trainer)

    def train(self, data: TrainingData, params: Mapping[str, Any], out_dir: Path) -> TrainOutput:
        """학습 예제로 학습해 `out_dir` 안에 산출물 파일 하나를 쓰고 그 경로·학습 지표를 돌려준다.

        params는 정책 템플릿 위에 `dlp train run --param`을 덮은 값이다. 산출물 업로드·해시는
        호출자(`loop.run_training_job`)가 한다.
        """
        ...


@dataclass(frozen=True)
class LoadContext:
    """로더에 넘기는 문맥."""

    now: datetime  # 예측 레코드의 created_at (시간대 있는 UTC)
    ontology_version: str  # 예측 레코드에 적을 온톨로지 버전
    # (세션, 스트림) → 그 스트림의 정답과 세션의 타임라인 정답(stream_id 없음, 접촉·행동).
    # oracle-stub만 쓴다 (평가·테스트). 운영에서는 None
    truth: Callable[[str, str], list[LabelRecord]] | None = None


class ModelLoader(Protocol):
    """산출물 로더 인터페이스."""

    def load(self, artifact: Path, *, version: str, ctx: LoadContext) -> Predictor:
        """산출물을 Predictor로 읽는다. 쓸 수 없으면 ModelUnavailableError."""
        ...
