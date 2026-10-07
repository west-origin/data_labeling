"""배포된 재학습 모델을 단계(프리라벨·프라이버시)에 붙인다.

단계 CLI는 기본 어댑터를 만든 뒤 이 모듈로 배포 모델을 읽어, 정책의 replaces에 적힌 기본 어댑터를
빼고 배포 모델을 넣는다. 산출물 해시가 레지스트리와 다르거나 로더가 쓸 수 없으면(예: oracle-stub은
정답이 있어야 한다) 기본 어댑터를 그대로 쓰고 이유를 남긴다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import sqlalchemy as sa

from dlp_media.storage import ObjectStore
from dlp_schema.db.repository import list_model_versions
from dlp_schema.lineage import ModelStatus
from dlp_schema.predictor import ModelUnavailableError, Predictor
from dlp_train.loop import TrainingError, _download  # pyright: ignore[reportPrivateUsage]
from dlp_train.policy import TrainingPolicy
from dlp_train.trainers import LOADERS, LoadContext, ModelLoader


@dataclass
class Deployed:
    predictors: list[Predictor] = field(default_factory=list[Predictor])
    replaces: set[str] = field(default_factory=set[str])  # 빼야 할 기본 어댑터 이름
    notes: list[str] = field(default_factory=list[str])

    def apply(self, base: list[Predictor]) -> list[Predictor]:
        return [p for p in base if p.name not in self.replaces] + self.predictors


def deployed_predictors(
    conn: sa.Connection,
    artifacts: ObjectStore,
    policy: TrainingPolicy,
    stage: str,
    ctx: LoadContext,
    work: Path,
    loaders: dict[str, ModelLoader] | None = None,
    *,
    strict: bool = False,
) -> Deployed:
    """strict: 배포 모델을 쓸 수 없으면 기본 어댑터로 넘어가지 않고 실패한다.

    프라이버시 단계용: 사람이 승인한 블러 모델을 조용히 빼면 블러 재현이 떨어진다.
    """
    loaders = LOADERS if loaders is None else loaders
    out = Deployed()
    for task, spec in policy.tasks.items():
        if spec.stage != stage:
            continue
        deployed = list_model_versions(conn, task, ModelStatus.DEPLOYED)
        if not deployed:
            continue
        mv = deployed[-1]
        loader = loaders.get(mv.trainer)
        if loader is None:
            if strict:
                raise TrainingError(f"{task}: 배포 모델 {mv.model_version}의 로더가 없습니다")
            out.notes.append(
                f"{task}: 배포 모델 {mv.model_version}의 로더가 없어 기본 어댑터를 씁니다"
            )
            continue
        try:
            predictor = loader.load(
                _download(artifacts, mv, work), version=mv.model_version, ctx=ctx
            )
        except (ModelUnavailableError, TrainingError) as exc:
            if strict:
                raise TrainingError(
                    f"{task}: 배포 모델 {mv.model_version}을 쓸 수 없습니다 ({exc})"
                ) from exc
            out.notes.append(
                f"{task}: 배포 모델 {mv.model_version}을 쓸 수 없어 기본 어댑터를 씁니다 ({exc})"
            )
            continue
        out.predictors.append(predictor)
        out.replaces |= set(spec.replaces)
        out.notes.append(f"{task}: 배포 모델 {mv.model_version}")
    return out
