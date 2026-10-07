"""배포된 재학습 모델을 단계(프리라벨·프라이버시)에 붙인다.

단계 CLI는 기본 어댑터를 만든 뒤 이 모듈로 배포 모델을 읽어, 정책의 replaces에 적힌 기본 어댑터를
빼고 배포 모델을 넣는다. 산출물 해시가 레지스트리와 다르거나 로더가 쓸 수 없으면(예: oracle-stub은
정답이 있어야 한다) 기본 어댑터를 그대로 쓰고 이유를 남긴다.

WP13, ADR 0016. 사용처: `dlp_cli.prelabel_cmds`(stage "prelabel"), `dlp_cli.privacy_cmds`
(stage "privacy"). 행동 단계는 아직 읽지 않는다 (training.yaml의 TODO(real-model)).
DB는 읽기만 한다 (model_versions). 산출물은 학습 산출물 저장소(buckets.mlflow)에서 `work`로 받는다.
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
    """단계에 붙일 배포 모델과 뺄 기본 어댑터."""

    predictors: list[Predictor] = field(default_factory=list[Predictor])  # 쓸 수 있는 배포 모델
    replaces: set[str] = field(default_factory=set[str])  # 빼야 할 기본 어댑터 이름
    notes: list[str] = field(default_factory=list[str])  # 사람이 읽는 처리 기록 (CLI가 출력)

    def apply(self, base: list[Predictor]) -> list[Predictor]:
        """기본 어댑터 목록에서 `replaces`를 빼고 배포 모델을 뒤에 붙인 새 목록."""
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

    Args:
        conn: DB 연결 (model_versions 읽기).
        artifacts: 학습 산출물 저장소.
        policy: 학습 정책 (과제별 stage·replaces).
        stage: 붙일 단계 ("prelabel" 또는 "privacy"). 그 단계 과제만 본다.
        ctx: 로더 문맥 (운영에서는 truth=None이라 oracle-stub은 못 쓴다).
        work: 산출물을 받을 로컬 디렉터리.
        loaders: 학습기 이름 → 로더 (테스트용 주입, 기본 `LOADERS`).

    Returns:
        `Deployed`. 과제마다 가장 최근(created_at 순 마지막) deployed 모델 하나를 쓴다. 로더가
        없거나 `ModelUnavailableError`·`TrainingError`(해시 불일치)면 그 과제는 기본 어댑터를
        그대로 둔다 (replaces도 더하지 않는다).

    Raises:
        TrainingError: strict이고 배포 모델을 쓸 수 없을 때.
    """
    loaders = LOADERS if loaders is None else loaders
    out = Deployed()
    for task, spec in policy.tasks.items():
        if spec.stage != stage:
            continue
        deployed = list_model_versions(conn, task, ModelStatus.DEPLOYED)
        if not deployed:
            continue
        # 배포는 과제마다 하나지만, 혹시 여럿이면 가장 최근 것 (list_model_versions는 created_at 순)
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
            # 산출물을 받아 레지스트리 sha256과 대조한다 (다르면 TrainingError)
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
