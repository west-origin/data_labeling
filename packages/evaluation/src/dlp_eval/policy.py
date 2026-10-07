"""평가 정책 (config/policies/evaluation.yaml)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field

from dlp_schema.common import Contract

Task = Literal[
    "objects", "hands", "body", "contact", "actions", "relations", "states", "coverage", "privacy"
]
TASKS: tuple[Task, ...] = (
    "objects",
    "hands",
    "body",
    "contact",
    "actions",
    "relations",
    "states",
    "coverage",
    "privacy",
)


class Tolerances(Contract):
    contact: int
    contact_glove: int
    boundary: int
    state: int


class PrivacyEval(Contract):
    coverage: float = Field(
        gt=0, le=1, description="정답 블러 박스 면적 중 예측 블러가 덮어야 하는 비율"
    )
    precision_overlap: float = Field(
        gt=0, le=1, description="예측 블러 박스 면적 중 정답 대상 위에 있어야 맞은 것으로 보는 비율"
    )


class InterpGaps(Contract):
    """예측 트랙을 정답 시각에 맞출 때 보간할 최대 키프레임 간격 (과제별).

    프리라벨 트래커는 max_gap_ms보다 짧은 끊김을 한 트랙으로 잇되 그 사이에 키프레임을 만들지 않는다
    (OWLv2 도구는 frame_stride_ms마다만 추론한다). 그래서 과제의 값은 그 과제 예측을 내는 어댑터의
    트랙 간격 이상이어야 한다. 짧으면 같은 박스도 정답 시각에서 예측이 없는 것으로 세어 기존 지표가
    0에 가까워지고, 약한 후보도 게이트를 통과한다.
    """

    default: int = Field(ge=0)
    tasks: dict[Task, int] = Field(default_factory=dict[Task, int])

    def for_task(self, task: Task) -> int:
        return self.tasks.get(task, self.default)


class GateRule(Contract):
    primary: str
    min_gain: float
    max_drop: float = Field(ge=0)
    max_drop_by: dict[str, float] = Field(
        default_factory=dict[str, float],
        description="지표별 허용 하락 (단위가 다른 지표용, 예: 오차 ms). 없으면 max_drop",
    )
    guards: tuple[str, ...]

    def allowed_drop(self, metric: str) -> float:
        return self.max_drop_by.get(metric, self.max_drop)

    first_deploy: float


class EvaluationPolicy(Contract):
    version: int
    tolerance_ms: Tolerances
    max_interp_ms: InterpGaps
    track_iou: float = Field(gt=0, le=1)
    segment_iou: tuple[float, ...]
    relation_iou: float = Field(gt=0, le=1)
    match_iou: float = Field(gt=0, le=1)
    pck_alpha: float = Field(gt=0)
    ece_bins: int = Field(ge=1)
    min_samples_per_class: int = Field(ge=1)
    privacy: PrivacyEval
    subgroups: tuple[Literal["glove", "site"], ...]
    gate: dict[Task, GateRule]


def load_policy(root: Path) -> EvaluationPolicy:
    data: Any = yaml.safe_load(
        (root / "config" / "policies" / "evaluation.yaml").read_text("utf-8")
    )
    return EvaluationPolicy.model_validate(data)
