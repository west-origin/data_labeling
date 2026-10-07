"""학습기와 로더.

- Trainer: 학습 예제 → 산출물 파일 하나 (가중치·설정). 학습 지표를 함께 낸다.
- ModelLoader: 산출물 → 단계가 쓰는 부품 (프리라벨 Predictor, 프라이버시 FrameDetector 등).
  배포된 모델은 레지스트리의 trainer 이름으로 로더를 찾는다.

CPU·CI용 stub은 oracle-stub 하나다. 학습 예제에서 본 클래스만 기억하고, 예측할 때는 정답을 (흔들림을
넣어) 그 클래스만 돌려준다. 그래서 "학습 데이터가 늘면 좋아지고, 흔들림이 크면 나빠지는" 루프를
실제 모델 없이 시험할 수 있다. 정답이 필요하므로 평가·테스트에서만 쓸 수 있다.

TODO(real-model): 실제 학습기·로더 (객체 YOLOX 미세조정, 손·전신 RTMPose 미세조정, 접촉 영상 분류기,
  블러 탐지기 미세조정). GPU 학습 서버가 필요하다. 미세조정 출발 가중치의 상업 사용 분류는
  config/models.yaml을 따른다 (ADR 0010).
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from dlp_schema.episode import version_tag
from dlp_schema.labels import (
    BlurTrackPayload,
    BoxKeyframe,
    BoxTrackPayload,
    KeypointTrackPayload,
    LabelPayload,
    LabelRecord,
    Provenance,
    Source,
    Verification,
)
from dlp_schema.predictor import Clip, ModelUnavailableError, Predictor
from dlp_train.extract import TrainingData, class_key, matches


@dataclass(frozen=True)
class TrainOutput:
    artifact: Path
    metrics: dict[str, float] = field(default_factory=dict[str, float])


class Trainer(Protocol):
    name: str

    def train(
        self, data: TrainingData, params: Mapping[str, Any], out_dir: Path
    ) -> TrainOutput: ...


@dataclass(frozen=True)
class LoadContext:
    now: datetime
    ontology_version: str
    # (세션, 스트림) → 정답 라벨. oracle-stub만 쓴다 (평가·테스트). 운영에서는 None
    truth: Callable[[str, str], list[LabelRecord]] | None = None


class ModelLoader(Protocol):
    def load(self, artifact: Path, *, version: str, ctx: LoadContext) -> Predictor:
        """산출물을 Predictor로 읽는다. 쓸 수 없으면 ModelUnavailableError."""
        ...


# ---------------------------------------------------------------- oracle-stub


class OracleStubTrainer:
    name = "oracle-stub"

    def train(self, data: TrainingData, params: Mapping[str, Any], out_dir: Path) -> TrainOutput:
        classes = Counter(class_key(e.label) for e in data.examples if e.change != "deleted")
        artifact = out_dir / "oracle-stub.json"
        artifact.write_text(
            json.dumps(
                {
                    "task": data.task,
                    "classes": sorted(classes),
                    "jitter_px": float(params.get("jitter_px", 0.0)),
                    "seed": int(params.get("seed", 0)),
                    "examples": len(data.examples),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        counts = data.counts()
        return TrainOutput(
            artifact,
            {"examples": float(len(data.examples)), "classes": float(len(classes))}
            | {f"examples/{k}": float(v) for k, v in counts.items()},
        )


def _jitter_box(k: BoxKeyframe, rng: np.random.Generator, px: float) -> BoxKeyframe:
    if k.outside or not px:
        return k
    return k.model_copy(
        update={"x": k.x + float(rng.normal(0, px)), "y": k.y + float(rng.normal(0, px))}
    )


def _jitter(p: LabelPayload, rng: np.random.Generator, px: float) -> LabelPayload:
    if not px:
        return p
    match p:
        case BoxTrackPayload() | BlurTrackPayload():
            return p.model_copy(
                update={"keyframes": tuple(_jitter_box(k, rng, px) for k in p.keyframes)}
            )
        case KeypointTrackPayload():
            frames = tuple(
                f.model_copy(
                    update={
                        "points": tuple(
                            q.model_copy(
                                update={
                                    "x": q.x + float(rng.normal(0, px)),
                                    "y": q.y + float(rng.normal(0, px)),
                                }
                            )
                            for q in f.points
                        )
                    }
                )
                for f in p.keyframes
            )
            return p.model_copy(update={"keyframes": frames})
        case _:
            return p


class OracleStubPredictor:
    def __init__(self, spec: dict[str, Any], *, version: str, ctx: LoadContext) -> None:
        if ctx.truth is None:
            raise ModelUnavailableError("oracle-stub은 정답이 있어야 합니다 (평가·테스트 전용)")
        self.task = spec["task"]
        self.name = f"trained-{self.task}"
        self.version = version
        self.classes = set(spec["classes"])
        self.jitter = float(spec["jitter_px"])
        self.seed = int(spec["seed"])
        self.truth = ctx.truth
        self.ctx = ctx

    def run(self, clip: Clip) -> list[LabelRecord]:
        rng = np.random.default_rng(self.seed)
        prefix = f"{clip.session_id}-{clip.stream_id}-{self.name}-{version_tag(self.version)}"
        out: list[LabelRecord] = []
        truth = sorted(self.truth(clip.session_id, clip.stream_id), key=lambda x: x.label_id)
        for i, x in enumerate(truth):
            if not matches(self.task, x) or class_key(x) not in self.classes:
                continue
            out.append(
                x.model_copy(
                    update={
                        "label_id": f"{prefix}-{i:04d}",
                        "parent_label_id": None,
                        "provenance": Provenance(source=Source.MODEL, model_version=self.version),
                        "verification": Verification(),
                        "confidence": 0.9,
                        "seeded_error": False,
                        "measurement": None,
                        "retracted": False,
                        "ontology_version": self.ctx.ontology_version,
                        "created_at": self.ctx.now,
                        "payload": _jitter(x.payload, rng, self.jitter),
                    }
                )
            )
        return out


class OracleStubLoader:
    def load(self, artifact: Path, *, version: str, ctx: LoadContext) -> Predictor:
        spec: dict[str, Any] = json.loads(artifact.read_text(encoding="utf-8"))
        return OracleStubPredictor(spec, version=version, ctx=ctx)


TRAINERS: dict[str, Trainer] = {"oracle-stub": OracleStubTrainer()}
LOADERS: dict[str, ModelLoader] = {"oracle-stub": OracleStubLoader()}
