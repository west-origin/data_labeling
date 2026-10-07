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

WP13, ADR 0016. 새 학습기를 추가하려면 `Trainer`와 `ModelLoader`를 같은 이름으로
`TRAINERS`·`LOADERS`에
등록하고, CPU용 stub 경로를 함께 둔다 (CLAUDE.md "새 모델은 Predictor 어댑터와 stub을 함께").
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
    """학습 결과."""

    artifact: Path  # 산출물 파일 (루프가 sha256을 재고 저장소·MLflow에 올린다)
    # 학습 지표 (루프가 MLflow에 train/ 접두로 기록)
    metrics: dict[str, float] = field(default_factory=dict[str, float])


class Trainer(Protocol):
    """학습기 인터페이스."""

    name: str  # 등록 이름 (정책 trainer, 레지스트리 model_versions.trainer)

    def train(
        self, data: TrainingData, params: Mapping[str, Any], out_dir: Path
    ) -> TrainOutput: ...


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


# ---------------------------------------------------------------- oracle-stub


class OracleStubTrainer:
    """CI·CPU용 stub 학습기. 학습 예제의 클래스 목록과 파라미터를 JSON 산출물로 쓴다."""

    name = "oracle-stub"

    def train(self, data: TrainingData, params: Mapping[str, Any], out_dir: Path) -> TrainOutput:
        """학습 예제에서 본 클래스(지운 예제 제외)를 기억하는 산출물 `oracle-stub.json`을 쓴다.

        Args:
            data: 학습 데이터. deleted 예제(오탐)의 클래스는 배울 클래스가 아니다.
            params: jitter_px(예측 좌표 흔들림 표준편차, px, 기본 0), seed(난수 씨앗, 기본 0),
                confidence(예측 신뢰도, 기본 0.9).
            out_dir: 산출물을 쓸 디렉터리 (호출자의 임시 디렉터리).

        Returns:
            산출물 경로와 지표 (examples, classes, examples/<분할>/<변화 종류>).
        """
        classes = Counter(class_key(e.label) for e in data.examples if e.change != "deleted")
        artifact = out_dir / "oracle-stub.json"
        artifact.write_text(
            json.dumps(
                {
                    "task": data.task,
                    "classes": sorted(classes),
                    "jitter_px": float(params.get("jitter_px", 0.0)),
                    "seed": int(params.get("seed", 0)),
                    "confidence": float(params.get("confidence", 0.9)),
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
    """박스 키프레임의 x·y를 정규분포(표준편차 px)로 흔든다. 화면 밖이거나 px 0이면 그대로."""
    if k.outside or not px:
        return k
    return k.model_copy(
        update={"x": k.x + float(rng.normal(0, px)), "y": k.y + float(rng.normal(0, px))}
    )


def _jitter(p: LabelPayload, rng: np.random.Generator, px: float) -> LabelPayload:
    """공간 페이로드(박스·블러·키포인트)의 좌표를 흔든 사본. 그 밖의 종류는 그대로 (시간은 흔들지
    않는다)."""
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
    """정답을 베껴 (흔들어) 돌려주는 stub Predictor. 정답 조회(ctx.truth)가 있어야 한다."""

    def __init__(self, spec: dict[str, Any], *, version: str, ctx: LoadContext) -> None:
        """spec: oracle-stub.json 내용. version: 예측 레코드에 적을 모델 버전.

        Raises:
            ModelUnavailableError: ctx.truth가 None일 때 (운영 환경).
        """
        if ctx.truth is None:
            raise ModelUnavailableError("oracle-stub은 정답이 있어야 합니다 (평가·테스트 전용)")
        self.task = spec["task"]
        self.name = f"trained-{self.task}"  # Predictor.name (기본 어댑터와 겹치지 않게)
        self.version = version
        self.classes = set(spec["classes"])
        self.jitter = float(spec["jitter_px"])
        self.seed = int(spec["seed"])
        self.confidence = float(spec.get("confidence", 0.9))
        self.truth = ctx.truth
        self.ctx = ctx

    def run(self, clip: Clip) -> list[LabelRecord]:
        """클립의 (세션, 스트림) 정답 중 과제·학습 클래스에 맞는 것을 모델 예측으로 바꿔 돌려준다.

        영상 파일은 읽지 않는다. 레코드 ID는 `<세션>-<스트림>-trained-<과제>-<버전 태그>-<번호>`로
        결정적이다 (같은 버전·같은 정답이면 같은 ID). 난수는 run마다 seed로 다시 시작한다.
        """
        rng = np.random.default_rng(self.seed)
        prefix = f"{clip.session_id}-{clip.stream_id}-{self.name}-{version_tag(self.version)}"
        out: list[LabelRecord] = []
        # label_id 순으로 정렬해 번호·흔들림을 결정적으로 만든다
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
                        "confidence": self.confidence,
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
    """oracle-stub 산출물(JSON) 로더."""

    def load(self, artifact: Path, *, version: str, ctx: LoadContext) -> Predictor:
        """산출물을 읽어 `OracleStubPredictor`를 만든다 (정답이 없으면 ModelUnavailableError)."""
        spec: dict[str, Any] = json.loads(artifact.read_text(encoding="utf-8"))
        return OracleStubPredictor(spec, version=version, ctx=ctx)


# 학습기·로더 등록부 (이름 → 객체). 같은 이름으로 짝지어 둔다
TRAINERS: dict[str, Trainer] = {"oracle-stub": OracleStubTrainer()}
LOADERS: dict[str, ModelLoader] = {"oracle-stub": OracleStubLoader()}
