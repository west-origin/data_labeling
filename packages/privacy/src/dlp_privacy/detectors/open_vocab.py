"""오픈 보캐뷸러리 탐지 (OWLv2): 문서·화면·사진물·문패·송장과 반사면 영역.

CPU에서 프레임당 수 초가 걸려 frame_stride_ms마다 한 번만 추론한다. 그 사이 프레임에는 마지막
추론 결과를 그대로 돌려준다. 같은 시각에 여러 번 불려도(대상 탐지 + reflection 탐지기의 영역
탐지) 추론은 한 번이다. 파이프라인은 프레임을 시간 순서로 넘긴다는 가정에 기댄다.

WP5, ADR 0009. 모델 실행은 `dlp_models.owlv2.Owlv2`(공용 ONNX 런타임)가 한다. 정책:
privacy.yaml `detectors.open_vocab` (model, tokenizer, frame_stride_ms, score_threshold, score_full,
queries). 버전은 `<가중치 버전>-s<frame_stride_ms>`이다 (간격이 바뀌면 결과가 바뀌므로).
주의: 간격 사이에 잠깐 나타났다 사라진 대상은 놓칠 수 있다 (블러 전수 검수가 보완).
"""

from __future__ import annotations

from typing import Protocol

from dlp_models.owlv2 import OwlDetection
from dlp_privacy.detection import Detection, Image
from dlp_privacy.geometry import Box


class QueryDetector(Protocol):
    """dlp_models.owlv2.Owlv2와 같은 꼴 (테스트에서 가짜로 바꿔 끼운다)."""

    # 질의 문장 목록 (OwlDetection.query_index가 가리킨다)
    queries: list[str]

    def detect(self, image: Image, thresholds: list[float]) -> list[OwlDetection]:
        """질의별 원점수 문턱으로 탐지한다. thresholds[i]는 queries[i]의 문턱."""
        ...


class OpenVocabDetector:
    """OWLv2 질의 탐지를 대상 탐지로 바꾸는 `FrameDetector` (프레임 간격 캐시, `Resettable`)."""

    def __init__(
        self,
        name: str,
        model: QueryDetector,
        targets: list[str],
        *,
        version: str,
        frame_stride_ms: int,
        score_threshold: float,
        score_full: float,
    ) -> None:
        """
        Args:
            name: 정책 탐지기 이름.
            model: 질의 탐지 모델 (`Owlv2` 또는 테스트용 가짜).
            targets: 질의 i → 대상 ID (model.queries와 길이가 같아야 한다).
            version: 가중치 버전 (`registry.resolve`).
            frame_stride_ms: 추론 간격 ms (0이면 매 프레임).
            score_threshold: OWLv2 원점수 문턱 (모델 쪽 필터).
            score_full: 신뢰도 정규화 기준 (원점수 / score_full, 최대 1).

        Raises:
            ValueError: 질의 수와 대상 수가 다를 때.
        """
        if len(targets) != len(model.queries):
            raise ValueError("질의 수와 대상 수가 다릅니다")
        self.name = name
        self.version = f"{version}-s{frame_stride_ms}"
        self.model = model
        self.targets = targets
        self.stride = frame_stride_ms
        self.score_threshold = score_threshold
        self.score_full = score_full
        # 마지막 추론 시각 (ms). None이면 아직 추론하지 않았다.
        self._last_run: int | None = None
        # 마지막 추론 결과: (대상, 박스, 정규화 신뢰도). 문턱 적용 전 전체.
        self._cache: list[tuple[str, Box, float]] = []

    def reset(self) -> None:
        """새 영상을 시작할 때 부른다. 앞 영상의 추론 결과를 다음 영상에 쓰지 않게 한다."""
        self._last_run = None
        self._cache = []

    def _infer(self, image: Image) -> list[tuple[str, Box, float]]:
        """한 번 추론해 (대상, 박스, 신뢰도) 목록을 만든다. 신뢰도 = min(1, 원점수 / score_full)."""
        thresholds = [self.score_threshold] * len(self.targets)
        return [
            (self.targets[d.query_index], Box(*d.box), min(1.0, d.score / self.score_full))
            for d in self.model.detect(image, thresholds)
        ]

    def detect(self, image: Image, t_ms: int, threshold: float) -> list[Detection]:
        """추론할 때가 됐으면 추론하고, 아니면 캐시를 쓴다. 신뢰도가 문턱 이상인 것만 낸다.

        문턱은 캐시에 매번 적용하므로 같은 시각에 다른 문턱으로 불러도(reflection 탐지기 등)
        추론을 다시 하지 않는다.
        """
        due = (
            self._last_run is None
            or t_ms < self._last_run  # 새 영상
            or t_ms - self._last_run >= self.stride
        )
        if due:
            self._cache = self._infer(image)
            self._last_run = t_ms
        return [Detection(tg, box, s, self.name) for tg, box, s in self._cache if s >= threshold]
