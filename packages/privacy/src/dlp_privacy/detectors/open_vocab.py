"""오픈 보캐뷸러리 탐지 (OWLv2): 문서·화면·사진물·문패·송장과 반사면 영역.

CPU에서 프레임당 수 초가 걸려 frame_stride_ms마다 한 번만 추론한다. 그 사이 프레임에는 마지막
추론 결과를 그대로 돌려준다. 같은 시각에 여러 번 불려도(대상 탐지 + reflection 탐지기의 영역
탐지) 추론은 한 번이다. 파이프라인은 프레임을 시간 순서로 넘긴다는 가정에 기댄다.
"""

from __future__ import annotations

from typing import Protocol

from dlp_models.owlv2 import OwlDetection
from dlp_privacy.detection import Detection, Image
from dlp_privacy.geometry import Box


class QueryDetector(Protocol):
    """dlp_models.owlv2.Owlv2와 같은 꼴 (테스트에서 가짜로 바꿔 끼운다)."""

    queries: list[str]

    def detect(self, image: Image, thresholds: list[float]) -> list[OwlDetection]: ...


class OpenVocabDetector:
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
        if len(targets) != len(model.queries):
            raise ValueError("질의 수와 대상 수가 다릅니다")
        self.name = name
        self.version = f"{version}-s{frame_stride_ms}"
        self.model = model
        self.targets = targets
        self.stride = frame_stride_ms
        self.score_threshold = score_threshold
        self.score_full = score_full
        self._last_run: int | None = None
        self._cache: list[tuple[str, Box, float]] = []

    def _infer(self, image: Image) -> list[tuple[str, Box, float]]:
        thresholds = [self.score_threshold] * len(self.targets)
        return [
            (self.targets[d.query_index], Box(*d.box), min(1.0, d.score / self.score_full))
            for d in self.model.detect(image, thresholds)
        ]

    def detect(self, image: Image, t_ms: int, threshold: float) -> list[Detection]:
        due = (
            self._last_run is None
            or t_ms < self._last_run  # 새 영상
            or t_ms - self._last_run >= self.stride
        )
        if due:
            self._cache = self._infer(image)
            self._last_run = t_ms
        return [Detection(tg, box, s, self.name) for tg, box, s in self._cache if s >= threshold]
