"""자동 모델의 공통 인터페이스.

모든 모델(탐지, 포즈, SLAM, VLM)은 Predictor를 구현하고 결과를 LabelRecord로 낸다.
각 모델에는 CPU에서 바로 도는 stub 구현을 함께 두며, CI와 다른 모듈 개발은 stub으로 진행한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from dlp_schema.labels import LabelRecord


class ModelUnavailableError(RuntimeError):
    """가중치·라이브러리가 없어 모델을 쓸 수 없다."""


@dataclass(frozen=True)
class Clip:
    session_id: str
    stream_id: str
    video: Path
    t_start_ms: int | None = None
    t_end_ms: int | None = None


class Predictor(Protocol):
    name: str
    version: str

    def run(self, clip: Clip) -> list[LabelRecord]: ...
