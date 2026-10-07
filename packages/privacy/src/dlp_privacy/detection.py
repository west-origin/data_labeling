"""프레임 단위 탐지 결과와 탐지기 인터페이스."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from dlp_privacy.geometry import Box

Image = NDArray[np.uint8]  # (H, W, 3) RGB


@dataclass(frozen=True)
class Detection:
    target: str  # 프라이버시 사전 ID (반사면 영역 등 중간 결과는 사전 밖 ID일 수 있다)
    box: Box
    score: float
    detector: str


@runtime_checkable
class Resettable(Protocol):
    """영상 사이에 지워야 할 상태(프레임 간격 캐시 등)를 가진 탐지기."""

    def reset(self) -> None: ...


class FrameDetector(Protocol):
    name: str
    version: str

    def detect(self, image: Image, t_ms: int, threshold: float) -> list[Detection]: ...
