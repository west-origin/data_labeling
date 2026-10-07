"""프레임 단위 탐지 결과와 탐지기 인터페이스.

WP5. 모든 프라이버시 탐지기(`detectors/`)가 따르는 공통 꼴이다. `pipeline.detect_video`가 프레임마다
`FrameDetector.detect`를 부르고, 결과 `Detection`을 `tracker`가 트랙으로 묶는다.

- `Image`: 탐지기 입력 이미지 타입 (H, W, 3) RGB uint8.
- `Detection`: 한 프레임의 탐지 하나 (대상, 박스, 점수, 탐지기 이름).
- `FrameDetector`: 탐지기 프로토콜 (`name`, `version`, `detect`).
- `Resettable`: 영상 사이에 상태를 지워야 하는 탐지기 프로토콜.

새 탐지기는 이 프로토콜을 따르고 CPU stub(예: `detectors.oracle.OracleDetector`)과 함께 추가한다
(CLAUDE.md 규칙).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from dlp_privacy.geometry import Box

Image = NDArray[np.uint8]  # (H, W, 3) RGB


@dataclass(frozen=True)
class Detection:
    """한 프레임의 탐지 하나."""

    target: str  # 프라이버시 사전 ID (반사면 영역 등 중간 결과는 사전 밖 ID일 수 있다)
    # 픽셀 좌표 박스 (x, y, w, h). 탐지기에 넘긴 이미지 기준 (reflection은 원 이미지로 되돌린다).
    box: Box
    # 신뢰도 0~1 (탐지기마다 척도가 다르다: YuNet 원점수, OWLv2는 원점수/score_full, 코드는 고정값).
    score: float
    # 낸 탐지기 이름 (정책 detectors의 키). 탐지기 불일치(disagreement) 판정에 쓴다.
    detector: str


@runtime_checkable
class Resettable(Protocol):
    """영상 사이에 지워야 할 상태(프레임 간격 캐시 등)를 가진 탐지기."""

    def reset(self) -> None:
        """새 영상을 시작하기 전에 상태를 지운다 (`pipeline.detect_video`가 영상마다 부른다)."""
        ...


class FrameDetector(Protocol):
    """프레임 탐지기 프로토콜."""

    # 정책 detectors의 키 (Detection.detector에 들어간다).
    name: str
    # 탐지기 버전 (가중치 버전 등). 탐지 모델 버전 문자열(pipeline.model_version)에 들어가므로
    # 결과가 바뀌는 변경이면 이 값도 바뀌어야 다시 탐지한다.
    version: str

    def detect(self, image: Image, t_ms: int, threshold: float) -> list[Detection]:
        """한 프레임에서 대상을 찾는다.

        Args:
            image: (H, W, 3) RGB uint8 프레임.
            t_ms: 그 프레임의 스트림 PTS 시각(정수 ms). 프레임 간격 캐시·결정적 난수에 쓴다.
                파이프라인은 한 영상 안에서 시각이 증가하는 순서로 부른다.
            threshold: 신뢰도 문턱. 이 미만은 내지 않는다 (탐지기마다 적용 방식이 다르다).

        Returns:
            탐지 목록 (없으면 빈 목록).
        """
        ...
