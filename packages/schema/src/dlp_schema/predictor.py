"""자동 모델의 공통 인터페이스.

클립 하나를 받아 라벨 레코드를 내는 모델 어댑터(프리라벨 어댑터, 배포된 재학습 모델)는
Predictor를 구현하고 결과를 LabelRecord로 낸다. 각 모델에는 CPU에서 바로 도는 stub 구현을 함께
두며, CI와 다른 모듈 개발은 stub으로 진행한다.
프레임 단위 블러 탐지기(`dlp_privacy.detection.FrameDetector`), 행동 VLM
(`dlp_actions.vlm.VlmClient`), 깊이 모델(`dlp_prelabel.lift3d.DepthModel`)은 단계별 인터페이스를
따로 둔다.

위치
    WP8(프리라벨 어댑터)의 공통 계약. `dlp prelabel run`이 프리라벨 어댑터를, `dlp privacy detect`가
    배포된 재학습 블러 모델을 이 인터페이스로 부른다. 재학습 루프(`dlp_train`)의 로더도 학습
    산출물을 Predictor로 읽는다 (`dlp_train.trainers.ModelLoader`).
    실제 모델 가중치는 `config/models.yaml`과 `dlp_models` 레지스트리가 관리한다.

주요 이름
    - `ModelUnavailableError`: 가중치·라이브러리가 없어 실제 모델을 쓸 수 없을 때 어댑터가 던진다.
    - `Clip`: 모델 입력 단위 (세션의 한 스트림 영상, 선택적으로 시간 범위).
    - `Predictor`: `name`·`version`·`run(clip)`을 가진 구조적 프로토콜.

주의
    - 어댑터가 낸 `LabelRecord`는 출처 `model`이어야 하며 `model_version`과 `confidence`가 필수다
      (`labels.LabelRecord` 검증). 모델 출처 라벨 ID에는 `episode.version_tag(모델 버전)`을 넣는다.
    - 공간 라벨의 키프레임 시각은 그 스트림 영상의 PTS ms다 (ADR 0019).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from dlp_schema.labels import LabelRecord


class ModelUnavailableError(RuntimeError):
    """가중치·라이브러리가 없어 모델을 쓸 수 없다.

    어댑터가 실제 모델을 불러오다 실패하면 던진다. 호출자는 stub으로 대체하거나 단계를 건너뛰는
    식으로 처리한다 (어떻게 할지는 각 단계의 정책이 정한다).
    """


@dataclass(frozen=True)
class Clip:
    """모델에 넘기는 입력 단위: 세션의 한 스트림 영상과 선택적 시간 범위.

    Attributes:
        session_id: 세션 ID.
        stream_id: 영상이 속한 스트림 ID (예: `bodycam`, 3인칭 스트림).
        video: 로컬 영상 파일 경로 (블러본 또는 원본. 원본이면 호출자가 감사 저장소로 받아 와야
            한다, ADR 0020).
        t_start_ms: 처리 시작 시각 (그 스트림 PTS ms, 포함). None이면 영상 처음부터.
        t_end_ms: 처리 끝 시각 (그 스트림 PTS ms). None이면 영상 끝까지.
    """

    session_id: str
    stream_id: str
    video: Path
    t_start_ms: int | None = None
    t_end_ms: int | None = None


class Predictor(Protocol):
    """자동 모델 어댑터의 구조적 인터페이스 (상속 없이 같은 속성·메서드만 있으면 된다).

    Attributes:
        name: 어댑터 이름 (예: `mediapipe_hands`, `oracle_stub`).
        version: 모델 버전 문자열. 정책 절 해시를 붙여 정책이 바뀌면 버전도 바뀌게 한다 (CLAUDE.md).
    """

    name: str
    version: str

    def run(self, clip: Clip) -> list[LabelRecord]:
        """클립 하나를 처리해 라벨 레코드 목록을 돌려준다 (DB에 쓰지 않는다. 저장은 호출자 몫)."""
        ...
