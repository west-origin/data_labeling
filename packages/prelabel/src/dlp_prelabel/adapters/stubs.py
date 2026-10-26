"""CPU용 stub Predictor.

- Oracle*: 정답 라벨을 (흔들림을 넣어) 그대로 돌려준다. 실제 모델 대신 CI와 다른 모듈 개발에 쓴다.
- Unavailable*: 실제 모델을 아직 연동하지 못한 기능. 빈 결과를 내고 이유를 남긴다.
- `ScriptedToolDetector`: 고정 클래스 도구 탐지기(`yolox_tools.ToolDetector`)의 CPU stub. 호출
  순서대로 정해 둔 탐지를 낸다 (어댑터·러너 시험용).

CLAUDE.md 규칙: 새 모델은 공통 Predictor 어댑터와 CPU stub을 함께 둔다. 미연동 자리는
"TODO(real-model)" 주석으로 표시한다 (`make todo-models`). 관련: WP8, ADR 0008.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

from dlp_models.yolox import YoloxDetection
from dlp_prelabel.common import model_label
from dlp_schema.episode import version_tag
from dlp_schema.labels import (
    BoxTrackPayload,
    KeypointTrackPayload,
    LabelRecord,
)
from dlp_schema.predictor import Clip


class OraclePredictor:
    """정답 라벨 중 kinds에 해당하는 것을 좌표 흔들림(jitter_px)을 넣어 모델 출처로 돌려준다.

    키포인트 트랙은 모든 점의 x·y에, 박스 트랙은 x에만 정규분포 흔들림(표준편차 jitter_px)을
    넣는다. 키프레임의 손 중심 3D 점(world, MediaPipe hand_world_landmarks 흉내, ADR 0044)은 정답
    그대로 낸다 (정답 3D를 아는 픽스처 `dlp_fixtures.hand3d`가 채운다. 없는 정답이면 없다). 다른
    종류는 그대로 낸다. 같은 seed면 결과가 같다 (결정적). version은 `oracle-<이름>-j<jitter_px>`라
    흔들림을 바꾸면 새 버전이 된다 (재실행·지우기 시험에 쓴다).
    """

    def __init__(
        self,
        name: str,
        truth: list[LabelRecord],
        kinds: tuple[str, ...],
        *,
        jitter_px: float = 0.0,
        seed: int = 0,
        ontology_version: str = "1.0.0",
        now: datetime,
    ) -> None:
        """Args:
        name: 어댑터 이름 (라벨 ID 접두사, 실제 어댑터 이름과 같게 주면 그 단계를 흉내 낸다).
        truth: 정답 라벨 (합성 픽스처 `dlp_fixtures`의 출력).
        kinds: 돌려줄 라벨 종류 (예: ("keypoint_track",)).
        jitter_px: 좌표 흔들림 표준편차(px). 0이면 그대로.
        seed: 난수 시드. ontology_version, now: 새 라벨에 넣을 값.
        """
        self.name = name
        self.version = f"oracle-{name}-j{jitter_px}"
        self.truth = [x for x in truth if x.kind in kinds]
        self.jitter, self.seed = jitter_px, seed
        self.ontology_version, self.now = ontology_version, now

    def run(self, clip: Clip) -> list[LabelRecord]:
        """clip의 세션·스트림으로 정답을 다시 써서 모델 라벨로 돌려준다.

        시각(t_start/t_end, 키프레임)은 정답 그대로다. 신뢰도는 0.9 고정. clip.video는 읽지
        않는다.
        """
        rng = np.random.default_rng(self.seed)
        out: list[LabelRecord] = []
        for i, label in enumerate(self.truth):
            p = label.payload
            if isinstance(p, KeypointTrackPayload):
                p = p.model_copy(
                    update={
                        "keyframes": tuple(
                            f.model_copy(
                                update={
                                    "points": tuple(
                                        pt.model_copy(
                                            update={
                                                "x": pt.x + rng.normal(0, self.jitter)
                                                if self.jitter
                                                else pt.x,
                                                "y": pt.y + rng.normal(0, self.jitter)
                                                if self.jitter
                                                else pt.y,
                                            }
                                        )
                                        for pt in f.points
                                    )
                                }
                            )
                            for f in p.keyframes
                        )
                    }
                )
            elif isinstance(p, BoxTrackPayload):
                p = p.model_copy(
                    update={
                        "keyframes": tuple(
                            k.model_copy(
                                update={
                                    "x": k.x + (rng.normal(0, self.jitter) if self.jitter else 0.0)
                                }
                            )
                            for k in p.keyframes
                        )
                    }
                )
            out.append(
                model_label(
                    label_id=f"{clip.session_id}-{clip.stream_id}-{self.name}-{version_tag(self.version)}-{i:03d}",
                    session_id=clip.session_id,
                    stream_id=clip.stream_id,
                    t_start_ms=label.t_start_ms,
                    t_end_ms=label.t_end_ms,
                    ontology_version=self.ontology_version,
                    model_version=self.version,
                    confidence=0.9,
                    payload=p,
                    now=self.now,
                )
            )
        return out


class UnavailablePredictor:
    """아직 실제 모델이 없는 기능의 자리. 빈 결과를 낸다 (stub)."""

    version = "unavailable"

    def __init__(self, name: str, reason: str) -> None:
        """name: 기능 이름. reason: 왜 없는지 (CLI·테스트가 보여 준다)."""
        self.name = name
        self.reason = reason

    def run(self, clip: Clip) -> list[LabelRecord]:
        """항상 빈 목록."""
        return []


# TODO(real-model): 도구 작용부·파지부 부분 마스크 (mask_track, part=작용부). 객체 전체 모양
#   마스크는 마스크 단계(`dlp_prelabel.masks`, SAM 2.1 tiny, ADR 0059)가 낸다. 부분은 그 마스크
#   안에서 작용부를 가리키는 프롬프트(부분 점·부분 박스)나 부분 분할 모델이 필요하다. 아직 없다.
#   이 부분 마스크와 메트릭 깊이로 작용부 3D 궤적을 만들어야 도구-표면 접촉·커버리지(WP9)가 실제
#   영상에서 나온다 (relations.yaml tool_surface의 TODO).
TOOL_PART_MASKS = UnavailablePredictor(
    "tool_part_masks", "도구 작용부·파지부 부분 마스크 미연동 (전체 모양 마스크는 masks 단계)"
)

# TODO(real-model): 루프 닫힘·묶음 조정이 있는 외부 시각-관성 SLAM 엔진 (바디캠 6자유도).
#   지금은 내장 IMU 시각 오도메트리(`dlp_prelabel.campose.ImuVisualOdometry`, ADR 0046)만 있어
#   거리에 비례해 드리프트가 쌓인다. 후보는 Basalt(BSD-3, CPU, C++ 빌드, 카메라-IMU 외부 파라미터
#   캘리브레이션 필요)이고 `campose.CameraPoseEstimator` 어댑터로 붙인다. ORB-SLAM3·OpenVINS
#   (GPL-3)와 MASt3R·DUSt3R(CC BY-NC)는 쓰지 않는다 (ADR 0046 라이선스 조사).
SLAM_ENGINE = UnavailablePredictor(
    "slam_engine",
    "외부 시각-관성 SLAM 엔진 미연동 (내장 IMU 시각 오도메트리만, 드리프트 보정 없음)",
)

# TODO(real-model): 영상만으로 접촉을 판정하는 학습 분류기. 장갑 세션의 접촉 구간을 정답으로
#   재학습 루프(`dlp train run contact`, WP13)에서 학습한다. 루프는 있지만 실제 학습기가 없다
#   (dlp_train.trainers). 지금은 contact.video_contact_intervals 휴리스틱을 쓴다.
LEARNED_CONTACT = UnavailablePredictor(
    "learned_contact", "재학습 루프의 실제 접촉 분류기 학습기 미연동 (dlp train run contact)"
)

# TODO(real-model): 사내 VLM 서버 없이 도는 로컬 장면 분류기 (ADR 0055). 상업 사용 가능한 가중치를
#   찾지 못했다 (Places365 계열은 비상업). 지금은 사내 VLM(`scene.VlmScene`)·사람이 준
#   장면(--scene)·고정 기본값(unknown, 전체 질의)뿐이다.
LOCAL_SCENE = UnavailablePredictor(
    "local_scene", "로컬 장면 분류기 미연동 (사내 VLM 또는 --scene, 없으면 unknown·전체 도구 질의)"
)

# TODO(real-model): 고정 클래스 도구 탐지 가중치 (ADR 0060). 공개 데이터
#   (config/pretrain/objects.yaml)로 yolox_s를 미세조정한 ONNX가 아직 없다 (1단계 학습 전).
#   어댑터(`yolox_tools.YoloxTools`)와 전후처리(`dlp_models.yolox`)는 있고, 정책
#   `prelabel.yaml models.closed_vocab_tools`가 null이라 단계가 없다.
#   학습·우리 골든셋 게이트 통과 뒤 models.yaml에 등록하고 정책에 이름을 넣는다.
CLOSED_VOCAB_WEIGHTS = UnavailablePredictor(
    "closed_vocab_weights",
    "고정 클래스 도구 탐지(YOLOX 공개 데이터 미세조정) 가중치 없음, closed_tools 단계 꺼짐 "
    "(ADR 0060)",
)

UNAVAILABLE = (TOOL_PART_MASKS, SLAM_ENGINE, LEARNED_CONTACT, LOCAL_SCENE, CLOSED_VOCAB_WEIGHTS)


class ScriptedToolDetector:
    """고정 클래스 도구 탐지기의 CPU stub: `detect`를 부른 순서대로 frames의 탐지를 낸다.

    frames를 다 쓰면 빈 목록이다. 이미지는 보지 않는다. 문턱은 실제 탐지기처럼 적용한다 (클래스
    번호의 문턱보다 낮은 탐지는 빠진다). `calls`에 받은 (이미지 크기, 문턱)을 남긴다.
    """

    def __init__(self, frames: Sequence[Sequence[YoloxDetection]]) -> None:
        """frames: 호출마다 낼 탐지 목록 (클래스 번호는 정책 classes 순서)."""
        self.frames = [list(f) for f in frames]
        self.calls: list[tuple[tuple[int, int], tuple[float, ...]]] = []

    def detect(self, rgb: NDArray[np.uint8], thresholds: Sequence[float]) -> list[YoloxDetection]:
        """다음 차례 탐지 중 문턱 이상인 것."""
        i = len(self.calls)
        self.calls.append(((int(rgb.shape[1]), int(rgb.shape[0])), tuple(thresholds)))
        found = self.frames[i] if i < len(self.frames) else []
        return [d for d in found if d.score >= thresholds[d.class_index]]
