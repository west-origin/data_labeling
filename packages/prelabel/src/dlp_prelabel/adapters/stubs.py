"""CPU용 stub Predictor.

- Oracle*: 정답 라벨을 (흔들림을 넣어) 그대로 돌려준다. 실제 모델 대신 CI와 다른 모듈 개발에 쓴다.
- Unavailable*: 실제 모델을 아직 연동하지 못한 기능. 빈 결과를 내고 이유를 남긴다.

CLAUDE.md 규칙: 새 모델은 공통 Predictor 어댑터와 CPU stub을 함께 둔다. 미연동 자리는
"TODO(real-model)" 주석으로 표시한다 (`make todo-models`). 관련: WP8, ADR 0008.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

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
    넣는다. 다른 종류는 그대로 낸다. 같은 seed면 결과가 같다 (결정적). version은
    `oracle-<이름>-j<jitter_px>`라 흔들림을 바꾸면 새 버전이 된다 (재실행·지우기 시험에 쓴다).
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


# TODO(real-model): 도구 작용부·파지부 마스크 (mask_track, part=작용부). 도구 박스는 OWLv2가
#   내므로(adapters/owl_objects.py) 그 박스를 프롬프트로 쓰는 SAM 2.1(tiny, Apache-2.0) 마스크가
#   필요하다. 아직 연동하지 않았다. CPU로도 느리지만 돈다 (OWLv2처럼 프레임 간격 추론, ONNX
#   공개본 있음).
#   이 마스크와 메트릭 깊이로 작용부 3D 궤적을 만들어야 도구-표면 접촉·커버리지(WP9)가 실제 영상에서
#   나온다 (relations.yaml tool_surface의 TODO).
TOOL_PART_MASKS = UnavailablePredictor("tool_part_masks", "SAM 2.1 마스크 미연동")

# TODO(real-model): 바디캠 6자유도 궤적 (trajectory3d, entity_id="camera"). IMU가 있으면 시각-관성
#   SLAM(Basalt BSD-3, CPU), 없으면 메트릭 깊이 + RGB-D 오도메트리(Open3D MIT, CPU)가 후보다.
#   아직 연동하지 않았다. Basalt는 C++ 빌드와 카메라-IMU 캘리브레이션이 필요하다 (GPU는 필요 없다).
CAMERA_POSE = UnavailablePredictor("camera_pose", "카메라 자세(SLAM·RGB-D 오도메트리) 미연동")

# TODO(real-model): 영상만으로 접촉을 판정하는 학습 분류기. 장갑 세션의 접촉 구간을 정답으로
#   재학습 루프(`dlp train run contact`, WP13)에서 학습한다. 루프는 있지만 실제 학습기가 없다
#   (dlp_train.trainers). 지금은 contact.video_contact_intervals 휴리스틱을 쓴다.
LEARNED_CONTACT = UnavailablePredictor(
    "learned_contact", "재학습 루프의 실제 접촉 분류기 학습기 미연동 (dlp train run contact)"
)

UNAVAILABLE = (TOOL_PART_MASKS, CAMERA_POSE, LEARNED_CONTACT)
