"""CPU용 stub Predictor.

- Oracle*: 정답 라벨을 (흔들림을 넣어) 그대로 돌려준다. 실제 모델 대신 CI와 다른 모듈 개발에 쓴다.
- Unavailable*: 실제 모델을 아직 연동하지 못한 기능. 빈 결과를 내고 이유를 남긴다.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

from dlp_prelabel.common import model_label
from dlp_schema.labels import (
    BoxTrackPayload,
    KeypointTrackPayload,
    LabelRecord,
)
from dlp_schema.predictor import Clip


class OraclePredictor:
    """정답 라벨 중 kinds에 해당하는 것을 좌표 흔들림(jitter_px)을 넣어 모델 출처로 돌려준다."""

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
        self.name = name
        self.version = f"oracle-{name}-j{jitter_px}"
        self.truth = [x for x in truth if x.kind in kinds]
        self.jitter, self.seed = jitter_px, seed
        self.ontology_version, self.now = ontology_version, now

    def run(self, clip: Clip) -> list[LabelRecord]:
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
                    label_id=f"{clip.session_id}-{clip.stream_id}-{self.name}-{i:03d}",
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
        self.name = name
        self.reason = reason

    def run(self, clip: Clip) -> list[LabelRecord]:
        return []


# TODO(real-model): 도구 작용부·파지부 마스크 (mask_track, part=작용부). SAM 2 마스크 추적과
#   오픈 보캐뷸러리 탐지가 필요하다 (GPU, Hugging Face 접근). 커버리지 계산(WP9)이 이 결과를 쓴다.
TOOL_PART_MASKS = UnavailablePredictor(
    "tool_part_masks", "SAM 2 + 오픈 보캐뷸러리 탐지 필요 (GPU, Hugging Face)"
)

# TODO(real-model): 바디캠 6자유도 궤적 (trajectory3d, entity_id="camera"). IMU가 있으면 시각-관성
#   SLAM(Basalt BSD-3, ORB-SLAM3 GPL-3.0), 없으면 시각 SLAM(DROID-SLAM). C++ 빌드와 GPU가 필요하다.
CAMERA_POSE = UnavailablePredictor("camera_pose", "시각-관성 SLAM 미연동 (C++ 빌드, GPU)")

# TODO(real-model): 단안 메트릭 깊이로 손·객체·작용부를 3D로 올리기
#   (Depth Anything V2 Small, Apache 2.0).
#   Hugging Face 접근이 막혀 가중치를 받을 수 없다.
MONO_DEPTH = UnavailablePredictor("mono_depth", "Depth Anything V2 가중치 접근 불가 (Hugging Face)")

# TODO(real-model): 영상만으로 접촉을 판정하는 학습 분류기. 장갑 세션의 접촉 구간을 정답으로
#   WP13에서 학습한다. 지금은 contact.video_contact_intervals 휴리스틱을 쓴다.
LEARNED_CONTACT = UnavailablePredictor(
    "learned_contact", "학습 데이터(장갑 세션) 누적 후 WP13에서 학습"
)

UNAVAILABLE = (TOOL_PART_MASKS, CAMERA_POSE, MONO_DEPTH, LEARNED_CONTACT)
