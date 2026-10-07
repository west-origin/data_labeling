"""프리라벨 정책 (config/policies/prelabel.yaml).

`dlp prelabel run`의 모든 단계가 읽는다. 각 어댑터·단계는 자기 정책 절의
해시(`PrelabelPolicy.digest`)를 모델 버전에 넣어, 그 절의 값이 바뀌면 다시 돌고 검수 전인 이전
결과를 지운다 (ADR 0015). 해시는 파싱·검증된 값(`model_dump`)으로 계산하므로 YAML 주석·순서는 해시에
영향이 없다.

절 → 쓰는 곳:
- models → 모든 어댑터 (`config/models.yaml` 이름), hands → `MediaPipeHands`, body → `RtmPose`,
  objects → `MediaPipeObjects`, open_vocab_objects → `OwlObjects`, depth → `lift3d.DepthLifter`,
  contact → `runner._contacts`, wearer_matching → `runner._wearer`, timeline_predictors →
  `runner.run_prelabel` (마스터 타임라인 구간을 내는 배포 모델).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator

from dlp_schema.common import Contract, OntologyId


class ModelRoles(Contract):
    """역할 → config/models.yaml 이름.

    - hands: 손 랜드마크. body_detector: 사람 박스(YOLOX). body: 전신 포즈(RTMPose).
    - coco_objects: COCO 객체. open_vocab: OWLv2 ONNX. open_vocab_tokenizer: OWLv2 토크나이저.
    - depth: 메트릭 깊이.

    이 절은 어떤 `digest`에도 들어가지 않는다 (가중치가 바뀌면 `resolve`의 파일 해시로 버전이
    바뀐다).
    """

    hands: str
    body_detector: str
    body: str
    coco_objects: str
    open_vocab: str
    open_vocab_tokenizer: str
    depth: str


class HandsPolicy(Contract):
    """손 21관절(MediaPipe) 정책 (`hands` 절).

    min_score: 손 탐지 최소 신뢰도 (0~1, MediaPipe `min_hand_detection_confidence`).
    input_is_mirrored: 입력이 거울상(셀카)인가. 바디캠은 False라 왼손·오른손 판정을 뒤집는다.
    """

    min_score: float = Field(ge=0, le=1)
    input_is_mirrored: bool


class BodyPolicy(Contract):
    """전신 포즈(YOLOX → RTMPose) 정책 (`body` 절).

    detector_score: YOLOX 사람 박스 점수 하한 (0~1).
    min_score: 사람 하나의 관절 점수 평균 하한. 미만이면 버린다.
    keypoint_score: 관절 점수가 이 미만이면 visibility=1(가려짐 추정), 이상이면 2.
    max_people: 프레임당 사람 수 상한. track_iou / max_gap_ms: 사람 트랙 잇기 IoU 문턱과 최대
    끊김(ms) (`common.track_boxes`).
    """

    detector_score: float = Field(ge=0, le=1)
    min_score: float = Field(ge=0, le=1)
    keypoint_score: float = Field(ge=0, le=1)
    max_people: int = Field(ge=1)
    track_iou: float = Field(gt=0, le=1)
    max_gap_ms: float = Field(ge=0)


class ObjectsPolicy(Contract):
    """COCO 객체 탐지(MediaPipe EfficientDet) 정책 (`objects` 절).

    min_score: 탐지 점수 하한. track_iou / max_gap_ms: 트랙 잇기 (`common.track_boxes`).
    coco_to_ontology: COCO 클래스 이름 → 온톨로지 객체 ID. 없는 클래스는 버린다.
    """

    min_score: float = Field(ge=0, le=1)
    track_iou: float = Field(gt=0, le=1)
    max_gap_ms: float = Field(ge=0)
    coco_to_ontology: dict[str, OntologyId]


class OpenVocabObjectsPolicy(Contract):
    """OWLv2 도구 탐지 정책 (`open_vocab_objects` 절).

    frame_stride_ms: 추론 간격 (ms, 스트림 PTS 기준). CPU 프레임당 수 초라 간격을 둔다.
    min_score: 모든 질의에 같은 점수 문턱. nms_iou: 같은 질의 박스끼리 NMS 문턱. track_iou /
    max_gap_ms: 트랙 잇기. max_gap_ms는 frame_stride_ms보다 커야 트랙이 이어진다.
    queries: 영어 질의 문장 → 온톨로지 객체 ID (최소 1개).
    """

    frame_stride_ms: int = Field(ge=0)
    min_score: float = Field(ge=0, le=1)
    track_iou: float = Field(gt=0, le=1)
    max_gap_ms: float = Field(ge=0)
    nms_iou: float = Field(gt=0, le=1)
    queries: dict[str, OntologyId] = Field(min_length=1)


# hand21 번호 → 온톨로지 hand_joints 이름 (3D 궤적 part)
# 3D로 올릴 수 있는 관절만 이름이 있다 (온톨로지 hand_joints와 같아야 한다). 0 손목, 4 엄지 끝,
# 8 검지 끝, 12 중지 끝, 16 약지 끝, 20 새끼 끝
HAND21_JOINTS = {
    0: "wrist", 4: "thumb_tip", 8: "index_tip", 12: "middle_tip", 16: "ring_tip", 20: "pinky_tip",
}  # fmt: skip


class DepthPolicy(Contract):
    """메트릭 깊이 3D 올리기 정책 (`depth` 절).

    frame_stride_ms: 깊이 추론 간격 (ms). default_hfov_deg: 캘리브레이션이 없을 때 가로 화각(도).
    patch_radius_px: 깊이 중앙값을 읽을 패치 반지름. min_depth_m / max_depth_m: 받아들일 깊이
    범위(m).
    hand_points: 올릴 hand21 관절 번호 (`HAND21_JOINTS`에 이름이 있는 번호만).
    confidence: 3D 궤적 라벨 신뢰도 (단안 깊이라 낮게).
    """

    frame_stride_ms: int = Field(ge=0)
    default_hfov_deg: float = Field(gt=0, lt=180)
    patch_radius_px: int = Field(ge=0)
    min_depth_m: float = Field(gt=0)
    max_depth_m: float = Field(gt=0)
    hand_points: tuple[int, ...] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)

    @field_validator("hand_points")
    @classmethod
    def _named_points(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        # 3D 궤적의 part는 온톨로지 hand_joints 이름이어야 하므로 이름이 있는 번호만 받는다
        """hand_points가 모두 `HAND21_JOINTS`에 있는지 검사한다. 없으면 ValueError."""
        bad = sorted(set(v) - set(HAND21_JOINTS))
        if bad:
            raise ValueError(
                f"hand_points {bad}: 이름이 없는 hand21 번호 (허용 {sorted(HAND21_JOINTS)})"
            )
        return v


class GloveContactPolicy(Contract):
    """장갑 압력 접촉 정책 (`contact.glove` 절). 단위는 장갑 압력 합(정규화 값).

    on_threshold: 이 값을 넘으면 접촉 시작. off_threshold: 이 값 아래로 내려가면 종료
    (히스테리시스).
    min_duration_ms: 이보다 짧은 구간은 버린다. on_threshold > off_threshold여야 의미가 있지만
    검증하지 않는다.
    """

    on_threshold: float
    off_threshold: float
    min_duration_ms: float


class VideoContactPolicy(Contract):
    """영상 접촉 휴리스틱 정책 (`contact.video` 절).

    max_distance_px: 손가락 끝과 박스 사이 거리 문턱(px, 스트림 해상도 기준).
    min_duration_ms: 최소 구간 길이. merge_gap_ms: 같은 대상 프레임이 이 이하로 끊기면 잇는다.
    box_max_gap_ms: 박스 키프레임 사이를 보간할 최대 간격.
    """

    max_distance_px: float
    min_duration_ms: float
    merge_gap_ms: float
    box_max_gap_ms: float = Field(ge=0)


class ContactConfidence(Contract):
    """접촉 구간 출처별 신뢰도. 장갑과 영상이 모두 접촉이면 fused.

    검수 우선순위(`review.yaml contact_mismatch_max_confidence`)가 이 값의 대소를 쓴다:
    glove·video < 그 문턱 < fused.
    """

    fused: float = Field(ge=0, le=1)
    glove: float = Field(ge=0, le=1)
    video: float = Field(ge=0, le=1)


class ContactPolicy(Contract):
    """접촉 단계 정책 (`contact` 절, `runner.contact_version`의 해시 대상).

    unresolved_target_id: 장갑만 잡아 대상을 모르는 접촉의 target_id. 관계(`relations.yaml
    unresolved_target_ids`)·행동 단계가 이 값을 버린다. 두 YAML을 함께 바꿔야 한다.
    """

    glove: GloveContactPolicy
    video: VideoContactPolicy
    confidence: ContactConfidence
    unresolved_target_id: str = Field(min_length=1)


class WearerPolicy(Contract):
    """착용자 매칭 정책 (`wearer_matching` 절).

    rate_hz: 두 신호를 다시 표본할 격자 주파수. min_correlation: 착용자로 볼 최소 피어슨 상관
    (-1~1).
    min_overlap_samples: 겹치는 길이가 이 샘플 수(격자 기준)보다 짧으면 비교하지 않는다.
    """

    rate_hz: float = Field(gt=0)
    min_correlation: float = Field(ge=-1, le=1)
    min_overlap_samples: int = Field(ge=2)


class PrelabelPolicy(Contract):
    """`config/policies/prelabel.yaml` 전체.

    version: 정책 파일 형식 버전. 나머지 필드는 각 절 클래스 docstring 참고.
    """

    version: int
    models: ModelRoles
    hands: HandsPolicy
    body: BodyPolicy
    objects: ObjectsPolicy
    open_vocab_objects: OpenVocabObjectsPolicy
    contact: ContactPolicy
    wearer_matching: WearerPolicy
    depth: DepthPolicy
    # 마스터 타임라인 구간(stream_id 없음)을 내는 예측기 이름. 기준 스트림에서 세션당 한 번만 돌린다
    timeline_predictors: tuple[str, ...] = ()

    def digest(self, *sections: str) -> str:
        """정책 절들의 짧은 해시. 모델 버전에 붙여 정책 값이 바뀌면 다시 돌게 한다.

        Args:
            *sections: 절 이름 (예: "hands", "contact"). 속성 이름과 같아야 한다 (없으면
                AttributeError).

        Returns:
            {절: model_dump(json)}를 키 정렬 JSON으로 만든 sha256의 앞 8자. 파싱된 값만 보므로 YAML
                주석·
            키 순서·따옴표 차이는 해시를 바꾸지 않는다.
        """
        # 파싱·검증된 값만 해시한다 (YAML 원문이 아니다). sort_keys로 키 순서와 무관하게 만든다
        data = {name: getattr(self, name).model_dump(mode="json") for name in sections}
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:8]


def load_policy(root: Path) -> PrelabelPolicy:
    """`<root>/config/policies/prelabel.yaml`을 읽어 검증한다.

    Raises:
        FileNotFoundError: 파일이 없을 때. pydantic.ValidationError: 값 범위·형식이 틀릴 때.
    """
    data: Any = yaml.safe_load((root / "config" / "policies" / "prelabel.yaml").read_text("utf-8"))
    return PrelabelPolicy.model_validate(data)
