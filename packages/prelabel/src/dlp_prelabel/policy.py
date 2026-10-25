"""프리라벨 정책 (config/policies/prelabel.yaml).

`dlp prelabel run`의 모든 단계가 읽는다. 각 어댑터·단계는 자기 정책 절의
해시(`PrelabelPolicy.digest`)를 모델 버전에 넣어, 그 절의 값이 바뀌면 다시 돌고 검수 전인 이전
결과를 지운다 (ADR 0015). 해시는 파싱·검증된 값(`model_dump`)으로 계산하므로 YAML 주석·순서는 해시에
영향이 없다.

절 → 쓰는 곳:
- models → 모든 어댑터 (`config/models.yaml` 이름), hands → `MediaPipeHands`, body → `RtmPose`,
  objects → `MediaPipeObjects`, open_vocab_objects → `OwlObjects`, depth → `lift3d.DepthLifter`,
  camera_pose → `campose`(바디캠 자세), world_frame → `worldframe`(월드 좌표 3D),
  contact → `runner._contacts`, wearer_matching → `runner._wearer`, timeline_predictors →
  `runner.run_prelabel` (마스터 타임라인 구간을 내는 배포 모델), wearer_suppression →
  `MediaPipeObjects`·`OwlObjects`·`RtmPose`(착용자 몸 억제, ADR 0058), arm → `RtmPose`(착용자 팔),
  masks → `masks.MaskMaker`(객체·손 모양 마스크, ADR 0059).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, field_validator, model_validator

from dlp_prelabel.posemath import rotation_problem
from dlp_schema.common import Contract, OntologyId
from dlp_schema.labels import Constraint3DFlag


class TrackFilterPolicy(Contract):
    """박스 트랙 거르기 (`objects.track_filter`·`open_vocab_objects.track_filter`, ADR 0054).

    start_score: 트랙을 여는 점수 (탐지 문턱 min_score보다 크거나 같게, 히스테리시스).
    min_keyframes / min_duration_ms: 남길 트랙의 최소 키프레임 수·길이(ms).
    """

    start_score: float = Field(ge=0, le=1)
    min_keyframes: int = Field(ge=1)
    min_duration_ms: int = Field(ge=0)


class ScenePolicy(Contract):
    """장면 분류 (`scene` 절, ADR 0055, `dlp_prelabel.scene`). 도구 단계 버전에 분류 결과가
    들어간다.

    classifier: fixed(늘 default, CPU stub) | vlm(사내 VLM, `DLP_VLM_URL`). labels: 장면 이름 목록
    (unknown 포함). default: fixed 분류기·분류 실패 때의 장면. min_confidence: 이 미만이면 unknown.
    sample_frames: 보낼 프레임 수. max_side: 보낼 프레임 긴 변(px). blur_tolerance_ms /
    blur_cell_px: 보낼 프레임에 블러 트랙 박스를 모자이크할 때 시각 허용 오차와 칸 크기. vlm_model:
    `config/models.yaml external` 이름. timeout_s: 요청 시간 초과.
    """

    classifier: Literal["fixed", "vlm"]
    labels: tuple[str, ...] = Field(min_length=1)
    default: str
    min_confidence: float = Field(ge=0, le=1)
    sample_frames: int = Field(ge=1)
    max_side: int = Field(ge=64)
    blur_tolerance_ms: int = Field(ge=0)
    blur_cell_px: int = Field(ge=2)
    vlm_model: str
    timeout_s: float = Field(gt=0)

    @model_validator(mode="after")
    def _default_known(self) -> ScenePolicy:
        """default·unknown이 labels 안인지 검사한다."""
        for name in (self.default, "unknown"):
            if name not in self.labels:
                raise ValueError(f"scene.labels에 {name}이 없습니다")
        return self


class SmoothingPolicy(Contract):
    """손 2D One Euro 평활 (`hands.smoothing`, ADR 0054). 값 단위는 정규화 좌표(0~1)/초.

    min_cutoff_hz: 정지 때 차단 주파수 (낮을수록 강한 평활). beta: 속도에 따른 차단 주파수 증가.
    d_cutoff_hz: 속도 추정 저역 통과. reset_gap_ms: 이보다 긴 끊김 뒤에는 필터를 새로 시작.
    """

    enabled: bool
    min_cutoff_hz: float = Field(gt=0)
    beta: float = Field(ge=0)
    d_cutoff_hz: float = Field(gt=0)
    reset_gap_ms: int = Field(ge=0)


class GapFillPolicy(Contract):
    """손 짧은 끊김 채우기 (`hands.gap_fill`, ADR 0054).

    max_gap_ms: 이 이하로 끊긴 사이만 채운다 (0이면 채우지 않음). confidence: 채운 관절 신뢰도.
    max_jump_scale: 끊김 앞뒤 손목 거리가 손 크기(손목→중지 뿌리 px)의 이 배를 넘으면 채우지 않는다
        (빠르게 움직였거나 다른 손으로 이어진 끊김).
    """

    max_gap_ms: int = Field(ge=0)
    confidence: float = Field(ge=0, le=1)
    max_jump_scale: float = Field(gt=0)


class Constraints3DPolicy(Contract):
    """손 3D 물리 제약 (`depth.constraints`, ADR 0054, `dlp_prelabel.constraints3d`).

    reach_m: 카메라→손목 거리 타당 범위(m). max_speed_mps / max_accel_mps2: 손목 속도·가속도 상한.
    forearm_m / forearm_tolerance: 착용자 아래팔(손목→팔꿈치) 길이 범위(m)와 세션 중앙값 허용 비율
    (ADR 0058).
    max_dt_ms: 이보다 먼 이웃 표본으로는 속도·가속도를 세지 않는다. bone_tolerance: 손끝의 손목
    거리가
    세션 중앙값에서 이 비율 넘게 다르면 bone_length. min_bone_samples: 중앙값을 낼 최소 표본 수.
    actions: 표시 → downweight(신뢰도 x downweight) | drop(표본 빼기).
    """

    enabled: bool
    reach_m: tuple[float, float]
    max_speed_mps: float = Field(gt=0)
    max_accel_mps2: float = Field(gt=0)
    max_dt_ms: int = Field(gt=0)
    bone_tolerance: float = Field(gt=0)
    min_bone_samples: int = Field(ge=1)
    actions: dict[Constraint3DFlag, Literal["downweight", "drop"]]
    downweight: float = Field(gt=0, le=1)
    # 착용자 아래팔 (ADR 0058): 손목(손 3D)→팔꿈치 거리 타당 범위(m)와 세션 중앙값에서 벗어남
    # 허용 비율.
    # 어기면 팔꿈치 표본에 forearm_length (중앙값은 min_bone_samples개 이상일 때만)
    forearm_m: tuple[float, float]
    forearm_tolerance: float = Field(gt=0)

    @model_validator(mode="after")
    def _complete(self) -> Constraints3DPolicy:
        """모든 표시에 동작이 있고 reach_m·forearm_m이 0 < 하한 < 상한인지 검사한다."""
        missing = sorted(set(Constraint3DFlag) - set(self.actions))
        if missing:
            raise ValueError(f"depth.constraints.actions에 없는 표시: {missing}")
        for name in ("reach_m", "forearm_m"):
            low, high = getattr(self, name)
            if not 0 < low < high:
                raise ValueError(f"depth.constraints.{name}: 0 < 하한 < 상한이어야 합니다")
        return self


class ModelRoles(Contract):
    """역할 → config/models.yaml 이름.

    - hands: 손 랜드마크. body_detector: 사람 박스(YOLOX). body: 전신 포즈(RTMPose).
    - coco_objects: COCO 객체. open_vocab: OWLv2 ONNX. open_vocab_tokenizer: OWLv2 토크나이저.
    - depth: 메트릭 깊이.
    - mask_encoder·mask_decoder: SAM 2.1 그래프, mask_encoder_data·mask_decoder_data: 그 외부 가중치
      파일 (ADR 0059).

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
    mask_encoder: str
    mask_encoder_data: str
    mask_decoder: str
    mask_decoder_data: str


class HandsPolicy(Contract):
    """손 21관절(MediaPipe) 정책 (`hands` 절).

    min_score: 손 탐지 최소 신뢰도 (0~1, MediaPipe `min_hand_detection_confidence`).
    swap_handedness: MediaPipe 손 판정(거울상 셀카 기준)을 뒤집을지. 1인칭(바디캠·머리캠)은
        손등 쪽에서 보므로 판정이 그대로 착용자 기준이라 False다. 거울상이 아닌 카메라를 사람이
        마주 보는 영상만 True (F-PAR-61, `adapters.mediapipe_models.flip_hand`).
    """

    min_score: float = Field(ge=0, le=1)
    swap_handedness: bool
    # 한 프레임의 두 손 탐지가 관절 외접 박스 IoU 이 이상으로 겹치면 같은 손을 두 번 본 것으로 보고
    # 점수 높은 것만 남긴다 (`adapters.mediapipe_models.drop_duplicate_hands`, 2026-10-23 파일럿)
    duplicate_iou: float = Field(gt=0, le=1)
    smoothing: SmoothingPolicy  # 2D One Euro 평활 (ADR 0054)
    gap_fill: GapFillPolicy  # 짧은 끊김 채우기 (ADR 0054)


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
    track_filter: TrackFilterPolicy  # 트랙 거르기 (ADR 0054)


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
    track_filter: TrackFilterPolicy  # 트랙 거르기 (ADR 0054)
    # 장면 → 그 장면에서 쓸 질의 문장 (queries의 키). 없는 장면(unknown 등)은 전체 질의 (ADR 0055)
    scene_queries: dict[str, tuple[str, ...]] = Field(default_factory=dict[str, tuple[str, ...]])

    @model_validator(mode="after")
    def _scene_queries_known(self) -> OpenVocabObjectsPolicy:
        """scene_queries의 질의가 모두 queries에 있는지 검사한다."""
        for scene, names in self.scene_queries.items():
            bad = sorted(set(names) - set(self.queries))
            if bad or not names:
                raise ValueError(f"scene_queries.{scene}: queries에 없는 질의 {bad} (또는 빈 목록)")
        return self


# hand21 번호 → 온톨로지 hand_joints 이름 (3D 궤적 part)
# 3D로 올릴 수 있는 관절만 이름이 있다 (온톨로지 hand_joints와 같아야 한다). 0 손목, 4 엄지 끝,
# 8 검지 끝, 12 중지 끝, 16 약지 끝, 20 새끼 끝
HAND21_JOINTS = {
    0: "wrist", 4: "thumb_tip", 8: "index_tip", 12: "middle_tip", 16: "ring_tip", 20: "pinky_tip",
}  # fmt: skip


class HandFitPolicy(Contract):
    """손 중심 3D 점 맞춤 정책 (`depth.hand_fit` 절, ADR 0044, `dlp_prelabel.handfit`).

    min_points: 맞춤에 쓰는 보이는 관절 수 하한 (이보다 적으면 맞춤 없이 손바닥 관절만 깊이로).
    max_reprojection_ratio: 맞춘 손을 다시 투영한 2D 오차 RMS를 손 2D 크기(관절이 중심에서 떨어진
        거리의 RMS, px)로 나눈 상대값의 상한. 넘으면 맞춤을 버린다. 해상도와 무관하다 (F-PAR-43).
    rotation_prior: 회전 다듬기(손 중심 기준 작은 회전, F-PAR-42)의 0 쪽 사전값 가중 (상대 재투영
        잔차 단위, 회전 rad당). 클수록 손 모델 자세를 더 믿는다. 0이면 회전을 다듬지 않는다.
    max_rotation_deg: 회전 다듬기가 이보다 크게 돌아야 맞으면 다듬은 값을 버리고 이동량만 쓴다.
    anchor_points: 메트릭 깊이로 크기를 정할 때 깊이를 읽는 손바닥 관절 번호 (0~20, 얇은 손가락
        끝은 배경 깊이가 섞이기 쉬워 넣지 않는다).
    reference_palm_m: 손목(0)→중지 MCP(9) 기준 길이(m). None이면 손 모델 크기를 그대로 믿는다.
    depth_weight: 크기 융합에서 깊이 쪽 가중치 (0이면 손 크기만, 1이면 깊이만, 로그 평균).
    max_scale_ratio: 깊이 쪽과 손 크기 쪽 배율의 비가 이보다 크면 깊이를 믿지 않는다 (배경·가림).
    """

    min_points: int = Field(ge=3, le=21)
    max_reprojection_ratio: float = Field(gt=0)
    rotation_prior: float = Field(ge=0)
    max_rotation_deg: float = Field(ge=0, le=90)
    anchor_points: tuple[int, ...] = Field(min_length=1)
    reference_palm_m: float | None = Field(default=None, gt=0)
    depth_weight: float = Field(ge=0, le=1)
    max_scale_ratio: float = Field(ge=1)

    @field_validator("anchor_points")
    @classmethod
    def _joint_numbers(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        """anchor_points가 hand21 번호(0~20)인지 검사한다. 아니면 ValueError."""
        bad = sorted(i for i in v if not 0 <= i <= 20)
        if bad:
            raise ValueError(f"anchor_points {bad}: hand21 번호(0~20)가 아닙니다")
        return v


class SampleConfidencePolicy(Contract):
    """3D 표본 신뢰도 정책 (`depth.sample_confidence` 절, ADR 0045, `handfit.fit_confidence`).

    hand_fit: 손 맞춤 표본의 최고 신뢰도 (재투영 0, 21관절 보임, 깊이·사전값 일치, 캘리브레이션).
    point_depth: 관절별 깊이·박스 중심 표본의 신뢰도 (그 픽셀 깊이 하나로 역투영).
    reprojection_floor: 상대 재투영 오차가 `hand_fit.max_reprojection_ratio`일 때 곱하는 값 (0이면
        그 끝에서 0).
    points_floor: 보이는 관절 수가 `hand_fit.min_points`일 때 곱하는 값 (21관절이면 1).
    agreement_floor: 깊이·손 크기 배율 비가 `hand_fit.max_scale_ratio`일 때 곱하는 값 (비 1이면 1).
    no_depth: 깊이를 못 쓰거나 버리고 손 크기 사전값만으로 크기를 정한 맞춤에 곱하는 값.
    hfov_approx: 캘리브레이션 없이 기본 화각 근사 내부 파라미터를 쓴 표본에 곱하는 값.
    estimated_hand_size: 손 크기로 추정한 초점 거리(ADR 0056)를 쓴 표본에 곱하는 값.
    """

    hand_fit: float = Field(ge=0, le=1)
    point_depth: float = Field(ge=0, le=1)
    reprojection_floor: float = Field(ge=0, le=1)
    points_floor: float = Field(ge=0, le=1)
    agreement_floor: float = Field(ge=0, le=1)
    no_depth: float = Field(ge=0, le=1)
    hfov_approx: float = Field(ge=0, le=1)
    estimated_hand_size: float = Field(ge=0, le=1)


class FocalEstimatePolicy(Contract):
    """손 크기로 세션 초점 거리 추정 (`depth.focal_estimate`, ADR 0056, `dlp_prelabel.fov`).

    캘리브레이션(세션·기기 등록)이 없는 세션에서만 쓴다. frame_stride_ms: 추정용 깊이 추론 간격.
    min_samples: 쓸 맞춤 표본 수 하한. max_spread: 손바닥 길이 비의 (사분위 범위 / 중앙값) 상한.
    hfov_range_deg: 받아들일 가로 화각 범위(도). 밖이면 추정을 버리고 화각 근사를 쓴다.
    """

    enabled: bool
    frame_stride_ms: int = Field(ge=0)
    min_samples: int = Field(ge=3)
    max_spread: float = Field(gt=0)
    hfov_range_deg: tuple[float, float]


class DepthPolicy(Contract):
    """메트릭 깊이 3D 올리기 정책 (`depth` 절).

    frame_stride_ms: 깊이 추론 간격 (ms). default_hfov_deg: 캘리브레이션이 없을 때 가로 화각(도).
    patch_radius_px: 깊이 중앙값을 읽을 패치 반지름. min_depth_m / max_depth_m: 받아들일 깊이
    범위(m).
    hand_points: 올릴 hand21 관절 번호 (`HAND21_JOINTS`에 이름이 있는 번호만).
    fingertip_fallback: 맞춤이 없는 키프레임(world 없음·맞춤 실패)의 손바닥 밖 관절(손가락 끝)을
        어떻게 할지. "drop"(기본): 내지 않는다 (픽셀 깊이가 배경을 읽어 수백 mm 틀린다, F-PAR-40).
        "point_depth": 예전처럼 그 픽셀 깊이로 올린다. 손바닥 관절(`hand_fit.anchor_points`)은
        어느 쪽이든 깊이로 올린다.
    max_intrinsics_aspect_mismatch: 캘리브레이션 해상도와 디코드한 프레임의 화면비(가로/세로)가
        이 상대 차이보다 크면 그 캘리브레이션을 쓰지 않고 화각 근사로 떨어진다 (같은 기기의 다른
        촬영 모드는 센서를 잘라 비례 조정이 틀린다, F-PAR-44).
    hand_fit: 손 중심 3D 점이 있는 손 트랙의 맞춤 (`HandFitPolicy`, ADR 0044).
    sample_confidence: 표본 신뢰도 (`SampleConfidencePolicy`, ADR 0045). 궤적 라벨 신뢰도는 그
        궤적 표본 신뢰도의 평균이다.
    """

    frame_stride_ms: int = Field(ge=0)
    default_hfov_deg: float = Field(gt=0, lt=180)
    patch_radius_px: int = Field(ge=0)
    min_depth_m: float = Field(gt=0)
    max_depth_m: float = Field(gt=0)
    hand_points: tuple[int, ...] = Field(min_length=1)
    # 3D로 올릴 착용자 팔(arm3) 점 이름 (`ARM3_JOINTS`, 손목은 손 3D가 맡아 넣지 않는다, ADR 0058)
    arm_points: tuple[Literal["elbow", "shoulder"], ...] = ()
    fingertip_fallback: Literal["drop", "point_depth"]
    max_intrinsics_aspect_mismatch: float = Field(ge=0)
    hand_fit: HandFitPolicy
    sample_confidence: SampleConfidencePolicy
    constraints: Constraints3DPolicy  # 손 3D 물리 제약 (ADR 0054)
    focal_estimate: FocalEstimatePolicy  # 손 크기로 초점 거리 추정 (ADR 0056)

    @field_validator("hand_points")
    @classmethod
    def _named_points(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        """hand_points가 모두 `HAND21_JOINTS`에 있는지 검사한다. 없으면 ValueError."""
        # 3D 궤적의 part는 온톨로지 hand_joints 이름이어야 하므로 이름이 있는 번호만 받는다
        bad = sorted(set(v) - set(HAND21_JOINTS))
        if bad:
            raise ValueError(
                f"hand_points {bad}: 이름이 없는 hand21 번호 (허용 {sorted(HAND21_JOINTS)})"
            )
        return v


class PalmEstimatePolicy(Contract):
    """작업자 손 크기 추정 정책 (`palm_estimate` 절, ADR 0045, `dlp_prelabel.palm`).

    캘리브레이션이 있는 세션의 손 맞춤 중 품질이 좋은 시각마다 깊이 배율만으로 잰 손목 → 중지 MCP
    길이의 중앙값을 낸다. 3D 단계 버전(`depth` 절 해시)에는 들어가지 않는다 (추정 명령만 쓴다).

    max_reprojection_ratio: 쓸 맞춤의 상대 재투영 오차 상한 (`HandFit.rms_rel`, 맞춤 기본값보다
        엄격하게. 해상도와 무관, F-PAR-43).
    min_visible: 쓸 맞춤의 보이는 관절 수 하한.
    min_samples: 이보다 적으면 추정하지 않는다 (등록하지 않음).
    palm_range_m: 받아들일 손바닥 길이 범위 (m, [하한, 상한]). 밖의 값은 버린다 (깊이 오류).
    """

    max_reprojection_ratio: float = Field(gt=0)
    min_visible: int = Field(ge=3, le=21)
    min_samples: int = Field(ge=1)
    palm_range_m: tuple[float, float]


class PoseImuPolicy(Contract):
    """카메라 자세의 IMU 회전 정책 (`camera_pose.imu` 절, ADR 0046, `campose.imu_orientation`).

    init_window_ms: IMU가 덮는 첫 프레임부터 이 길이의 IMU로 중력 방향과 자이로 편향을 정한다.
    static_gyro_std: 초기 구간 자이로 크기의 표준편차가 이 값(rad/s) 이하이고, 자이로 평균 벡터의
        크기가 `static_gyro_mean_max`(rad/s) 이하이고, 가속도 크기 표준편차가 `static_acc_std`(m/s²)
        이하이고, 가속도 크기 평균이 `gravity`와 `accel_tolerance` 안이면 정지로 보고 자이로 편향을
        그 평균으로 뺀다. 아니면 편향 0 (신뢰도에 `confidence.no_static_init`를 곱한다).
    static_gyro_mean_max: 정지로 볼 자이로 평균 크기 상한(rad/s). 실제 영점 편향보다 조금 크게
        잡는다. 일정한 속도로 돌며 녹화를 시작하면 크기 표준편차는 0에 가깝지만 평균이 커서 정지가
        아니다 (F-PAR-46).
    max_gap_ms: IMU 표본 간격이 이보다 크면(기록 끊김) 그 앞에서 자세를 멈춘다. 빈틈을 건너
        적분하면 회전을 지어낸다 (F-PAR-48). 초기화 구간 안에 빈틈이 있으면 자세를 만들지 않는다.
    rotation_tolerance: 카메라-IMU 외부 파라미터(세션 값·`default_imu_to_camera`)의 회전 부분이
        직교(R Rᵀ = I)에서 벗어나도 되는 원소별 허용 오차. 넘거나 행렬식이 음수(반사)면 자세를
        만들지 않는다. 허용 오차 안이면 가장 가까운 회전으로 고쳐 쓴다 (F-PAR-49).
    accel_gain: 가속도 기울기 보정 이득 (1/s, 상보 필터). 0이면 자이로만 적분한다.
    accel_tolerance: 가속도 크기가 중력과 이만큼(m/s²) 안일 때만 기울기를 보정한다 (빠른 움직임은
        보정하지 않는다).
    gravity: 중력 가속도 크기 (m/s²).
    default_imu_to_camera: 세션 캘리브레이션에 카메라-IMU 외부 파라미터가 없을 때 쓸 IMU → 카메라
        회전 (3x3 행 우선). None이면 그런 세션은 자세를 만들지 않는다 (기종마다 축이 달라 기본값이
        틀릴 수 있다).
    """

    init_window_ms: int = Field(gt=0)
    static_gyro_std: float = Field(ge=0)
    static_gyro_mean_max: float = Field(ge=0)
    static_acc_std: float = Field(ge=0)
    accel_gain: float = Field(ge=0)
    accel_tolerance: float = Field(gt=0)
    gravity: float = Field(gt=0)
    max_gap_ms: float = Field(gt=0)
    rotation_tolerance: float = Field(gt=0)
    default_imu_to_camera: tuple[tuple[float, float, float], ...] | None = None

    @model_validator(mode="after")
    def _rotation(self) -> PoseImuPolicy:
        """default_imu_to_camera가 3x3 회전 행렬(직교, 행렬식 +1)인지 검사한다. 아니면 ValueError.

        세션 값과 같은 검사(`posemath.rotation_problem`, 허용 오차 `rotation_tolerance`)다.
        """
        if self.default_imu_to_camera is None:
            return self

        problem = rotation_problem(self.default_imu_to_camera, self.rotation_tolerance)
        if problem is not None:
            raise ValueError(f"default_imu_to_camera: {problem}")
        return self


class PoseFeaturePolicy(Contract):
    """특징점 추적 정책 (`camera_pose.features` 절, OpenCV goodFeaturesToTrack + 피라미드 LK).

    max_corners / quality_level / min_distance_px: 특징점 검출. lk_window_px / lk_levels: LK 창
    크기와 피라미드 단계. lk_max_iterations / lk_epsilon: LK 반복 종료 기준 (반복 수, 이동량 px.
    OpenCV 기본값 30, 0.01과 같다, F-PAR-55). fb_max_px: 앞뒤 추적 왕복 오차 상한 (넘으면 버린다).
    """

    max_corners: int = Field(ge=8)
    quality_level: float = Field(gt=0, lt=1)
    min_distance_px: float = Field(gt=0)
    lk_window_px: int = Field(ge=5)
    lk_levels: int = Field(ge=0)
    lk_max_iterations: int = Field(ge=1)
    lk_epsilon: float = Field(gt=0)
    fb_max_px: float = Field(gt=0)


class PoseTranslationPolicy(Contract):
    """회전을 아는 이동량 풀이 정책 (`camera_pose.translation` 절, `campose.solve_translation`).

    ransac_iterations: 두 점 표본 RANSAC 반복 수. inlier_px: 재투영 오차 인라이어 문턱(px).
    min_inliers: 이보다 적으면 그 프레임은 자세를 내지 않는다. seed: RANSAC 난수 시드 (결정적).
    depth_patch_px / min_depth_m / max_depth_m: 특징점 깊이를 읽는 패치 반지름과 받는 깊이 범위.
    """

    ransac_iterations: int = Field(ge=1)
    inlier_px: float = Field(gt=0)
    min_inliers: int = Field(ge=3)
    seed: int
    depth_patch_px: int = Field(ge=0)
    min_depth_m: float = Field(gt=0)
    max_depth_m: float = Field(gt=0)


class PoseConfidencePolicy(Contract):
    """자세 표본 신뢰도 (`camera_pose.confidence` 절, `campose.pose_confidence`).

    base: 최고값. good_inliers: 인라이어가 이 수 이상이면 인라이어 항 1 (적으면 비례해 낮춤).
    reprojection_floor: 인라이어 재투영 RMS가 `translation.inlier_px`일 때 곱하는 값.
    no_static_init: 초기 정지 구간이 없어 자이로 편향을 빼지 못했을 때 곱하는 값.
    hfov_approx: 캘리브레이션 없이 화각 근사 내부 파라미터를 썼을 때 곱하는 값.
    """

    base: float = Field(ge=0, le=1)
    good_inliers: int = Field(ge=1)
    reprojection_floor: float = Field(ge=0, le=1)
    no_static_init: float = Field(ge=0, le=1)
    hfov_approx: float = Field(ge=0, le=1)


class CameraPosePolicy(Contract):
    """바디캠 6자유도 자세 정책 (`camera_pose` 절, ADR 0046, `dlp_prelabel.campose`).

    enabled: 거짓이면 자세·월드 좌표 단계를 돌리지 않는다.
    frame_stride_ms: 자세를 내는 프레임 간격 (깊이 추론 1회/프레임).
    max_lost_ms: 마지막으로 추적한 프레임에서 이만큼 지나도록 추적을 못 하면 그 뒤는 자세를 내지
        않는다 (같은 월드 좌표로 이을 수 없다).
    """

    enabled: bool
    frame_stride_ms: int = Field(ge=0)
    max_lost_ms: int = Field(ge=0)
    imu: PoseImuPolicy
    features: PoseFeaturePolicy
    translation: PoseTranslationPolicy
    confidence: PoseConfidencePolicy


class WorldFramePolicy(Contract):
    """카메라 좌표 3D 궤적 → 세션 월드 좌표 (`world_frame` 절, ADR 0046, `dlp_prelabel.worldframe`).

    max_pose_gap_ms: 표본 시각 양쪽 자세 표본 간격이 이보다 크면 그 표본은 월드 좌표로 바꾸지
    않는다.
    """

    max_pose_gap_ms: int = Field(ge=0)


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


class WearerSuppressionPolicy(Contract):
    """착용자 자신의 몸 억제 (`wearer_suppression` 절, ADR 0058, `dlp_prelabel.wearer_suppress`).

    바디캠 스트림에서 착용자 자신의 팔·몸통을 잡은 박스·자세를 라벨로 만들지 않는다. 판정 규칙은
    `dlp_schema.wearer_body`. 3인칭 스트림·손 증거가 없는 스트림에는 적용하지 않는다. 프라이버시
    탐지(블러)는 이 절과 무관하다 (블러를 약하게 하지 않는다).

    enabled: 끄면 억제하지 않는다 (버전 해시는 그대로 들어간다).
    stages: 억제를 적용할 어댑터 이름 (objects·tools·body). 이 어댑터 버전에 이 절의 해시가
        들어간다.
    time_tolerance_ms: 손 키프레임을 같은 시각으로 볼 허용 오차. margin: 박스를 넓히는 비율.
    bottom_band: 박스·자세 아래 변이 이 화면 높이 비율의 아래 띠 안이어야 착용자로 본다.
    track_share: 박스 트랙 키프레임 중 착용자 프레임 비율이 이 이상이면 트랙째 라벨로 만들지 않는다.
    person_classes: 사람 클래스 (온톨로지 객체 ID). 이 클래스 박스는 착용자 프레임 규칙만 본다.
    other_min_area: 사람 아닌 클래스(도구 등) 박스는 착용자 프레임 규칙에 더해 박스 면적이 화면의 이
        비율 이상이어야 착용자 프레임이다 (손에 든 도구를 지우지 않게). None이면 사람 클래스만 본다.
    wrist_match_palms: 전신 자세 손목이 착용자 손목에서 손 크기(손목→중지 뿌리 px) x 이 값 안이면
        착용자 자세다.
    """

    enabled: bool
    stages: tuple[Literal["objects", "tools", "body"], ...]
    time_tolerance_ms: int = Field(ge=0)
    margin: float = Field(ge=0)
    bottom_band: float = Field(gt=0, le=1)
    track_share: float = Field(gt=0, le=1)
    person_classes: tuple[OntologyId, ...]
    other_min_area: float | None = Field(default=None, gt=0, le=1)
    wrist_match_palms: float = Field(gt=0)


class ArmBoxPolicy(Contract):
    """팔 단계의 합성 착용자 박스 (`arm.box`). 보이는 손 박스를 옆으로 side_palms, 위로 up_palms 손
    크기만큼 넓히고 아래는 화면 끝까지 늘린다 (아래팔은 화면 아래에서 들어온다)."""

    side_palms: float = Field(ge=0)
    up_palms: float = Field(ge=0)


class ArmPolicy(Contract):
    """착용자 팔 골격 arm3 (`arm` 절, ADR 0058, `dlp_prelabel.arm`). 전신 어댑터(`RtmPose`)가 돈다.

    enabled: 끄면 팔 트랙을 내지 않는다. frame_stride_ms: 팔 키프레임 간격 (손 키프레임 중 고른다).
    wrist_match_palms: 자세의 팔 사슬(손목·팔꿈치·어깨) 손목이 hand21 손목에서 손 크기 x 이 값
        안이어야
        그 손의 팔로 쓴다. 손목은 언제나 hand21 손목이고, 일치 정도를 손목 점 신뢰도에 싣는다.
    elbow_min_score / shoulder_min_score: 그 점 모델 점수 하한. edge_margin_px: 화면 가장자리에서 이
        안쪽이어야 보임 (밖이거나 가장자리면 화면 밖 = 보이지 않음 0, 지어내지 않는다).
    forearm_palms: 화면의 손목→팔꿈치 거리 / 손 크기 타당 범위. max_hand_cos: 손목→팔꿈치와
        손목→중지
        뿌리 방향의 코사인 상한 (팔꿈치가 손가락 쪽에 있으면 틀린 것).
    cross_margin_palms: 두 팔꿈치가 모두 보일 때 왼팔 팔꿈치가 오른팔 팔꿈치보다 손 크기 x 이 값
        넘게
        오른쪽이면 좌우 충돌로 보고 둘 다 보이지 않음으로 둔다.
    emit_shoulder: 어깨를 낼지 (가슴 장착은 어깨가 거의 화면 밖이라 기본 끔: 늘 0).
    box: 합성 착용자 박스 (`ArmBoxPolicy`).
    """

    enabled: bool
    frame_stride_ms: int = Field(ge=0)
    wrist_match_palms: float = Field(gt=0)
    elbow_min_score: float = Field(ge=0, le=1)
    shoulder_min_score: float = Field(ge=0, le=1)
    edge_margin_px: float = Field(ge=0)
    forearm_palms: tuple[float, float]
    max_hand_cos: float = Field(ge=-1, le=1)
    cross_margin_palms: float = Field(ge=0)
    emit_shoulder: bool
    box: ArmBoxPolicy

    @model_validator(mode="after")
    def _range(self) -> ArmPolicy:
        """forearm_palms가 0 < 하한 < 상한인지 검사한다."""
        low, high = self.forearm_palms
        if not 0 < low < high:
            raise ValueError("arm.forearm_palms: 0 < 하한 < 상한이어야 합니다")
        return self


class MasksPolicy(Contract):
    """객체·손 모양 마스크 (`masks`, `dlp_prelabel.masks.MaskMaker`, ADR 0059). 뜻은 YAML 주석."""

    enabled: bool
    frame_stride_ms: int = Field(ge=1)
    box_max_gap_ms: int = Field(ge=0)
    min_track_confidence: float = Field(ge=0, le=1)
    exclude_classes: tuple[OntologyId, ...] = ()
    min_box_px: float = Field(ge=0)
    max_prompts_per_frame: int = Field(ge=1)
    hands: bool
    hand_class: OntologyId
    hand_min_points: int = Field(ge=1, le=21)
    hand_box_margin: float = Field(ge=0)
    hand_time_tolerance_ms: int = Field(ge=0)
    hand_point_joints: tuple[int, ...] = ()
    min_score: float = Field(ge=0, le=1)
    min_area_ratio: float = Field(ge=0, le=1)
    outside_margin: float = Field(ge=0)
    max_outside_ratio: float = Field(ge=0, le=1)
    largest_component: bool
    subtract_hands: bool
    person_classes: tuple[OntologyId, ...] = ()
    person_bottom_band: float | None = Field(default=None, ge=0, le=1)

    @field_validator("hand_point_joints")
    @classmethod
    def _joints(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        """hand21 관절 번호(0~20)만. Raises: ValueError."""
        if any(not 0 <= j <= 20 for j in v):
            raise ValueError("masks.hand_point_joints: hand21 관절 번호(0~20)여야 합니다")
        return v


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
    palm_estimate: PalmEstimatePolicy
    camera_pose: CameraPosePolicy
    world_frame: WorldFramePolicy
    scene: ScenePolicy  # 장면 분류 → 도구 질의 목록 (ADR 0055)
    wearer_suppression: WearerSuppressionPolicy  # 착용자 자신의 몸 억제 (ADR 0058)
    arm: ArmPolicy  # 착용자 팔 골격 arm3 (ADR 0058)
    masks: MasksPolicy  # 객체·손 모양 마스크 (ADR 0059)
    # 마스터 타임라인 구간(stream_id 없음)을 내는 예측기 이름. 기준 스트림에서 세션당 한 번만 돌린다
    timeline_predictors: tuple[str, ...] = ()

    def digest(self, *sections: str) -> str:
        """정책 절들의 짧은 해시. 모델 버전에 붙여 정책 값이 바뀌면 다시 돌게 한다.

        Args:
            *sections: 절 이름 (예: "hands", "contact"). 속성 이름과 같아야 한다 (없으면
                AttributeError).

        Returns:
            {절: model_dump(json)}를 키 정렬 JSON으로 만든 sha256의 앞 8자. 파싱된 값만 보므로
            YAML 주석·키 순서·따옴표 차이는 해시를 바꾸지 않는다.
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
