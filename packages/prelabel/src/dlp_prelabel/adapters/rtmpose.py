"""RTMPose 전신 포즈 (COCO 17점, CPU): YOLOX-m 사람 탐지(COCO) → RTMPose-m (Body7).

rtmlib로 ONNX Runtime에서 돌린다. 가중치는 `make models`가 받는다 (config/models.yaml
yolox_m_coco: Megvii 공식, rtmpose_m_body7: OpenMMLab). 사람 탐지기는 rtmlib 기본값인 Human-Art판
대신 COCO판을 쓴다. Human-Art 데이터가 비상업 라이선스이고, 실사 영상에서 성능 차이가 거의 없다 (ADR
0010). 3인칭 영상의 착용자 매칭과 바디캠에 보이는 다른 사람(환자 등) 자세에 쓴다. 바디캠에서는
착용자 자신의 자세를 coco17 트랙에서 빼고(ADR 0058, `dlp_prelabel.wearer_suppress`), 착용자 팔 골격
arm3(손목·팔꿈치·어깨, `dlp_prelabel.arm`)을 낸다. 새 가중치는 없다.

모델 입출력 (rtmlib가 전처리를 맡는다):
- YOLOX: BGR 이미지 → 640x640 letterbox 입력 → 사람 박스(x1, y1, x2, y2 픽셀)와 클래스 (0 = 사람).
- RTMPose: BGR 이미지 + 사람 박스 → 박스를 192x256(너비x높이)로 잘라 SimCC 디코드 → 관절 17개 (x,
  y 원래 이미지 픽셀)와 관절별 점수. 출력 좌표계: 스트림 픽셀, 왼쪽 위 원점. skeleton="coco17".
  키프레임 시각은 스트림 PTS ms. 라이선스: `config/models.yaml yolox_m_coco`, `rtmpose_m_body7`
  모두 review (COCO 등 학습 데이터).
"""

# rtmlib에는 타입 정보가 없어 이 모듈에서만 알 수 없는 타입 경고를 끈다.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from dlp_models.registry import resolve
from dlp_prelabel.arm import (
    ArmFrame,
    ArmReport,
    PoseCandidate,
    arm_tracks,
    estimate_arms,
    synthetic_box,
)
from dlp_prelabel.common import iter_frames, model_label, strictly_increasing, track_boxes
from dlp_prelabel.policy import PrelabelPolicy
from dlp_prelabel.wearer_suppress import WearerAware, WearerReport, wearer_tag
from dlp_schema.episode import version_tag
from dlp_schema.labels import Hand, Keypoint, KeypointFrame, KeypointTrackPayload, LabelRecord
from dlp_schema.predictor import Clip
from dlp_schema.wearer_body import HandAt, WearerEvidence, wearer_pose_frame

# (트랙 키, (x, y, w, h) 픽셀, 점수). 사람은 키가 항상 "person"
Det = tuple[str, tuple[float, float, float, float], float]
# BGR 프레임 → (N, 4) 사람 박스 x1, y1, x2, y2 (점수 순)
DetectFn = Callable[[NDArray[np.uint8]], NDArray[np.float32]]
# (BGR 프레임, 박스 목록) → ((N, 17, 2) 관절 픽셀, (N, 17) 관절 점수)
PoseFn = Callable[[NDArray[np.uint8], list[list[float]]], tuple[NDArray[Any], NDArray[Any]]]


@dataclass
class BodyPass:
    """영상 한 번 훑기의 결과 (`body_pass`).

    detections / frames: 다른 사람 자세 (추적 입력, 시각·관절 박스 → 키프레임).
    arms: 손 → 팔 추정 (팔 단계가 돌았을 때). wearer: 억제 요약 (억제하지 않았으면 None).
    arm_report: 팔 요약 (팔 단계가 돌지 않았으면 None).
    """

    detections: list[tuple[int, list[Det]]] = field(default_factory=list[tuple[int, list[Det]]])
    frames: dict[tuple[int, tuple[float, float, float, float]], KeypointFrame] = field(
        default_factory=dict[tuple[int, tuple[float, float, float, float]], KeypointFrame]
    )
    arms: dict[Hand, list[ArmFrame]] = field(default_factory=dict[Hand, list[ArmFrame]])
    wearer: WearerReport | None = None
    arm_report: ArmReport | None = None


class _Pass:
    """`body_pass`의 프레임 처리 (자세 → 착용자 자세 분리 → 다른 사람 탐지, 팔 시각이면 팔)."""

    def __init__(
        self,
        detect: DetectFn,
        pose: PoseFn,
        policy: PrelabelPolicy,
        wearer: WearerEvidence | None,
    ) -> None:
        self.detect, self.pose, self.policy, self.wearer = detect, pose, policy, wearer
        ws = policy.wearer_suppression
        self.suppress = wearer is not None and ws.enabled and "body" in ws.stages
        self.arm_on = wearer is not None and policy.arm.enabled
        self.out = BodyPass(
            wearer=WearerReport() if self.suppress else None,
            arm_report=ArmReport() if self.arm_on else None,
        )
        self.last_arm: int | None = None

    def _points(self, kp: NDArray[Any], sc: NDArray[Any]) -> tuple[Keypoint, ...]:
        """관절 → Keypoint (점수가 keypoint_score 미만이면 visibility 1)."""
        cut = self.policy.body.keypoint_score
        return tuple(
            Keypoint(x=float(x), y=float(y), visibility=2 if s >= cut else 1)
            for (x, y), s in zip(kp, sc, strict=True)
        )

    def frame(self, t: int, rgb: NDArray[np.uint8]) -> None:
        """프레임 하나를 처리한다."""
        bp, ws = self.policy.body, self.policy.wearer_suppression
        # rtmlib은 OpenCV(BGR) 입력을 가정한다
        bgr = np.asarray(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), dtype=np.uint8)
        size = (int(rgb.shape[1]), int(rgb.shape[0]))
        boxes = np.asarray(self.detect(bgr), dtype=np.float32).reshape(-1, 4)[: bp.max_people]
        hands = self.wearer.hands_near(t, 0) if self.wearer is not None else []
        dets: list[Det] = []
        own: list[PoseCandidate] = []
        if len(boxes):
            keypoints, scores = self.pose(bgr, boxes.tolist())
            for kp, sc in zip(np.asarray(keypoints), np.asarray(scores), strict=True):
                points = self._points(kp, sc)
                # 착용자 자세: 손목이 착용자 손목과 맞고 아래 띠에 닿는다 (팔 후보로는 쓴다)
                if hands and wearer_pose_frame(
                    points, hands, frame_h=size[1], bottom_band=ws.bottom_band,
                    wrist_match_palms=ws.wrist_match_palms,
                ):  # fmt: skip
                    own.append(PoseCandidate(np.asarray(kp, float), np.asarray(sc, float)))
                    if self.out.wearer is not None:
                        self.out.wearer.poses += 1
                        continue
                # 사람 점수 = 관절 점수 평균 (rtmlib SimCC 점수, 0~1 근처)
                score = float(np.mean(sc))
                if score < bp.min_score:
                    continue
                # 추적용 박스: 탐지 박스 대신 관절을 감싸는 박스 (x, y, w, h)
                xs, ys = kp[:, 0], kp[:, 1]
                box = (
                    float(xs.min()),
                    float(ys.min()),
                    float(xs.max() - xs.min()),
                    float(ys.max() - ys.min()),
                )
                dets.append(("person", box, score))
                self.out.frames[(t, box)] = KeypointFrame(t_ms=t, points=points)
        self.out.detections.append((t, dets))
        if self.arm_on and hands and self._arm_due(t):
            self._arms(t, bgr, hands, own, size)

    def _arm_due(self, t: int) -> bool:
        """팔 키프레임 시각인가 (마지막 팔 시각에서 frame_stride_ms가 지났다)."""
        if self.last_arm is not None and t - self.last_arm < self.policy.arm.frame_stride_ms:
            return False
        self.last_arm = t
        return True

    def _arms(
        self,
        t: int,
        bgr: NDArray[np.uint8],
        hands: list[HandAt],
        own: list[PoseCandidate],
        size: tuple[int, int],
    ) -> None:
        """합성 착용자 박스 자세를 더해 손마다 팔을 고른다."""
        by_hand: dict[Hand, HandAt] = {}
        for h in hands:
            by_hand.setdefault(h.hand, h)
        x0, y0, x1, y1 = synthetic_box(list(by_hand.values()), size, self.policy.arm.box)
        kp, sc = self.pose(bgr, [[x0, y0, x1, y1]])
        cands = [*own, PoseCandidate(np.asarray(kp[0], float), np.asarray(sc[0], float))]
        assert self.out.arm_report is not None
        for f in estimate_arms(t, by_hand, cands, size, self.policy.arm).values():
            self.out.arms.setdefault(f.hand, []).append(f)
            self.out.arm_report.add(f)


def body_pass(
    frames: Iterable[tuple[int, NDArray[np.uint8]]],
    detect: DetectFn,
    pose: PoseFn,
    policy: PrelabelPolicy,
    wearer: WearerEvidence | None,
) -> BodyPass:
    """영상을 한 번 훑어 다른 사람 자세와 착용자 팔을 모은다 (모델 없이 시험할 수 있게 분리).

    프레임마다: 사람 박스(최대 max_people) → 자세. 착용자 자세(`wearer_pose_frame`, 착용자 증거가
    있을 때)는 억제가 켜져 있으면 다른 사람 트랙에 넣지 않고 팔 후보로만 쓴다. 관절 점수 평균이
    min_score 미만인 다른 사람은 버린다. 팔 시각(arm.frame_stride_ms 간격, 그 시각 손이 보일 때)에는
    합성 착용자 박스 자세를 더해 손마다 팔을 고른다 (`dlp_prelabel.arm`).

    Args:
        frames: (스트림 PTS ms, RGB) 순회 (시각이 엄격히 증가해야 한다).
        detect / pose: 사람 탐지·자세 함수 (rtmlib YOLOX·RTMPose, 시험은 가짜).
        policy: 프리라벨 정책 (body·wearer_suppression·arm 절).
        wearer: 착용자 증거 (러너가 바디캠에만 넘긴다). None이면 억제·팔 단계를 하지 않는다.
    """
    p = _Pass(detect, pose, policy, wearer)
    for t, rgb in frames:
        p.frame(t, rgb)
    return p.out


class RtmPose(WearerAware):
    """전신 포즈 어댑터 (name="body"). 바디캠에서는 착용자 팔 골격 arm3도 낸다 (ADR 0058).

    version: `rtmpose-<포즈 가중치 버전>+<탐지 가중치 버전>+p<body 절 해시>+w<wearer_suppression
    해시>+a<arm 절 해시>`. 다른 사람 라벨 ID는 `<세션>-<스트림>-body-<version_tag>-<트랙 순번>`,
    entity_id는 `person_<순번>`. 팔 트랙 ID는 `<세션>-<스트림>-body-<version_tag>-arm-<손>`,
    entity_id는 `<손>_arm` (`dlp_prelabel.arm.arm_tracks`).
    """

    name = "body"

    def __init__(
        self, root: Path, policy: PrelabelPolicy, *, ontology_version: str, now: datetime
    ) -> None:
        """두 가중치를 확인하고 버전을 정한다 (모델은 `run`에서 연다).

        Raises:
            ModelUnavailableError: 가중치가 없을 때 (`make models`).
        """
        self.detector_path, det_version = resolve(root, policy.models.body_detector)
        self.pose_path, pose_version = resolve(root, policy.models.body)
        self.version = (
            f"rtmpose-{pose_version}+{det_version}+p{policy.digest('body')}"
            f"{wearer_tag(policy, self.name)}+a{policy.digest('arm')}"
        )
        self.policy, self.ontology_version, self.now = policy, ontology_version, now
        self.last_arm: ArmReport | None = None

    def run(self, clip: Clip) -> list[LabelRecord]:
        """영상 전체 프레임에서 사람 자세를 찾아 사람마다 coco17 키포인트 트랙을 돌려준다.

        착용자 증거(`set_wearer`, 러너가 바디캠에만 넘긴다)가 있으면 착용자 자세는 coco17 트랙에서
        빼고, 손마다 arm3 팔 트랙을 더한다 (`body_pass`). 추적에는 탐지 박스가 아니라 관절 좌표를
        감싸는 박스를 쓴다. 트랙 신뢰도는 프레임 점수 평균.
        """
        # rtmlib은 무겁고 onnxruntime 세션을 바로 만들므로 실제로 돌릴 때만 불러온다
        from rtmlib import YOLOX, RTMPose

        bp = self.policy.body
        detector = YOLOX(
            str(self.detector_path),
            model_input_size=(640, 640),
            det_mode="multiclass",
            score_thr=bp.detector_score,
            device="cpu",
        )
        model = RTMPose(str(self.pose_path), model_input_size=(192, 256), device="cpu")

        def detect(bgr: NDArray[np.uint8]) -> NDArray[np.float32]:
            # found: (N, 4) x1, y1, x2, y2 픽셀, classes: (N,) COCO 클래스 번호 (0 = 사람, 점수 순)
            found, classes = detector(bgr)
            return np.asarray(found, dtype=np.float32).reshape(-1, 4)[np.asarray(classes) == 0]

        def pose(bgr: NDArray[np.uint8], boxes: list[list[float]]) -> tuple[Any, Any]:
            return model(bgr, bboxes=boxes)

        # 반올림한 PTS ms가 앞 프레임과 같은 프레임은 건너뛴다: 한 사람 트랙에 같은 시각 키프레임이
        # 두 번 들어가면 계약 검사(키프레임 시각 유일)에 걸린다 (`common.strictly_increasing`)
        result = body_pass(
            strictly_increasing(iter_frames(clip.video)), detect, pose, self.policy, self.wearer
        )
        self.last_wearer, self.last_arm = result.wearer, result.arm_report
        return [*self._people(result, clip), *self._arm_labels(result, clip)]

    def _people(self, result: BodyPass, clip: Clip) -> list[LabelRecord]:
        """다른 사람 자세 → coco17 트랙 라벨."""
        bp = self.policy.body
        tracks = track_boxes(result.detections, iou_match=bp.track_iou, max_gap_ms=bp.max_gap_ms)
        out: list[LabelRecord] = []
        for i, tr in enumerate(tracks):
            times = sorted(tr.frames)
            payload = KeypointTrackPayload(
                entity_id=f"person_{i:02d}",
                skeleton="coco17",
                keyframes=tuple(result.frames[(t, tr.frames[t][0])] for t in times),
            )
            out.append(
                model_label(
                    label_id=f"{clip.session_id}-{clip.stream_id}-body-{version_tag(self.version)}-{i:03d}",
                    session_id=clip.session_id,
                    stream_id=clip.stream_id,
                    t_start_ms=times[0],
                    t_end_ms=times[-1],
                    ontology_version=self.ontology_version,
                    model_version=self.version,
                    confidence=sum(s for _, s in tr.frames.values()) / len(tr.frames),
                    payload=payload,
                    now=self.now,
                )
            )
        return out

    def _arm_labels(self, result: BodyPass, clip: Clip) -> list[LabelRecord]:
        """착용자 팔 트랙 라벨 (손마다 하나, 보이는 팔꿈치가 없으면 없음)."""
        labels = arm_tracks(
            result.arms, clip, prefix=self.name, version=self.version,
            ontology_version=self.ontology_version, now=self.now,
        )  # fmt: skip
        if result.arm_report is not None:
            result.arm_report.tracks = len(labels)
        return labels
