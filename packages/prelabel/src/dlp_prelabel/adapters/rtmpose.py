"""RTMPose 전신 포즈 (COCO 17점, CPU): YOLOX-m 사람 탐지(COCO) → RTMPose-m (Body7).

rtmlib로 ONNX Runtime에서 돌린다. 가중치는 `make models`가 받는다 (config/models.yaml
yolox_m_coco: Megvii 공식, rtmpose_m_body7: OpenMMLab). 사람 탐지기는 rtmlib 기본값인 Human-Art판
대신 COCO판을 쓴다. Human-Art 데이터가 비상업 라이선스이고, 실사 영상에서 성능 차이가 거의 없다 (ADR
0010). 3인칭 영상의 착용자 매칭과 바디캠에 보이는 다른 사람(환자 등) 자세에 쓴다.

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

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from dlp_models.registry import resolve
from dlp_prelabel.common import iter_frames, model_label, strictly_increasing, track_boxes
from dlp_prelabel.policy import PrelabelPolicy
from dlp_schema.episode import version_tag
from dlp_schema.labels import Keypoint, KeypointFrame, KeypointTrackPayload, LabelRecord
from dlp_schema.predictor import Clip

# (트랙 키, (x, y, w, h) 픽셀, 점수). 사람은 키가 항상 "person"
Det = tuple[str, tuple[float, float, float, float], float]


class RtmPose:
    """전신 포즈 어댑터 (name="body").

    version: `rtmpose-<포즈 가중치 버전>+<탐지 가중치 버전>+p<body 절 해시>`. 라벨 ID는
    `<세션>-<스트림>-body-<version_tag>-<트랙 순번>`, entity_id는 `person_<순번>`.
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
        self.version = f"rtmpose-{pose_version}+{det_version}+p{policy.digest('body')}"
        self.policy, self.ontology_version, self.now = policy, ontology_version, now

    def run(self, clip: Clip) -> list[LabelRecord]:
        """영상 전체 프레임에서 사람 자세를 찾아 사람마다 coco17 키포인트 트랙을 돌려준다.

        프레임마다: 사람 박스(최대 max_people) → 자세. 관절 점수 평균이 min_score 미만인 사람은
        버린다. 추적에는 탐지 박스가 아니라 관절 좌표를 감싸는 박스를 쓴다. 트랙 신뢰도는 프레임
        점수 평균.
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
        pose = RTMPose(str(self.pose_path), model_input_size=(192, 256), device="cpu")
        detections: list[tuple[int, list[Det]]] = []
        # (시각, 관절 감싸는 박스) → 키프레임. 추적 결과(시각, 박스)로 키프레임을 되찾는다
        frames: dict[tuple[int, tuple[float, float, float, float]], KeypointFrame] = {}
        # 반올림한 PTS ms가 앞 프레임과 같은 프레임은 건너뛴다: 한 사람 트랙에 같은 시각 키프레임이
        # 두 번 들어가면 계약 검사(키프레임 시각 유일)에 걸린다 (`common.strictly_increasing`)
        for t, rgb in strictly_increasing(iter_frames(clip.video)):
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)  # rtmlib은 OpenCV(BGR) 입력을 가정한다
            # found: (N, 4) x1, y1, x2, y2 픽셀, classes: (N,) COCO 클래스 번호
            found, classes = detector(bgr)
            people = np.asarray(found, dtype=np.float32).reshape(-1, 4)[np.asarray(classes) == 0]
            boxes = people[: bp.max_people]  # COCO 0번 = 사람, 점수 순
            dets: list[Det] = []
            if len(boxes):
                keypoints, scores = pose(bgr, bboxes=boxes.tolist())
                for kp, sc in zip(np.asarray(keypoints), np.asarray(scores), strict=True):
                    # 사람 점수 = 관절 점수 평균 (rtmlib SimCC 점수, 0~1 근처)
                    score = float(np.mean(sc))
                    if score < bp.min_score:
                        continue
                    points = tuple(
                        Keypoint(
                            x=float(x), y=float(y), visibility=2 if s >= bp.keypoint_score else 1
                        )
                        for (x, y), s in zip(kp, sc, strict=True)
                    )
                    # 추적용 박스: 탐지 박스 대신 관절을 감싸는 박스 (x, y, w, h)
                    xs, ys = kp[:, 0], kp[:, 1]
                    box = (
                        float(xs.min()),
                        float(ys.min()),
                        float(xs.max() - xs.min()),
                        float(ys.max() - ys.min()),
                    )
                    dets.append(("person", box, score))
                    frames[(t, box)] = KeypointFrame(t_ms=t, points=points)
            detections.append((t, dets))
        tracks = track_boxes(detections, iou_match=bp.track_iou, max_gap_ms=bp.max_gap_ms)
        out: list[LabelRecord] = []
        for i, tr in enumerate(tracks):
            times = sorted(tr.frames)
            payload = KeypointTrackPayload(
                entity_id=f"person_{i:02d}",
                skeleton="coco17",
                keyframes=tuple(frames[(t, tr.frames[t][0])] for t in times),
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
