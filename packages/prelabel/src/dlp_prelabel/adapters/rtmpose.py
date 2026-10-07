"""RTMPose 전신 포즈 (COCO 17점, CPU): YOLOX-m 사람 탐지(COCO) → RTMPose-m (Body7).

rtmlib로 ONNX Runtime에서 돌린다. 가중치는 `make models`가 받는다 (config/models.yaml
yolox_m_coco: Megvii 공식, rtmpose_m_body7: OpenMMLab). 사람 탐지기는 rtmlib 기본값인 Human-Art판
대신 COCO판을 쓴다. Human-Art 데이터가 비상업 라이선스이고, 실사 영상에서 성능 차이가 거의 없다
(ADR 0010). 3인칭 영상의 착용자 매칭과 바디캠에 보이는 다른 사람(환자 등) 자세에 쓴다.
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
from dlp_prelabel.common import iter_frames, model_label, track_boxes
from dlp_prelabel.policy import PrelabelPolicy
from dlp_schema.episode import version_tag
from dlp_schema.labels import Keypoint, KeypointFrame, KeypointTrackPayload, LabelRecord
from dlp_schema.predictor import Clip

Det = tuple[str, tuple[float, float, float, float], float]


class RtmPose:
    name = "body"

    def __init__(
        self, root: Path, policy: PrelabelPolicy, *, ontology_version: str, now: datetime
    ) -> None:
        self.detector_path, det_version = resolve(root, policy.models.body_detector)
        self.pose_path, pose_version = resolve(root, policy.models.body)
        self.version = f"rtmpose-{pose_version}+{det_version}+p{policy.digest('body')}"
        self.policy, self.ontology_version, self.now = policy, ontology_version, now

    def run(self, clip: Clip) -> list[LabelRecord]:
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
        frames: dict[tuple[int, tuple[float, float, float, float]], KeypointFrame] = {}
        for t, rgb in iter_frames(clip.video):
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)  # rtmlib은 OpenCV(BGR) 입력을 가정한다
            found, classes = detector(bgr)
            people = np.asarray(found, dtype=np.float32).reshape(-1, 4)[np.asarray(classes) == 0]
            boxes = people[: bp.max_people]  # COCO 0번 = 사람, 점수 순
            dets: list[Det] = []
            if len(boxes):
                keypoints, scores = pose(bgr, bboxes=boxes.tolist())
                for kp, sc in zip(np.asarray(keypoints), np.asarray(scores), strict=True):
                    score = float(np.mean(sc))
                    if score < bp.min_score:
                        continue
                    points = tuple(
                        Keypoint(
                            x=float(x), y=float(y), visibility=2 if s >= bp.keypoint_score else 1
                        )
                        for (x, y), s in zip(kp, sc, strict=True)
                    )
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
