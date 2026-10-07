"""MediaPipe 실제 모델 어댑터 (CPU): 손 21관절, COCO 객체 탐지.

모델 파일은 `make models`로 받는다 (config/models.yaml). MediaPipe는 EGL/GLES 시스템
라이브러리가 필요하다 (Ubuntu: libegl1 libgles2).
"""

# MediaPipe에는 타입 스텁이 없어 이 모듈에서만 알 수 없는 타입 경고를 끈다.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from dlp_models.registry import resolve
from dlp_prelabel.common import iter_frames, model_label, track_boxes
from dlp_prelabel.policy import PrelabelPolicy
from dlp_schema.episode import version_tag
from dlp_schema.labels import (
    BoxKeyframe,
    BoxTrackPayload,
    Hand,
    Keypoint,
    KeypointFrame,
    KeypointTrackPayload,
    LabelRecord,
)
from dlp_schema.predictor import Clip, ModelUnavailableError


def _model(root: Path, name: str) -> tuple[Path, str]:
    path, version = resolve(root, name)
    return path, f"mediapipe-{version}"


def _vision() -> Any:
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python import vision
    except OSError as exc:  # libEGL 등 시스템 라이브러리 없음
        raise ModelUnavailableError(f"MediaPipe를 불러올 수 없습니다: {exc}") from exc
    return mp, mpt, vision


def flip_hand(label: str, input_is_mirrored: bool) -> Hand:
    """MediaPipe 손 판정은 거울상 입력을 가정한다. 거울상이 아니면 왼손·오른손을 뒤집는다."""
    hand = Hand.LEFT if label.lower() == "left" else Hand.RIGHT
    if input_is_mirrored:
        return hand
    return Hand.RIGHT if hand is Hand.LEFT else Hand.LEFT


class MediaPipeHands:
    name = "hands"

    def __init__(
        self, root: Path, policy: PrelabelPolicy, *, ontology_version: str, now: datetime
    ) -> None:
        self.path, version = _model(root, policy.models.hands)
        self.version = f"{version}+p{policy.digest('hands')}"
        self.policy, self.ontology_version, self.now = policy, ontology_version, now

    def run(self, clip: Clip) -> list[LabelRecord]:
        mp, mpt, vision = _vision()
        options = vision.HandLandmarkerOptions(
            base_options=mpt.BaseOptions(
                model_asset_path=str(self.path), delegate=mpt.BaseOptions.Delegate.CPU
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=self.policy.hands.min_score,
        )
        frames: dict[Hand, list[tuple[KeypointFrame, float]]] = {Hand.LEFT: [], Hand.RIGHT: []}
        with vision.HandLandmarker.create_from_options(options) as model:
            for t, img in iter_frames(clip.video):
                h, w = img.shape[:2]
                result = model.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=img), t
                )
                for landmarks, handed in zip(result.hand_landmarks, result.handedness, strict=True):
                    hand = flip_hand(handed[0].category_name, self.policy.hands.input_is_mirrored)
                    points = tuple(
                        Keypoint(x=lm.x * w, y=lm.y * h, visibility=2) for lm in landmarks
                    )
                    frames[hand].append(
                        (KeypointFrame(t_ms=t, points=points), float(handed[0].score))
                    )
        out: list[LabelRecord] = []
        for hand, items in frames.items():
            if not items:
                continue
            payload = KeypointTrackPayload(
                entity_id=f"{hand.value}_hand",
                skeleton="hand21",
                hand=hand,
                keyframes=tuple(f for f, _ in items),
            )
            out.append(
                model_label(
                    label_id=f"{clip.session_id}-{clip.stream_id}-hands-{version_tag(self.version)}-{hand.value}",
                    session_id=clip.session_id,
                    stream_id=clip.stream_id,
                    t_start_ms=items[0][0].t_ms,
                    t_end_ms=items[-1][0].t_ms,
                    ontology_version=self.ontology_version,
                    model_version=self.version,
                    confidence=sum(s for _, s in items) / len(items),
                    payload=payload,
                    now=self.now,
                )
            )
        return out


class MediaPipeObjects:
    """COCO 객체 탐지. 온톨로지에 대응하는 클래스만 남긴다.

    COCO에 없는 청소 도구는 OWLv2 어댑터(adapters/owl_objects.py)가 맡는다.
    """

    name = "objects"

    def __init__(
        self, root: Path, policy: PrelabelPolicy, *, ontology_version: str, now: datetime
    ) -> None:
        self.path, version = _model(root, policy.models.coco_objects)
        self.version = f"{version}+p{policy.digest('objects')}"
        self.policy, self.ontology_version, self.now = policy, ontology_version, now

    def run(self, clip: Clip) -> list[LabelRecord]:
        mp, mpt, vision = _vision()
        op = self.policy.objects
        options = vision.ObjectDetectorOptions(
            base_options=mpt.BaseOptions(
                model_asset_path=str(self.path), delegate=mpt.BaseOptions.Delegate.CPU
            ),
            running_mode=vision.RunningMode.VIDEO,
            score_threshold=op.min_score,
        )
        detections: list[
            tuple[int, list[tuple[str, tuple[float, float, float, float], float]]]
        ] = []
        with vision.ObjectDetector.create_from_options(options) as model:
            for t, img in iter_frames(clip.video):
                result = model.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=img), t
                )
                dets: list[tuple[str, tuple[float, float, float, float], float]] = []
                for d in result.detections:
                    cat = d.categories[0]
                    cls = op.coco_to_ontology.get(cat.category_name)
                    if cls is None:
                        continue
                    b = d.bounding_box
                    dets.append(
                        (
                            cls,
                            (float(b.origin_x), float(b.origin_y), float(b.width), float(b.height)),
                            float(cat.score),
                        )
                    )
                detections.append((t, dets))
        return boxes_to_labels(
            track_boxes(detections, iou_match=op.track_iou, max_gap_ms=op.max_gap_ms),
            clip,
            self.version,
            self.ontology_version,
            self.now,
            prefix="objects",
        )


def boxes_to_labels(
    tracks: list[Any],
    clip: Clip,
    version: str,
    ontology_version: str,
    now: datetime,
    *,
    prefix: str,
    entity_prefix: str = "",
) -> list[LabelRecord]:
    out: list[LabelRecord] = []
    counters: dict[str, int] = {}
    for tr in sorted(tracks, key=lambda x: (min(x.frames), x.key)):
        n = counters[tr.key] = counters.get(tr.key, 0) + 1
        times = sorted(tr.frames)
        keyframes = tuple(
            BoxKeyframe(
                t_ms=t,
                x=tr.frames[t][0][0],
                y=tr.frames[t][0][1],
                w=tr.frames[t][0][2],
                h=tr.frames[t][0][3],
            )
            for t in times
        )
        payload = BoxTrackPayload(
            entity_id=f"{entity_prefix}{tr.key}_{n:02d}", class_id=tr.key, keyframes=keyframes
        )
        out.append(
            model_label(
                label_id=f"{clip.session_id}-{clip.stream_id}-{prefix}-{version_tag(version)}-{tr.key}-{n:02d}",
                session_id=clip.session_id,
                stream_id=clip.stream_id,
                t_start_ms=times[0],
                t_end_ms=times[-1],
                ontology_version=ontology_version,
                model_version=version,
                confidence=sum(s for _, s in tr.frames.values()) / len(tr.frames),
                payload=payload,
                now=now,
            )
        )
    return out
