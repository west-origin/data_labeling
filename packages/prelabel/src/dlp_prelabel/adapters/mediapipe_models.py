"""MediaPipe 실제 모델 어댑터 (CPU): 손 21관절, COCO 객체 탐지.

모델 파일은 `make models`로 받는다 (config/models.yaml). MediaPipe는 EGL/GLES 시스템 라이브러리가
필요하다 (Ubuntu: libegl1 libgles2).

모델 입출력 (MediaPipe Tasks, VIDEO 모드: 프레임마다 증가하는 시각(ms)을 넘겨야 한다):
- HandLandmarker(`hand_landmarker.task`): RGB 프레임 → 손마다 21관절(이미지 크기로 정규화한 0~1 x,
  y)과 손 판정(Left/Right, 거울상 가정). 여기서 픽셀로 바꾼다 (x*W, y*H). skeleton="hand21"
  (MediaPipe 관절 순서: 0 손목, 4 엄지 끝, 8 검지 끝, 12 중지 끝, 16 약지 끝, 20 새끼 끝).
- ObjectDetector(`efficientdet_lite0.tflite`): RGB 프레임 → COCO 클래스 이름·점수·박스 (origin_x,
  origin_y, width, height 픽셀). 출력 좌표계: 스트림 픽셀, 왼쪽 위 원점. 키프레임 시각은 스트림
  PTS ms. 라이선스: `hand_landmarker` allowed, `object_detector` review (COCO 학습), ADR 0010.
"""

# MediaPipe에는 타입 스텁이 없어 이 모듈에서만 알 수 없는 타입 경고를 끈다.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

from collections.abc import Iterable
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
    """(가중치 경로, `mediapipe-<이름>-<해시 12자>` 버전). 없으면 ModelUnavailableError."""
    path, version = resolve(root, name)
    return path, f"mediapipe-{version}"


def _vision() -> Any:
    """MediaPipe 모듈을 늦게 불러온다 (mp, tasks.python, vision).

    libEGL 등 시스템 라이브러리가 없으면 import가 OSError를 내므로 ModelUnavailableError로
    바꾼다.
    """
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python import vision
    except OSError as exc:  # libEGL 등 시스템 라이브러리 없음
        raise ModelUnavailableError(f"MediaPipe를 불러올 수 없습니다: {exc}") from exc
    return mp, mpt, vision


def flip_hand(label: str, input_is_mirrored: bool) -> Hand:
    """MediaPipe 손 판정은 거울상 입력을 가정한다. 거울상이 아니면 왼손·오른손을 뒤집는다.

    Args:
        label: MediaPipe category_name ("Left"/"Right", 대소문자 무관).
        input_is_mirrored: 정책 `hands.input_is_mirrored` (바디캠은 False).
    """
    hand = Hand.LEFT if label.lower() == "left" else Hand.RIGHT
    if input_is_mirrored:
        return hand
    return Hand.RIGHT if hand is Hand.LEFT else Hand.LEFT


def best_per_hand[T](
    detections: Iterable[tuple[Hand, float, T]],
) -> dict[Hand, tuple[float, T]]:
    """한 프레임의 손 탐지에서 손(왼·오른)마다 점수가 가장 높은 것 하나만 남긴다.

    num_hands=2면 두 탐지가 같은 손으로 판정될 수 있다. 둘 다 넣으면 한 트랙에 같은 시각
    키프레임이 두 번 들어간다.
    """
    best: dict[Hand, tuple[float, T]] = {}
    for hand, score, item in detections:
        if hand not in best or score > best[hand][0]:
            best[hand] = (score, item)
    return best


class MediaPipeHands:
    """손 21관절 어댑터 (name="hands").

    version: `mediapipe-<가중치 버전>+p<hands 절 해시>`. 손(왼·오른)마다 트랙 하나를 낸다. 라벨
    ID는 `<세션>-<스트림>-hands-<version_tag>-<left|right>`, entity_id는 `<left|right>_hand`.
    """

    name = "hands"

    def __init__(
        self, root: Path, policy: PrelabelPolicy, *, ontology_version: str, now: datetime
    ) -> None:
        """가중치를 확인하고 버전을 정한다. Raises: ModelUnavailableError."""
        self.path, version = _model(root, policy.models.hands)
        self.version = f"{version}+p{policy.digest('hands')}"
        self.policy, self.ontology_version, self.now = policy, ontology_version, now

    def run(self, clip: Clip) -> list[LabelRecord]:
        """영상 전체 프레임에서 손을 찾아 손마다 hand21 키포인트 트랙을 돌려준다.

        프레임마다 손별 최고 점수 탐지 하나만 쓴다 (`best_per_hand`). 관절 visibility는 모두
        2(보임)로 둔다 (MediaPipe가 관절별 가림을 주지 않는다). 트랙 신뢰도는 손 판정 점수의
        평균이다. 한 손이 잠시 사라져도 같은 트랙에 이어 붙인다 (키프레임 사이에 빈 시간이 생길
        수 있다).
        """
        mp, mpt, vision = _vision()
        options = vision.HandLandmarkerOptions(
            base_options=mpt.BaseOptions(
                model_asset_path=str(self.path), delegate=mpt.BaseOptions.Delegate.CPU
            ),
            running_mode=vision.RunningMode.VIDEO,
            # 바디캠에서는 착용자의 두 손만 본다 (다른 사람 손은 이 어댑터가 구분하지 않는다)
            num_hands=2,
            min_hand_detection_confidence=self.policy.hands.min_score,
        )
        frames: dict[Hand, list[tuple[KeypointFrame, float]]] = {Hand.LEFT: [], Hand.RIGHT: []}
        with vision.HandLandmarker.create_from_options(options) as model:
            for t, img in iter_frames(clip.video):
                # VIDEO 모드는 넘기는 시각(ms)이 계속 증가해야 한다. 스트림 PTS ms를 그대로 넘긴다
                # (PTS 반올림으로 같은 ms가 두 번 나오면 MediaPipe가 오류를 낸다)
                h, w = img.shape[:2]
                result = model.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=img), t
                )
                found = best_per_hand(
                    (
                        flip_hand(handed[0].category_name, self.policy.hands.input_is_mirrored),
                        float(handed[0].score),
                        landmarks,
                    )
                    for landmarks, handed in zip(
                        result.hand_landmarks, result.handedness, strict=True
                    )
                )
                for hand, (score, landmarks) in found.items():
                    points = tuple(
                        # 정규화 좌표(0~1) → 픽셀
                        Keypoint(x=lm.x * w, y=lm.y * h, visibility=2)
                        for lm in landmarks
                    )
                    frames[hand].append((KeypointFrame(t_ms=t, points=points), score))
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
        """가중치를 확인하고 버전을 정한다 (`mediapipe-...+p<objects 절 해시>`)."""
        self.path, version = _model(root, policy.models.coco_objects)
        self.version = f"{version}+p{policy.digest('objects')}"
        self.policy, self.ontology_version, self.now = policy, ontology_version, now

    def run(self, clip: Clip) -> list[LabelRecord]:
        """영상 전체 프레임에서 COCO 객체를 찾아 온톨로지 클래스별 박스 트랙을 돌려준다.

        탐지마다 첫 번째(최고 점수) 카테고리만 보고, `coco_to_ontology`에 없는 클래스는 버린다.
        라벨 ID는 `<세션>-<스트림>-objects-<version_tag>-<클래스>-<순번>`.
        """
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
                    # 첫 카테고리가 최고 점수다
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
    """박스 트랙들을 box_track 라벨로 바꾼다 (객체·도구 어댑터 공용).

    트랙은 (첫 시각, 클래스) 순으로 정렬해 클래스마다 01부터 번호를 매긴다 (같은 입력이면 같은 ID).

    Args:
        tracks: `common.BoxTrack` 목록 (key = 온톨로지 클래스 ID).
        clip: 세션·스트림. version: 모델 버전. ontology_version, now: 라벨 값.
        prefix: 라벨 ID의 단계 이름 (어댑터 `name`과 같아야 러너가 이전 버전을 찾는다).
        entity_prefix: entity_id 앞에 붙일 문자열 (OWLv2는 "ov_").

    Returns:
        트랙마다 라벨 하나. 신뢰도는 키프레임 점수 평균.
    """
    out: list[LabelRecord] = []
    counters: dict[str, int] = {}
    # 결정적 번호: 첫 등장 시각, 클래스 순으로 정렬해 클래스별 번호를 매긴다
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
