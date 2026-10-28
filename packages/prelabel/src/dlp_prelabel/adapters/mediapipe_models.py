"""MediaPipe 실제 모델 어댑터 (CPU): 손 21관절, COCO 객체 탐지.

모델 파일은 `make models`로 받는다 (config/models.yaml). MediaPipe는 EGL/GLES 시스템 라이브러리가
필요하다 (Ubuntu: libegl1 libgles2).

모델 입출력 (MediaPipe Tasks, VIDEO 모드: 프레임마다 엄격히 증가하는 시각(ms)을 넘겨야 한다.
반올림한 PTS ms가 앞 프레임과 같은 프레임은 `common.strictly_increasing`이 건너뛴다):
- HandLandmarker(`hand_landmarker.task`): RGB 프레임 → 손마다 21관절(이미지 크기로 정규화한 0~1 x,
  y), 손 중심 3D 21점(hand_world_landmarks, 미터, 원점은 손의 대략적인 기하 중심)과 손 판정
  (Left/Right, 거울상 셀카 기준이라 1인칭 화면은 그대로 착용자 기준, `flip_hand`). 2D는 여기서
  픽셀로 바꾸고 (x*W, y*H), 3D는 키프레임 world에 싣는다
  (ADR 0044, `lift3d`가 손가락 끝 3D에 쓴다). skeleton="hand21"
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
from dlp_prelabel.common import iou, iter_frames, model_label, strictly_increasing, track_boxes
from dlp_prelabel.policy import PrelabelPolicy
from dlp_prelabel.temporal import fill_gaps, filter_tracks, smooth_hand_frames
from dlp_prelabel.wearer_suppress import WearerAware, suppress_box_tracks, wearer_tag
from dlp_schema.episode import version_tag
from dlp_schema.labels import (
    BoxKeyframe,
    BoxTrackPayload,
    Hand,
    Keypoint,
    KeypointFrame,
    KeypointTrackPayload,
    LabelRecord,
    WorldPoint,
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


def flip_hand(label: str, swap_handedness: bool) -> Hand:
    """MediaPipe 손 판정(category_name) → 착용자 기준 손 (F-PAR-61).

    MediaPipe 손 판정은 손 모양의 좌우성(엄지가 어느 쪽인가)으로 정한다. 학습 기준은 "거울상 셀카"
    화면이다: 사람이 카메라를 마주 보고 화면을 좌우로 뒤집은 영상이다. 손등 쪽에서 손을 보는 1인칭
    (바디캠·머리캠, 거울상 아님) 화면에서 손의 2D 모양은 거울상 셀카 화면의 같은 손과 좌우가 같다.
    카메라를 등지고 손을 보는 시점은 마주 본 화면을 좌우로 뒤집은 것과 같기 때문이다. 그래서 1인칭
    화면은 뒤집지 않는다. 뒤집어야 하는 것은 사람이 거울상이 아닌 카메라를 마주 볼 때뿐이다 (3인칭
    정면 등). 2026-10-23 파일럿 확인: 바로 세운 1인칭 실영상에서 뒤집지 않은 판정의 왼손이
    두 손 프레임 대부분에서 화면 왼쪽에 왔다 (pilot-real-2026-10-22.md).

    Args:
        label: MediaPipe category_name ("Left"/"Right", 대소문자 무관).
        swap_handedness: 정책 `hands.swap_handedness`. 1인칭 바디캠은 False.
    """
    hand = Hand.LEFT if label.lower() == "left" else Hand.RIGHT
    if not swap_handedness:
        return hand
    return Hand.RIGHT if hand is Hand.LEFT else Hand.LEFT


def _bbox(landmarks: Any) -> tuple[float, float, float, float]:
    """정규화 관절 좌표의 외접 박스 (x, y, w, h)."""
    xs, ys = [float(p.x) for p in landmarks], [float(p.y) for p in landmarks]
    return (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))


def drop_duplicate_hands[T](
    detections: list[tuple[Hand, float, T]],
    boxes: list[tuple[float, float, float, float]],
    iou_max: float,
) -> list[tuple[Hand, float, T]]:
    """같은 손을 두 번 본 탐지(관절 외접 박스 IoU >= iou_max)는 점수 높은 것만 남긴다.

    MediaPipe는 num_hands=2에서 한 손을 Left·Right 두 탐지로 낼 때가 있다. 그대로 두면 두 손 트랙이
    한 손에 겹친다 (2026-10-23 파일럿). 순서는 입력 순서를 지킨다.
    """
    order = sorted(range(len(detections)), key=lambda i: -detections[i][1])
    kept: list[int] = []
    for i in order:
        if all(iou(boxes[i], boxes[j]) < iou_max for j in kept):
            kept.append(i)
    return [detections[i] for i in sorted(kept)]


def _frame_box(frame: KeypointFrame) -> tuple[float, float, float, float]:
    """키프레임 관절(픽셀)의 외접 박스."""
    xs, ys = [p.x for p in frame.points], [p.y for p in frame.points]
    return (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))


def drop_filled_overlaps(
    tracks: dict[Hand, list[tuple[KeypointFrame, float]]], iou_max: float
) -> dict[Hand, list[tuple[KeypointFrame, float]]]:
    """채운(보간) 키프레임이 같은 시각 다른 손의 관측 키프레임과 겹치면(IoU >= iou_max) 뺀다.

    MediaPipe가 한 손의 판정을 프레임마다 Left·Right로 바꾸면, 한쪽 트랙의 짧은 끊김을 채운 손이
    다른 트랙의 관측 손 위에 놓인다 (2026-10-23 파일럿: 채운 두 손 프레임의 두 손목 간격 중앙값 = 손
    크기의 0.08배). 관측 키프레임은 건드리지 않는다.
    """
    observed = {
        hand: {f.t_ms: _frame_box(f) for f, _ in items if not f.interpolated}
        for hand, items in tracks.items()
    }
    out: dict[Hand, list[tuple[KeypointFrame, float]]] = {}
    for hand, items in tracks.items():
        others = [boxes for h, boxes in observed.items() if h is not hand]
        out[hand] = [
            (f, s)
            for f, s in items
            if not f.interpolated
            or all(f.t_ms not in o or iou(_frame_box(f), o[f.t_ms]) < iou_max for o in others)
        ]
    return out


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


def hand_frame(t_ms: int, landmarks: Any, world: Any, width: int, height: int) -> KeypointFrame:
    """MediaPipe 손 하나 → 키프레임 (2D 픽셀 점 + 손 중심 3D 점, ADR 0044).

    Args:
        t_ms: 스트림 PTS ms.
        landmarks: `hand_landmarks[i]` (정규화 0~1 x, y). 픽셀로 바꾼다 (x*W, y*H).
        world: `hand_world_landmarks[i]` (미터, 원점은 손의 대략적인 기하 중심, 카메라와 같은
            축). 0.1 mm(소수 넷째 자리)로 반올림해 싣는다. 점 수가 다르면 싣지 않는다.
        width, height: 프레임 크기(px).
    """
    points = tuple(Keypoint(x=lm.x * width, y=lm.y * height, visibility=2) for lm in landmarks)
    world_points = tuple(
        WorldPoint(x=round(p.x, 4), y=round(p.y, 4), z=round(p.z, 4)) for p in world
    )
    return KeypointFrame(
        t_ms=t_ms, points=points, world=world_points if len(world_points) == len(points) else None
    )


class MediaPipeHands:
    """손 21관절 어댑터 (name="hands").

    version: `mediapipe-<가중치 버전>+rgb1+w1+hd2+sm1+dd2+p<hands 절 해시>`.
    RGB 행 패딩 제거(rgb1), 손 중심 3D 점 싣기(w1, ADR 0044), 1인칭 손 판정을 뒤집지 않게 바로잡은
    손 좌우 의미(hd2, F-PAR-61), 2D 평활·짧은 끊김 채우기(sm1, ADR 0054), 한 손을 두 번 본 탐지
    빼기와 다른 손 위에 놓인 채운 키프레임 빼기(dd2) 버전을 포함한다.
    손(왼·오른)마다
    트랙 하나를 낸다. 라벨 ID는 `<세션>-<스트림>-hands-<version_tag>-<left|right>`, entity_id는
    `<left|right>_hand`.
    """

    name = "hands"

    def __init__(
        self, root: Path, policy: PrelabelPolicy, *, ontology_version: str, now: datetime
    ) -> None:
        """가중치를 확인하고 버전을 정한다. Raises: ModelUnavailableError."""
        self.path, version = _model(root, policy.models.hands)
        self.version = f"{version}+rgb1+w1+hd2+sm1+dd2+p{policy.digest('hands')}"
        self.policy, self.ontology_version, self.now = policy, ontology_version, now

    def run(self, clip: Clip) -> list[LabelRecord]:
        """영상 전체 프레임에서 손을 찾아 손마다 hand21 키포인트 트랙을 돌려준다.

        프레임마다 손별 최고 점수 탐지 하나만 쓴다 (`best_per_hand`). 키프레임에는 2D 점과 손 중심
        3D 점(hand_world_landmarks, `hand_frame`)을 함께 싣는다. 관절 visibility는 모두
        2(보임)로 둔다 (MediaPipe가 관절별 가림을 주지 않는다). 트랙 신뢰도는 손 판정 점수의
        평균이다. 한 손이 잠시 사라져도 같은 트랙에 이어 붙인다 (키프레임 사이에 빈 시간이 생길
        수 있다). 라벨로 만들기 전에 손마다 2D를 평활하고(`hands.smoothing`), 짧은 끊김을 보간
        키프레임으로 채운다(`hands.gap_fill`, 채운 키프레임 점수는 채움 신뢰도, ADR 0054).
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
        times: list[int] = []
        size = (1, 1)
        with vision.HandLandmarker.create_from_options(options) as model:
            # VIDEO 모드는 넘기는 시각(ms)이 엄격히 증가해야 한다. 스트림 PTS ms를 그대로 넘기되,
            # PTS 반올림으로 앞 프레임과 같은 ms가 된 프레임은 건너뛴다 (`strictly_increasing`).
            # 그대로 넘기면 MediaPipe가 예외를 내 세션 전체 프리라벨이 실패했다.
            for t, img in strictly_increasing(iter_frames(clip.video)):
                h, w = img.shape[:2]
                times.append(t)
                size = (w, h)
                result = model.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=img), t
                )
                dets = [
                    (
                        flip_hand(handed[0].category_name, self.policy.hands.swap_handedness),
                        float(handed[0].score),
                        (landmarks, world),
                    )
                    for landmarks, world, handed in zip(
                        result.hand_landmarks,
                        result.hand_world_landmarks,
                        result.handedness,
                        strict=True,
                    )
                ]
                boxes = [_bbox(lm) for lm, _ in (d[2] for d in dets)]
                found = best_per_hand(
                    drop_duplicate_hands(dets, boxes, self.policy.hands.duplicate_iou)
                )
                for hand, (score, (landmarks, world)) in found.items():
                    frames[hand].append((hand_frame(t, landmarks, world, w, h), score))
        out: list[LabelRecord] = []
        cleaned = drop_filled_overlaps(
            {hand: self._clean(obs, times, size) for hand, obs in frames.items() if obs},
            self.policy.hands.duplicate_iou,
        )
        for hand, items in cleaned.items():
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

    def _clean(
        self, observed: list[tuple[KeypointFrame, float]], times: list[int], size: tuple[int, int]
    ) -> list[tuple[KeypointFrame, float]]:
        """한 손 관측을 평활하고 짧은 끊김을 채운 (키프레임, 점수) 목록 (ADR 0054)."""
        hp = self.policy.hands
        score = {f.t_ms: s for f, s in observed}
        smooth = smooth_hand_frames([f for f, _ in observed], size[0], size[1], hp.smoothing)
        filled = fill_gaps(smooth, times, hp.gap_fill)
        return [(f, hp.gap_fill.confidence if f.interpolated else score[f.t_ms]) for f in filled]


class MediaPipeObjects(WearerAware):
    """COCO 객체 탐지. 온톨로지에 대응하는 클래스만 남긴다.

    COCO에 없는 청소 도구는 OWLv2 어댑터(adapters/owl_objects.py)가 맡는다. 바디캠에서는 착용자
    자신의 팔·몸통을 잡은 박스 트랙(대부분 person)을 라벨로 만들지 않는다 (ADR 0058,
    `wearer_suppress.suppress_box_tracks`, 착용자 증거는 러너가 `set_wearer`로 넘긴다).
    """

    name = "objects"

    def __init__(
        self, root: Path, policy: PrelabelPolicy, *, ontology_version: str, now: datetime
    ) -> None:
        """가중치를 확인하고 버전을 정한다.

        버전: `mediapipe-...+rgb1+p<objects 절 해시>+w<wearer_suppression 절 해시>`.
        """
        self.path, version = _model(root, policy.models.coco_objects)
        self.version = f"{version}+rgb1+p{policy.digest('objects')}{wearer_tag(policy, self.name)}"
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
        size = (0, 0)
        with vision.ObjectDetector.create_from_options(options) as model:
            # 손 어댑터와 같은 이유로 같은 ms의 두 번째 프레임은 건너뛴다
            # (VIDEO 모드 시각 증가 조건)
            for t, img in strictly_increasing(iter_frames(clip.video)):
                size = (int(img.shape[1]), int(img.shape[0]))
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
        tracks = track_boxes(detections, iou_match=op.track_iou, max_gap_ms=op.max_gap_ms)
        # 착용자 자신의 몸 박스 트랙은 라벨로 만들지 않는다 (ADR 0058, 요약 last_wearer에만 센다).
        # 버전 태그(`wearer_tag`)와 같은 조건(stages에 objects)일 때만 억제한다 (F-PAR-73)
        kept, self.last_wearer = filter_tracks(tracks, op.track_filter), None
        ws = self.policy.wearer_suppression
        if self.name in ws.stages:
            kept, self.last_wearer = suppress_box_tracks(kept, self.wearer, ws, size)
        return boxes_to_labels(
            kept, clip, self.version, self.ontology_version, self.now, prefix="objects"
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
