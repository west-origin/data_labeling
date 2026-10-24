"""정답을 아는 1인칭 착용자 팔·몸 장면 (ADR 0058).

가슴 장착 바디캠 화면(기본 640x480)에서 왼손·오른손 손목이 화면 아래쪽에 있고, 아래팔은 손가락
반대 방향으로 화면 아래 가장자리 쪽으로 뻗는다. 정답은 프레임·손마다 팔꿈치 픽셀 위치와 화면 안
여부다 (`ArmTruth`). 후반 절반에는 왼손이 화면 아래로 내려가 왼팔꿈치가 화면 밖이 된다.

함께 내는 것
- `hand_labels`: 손마다 hand21 트랙 (정답 출처). 손목(0)·중지 뿌리(9)는 정답 손 방향을 따른다.
- `pose_for(box)`: 가짜 RTMPose (COCO17 관절 (17, 2)와 점수 (17,)). 착용자 박스(합성 박스 또는
  `wearer_box`)면 착용자 자세, `other_box`면 다른 사람(대상자) 자세를 낸다. 착용자 자세는 1인칭
  RTMPose처럼 팔 사슬의 COCO 좌우가 뒤바뀌어 있고(왼손 → COCO 오른팔 10·8·6), 화면 밖 팔꿈치는
  화면 밖 좌표 그대로, 어깨는 화면 밖, 머리 관절은 화면 아래에 지어낸 "보임" 점이다.
- `boxes`: 객체 단계 박스 트랙 (모델 출처): 착용자 몸을 덮은 COCO 사람 박스(억제 대상), 다른 사람
  박스(남길 것), 손에 든 작은 걸레 박스(남길 것), 착용자 몸통을 덮은 큰 수건 박스(억제 대상).
  정답 = `suppressed` (억제해야 할 박스 라벨 ID).
- `other_person`: 다른 사람의 coco17 트랙 (보이는 머리 관절, 착용자 증거의 머리).

시각: 스트림 PTS ms (frame_times). 좌표: 스트림 픽셀, 왼쪽 위 원점. 메모리에서만 쓴다.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from dlp_schema.labels import (
    BoxKeyframe,
    BoxTrackPayload,
    Hand,
    Keypoint,
    KeypointFrame,
    KeypointTrackPayload,
    LabelRecord,
    Provenance,
    Source,
)
from dlp_schema.testing import FIXED_TIME

# 손 크기 (손목→중지 뿌리 px)와 아래팔 길이 (손 크기 배)
PALM_PX = 40.0
FOREARM_PALMS = 3.5
# COCO17 팔 사슬 (손목, 팔꿈치, 어깨). 착용자 자세는 손과 사슬의 COCO 좌우가 뒤바뀐다
_SWAPPED = {Hand.LEFT: (10, 8, 6), Hand.RIGHT: (9, 7, 5)}


@dataclass(frozen=True)
class ArmTruth:
    """한 시각 한 손의 정답: 손목·팔꿈치 픽셀과 팔꿈치가 화면 안인지."""

    wrist: tuple[float, float]
    elbow: tuple[float, float]
    elbow_in_frame: bool


@dataclass
class ArmScenario:
    """착용자 팔·몸 장면 (모듈 docstring)."""

    session_id: str
    size: tuple[int, int]
    frame_times: list[int]
    truth: dict[tuple[int, Hand], ArmTruth]
    hand_labels: list[LabelRecord]
    wearer_poses: dict[int, tuple[NDArray[np.float64], NDArray[np.float64]]]
    other_box: tuple[float, float, float, float]
    other_pose: tuple[NDArray[np.float64], NDArray[np.float64]]
    other_person: LabelRecord
    boxes: list[LabelRecord] = field(default_factory=list[LabelRecord])
    suppressed: set[str] = field(default_factory=set[str])
    wearer_box: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    def pose_for(
        self, t_ms: int, box: Sequence[float]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """가짜 자세 모델: 박스 (x1, y1, x2, y2)가 다른 사람 박스면 그 사람, 아니면 착용자 자세."""
        x1, y1, x2, y2 = (float(v) for v in box)
        ox, oy, ow, oh = self.other_box
        if (
            abs(x1 - ox) < 1
            and abs(y1 - oy) < 1
            and abs(x2 - ox - ow) < 1
            and abs(y2 - oy - oh) < 1
        ):
            return self.other_pose
        return self.wearer_poses[t_ms]


def _unit(x: float, y: float) -> tuple[float, float]:
    """단위 벡터."""
    n = math.hypot(x, y)
    return x / n, y / n


def _hand_frame(t: int, wrist: tuple[float, float], up: tuple[float, float]) -> KeypointFrame:
    """정답 손 방향(up = 손목→중지 뿌리)으로 놓은 hand21 키프레임 (모든 점 보임)."""
    side = (-up[1], up[0])
    pts: list[Keypoint] = []
    for j in range(21):
        if j == 0:
            x, y = wrist
        elif j == 9:
            x, y = wrist[0] + up[0] * PALM_PX, wrist[1] + up[1] * PALM_PX
        else:
            # 손가락 점: 손 방향으로 손 크기 0.5~1.9배, 손가락마다 옆으로 벌린다
            finger, joint = (j - 1) // 4, (j - 1) % 4
            along = PALM_PX * (0.5 + 0.35 * joint + (0.4 if finger else 0.0))
            lateral = PALM_PX * 0.35 * (finger - 2)
            x = wrist[0] + up[0] * along + side[0] * lateral
            y = wrist[1] + up[1] * along + side[1] * lateral
        pts.append(Keypoint(x=round(x, 3), y=round(y, 3), visibility=2))
    return KeypointFrame(t_ms=t, points=tuple(pts))


def _wearer_pose(
    truths: dict[Hand, ArmTruth], size: tuple[int, int], rng: np.random.Generator
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """착용자 자세 (팔 사슬 좌우 뒤바뀜, 손목에 손 크기 0.1배 잡음, 머리는 화면 아래에 지어냄)."""
    w, h = size
    xy = np.tile([w / 2, h * 0.95], (17, 1)).astype(float)
    scores = np.full(17, 0.1)
    xy[:5] = [[w / 2 + 10 * i, h * 0.9] for i in range(5)]
    scores[:5] = 0.5  # 1인칭 RTMPose가 지어내는 "보임" 머리
    for hand, tr in truths.items():
        wi, ei, si = _SWAPPED[hand]
        xy[wi] = np.asarray(tr.wrist) + rng.normal(0, 0.1 * PALM_PX, 2)
        xy[ei] = tr.elbow
        xy[si] = (tr.elbow[0], h + 200.0)  # 어깨는 화면 밖
        scores[[wi, ei, si]] = [0.9, 0.8, 0.2]
    return xy, scores


def _label(
    label_id: str, sid: str, times: Sequence[int], payload: object, model: bool
) -> LabelRecord:
    """정답(사람 출처) 또는 모델 출처 라벨."""
    prov = (
        Provenance(source=Source.MODEL, model_version="fixture-1")
        if model
        else Provenance(source=Source.HUMAN)
    )
    return LabelRecord.model_validate(
        {
            "label_id": label_id, "session_id": sid, "stream_id": "bodycam",
            "t_start_ms": times[0], "t_end_ms": times[-1], "ontology_version": "1.0.0",
            "provenance": prov, "confidence": 0.9 if model else None, "created_at": FIXED_TIME,
            "payload": payload,
        }
    )  # fmt: skip


def _box_track(
    label_id: str, sid: str, times: Sequence[int], class_id: str, box: tuple[float, ...]
) -> LabelRecord:
    """같은 박스가 모든 시각에 있는 모델 박스 트랙."""
    x, y, w, h = box
    kfs = tuple(BoxKeyframe(t_ms=t, x=x, y=y, w=w, h=h) for t in times)
    payload = BoxTrackPayload(
        entity_id=label_id.rsplit("-", 1)[-1], class_id=class_id, keyframes=kfs
    )
    return _label(label_id, sid, times, payload, True)


def _other_person(
    sid: str, times: Sequence[int], size: tuple[int, int]
) -> tuple[
    tuple[float, float, float, float], tuple[NDArray[np.float64], NDArray[np.float64]], LabelRecord
]:
    """화면 위쪽 가운데 서 있는 다른 사람 (박스, 자세, coco17 트랙). 머리 관절이 보인다."""
    w, h = size
    box = (w * 0.4, h * 0.05, w * 0.2, h * 0.45)
    cx, top = w * 0.5, h * 0.08
    xy = np.array([[cx + (i % 3 - 1) * 6, top + 6 * (i // 3)] for i in range(17)], dtype=float)
    xy[5:] = [[cx + (-1) ** i * 25, top + 30 + 9 * (i - 5)] for i in range(5, 17)]
    scores = np.full(17, 0.8)
    points = tuple(Keypoint(x=float(a), y=float(b), visibility=2) for a, b in xy)
    track = KeypointTrackPayload(
        entity_id="person_00", skeleton="coco17",
        keyframes=tuple(KeypointFrame(t_ms=t, points=points) for t in times),
    )  # fmt: skip
    return box, (xy, scores), _label(f"{sid}-body-person", sid, times, track, True)


def generate_arm_scenario(
    seed: int = 0, n_frames: int = 20, step_ms: int = 100, size: tuple[int, int] = (640, 480)
) -> ArmScenario:
    """착용자 팔·몸 장면을 만든다 (같은 seed면 같은 결과).

    왼손: 처음 절반은 손목이 화면 높이 0.55 근처(팔꿈치 화면 안), 나머지 절반은 0.85 근처라 팔꿈치가
    화면 아래 밖이다. 오른손: 늘 0.55 근처(팔꿈치 화면 안). 손 방향은 위쪽(조금 안쪽)이고 아래팔은
    그 반대로 손 크기 x FOREARM_PALMS 길이다.
    """
    rng = np.random.default_rng(seed)
    w, h = size
    sid = f"arm-{seed:03d}"
    times = [i * step_ms for i in range(n_frames)]
    truth: dict[tuple[int, Hand], ArmTruth] = {}
    frames: dict[Hand, list[KeypointFrame]] = {Hand.LEFT: [], Hand.RIGHT: []}
    poses: dict[int, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
    for i, t in enumerate(times):
        now: dict[Hand, ArmTruth] = {}
        for hand, x0, lean in ((Hand.LEFT, 0.33, 0.25), (Hand.RIGHT, 0.67, -0.25)):
            low = hand is Hand.LEFT and i >= n_frames // 2
            # 키프레임 좌표와 같게 소수 셋째 자리로 둔다 (정답 = 라벨 값)
            wrist = (
                round(w * x0 + 5 * math.sin(i / 3), 3),
                round(h * (0.85 if low else 0.55) + 4 * math.cos(i / 4), 3),
            )
            up = _unit(lean, -1.0)
            length = PALM_PX * FOREARM_PALMS
            elbow = (wrist[0] - up[0] * length, wrist[1] - up[1] * length)
            inside = 0 <= elbow[0] < w and 0 <= elbow[1] < h
            now[hand] = ArmTruth(wrist, elbow, inside)
            truth[(t, hand)] = now[hand]
            frames[hand].append(_hand_frame(t, wrist, up))
        poses[t] = _wearer_pose(now, size, rng)
    hand_labels = [
        _label(
            f"{sid}-kp-{hand.value}", sid, times,
            KeypointTrackPayload(
                entity_id=f"{hand.value}_hand", skeleton="hand21", hand=hand,
                keyframes=tuple(frames[hand]),
            ),
            False,
        )
        for hand in (Hand.LEFT, Hand.RIGHT)
    ]  # fmt: skip
    other_box, other_pose, other = _other_person(sid, times, size)
    wearer_box = (w * 0.2, h * 0.4, w * 0.6, h * 0.6)
    boxes = [
        _box_track(f"{sid}-objects-wearer", sid, times, "person", wearer_box),
        _box_track(f"{sid}-objects-other", sid, times, "person", other_box),
        # 왼손에 든 작은 걸레: 손목을 담지만 면적이 작아 남긴다
        _box_track(f"{sid}-tools-rag", sid, times, "rag", (w * 0.28, h * 0.5, 50.0, 40.0)),
        # 착용자 몸통·두 팔을 덮은 큰 "수건" 박스 (OWLv2 오탐): 억제
        _box_track(
            f"{sid}-tools-towel", sid, times, "towel", (w * 0.15, h * 0.35, w * 0.7, h * 0.65)
        ),
    ]
    return ArmScenario(
        session_id=sid, size=size, frame_times=times, truth=truth, hand_labels=hand_labels,
        wearer_poses=poses, other_box=other_box, other_pose=other_pose, other_person=other,
        boxes=boxes, suppressed={f"{sid}-objects-wearer", f"{sid}-tools-towel"},
        wearer_box=wearer_box,
    )  # fmt: skip
