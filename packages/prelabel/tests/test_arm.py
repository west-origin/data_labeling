"""착용자 팔 골격 arm3와 착용자 몸 억제 (ADR 0058).

정답은 합성 픽스처 `dlp_fixtures.arm.generate_arm_scenario`가 낸다: 프레임·손마다 정답 팔꿈치
위치와 화면 안 여부, 억제해야 할 박스. 자세 모델은 픽스처의 가짜 RTMPose(1인칭처럼 팔 사슬 좌우가
뒤바뀜)로 대신한다. 서비스 없이 돈다.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from dlp_fixtures.arm import PALM_PX, ArmScenario, generate_arm_scenario
from dlp_prelabel.adapters.rtmpose import BodyPass, body_pass
from dlp_prelabel.arm import ArmFrame, PoseCandidate, arm_tracks, estimate_arms, synthetic_box
from dlp_prelabel.common import BoxTrack
from dlp_prelabel.constraints3d import forearm_lengths
from dlp_prelabel.lift3d import lift_key, lift_tracks
from dlp_prelabel.policy import PrelabelPolicy, load_policy
from dlp_prelabel.wearer_suppress import suppress_box_tracks, wearer_tag
from dlp_schema.labels import (
    BoxTrackPayload,
    Constraint3DFlag,
    Hand,
    KeypointTrackPayload,
    LabelRecord,
)
from dlp_schema.ontology import load_ontology
from dlp_schema.predictor import Clip
from dlp_schema.testing import FIXED_TIME
from dlp_schema.validation import check_label
from dlp_schema.wearer_body import HandAt, WearerEvidence, hand_at

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def policy() -> PrelabelPolicy:
    """저장소 프리라벨 정책."""
    return load_policy(ROOT)


@pytest.fixture(scope="module")
def scene() -> ArmScenario:
    """착용자 팔·몸 장면 (seed 0)."""
    return generate_arm_scenario(0)


def _hands_at(scene: ArmScenario, t: int) -> dict[Hand, HandAt]:
    """시각 t의 정답 손 (hand21 키프레임)."""
    out: dict[Hand, HandAt] = {}
    for x in scene.hand_labels:
        p = x.payload
        assert isinstance(p, KeypointTrackPayload) and p.hand is not None
        frame = next(f for f in p.keyframes if f.t_ms == t)
        h = hand_at(p.hand, frame)
        assert h is not None
        out[p.hand] = h
    return out


def _cands(scene: ArmScenario, t: int) -> list[PoseCandidate]:
    """시각 t의 착용자 자세 후보 하나 (가짜 RTMPose)."""
    xy, sc = scene.wearer_poses[t]
    return [PoseCandidate(xy, sc)]


def test_elbow_follows_hand_side_and_off_frame_elbow_is_not_visible(
    scene: ArmScenario, policy: PrelabelPolicy
) -> None:
    """정답: 화면 안 팔꿈치는 정답 위치 그대로 보임(2), 화면 밖 팔꿈치는 보이지 않음(0, 손목 자리).

    가짜 자세의 팔 사슬은 COCO 좌우가 뒤바뀌어 있지만 팔은 hand21 손 판정을 따른다. 손목은 늘
    hand21 손목이고, 사슬 손목과의 일치 정도(잡음 0.1 손 크기)가 손목 점 신뢰도다. 어깨는 늘 0이다.
    """
    seen = {Hand.LEFT: 0, Hand.RIGHT: 0}
    for t in scene.frame_times:
        arms = estimate_arms(t, _hands_at(scene, t), _cands(scene, t), scene.size, policy.arm)
        for hand, f in arms.items():
            truth = scene.truth[(t, hand)]
            wrist, elbow, shoulder = f.points
            assert (wrist.x, wrist.y, wrist.visibility) == (*truth.wrist, 2)
            assert wrist.confidence is not None and wrist.confidence > 0.5
            assert shoulder.visibility == 0
            if truth.elbow_in_frame:
                assert f.reason == "ok" and elbow.visibility == 2
                assert math.dist((elbow.x, elbow.y), truth.elbow) < 0.01
                seen[hand] += 1
            else:
                assert f.reason == "off_frame" and elbow.visibility == 0
                assert (elbow.x, elbow.y) == truth.wrist
    half = len(scene.frame_times) // 2
    assert seen == {Hand.LEFT: half, Hand.RIGHT: len(scene.frame_times)}


def test_implausible_elbows_are_not_invented(scene: ArmScenario, policy: PrelabelPolicy) -> None:
    """손가락 쪽 팔꿈치(direction), 낮은 점수(low_score), 먼 사슬(no_chain), 좌우가 엇갈린 두 팔꿈치
    (side_conflict)는 모두 보이지 않음이다."""
    t = scene.frame_times[0]
    hands = _hands_at(scene, t)
    xy, sc = scene.wearer_poses[t]
    left = scene.truth[(t, Hand.LEFT)]
    # 왼팔 팔꿈치를 손가락 방향(손목 위)으로 옮긴다: 길이는 같고 방향만 반대
    flipped = xy.copy()
    flipped[8] = 2 * np.asarray(left.wrist) - np.asarray(left.elbow)
    arms = estimate_arms(t, hands, [PoseCandidate(flipped, sc)], scene.size, policy.arm)
    assert arms[Hand.LEFT].reason == "direction" and arms[Hand.LEFT].points[1].visibility == 0
    low = sc.copy()
    low[8] = 0.2
    arms = estimate_arms(t, hands, [PoseCandidate(xy, low)], scene.size, policy.arm)
    assert arms[Hand.LEFT].reason == "low_score"
    far = xy.copy()
    far[10] += 5 * PALM_PX
    arms = estimate_arms(t, hands, [PoseCandidate(far, sc)], scene.size, policy.arm)
    assert arms[Hand.LEFT].reason == "no_chain" and arms[Hand.LEFT].wrist_agreement is None
    assert arms[Hand.LEFT].points[0].confidence is None
    # 두 팔꿈치가 모두 타당(길이·방향)하지만 왼팔 팔꿈치(330)가 오른팔 팔꿈치(300)보다 오른쪽
    crossed = xy.copy()
    crossed[8] = (330.0, 380.0)  # 왼손 사슬 = COCO 오른팔 (10, 8, 6)
    crossed[7] = (300.0, 380.0)  # 오른손 사슬 = COCO 왼팔 (9, 7, 5)
    arms = estimate_arms(t, hands, [PoseCandidate(crossed, sc)], scene.size, policy.arm)
    assert {f.reason for f in arms.values()} == {"side_conflict"}
    assert all(f.points[1].visibility == 0 for f in arms.values())


def test_arm_tracks_link_to_the_hand_and_validate(
    scene: ArmScenario, policy: PrelabelPolicy
) -> None:
    """팔 트랙: 손마다 하나, entity `<손>_arm`, hand = 손, arm3. 보이는 팔꿈치가 없는 손은 트랙이
    없다.
    키프레임 시각은 손 키프레임 시각이고 온톨로지 검증을 통과한다."""
    frames: dict[Hand, list[ArmFrame]] = {}
    for t in scene.frame_times:
        for f in estimate_arms(
            t, _hands_at(scene, t), _cands(scene, t), scene.size, policy.arm
        ).values():
            frames.setdefault(f.hand, []).append(f)
    clip = Clip(scene.session_id, "bodycam", Path("unused.mp4"))
    labels = arm_tracks(
        frames, clip, prefix="body", version="v1", ontology_version="1.0.0", now=FIXED_TIME
    )
    ontology = load_ontology(ROOT / "config/ontology/v1")
    assert [x.label_id.rsplit("-", 2)[-2:] for x in labels] == [["arm", "left"], ["arm", "right"]]
    for x in labels:
        p = x.payload
        assert isinstance(p, KeypointTrackPayload)
        assert p.hand is not None and (p.skeleton, p.entity_id) == ("arm3", f"{p.hand.value}_arm")
        assert [f.t_ms for f in p.keyframes] == scene.frame_times
        assert check_label(x, ontology) == []
    # 왼손 팔꿈치를 모두 지우면 왼팔 트랙은 없다
    hidden = {
        Hand.LEFT: [
            ArmFrame(
                f.t_ms, f.hand, (f.points[0], f.points[2], f.points[2]), None, "off_frame", None
            )
            for f in frames[Hand.LEFT]
        ],
        Hand.RIGHT: frames[Hand.RIGHT],
    }
    only = arm_tracks(
        hidden, clip, prefix="body", version="v1", ontology_version="1.0.0", now=FIXED_TIME
    )
    assert [x.payload.hand for x in only if isinstance(x.payload, KeypointTrackPayload)] == [
        Hand.RIGHT
    ]


def test_synthetic_box_reaches_the_bottom_edge(scene: ArmScenario, policy: PrelabelPolicy) -> None:
    """합성 착용자 박스: 손 박스를 옆·위로 넓히고 아래는 화면 끝, 화면 밖으로 나가지 않는다."""
    hands = list(_hands_at(scene, 0).values())
    x0, y0, x1, y1 = synthetic_box(hands, scene.size, policy.arm.box)
    assert y1 == scene.size[1] and 0 <= x0 < x1 <= scene.size[0]
    assert all(x0 <= h.wrist[0] <= x1 and y0 <= h.wrist[1] for h in hands)


def _run_body(
    scene: ArmScenario, policy: PrelabelPolicy, wearer: WearerEvidence | None
) -> BodyPass:
    """가짜 탐지기(착용자 박스 + 다른 사람 박스)·가짜 자세로 `body_pass`를 돌린다."""
    w, h = scene.size
    now = {"t": 0}

    def frames() -> Iterator[tuple[int, NDArray[np.uint8]]]:
        for t in scene.frame_times:
            now["t"] = t
            yield t, np.zeros((h, w, 3), np.uint8)

    wx, wy, ww, wh = scene.wearer_box
    ox, oy, ow, oh = scene.other_box

    def detect(_: NDArray[np.uint8]) -> NDArray[np.float32]:
        return np.array([[wx, wy, wx + ww, wy + wh], [ox, oy, ox + ow, oy + oh]], np.float32)

    def pose(
        _: NDArray[np.uint8], boxes: list[list[float]]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        out = [scene.pose_for(now["t"], b) for b in boxes]
        return np.stack([k for k, _ in out]), np.stack([s for _, s in out])

    return body_pass(frames(), detect, pose, policy, wearer)


def test_body_pass_drops_wearer_poses_keeps_the_other_person_and_emits_arms(
    scene: ArmScenario, policy: PrelabelPolicy
) -> None:
    """착용자 자세(손목이 착용자 손목, 화면 아래)는 사람 탐지에서 빠지고 팔 후보가 된다. 다른 사람
    (위쪽, 머리 보임)은 남는다. 팔 키프레임은 arm.frame_stride_ms(100) 간격 = 모든 프레임이다."""
    evidence = WearerEvidence.from_labels([*scene.hand_labels, scene.other_person], "bodycam")
    result = _run_body(scene, policy, evidence)
    n = len(scene.frame_times)
    assert result.wearer is not None and result.wearer.poses == n
    assert all(len(dets) == 1 for _, dets in result.detections)
    assert result.arm_report is not None
    assert result.arm_report.keyframes == {"left": n, "right": n}
    assert result.arm_report.elbows == {"left": n // 2, "right": n}
    assert {f.reason for f in result.arms[Hand.LEFT][n // 2 :]} == {"off_frame"}
    # 착용자 증거가 없으면(3인칭 스트림) 억제·팔 단계를 하지 않는다: 두 사람 모두 탐지에 남는다
    plain = _run_body(scene, policy, None)
    assert plain.wearer is None and plain.arm_report is None and not plain.arms
    assert all(len(dets) == 2 for _, dets in plain.detections)


def _box_tracks(labels: list[LabelRecord]) -> list[BoxTrack]:
    """박스 트랙 라벨 → 어댑터 내부 `BoxTrack` (키 = 클래스, 라벨 ID는 대응표로 돌려준다)."""
    out: list[BoxTrack] = []
    for x in labels:
        p = x.payload
        assert isinstance(p, BoxTrackPayload)
        out.append(BoxTrack(p.class_id, {k.t_ms: ((k.x, k.y, k.w, k.h), 0.9) for k in p.keyframes}))
    return out


def test_wearer_boxes_are_suppressed_and_other_people_and_held_tools_kept(
    scene: ArmScenario, policy: PrelabelPolicy
) -> None:
    """정답(`scene.suppressed`): 착용자 몸을 덮은 사람 박스와 몸통을 덮은 큰 도구 박스는 억제,
    다른 사람 박스와 손에 든 작은 걸레는 남는다. 증거가 없거나 정책이 꺼지면 그대로다."""
    evidence = WearerEvidence.from_labels([*scene.hand_labels, scene.other_person], "bodycam")
    tracks = _box_tracks(scene.boxes)
    kept, report = suppress_box_tracks(tracks, evidence, policy.wearer_suppression, scene.size)
    ids = {id(t): x.label_id for t, x in zip(tracks, scene.boxes, strict=True)}
    assert {ids[id(t)] for t in tracks if t not in kept} == scene.suppressed
    assert report is not None and report.tracks == 2 and report.kept_person_tracks == 1
    assert report.by_class == {"person": 1, "towel": 1}
    same, none = suppress_box_tracks(tracks, None, policy.wearer_suppression, scene.size)
    assert same == tracks and none is None
    off = policy.wearer_suppression.model_copy(update={"enabled": False})
    assert suppress_box_tracks(tracks, evidence, off, scene.size)[0] == tracks
    # 사람 아닌 클래스 규칙을 끄면 큰 도구 박스는 남는다
    persons = policy.wearer_suppression.model_copy(update={"other_min_area": None})
    _, only = suppress_box_tracks(tracks, evidence, persons, scene.size)
    assert only is not None and only.by_class == {"person": 1}


def test_wearer_policy_hash_is_in_the_suppressing_stages_versions(policy: PrelabelPolicy) -> None:
    """억제 정책 해시는 억제를 적용하는 단계 버전에만 들어간다 (바뀌면 그 단계가 다시 돈다)."""
    tag = wearer_tag(policy, "objects")
    assert tag.startswith("+w") and tag == wearer_tag(policy, "tools") == wearer_tag(policy, "body")
    assert wearer_tag(policy, "hands") == ""
    changed = policy.model_copy(
        update={"wearer_suppression": policy.wearer_suppression.model_copy(update={"margin": 0.2})}
    )
    assert wearer_tag(changed, "objects") != tag


class _FlatDepth:
    """모든 픽셀 깊이가 1 m인 가짜 깊이 모델."""

    def predict(self, image: NDArray[np.uint8]) -> NDArray[np.float32]:
        return np.ones(image.shape[:2], np.float32)


def test_elbow_is_lifted_to_3d_with_forearm_length_flags(
    scene: ArmScenario, policy: PrelabelPolicy
) -> None:
    """팔꿈치 3D(`<손>_arm`, elbow): 깊이 1 m 평면이면 화면 거리 / fx가 아래팔 길이다.

    정답: 화각 90도 근사(fx = 가로/2 = 320 px)에서 아래팔 화면 길이 140 px → 0.4375 m. 저장소 범위
    [0.15, 0.40] 밖이라 모든 팔꿈치 표본에 forearm_length가 붙고 신뢰도가 낮아진다. 범위를 넓히면
    표시가 없다. 화면 밖 팔꿈치(보이지 않음)는 3D가 없다. 손목 3D는 손 트랙이 낸다.
    """
    evidence = WearerEvidence.from_labels([*scene.hand_labels, scene.other_person], "bodycam")
    result = _run_body(scene, policy, evidence)
    clip = Clip(scene.session_id, "bodycam", Path("unused.mp4"))
    arms = arm_tracks(
        result.arms, clip, prefix="body", version="v1", ontology_version="1.0.0", now=FIXED_TIME
    )
    assert all(lift_key(x.payload) is not None for x in arms)
    w, h = scene.size

    def frames() -> Iterator[tuple[int, NDArray[np.uint8]]]:
        return ((t, np.zeros((h, w, 3), np.uint8)) for t in scene.frame_times)

    depth = policy.depth.model_copy(update={"frame_stride_ms": 0})
    out = lift_tracks(frames(), _FlatDepth(), [*scene.hand_labels, *arms], None, depth)
    left, right = out[("left_arm", "elbow")], out[("right_arm", "elbow")]
    assert len(right) == len(scene.frame_times) and len(left) == len(scene.frame_times) // 2
    lengths = forearm_lengths(out[("right_hand", "wrist")], right)
    assert all(abs(d - 140 / 320) < 0.01 for d in lengths.values())
    assert all(s.flags == (Constraint3DFlag.FOREARM_LENGTH,) for s in right)
    wide = depth.model_copy(
        update={"constraints": depth.constraints.model_copy(update={"forearm_m": (0.15, 0.5)})}
    )
    loose = lift_tracks(frames(), _FlatDepth(), [*scene.hand_labels, *arms], None, wide)
    assert all(s.flags is None for s in loose[("right_arm", "elbow")])
    assert all(
        s.confidence is not None and t.confidence is not None and s.confidence < t.confidence
        for s, t in zip(right, loose[("right_arm", "elbow")], strict=True)
    )
