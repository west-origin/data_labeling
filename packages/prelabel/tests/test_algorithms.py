"""접촉·착용자 알고리즘 단위 테스트 (DB 없음).

정답은 합성 행동 시나리오(`dlp_fixtures.actions.generate_action_scenario`)가 아는 접촉
구간·대상과 장갑 압력이다. 영상 접촉은 고정 위치 개체(서랍·버킷·싱크) 박스만 정답 박스로 쓴다
(잡아서 옮기는 개체는 박스 정답이 없다). 착용자는 합성 운동 신호로 본다. 관련: WP8, ADR 0015,
0026.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from dlp_fixtures.actions import ENTITIES, ActionScenario, generate_action_scenario
from dlp_prelabel.contact import (
    ContactInterval,
    box_at,
    fuse_contacts,
    glove_contact_intervals,
    video_contact_intervals,
)
from dlp_prelabel.policy import PrelabelPolicy, load_policy
from dlp_prelabel.wearer import match_wearer, wrist_speed
from dlp_schema.labels import (
    BoxKeyframe,
    BoxTrackPayload,
    HandStatePayload,
    Keypoint,
    KeypointFrame,
    KeypointTrackPayload,
)

ROOT = Path(__file__).resolve().parents[3]
STATIC = {"drawer_01", "bucket_01", "sink_01"}  # 잡아서 옮기는 묶음의 대상이 아니라 위치가 고정이다


@pytest.fixture(scope="module")
def policy() -> PrelabelPolicy:
    """저장소의 `prelabel.yaml` 정책 (모듈 범위 픽스처)."""
    return load_policy(ROOT)


def _gt(sc: ActionScenario) -> list[tuple[int, int, str | None]]:
    """정답 접촉 구간 (시작, 끝, 대상): 접촉 대상 종류가 none이 아닌 손 상태 라벨."""
    return [
        (x.t_start_ms, x.t_end_ms, x.payload.target_id)
        for x in sc.labels
        if isinstance(x.payload, HandStatePayload) and x.payload.contact_target_kind != "none"
    ]


def _static_boxes(sc: ActionScenario) -> list[BoxTrackPayload]:
    """고정 개체(STATIC)의 정답 위치 둘레 30x30 박스 트랙 (영상 모든 프레임 시각)."""
    return [
        BoxTrackPayload(
            entity_id=eid,
            class_id=cls,
            keyframes=tuple(
                BoxKeyframe(t_ms=t, x=p[0] - 15, y=p[1] - 15, w=30, h=30) for t in sc.frame_times
            ),
        )
        for eid, cls, _, p, _ in ENTITIES
        if eid in STATIC
    ]


@pytest.mark.parametrize("seed", range(4))
def test_glove_contact_timing_within_one_frame(seed: int, policy: PrelabelPolicy) -> None:
    """장갑 압력만으로 찾은 접촉 구간 수가 정답과 같고 경계가 한 프레임(33 ms) 안인지 본다."""
    sc = generate_action_scenario(seed, n_units=10)
    found = glove_contact_intervals(sc.glove_t_ms, sc.glove_pressure, policy.contact.glove)
    truth = _gt(sc)
    assert len(found) == len(truth)
    for c, (start, end, _) in zip(found, truth, strict=True):
        assert abs(c.start_ms - start) <= 33 and abs(c.end_ms - end) <= 33


@pytest.mark.parametrize("seed", range(4))
def test_fused_contacts_pick_the_touched_object(seed: int, policy: PrelabelPolicy) -> None:
    """융합 접촉에서 장갑 구간의 대상이 정답 대상(고정 개체일 때)과 같고 출처가 fused인지 본다."""
    sc = generate_action_scenario(seed, n_units=10)
    hand = next(x.payload for x in sc.labels if isinstance(x.payload, KeypointTrackPayload))
    video = video_contact_intervals(hand, _static_boxes(sc), policy.contact.video)
    fused = fuse_contacts(
        glove_contact_intervals(sc.glove_t_ms, sc.glove_pressure, policy.contact.glove), video
    )
    checked = 0
    # 장갑 구간에서 나온 것만 정답과 맞춘다 (장갑이 놓친 영상 구간은 영상 출처로 따로 남는다)
    from_glove = [c for c in fused if c.source != "video"]
    for c, (_, _, target) in zip(from_glove, _gt(sc), strict=True):
        if target in STATIC:
            assert c.target_id == target and c.source == "fused"
            checked += 1
    assert checked > 0


def test_video_contact_without_glove(policy: PrelabelPolicy) -> None:
    """장갑 없이 영상 휴리스틱만으로 고정 개체 접촉마다 같은 대상 구간이 정답 길이의 절반 넘게
    겹치는지 본다.
    """
    sc = generate_action_scenario(2, n_units=10)
    hand = next(x.payload for x in sc.labels if isinstance(x.payload, KeypointTrackPayload))
    video = video_contact_intervals(hand, _static_boxes(sc), policy.contact.video)
    truth = [(s, e, t) for s, e, t in _gt(sc) if t in STATIC]
    for start, end, target in truth:
        overlap = [
            min(end, v.end_ms) - max(start, v.start_ms)
            for v in video
            if v.target_id == target and min(end, v.end_ms) > max(start, v.start_ms)
        ]
        assert overlap and max(overlap) > 0.5 * (end - start)


def test_video_contact_interpolates_sparse_tool_boxes(policy: PrelabelPolicy) -> None:
    """감사 회귀: 도구 박스는 500 ms마다만 있다. 손 키프레임 시각에서 보간해 접촉을 찾는다.

    정답 근거: 프레임마다 박스가 있을 때(dense)와 결과가 같아야 한다. box_max_gap_ms를 간격의
    절반으로 줄이면 보간하지 않아 접촉이 없다.
    """
    sc = generate_action_scenario(2, n_units=10)
    hand = next(x.payload for x in sc.labels if isinstance(x.payload, KeypointTrackPayload))
    dense = _static_boxes(sc)
    stride = 500
    sparse = [
        b.model_copy(
            update={
                "keyframes": tuple(
                    k
                    for i, k in enumerate(b.keyframes)
                    if i == 0 or k.t_ms // stride != b.keyframes[i - 1].t_ms // stride
                )
            }
        )
        for b in dense
    ]
    assert all(len(s.keyframes) < len(d.keyframes) / 10 for s, d in zip(sparse, dense, strict=True))
    want = video_contact_intervals(hand, dense, policy.contact.video)
    got = video_contact_intervals(hand, sparse, policy.contact.video)
    assert want and [(c.start_ms, c.end_ms, c.target_id) for c in got] == [
        (c.start_ms, c.end_ms, c.target_id) for c in want
    ]
    # 앞뒤 박스 간격이 box_max_gap_ms보다 길면 보간하지 않는다
    tight = policy.contact.video.model_copy(update={"box_max_gap_ms": stride / 2})
    assert video_contact_intervals(hand, sparse, tight) == []


def test_box_interpolation_and_outside_keyframes() -> None:
    """`box_at`의 선형 보간(50 ms → 중간값), 키프레임 시각 정확 일치, 화면 밖 키프레임과의 보간
    금지, 간격 초과, 트랙 범위 밖을 본다.
    """
    track = BoxTrackPayload(
        entity_id="rag_01",
        class_id="rag",
        keyframes=(
            BoxKeyframe(t_ms=0, x=0, y=0, w=10, h=10),
            BoxKeyframe(t_ms=100, x=10, y=20, w=20, h=10),
            BoxKeyframe(t_ms=200, x=10, y=20, w=20, h=10, outside=True),
            BoxKeyframe(t_ms=300, x=10, y=20, w=20, h=10),
        ),
    )
    assert box_at(track, 50, 1000) == (5.0, 10.0, 15.0, 10.0)
    assert box_at(track, 100, 1000) == (10, 20, 20, 10)
    assert box_at(track, 150, 1000) is None  # 화면 밖 키프레임과는 보간하지 않는다
    assert box_at(track, 50, 99) is None
    assert box_at(track, -1, 1000) is None and box_at(track, 301, 1000) is None


def test_fusion_keeps_video_contacts_the_glove_missed() -> None:
    """감사 회귀: 장갑 구간과 겹치지 않는 영상 구간도 영상 출처로 남는다."""
    glove = [ContactInterval(1000, 2000, None, "glove")]
    video = [
        ContactInterval(1200, 1800, "bucket_01", "video"),
        ContactInterval(3000, 3500, "sink_01", "video"),
    ]
    fused = fuse_contacts(glove, video)
    assert fused == [
        ContactInterval(1000, 2000, "bucket_01", "fused"),
        ContactInterval(3000, 3500, "sink_01", "video"),
    ]


def _person(
    entity: str, t: NDArray[np.float64], wrist_xy: NDArray[np.float64]
) -> KeypointTrackPayload:
    """coco17 인물 트랙: 두 손목(9, 10번)을 wrist_xy 궤적으로 움직이고 나머지 관절은 원점."""
    frames: list[KeypointFrame] = []
    for i, tm in enumerate(t):
        pts = [Keypoint(x=0.0, y=0.0, visibility=2) for _ in range(17)]
        pts[9] = Keypoint(x=float(wrist_xy[i, 0]), y=float(wrist_xy[i, 1]), visibility=2)
        pts[10] = pts[9]
        frames.append(KeypointFrame(t_ms=int(tm), points=tuple(pts)))
    return KeypointTrackPayload(entity_id=entity, skeleton="coco17", keyframes=tuple(frames))


def test_wearer_is_the_person_whose_motion_matches_the_bodycam(policy: PrelabelPolicy) -> None:
    """IMU 신호와 같은 순간에 움직인 인물이 착용자로 뽑히는지 본다.

    시나리오: 착용자는 무작위 순간 12번 움직이고 IMU도 그때 흔들린다. 다른 사람 둘은 잡음 운동과
    400 샘플 늦게 움직인 같은 패턴이다. 착용자 상관 > 0.8 > 나머지. IMU가 잡음이면 착용자 없음,
    겹침 요구가 너무 길면 비교 자체를 하지 않는다.
    """
    rng = np.random.default_rng(0)
    t = np.arange(0, 20_000, 33.0)
    bursts = np.zeros(t.size)
    for s in rng.uniform(0, 19_000, 12):  # 착용자가 움직인 순간들
        bursts += np.exp(-(((t - s) / 150) ** 2))
    people: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
    for name, motion in {
        "wearer_track": bursts,
        "other_a": np.convolve(rng.random(t.size), np.ones(15) / 15, "same"),
        "other_b": np.roll(bursts, 400),  # 비슷하지만 다른 때 움직인 사람
    }.items():
        xy = np.cumsum(
            np.stack([motion * 8 + rng.normal(0, 0.3, t.size), np.zeros(t.size)], axis=1), axis=0
        )
        people[name] = wrist_speed(_person(name, t, xy))
    imu_t = np.arange(0, 20_000, 5.0)
    imu_v = np.interp(imu_t, t, bursts) * 3 + rng.normal(0, 0.05, imu_t.size)
    match = match_wearer(
        imu_t,
        imu_v,
        people,
        rate_hz=policy.wearer_matching.rate_hz,
        min_correlation=policy.wearer_matching.min_correlation,
        min_overlap_samples=policy.wearer_matching.min_overlap_samples,
    )
    assert match.entity_id == "wearer_track"
    assert (
        match.scores["wearer_track"] > 0.8 > max(match.scores["other_a"], match.scores["other_b"])
    )

    none = match_wearer(
        imu_t,
        rng.normal(0, 1, imu_t.size),
        {"other_a": people["other_a"]},
        rate_hz=30,
        min_correlation=0.5,
        min_overlap_samples=20,
    )
    assert none.entity_id is None
    # 겹치는 길이가 min_overlap_samples보다 짧으면 비교하지 않는다
    short = match_wearer(
        imu_t, imu_v, people, rate_hz=30, min_correlation=0.5, min_overlap_samples=10**6
    )
    assert short.entity_id is None and short.scores == {}
