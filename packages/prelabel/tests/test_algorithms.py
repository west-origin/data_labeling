from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from dlp_fixtures.actions import ENTITIES, ActionScenario, generate_action_scenario
from dlp_prelabel.contact import fuse_contacts, glove_contact_intervals, video_contact_intervals
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
    return load_policy(ROOT)


def _gt(sc: ActionScenario) -> list[tuple[int, int, str | None]]:
    return [
        (x.t_start_ms, x.t_end_ms, x.payload.target_id)
        for x in sc.labels
        if isinstance(x.payload, HandStatePayload) and x.payload.contact_target_kind != "none"
    ]


def _static_boxes(sc: ActionScenario) -> list[BoxTrackPayload]:
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
    sc = generate_action_scenario(seed, n_units=10)
    found = glove_contact_intervals(sc.glove_t_ms, sc.glove_pressure, policy.contact.glove)
    truth = _gt(sc)
    assert len(found) == len(truth)
    for c, (start, end, _) in zip(found, truth, strict=True):
        assert abs(c.start_ms - start) <= 33 and abs(c.end_ms - end) <= 33


@pytest.mark.parametrize("seed", range(4))
def test_fused_contacts_pick_the_touched_object(seed: int, policy: PrelabelPolicy) -> None:
    sc = generate_action_scenario(seed, n_units=10)
    hand = next(x.payload for x in sc.labels if isinstance(x.payload, KeypointTrackPayload))
    video = video_contact_intervals(hand, _static_boxes(sc), policy.contact.video)
    fused = fuse_contacts(
        glove_contact_intervals(sc.glove_t_ms, sc.glove_pressure, policy.contact.glove), video
    )
    checked = 0
    for c, (_, _, target) in zip(fused, _gt(sc), strict=True):
        if target in STATIC:
            assert c.target_id == target and c.source == "fused"
            checked += 1
    assert checked > 0


def test_video_contact_without_glove(policy: PrelabelPolicy) -> None:
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


def _person(
    entity: str, t: NDArray[np.float64], wrist_xy: NDArray[np.float64]
) -> KeypointTrackPayload:
    frames: list[KeypointFrame] = []
    for i, tm in enumerate(t):
        pts = [Keypoint(x=0.0, y=0.0, visibility=2) for _ in range(17)]
        pts[9] = Keypoint(x=float(wrist_xy[i, 0]), y=float(wrist_xy[i, 1]), visibility=2)
        pts[10] = pts[9]
        frames.append(KeypointFrame(t_ms=int(tm), points=tuple(pts)))
    return KeypointTrackPayload(entity_id=entity, skeleton="coco17", keyframes=tuple(frames))


def test_wearer_is_the_person_whose_motion_matches_the_bodycam(policy: PrelabelPolicy) -> None:
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
    )
    assert none.entity_id is None
