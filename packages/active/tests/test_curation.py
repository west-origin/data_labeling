from __future__ import annotations

import math
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from dlp_active.curation import push_to_fiftyone, rates_info, score_fields, stream_sample
from dlp_active.policy import ActivePolicy, load_policy
from dlp_active.rates import correction_rates
from dlp_active.select import SessionScore
from dlp_fixtures.video import vfr_times, write_video
from dlp_media.pts import build_pts_index
from dlp_schema.labels import LabelRecord, Provenance, Source
from dlp_schema.session import Stream, StreamKind, SyncMethod
from dlp_schema.testing import make_label

ROOT = Path(__file__).resolve().parents[3]
W, H = 64, 48


@pytest.fixture(scope="module")
def policy() -> ActivePolicy:
    return load_policy(ROOT)


@pytest.fixture(scope="module")
def video(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[int]]:
    times = vfr_times(np.random.default_rng(3), 1000)
    path = tmp_path_factory.mktemp("v") / "blurred.mp4"
    write_video(path, ((t, np.zeros((H, W, 3), np.uint8)) for t in times), width=W, height=H)
    return path, times


def labels(times: list[int], offset: int) -> list[LabelRecord]:
    m = Provenance(source=Source.MODEL, model_version="m1")
    # 공간 라벨 키프레임은 스트림 PTS 시각, 시간 구간 라벨은 마스터 시각 (= 스트림 시각 + 오프셋)
    t3, t5 = times[3], times[5]
    m3, m5 = t3 + offset, t5 + offset
    return [
        make_label(
            {"kind": "box_track", "entity_id": "cup_1", "class_id": "cup",
             "keyframes": [{"t_ms": t3, "x": 16, "y": 12, "w": 32, "h": 24},
                           {"t_ms": t3 + 7, "x": 0, "y": 0, "w": 8, "h": 8},  # 프레임 사이 → 버림
                           {"t_ms": t5, "x": 0, "y": 0, "w": 8, "h": 8, "outside": True}]},
            label_id="box", stream_id="bodycam", provenance=m, confidence=0.7,
            t_start_ms=t3, t_end_ms=t5,
        ),
        make_label(
            {"kind": "keypoint_track", "entity_id": "hand_r", "skeleton": "hand21", "hand": "right",
             "keyframes": [{"t_ms": t5, "points": [{"x": 32, "y": 24, "visibility": 2}]
                            + [{"x": 0, "y": 0, "visibility": 0}] * 20}]},
            label_id="kp", stream_id="bodycam", t_start_ms=t5, t_end_ms=t5,
        ),
        make_label(
            {"kind": "action", "action_id": "a1", "hand": "right", "verb": "wipe",
             "t_approach_ms": m3, "t_end_ms": m5},
            label_id="act", t_start_ms=m3, t_end_ms=m5,
        ),
        make_label(
            {"kind": "blur_track", "target": "face",
             "keyframes": [{"t_ms": t3, "x": 1, "y": 1, "w": 5, "h": 5}]},
            label_id="blur", stream_id="bodycam", t_start_ms=t3, t_end_ms=t3,
        ),
    ]  # fmt: skip


def stream(offset: int) -> Stream:
    return Stream(
        stream_id="bodycam",
        kind=StreamKind.BODYCAM,
        uri="local://dlp-labeling/x.mp4",
        sync_method=SyncMethod.REFERENCE,
        offset_ms=offset,
    )


def sample(video: tuple[Path, list[int]], policy: ActivePolicy, offset: int = 250) -> Any:
    path, times = video
    score = SessionScore("s1", 1.5, {"correction_rate": 1.5}, 2, [("box_track/cup", 1.5)])
    return stream_sample(
        "s1",
        path,
        build_pts_index(path),
        (W, H),
        stream(offset),
        labels(times, offset),
        policy,
        score_fields(1, score),
    )


def test_labels_map_to_blurred_video_frames(
    video: tuple[Path, list[int]], policy: ActivePolicy
) -> None:
    s = sample(video, policy)
    # VFR 영상에서도 PTS로 맞춘다: 4번째·6번째 프레임 (FiftyOne 번호는 1부터)
    assert sorted(s.frames) == [4, 6]
    (det,) = s.frames[4]
    assert det.box == (0.25, 0.25, 0.5, 0.5)
    assert det.label == "cup" and det.verification == "unreviewed" and det.confidence == 0.7
    (kp,) = s.frames[6]
    assert kp.points is not None and kp.points[0] == (0.5, 0.5) and math.isnan(kp.points[1][0])
    (seg,) = s.temporal
    assert seg.support == (4, 6) and seg.label == "wipe"
    # 블러 라벨은 넣지 않는다
    assert all(x.kind != "blur_track" for xs in s.frames.values() for x in xs)
    assert s.fields["active_top_classes"] == ["box_track/cup"]


def test_push_to_fiftyone_round_trip(video: tuple[Path, list[int]], policy: ActivePolicy) -> None:
    fo: Any = pytest.importorskip("fiftyone")
    rates = correction_rates([], policy)
    name = f"dlp-test-{uuid.uuid4().hex[:8]}"
    ds = push_to_fiftyone(name, [sample(video, policy)], rates_info(rates))
    try:
        loaded = fo.load_dataset(name)
        assert len(loaded) == 1 and loaded.info["overall_correction_rate"] == 0.0
        s = loaded.first()
        assert s.session_id == "s1" and s.active_rank == 1 and s.active_score == 1.5
        dets = s.frames[4].labels.detections
        assert [d.label for d in dets] == ["cup"] and dets[0].label_id == "box"
        assert s.frames[6].keypoints.keypoints[0].label == "hand21/right"
        (seg,) = s.segments.detections
        assert seg.label == "action/wipe" and list(seg.support) == [4, 6]
        # 수정률 분석: 미검수 모델 라벨만 거르기
        view = loaded.filter_labels("frames.labels", fo.ViewField("verification") == "unreviewed")
        assert view.count("frames.labels.detections") == 1
    finally:
        ds.delete()
