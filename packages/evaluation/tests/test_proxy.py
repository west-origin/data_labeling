"""정답 없는 품질 대리 지표(`dlp_eval.proxy`) 시험.

정답: 깜박임 구간 수·짧은 트랙 수·손 화면 위치·손목 일치 거리·얼굴과 사람 박스 위치·손목 3D 거리와
점프를 알고 만든 합성 라벨에서 계산한 값.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from dlp_eval.policy import ProxyPolicy, load_policy
from dlp_eval.proxy import markdown, motion, proxy_report, runs
from dlp_schema.labels import LabelRecord, Provenance, Source, Trajectory3DSample
from dlp_schema.testing import make_label

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def policy() -> ProxyPolicy:
    """저장소 정책의 proxy 절."""
    return load_policy(ROOT).proxy


def _model(
    label_id: str, payload: dict[str, Any], times: Sequence[int], version: str
) -> LabelRecord:
    """모델 출처 라벨 (스트림 bodycam)."""
    return make_label(
        payload,
        label_id=label_id,
        t_start_ms=min(times),
        t_end_ms=max(times),
        stream_id="bodycam",
        provenance=Provenance(source=Source.MODEL, model_version=version),
        confidence=0.5,
    )


def _box(label_id: str, cls: str, times: Sequence[int], xywh: tuple[float, ...]) -> LabelRecord:
    """박스 트랙 (모든 키프레임 같은 박스)."""
    x, y, w, h = xywh
    kfs = [{"t_ms": t, "x": x, "y": y, "w": w, "h": h} for t in times]
    payload = {"kind": "box_track", "entity_id": label_id, "class_id": cls, "keyframes": kfs}
    return _model(label_id, payload, times, "owlv2-abc+p1")


def _hand(hand: str, times: Sequence[int], wrist_x: float, tag: str = "") -> LabelRecord:
    """hand21 트랙: 손목 (wrist_x, 400), 중지 뿌리는 손목 위 50 px (손 크기 50)."""
    points = [
        {"x": wrist_x, "y": 400.0 - (50.0 if j == 9 else 0.0), "visibility": 2} for j in range(21)
    ]
    payload = {
        "kind": "keypoint_track",
        "entity_id": f"{hand}_hand",
        "skeleton": "hand21",
        "hand": hand,
        "keyframes": [{"t_ms": t, "points": points} for t in times],
    }
    return _model(f"hands-{hand}{tag}", payload, times, "mediapipe-hand_landmarker-x+p1")


def test_runs_split_on_gaps_relative_to_median_step() -> None:
    """정답: 100 ms 간격 시각에 600 ms 끊김 두 번 → 구간 3개 (기준 100 x 2.5 = 250 ms)."""
    times = [0, 100, 200, 800, 900, 1500]
    assert runs(times, 2.5) == [(0, 200), (800, 900), (1500, 1500)]
    assert runs([], 2.5) == []


def test_flicker_and_short_tracks(policy: ProxyPolicy) -> None:
    """깜박이는 박스 묶음의 구간 수·짧은 트랙 비율·프레임당 수가 정답과 같다.

    rag: 0~2000 ms(100 ms 간격) 연속 트랙 하나 + 3000 ms·4000 ms 키프레임 하나짜리 트랙 둘
    → 트랙 3, 짧은(500 ms 미만) 2, 존재 구간 3, 짧은 구간 2/3. 3000 ms에 다른 rag가 하나 더 있어
    그 시각 탐지 수 2.
    """
    long = _box("rag-1", "rag", range(0, 2001, 100), (0, 0, 10, 10))
    short_a = _box("rag-2", "rag", [3000], (50, 50, 10, 10))
    short_b = _box("rag-3", "rag", [4000], (50, 50, 10, 10))
    dup = _box("rag-4", "rag", [3000], (90, 90, 10, 10))
    report = proxy_report([long, short_a, short_b, dup], policy, duration_ms=60_000)
    g = report["groups"]["box:owlv2:rag"]
    assert g["track"]["tracks"] == 4
    assert g["track"]["short_frac"] == 0.75
    assert g["track"]["tracks_per_min"] == 4.0
    assert g["presence"]["runs"] == 3
    assert g["presence"]["short_run_frac"] == round(2 / 3, 4)
    assert g["presence"]["per_frame"]["max"] == 2.0


def test_handedness_screen_side(policy: ProxyPolicy) -> None:
    """두 손 프레임 10개 중 8개는 왼손이 화면 왼쪽(x 100 < 500), 2개는 반대 → 0.8."""
    times = list(range(0, 330, 33))
    left_ok = _hand("left", times[:8], 100.0)
    right_ok = _hand("right", times[:8], 500.0)
    left_bad = _hand("left", times[8:], 600.0, "-b")
    right_bad = _hand("right", times[8:], 200.0, "-b")
    report = proxy_report([left_ok, right_ok, left_bad, right_bad], policy, duration_ms=330)
    assert report["hands_2d"]["two_hand_frames"] == 10
    assert report["hands_2d"]["left_on_screen_left_frac"] == 0.8
    assert report["hands_2d"]["left_on_screen_left_frac_observed"] == 0.8
    # 손목이 움직이지 않는 구간은 떨림 0
    assert report["hands_2d"]["left"]["wrist_jitter"]["p50"] == 0.0


def _coco(times: Sequence[int], left_wrist: tuple[float, float], tag: str = "1") -> LabelRecord:
    """COCO 17점 트랙: 머리 다섯 점은 (300, 100), 왼 손목은 주어진 위치, 나머지는 안 보임."""
    pts: list[dict[str, Any]] = []
    for j in range(17):
        if j < 5:
            pts.append({"x": 300.0, "y": 100.0, "visibility": 2})
        elif j == 9:
            pts.append({"x": left_wrist[0], "y": left_wrist[1], "visibility": 2})
        else:
            pts.append({"x": 0.0, "y": 0.0, "visibility": 0})
    payload = {
        "kind": "keypoint_track",
        "entity_id": "person_1",
        "skeleton": "coco17",
        "keyframes": [{"t_ms": t, "points": pts} for t in times],
    }
    return _model(f"body-{tag}", payload, times, "rtmpose-m+p1")


def test_wrist_agreement(policy: ProxyPolicy) -> None:
    """손 크기 50 px, 일치 기준 1.5배 = 75 px. 0~2번 프레임은 COCO 왼 손목이 30 px(일치), 3번은
    200 px(불일치) → 일치 3/4, 일치한 쌍은 모두 좌우 같음."""
    hand = _hand("left", [0, 33, 66, 99], 100.0)
    near = _coco([0, 33, 66], (130.0, 400.0))
    far = _coco([99], (300.0, 400.0), "2")
    wrist = proxy_report([hand, near, far], policy)["agreement"]["wrist"]
    assert wrist == {"pairs": 4, "agree_frac": 0.75, "side_match_frac": 1.0}


def test_face_support(policy: ProxyPolicy) -> None:
    """얼굴 키프레임 2개: 머리 관절(300, 100)을 덮는 것 하나, 싱크 위(800, 800) 하나.

    COCO 점 외접 박스(머리~손목)가 사람 박스다 → 사람 근처 1/2, 머리 근처 1/2.
    """
    body = _coco([0], (130.0, 400.0))
    kfs = [{"t_ms": 0, "x": 280.0, "y": 80.0, "w": 40.0, "h": 40.0}]
    sink = [{"t_ms": 0, "x": 780.0, "y": 780.0, "w": 40.0, "h": 40.0}]
    face = _model(
        "face-1", {"kind": "blur_track", "target": "face", "keyframes": kfs}, [0], "iou-1"
    )
    fp = _model("face-2", {"kind": "blur_track", "target": "face", "keyframes": sink}, [0], "iou-1")
    report = proxy_report([body, face, fp], policy)["agreement"]["face"]
    assert report == {"face_keyframes": 2, "near_person_frac": 0.5, "near_head_frac": 0.5}


def _traj(part: str, samples: Sequence[tuple[int, float, float, float]]) -> LabelRecord:
    """오른손 카메라 좌표 3D 궤적."""
    payload = {
        "kind": "trajectory3d",
        "entity_id": "right_hand",
        "part": part,
        "frame": "camera",
        "source_3d": "mono_depth",
        "samples": [{"t_ms": t, "x": x, "y": y, "z": z} for t, x, y, z in samples],
    }
    return _model(f"3d-{part}", payload, [s[0] for s in samples], "depth3d-x+p1")


def test_hand_3d_physics(policy: ProxyPolicy) -> None:
    """손목 z: 0.4, 0.4, 1.2(점프, 팔 길이 밖), 0.4 m (200 ms 간격). 중지 끝은 손목에서 늘 0.19 m.

    정답: 범위 밖 1/4, 속도 0, 4, 4, 0... 0.8 m / 0.2 s = 4 m/s > 3 → 속도 초과 2/3, 손끝 길이
    변동계수 0, 범위 밖 0, 손끝 표본 4/4.
    """
    zs = [0.4, 0.4, 1.2, 0.4]
    wrist = _traj("wrist", [(i * 200, 0.0, 0.0, z) for i, z in enumerate(zs)])
    tip = _traj("middle_tip", [(i * 200, 0.0, -0.19, z) for i, z in enumerate(zs)])
    h = proxy_report([wrist, tip], policy)["hand_3d"]["right_hand"]
    assert h["samples"] == 4
    assert h["out_of_reach_frac"] == 0.25
    assert h["over_speed_frac"] == round(2 / 3, 4)
    assert h["length_cv"]["middle_tip"] == 0.0
    assert h["length_out_of_range_frac"] == 0.0
    assert h["tip_coverage_frac"] == 1.0
    assert "right_hand 3D" in markdown(proxy_report([wrist, tip], policy), "t")


def test_wearer_arm_and_wearer_like_people(policy: ProxyPolicy) -> None:
    """ADR 0058. 오른팔 arm3 4 키프레임: 팔꿈치 보임 3, 사슬 일치(손목 신뢰도 있음) 4, 일치 정도
    0.8. 팔꿈치 3D는 손목에서 0.25, 0.25, 0.25, 0.5 m(범위 [0.18, 0.36] 밖 1/4).

    사람 박스 둘: 착용자 오른손목(x 300, y 400)을 담는 박스(착용자처럼 보임)와 손목 밖 박스.
    """
    times = [0, 200, 400, 600]
    frames = [
        {"t_ms": t, "points": [
            {"x": 300.0, "y": 400.0, "visibility": 2, "confidence": 0.8},
            {"x": 250.0, "y": 470.0, "visibility": 0 if t == 600 else 2, "confidence": 0.7},
            {"x": 300.0, "y": 400.0, "visibility": 0},
        ]}
        for t in times
    ]  # fmt: skip
    arm = _model(
        "body-arm-right",
        {"kind": "keypoint_track", "entity_id": "right_arm", "skeleton": "arm3", "hand": "right",
         "keyframes": frames},
        times, "rtmpose-x+p1",
    )  # fmt: skip
    wrist = _traj("wrist", [(t, 0.0, 0.0, 0.4) for t in times])
    zs = [0.25, 0.25, 0.25, 0.5]
    elbow = make_label(
        {"kind": "trajectory3d", "entity_id": "right_arm", "part": "elbow", "frame": "camera",
         "source_3d": "mono_depth",
         "samples": [{"t_ms": t, "x": 0.0, "y": z, "z": 0.4}
                     for t, z in zip(times, zs, strict=True)]},
        label_id="3d-elbow", t_start_ms=0, t_end_ms=600, stream_id="bodycam",
        provenance=Provenance(source=Source.MODEL, model_version="depth3d-x"), confidence=0.5,
    )  # fmt: skip
    hand = _hand("right", times, 300.0)
    people = [
        _box("p-wearer", "person", times, (200.0, 300.0, 300.0, 180.0)),
        _box("p-other", "person", times, (10.0, 10.0, 50.0, 100.0)),
    ]
    report = proxy_report([arm, wrist, elbow, hand, *people], policy)["wearer"]
    a = report["arm"]["right"]
    assert (a["keyframes"], a["elbow_visible_frac"], a["chain_frac"]) == (4, 0.75, 1.0)
    assert a["wrist_agreement"]["p50"] == 0.8
    assert a["forearm"]["out_of_range_frac"] == 0.25 and a["forearm"]["length_cv"] is not None
    assert report["people"] == {"person_tracks": 2, "wearer_like_tracks": 1}
    assert "right 팔" in markdown(proxy_report([arm, wrist, elbow, hand, *people], policy), "t")


def test_motion_skips_far_neighbours() -> None:
    """이웃 간격이 max_dt_ms를 넘으면 속도·가속도를 세지 않는다 (정답: 1 m / 0.1 s = 10 m/s)."""
    s = [
        Trajectory3DSample(t_ms=0, x=0, y=0, z=0.5),
        Trajectory3DSample(t_ms=100, x=1, y=0, z=0.5),
        Trajectory3DSample(t_ms=5000, x=0, y=0, z=0.5),
    ]
    speeds, accels = motion(s, 400)
    assert speeds == [pytest.approx(10.0)]
    assert accels == []


def test_policy_ranges_are_checked() -> None:
    """범위 값의 하한이 상한보다 크면 정책을 거부한다."""
    data = load_policy(ROOT).proxy.model_dump()
    data["reach_m"] = (1.0, 0.5)
    with pytest.raises(ValueError, match="reach_m"):
        ProxyPolicy.model_validate(data)
