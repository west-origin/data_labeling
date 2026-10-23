"""오탐 의심 힌트(`dlp_privacy.hints`, `dlp_review.object_hints`)와 TTA(`geometry.tta_tracks`).

정답을 아는 합성 라벨: 머리 관절 위의 얼굴(진짜), 손 위의 얼굴(오탐), 0.1초 깜박임, 착용자 손목을
담은 사람 박스(착용자 몸)와 머리가 보이는 사람 박스(다른 사람). TTA는 흰 사각형이 있는 합성
영상과 흰 화소 덩어리를 찾는 결정적 탐지기로 본다: 사각형 위 박스는 모든 변환에서 다시 찾고(1.0),
빈 곳 박스는 못 찾는다(0.0). 힌트는 블러를 지우거나 바꾸지 않는다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from dlp_media.geometry import Box, TtaSpec, tta_tracks
from dlp_media.storage import LocalStore
from dlp_privacy.hints import hint_segments, privacy_hints, read_hint_segments, write_hint_segments
from dlp_privacy.policy import load_policy
from dlp_review.object_hints import object_hints, read_object_hints, write_object_hints
from dlp_review.ops.policy import load_policy as load_review_policy
from dlp_schema.labels import LabelRecord, Provenance, Source
from dlp_schema.testing import make_label

ROOT = Path(__file__).resolve().parents[1]
TIMES = list(range(0, 1000, 100))


def _model(label_id: str, payload: dict[str, Any], times: list[int]) -> LabelRecord:
    """모델 출처 라벨 (bodycam)."""
    return make_label(
        payload, label_id=label_id, t_start_ms=times[0], t_end_ms=times[-1], stream_id="bodycam",
        provenance=Provenance(source=Source.MODEL, model_version="m-1"), confidence=0.5,
    )  # fmt: skip


def _box_track(label_id: str, kind: str, key: str, box: Box, times: list[int]) -> LabelRecord:
    """같은 박스의 블러·박스 트랙."""
    x, y, w, h = box
    kfs = [{"t_ms": t, "x": x, "y": y, "w": w, "h": h} for t in times]
    if kind == "blur_track":
        return _model(label_id, {"kind": kind, "target": key, "keyframes": kfs}, times)
    payload = {"kind": kind, "entity_id": label_id, "class_id": key, "keyframes": kfs}
    return _model(label_id, payload, times)


def _coco(head: tuple[float, float]) -> LabelRecord:
    """머리 다섯 점이 head에 보이는 COCO 트랙."""
    pts = [
        {"x": head[0], "y": head[1], "visibility": 2 if j < 5 else 0} for j in range(17)
    ]  # fmt: skip
    payload = {
        "kind": "keypoint_track", "entity_id": "person_1", "skeleton": "coco17",
        "keyframes": [{"t_ms": t, "points": pts} for t in TIMES],
    }  # fmt: skip
    return _model("body-1", payload, TIMES)


def _hand(center: tuple[float, float]) -> LabelRecord:
    """손 관절이 center 둘레 ±30 px에 퍼진 오른손 트랙 (손목 = center)."""
    cx, cy = center
    pts = [
        {"x": cx + (j % 5 - 2) * 15, "y": cy - (j // 5) * 10, "visibility": 2} for j in range(21)
    ]
    pts[0] = {"x": cx, "y": cy, "visibility": 2}
    payload = {
        "kind": "keypoint_track", "entity_id": "right_hand", "skeleton": "hand21", "hand": "right",
        "keyframes": [{"t_ms": t, "points": pts} for t in TIMES],
    }  # fmt: skip
    return _model("hands-right", payload, TIMES)


def test_privacy_hints_flag_without_touching_blur(tmp_path: Path) -> None:
    """정답: 머리 위 얼굴은 힌트 없음, 손 위 얼굴은 no_head·on_hand, 깜박임은 short_flicker,
    TTA 지지 0.2인 얼굴은 tta_unstable. 힌트 구간은 누락 위험 구간 뒤 순위이고 왕복 저장된다."""
    policy = load_policy(ROOT)
    real = _box_track("face-real", "blur_track", "face", (280, 80, 40, 40), TIMES)
    on_hand = _box_track("face-hand", "blur_track", "face", (580, 580, 40, 40), TIMES)
    blip = _box_track("face-blip", "blur_track", "face", (280, 80, 40, 40), [500, 600])
    labels = [real, on_hand, blip, _coco((300, 100)), _hand((600, 600))]
    tta = {"face-real": 1.0, "face-hand": 0.2}
    hints = privacy_hints(labels, "bodycam", policy.hints, tta)
    got = {(h.label_id, h.reason) for h in hints}
    assert got == {
        ("face-hand", "no_head"), ("face-hand", "on_hand"), ("face-hand", "tta_unstable"),
        ("face-blip", "short_flicker"),
    }  # fmt: skip
    segs = hint_segments(hints, len(policy.review_priority))
    assert {s.reason for s in segs} == {"suspected_false_positive"}
    assert all(s.priority == len(policy.review_priority) for s in segs)
    store = LocalStore(tmp_path / "raw", "dlp-raw")
    write_hint_segments(store, "s001", "bodycam", segs, tmp_path)
    assert read_hint_segments(store, "s001", "bodycam", tmp_path) == segs
    # 블러 라벨은 그대로다 (힌트는 따로 저장할 뿐)
    assert labels[1] == on_hand
    # COCO 트랙이 없으면 머리 규칙은 판단하지 않는다
    no_body = privacy_hints([on_hand, _hand((600, 600))], "bodycam", policy.hints)
    assert {h.reason for h in no_body} == {"on_hand"}


def test_object_hints_wearer_body(tmp_path: Path) -> None:
    """정답: 착용자 손목을 담고 머리가 없는 사람 박스 → wearer_body, 머리가 보이는 박스 → 없음."""
    policy = load_review_policy(ROOT).object_hints
    assert policy is not None
    wearer = _box_track("person-wearer", "box_track", "person", (500, 500, 300, 300), TIMES)
    other = _box_track("person-other", "box_track", "person", (250, 50, 120, 300), TIMES)
    labels = [wearer, other, _coco((300, 100)), _hand((600, 600))]
    hints = object_hints(labels, "bodycam", policy, {"person-other": 0.1})
    assert {(h.label_id, h.reason) for h in hints} == {
        ("person-wearer", "wearer_body"), ("person-other", "tta_unstable"),
    }  # fmt: skip
    store = LocalStore(tmp_path / "lab", "dlp-labeling")
    write_object_hints(store, "s001", "bodycam", hints, tmp_path)
    assert read_object_hints(store, "s001", "bodycam", tmp_path) == hints


def _video(path: Path) -> None:
    """흰 사각형(100, 60, 40, 30)이 있는 320x240 10 fps 1초 영상."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), 10, (320, 240))
    for _ in range(10):
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        img[60:90, 100:140] = 255
        writer.write(img)
    writer.release()


def _white(img: np.ndarray) -> list[tuple[str, Box]]:
    """결정적 탐지기: 밝은 화소 덩어리의 외접 박스 (키 "face")."""
    mask = (img.mean(axis=2) > 128).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    return [("face", (float(s[0]), float(s[1]), float(s[2]), float(s[3]))) for s in stats[1:n]]


def test_tta_tracks_on_synthetic_video(tmp_path: Path) -> None:
    """정답: 사각형 위 박스는 뒤집기·배율 화면에서도 다시 찾고(1.0), 빈 곳 박스는 못 찾는다(0.0)."""
    video = tmp_path / "v.mp4"
    _video(video)
    spec = TtaSpec(transforms=("hflip", "scale:0.75"), iou=0.5, max_tracks=10, frames_per_track=2)
    tracks = [
        ("real", "face", [(0, (100.0, 60.0, 40.0, 30.0)), (500, (100.0, 60.0, 40.0, 30.0))]),
        ("ghost", "face", [(300, (220.0, 150.0, 40.0, 30.0))]),
    ]
    support = tta_tracks(video, tracks, _white, spec)
    assert support["real"] == 1.0
    assert support["ghost"] == 0.0
