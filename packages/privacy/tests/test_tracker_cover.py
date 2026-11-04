"""트래커 덮기 불변식: 블러 박스가 매 프레임의 모든 탐지(관측·간격 캐시)를 덮는다 (F-PMS-49).

키프레임이 트랙 구간의 모든 프레임에 있으므로 `track_frames` 박스가 곧 렌더 박스다.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import pytest

from dlp_privacy.detection import Detection
from dlp_privacy.geometry import Box
from dlp_privacy.tracker import build_tracks, track_frames

W, H = 640, 480
IOU_MATCH, MAX_GAP_MS, HOLD_MS, MARGIN = 0.2, 600, 200, 0.05
TIMES = [round(i * 1000 / 30) for i in range(90)]


def _uncovered(frames: list[tuple[int, list[Detection]]]) -> list[tuple[int, Detection]]:
    """어떤 트랙 박스에도 덮이지 않은 (프레임 시각, 탐지) 목록 (화면 밖으로 잘린 탐지는 뺀다)."""
    tracks = build_tracks(frames, iou_match=IOU_MATCH, max_gap_ms=MAX_GAP_MS)
    times = [t for t, _ in frames]
    boxes: dict[int, list[Box]] = {t: [] for t in times}
    for tr in tracks:
        for f in track_frames(tr, times, hold_ms=HOLD_MS, margin=MARGIN, width=W, height=H):
            if f.box is not None:
                boxes[f.t_ms].append(f.box)
    bad: list[tuple[int, Detection]] = []
    for t, dets in frames:
        for d in dets:
            c = d.box.clipped(W, H)
            if c is None:
                continue
            inner = Box(c.x + 1e-6, c.y + 1e-6, c.w - 2e-6, c.h - 2e-6)
            if not any(b.contains(inner) for b in boxes[t]):
                bad.append((t, d))
    return bad


@dataclass(frozen=True)
class _Obj:
    """등속으로 움직이는 대상 (프레임 i의 박스는 시작 박스에 속도의 i배를 더한다)."""

    target: str
    x: float
    y: float
    w: float
    h: float
    vx: float
    vy: float
    present: list[bool]

    def box(self, i: int, dx: float = 0.0, sw: float = 1.0, sh: float = 1.0) -> Box:
        return Box(self.x + self.vx * i + dx, self.y + self.vy * i, self.w * sw, self.h * sh)


def _two_detector_frames(seed: int, stride_ms: int) -> list[tuple[int, list[Detection]]]:
    """대상 1~5개를 간격 추론 탐지기(ov, 사이 프레임은 캐시)와 매 프레임 탐지기(qr)로 본다."""
    rnd = random.Random(seed)
    objs = [
        _Obj(
            rnd.choice(["document", "document", "screen"]),
            rnd.uniform(-50, 600),
            rnd.uniform(-50, 440),
            rnd.uniform(5, 150),
            rnd.uniform(5, 150),
            rnd.uniform(-40, 40),
            rnd.uniform(-15, 15),
            [rnd.random() < 0.85 for _ in TIMES],
        )
        for _ in range(rnd.randint(1, 5))
    ]
    frames: list[tuple[int, list[Detection]]] = []
    last: int | None = None
    cache: list[Detection] = []
    for i, t in enumerate(TIMES):
        if last is None or t - last >= stride_ms:
            cache = [
                Detection(o.target, o.box(i), rnd.uniform(0.3, 1), "ov")
                for o in objs
                if o.present[i] and rnd.random() < 0.9
            ]
            last = t
        dets = [Detection(d.target, d.box, d.score, d.detector, cached=t != last) for d in cache]
        dets += [
            Detection(
                o.target,
                o.box(i, rnd.uniform(-20, 20), rnd.uniform(0.2, 1.2), 0.5),
                rnd.uniform(0.3, 1),
                "qr",
            )
            for o in objs
            if o.present[i] and rnd.random() < 0.5
        ]
        frames.append((t, dets))
    return frames


def test_cached_box_between_observations_stays_covered_f_pms_49() -> None:
    """관측 사이에 낀 캐시 박스가 뒤 관측이 붙어도 그 프레임 블러에 남는다 (F-PMS-49)."""
    x = Box(0, 0, 60, 60)
    c = Box(0, 0, 100, 100)
    frames = [
        (0, [Detection("shipping_label", x, 0.9, "codes")]),
        (33, [Detection("shipping_label", c, 0.9, "open_vocab", cached=True)]),
        (67, [Detection("shipping_label", x, 0.9, "codes")]),
    ]
    assert _uncovered(frames) == []


@pytest.mark.parametrize("stride_ms", [0, 500])
def test_blur_covers_every_detection_with_cached_second_detector_f_pms_49(stride_ms: int) -> None:
    """탐지기 2개(하나는 간격 캐시)의 무작위 장면에서 덮이지 않은 탐지가 없다 (F-PMS-49).

    시드 34(stride 500)는 고치기 전 캐시 박스가 노출되던 장면이다.
    """
    failing = {s: bad for s in range(50) if (bad := _uncovered(_two_detector_frames(s, stride_ms)))}
    assert failing == {}
