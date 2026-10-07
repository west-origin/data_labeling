"""탐지를 트랙으로 묶고, 끊김을 보간하고, 소실 후에도 블러를 유지한다.

1. 같은 프레임·같은 대상의 여러 탐지기 결과가 겹치면 하나로 합친다 (박스는 합집합, 재현율 우선).
2. 대상별로 IoU 탐욕 매칭으로 트랙을 잇는다. max_gap_ms보다 오래 끊기면 새 트랙이다.
3. 트랙 안의 관측 사이 프레임은 선형 보간한다 (보간 구간은 검수 우선 구간이 된다).
4. 마지막 관측 뒤 hold_ms 동안 마지막 박스를 유지한다 (가려지거나 놓쳐도 노출되지 않게).
   처리가 오프라인이므로 첫 관측 앞으로도 hold_ms만큼 첫 박스를 미리 둔다. 탐지기가 대상이
   처음 나타난 몇 프레임을 놓쳐도 노출되지 않는다.
5. 박스에 대상별 여유(margin)를 더하고 화면 안으로 자른다.
트랙 하나가 blur_track 라벨 하나가 된다. 키프레임은 트랙 구간의 모든 프레임에 둔다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dlp_privacy.detection import Detection
from dlp_privacy.geometry import Box


@dataclass
class Observation:
    box: Box
    score: float
    detectors: frozenset[str]


@dataclass
class Track:
    target: str
    obs: dict[int, Observation] = field(default_factory=dict[int, Observation])

    @property
    def last_t(self) -> int:
        return max(self.obs)

    def last_box(self) -> Box:
        return self.obs[self.last_t].box


@dataclass(frozen=True)
class TrackFrame:
    t_ms: int
    box: Box | None  # None이면 화면 밖
    kind: str  # observed | interpolated | held | outside
    score: float | None
    detectors: frozenset[str]


def fuse(detections: list[Detection], iou: float) -> list[tuple[str, Observation]]:
    """같은 대상끼리 겹치는 탐지를 합친다."""
    out: list[tuple[str, Observation]] = []
    for d in sorted(detections, key=lambda d: -d.score):
        for target, o in out:
            if target == d.target and o.box.iou(d.box) >= iou:
                x1, y1 = min(o.box.x, d.box.x), min(o.box.y, d.box.y)
                x2 = max(o.box.x + o.box.w, d.box.x + d.box.w)
                y2 = max(o.box.y + o.box.h, d.box.y + d.box.h)
                o.box = Box(x1, y1, x2 - x1, y2 - y1)
                o.detectors = o.detectors | {d.detector}
                break
        else:
            out.append((d.target, Observation(d.box, d.score, frozenset({d.detector}))))
    return out


def build_tracks(
    frames: list[tuple[int, list[Detection]]], *, iou_match: float, max_gap_ms: float
) -> list[Track]:
    tracks: list[Track] = []
    for t_ms, detections in frames:
        open_tracks = [tr for tr in tracks if t_ms - tr.last_t <= max_gap_ms]
        used: set[int] = set()
        for target, o in sorted(fuse(detections, iou_match), key=lambda x: -x[1].score):
            best, best_iou = None, iou_match
            for i, tr in enumerate(open_tracks):
                if i in used or tr.target != target:
                    continue
                v = tr.last_box().iou(o.box)
                if v >= best_iou:
                    best, best_iou = i, v
            if best is None:
                tracks.append(Track(target, {t_ms: o}))
            else:
                used.add(best)
                open_tracks[best].obs[t_ms] = o
    return tracks


def track_frames(
    track: Track,
    frame_times: list[int],
    *,
    hold_ms: float,
    margin: float,
    width: int,
    height: int,
) -> list[TrackFrame]:
    """첫 관측 hold_ms 전부터 마지막 관측 hold_ms 뒤 다음 프레임(outside)까지의 프레임별 박스."""
    times = sorted(track.obs)
    first, last = times[0], times[-1]
    out: list[TrackFrame] = []
    for t in frame_times:
        if t < first - hold_ms:
            continue
        if t > last + hold_ms:
            out.append(TrackFrame(t, None, "outside", None, frozenset()))
            break
        if t in track.obs:
            o = track.obs[t]
            box, kind, score, dets = o.box, "observed", o.score, o.detectors
        elif t > last:
            box, kind, score, dets = track.obs[last].box, "held", None, frozenset[str]()
        elif t < first:
            box, kind, score, dets = track.obs[first].box, "held", None, frozenset[str]()
        else:
            before = max(x for x in times if x < t)
            after = min(x for x in times if x > t)
            s = (t - before) / (after - before)
            box = track.obs[before].box.lerp(track.obs[after].box, s)
            kind, score, dets = "interpolated", None, frozenset[str]()
        clipped = box.padded(margin).clipped(width, height)
        out.append(TrackFrame(t, clipped, kind if clipped else "outside", score, dets))
    return out
