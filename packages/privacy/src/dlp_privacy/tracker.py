"""탐지를 트랙으로 묶고, 끊김을 보간하고, 소실 후에도 블러를 유지한다.

1. 같은 프레임·같은 대상의 여러 탐지기 결과가 겹치면 하나로 합친다 (박스는 합집합, 재현율 우선).
2. 대상별로 IoU 탐욕 매칭으로 트랙을 잇는다. max_gap_ms보다 오래 끊기면 새 트랙이다.
3. 트랙 안의 관측 사이 프레임은 선형 보간한다 (보간 구간은 검수 우선 구간이 된다).
4. 마지막 관측 뒤 hold_ms 동안 마지막 박스를 유지한다 (가려지거나 놓쳐도 노출되지 않게).
   처리가 오프라인이므로 첫 관측 앞으로도 hold_ms만큼 첫 박스를 미리 둔다. 탐지기가 대상이
   처음 나타난 몇 프레임을 놓쳐도 노출되지 않는다.
5. 박스에 대상별 여유(margin)를 더하고 화면 안으로 자른다.
트랙 하나가 blur_track 라벨 하나가 된다. 키프레임은 트랙 구간의 모든 프레임에 둔다.

WP5. `pipeline.detect_video`가 쓴다. 이 모듈의 동작을 바꾸면 `pipeline.TRACKER_VERSION`을 올려야
다시 탐지된다. 정책 값: privacy.yaml `tracker`(iou_match, max_gap_ms), `targets.*.margin`,
defaults.yaml `privacy.blur_hold_ms`.

시간: 모든 시각은 그 스트림의 프레임 PTS 시각(정수 ms)이다. 보간은 여기서는 시각 비율로 한다
(모든 프레임에 키프레임을 두므로 렌더·CVAT의 프레임 번호 보간과 결과가 같다).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dlp_privacy.detection import Detection
from dlp_privacy.geometry import Box


@dataclass
class Observation:
    """한 프레임에서 트랙에 붙은 (합친) 관측. `fuse`가 박스·탐지기를 고치므로 가변이다."""

    # 합친 박스 (겹친 탐지들의 합집합 외접 박스)
    box: Box
    # 가장 높은 탐지 점수 (fuse가 점수 내림차순으로 처리하므로 첫 탐지의 점수)
    score: float
    # 이 관측에 참여한 탐지기 이름들 (disagreement 판정용)
    detectors: frozenset[str]


@dataclass
class Track:
    """대상 하나의 트랙. obs: 관측한 프레임 시각(ms) → 관측."""

    target: str
    obs: dict[int, Observation] = field(default_factory=dict[int, Observation])

    @property
    def last_t(self) -> int:
        """마지막 관측 시각 (ms)."""
        return max(self.obs)

    def last_box(self) -> Box:
        """마지막 관측 박스 (다음 프레임 매칭 기준)."""
        return self.obs[self.last_t].box


@dataclass(frozen=True)
class TrackFrame:
    """트랙의 프레임별 결과 (여유를 더하고 화면으로 자른 박스)."""

    # 프레임 시각 (스트림 PTS ms)
    t_ms: int
    box: Box | None  # None이면 화면 밖
    # observed: 관측, interpolated: 관측 사이 보간, held: 첫 관측 전·마지막 관측 뒤 유지,
    # outside: 박스 없음 (유지 시간이 끝났거나 박스가 화면 밖으로 잘렸다)
    kind: str  # observed | interpolated | held | outside
    # 관측 점수 (관측 프레임만, 나머지는 None)
    score: float | None
    # 관측에 참여한 탐지기 (관측 프레임만, 나머지는 빈 집합)
    detectors: frozenset[str]


def fuse(detections: list[Detection], iou: float) -> list[tuple[str, Observation]]:
    """같은 대상끼리 겹치는 탐지를 합친다.

    점수 높은 탐지부터 보며, 이미 만든 관측 중 같은 대상이고 IoU가 iou 이상인 첫 관측에 합친다
    (박스는 두 박스를 모두 덮는 외접 박스로 키우고 탐지기 이름을 더한다). 합칠 곳이 없으면 새 관측.
    재현율 우선이라 박스를 줄이지 않는다.

    Args:
        detections: 한 프레임의 탐지.
        iou: 합칠 최소 IoU (정책 `tracker.iou_match`를 그대로 쓴다).

    Returns:
        (대상, 관측) 목록.
    """
    out: list[tuple[str, Observation]] = []
    for d in sorted(detections, key=lambda d: -d.score):
        for target, o in out:
            # 비교 기준은 지금까지 합쳐 커진 박스다
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
    """프레임별 탐지를 트랙으로 잇는다 (대상별 IoU 탐욕 매칭).

    프레임마다:
    1. 마지막 관측이 max_gap_ms 이내인 트랙만 열린 트랙으로 본다.
    2. `fuse`로 합친 관측을 점수 높은 순으로, 같은 대상이고 아직 이번 프레임에 쓰이지 않은
       열린 트랙 중 마지막 박스와 IoU가 가장 큰(iou_match 이상) 트랙에 붙인다.
    3. 붙일 트랙이 없으면 새 트랙을 연다.

    Args:
        frames: (프레임 시각 ms, 탐지) 목록, 시각 증가 순.
        iou_match: privacy.yaml `tracker.iou_match`.
        max_gap_ms: privacy.yaml `tracker.max_gap_ms`. 이보다 오래 끊기면 새 트랙이 된다.

    Returns:
        트랙 목록 (만든 순서).
    """
    tracks: list[Track] = []
    for t_ms, detections in frames:
        open_tracks = [tr for tr in tracks if t_ms - tr.last_t <= max_gap_ms]
        used: set[int] = set()  # 이번 프레임에 이미 관측을 붙인 open_tracks 위치
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
    """첫 관측 hold_ms 전부터 마지막 관측 hold_ms 뒤 다음 프레임(outside)까지의 프레임별 박스.

    Args:
        track: 트랙 (관측이 하나 이상).
        frame_times: 영상 전체 프레임 시각 (증가 순).
        hold_ms: defaults.yaml `privacy.blur_hold_ms`. 첫 관측 전·마지막 관측 뒤 박스 유지 시간.
        margin: 대상 여유 (`Box.padded`).
        width, height: 영상 크기 (박스를 화면 안으로 자른다).

    Returns:
        구간 안 프레임마다 `TrackFrame`. 마지막 관측 + hold_ms 뒤 첫 프레임 하나를 outside로 넣고
        끝낸다 (영상 끝에 닿으면 outside 없이 끝난다). 여유를 더한 박스가 화면 밖이면 그 프레임도
        outside다.
    """
    times = sorted(track.obs)
    first, last = times[0], times[-1]
    out: list[TrackFrame] = []
    for t in frame_times:
        if t < first - hold_ms:
            continue
        if t > last + hold_ms:
            # 유지 시간 끝: 렌더·CVAT가 여기서 블러를 끄도록 outside 키프레임 하나를 둔다
            out.append(TrackFrame(t, None, "outside", None, frozenset()))
            break
        if t in track.obs:
            o = track.obs[t]
            box, kind, score, dets = o.box, "observed", o.score, o.detectors
        elif t > last:
            # 마지막 관측 뒤 유지
            box, kind, score, dets = track.obs[last].box, "held", None, frozenset[str]()
        elif t < first:
            # 첫 관측 앞 미리 가림 (오프라인 처리라 가능)
            box, kind, score, dets = track.obs[first].box, "held", None, frozenset[str]()
        else:
            # 관측 사이: 앞뒤 관측 박스를 시각 비율로 선형 보간
            before = max(x for x in times if x < t)
            after = min(x for x in times if x > t)
            s = (t - before) / (after - before)
            box = track.obs[before].box.lerp(track.obs[after].box, s)
            kind, score, dets = "interpolated", None, frozenset[str]()
        clipped = box.padded(margin).clipped(width, height)
        out.append(TrackFrame(t, clipped, kind if clipped else "outside", score, dets))
    return out
