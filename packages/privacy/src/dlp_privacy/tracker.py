"""탐지를 트랙으로 묶고, 끊김을 보간하고, 소실 후에도 블러를 유지한다.

1. 같은 프레임·같은 대상의 여러 탐지기 결과가 겹치면 하나로 합친다 (박스는 합집합, 재현율 우선).
   간격 캐시(`Detection.cached`)만으로 된 묶음은 관측이 아니다 (F-PMS-45).
2. 대상별로 IoU 탐욕 매칭으로 트랙을 잇는다. max_gap_ms보다 오래 끊기면 새 트랙이다. 캐시만인
   묶음은 같은 대상 트랙의 마지막 박스와 맞으면 그 트랙의 유지 근거(`Track.held_until`)가 된다.
3. 트랙 안의 관측 사이 프레임은 앞뒤 관측 박스를 모두 덮는 외접 박스로 가린다 (간격 캐시 사이에
   움직인 대상도 덮게, 재현율 우선. 보간 구간은 검수 우선 구간이 된다).
4. 마지막 관측(캐시 유지 근거가 있으면 그 마지막 시각) 뒤 hold_ms 동안 마지막 박스를 유지한다
   (가려지거나 놓쳐도 노출되지 않게).
   처리가 오프라인이므로 첫 관측 앞으로도 hold_ms만큼 첫 박스를 미리 둔다. 탐지기가 대상이
   처음 나타난 몇 프레임을 놓쳐도 노출되지 않는다.
5. 박스에 대상별 여유(margin)를 더하고 화면 안으로 자른다.
트랙 하나가 blur_track 라벨 하나가 된다. 키프레임은 트랙 구간의 모든 프레임에 둔다.

WP5. `pipeline.detect_video`가 쓴다. 이 모듈의 동작을 바꾸면 `pipeline.TRACKER_VERSION`을 올려야
다시 탐지된다. 정책 값: privacy.yaml `tracker`(iou_match, max_gap_ms), `targets.*.margin`,
defaults.yaml `privacy.blur_hold_ms`.

시간: 모든 시각은 그 스트림의 프레임 PTS 시각(정수 ms)이다. 보간 박스는 시각과 무관한 외접
박스다 (모든 프레임에 키프레임을 두므로 렌더·CVAT의 프레임 번호 보간과 결과가 같다).
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
    # 이 프레임에서 실제로 추론한 탐지가 하나라도 있는가 (False면 간격 캐시뿐, 관측이 아니다)
    fresh: bool = True


@dataclass
class Track:
    """대상 하나의 트랙. obs: 관측한 프레임 시각(ms) → 관측."""

    target: str
    obs: dict[int, Observation] = field(default_factory=dict[int, Observation])
    # 마지막 관측 뒤 간격 캐시가 이 트랙을 가리킨 마지막 시각 (다음 추론 전까지 대상이 있다고 본다)
    held_until: int | None = None
    # 그 캐시 박스들의 외접 박스
    held_box: Box | None = None

    def add(self, t_ms: int, o: Observation) -> None:
        """관측을 붙인다. 앞서 모은 캐시 유지 근거는 새 관측 앞이므로 버린다."""
        self.obs[t_ms] = o
        self.held_until, self.held_box = None, None

    def hold(self, t_ms: int, box: Box) -> None:
        """간격 캐시 묶음을 유지 근거로 더한다 (마지막 관측 뒤만)."""
        self.held_until = t_ms
        self.held_box = box if self.held_box is None else self.held_box.hull(box)

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
    재현율 우선이라 박스를 줄이지 않는다. 관측의 `fresh`는 참여한 탐지 중 캐시가 아닌 것이 있는가다.

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
                o.box = o.box.hull(d.box)
                o.detectors = o.detectors | {d.detector}
                o.fresh = o.fresh or not d.cached
                break
        else:
            obs = Observation(d.box, d.score, frozenset({d.detector}), fresh=not d.cached)
            out.append((d.target, obs))
    return out


def build_tracks(
    frames: list[tuple[int, list[Detection]]], *, iou_match: float, max_gap_ms: float
) -> list[Track]:
    """프레임별 탐지를 트랙으로 잇는다 (대상별 IoU 탐욕 매칭).

    프레임마다:
    1. 마지막 관측이 max_gap_ms 이내인 트랙만 열린 트랙으로 본다.
    2. `fuse`로 합친 관측을 점수 높은 순으로, 같은 대상이고 아직 이번 프레임에 쓰이지 않은
       열린 트랙 중 마지막 박스와 IoU가 가장 큰(iou_match 이상) 트랙에 붙인다.
    3. 붙일 트랙이 없으면, 바로 앞 프레임까지 간격 캐시로 유지되던(그래서 이번이 다시 추론한
       프레임인) 같은 대상 트랙 중 마지막 박스 중심이 가장 가까운 트랙에 잇는다. 간격 사이에 대상이
       제 크기 넘게 움직이면 IoU로는 못 잇기 때문이다. 그런 트랙도 없으면 새 트랙을 연다.
    4. 간격 캐시만인 묶음(`fresh=False`)은 관측으로 붙이지 않는다. 같은 대상 트랙 중 마지막 관측이
       그 시각 이전이고 마지막 박스와 IoU가 가장 큰(iou_match 이상) 트랙의 유지 근거로 삼는다
       (`Track.hold`). 맞는 트랙이 없으면(정상이라면 없다: 캐시는 앞 추론의 관측을 되풀이한다)
       탐지를 버리지 않도록 관측으로 새 트랙을 연다.

    Args:
        frames: (프레임 시각 ms, 탐지) 목록, 시각 증가 순.
        iou_match: privacy.yaml `tracker.iou_match`.
        max_gap_ms: privacy.yaml `tracker.max_gap_ms`. 이보다 오래 끊기면 새 트랙이 된다.

    Returns:
        트랙 목록 (만든 순서).
    """
    tracks: list[Track] = []
    t_prev: int | None = None
    for t_ms, detections in frames:
        open_tracks = [tr for tr in tracks if t_ms - tr.last_t <= max_gap_ms]
        used: set[int] = set()  # 이번 프레임에 이미 관측을 붙인 open_tracks 위치
        fused = sorted(fuse(detections, iou_match), key=lambda x: -x[1].score)
        unmatched: list[tuple[str, Observation]] = []
        for target, o in (x for x in fused if x[1].fresh):
            best = _best_match(open_tracks, target, o.box, iou_match, skip=used)
            if best is None:
                unmatched.append((target, o))
            else:
                used.add(best)
                open_tracks[best].add(t_ms, o)
        # 앞 프레임까지 캐시로 유지되던 트랙은 이번이 다시 본 프레임이다. IoU로 못 이은 새 관측을
        # 그 트랙에 잇는다 (간격 사이에 대상이 제 크기 넘게 움직였다). 이어야 사이 프레임이 앞뒤
        # 박스의 외접 박스로 덮인다 (F-PMS-45).
        awaiting = [
            tr
            for i, tr in enumerate(open_tracks)
            if i not in used and t_prev is not None and tr.held_until == t_prev
        ]
        for target, o in unmatched:
            tr = _nearest(awaiting, target, o.box)
            if tr is None:
                tracks.append(Track(target, {t_ms: o}))
            else:
                awaiting.remove(tr)
                tr.add(t_ms, o)
        for target, o in (x for x in fused if not x[1].fresh):
            earlier = [tr for tr in tracks if tr.last_t < t_ms]
            best = _best_match(earlier, target, o.box, iou_match, skip=set())
            if best is None:
                tracks.append(Track(target, {t_ms: o}))
            else:
                earlier[best].hold(t_ms, o.box)
        t_prev = t_ms
    return tracks


def _nearest(tracks: list[Track], target: str, box: Box) -> Track | None:
    """같은 대상 트랙 중 마지막 박스 중심이 box 중심에 가장 가까운 트랙 (없으면 None)."""
    same = [tr for tr in tracks if tr.target == target]
    if not same:
        return None
    cx, cy = box.x + box.w / 2, box.y + box.h / 2

    def dist(tr: Track) -> float:
        b = tr.last_box()
        return (b.x + b.w / 2 - cx) ** 2 + (b.y + b.h / 2 - cy) ** 2

    return min(same, key=dist)


def _best_match(
    tracks: list[Track], target: str, box: Box, iou_match: float, *, skip: set[int]
) -> int | None:
    """같은 대상 트랙 중 마지막 박스와 IoU가 가장 큰(iou_match 이상) 트랙의 위치 (skip 제외)."""
    best, best_iou = None, iou_match
    for i, tr in enumerate(tracks):
        if i in skip or tr.target != target:
            continue
        v = tr.last_box().iou(box)
        if v >= best_iou:
            best, best_iou = i, v
    return best


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

    마지막 관측 뒤 간격 캐시 유지 근거(`Track.held_until`)가 있으면 그 시각 + hold_ms까지 유지하고,
    유지 박스는 마지막 관측 박스와 캐시 박스의 외접 박스다. 관측 사이는 앞뒤 관측 박스의 외접
    박스다 (F-PMS-45).

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
    held_end = max(last, track.held_until or last)
    held_box = track.obs[last].box
    if track.held_box is not None:
        held_box = held_box.hull(track.held_box)
    out: list[TrackFrame] = []
    for t in frame_times:
        if t < first - hold_ms:
            continue
        if t > held_end + hold_ms:
            # 유지 시간 끝: 렌더·CVAT가 여기서 블러를 끄도록 outside 키프레임 하나를 둔다
            out.append(TrackFrame(t, None, "outside", None, frozenset()))
            break
        if t in track.obs:
            o = track.obs[t]
            box, kind, score, dets = o.box, "observed", o.score, o.detectors
        elif t > last:
            # 마지막 관측 뒤 유지 (간격 캐시가 가리킨 동안 + hold_ms)
            box, kind, score, dets = held_box, "held", None, frozenset[str]()
        elif t < first:
            # 첫 관측 앞 미리 가림 (오프라인 처리라 가능)
            box, kind, score, dets = track.obs[first].box, "held", None, frozenset[str]()
        else:
            # 관측 사이: 앞뒤 관측 박스를 모두 덮는 외접 박스. 선형 보간은 간격(OWLv2 500 ms 등)
            # 사이에 곧게 움직이지 않은 대상을 내놓는다 (재현율 우선, F-PMS-45)
            before = max(x for x in times if x < t)
            after = min(x for x in times if x > t)
            box = track.obs[before].box.hull(track.obs[after].box)
            kind, score, dets = "interpolated", None, frozenset[str]()
        clipped = box.padded(margin).clipped(width, height)
        out.append(TrackFrame(t, clipped, kind if clipped else "outside", score, dets))
    return out
