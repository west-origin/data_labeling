"""접촉 구간 추정 (모델이 아니라 알고리즘).

- 장갑: 압력 합을 히스테리시스 문턱으로 나눈다. 시작은 on 문턱을 넘은 샘플에서 off 이하였던
  마지막 샘플 다음까지 되짚고, 끝은 off 아래로 내려간 첫 샘플이다 (장갑 접촉 시각이 접촉 모델의
  정답 신호다).
- 영상: 손가락 끝 다섯 점과 객체 박스 사이 최소 거리가 문턱 이하인 프레임을 접촉 후보로 본다. 손
  키프레임 시각에 박스 키프레임이 없으면 앞뒤 박스 키프레임을 선형 보간한다 (간격이 box_max_gap_ms
  이하일 때만). 도구 박스는 프레임마다가 아니라 frame_stride_ms마다 나온다.
- 융합: 장갑 구간의 시각을 쓰고, 대상은 그 구간과 가장 많이 겹치는 영상 구간의 객체로 정한다. 장갑
  구간과 겹치지 않는 영상 구간은 영상 출처(낮은 신뢰도)로 그대로 낸다 (장갑이 놓친 접촉일 수 있어
  검수자가 먼저 본다).

파이프라인 위치: `runner._contacts`가 손·박스 트랙과 장갑 신호로 부르고 결과를 hand_state 라벨로
쓴다. 정책: `prelabel.yaml contact.glove`, `contact.video`. 관련: WP8, ADR 0015, 0026.

시간 단위 주의: 장갑 구간은 러너가 장갑 시각을 마스터 ms로 바꿔 넘긴다. 영상 구간은 손 키프레임의
스트림 PTS ms다. 러너는 바디캠(기준 스트림, ADR 0019)만 쓰므로 둘이 같은 축이 된다.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from dlp_prelabel.policy import GloveContactPolicy, VideoContactPolicy
from dlp_schema.labels import BoxTrackPayload, KeypointTrackPayload

# hand21 손가락 끝 번호 (엄지·검지·중지·약지·새끼). 손바닥 접촉은 이 휴리스틱이 못 잡는다
FINGERTIPS = (4, 8, 12, 16, 20)


@dataclass(frozen=True)
class ContactInterval:
    """접촉 구간 하나.

    start_ms, end_ms: 구간 경계 (정수 ms, 위 시간 단위 주의 참고). 양 끝을 포함하는 샘플 시각이다.
    target_id: 접촉 대상 개체 ID (박스 트랙 entity_id). 장갑만 잡아 모르면 None.
    source: "glove"(장갑만) | "video"(영상만) | "fused"(장갑 시각 + 영상 대상). 신뢰도·증거
    종류를 정한다.
    """

    start_ms: int
    end_ms: int
    target_id: str | None
    source: str  # glove | video | fused


def glove_contact_intervals(
    t_ms: NDArray[np.float64], pressure: NDArray[np.float64], policy: GloveContactPolicy
) -> list[ContactInterval]:
    """장갑 압력 합 시계열 → 접촉 구간 (대상 없음, source="glove").

    알고리즘: on 문턱을 넘는 샘플 i를 찾으면, 시작은 압력이 off 문턱 위로 이어지는 동안 앞으로
    되짚고, 끝은 off 문턱 이상이 이어지는 동안 뒤로 늘린다 (off 아래로 내려간 첫 샘플이 끝 시각).
    min_duration_ms보다 짧은 구간은 버리고, 다음 탐색은 끝 샘플 다음부터 한다.

    Args:
        t_ms: 샘플 시각 (ms, 오름차순). 러너는 마스터 ms로 바꿔 넘긴다.
        pressure: 같은 길이의 압력 합.
        policy: `prelabel.yaml contact.glove`.

    Returns:
        시각 순 구간 목록. 경계는 정수 ms로 반올림한다.
    """
    out: list[ContactInterval] = []
    i, n = 0, t_ms.size
    while i < n:
        # on 문턱을 넘는 샘플이 나올 때까지 넘긴다
        if pressure[i] <= policy.on_threshold:
            i += 1
            continue
        start = i
        # 시작 되짚기: on을 넘기 전에 이미 off 위로 올라와 있던 샘플까지 앞으로 넓힌다
        while start > 0 and pressure[start - 1] > policy.off_threshold:
            start -= 1
        end = i
        # 끝 늘리기: off 아래로 내려간 첫 샘플(end)에서 멈춘다. 끝까지 접촉이면 end == n
        while end < n and pressure[end] >= policy.off_threshold:
            end += 1
        # 끝 시각은 off 아래로 내려간 첫 샘플 시각 (신호 끝까지 접촉이면 마지막 샘플)
        s_ms, e_ms = float(t_ms[start]), float(t_ms[min(end, n - 1)])
        if e_ms - s_ms >= policy.min_duration_ms:
            out.append(ContactInterval(round(s_ms), round(e_ms), None, "glove"))
        # end 샘플은 off 아래라 새 구간의 시작이 될 수 없으므로 그다음부터 찾는다
        i = end + 1
    return out


def _box_distance(px: float, py: float, box: tuple[float, float, float, float]) -> float:
    """점 (px, py)과 박스 (x, y, w, h) 사이 최소 유클리드 거리(px). 점이 박스 안이면 0."""
    x, y, w, h = box
    # 점이 박스 왼쪽이면 x - px, 오른쪽이면 px - (x + w), 안이면 0 (세로도 같다)
    dx = max(x - px, 0.0, px - (x + w))
    dy = max(y - py, 0.0, py - (y + h))
    return float(np.hypot(dx, dy))


Box = tuple[float, float, float, float]


def box_at(
    track: BoxTrackPayload, t_ms: int, max_gap_ms: float, times: list[int] | None = None
) -> Box | None:
    """t_ms의 박스. 키프레임이 없으면 앞뒤 키프레임을 선형 보간한다.

    앞뒤 중 하나가 화면 밖(outside)이거나 간격이 max_gap_ms보다 길면 없음.
    times: 키프레임 시각 목록 (여러 번 부를 때 미리 만들어 넘긴다).

    Args:
        track: 박스 트랙 (키프레임이 시각 오름차순이라고 가정한다).
        t_ms: 찾을 시각 (스트림 PTS ms).
        max_gap_ms: 보간할 최대 키프레임 간격 (`contact.video.box_max_gap_ms`).

    Returns:
        (x, y, w, h) 픽셀 또는 None (트랙 범위 밖, 화면 밖, 간격 초과).
    """
    frames = track.keyframes
    # i: t_ms 이상인 첫 키프레임 번호
    i = bisect.bisect_left(times if times is not None else [k.t_ms for k in frames], t_ms)
    if i < len(frames) and frames[i].t_ms == t_ms:
        k = frames[i]
        return None if k.outside else (k.x, k.y, k.w, k.h)
    if i == 0 or i == len(frames):
        return None
    a, b = frames[i - 1], frames[i]
    if a.outside or b.outside or b.t_ms - a.t_ms > max_gap_ms:
        return None
    # 앞뒤 키프레임 사이 비율 (0~1)로 x, y, w, h를 각각 선형 보간한다
    r = (t_ms - a.t_ms) / (b.t_ms - a.t_ms)
    return (
        a.x + (b.x - a.x) * r,
        a.y + (b.y - a.y) * r,
        a.w + (b.w - a.w) * r,
        a.h + (b.h - a.h) * r,
    )


def video_contact_intervals(
    hand: KeypointTrackPayload, objects: list[BoxTrackPayload], policy: VideoContactPolicy
) -> list[ContactInterval]:
    """손 키포인트 트랙과 객체 박스 트랙 → 영상 접촉 구간 (source="video").

    프레임(손 키프레임)마다 손가락 끝 다섯 점(hand21의 4, 8, 12, 16, 20)과 각 박스 사이 최소 거리를
    재고, 문턱 이하인 박스 중 가장 가까운 것을 그 프레임의 대상으로 둔다. 직전 구간과 같은 대상이
    merge_gap_ms 안에 다시 나오면 구간을 잇고(사이에 다른 대상이 끼면 새 구간), min_duration_ms보다
    짧은 구간은 버린다.

    Args:
        hand: hand21 키포인트 트랙 (바디캠, 스트림 PTS ms).
        objects: 같은 스트림의 박스 트랙들 (객체·도구).
        policy: `prelabel.yaml contact.video`.

    Returns:
        시각 순 구간 목록. 대상이 바뀌면 새 구간이 된다.
    """
    # 키프레임 시각 목록을 한 번만 만들어 box_at의 이분 탐색에 넘긴다
    tracks = [(obj, [k.t_ms for k in obj.keyframes]) for obj in objects if obj.keyframes]
    per_frame: list[tuple[int, str | None]] = []
    for f in hand.keyframes:
        best: tuple[float, str] | None = None
        for obj, times in tracks:
            # 트랙이 존재하는 시간 밖이면 비교하지 않는다
            if f.t_ms < times[0] or f.t_ms > times[-1]:
                continue
            box = box_at(obj, f.t_ms, policy.box_max_gap_ms, times)
            if box is None:
                continue
            entity = obj.entity_id
            d = min(_box_distance(f.points[i].x, f.points[i].y, box) for i in FINGERTIPS)
            if d <= policy.max_distance_px and (best is None or d < best[0]):
                best = (d, entity)
        per_frame.append((f.t_ms, best[1] if best else None))

    raw: list[ContactInterval] = []
    for t, target in per_frame:
        if target is None:
            continue
        # 직전 구간과 대상이 같고 merge_gap_ms 안이면 잇는다. 대상이 바뀌면 새 구간이다
        if raw and raw[-1].target_id == target and t - raw[-1].end_ms <= policy.merge_gap_ms:
            raw[-1] = ContactInterval(raw[-1].start_ms, t, target, "video")
        else:
            raw.append(ContactInterval(t, t, target, "video"))
    return [c for c in raw if c.end_ms - c.start_ms >= policy.min_duration_ms]


def _overlap(a: ContactInterval, b: ContactInterval) -> int:
    """두 구간의 겹친 길이(ms). 겹치지 않으면 0 이하."""
    return min(a.end_ms, b.end_ms) - max(a.start_ms, b.start_ms)


def fuse_contacts(
    glove: list[ContactInterval], video: list[ContactInterval]
) -> list[ContactInterval]:
    """장갑 구간마다 대상을 붙이고, 어느 장갑 구간과도 겹치지 않는 영상 구간은 영상 출처로 낸다.

    장갑 구간의 대상은 가장 오래 겹친 영상 구간의 대상이다 (겹침이 같으면 대상 ID가 큰 쪽).
    대상을 찾으면 source="fused", 못 찾으면 "glove" 그대로(대상 None). 결과는 (시작, 끝) 순 정렬.
    """
    out: list[ContactInterval] = []
    for g in glove:
        overlaps = [(_overlap(g, v), v.target_id) for v in video if _overlap(g, v) > 0]
        # 가장 오래 겹친 영상 구간의 대상 (겹침이 같으면 튜플 비교로 대상 ID가 큰 쪽)
        target = max(overlaps)[1] if overlaps else None
        out.append(ContactInterval(g.start_ms, g.end_ms, target, "fused" if target else "glove"))
    out += [v for v in video if not any(_overlap(g, v) > 0 for g in glove)]
    return sorted(out, key=lambda c: (c.start_ms, c.end_ms))
