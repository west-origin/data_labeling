"""도구 작용부-표면 접촉 구간 (평면까지 거리, 히스테리시스).

작용부 샘플마다 같은 시각의 표면 좌표로 바꿔 평면까지 거리와 표면 안 여부를 본다. 거리가
on_distance_m 이하로 내려오면 접촉을 시작하고 off_distance_m을 넘거나 표면 밖으로 나가면 끝낸다.
구간 경계는 on 문턱 아래 첫·마지막 샘플이다 (off 문턱은 잡음에 의한 끊김만 막는다).

정책: `relations.yaml tool_surface`. `derive`가 (도구 작용부, 표면)마다 부르고, 결과는 관계 규칙
(source=tool_surface)과 커버리지 입력이 된다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from dlp_relations.geometry import SurfaceFrame, interpolate
from dlp_relations.policy import ToolSurfacePolicy
from dlp_schema.labels import Trajectory3DPayload


@dataclass(frozen=True)
class Track3D:
    """시각 순으로 정렬한 3D 궤적 (계산용).

    entity_id, part: 개체·부위. times: (N,) int64 ms. xyz: (N, 3) m (궤적의 좌표계 그대로).
    """

    entity_id: str
    part: str | None
    times: NDArray[np.int64]
    xyz: NDArray[np.float64]  # (N, 3)

    @classmethod
    def from_payload(cls, p: Trajectory3DPayload) -> Track3D:
        """trajectory3d 페이로드 → 시각 순 정렬 (같은 시각은 원래 순서 유지)."""
        times = np.array([s.t_ms for s in p.samples], dtype=np.int64)
        xyz = np.array([[s.x, s.y, s.z] for s in p.samples], dtype=np.float64)
        order = np.argsort(times, kind="stable")
        return cls(p.entity_id, p.part, times[order], xyz[order])


@dataclass
class SurfaceContact:
    """작용부-표면 접촉 구간 하나.

    tool_id, tool_part: 도구 개체와 작용부. surface_id: 표면 개체. start_ms, end_ms: on 문턱 아래
    첫·마지막 샘플 시각. points: 구간 안 샘플의 표면 좌표 (시각, 가로 비율, 세로 비율). sizes: 그
    시각 표면 크기(m).
    """

    tool_id: str
    tool_part: str | None
    surface_id: str
    start_ms: int
    end_ms: int
    # 구간 안 샘플의 표면 좌표 (시각, 가로 비율, 세로 비율)와 그 시각 표면 크기 (m)
    points: list[tuple[int, float, float]] = field(default_factory=list[tuple[int, float, float]])
    sizes: list[tuple[float, float]] = field(default_factory=list[tuple[float, float]])


def surface_frame_at(corners: list[Track3D], t: int, max_gap_ms: int) -> SurfaceFrame | None:
    """시각 t의 표면 좌표계. 꼭짓점 하나라도 max_gap_ms 안 샘플이 없으면 None.

    corners는 corner_parts 순서(0, 1, 2, 3)로 넘겨야 한다.
    """
    points: list[NDArray[np.float64]] = []
    for c in corners:
        p = interpolate(c.times, c.xyz, t, max_gap_ms)
        if p is None:
            return None
        points.append(p)
    return SurfaceFrame.from_corners(np.stack(points))


def _inside(a: float, b: float, margin: float) -> bool:
    """표면 좌표 (a, b)가 [-margin, 1+margin] 사각형 안인가."""
    return -margin <= a <= 1 + margin and -margin <= b <= 1 + margin


def tool_surface_contacts(
    tool: Track3D,
    surface_id: str,
    corners: list[Track3D],
    grasped: list[tuple[int, int]],
    policy: ToolSurfacePolicy,
) -> list[SurfaceContact]:
    """작용부 궤적 하나와 표면 하나 사이 접촉 구간들.

    알고리즘 (작용부 샘플마다):
    1. 같은 시각 표면 좌표계를 만들고(꼭짓점 보간) 작용부를 (a, b, 거리)로 바꾼다.
    2. 표면 안·쥐는 중(`require_grasp`)·거리 ≤ on이면 접촉 시작. 접촉 중 표면 밖·안 쥠·거리 > off면
       끝. 끝 시각은 마지막으로 거리 ≤ on이었던 샘플이다.
    3. 구간 밖(on 위) 샘플 좌표는 커버리지에서 뺀다.
    4. merge_gap_ms 이하로 끊긴 구간을 잇고, min_duration_ms 미만 구간은 버린다.

    Args:
        tool: 작용부 궤적. surface_id: 표면 개체 ID. corners: 꼭짓점 궤적 4개 (corner_parts 순서).
        grasped: 손이 이 도구를 쥔 구간 [(시작, 끝)] (양 끝 포함).
        policy: `relations.yaml tool_surface`.
    """

    def held(t: int) -> bool:
        """시각 t에 손이 도구를 쥐고 있는가 (require_grasp가 꺼져 있으면 항상 참)."""
        return not policy.require_grasp or any(s <= t <= e for s, e in grasped)

    runs: list[SurfaceContact] = []
    current: SurfaceContact | None = None
    last_on: int | None = None
    for t_raw, p in zip(tool.times, tool.xyz, strict=True):
        t = int(t_raw)
        # 작용부 샘플 시각의 표면 좌표계 (꼭짓점 보간). 못 만들면 이 샘플은 표면 밖으로 본다
        frame = surface_frame_at(corners, t, policy.max_time_gap_ms)
        local = frame.local(p) if frame is not None else None
        ok = local is not None and _inside(local[0], local[1], policy.inside_margin) and held(t)
        # 접촉 밖: 안·쥠·거리 ≤ on이면 시작. 접촉 중: 밖·안 쥠·거리 > off면 끝 (히스테리시스)
        if current is None:
            if ok and local is not None and local[2] <= policy.on_distance_m:
                current = SurfaceContact(tool.entity_id, tool.part, surface_id, t, t)
                last_on = t
        elif not ok or local is None or local[2] > policy.off_distance_m:
            # 끝 시각은 마지막으로 on 문턱 아래였던 샘플 (off 사이 꼬리는 구간에 넣지 않는다)
            assert last_on is not None
            current.end_ms = last_on
            runs.append(current)
            current, last_on = None, None
        # 접촉 중 샘플은 일단 좌표를 모으고, on 문턱 아래면 last_on을 갱신한다
        if current is not None and local is not None and frame is not None:
            current.points.append((t, local[0], local[1]))
            current.sizes.append(frame.size_m)
            if local[2] <= policy.on_distance_m:
                last_on = t
    if current is not None and last_on is not None:
        current.end_ms = last_on
        runs.append(current)
    for r in runs:  # 경계 밖(on 문턱 위) 샘플은 커버리지에서 뺀다
        r.points = [x for x in r.points if r.start_ms <= x[0] <= r.end_ms]

    # merge_gap_ms 이하로 끊긴 구간을 잇는다 (커버리지 점·크기도 합친다)
    merged: list[SurfaceContact] = []
    for r in runs:
        if merged and r.start_ms - merged[-1].end_ms <= policy.merge_gap_ms:
            merged[-1].end_ms = r.end_ms
            merged[-1].points += r.points
            merged[-1].sizes += r.sizes
        else:
            merged.append(r)
    return [r for r in merged if r.end_ms - r.start_ms >= policy.min_duration_ms]
