"""도구 작용부-표면 접촉 구간 (평면까지 거리, 히스테리시스).

작용부 샘플마다 같은 시각의 표면 좌표로 바꿔 평면까지 거리와 표면 안 여부를 본다. 거리가
on_distance_m 이하로 내려오면 접촉을 시작하고 off_distance_m을 넘거나 표면 밖으로 나가면 끝낸다.
구간 경계는 on 문턱 아래 첫·마지막 샘플이다 (off 문턱은 잡음에 의한 끊김만 막는다).
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
    entity_id: str
    part: str | None
    times: NDArray[np.int64]
    xyz: NDArray[np.float64]  # (N, 3)

    @classmethod
    def from_payload(cls, p: Trajectory3DPayload) -> Track3D:
        times = np.array([s.t_ms for s in p.samples], dtype=np.int64)
        xyz = np.array([[s.x, s.y, s.z] for s in p.samples], dtype=np.float64)
        order = np.argsort(times, kind="stable")
        return cls(p.entity_id, p.part, times[order], xyz[order])


@dataclass
class SurfaceContact:
    tool_id: str
    tool_part: str | None
    surface_id: str
    start_ms: int
    end_ms: int
    # 구간 안 샘플의 표면 좌표 (시각, 가로 비율, 세로 비율)와 그 시각 표면 크기 (m)
    points: list[tuple[int, float, float]] = field(default_factory=list[tuple[int, float, float]])
    sizes: list[tuple[float, float]] = field(default_factory=list[tuple[float, float]])


def surface_frame_at(corners: list[Track3D], t: int, max_gap_ms: int) -> SurfaceFrame | None:
    points: list[NDArray[np.float64]] = []
    for c in corners:
        p = interpolate(c.times, c.xyz, t, max_gap_ms)
        if p is None:
            return None
        points.append(p)
    return SurfaceFrame.from_corners(np.stack(points))


def _inside(a: float, b: float, margin: float) -> bool:
    return -margin <= a <= 1 + margin and -margin <= b <= 1 + margin


def tool_surface_contacts(
    tool: Track3D,
    surface_id: str,
    corners: list[Track3D],
    grasped: list[tuple[int, int]],
    policy: ToolSurfacePolicy,
) -> list[SurfaceContact]:
    def held(t: int) -> bool:
        return not policy.require_grasp or any(s <= t <= e for s, e in grasped)

    runs: list[SurfaceContact] = []
    current: SurfaceContact | None = None
    last_on: int | None = None
    for t_raw, p in zip(tool.times, tool.xyz, strict=True):
        t = int(t_raw)
        frame = surface_frame_at(corners, t, policy.max_time_gap_ms)
        local = frame.local(p) if frame is not None else None
        ok = local is not None and _inside(local[0], local[1], policy.inside_margin) and held(t)
        if current is None:
            if ok and local is not None and local[2] <= policy.on_distance_m:
                current = SurfaceContact(tool.entity_id, tool.part, surface_id, t, t)
                last_on = t
        elif not ok or local is None or local[2] > policy.off_distance_m:
            assert last_on is not None
            current.end_ms = last_on
            runs.append(current)
            current, last_on = None, None
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

    merged: list[SurfaceContact] = []
    for r in runs:
        if merged and r.start_ms - merged[-1].end_ms <= policy.merge_gap_ms:
            merged[-1].end_ms = r.end_ms
            merged[-1].points += r.points
            merged[-1].sizes += r.sizes
        else:
            merged.append(r)
    return [r for r in merged if r.end_ms - r.start_ms >= policy.min_duration_ms]
