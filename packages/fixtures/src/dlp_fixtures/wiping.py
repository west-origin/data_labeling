"""정답 접촉 구간·커버리지를 아는 걸레질 시나리오 (WP9).

오른손에 쥔 걸레(rag_01)로 탁자(table_01) 위를 줄 단위로 닦는다. 줄마다 작용부(cloth_face)를
내려 닿게 하고(하강), 표면을 따라 밀고, 들어 올려(상승) 다음 줄 위로 옮긴다.

정답:
- 도구-표면 접촉 구간: 작용부 높이가 0인 구간 (하강이 끝난 시각 ~ 상승이 시작한 시각)
- 커버리지: 접촉 중 작용부 경로를 반지름 footprint_m 원으로 쓸어 덮은 면적 / 탁자 면적.
  접촉 경로는 줄마다 곧은 선분이므로 선분까지 거리로 2 mm 격자에서 계산한다.

입력 라벨: 손 상태(걸레 파지), 걸레·탁자 박스 트랙, 작용부와 탁자 네 꼭짓점의 3D 궤적
(카메라 좌표, 30 fps, 2 mm 잡음). 카메라가 움직이므로 같은 점도 시각마다 좌표가 다르다.
꼭짓점 순서는 corner_0(원점) → corner_1(가로) → corner_2 → corner_3(세로)이다.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from dlp_schema.labels import (
    BoxKeyframe,
    BoxTrackPayload,
    CoordinateFrame,
    Hand,
    HandStatePayload,
    LabelPayload,
    LabelRecord,
    Provenance,
    RelationPayload,
    RelationPredicate,
    Source,
    Source3D,
    Trajectory3DPayload,
    Trajectory3DSample,
)
from dlp_schema.testing import FIXED_TIME

FRAME_MS = 1000 / 30
SURFACE_ID, SURFACE_CLASS = "table_01", "table"
TOOL_ID, TOOL_CLASS, TOOL_PART = "rag_01", "rag", "cloth_face"
CORNER_PARTS = ("corner_0", "corner_1", "corner_2", "corner_3")
WIDTH_M, DEPTH_M = 0.8, 0.5  # 탁자 크기
LIFT_M = 0.08  # 줄 사이를 옮길 때 작용부 높이
DESCEND_MS, ASCEND_MS, TRANSIT_MS = 250, 250, 400
TRUTH_GRID_M = 0.002


@dataclass(frozen=True)
class _Move:
    t0: int
    t1: int
    start: tuple[float, float, float]  # 표면 좌표 (가로 m, 세로 m, 높이 m)
    end: tuple[float, float, float]


@dataclass
class WipingScenario:
    session_id: str
    ontology_version: str
    duration_ms: int
    footprint_m: float
    labels: list[LabelRecord]  # 입력: 손 상태, 박스 트랙, 3D 궤적
    truth_relations: list[LabelRecord]  # 정답 관계 (파지, 도구-표면 접촉)
    contacts: list[tuple[int, int]]  # 정답 도구-표면 접촉 구간
    grasp: tuple[int, int]
    strokes: list[tuple[tuple[float, float], tuple[float, float]]]  # 줄별 접촉 선분 (표면 좌표 m)
    coverage: float  # 정답 커버리지 (0~1)


def _smooth(u: float) -> float:
    return u * u * (3 - 2 * u)


def _position(moves: list[_Move], t: float) -> NDArray[np.float64]:
    for m in moves:
        if m.t0 <= t <= m.t1:
            u = 0.0 if m.t1 == m.t0 else _smooth((t - m.t0) / (m.t1 - m.t0))
            return np.array(m.start) + (np.array(m.end) - np.array(m.start)) * u
    return np.array(moves[0].start if t < moves[0].t0 else moves[-1].end)


def _camera_transform(seed: int) -> Callable[[float, NDArray[np.float64]], NDArray[np.float64]]:
    """표면 좌표 → 카메라 좌표. 탁자는 앞으로 기울어져 있고 카메라는 천천히 흔들린다."""
    rng = np.random.default_rng([seed, 7])
    tilt = np.deg2rad(rng.uniform(50, 70))
    # 표면 축: 가로 = 카메라 X, 세로 = 카메라 Z 쪽으로 기울어짐, 법선 = 카메라 쪽(-Y 성분)
    ex = np.array([1.0, 0.0, 0.0])
    ey = np.array([0.0, -np.cos(tilt), np.sin(tilt)])
    n = np.cross(ex, ey)
    origin = np.array([-WIDTH_M / 2, 0.35, 0.55])
    yaw_amp, period = rng.uniform(0.03, 0.08), rng.uniform(2500, 4000)

    def to_camera(t: float, p: NDArray[np.float64]) -> NDArray[np.float64]:
        world = origin + p[0] * ex + p[1] * ey + p[2] * n
        yaw = yaw_amp * np.sin(2 * np.pi * t / period)
        c, s = np.cos(yaw), np.sin(yaw)
        rot = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        shift = np.array([0.02 * np.sin(2 * np.pi * t / period), 0.01 * np.cos(t / 500), 0.0])
        return rot @ world + shift

    return to_camera


def _truth_coverage(
    strokes: list[tuple[tuple[float, float], tuple[float, float]]], radius: float
) -> float:
    xs = np.arange(TRUTH_GRID_M / 2, WIDTH_M, TRUTH_GRID_M)
    ys = np.arange(TRUTH_GRID_M / 2, DEPTH_M, TRUTH_GRID_M)
    gx, gy = np.meshgrid(xs, ys)
    covered = np.zeros(gx.shape, dtype=bool)
    for (ax, ay), (bx, by) in strokes:
        dx, dy = bx - ax, by - ay
        u = np.clip(((gx - ax) * dx + (gy - ay) * dy) / (dx * dx + dy * dy), 0, 1)
        covered |= np.hypot(gx - (ax + u * dx), gy - (ay + u * dy)) <= radius
    return float(covered.mean())


def generate_wiping_scenario(
    seed: int = 0,
    *,
    session_id: str = "wipe-0000",
    ontology_version: str = "1.0.0",
    footprint_m: float = 0.06,
    noise_m: float = 0.002,
) -> WipingScenario:
    rng = np.random.default_rng(seed)
    rows = int(rng.integers(2, 5))
    spacing = float(rng.uniform(0.08, 0.12))
    a0, a1 = float(rng.uniform(0.05, 0.15)), float(rng.uniform(0.5, 0.75))
    speed = float(rng.uniform(0.3, 0.6))  # m/s
    b0 = 0.08

    grasp_start = 400
    t = grasp_start + 300
    moves: list[_Move] = []
    contacts: list[tuple[int, int]] = []
    strokes: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for k in range(rows):
        b = b0 + k * spacing
        left, right = (a0, a1) if k % 2 == 0 else (a1, a0)  # 지그재그
        if k > 0:
            prev = moves[-1].end
            moves.append(_Move(t, t + TRANSIT_MS, prev, (left, b, LIFT_M)))
            t += TRANSIT_MS
        moves.append(_Move(t, t + DESCEND_MS, (left, b, LIFT_M), (left, b, 0.0)))
        t += DESCEND_MS
        wipe_ms = round(abs(right - left) / speed * 1000)
        moves.append(_Move(t, t + wipe_ms, (left, b, 0.0), (right, b, 0.0)))
        contacts.append((t, t + wipe_ms))
        strokes.append(((left, b), (right, b)))
        t += wipe_ms
        moves.append(_Move(t, t + ASCEND_MS, (right, b, 0.0), (right, b, LIFT_M)))
        t += ASCEND_MS
    grasp_end = t + 300
    duration = grasp_end + 400

    to_camera = _camera_transform(seed)
    times = [round(i * FRAME_MS) for i in range(int(duration / FRAME_MS) + 1)]

    def noisy(p: NDArray[np.float64]) -> tuple[float, float, float]:
        q = p + rng.normal(0, noise_m, 3)
        return (round(float(q[0]), 5), round(float(q[1]), 5), round(float(q[2]), 5))

    tool_samples = tuple(
        Trajectory3DSample(t_ms=tm, x=x, y=y, z=z)
        for tm in times
        for x, y, z in [noisy(to_camera(tm, _position(moves, tm)))]
    )
    corner_local = [(0.0, 0.0), (WIDTH_M, 0.0), (WIDTH_M, DEPTH_M), (0.0, DEPTH_M)]
    corner_samples = {
        part: tuple(
            Trajectory3DSample(t_ms=tm, x=x, y=y, z=z)
            for tm in times
            for x, y, z in [noisy(to_camera(tm, np.array([ca, cb, 0.0])))]
        )
        for part, (ca, cb) in zip(CORNER_PARTS, corner_local, strict=True)
    }

    counter = iter(range(10_000))

    def record(
        payload: LabelPayload, start: int, end: int, stream: str | None = None
    ) -> LabelRecord:
        return LabelRecord(
            label_id=f"{session_id}-{next(counter):04d}",
            session_id=session_id,
            stream_id=stream,
            t_start_ms=start,
            t_end_ms=end,
            ontology_version=ontology_version,
            provenance=Provenance(source=Source.HUMAN),
            created_at=FIXED_TIME,
            payload=payload,
        )

    def traj(entity: str, part: str, samples: tuple[Trajectory3DSample, ...]) -> LabelRecord:
        payload = Trajectory3DPayload(
            entity_id=entity,
            part=part,
            frame=CoordinateFrame.CAMERA,
            source_3d=Source3D.MONO_DEPTH,
            samples=samples,
        )
        return record(payload, samples[0].t_ms, samples[-1].t_ms, "bodycam")

    def box(entity: str, cls: str, xywh: tuple[float, float, float, float]) -> LabelRecord:
        x, y, w, h = xywh
        frames = tuple(BoxKeyframe(t_ms=tm, x=x, y=y, w=w, h=h) for tm in (0, times[-1]))
        payload = BoxTrackPayload(entity_id=entity, class_id=cls, keyframes=frames)
        return record(payload, 0, times[-1], "bodycam")

    none = HandStatePayload(hand=Hand.RIGHT, contact_target_kind="none", role="inactive")
    grip = HandStatePayload(
        hand=Hand.RIGHT,
        contact_target_kind="tool",
        target_id=TOOL_ID,
        grasp_type="tool_grip",
        role="active",
    )
    labels = [
        record(none, 0, grasp_start),
        record(grip, grasp_start, grasp_end),
        record(none, grasp_end, duration),
        box(TOOL_ID, TOOL_CLASS, (150, 150, 60, 40)),
        box(SURFACE_ID, SURFACE_CLASS, (20, 100, 280, 130)),
        traj(TOOL_ID, TOOL_PART, tool_samples),
        *(traj(SURFACE_ID, part, corner_samples[part]) for part in CORNER_PARTS),
    ]
    truth = [
        record(
            RelationPayload(
                subject_id="right_hand", predicate=RelationPredicate.GRASP, object_id=TOOL_ID
            ),
            grasp_start,
            grasp_end,
        ),
        *(
            record(
                RelationPayload(
                    subject_id=TOOL_ID,
                    subject_part=TOOL_PART,
                    predicate=RelationPredicate.CONTACT,
                    object_id=SURFACE_ID,
                ),
                s,
                e,
            )
            for s, e in contacts
        ),
    ]
    return WipingScenario(
        session_id=session_id,
        ontology_version=ontology_version,
        duration_ms=duration,
        footprint_m=footprint_m,
        labels=labels,
        truth_relations=truth,
        contacts=contacts,
        grasp=(grasp_start, grasp_end),
        strokes=strokes,
        coverage=_truth_coverage(strokes, footprint_m),
    )
