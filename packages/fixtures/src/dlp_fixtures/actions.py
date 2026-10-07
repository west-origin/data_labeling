"""정답 경계를 아는 행동 시퀀스: 손 키포인트 궤적, 장갑 압력, 정답 라벨.

한 손(기본 오른손)이 대기 → 행동 → 행동 … 을 이어서 한다. 행동마다 접근(최소 저크 궤적,
빠름) → 접촉(동사별 느린 움직임, 압력 있음) → 이탈(쉬는 위치 쪽으로 절반 복귀)의 세 국면을
가진다.
잡다 → 옮기다 → 놓다 묶음은 접촉이 이어지므로 접촉 시작은 잡다에, 접촉 종료는 놓다에만 둔다.

경계 후보 생성기(WP10)는 손 속도와 압력에서 이 정답 경계를 다시 찾아야 한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from dlp_fixtures.io import write_jsonl, write_parquet
from dlp_schema.episode import Entity, EntityKind
from dlp_schema.labels import (
    ActionPayload,
    GapPayload,
    Hand,
    HandStatePayload,
    Keypoint,
    KeypointFrame,
    KeypointTrackPayload,
    LabelPayload,
    LabelRecord,
    Provenance,
    SegmentLevel,
    SegmentPayload,
    Source,
)
from dlp_schema.testing import FIXED_TIME

FRAME_MS = 1000 / 30
GLOVE_RATE = 100.0
REST = np.array([160.0, 200.0])

# (개체 ID, 클래스, 개체 종류, 화면 위치, 손 접촉 대상 종류)
ENTITIES: list[tuple[str, str, EntityKind, tuple[float, float], str]] = [
    ("cup_01", "cup", EntityKind.OBJECT, (90.0, 120.0), "object"),
    ("spray_bottle_01", "spray_bottle", EntityKind.TOOL, (240.0, 110.0), "tool"),
    ("drawer_01", "drawer", EntityKind.OBJECT, (60.0, 180.0), "object"),
    ("bucket_01", "bucket", EntityKind.OBJECT, (260.0, 190.0), "object"),
    ("sink_01", "sink", EntityKind.SURFACE, (170.0, 80.0), "fixed_surface"),
]
# 단일 접촉 동사 → (대상 개체, 파지 유형)
SINGLE_ACTIONS: dict[str, tuple[str, str]] = {
    "press": ("spray_bottle_01", "tool_grip"),
    "push": ("cup_01", "palm_push_wipe"),
    "pull": ("drawer_01", "hook"),
    "rotate": ("cup_01", "power"),
    "rub": ("sink_01", "palm_push_wipe"),
    "support": ("bucket_01", "palm_support"),
}
CHAIN_TARGETS = ("cup_01", "spray_bottle_01")

# 손 21관절 템플릿 (손목 기준 상대 위치, 픽셀). 손가락 5개, 각 4관절.
_FINGER_DIRS = np.deg2rad(np.array([-60.0, -25.0, -5.0, 15.0, 35.0]))
_TEMPLATE = np.vstack(
    [np.zeros((1, 2))]
    + [
        np.stack([np.sin(a) * r, -np.cos(a) * r], axis=1)
        for a in _FINGER_DIRS
        for r in [np.array([8.0, 14.0, 19.0, 23.0])]
    ]
)


@dataclass
class _Plan:
    """한 행동의 정답 시각과 대상."""

    verb: str
    target: str
    grasp: str
    t_approach: int
    t_contact_start: int | None
    t_contact_end: int | None
    t_end: int
    contact_held: bool = False


@dataclass
class ActionScenario:
    session_id: str
    hand: Hand
    duration_ms: int
    labels: list[LabelRecord]
    entities: list[Entity]
    frame_times: list[int] = field(repr=False)
    wrist: NDArray[np.float64] = field(repr=False)
    glove_t_ms: NDArray[np.float64] = field(repr=False)
    glove_pressure: NDArray[np.float64] = field(repr=False)

    @property
    def actions(self) -> list[ActionPayload]:
        return [x.payload for x in self.labels if isinstance(x.payload, ActionPayload)]

    def write(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(out_dir / "labels.jsonl", self.labels)
        write_jsonl(out_dir / "entities.jsonl", self.entities)
        write_parquet(
            out_dir / f"glove_{self.hand.value}.parquet",
            {"t_ms": self.glove_t_ms, "pressure_0": self.glove_pressure},
            {"clock": "bodycam"},
        )


def generate_action_scenario(
    seed: int = 0,
    *,
    session_id: str = "syn-actions",
    n_units: int = 8,
    hand: Hand = Hand.RIGHT,
    ontology_version: str = "1.0.0",
) -> ActionScenario:
    rng = np.random.default_rng(seed)
    pos = {eid: np.array(p) for eid, _, _, p, _ in ENTITIES}
    contact_kind = {eid: k for eid, _, _, _, k in ENTITIES}

    plans: list[_Plan] = []
    gaps: list[tuple[int, int]] = []
    # 손목 위치를 정할 제어점: (시각, 위치, 접촉 중 움직임 종류)
    keys: list[tuple[int, NDArray[np.float64], str]] = [(0, REST.copy(), "still")]
    t = 0

    def idle(duration: int) -> None:
        nonlocal t
        gaps.append((t, t + duration))
        keys.append((t + duration, keys[-1][1].copy(), "still"))
        t += duration

    def approach(target: str) -> int:
        nonlocal t
        start = t
        t += int(rng.integers(300, 700))
        keys.append((t, pos[target] + rng.normal(0, 3, 2), "minjerk"))
        return start

    def contact(motion: str, duration: int, shift: tuple[float, float] = (0.0, 0.0)) -> None:
        nonlocal t
        t += duration
        keys.append((t, keys[-1][1] + np.array(shift), motion))

    def retreat() -> None:
        # 쉬는 위치 쪽으로 절반 돌아간다. 다음 접근이 같은 대상이어도 접근 동작이 뚜렷하게 남는다.
        nonlocal t
        t += int(rng.integers(200, 350))
        here = keys[-1][1]
        keys.append((t, here + (REST - here) * 0.5, "minjerk"))

    idle(int(rng.integers(400, 900)))
    for _ in range(n_units):
        if rng.random() < 0.3:
            target = str(rng.choice(CHAIN_TARGETS))
            a0 = approach(target)
            c0 = t
            contact("still", int(rng.integers(200, 400)))  # 잡다
            g_end = t
            dest = (float(rng.uniform(-60, 60)), float(rng.uniform(-40, 20)))
            contact("minjerk", int(rng.integers(600, 1_100)), dest)  # 옮기다
            c_end = t
            contact("still", int(rng.integers(150, 300)))  # 놓다
            r0 = t
            retreat()
            pos[target] = keys[-2][1].copy()
            plans += [
                _Plan("grasp", target, "power", a0, c0, None, g_end),
                _Plan("carry", target, "power", g_end, None, None, c_end, contact_held=True),
                _Plan("release", target, "power", c_end, None, r0, t, contact_held=True),
            ]
        else:
            verb = str(rng.choice(list(SINGLE_ACTIONS)))
            target, grasp = SINGLE_ACTIONS[verb]
            a0 = approach(target)
            c0 = t
            motion, shift = {
                "push": ("minjerk", (25.0, 0.0)),
                "pull": ("minjerk", (0.0, 25.0)),
                "rotate": ("circle", (0.0, 0.0)),
                "rub": ("rub", (0.0, 0.0)),
            }.get(verb, ("still", (0.0, 0.0)))
            contact(motion, int(rng.integers(500, 1_300)), shift)
            c1 = t
            retreat()
            plans.append(_Plan(verb, target, grasp, a0, c0, c1, t))
        if rng.random() < 0.5:
            idle(int(rng.integers(300, 1_000)))
    idle(500)
    duration = t

    frame_times = [round(i * FRAME_MS) for i in range(int(duration / FRAME_MS) + 1)]
    wrist = _wrist_track(keys, np.asarray(frame_times, dtype=float), rng)
    glove_t = np.arange(0, duration, 1000 / GLOVE_RATE)
    pressure = _pressure(plans, glove_t, rng)

    labels = _labels(
        session_id, hand, ontology_version, plans, gaps, contact_kind, frame_times, wrist, duration
    )
    entities = [Entity(entity_id=eid, kind=k, class_id=c) for eid, c, k, _, _ in ENTITIES]
    entities.append(Entity(entity_id=f"{hand.value}_hand", kind=EntityKind.HAND))
    return ActionScenario(
        session_id, hand, duration, labels, entities, frame_times, wrist, glove_t, pressure
    )


# ---------------------------------------------------------------- 궤적과 신호


def _wrist_track(
    keys: list[tuple[int, NDArray[np.float64], str]],
    times: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    out = np.zeros((times.size, 2))
    for i, tm in enumerate(times):
        j = max(k for k in range(len(keys)) if keys[k][0] <= tm) if tm >= 0 else 0
        if j == len(keys) - 1:
            out[i] = keys[j][1]
            continue
        (t0, p0, _), (t1, p1, motion) = keys[j], keys[j + 1]
        s = (tm - t0) / (t1 - t0)
        if motion == "still":
            out[i] = p0
        elif motion == "minjerk":
            out[i] = p0 + (p1 - p0) * (10 * s**3 - 15 * s**4 + 6 * s**5)
        elif motion == "rub":
            out[i] = p0 + np.array([10.0 * np.sin(2 * np.pi * 3.0 * (tm - t0) / 1000), 0.0])
        else:  # circle
            ang = 2 * np.pi * (tm - t0) / 1000
            out[i] = p0 + np.array([6.0 * np.sin(ang), 6.0 * (1 - np.cos(ang))])
    return out + rng.normal(0, 0.3, out.shape)


def _pressure(
    plans: list[_Plan], t: NDArray[np.float64], rng: np.random.Generator
) -> NDArray[np.float64]:
    p = np.zeros(t.size)
    start: int | None = None
    for plan in plans:
        if plan.t_contact_start is not None:
            start = plan.t_contact_start
        if plan.t_contact_end is not None and start is not None:
            level = rng.uniform(0.6, 1.0)
            ramp = np.clip(np.minimum(t - start, plan.t_contact_end - t) / 30.0, 0, 1)
            p = np.maximum(p, ramp * level)
            start = None
    return np.clip(p + rng.normal(0, 0.01, t.size), 0, None)


# ---------------------------------------------------------------- 정답 라벨


def _labels(
    session_id: str,
    hand: Hand,
    ontology_version: str,
    plans: list[_Plan],
    gaps: list[tuple[int, int]],
    contact_kind: dict[str, str],
    frame_times: list[int],
    wrist: NDArray[np.float64],
    duration: int,
) -> list[LabelRecord]:
    counter = iter(range(10_000))

    def record(
        payload: LabelPayload, start: int, end: int, stream_id: str | None = None
    ) -> LabelRecord:
        return LabelRecord(
            label_id=f"{session_id}-{next(counter):04d}",
            session_id=session_id,
            t_start_ms=start,
            t_end_ms=end,
            ontology_version=ontology_version,
            provenance=Provenance(source=Source.HUMAN),
            created_at=FIXED_TIME,
            payload=payload,
            stream_id=stream_id,
        )

    labels: list[LabelRecord] = []
    for i, plan in enumerate(plans):
        payload = ActionPayload(
            action_id=f"{session_id}-a{i:03d}",
            hand=hand,
            verb=plan.verb,
            target_id=plan.target,
            t_approach_ms=plan.t_approach,
            t_contact_start_ms=plan.t_contact_start,
            t_contact_end_ms=plan.t_contact_end,
            t_end_ms=plan.t_end,
            contact_held=plan.contact_held,
        )
        labels.append(record(payload, plan.t_approach, plan.t_end))
    for start, end in gaps:
        labels.append(record(GapPayload(hand=hand, gap_type="idle"), start, end))

    # 손 상태: 접촉 구간과 그 사이(접촉 없음) 구간
    contacts: list[tuple[int, int, _Plan]] = []
    open_plan: tuple[int, _Plan] | None = None
    for plan in plans:
        if plan.t_contact_start is not None:
            open_plan = (plan.t_contact_start, plan)
        if plan.t_contact_end is not None and open_plan is not None:
            contacts.append((open_plan[0], plan.t_contact_end, open_plan[1]))
            open_plan = None
    cursor = 0
    for start, end, plan in contacts:
        if start > cursor:
            labels.append(record(_no_contact(hand), cursor, start))
        state = HandStatePayload(
            hand=hand,
            contact_target_kind=contact_kind[plan.target],
            target_id=plan.target,
            grasp_type=plan.grasp,
            role="active",
        )
        labels.append(record(state, start, end))
        cursor = end
    labels.append(record(_no_contact(hand), cursor, duration))

    keypoints = KeypointTrackPayload(
        entity_id=f"{hand.value}_hand",
        skeleton="hand21",
        hand=hand,
        keyframes=tuple(
            KeypointFrame(
                t_ms=tm,
                points=tuple(
                    Keypoint(x=float(x), y=float(y), visibility=2) for x, y in w + _TEMPLATE
                ),
            )
            for tm, w in zip(frame_times, wrist, strict=True)
        ),
    )
    labels.append(record(keypoints, frame_times[0], frame_times[-1], stream_id="bodycam"))
    task = SegmentPayload(
        segment_id=f"{session_id}-task", level=SegmentLevel.TASK, ref_id="surface_wipe_disinfect"
    )
    labels.append(record(task, 0, duration))
    return sorted(labels, key=lambda x: (x.t_start_ms, x.label_id))


def _no_contact(hand: Hand) -> HandStatePayload:
    return HandStatePayload(hand=hand, contact_target_kind="none", role="inactive")
