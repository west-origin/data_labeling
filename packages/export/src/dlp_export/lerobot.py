"""LeRobot 에피소드 (형식 v3.0, 공식 쓰기·읽기 API).

세션의 바디캠 블러본 하나가 에피소드 하나다. 이 모듈은 고정 프레임률 시각마다의 특징 표와 그 시각에
보이던 블러본 프레임(PTS 인덱스로 고름)을 정하고, 격리된 일회용 환경의 scripts/lerobot_write.py가
LeRobot 공식 API로 쓴다 (PyTorch가 작업공간 의존성과 충돌하므로). 쓴 뒤 scripts/lerobot_check.py가
공식 로더로 다시 읽어 확인한다.

프레임 특징 (손은 왼손·오른손 순):
- observation.state (float32): 손 21관절 2D(정규화 x, y, COCO 보임 v: 0 없음·1 가려짐·2 보임)
  + 손 3D 점(카메라 좌표 x, y, z, 있음) + 쥔 도구 작용부 3D(x, y, z, 있음).
  키포인트·궤적은 interp_max_gap_ms 안에서만 선형 보간한다.
- action (float32): 다음 프레임의 observation.state (사람 시연 데이터의 관례, 마지막은 그대로).
- annotation.hand_state (int64): 손마다 [접촉, 접촉 대상 종류, 파지 유형, 역할]
  (어휘 번호, 라벨 없음 -1).
- annotation.tool_surface_contact (int64): 손마다 쥔 도구 작용부의 표면 접촉
  (0/1, 쥔 도구 없음 -1).
- annotation.verb (int64): 손마다 그 시각 원시 동작의 동사 번호 (없음 -1).
- annotation.verification (int64): [손 2D, 손 3D, 손 상태, 행동] 묶음마다 그 시각에 쓴 라벨의
  가장 낮은 검증 등급 (미검수 0, 표본 검증 1, 사람 승인 2, 사람 수정·작성 3, 라벨 없음 -1).
- task: 그 시각의 작업(task 수준 구간) ID, 없으면 세션 도메인.
어휘(번호 → ID)는 meta/dlp_vocab.json에 둔다.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dlp_export.frames import stream_ms
from dlp_export.policy import ExportPolicy, LeRobotPolicy
from dlp_media.pts import PtsIndex
from dlp_schema.labels import (
    ActionPayload,
    Hand,
    HandStatePayload,
    KeypointTrackPayload,
    LabelRecord,
    RelationPayload,
    RelationPredicate,
    SegmentLevel,
    SegmentPayload,
    Source,
    Trajectory3DPayload,
    VerificationState,
)
from dlp_schema.ontology import Ontology
from dlp_schema.session import Session, Stream

HANDS = (Hand.LEFT, Hand.RIGHT)
GRADE = {
    VerificationState.UNREVIEWED: 0,
    VerificationState.SAMPLE_VERIFIED: 1,
    VerificationState.HUMAN_APPROVED: 2,
    VerificationState.HUMAN_CORRECTED: 3,
}
GROUPS = ("hands_2d", "hands_3d", "hand_state", "actions")


def grade(x: LabelRecord) -> int:
    return 3 if x.provenance.source is Source.HUMAN else GRADE[x.verification.state]


@dataclass(frozen=True)
class Vocab:
    verbs: tuple[str, ...]
    contact_target_kinds: tuple[str, ...]
    grasp_types: tuple[str, ...]
    hand_roles: tuple[str, ...]
    working_parts: frozenset[str]
    tasks: tuple[str, ...]  # 에피소드 task 문자열 (작업 ID와 도메인)

    @classmethod
    def from_ontology(cls, o: Ontology) -> Vocab:
        parts = {p for obj in o.objects.values() if obj.tool for p in obj.tool.working_parts}
        return cls(
            tuple(sorted(o.verbs)),
            tuple(o.contact_target_kinds),
            tuple(o.grasp_types),
            tuple(o.hand_roles),
            frozenset(parts),
            tuple(sorted(o.tasks)) + tuple(sorted(o.domains)),
        )

    def as_json(self) -> dict[str, Any]:
        return {
            "verb": list(self.verbs),
            "contact_target_kind": list(self.contact_target_kinds),
            "grasp_type": list(self.grasp_types),
            "hand_role": list(self.hand_roles),
            "verification_grade": [
                "unreviewed",
                "sample_verified",
                "human_approved",
                "human_corrected_or_created",
            ],
            "verification_groups": list(GROUPS),
            "missing": -1,
        }


def state_names(policy: LeRobotPolicy) -> list[str]:
    from dlp_export.coco import HAND21

    names: list[str] = []
    for h in HANDS:
        for j in HAND21:
            names += [f"{h.value}.{j}.x", f"{h.value}.{j}.y", f"{h.value}.{j}.v"]
    for h in HANDS:
        for part in policy.hand_points_3d:
            names += [f"{h.value}.{part}.{a}" for a in ("x3d", "y3d", "z3d", "present")]
    for h in HANDS:
        names += [f"{h.value}.tool_tip.{a}" for a in ("x3d", "y3d", "z3d", "present")]
    return names


def features(policy: LeRobotPolicy, height: int, width: int) -> dict[str, Any]:
    n = len(state_names(policy))
    hands = [h.value for h in HANDS]
    return {
        "observation.images.bodycam": {
            "dtype": "video", "shape": [height, width, 3], "names": ["height", "width", "channels"],
        },
        "observation.state": {"dtype": "float32", "shape": [n], "names": state_names(policy)},
        "action": {"dtype": "float32", "shape": [n], "names": state_names(policy)},
        "annotation.hand_state": {
            "dtype": "int64", "shape": [8],
            "names": [
                f"{h}.{a}" for h in hands for a in ("contact", "target_kind", "grasp", "role")
            ],
        },
        "annotation.tool_surface_contact": {"dtype": "int64", "shape": [2], "names": hands},
        "annotation.verb": {"dtype": "int64", "shape": [2], "names": hands},
        "annotation.verification": {"dtype": "int64", "shape": [4], "names": list(GROUPS)},
    }  # fmt: skip


def _interp(
    series: list[tuple[int, NDArray[np.float64]]], t: float, max_gap: int
) -> NDArray[np.float64] | None:
    """(시각, 값) 목록에서 t의 값.

    양쪽 표본 간격이 max_gap보다 크면 None. 값의 NaN은 그대로 퍼진다.
    """
    if not series:
        return None
    times = [s[0] for s in series]
    i = int(np.searchsorted(times, t))
    if i < len(times) and times[i] == t:
        return series[i][1]
    if i == 0 or i == len(times):
        return None
    (t0, v0), (t1, v1) = series[i - 1], series[i]
    if t1 - t0 > max_gap:
        return None
    w = (t - t0) / (t1 - t0)
    return v0 * (1 - w) + v1 * w


def _covering(labels: list[LabelRecord], t: float) -> list[LabelRecord]:
    return [x for x in labels if x.t_start_ms <= t < x.t_end_ms or x.t_start_ms == x.t_end_ms == t]


@dataclass
class Episode:
    frame_index: NDArray[np.int64]  # 블러본 표시 순서 프레임 번호 (쓰기 전용, 저장하지 않는다)
    times_ms: NDArray[np.float64]  # 마스터 시각
    state: NDArray[np.float32]
    action: NDArray[np.float32]
    hand_state: NDArray[np.int64]
    tool_surface_contact: NDArray[np.int64]
    verb: NDArray[np.int64]
    verification: NDArray[np.int64]
    tasks: list[str]


def build_episode(
    session: Session,
    stream: Stream,
    index: PtsIndex,
    size: tuple[int, int],
    labels: list[LabelRecord],
    vocab: Vocab,
    policy: LeRobotPolicy,
) -> Episode:
    w, h = size
    gap = policy.interp_max_gap_ms
    # 에피소드 시각: 블러본 첫 프레임부터 끝까지 1/fps 간격 (마스터 시각)
    start = stream.to_master_ms(float(index.ms[0]))
    end = stream.to_master_ms(index.duration_ms)
    n = max(math.floor((end - start) * policy.fps / 1000), 1)
    times = start + np.arange(n) * 1000.0 / policy.fps
    frame_index = np.array([index.frame_at(stream_ms(stream, t)) for t in times], dtype=np.int64)

    # 손 2D: 스트림의 hand21 트랙 (같은 손에 트랙이 여럿이면 시각마다 먼저 값이 있는 것)
    kp2d: dict[Hand, list[tuple[list[tuple[int, NDArray[np.float64]]], LabelRecord]]] = {
        h_: [] for h_ in HANDS
    }
    for x in labels:
        p = x.payload
        if (
            isinstance(p, KeypointTrackPayload)
            and p.skeleton == "hand21"
            and p.hand
            and x.stream_id == stream.stream_id
        ):
            series = [
                (
                    f.t_ms,
                    np.array(
                        [
                            [q.x / w, q.y / h, q.visibility] if q.visibility else [np.nan] * 3
                            for q in f.points
                        ]
                    ),
                )
                for f in sorted(p.keyframes, key=lambda f: f.t_ms)
            ]
            kp2d[p.hand].append((series, x))
    # 3D 궤적: (개체, 부분) → 표본
    traj: dict[
        tuple[str, str | None], tuple[list[tuple[int, NDArray[np.float64]]], LabelRecord]
    ] = {}
    for x in labels:
        p = x.payload
        if isinstance(p, Trajectory3DPayload):
            traj[(p.entity_id, p.part)] = (
                [
                    (s.t_ms, np.array([s.x, s.y, s.z]))
                    for s in sorted(p.samples, key=lambda s: s.t_ms)
                ],
                x,
            )
    hand_states = [x for x in labels if isinstance(x.payload, HandStatePayload)]
    actions = [x for x in labels if isinstance(x.payload, ActionPayload)]
    tool_contacts = [
        x for x in labels
        if isinstance(x.payload, RelationPayload)
        and x.payload.predicate is RelationPredicate.CONTACT
        and x.payload.subject_part in vocab.working_parts
    ]  # fmt: skip
    tasks_lbl = [
        x for x in labels
        if isinstance(x.payload, SegmentPayload) and x.payload.level is SegmentLevel.TASK
    ]  # fmt: skip

    dim = len(state_names(policy))
    state = np.zeros((n, dim), np.float32)
    hs = np.full((n, 8), -1, np.int64)
    tsc = np.full((n, 2), -1, np.int64)
    verb = np.full((n, 2), -1, np.int64)
    ver = np.full((n, 4), -1, np.int64)
    task_names: list[str] = []
    n2d, n3d = 21 * 3, len(policy.hand_points_3d) * 4

    def worse(k: int, g: int, x: LabelRecord) -> None:
        cur = ver[k, g]
        ver[k, g] = grade(x) if cur < 0 else min(cur, grade(x))

    for k, t in enumerate(times):
        for hi, hand in enumerate(HANDS):
            # 2D
            for series, x in kp2d[hand]:
                v = _interp(series, t, gap)
                if v is not None:
                    # 양쪽 키프레임에 다 라벨이 있는 관절만 값이 있다. 보임 정도는 둘 중 낮은 쪽
                    # (보간한 v의 내림: 끝점에서는 그 값, 사이에서는 작은 값)
                    labeled = ~np.isnan(v[:, 0])
                    block = np.where(labeled[:, None], np.nan_to_num(v), 0.0)
                    block[:, 2] = np.where(labeled, np.floor(np.nan_to_num(v[:, 2])), 0.0)
                    state[k, hi * n2d : (hi + 1) * n2d] = block.reshape(-1)
                    worse(k, 0, x)
                    break
            # 3D 손 점
            base = 2 * n2d + hi * n3d
            for pi, part in enumerate(policy.hand_points_3d):
                got = traj.get((f"{hand.value}_hand", part))
                v = _interp(got[0], t, gap) if got else None
                if v is not None and got is not None:
                    state[k, base + pi * 4 : base + pi * 4 + 4] = [*v, 1.0]
                    worse(k, 1, got[1])
            # 손 상태와 쥔 도구
            cur = [x for x in _covering(hand_states, t) if x.payload.hand is hand]  # type: ignore[union-attr]
            tool_id: str | None = None
            if cur:
                x = cur[0]
                p = x.payload
                assert isinstance(p, HandStatePayload)
                kind = p.contact_target_kind
                hs[k, hi * 4 : hi * 4 + 4] = [
                    int(kind != "none"),
                    vocab.contact_target_kinds.index(kind)
                    if kind in vocab.contact_target_kinds
                    else -1,
                    vocab.grasp_types.index(p.grasp_type)
                    if p.grasp_type in vocab.grasp_types
                    else -1,
                    vocab.hand_roles.index(p.role) if p.role in vocab.hand_roles else -1,
                ]
                worse(k, 2, x)
                if kind == "tool":
                    tool_id = p.target_id
            tip_base = 2 * n2d + 2 * n3d + hi * 4
            if tool_id is not None:
                tsc[k, hi] = int(any(
                    x.payload.subject_id == tool_id  # type: ignore[union-attr]
                    for x in _covering(tool_contacts, t)
                ))  # fmt: skip
                for (ent, part), (series, x) in traj.items():
                    if ent == tool_id and part in vocab.working_parts:
                        v = _interp(series, t, gap)
                        if v is not None:
                            state[k, tip_base : tip_base + 4] = [*v, 1.0]
                            worse(k, 1, x)
                            break
            # 행동
            acts = [x for x in _covering(actions, t) if x.payload.hand is hand]  # type: ignore[union-attr]
            if acts:
                p = acts[0].payload
                assert isinstance(p, ActionPayload)
                verb[k, hi] = vocab.verbs.index(p.verb) if p.verb in vocab.verbs else -1
                worse(k, 3, acts[0])
        cover = _covering(tasks_lbl, t)
        seg = cover[0].payload if cover else None
        task_names.append(seg.ref_id if isinstance(seg, SegmentPayload) else session.domain.value)

    action = np.concatenate([state[1:], state[-1:]], axis=0)
    return Episode(frame_index, times, state, action, hs, tsc, verb, ver, task_names)


def write_package(
    episodes: list[tuple[Session, Path, Episode]],
    policy: ExportPolicy,
    size: tuple[int, int],
    pkg: Path,
) -> Path:
    """격리 환경의 쓰기 스크립트가 읽을 묶음 (package.json + 에피소드마다 npz)."""
    lp = policy.lerobot
    w, h = size
    if lp.max_width and w > lp.max_width:
        h, w = round(h * lp.max_width / w / 2) * 2, lp.max_width
    pkg.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = []
    for i, (session, video, ep) in enumerate(episodes):
        npz = pkg / f"episode_{i:06d}.npz"
        np.savez(
            npz,
            frame_index=ep.frame_index,
            state=ep.state,
            action=ep.action,
            hand_state=ep.hand_state,
            tool_surface_contact=ep.tool_surface_contact,
            verb=ep.verb,
            verification=ep.verification,
        )
        items.append({"session_id": session.session_id, "video": str(video), "npz": str(npz),
                      "tasks": ep.tasks})  # fmt: skip
    spec = {
        "repo_id": lp.repo_id, "fps": lp.fps, "robot_type": lp.robot_type, "vcodec": lp.vcodec,
        "width": w, "height": h, "features": features(lp, h, w), "episodes": items,
    }  # fmt: skip
    path = pkg / "package.json"
    path.write_text(json.dumps(spec, ensure_ascii=False), "utf-8")
    return path


def env_command(policy: LeRobotPolicy) -> list[str]:
    cmd = ["uv", "run", "--no-project", "--python", policy.env.python]
    for p in policy.env.packages:
        cmd += ["--with", p]
    return [*cmd, "--index", policy.env.index, "--index-strategy", "unsafe-best-match", "python"]


def run_script(root: Path, policy: LeRobotPolicy, script: str, *args: str) -> str:
    """격리 환경에서 scripts/<script>를 돌리고 표준 출력을 돌려준다 (허브에 접속하지 않는다)."""
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}
    proc = subprocess.run(
        [*env_command(policy), str(root / "scripts" / script), *args],
        cwd=root, env=env, check=True, capture_output=True, text=True,
    )  # fmt: skip
    return proc.stdout
