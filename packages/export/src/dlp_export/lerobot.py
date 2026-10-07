"""LeRobot 에피소드 (형식 v3.0, 공식 쓰기·읽기 API).

세션의 `lerobot.video_stream` 스트림(기본 바디캠) 블러본 하나가 에피소드 하나다. 이 모듈은 고정
프레임률 시각마다의 특징 표와 그 시각에 보이던 블러본 프레임(PTS 인덱스로 고름)을 정하고, 격리된
일회용 환경의 scripts/lerobot_write.py가 LeRobot 공식 API로 쓴다 (PyTorch가 작업공간 의존성과
충돌하므로, 환경은 scripts/lerobot-env/uv.lock에 고정). 쓴 뒤 scripts/lerobot_check.py가 공식
로더로 다시 읽어 확인한다.

흐름 (`runner._lerobot`이 부른다):
1. `build_episode`: 세션·스트림·라벨 → `Episode` (프레임별 특징 배열, 작업공간 numpy만 사용).
2. `write_package`: 에피소드들을 package.json + episode_XXXXXX.npz로 묶는다 (격리 환경과의 경계).
3. `run_script(..., "lerobot_write.py", pkg, dest)`: 격리 환경에서 공식 API로 데이터셋을 쓴다.
4. `run_script(..., "lerobot_check.py", dest, repo_id)`: 공식 로더로 다시 읽어 에피소드·프레임
   수 확인.

격리 환경(ADR 0021): `uv run --project scripts/lerobot-env --locked --isolated`. 전이 의존성까지
버전·해시를 uv.lock에 고정하고 실행마다 일회용 가상 환경을 쓴다.
허브에는 접속하지 않는다 (오프라인).

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
- annotation.verification (int64): `GROUPS` 묶음 [손 2D, 손 3D, 손 상태, 행동, 도구-표면 접촉, 작업]
  마다 그 시각에 쓴 라벨의 가장 낮은 검증 등급 (미검수 0, 표본 검증 1, 사람 승인 2,
  사람 수정·작성 3, 라벨 없음 -1).
- task: 그 시각의 작업(task 수준 구간) ID, 없으면 세션 도메인.
어휘(번호 → ID)는 meta/dlp_vocab.json에 둔다.

시각 규약 (ADR 0019): 에피소드 시각은 마스터 시각이다. 공간 라벨(키포인트·3D 궤적)은 `stream_ms`로
바꾼 스트림 시각에서 읽고, 손 상태·행동·관계·작업 구간은 마스터 시각에서 읽는다.
`Episode.frame_index`(블러본 프레임 번호)는 쓰기 스크립트에만 넘기고 데이터셋에 저장하지 않는다.
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
from dlp_export.pseudonym import Pseudonymizer
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

# 특징 배열의 손 순서 (왼손이 앞)
HANDS = (Hand.LEFT, Hand.RIGHT)
# 모델 라벨 검증 상태 → 등급 (높을수록 믿을 만함). 사람이 만든 라벨은 `grade`에서 3
GRADE = {
    VerificationState.UNREVIEWED: 0,
    VerificationState.SAMPLE_VERIFIED: 1,
    VerificationState.HUMAN_APPROVED: 2,
    VerificationState.HUMAN_CORRECTED: 3,
}
# annotation.verification 열 순서 (열 번호는 build_episode의 worse(k, g, …) 호출과 맞아야 한다)
GROUPS = ("hands_2d", "hands_3d", "hand_state", "actions", "tool_contact", "task")


def grade(x: LabelRecord) -> int:
    """라벨의 검증 등급 (0~3). 사람이 만든 라벨은 상태와 관계없이 3."""
    return 3 if x.provenance.source is Source.HUMAN else GRADE[x.verification.state]


def preferred(labels: list[LabelRecord]) -> list[LabelRecord]:
    """겹치는 라벨 중 먼저 쓸 순서: 검증 등급이 높은 것, 같으면 새 것.

    마지막 정렬 키(라벨 ID)는 같은 입력이면 같은 결과가 나오게 하는 동점 처리다.
    """
    return sorted(labels, key=lambda x: (-grade(x), -x.created_at.timestamp(), x.label_id))


@dataclass(frozen=True)
class Vocab:
    """정수 특징의 어휘 (번호 = 튜플 안 위치). 온톨로지 버전이 같으면 같다."""

    verbs: tuple[str, ...]  # 동사 ID (이름순)
    contact_target_kinds: tuple[str, ...]  # 접촉 대상 종류 (온톨로지 순서)
    grasp_types: tuple[str, ...]  # 파지 유형 (온톨로지 순서)
    hand_roles: tuple[str, ...]  # 손 역할 (온톨로지 순서)
    working_parts: frozenset[str]  # 도구 작용부 이름 (도구-표면 접촉·작용부 3D에 쓴다)
    tasks: tuple[str, ...]  # 에피소드 task 문자열 (작업 ID와 도메인)

    @classmethod
    def from_ontology(cls, o: Ontology) -> Vocab:
        """온톨로지에서 어휘를 만든다 (모든 도구 객체의 작용부를 모은다)."""
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
        """meta/dlp_vocab.json 내용 (구매자가 정수 특징을 ID로 되돌릴 때 쓴다)."""
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
    """observation.state 열 이름 (열 순서 정의).

    [왼손 21관절*(x, y, v), 오른손 21관절*(x, y, v),
     왼손 3D 점*(x3d, y3d, z3d, present), 오른손 3D 점*(…),
     왼손 도구 작용부(…), 오른손 도구 작용부(…)]
    길이 = 2*63 + 2*4*len(hand_points_3d) + 2*4.
    """
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


def video_key(policy: LeRobotPolicy) -> str:
    """에피소드 영상 특징 키 (예: observation.images.bodycam)."""
    return f"observation.images.{policy.video_stream}"


def features(policy: LeRobotPolicy, height: int, width: int) -> dict[str, Any]:
    """LeRobot `features` 정의 (dtype·shape·names). 쓰기 스크립트가 그대로 `create`에 넘긴다.

    Args:
        policy: LeRobot 정책.
        height, width: 에피소드 영상 크기 (`write_package`에서 max_width로 줄인 뒤).
    """
    n = len(state_names(policy))
    hands = [h.value for h in HANDS]
    return {
        video_key(policy): {
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
        "annotation.verification": {
            "dtype": "int64", "shape": [len(GROUPS)], "names": list(GROUPS),
        },
    }  # fmt: skip


def _interp(
    series: list[tuple[int, NDArray[np.float64]]], t: float, max_gap: int
) -> NDArray[np.float64] | None:
    """(시각, 값) 목록에서 t의 값.

    양쪽 표본 간격이 max_gap보다 크면 None. 값의 NaN은 그대로 퍼진다.

    Args:
        series: 시각(ms) 오름차순 표본. 값은 같은 모양의 배열.
        t: 읽을 시각 (series와 같은 시각 체계, 보통 스트림 시각 ms).
        max_gap: 보간을 허용하는 최대 표본 간격 (ms, `interp_max_gap_ms`).

    Returns:
        t와 같은 표본이 있으면 그 값, 두 표본 사이면 선형 보간 값, 범위 밖·간격 초과면 None.
    """
    if not series:
        return None
    times = [s[0] for s in series]
    i = int(np.searchsorted(times, t))
    if i < len(times) and times[i] == t:
        return series[i][1]
    # 첫 표본 앞이나 마지막 표본 뒤는 외삽하지 않는다
    if i == 0 or i == len(times):
        return None
    (t0, v0), (t1, v1) = series[i - 1], series[i]
    if t1 - t0 > max_gap:
        return None
    w = (t - t0) / (t1 - t0)
    return v0 * (1 - w) + v1 * w


def _covering(labels: list[LabelRecord], t: float) -> list[LabelRecord]:
    """시각 t(마스터 ms)를 덮는 구간 라벨 (반열린 [시작, 끝), 길이 0이면 그 시각에서만).

    입력 순서(보통 `preferred` 순)를 유지하므로 첫 원소가 먼저 쓸 라벨이다.
    """
    return [x for x in labels if x.t_start_ms <= t < x.t_end_ms or x.t_start_ms == x.t_end_ms == t]


@dataclass
class Episode:
    """에피소드 하나의 프레임별 특징 (n = 프레임 수)."""

    frame_index: NDArray[np.int64]  # 블러본 표시 순서 프레임 번호 (쓰기 전용, 저장하지 않는다)
    times_ms: NDArray[np.float64]  # 마스터 시각
    state: NDArray[np.float32]  # (n, len(state_names))
    action: NDArray[np.float32]  # (n, len(state_names)): 다음 프레임 state
    hand_state: NDArray[np.int64]  # (n, 8): 손마다 [접촉, 대상 종류, 파지, 역할]
    tool_surface_contact: NDArray[np.int64]  # (n, 2)
    verb: NDArray[np.int64]  # (n, 2)
    verification: NDArray[np.int64]  # (n, len(GROUPS))
    tasks: list[str]  # 길이 n, 프레임별 task 문자열
    used: frozenset[str] = frozenset()  # 실제로 어느 프레임 특징에 들어간 라벨 ID


def build_episode(
    session: Session,
    stream: Stream,
    index: PtsIndex,
    size: tuple[int, int],
    labels: list[LabelRecord],
    vocab: Vocab,
    policy: LeRobotPolicy,
) -> Episode:
    """세션 하나의 에피소드 특징을 만든다 (순수 함수, 파일·DB를 건드리지 않는다).

    Args:
        session: 세션 (도메인을 task 기본값으로 쓴다).
        stream: 에피소드 영상 스트림 (바디캠). 시각 변환에 오프셋·클럭 배율을 쓴다.
        index: 그 스트림 블러본의 PTS 인덱스.
        size: 블러본 원본 크기 (가로, 세로) 픽셀. 2D 관절을 0~1로 정규화하는 데 쓴다.
        labels: 내보낼 라벨 (`select_labels` 결과, 다른 스트림 라벨이 섞여 있어도 된다).
        vocab: 어휘.
        policy: LeRobot 정책 (fps, interp_max_gap_ms, hand_points_3d).

    Returns:
        `Episode`. 프레임 수는 최소 1.

    겹침 규칙: 같은 손·묶음에 라벨이 여럿이면 `preferred` 순(검증 등급 높은 것, 새 것)으로
    그 시각에 값이 있는 첫 라벨을 쓴다.
    """
    w, h = size
    gap = policy.interp_max_gap_ms
    # 에피소드 시각: 블러본 첫 프레임부터 끝까지 1/fps 간격 (마스터 시각)
    start = stream.to_master_ms(float(index.ms[0]))
    end = stream.to_master_ms(index.duration_ms)
    n = max(math.floor((end - start) * policy.fps / 1000), 1)
    times = start + np.arange(n) * 1000.0 / policy.fps
    # 각 시각에 화면에 보이던 프레임 (그 시각 이전의 마지막 PTS, VFR에서도 맞다)
    frame_index = np.array([index.frame_at(stream_ms(stream, t)) for t in times], dtype=np.int64)

    # 공간 라벨(키포인트·3D 궤적)은 스트림 시각, 손 상태·행동·관계·작업은 마스터 시각 (ADR 0019)
    # 손 2D: 스트림의 hand21 트랙 (같은 손에 트랙이 여럿이면 시각마다 등급 높은·새 것부터)
    kp2d: dict[Hand, list[tuple[list[tuple[int, NDArray[np.float64]]], LabelRecord]]] = {
        h_: [] for h_ in HANDS
    }
    for x in preferred(labels):
        p = x.payload
        if (
            isinstance(p, KeypointTrackPayload)
            and p.skeleton == "hand21"
            and p.hand
            and x.stream_id == stream.stream_id
        ):
            # 키프레임마다 (21, 3) 배열 [x/가로, y/세로, v]. 보이지 않는(v=0) 관절은 NaN으로 두어
            # 보간 때 "라벨 없음"이 퍼지게 한다
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
    # 3D 궤적: (개체, 부분) → 이 스트림의 궤적들 (등급 높은·새 것부터, 시각마다 값이 있는 첫 것)
    # stream_id가 없는 궤적도 받는다 (기준 스트림 시각으로 본다)
    traj: dict[
        tuple[str, str | None], list[tuple[list[tuple[int, NDArray[np.float64]]], LabelRecord]]
    ] = {}
    for x in preferred(labels):
        p = x.payload
        if isinstance(p, Trajectory3DPayload) and x.stream_id in (None, stream.stream_id):
            traj.setdefault((p.entity_id, p.part), []).append(
                (
                    [
                        (s.t_ms, np.array([s.x, s.y, s.z]))
                        for s in sorted(p.samples, key=lambda s: s.t_ms)
                    ],
                    x,
                )
            )

    def traj_at(
        entity: str, part: str | None, ts: float
    ) -> tuple[NDArray[np.float64], LabelRecord] | None:
        """개체·부분의 스트림 시각 ts 3D 좌표와 그 값을 준 라벨 (없으면 None)."""
        for series, x in traj.get((entity, part), []):
            v = _interp(series, ts, gap)
            if v is not None:
                return v, x
        return None

    # 시간 구간 라벨 (마스터 시각). 모두 preferred 순이라 _covering의 첫 원소를 쓴다
    ordered = preferred(labels)
    hand_states = [x for x in ordered if isinstance(x.payload, HandStatePayload)]
    actions = [x for x in ordered if isinstance(x.payload, ActionPayload)]
    # 도구-표면 접촉: 주어가 도구 작용부인 contact 관계 (dlp_relations가 도출)
    tool_contacts = [
        x for x in ordered
        if isinstance(x.payload, RelationPayload)
        and x.payload.predicate is RelationPredicate.CONTACT
        and x.payload.subject_part in vocab.working_parts
    ]  # fmt: skip
    tasks_lbl = [
        x for x in ordered
        if isinstance(x.payload, SegmentPayload) and x.payload.level is SegmentLevel.TASK
    ]  # fmt: skip

    dim = len(state_names(policy))
    # 값이 없으면 state는 0(present·v 열도 0), 정수 특징은 -1
    state = np.zeros((n, dim), np.float32)
    hs = np.full((n, 8), -1, np.int64)
    tsc = np.full((n, 2), -1, np.int64)
    verb = np.full((n, 2), -1, np.int64)
    ver = np.full((n, len(GROUPS)), -1, np.int64)
    task_names: list[str] = []
    used: set[str] = set()
    # state 안 블록 크기: 손 하나의 2D(21관절*3), 손 하나의 3D 점(점 수*4)
    n2d, n3d = 21 * 3, len(policy.hand_points_3d) * 4

    def worse(k: int, g: int, x: LabelRecord) -> None:
        """프레임 k의 묶음 g에 라벨 x를 썼다 (검증 등급을 낮추고, 쓴 라벨로 센다)."""
        used.add(x.label_id)
        cur = ver[k, g]
        ver[k, g] = grade(x) if cur < 0 else min(cur, grade(x))

    for k, t in enumerate(times):
        ts = stream_ms(stream, t)  # 공간 라벨을 읽을 스트림 시각
        for hi, hand in enumerate(HANDS):
            # 2D
            for series, x in kp2d[hand]:
                v = _interp(series, ts, gap)
                if v is not None:
                    # 양쪽 키프레임에 다 라벨이 있는 관절만 값이 있다. 보임 정도는 둘 중 낮은 쪽
                    # (보간한 v의 내림: 끝점에서는 그 값, 사이에서는 작은 값)
                    labeled = ~np.isnan(v[:, 0])
                    block = np.where(labeled[:, None], np.nan_to_num(v), 0.0)
                    block[:, 2] = np.where(labeled, np.floor(np.nan_to_num(v[:, 2])), 0.0)
                    state[k, hi * n2d : (hi + 1) * n2d] = block.reshape(-1)
                    worse(k, 0, x)
                    break
            # 3D 손 점 (개체 ID 규약: "<left|right>_hand", 부분 = hand_points_3d 이름)
            base = 2 * n2d + hi * n3d
            for pi, part in enumerate(policy.hand_points_3d):
                got = traj_at(f"{hand.value}_hand", part, ts)
                if got is not None:
                    state[k, base + pi * 4 : base + pi * 4 + 4] = [*got[0], 1.0]
                    worse(k, 1, got[1])
            # 손 상태와 쥔 도구
            cur = [x for x in _covering(hand_states, t) if x.payload.hand is hand]  # type: ignore[union-attr]
            tool_id: str | None = None
            if cur:
                x = cur[0]
                p = x.payload
                assert isinstance(p, HandStatePayload)
                kind = p.contact_target_kind
                # 어휘에 없는 값은 -1 (온톨로지와 라벨이 어긋난 경우)
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
            # 쥔 도구가 있을 때만: 도구-표면 접촉(0/1)과 작용부 3D. 없으면 -1·0 그대로
            tip_base = 2 * n2d + 2 * n3d + hi * 4
            if tool_id is not None:
                touching = [
                    x for x in _covering(tool_contacts, t)
                    if x.payload.subject_id == tool_id  # type: ignore[union-attr]
                ]  # fmt: skip
                tsc[k, hi] = int(bool(touching))
                if touching:
                    worse(k, 4, touching[0])
                # 작용부가 여럿이면 이름순으로 값이 있는 첫 작용부
                for part in sorted(p for e, p in traj if e == tool_id and p in vocab.working_parts):
                    got = traj_at(tool_id, part, ts)
                    if got is not None:
                        state[k, tip_base : tip_base + 4] = [*got[0], 1.0]
                        worse(k, 1, got[1])
                        break
            # 행동
            acts = [x for x in _covering(actions, t) if x.payload.hand is hand]  # type: ignore[union-attr]
            if acts:
                p = acts[0].payload
                assert isinstance(p, ActionPayload)
                verb[k, hi] = vocab.verbs.index(p.verb) if p.verb in vocab.verbs else -1
                worse(k, 3, acts[0])
        # 작업 (손과 무관): task 수준 구간의 ref_id, 없으면 세션 도메인
        cover = _covering(tasks_lbl, t)
        if cover:
            worse(k, 5, cover[0])
        seg = cover[0].payload if cover else None
        task_names.append(seg.ref_id if isinstance(seg, SegmentPayload) else session.domain.value)

    # action = 다음 프레임 state (마지막 프레임은 자기 자신)
    action = np.concatenate([state[1:], state[-1:]], axis=0)
    return Episode(
        frame_index, times, state, action, hs, tsc, verb, ver, task_names, frozenset(used)
    )


def write_package(
    episodes: list[tuple[Session, Path, Episode]],
    policy: ExportPolicy,
    size: tuple[int, int],
    pkg: Path,
    ids: Pseudonymizer,
) -> Path:
    """격리 환경의 쓰기 스크립트가 읽을 묶음 (package.json + 에피소드마다 npz).

    격리 환경은 작업공간 패키지를 import할 수 없으므로, 필요한 것을 모두 이 묶음에 담는다.

    Args:
        episodes: (세션, 블러본 로컬 경로, 에피소드) 목록. 순서가 에피소드 번호가 된다.
        policy: 내보내기 정책 (`lerobot` 절을 쓴다).
        size: 블러본 크기 (가로, 세로). `max_width`보다 넓으면 비율을 유지해 줄이고 세로는 짝수로
            맞춘다 (h264 요구).
        pkg: 묶음을 쓸 디렉터리 (만든다).
        ids: 가명 함수 (package.json의 session_id는 가명).

    Returns:
        package.json 경로.
    """
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
        items.append({"session_id": ids.session(session.session_id), "video": str(video),
                      "npz": str(npz), "tasks": ep.tasks})  # fmt: skip
    spec = {
        "repo_id": lp.repo_id, "fps": lp.fps, "robot_type": lp.robot_type, "vcodec": lp.vcodec,
        "width": w, "height": h, "features": features(lp, h, w), "episodes": items,
        "video_key": video_key(lp),
    }  # fmt: skip
    path = pkg / "package.json"
    path.write_text(json.dumps(spec, ensure_ascii=False), "utf-8")
    return path


def env_command(root: Path, policy: LeRobotPolicy) -> list[str]:
    """격리 환경 실행 명령.

    잠금 파일(전이 의존성까지 버전·해시 고정)을 그대로 따르고(--locked: pyproject와 어긋나면
    실패), 실행마다 일회용 가상 환경을 쓴다(--isolated). PyTorch만 CPU판 인덱스에서 받는다
    (explicit 인덱스).

    Args:
        root: 저장소 루트.
        policy: LeRobot 정책 (`env.project`, `env.python`).

    Returns:
        뒤에 스크립트 경로와 인자를 붙여 실행할 명령 목록.
    """
    return [
        "uv", "run", "--project", str(root / policy.env.project), "--locked", "--isolated",
        "--python", policy.env.python, "python",
    ]  # fmt: skip


def run_script(root: Path, policy: LeRobotPolicy, script: str, *args: str) -> str:
    """격리 환경에서 scripts/<script>를 돌리고 표준 출력을 돌려준다 (허브에 접속하지 않는다).

    HF_HUB_OFFLINE·HF_DATASETS_OFFLINE을 켜 Hugging Face 허브 접속을 막는다.
    처음 실행하면 uv가 잠금 파일대로 환경(약 1.5 GB)을 받으므로 오래 걸린다.

    Raises:
        subprocess.CalledProcessError: 스크립트가 0이 아닌 코드로 끝났을 때 (stderr 포함).
    """
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}
    proc = subprocess.run(
        [*env_command(root, policy), str(root / "scripts" / script), *args],
        cwd=root, env=env, check=True, capture_output=True, text=True,
    )  # fmt: skip
    return proc.stdout
