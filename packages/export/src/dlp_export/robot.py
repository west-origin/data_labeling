"""LeRobot 로봇 리타기팅 에피소드 (`dlp export lerobot --robot <ID>`, ADR 0047).

사람 1인칭 에피소드(`lerobot.build_episode`)의 각 프레임 시각에서 세션 월드 좌표 손 점(손목·엄지
끝·검지 끝, frame=world trajectory3d, ADR 0046)을 읽어 로봇 관절 궤적으로 옮긴다 (`dlp_retarget`).
영상은 그대로다.

LeRobot 특징 (로봇 모드):
- observation.state (float32): 로봇 관절 값 (팔 관절 + 그리퍼, rad, 이름 = URDF 관절 이름).
- action (float32): 다음 프레임 관절 값 (마지막은 그대로).
- observation.human_state (float32): 사람 모드의 observation.state 그대로 (손 2D·카메라 좌표 3D·
  작용부).
- annotation.* : 사람 모드와 같다.
- retarget.target_present (int64): 그 프레임에 손 목표(세 점 모두)가 있었나 (0/1). 없으면 관절은
  직전 해.
- retarget.reachable (int64): 허용 오차 안에 닿았나 (0/1). 0이면 관절은 가장 가까운 해(지어내지
  않음).
- retarget.position_error_m / retarget.orientation_error_rad (float32): 끝단 오차 (목표 없음 -1).
- retarget.velocity_limited (int64): 속도 한계로 잘린 프레임 (0/1).
- retarget.verification (int64): 그 프레임에 쓴 월드 손 3D 라벨의 가장 낮은 검증 등급 (없음 -1).

시각 규약 (ADR 0019): 에피소드 시각은 마스터, 3D 궤적은 그 스트림 시각(`stream_ms`)에서 읽고
`interp_max_gap_ms` 안에서만 선형 보간한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from dlp_retarget.ik import DlsIkSolver, IkSolver
from dlp_retarget.mink import MinkIkSolver
from dlp_retarget.policy import RetargetPolicy, SolverName
from dlp_retarget.retarget import HandPoints, RetargetResult, retarget_episode, retarget_version
from dlp_retarget.robot import LoadedRobot
from numpy.typing import NDArray

from dlp_export.frames import stream_ms
from dlp_export.lerobot import (
    FRAME_ARRAYS,
    Episode,
    PackageOverride,
    features,
    grade,
    interp_series,
    preferred,
    state_names,
)
from dlp_export.policy import LeRobotPolicy
from dlp_schema.labels import CoordinateFrame, LabelRecord, Trajectory3DPayload
from dlp_schema.session import Stream
from dlp_schema.trajectory_follow import CAMERA_POSE_REVIEW

# 로봇 모드에서 npz에 더하는 프레임 배열 (특징 이름 → npz 키)
ROBOT_ARRAYS = {
    "observation.human_state": "human_state",
    "retarget.target_present": "target_present",
    "retarget.reachable": "reachable",
    "retarget.position_error_m": "position_error_m",
    "retarget.orientation_error_rad": "orientation_error_rad",
    "retarget.velocity_limited": "velocity_limited",
    "retarget.verification": "retarget_verification",
}


@dataclass
class RobotEpisode:
    """에피소드 하나의 리타기팅 결과와 쓴 라벨."""

    result: RetargetResult
    used: frozenset[str]
    verification: NDArray[np.int64]  # (n,) 그 프레임 월드 손 3D의 가장 낮은 검증 등급 (없음 -1)


def make_solver(root: Path, policy: RetargetPolicy) -> IkSolver:
    """정책의 풀이기 (dls: 내장, mink: 격리 환경)."""
    if policy.solver is SolverName.MINK:
        return MinkIkSolver(root, policy.mink)
    return DlsIkSolver()


Series = list[tuple[int, NDArray[np.float64]]]


def _world_series(
    labels: list[LabelRecord], stream: Stream, entity: str, part: str
) -> list[tuple[Series, LabelRecord]]:
    """그 손·부위의 월드 좌표 궤적들 (등급 높은·새 것부터, `preferred` 순)."""
    out: list[tuple[Series, LabelRecord]] = []
    for x in preferred(labels):
        p = x.payload
        if (
            isinstance(p, Trajectory3DPayload)
            and p.frame is CoordinateFrame.WORLD
            and p.entity_id == entity
            and p.part == part
            and x.stream_id in (None, stream.stream_id)
        ):
            series = [
                (s.t_ms, np.array([s.x, s.y, s.z])) for s in sorted(p.samples, key=lambda s: s.t_ms)
            ]
            out.append((series, x))
    return out


def world_hand_points(
    stream: Stream,
    labels: list[LabelRecord],
    times_ms: NDArray[np.float64],
    robot: LoadedRobot,
    policy: RetargetPolicy,
    gap_ms: int,
) -> tuple[HandPoints, frozenset[str], NDArray[np.int64]]:
    """에피소드 프레임 시각(마스터)마다 그 로봇 팔이 따라갈 손의 월드 좌표 세 점.

    Returns:
        (손 점 (세 점 중 하나라도 없으면 그 프레임 세 점 모두 NaN), 쓴 라벨 ID, 프레임별 가장 낮은
        검증 등급 (없음 -1)).
    """
    entity = f"{robot.config.arm.hand.value}_hand"
    parts = policy.hand_parts
    roles = [
        _world_series(labels, stream, entity, part)
        for part in (parts.wrist, parts.thumb, parts.index)
    ]
    n = times_ms.size
    pts = np.full((3, n, 3), np.nan)
    ver = np.full(n, -1, np.int64)
    used: set[str] = set()
    for k, t in enumerate(times_ms):
        ts = stream_ms(stream, float(t))
        got: list[tuple[NDArray[np.float64], LabelRecord]] = []
        for series_list in roles:
            hit = next(
                (
                    (v, x)
                    for series, x in series_list
                    if (v := interp_series(series, ts, gap_ms)) is not None
                ),
                None,
            )
            if hit is None:
                break
            got.append(hit)
        if len(got) < 3:
            continue
        for r, (v, x) in enumerate(got):
            pts[r, k] = v
            used.add(x.label_id)
        ver[k] = min(grade(x) for _, x in got)
    return HandPoints(times_ms, pts[0], pts[1], pts[2]), frozenset(used), ver


def retarget_session(
    ep: Episode,
    stream: Stream,
    labels: list[LabelRecord],
    robot: LoadedRobot,
    policy: RetargetPolicy,
    lerobot: LeRobotPolicy,
    solver: IkSolver,
) -> RobotEpisode:
    """사람 에피소드 시각에서 손 점을 읽어 리타기팅한다."""
    points, used, ver = world_hand_points(
        stream, labels, ep.times_ms, robot, policy, lerobot.interp_max_gap_ms
    )
    return RobotEpisode(retarget_episode(points, robot, policy, solver), used, ver)


def robot_features(
    lerobot: LeRobotPolicy, robot: LoadedRobot, height: int, width: int
) -> dict[str, Any]:
    """로봇 모드 LeRobot `features` (모듈 설명)."""
    base = features(lerobot, height, width)
    joints = list(robot.joint_names)
    human = state_names(lerobot)
    out = {k: v for k, v in base.items() if k not in ("observation.state", "action")}
    one = ["value"]
    out.update({
        "observation.state": {"dtype": "float32", "shape": [len(joints)], "names": joints},
        "action": {"dtype": "float32", "shape": [len(joints)], "names": joints},
        "observation.human_state": {"dtype": "float32", "shape": [len(human)], "names": human},
        "retarget.target_present": {"dtype": "int64", "shape": [1], "names": one},
        "retarget.reachable": {"dtype": "int64", "shape": [1], "names": one},
        "retarget.position_error_m": {"dtype": "float32", "shape": [1], "names": one},
        "retarget.orientation_error_rad": {"dtype": "float32", "shape": [1], "names": one},
        "retarget.velocity_limited": {"dtype": "int64", "shape": [1], "names": one},
        "retarget.verification": {"dtype": "int64", "shape": [1], "names": one},
    })  # fmt: skip
    return out


def robot_arrays(ep: Episode, rob: RobotEpisode) -> dict[str, NDArray[Any]]:
    """npz에 넣을 로봇 모드 배열 (state·action은 관절 값으로 바꾼다)."""
    r = rob.result
    q = r.q.astype(np.float32)
    action = np.concatenate([q[1:], q[-1:]], axis=0)

    def col(values: NDArray[Any], dtype: type[np.generic]) -> NDArray[Any]:
        return values.reshape(-1, 1).astype(dtype)

    return {
        "state": q,
        "action": action,
        "human_state": ep.state,
        "target_present": col(r.target_present, np.int64),
        "reachable": col(r.reachable, np.int64),
        "position_error_m": col(np.nan_to_num(r.position_error_m, nan=-1.0), np.float32),
        "orientation_error_rad": col(np.nan_to_num(r.orientation_error_rad, nan=-1.0), np.float32),
        "velocity_limited": col(r.velocity_limited, np.int64),
        "retarget_verification": col(rob.verification, np.int64),
    }


def retarget_meta(
    robot: LoadedRobot, policy: RetargetPolicy, solver: IkSolver, reports: list[dict[str, Any]]
) -> dict[str, Any]:
    """meta/dlp_retarget.json·manifest details.retarget 내용 (로봇·모델 해시·판·에피소드 보고)."""
    cfg = robot.config
    return {
        "robot_id": cfg.robot_id,
        "display_name": cfg.display_name,
        "model_format": cfg.model.format,
        "model_sha256": robot.model_sha256,
        "model_license": cfg.model.license,
        "retarget_version": retarget_version(robot, policy, solver),
        "solver": f"{solver.name}-{solver.version}",
        "joint_names": list(robot.joint_names),
        "units": "rad (revolute), m (prismatic)",
        "hand": cfg.arm.hand.value,
        "placement": cfg.arm.placement.model_dump(mode="json"),
        "unreachable_policy": policy.unreachable.value,
        # 목표는 월드 좌표 손 3D라 자동 카메라 자세(사람 검수 대상 아님)에 기댄다 (ADR 0050)
        "camera_pose": CAMERA_POSE_REVIEW,
        "episodes": reports,
    }


def robot_override(
    episodes: list[tuple[Episode, RobotEpisode]], lerobot: LeRobotPolicy, robot: LoadedRobot
) -> PackageOverride:
    """`write_package`에 넘길 로봇 모드 특징·배열 (episodes는 내보낼 순서)."""
    return PackageOverride(
        features=lambda h, w: robot_features(lerobot, robot, h, w),
        arrays={**FRAME_ARRAYS, **ROBOT_ARRAYS},
        extra=[robot_arrays(ep, rob) for ep, rob in episodes],
        robot_type=robot.config.robot_id,
    )
