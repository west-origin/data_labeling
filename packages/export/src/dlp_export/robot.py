"""LeRobot 로봇 리타기팅 에피소드 (`dlp export lerobot --robot <ID>`, ADR 0047·0068).

사람 1인칭 에피소드(`lerobot.build_episode`)의 각 프레임 시각에서 세션 월드 좌표 손 점(손목·엄지
끝·검지 끝, frame=world trajectory3d, ADR 0046)을 읽어 로봇 관절 궤적으로 옮긴다 (`dlp_retarget`).
세 점의 부위 이름은 retarget.yaml `hand_parts`에서 읽고, 로봇 설정의 팔(`arm.hand`) 쪽 손만 쓴다.
영상은 그대로다. 정책이 버린(drop) 에피소드는 쓰지 않고 보고에만 남긴다 (`runner._RobotRun`).

로봇 종류는 로봇 설정이 정한다 (`load_retargeter`): 평행 그리퍼 로봇은 아래 특징 그대로이고,
다지 손 로봇(ADR 0068)은 손목과 대응한 손가락 끝 점(손목 부위는 `hand_parts.wrist`, 손가락 끝
부위는 로봇 설정의 `fingers[].human_part`)을 읽어 팔 관절 + 손가락 관절로 옮기고 아래 "다지 손
로봇" 특징을 더한다. 평행 그리퍼 로봇의 결과는 다지 손 지원 전과 같다.

LeRobot 특징 (로봇 모드):
- observation.state (float32): 로봇 관절 값 (팔 관절 + 그리퍼, 이름 = URDF 관절 이름). 단위는 회전
  관절 rad, 직선 관절 m다 (meta/dlp_retarget.json `units`).
- action (float32): 다음 프레임 관절 값 (마지막은 그대로).
- observation.human_state (float32): 사람 모드의 observation.state 그대로 (손 2D·카메라 좌표 3D·
  작용부).
- annotation.* : 사람 모드와 같다.
- retarget.target_present (int64): 그 프레임에 손 목표(세 점 모두)가 있었나 (0/1). 없으면 관절은
  직전 프레임 값 (첫 목표 전에는 시작 자세 = 관절 범위 가운데, 닫힌 그리퍼).
- retarget.reachable (int64): 허용 오차 안에 닿았나 (0/1). 기준은 meta/dlp_retarget.json `ik`
  (위치 position_tolerance_m, 회전 orientation_tolerance_rad, orientation_mode)와
  `reachable_definition`이다. 0이면 관절은 가장 가까운 해(지어내지 않음).
- retarget.position_error_m (float32): 끝단 위치 오차 (목표 없음 -1).
- retarget.orientation_error_rad (float32): 끝단 회전 오차. 뜻은 orientation_mode에 따른다
  (`orientation_error_definition`: approach는 접근 축 각, full은 접근·닫힘 축 각의 큰 쪽). 목표가
  없거나 none 모드(재지 않음)면 -1.
- retarget.velocity_limited (int64): 팔·그리퍼 관절 중 하나라도 속도 한계로 잘린 프레임 (0/1).
- retarget.verification (int64): 그 프레임에 쓴 월드 손 3D 라벨의 가장 낮은 검증 등급 (없음 -1).

다지 손 로봇 (ADR 0068): 위 특징의 뜻이 아래처럼 바뀌고 손가락별 특징 셋이 더 있다 (열 순서와
이름 = 로봇 설정의 손가락 대응 순서, 사람 손가락 끝 부위 이름).
- observation.state·action: 팔 관절 + 손가락 관절 (설정의 팔 관절 순서, 그 뒤 손가락 대응 순서로
  손가락마다 손목 쪽 관절부터). 손목 링크 아래의 관절 중 대응하지 않은 것은 싣지 않는다
  (meta `dexterous_hand.unmapped_joints`).
- retarget.target_present: 손목 목표가 있었나. 없으면 모든 관절이 직전 프레임 값이다.
- retarget.reachable: 손목과 목표가 있는 손가락 끝이 모두 허용 오차 안인가 (meta
  `reachable_definition`). 목표가 없는 손가락은 판정에 넣지 않는다 (아래 특징으로 가린다).
- retarget.position_error_m: 손목 위치 오차. retarget.orientation_error_rad: 늘 -1 (방향 목표가
  없어 재지 않는다). retarget.velocity_limited: 팔·손가락 관절 중 하나라도 잘린 프레임.
- retarget.verification: 그 프레임에 쓴 손목·손가락 끝 월드 3D 라벨의 가장 낮은 검증 등급.
- retarget.finger_target_present (int64, 손가락마다): 그 손가락 끝 목표가 있었나 (0/1). 0이면 그
  손가락의 관절은 직전 프레임 값이다 (한 번도 목표가 없었으면 시작 자세 = 관절 범위 가운데).
- retarget.finger_reachable (int64, 손가락마다): 목표가 있고 손가락 끝 위치 오차가
  `dexterous_hand.fingertip_tolerance_m` 안인가 (0/1).
- retarget.finger_position_error_m (float32, 손가락마다): 손가락 끝 위치 오차 (목표 없음 -1).

시각 규약 (ADR 0019): 에피소드 시각은 마스터, 3D 궤적은 그 스트림 시각(`stream_ms`)에서 읽고
`interp_max_gap_3d_ms` 안에서만 선형 보간한다.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from dlp_retarget.hand_ik import HandIkSolver, make_hand_solver
from dlp_retarget.ik import DlsIkSolver, IkSolver
from dlp_retarget.mink import MinkIkSolver
from dlp_retarget.policy import RetargetPolicy, SolverName
from dlp_retarget.retarget import (
    FingerPoints,
    HandPoints,
    HandRetargetResult,
    RetargetResult,
    finger_parts,
    retarget_episode,
    retarget_hand_episode,
    retarget_version,
)
from dlp_retarget.robot import AnyRobot, LoadedDexterousRobot, LoadedRobot, load_any_robot
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

# retarget.orientation_error_rad의 뜻 (orientation_mode 값 → 구매자용 영어 설명, 계약의 의미)
ORIENTATION_ERROR_DEFINITION = {
    "none": "not measured (-1); only the position is matched",
    "approach": "angle between the tool approach axis and the hand approach axis "
    "(wrist to grasp point); rotation about the approach axis (roll) is not evaluated",
    "full": "max of the approach-axis angle and the closing-axis angle "
    "(thumb to index direction); roll is evaluated",
}
# retarget.reachable의 뜻 (구매자용)
REACHABLE_DEFINITION = (
    "1 when position error <= ik.position_tolerance_m and (unless orientation_mode is none) "
    "orientation error <= ik.orientation_tolerance_rad, after the velocity limit "
    "(URDF velocity x ik.velocity_scale) has been applied"
)

# 다지 손 로봇의 retarget.reachable·retarget.finger_reachable의 뜻 (구매자용, ADR 0068)
HAND_REACHABLE_DEFINITION = (
    "1 when wrist position error <= ik.position_tolerance_m and every fingertip that has a target "
    "in that frame is within dexterous_hand.fingertip_tolerance_m, after the velocity limit "
    "(URDF velocity x ik.velocity_scale) has been applied; fingers without a target "
    "(retarget.finger_target_present = 0) are not evaluated and keep their previous joint values"
)
FINGER_REACHABLE_DEFINITION = (
    "per finger: 1 when the fingertip has a target in that frame and its position error <= "
    "dexterous_hand.fingertip_tolerance_m"
)
# 다지 손 로봇의 방향 오차 (재지 않는다)
HAND_ORIENTATION_ERROR_DEFINITION = (
    "not measured (-1); only the wrist and fingertip positions are matched, so the palm "
    "orientation and the finger knuckle angles can differ from the human hand"
)

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
# 다지 손 로봇이 더 싣는 손가락별 프레임 배열 (특징 이름 → npz 키, ADR 0068)
HAND_ARRAYS = {
    "retarget.finger_target_present": "finger_target_present",
    "retarget.finger_reachable": "finger_reachable",
    "retarget.finger_position_error_m": "finger_position_error_m",
}


@dataclass
class RobotEpisode:
    """에피소드 하나의 리타기팅 결과와 쓴 라벨 (결과는 로봇 종류에 따라 평행 그리퍼·다지 손)."""

    result: RetargetResult | HandRetargetResult
    used: frozenset[str]
    verification: NDArray[np.int64]  # (n,) 그 프레임 월드 손 3D의 가장 낮은 검증 등급 (없음 -1)


def make_solver(root: Path, policy: RetargetPolicy) -> IkSolver:
    """정책의 팔 풀이기 (dls: 내장, mink: 격리 환경). 다지 손 로봇은 `load_retargeter`가
    `hand_ik.make_hand_solver`로 따로 고른다."""
    if policy.solver is SolverName.MINK:
        return MinkIkSolver(root, policy.mink)
    return DlsIkSolver()


Series = list[tuple[int, NDArray[np.float64]]]
# 프레임 시각에 값이 있는 궤적 표본: (보간한 점, 그 값을 준 라벨)
Hit = tuple[NDArray[np.float64], LabelRecord]


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

    부위마다 `preferred` 순 궤적 중 그 시각에 값이 있는 첫 궤적을 쓴다. 보간 한도는 `gap_ms`
    (export.yaml `lerobot.interp_max_gap_3d_ms`)다. 좌표 단위는 세션 월드 좌표계의 m다.

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
        got: list[Hit] = []
        for series_list in roles:
            hit = _first_hit(series_list, ts, gap_ms)
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


def _first_hit(series_list: list[tuple[Series, LabelRecord]], ts: float, gap_ms: int) -> Hit | None:
    """`preferred` 순 궤적 중 스트림 시각 ts에 값이 있는 첫 궤적의 (점, 라벨). 없으면 None."""
    return next(
        (
            (v, x)
            for series, x in series_list
            if (v := interp_series(series, ts, gap_ms)) is not None
        ),
        None,
    )


def world_finger_points(
    stream: Stream,
    labels: list[LabelRecord],
    times_ms: NDArray[np.float64],
    robot: LoadedDexterousRobot,
    policy: RetargetPolicy,
    gap_ms: int,
) -> tuple[FingerPoints, frozenset[str], NDArray[np.int64]]:
    """에피소드 프레임 시각(마스터)마다 다지 손 로봇이 따라갈 손의 월드 좌표 손목·손가락 끝 점.

    부위는 `finger_parts`(손목 = retarget.yaml `hand_parts.wrist`, 손가락 끝 = 로봇 설정의
    `human_part`, 대응 순서)다. 궤적 고르기와 보간 한도는 `world_hand_points`와 같다. 평행 그리퍼와
    달리 점마다 따로 비운다: 손목이 없는 프레임은 모든 점이 NaN이고(목표 없음), 손목은 있고 손가락
    끝만 없으면 그 손가락 끝만 NaN이다 (그 손가락만 목표 없음).

    Returns:
        (손 점, 쓴 라벨 ID, 프레임별 가장 낮은 검증 등급 (손목이 없으면 -1)). 손목이 없는 프레임의
        손가락 끝 궤적은 쓴 라벨에 넣지 않는다.

    Raises:
        RobotError: 손가락의 human_part가 손목 부위와 같을 때 (`finger_parts`).
    """
    entity = f"{robot.config.arm.hand.value}_hand"
    roles = [_world_series(labels, stream, entity, part) for part in finger_parts(robot, policy)]
    n = times_ms.size
    wrist = np.full((n, 3), np.nan)
    tips = np.full((n, len(roles) - 1, 3), np.nan)
    ver = np.full(n, -1, np.int64)
    used: set[str] = set()
    for k, t in enumerate(times_ms):
        ts = stream_ms(stream, float(t))
        hits = [_first_hit(series_list, ts, gap_ms) for series_list in roles]
        if hits[0] is None:
            continue
        wrist[k] = hits[0][0]
        got = [hits[0]]
        for i, hit in enumerate(hits[1:]):
            if hit is not None:
                tips[k, i] = hit[0]
                got.append(hit)
        used |= {x.label_id for _, x in got}
        ver[k] = min(grade(x) for _, x in got)
    return FingerPoints(times_ms, wrist, tips), frozenset(used), ver


def retarget_session(
    ep: Episode,
    stream: Stream,
    labels: list[LabelRecord],
    robot: LoadedRobot,
    policy: RetargetPolicy,
    lerobot: LeRobotPolicy,
    solver: IkSolver,
) -> RobotEpisode:
    """사람 에피소드 시각에서 손 점을 읽어 리타기팅한다 (평행 그리퍼 로봇)."""
    points, used, ver = world_hand_points(
        stream, labels, ep.times_ms, robot, policy, lerobot.interp_max_gap_3d_ms
    )
    return RobotEpisode(retarget_episode(points, robot, policy, solver), used, ver)


def retarget_hand_session(
    ep: Episode,
    stream: Stream,
    labels: list[LabelRecord],
    robot: LoadedDexterousRobot,
    policy: RetargetPolicy,
    lerobot: LeRobotPolicy,
    solver: HandIkSolver,
) -> RobotEpisode:
    """사람 에피소드 시각에서 손목·손가락 끝 점을 읽어 리타기팅한다 (다지 손 로봇, ADR 0068)."""
    points, used, ver = world_finger_points(
        stream, labels, ep.times_ms, robot, policy, lerobot.interp_max_gap_3d_ms
    )
    return RobotEpisode(retarget_hand_episode(points, robot, policy, solver), used, ver)


# 에피소드 하나를 리타기팅하는 함수 (로봇·정책·풀이기를 묶어 둔 것)
EpisodeRetarget = Callable[[Episode, Stream, list[LabelRecord], LeRobotPolicy], RobotEpisode]


@dataclass(frozen=True)
class Retargeter:
    """로봇 하나의 리타기팅 준비: 확인한 로봇, 풀이기, 에피소드 하나를 옮기는 함수.

    로봇 종류에 따라 풀이기와 옮기는 함수가 짝으로 정해진다 (평행 그리퍼: `retarget_session` +
    팔 풀이기, 다지 손: `retarget_hand_session` + 다지 손 풀이기). solver는 판 문자열에 쓴다.
    """

    robot: AnyRobot
    solver: IkSolver | HandIkSolver
    episode: EpisodeRetarget


def load_retargeter(root: Path, robot_id: str, policy: RetargetPolicy) -> Retargeter:
    """로봇(모델 sha256 확인)을 읽고 그 종류에 맞는 풀이기와 에피소드 함수를 묶는다.

    Raises:
        RobotError: 로봇 설정·모델 확인 실패, 다지 손 로봇인데 정책 풀이기가 mink일 때
            (`hand_ik.make_hand_solver`), 손가락 부위가 손목 부위와 같을 때 (`finger_parts`).
    """
    robot = load_any_robot(root, robot_id)
    if isinstance(robot, LoadedDexterousRobot):
        hand, hand_solver = robot, make_hand_solver(policy)
        finger_parts(hand, policy)  # 내보내기를 시작하기 전에 부위 설정을 확인한다

        def hand_episode(
            ep: Episode, stream: Stream, labels: list[LabelRecord], lerobot: LeRobotPolicy
        ) -> RobotEpisode:
            return retarget_hand_session(ep, stream, labels, hand, policy, lerobot, hand_solver)

        return Retargeter(robot, hand_solver, hand_episode)
    arm, solver = robot, make_solver(root, policy)

    def arm_episode(
        ep: Episode, stream: Stream, labels: list[LabelRecord], lerobot: LeRobotPolicy
    ) -> RobotEpisode:
        return retarget_session(ep, stream, labels, arm, policy, lerobot, solver)

    return Retargeter(robot, solver, arm_episode)


def robot_features(
    lerobot: LeRobotPolicy, robot: AnyRobot, height: int, width: int
) -> dict[str, Any]:
    """로봇 모드 LeRobot `features` (모듈 설명). 다지 손 로봇은 손가락별 특징 셋을 더한다."""
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
    if isinstance(robot, LoadedDexterousRobot):
        fingers = [f.config.human_part for f in robot.fingers]
        per_finger = {"shape": [len(fingers)], "names": fingers}
        out.update({
            "retarget.finger_target_present": {"dtype": "int64", **per_finger},
            "retarget.finger_reachable": {"dtype": "int64", **per_finger},
            "retarget.finger_position_error_m": {"dtype": "float32", **per_finger},
        })  # fmt: skip
    return out


def robot_arrays(ep: Episode, rob: RobotEpisode) -> dict[str, NDArray[Any]]:
    """npz에 넣을 로봇 모드 배열 (state·action은 관절 값으로 바꾼다).

    다지 손 결과면 손가락별 배열(`HAND_ARRAYS`) (n, F)를 더한다 (오차의 목표 없음은 -1).
    """
    r = rob.result
    q = r.q.astype(np.float32)
    action = np.concatenate([q[1:], q[-1:]], axis=0)

    def col(values: NDArray[Any], dtype: type[np.generic]) -> NDArray[Any]:
        """(n,) 배열을 LeRobot 특징 모양 (n, 1)과 dtype으로 바꾼다."""
        return values.reshape(-1, 1).astype(dtype)

    out = {
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
    if isinstance(r, HandRetargetResult):
        errors = np.nan_to_num(r.finger_position_error_m, nan=-1.0)
        out.update(
            finger_target_present=r.finger_target_present.astype(np.int64),
            finger_reachable=r.finger_reachable.astype(np.int64),
            finger_position_error_m=errors.astype(np.float32),
        )
    return out


def retarget_meta(
    robot: AnyRobot,
    policy: RetargetPolicy,
    solver: IkSolver | HandIkSolver,
    reports: list[dict[str, Any]],
) -> dict[str, Any]:
    """meta/dlp_retarget.json·manifest details.retarget 내용.

    로봇·모델 해시·판·에피소드 보고와 함께, 도달·오차를 해석하는 기준(`ik`: 허용 오차·
    orientation_mode·velocity_scale 등, `hand_parts`: 손 점 역할 → 부위, 두 정의 문자열)을 싣는다.
    모델 파일 경로·작업자·장소 ID는 싣지 않는다. 다지 손 로봇은 `_hand_meta`가 손가락 대응·손
    배율·허용 오차·손가락별 정의를 싣는다 (에피소드 보고에 손가락별 통계가 있다).
    """
    if isinstance(robot, LoadedDexterousRobot):
        return _hand_meta(robot, policy, solver, reports)
    cfg = robot.config
    mode = cfg.ik.orientation_mode.value
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
        "ik": cfg.ik.model_dump(mode="json"),
        "hand_parts": policy.hand_parts.model_dump(mode="json"),
        "reachable_definition": REACHABLE_DEFINITION,
        "orientation_error_definition": ORIENTATION_ERROR_DEFINITION[mode],
        # 목표는 월드 좌표 손 3D라 자동 카메라 자세(사람 검수 대상 아님)에 기댄다 (ADR 0050)
        "camera_pose": CAMERA_POSE_REVIEW,
        "episodes": reports,
    }


def _hand_meta(
    robot: LoadedDexterousRobot,
    policy: RetargetPolicy,
    solver: IkSolver | HandIkSolver,
    reports: list[dict[str, Any]],
) -> dict[str, Any]:
    """다지 손 로봇의 meta/dlp_retarget.json·manifest details.retarget 내용 (ADR 0068).

    평행 그리퍼 meta와 같은 키에 더해 `dexterous_hand`(손목 링크, 손 배율, 손가락 끝 허용 오차,
    손가락마다 사람 부위 → 끝 링크와 그 관절, 싣지 않은 손 관절)와 `finger_reachable_definition`을
    싣는다. `hand_parts`는 손목 부위와 손가락 끝 부위(대응 순서)다. 에피소드 보고(`reports`)에는
    손가락별 통계(`fingers`)가 들어 있다. 모델 파일 경로·작업자·장소 ID는 싣지 않는다.
    """
    cfg = robot.config
    hand = cfg.arm.dexterous_hand
    wrist_part, *tip_parts = finger_parts(robot, policy)
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
        "ik": cfg.ik.model_dump(mode="json"),
        "hand_parts": {"wrist": wrist_part, "fingers": tip_parts},
        "dexterous_hand": {
            "wrist_link": hand.wrist_link,
            "arm_joints": list(robot.chain.names),
            "hand_scale": hand.hand_scale,
            "fingertip_tolerance_m": hand.fingertip_tolerance_m,
            "fingers": [
                {
                    "human_part": f.config.human_part,
                    "tip_link": f.config.tip_link,
                    "joints": list(f.chain.names),
                }
                for f in robot.fingers
            ],
            "unmapped_joints": list(robot.unmapped_joints),
        },
        "reachable_definition": HAND_REACHABLE_DEFINITION,
        "finger_reachable_definition": FINGER_REACHABLE_DEFINITION,
        "orientation_error_definition": HAND_ORIENTATION_ERROR_DEFINITION,
        # 목표는 월드 좌표 손 3D라 자동 카메라 자세(사람 검수 대상 아님)에 기댄다 (ADR 0050)
        "camera_pose": CAMERA_POSE_REVIEW,
        "episodes": reports,
    }


def robot_override(
    episodes: list[tuple[Episode, RobotEpisode]], lerobot: LeRobotPolicy, robot: AnyRobot
) -> PackageOverride:
    """`write_package`에 넘길 로봇 모드 특징·배열 (episodes는 내보낼 순서).

    다지 손 로봇이면 손가락별 배열(`HAND_ARRAYS`)도 싣는다.
    """
    hand = HAND_ARRAYS if isinstance(robot, LoadedDexterousRobot) else {}
    return PackageOverride(
        features=lambda h, w: robot_features(lerobot, robot, h, w),
        arrays={**FRAME_ARRAYS, **ROBOT_ARRAYS, **hand},
        extra=[robot_arrays(ep, rob) for ep, rob in episodes],
        robot_type=robot.config.robot_id,
    )
