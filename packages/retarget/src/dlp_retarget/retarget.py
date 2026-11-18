"""에피소드 하나를 로봇 관절 궤적으로 옮기고 도달 가능성 보고를 만든다 (ADR 0047·0068).

평행 그리퍼 로봇 (`retarget_episode`).
입력: 에피소드 프레임 시각과 그 시각의 손 점 3개(세션 월드 좌표, 없으면 NaN). 출력: 프레임별 관절 값
(팔 관절 + 그리퍼), 목표 유무, 도달 여부, 위치·회전 오차, 속도 한계 표시와 `FeasibilityReport`.

정직성 규칙:
- 목표가 없는 프레임은 관절 값을 직전 프레임 값으로 두되 target_present=0으로 표시한다. 첫 목표 앞의
  프레임은 풀 해가 아직 없어 시작 자세(관절 범위 가운데)와 닫힌 그리퍼(`gripper.closed`)다. 어느
  쪽도 사람 동작에서 온 값이 아니다.
- 닿지 못한 프레임은 가장 가까운 해를 내되 reachable=0과 오차를 남긴다.
- 속도 한계: 팔 관절은 풀이기가(`ik.limit_step`), 그리퍼 관절은 여기서 같은 규칙(URDF velocity x
  velocity_scale x 프레임 간격, 직전 프레임 값 기준)으로 자르고 `velocity_limited`에 합친다.
- 목표가 하나도 없는 에피소드는 정책과 관계없이 버리기로 결정한다 (로봇 관절이 모두 시작 자세라 로봇
  데이터로 의미가 없다). 정책 `unreachable: drop_episode`는 비율 기준까지 더 본다.
  결정(`decision`)만 하고 실제로 빼는 것은 내보내기가 한다.
- orientation_mode none은 회전을 재지 않으므로 회전 오차를 NaN(내보내기 -1)으로 둔다 (0이면 방향이
  완벽히 맞은 것과 구분되지 않는다).

다지 손 로봇 (`retarget_hand_episode`, ADR 0068).
입력: 프레임 시각과 손목·대응한 손가락 끝 점(`FingerPoints`). 출력: 프레임별 관절 값(팔 관절 +
손가락 관절), 프레임 표시와 손가락별 표시(목표 유무·도달·오차), `HandFeasibilityReport`(손가락별
통계 포함). 위 정직성 규칙을 같은 뜻으로 따르고, 다른 점은 아래다.
- 손목이 없는 프레임은 목표가 없다 (모든 관절 유지, target_present=0). 손목은 있고 손가락 끝만 없는
  손가락은 그 손가락 관절만 유지하고 finger_target_present=0으로 남긴다. 한 번도 목표가 없던
  손가락의 관절은 시작 자세(관절 범위 가운데) 그대로다. 어느 쪽도 사람 동작에서 온 값이 아니다.
- 프레임의 reachable은 손목과 목표가 있는 손가락 끝이 모두 허용 오차 안일 때다. 손가락별 도달은
  finger_reachable이다. 닿지 못한 손목·손가락은 가장 가까운 해와 실제 오차를 낸다.
- 회전은 재지 않는다 (방향 목표가 없어 회전 오차는 늘 NaN).
- 속도 한계는 풀이기가 팔·손가락 모든 관절에 적용한다 (그리퍼 관절이 없다).
- 에피소드 판정: 손목 목표가 하나도 없으면 버린다. `unreachable: drop_episode`는 프레임 기준에 더해
  손가락마다 목표 비율이 `min_target_fraction` 미만이거나 닿지 못한 비율이
  `max_unreachable_fraction`을 넘으면 버린다 (`mark`는 손가락 통계를 보고에만 남긴다).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from dlp_retarget.hand_ik import HandIkProblem, HandIkSolution, HandIkSolver
from dlp_retarget.ik import IkProblem, IkSolver, limit_step
from dlp_retarget.policy import RetargetPolicy, UnreachableAction
from dlp_retarget.robot import (
    AnyRobot,
    LoadedDexterousRobot,
    LoadedRobot,
    OrientationMode,
    RobotError,
)
from dlp_retarget.targets import HandTargets, finger_targets, gripper_values, hand_targets
from dlp_retarget.urdf import Chain
from dlp_schema.common import Contract

Vec = NDArray[np.float64]


@dataclass(frozen=True)
class HandPoints:
    """프레임 시각과 그 시각의 손 점 (n, 3) 세 개 (세션 월드 좌표 m, 없으면 NaN).

    times_ms는 에피소드 프레임의 마스터 타임라인 시각 ms(실수)다. 3D 궤적 표본은 스트림 PTS
    시각이므로 호출자(`dlp_export.robot.world_hand_points`)가 마스터 → 스트림 시각으로 바꿔 읽어
    채운다 (ADR 0019). 여기서는 시각을 속도 한계의 프레임 간격과 `longest_unreachable_ms`에만 쓴다.
    세 점 중 하나라도 없는 프레임은 세 점 모두 NaN이다.
    """

    times_ms: Vec
    wrist: NDArray[np.float64]
    thumb: NDArray[np.float64]
    index: NDArray[np.float64]


@dataclass(frozen=True)
class FingerPoints:
    """프레임 시각과 그 시각의 손목 (n, 3)·손가락 끝 (n, F, 3) (세션 월드 좌표 m, 없으면 NaN).

    다지 손 로봇의 입력이다. 손가락 순서는 로봇 설정의 대응 순서(`DexterousHandConfig.fingers`)고,
    times_ms의 뜻은 `HandPoints`와 같다. 손목과 손가락 끝을 따로 비운다: 손목이 없는 프레임은
    목표가 없고, 손목은 있는데 손가락 끝만 없는 칸은 그 손가락만 목표가 없다.
    """

    times_ms: Vec
    wrist: NDArray[np.float64]
    tips: NDArray[np.float64]


class FeasibilityReport(Contract):
    """에피소드 도달 가능성 보고 (내보내기 meta/dlp_retarget.json, manifest details).

    fraction은 0~1. 오차 통계는 목표가 있는 프레임만 (없으면 None). frames_out_of_reach: 목표가 팔
    첫 관절에서 최대 팔 길이보다 먼 프레임 (어떤 자세로도 닿을 수 없음). velocity_limited_frames:
    팔·그리퍼 중 하나라도 속도 한계로 잘린 프레임 수. longest_unreachable_ms: 목표가 있는데 닿지
    못한 프레임이 이어진 가장 긴 구간의 첫·마지막 프레임 시각 차 (홀로 떨어진 프레임 하나면 0,
    미도달 유무는 reachable_frames < frames_with_target으로 본다). decision: 내보낼지(keep)
    버릴지(drop).
    """

    frames: int
    frames_with_target: int
    target_fraction: float
    reachable_frames: int
    reachable_fraction: float
    position_error_mean_m: float | None
    position_error_max_m: float | None
    orientation_error_mean_rad: float | None
    orientation_error_max_rad: float | None
    frames_out_of_reach: int
    velocity_limited_frames: int
    longest_unreachable_ms: float
    decision: Literal["keep", "drop"]
    reason: str | None = None


class FingerReport(Contract):
    """다지 손 에피소드 보고의 손가락 하나 (`HandFeasibilityReport.fingers`).

    human_part / tip_link: 대응 (사람 손가락 끝 부위 → 로봇 손가락 끝 링크). frames_with_target: 그
    손가락 끝 목표가 있던 프레임 수 (손목 목표도 있는 프레임), target_fraction은 에피소드 프레임
    대비다. reachable_frames: 그중 오차가 손가락 끝 허용 오차 안인 프레임, reachable_fraction은
    목표가 있는 프레임 대비다. 오차 통계는 목표가 있는 프레임만 (없으면 None).
    frames_out_of_reach: 손목 목표에서 그 손가락 끝 목표까지 거리가 로봇 손목 → 손가락 끝 거리의
    상한(`finger_span`)보다 먼 프레임. 어떤 자세로도 손목과 그 손가락 끝을 함께 맞출 수 없다는
    뜻이고, 많으면 `hand_scale`이 로봇 손보다 크다.
    """

    human_part: str
    tip_link: str
    frames_with_target: int
    target_fraction: float
    reachable_frames: int
    reachable_fraction: float
    position_error_mean_m: float | None
    position_error_max_m: float | None
    frames_out_of_reach: int


class HandFeasibilityReport(FeasibilityReport):
    """다지 손 에피소드 도달 가능성 보고 (내보내기 meta/dlp_retarget.json, manifest details).

    `FeasibilityReport`의 항목을 다지 손의 뜻으로 쓴다. frames_with_target: 손목 목표가 있는 프레임.
    reachable_frames: 손목과 목표가 있는 손가락 끝이 모두 허용 오차 안인 프레임. 위치 오차: 손목
    위치 오차. 회전 오차: 재지 않아 None. frames_out_of_reach: 손목 목표가 팔 첫 관절에서 최대 팔
    길이보다 먼 프레임. velocity_limited_frames: 팔·손가락 관절 중 하나라도 속도 한계로 잘린
    프레임. fingers: 손가락별 통계 (로봇 설정의 대응 순서).
    """

    fingers: tuple[FingerReport, ...]


@dataclass
class RetargetResult:
    """프레임별 결과 (n = 프레임 수)와 보고."""

    joint_names: tuple[str, ...]  # 팔 관절 + 그리퍼 관절
    q: NDArray[np.float64]  # (n, len(joint_names)) 회전 관절은 rad, 직선 관절은 m (URDF 단위)
    target_present: NDArray[np.bool_]
    reachable: NDArray[np.bool_]
    position_error_m: Vec
    orientation_error_rad: Vec
    velocity_limited: NDArray[np.bool_]
    report: FeasibilityReport


@dataclass
class HandRetargetResult:
    """다지 손 로봇의 프레임별 결과 (n = 프레임 수, F = 손가락 수)와 보고.

    앞의 항목은 `RetargetResult`와 이름이 같고 뜻은 다지 손의 것이다: target_present는 손목 목표
    유무, reachable은 손목과 목표가 있는 손가락 끝이 모두 도달, position_error_m은 손목 위치 오차,
    orientation_error_rad는 늘 NaN(재지 않음)이다.
    """

    joint_names: tuple[str, ...]  # 팔 관절 + 손가락 관절 (대응 순서)
    q: NDArray[np.float64]  # (n, len(joint_names)) 회전 관절은 rad, 직선 관절은 m (URDF 단위)
    target_present: NDArray[np.bool_]
    reachable: NDArray[np.bool_]
    position_error_m: Vec
    orientation_error_rad: Vec
    velocity_limited: NDArray[np.bool_]
    finger_names: tuple[str, ...]  # 손가락 순서 (사람 손가락 끝 부위 이름)
    finger_target_present: NDArray[np.bool_]  # (n, F) 손목과 그 손가락 끝 목표가 있는가
    finger_reachable: NDArray[np.bool_]  # (n, F) 목표가 있고 손가락 끝 허용 오차 안
    finger_position_error_m: NDArray[np.float64]  # (n, F) 목표가 없으면 NaN
    report: HandFeasibilityReport


def retarget_version(
    robot: AnyRobot, policy: RetargetPolicy, solver: IkSolver | HandIkSolver
) -> str:
    """리타기팅 판 문자열.

    형식은 `retarget-<풀이기 이름>-<풀이기 판>+r<로봇 설정 해시 12자>+m<모델 sha256 앞 12자>+p<정책
    해시 8자>`다. 풀이기·로봇 설정·모델 파일·정책 중 하나라도 바뀌면 판이 바뀐다. 내보내기
    manifest에 남는다. 다지 손 로봇은 다지 손 풀이기의 이름·판(`dls`·`dlshand1`)이 들어간다.
    """
    return (
        f"retarget-{solver.name}-{solver.version}+r{robot.config.digest()}"
        f"+m{robot.model_sha256[:12]}+p{policy.digest()}"
    )


def max_reach(chain: Chain) -> tuple[Vec, float]:
    """(첫 움직이는 관절 위치, 그 점에서 끝까지 링크 길이 합). 기준 링크 좌표, m.

    q = 0 자세에서 이어진 움직이는 관절 위치와 끝 위치 사이 거리를 더하고, 직선 관절은 이동 범위의
    큰 쪽 절댓값을 더한다. 회전은 이웃 관절 사이 거리를 바꾸지 않으므로 끝은 어떤 자세로도 첫
    관절에서 이 길이보다 멀리 가지 못한다. `frames_out_of_reach`가 이 값을 쓴다.
    """
    q0 = np.zeros(len(chain.moving))
    f = chain.frames(q0)
    pts = [*f.origins, f.tip[:3, 3]]
    length = sum(float(np.linalg.norm(b - a)) for a, b in itertools.pairwise(pts))
    # prismatic 관절은 이동 범위만큼 더 닿는다
    length += sum(max(abs(j.lower), abs(j.upper)) for j in chain.moving if j.kind == "prismatic")
    return f.origins[0], length


def _longest_run_ms(mask: NDArray[np.bool_], times: Vec) -> float:
    """참이 이어진 가장 긴 구간의 길이 (ms, 첫 프레임 시각 ~ 마지막 프레임 시각, 한 프레임이면
    0)."""
    best, start = 0.0, None
    for k, m in enumerate(mask):
        if m and start is None:
            start = k
        if start is not None and (not m or k == mask.size - 1):
            end = k if m else k - 1
            best = max(best, float(times[end] - times[start]))
            start = None
    return best


def _decide(
    n: int, with_target: int, reachable: int, policy: RetargetPolicy
) -> tuple[Literal["keep", "drop"], str | None]:
    """정책에 따른 내보내기 결정과 이유 (목표가 없는 에피소드는 정책과 관계없이 drop)."""
    target_fraction = with_target / n if n else 0.0
    unreachable = 1 - reachable / with_target if with_target else 1.0
    if with_target == 0:
        return "drop", "손 목표가 있는 프레임이 없습니다 (월드 좌표 손 3D 없음)"
    if policy.unreachable is UnreachableAction.MARK:
        return "keep", None
    if target_fraction < policy.min_target_fraction:
        return (
            "drop",
            f"손 목표가 있는 프레임 비율 {target_fraction:.2f} < {policy.min_target_fraction}",
        )
    if unreachable > policy.max_unreachable_fraction:
        return "drop", f"도달하지 못한 비율 {unreachable:.2f} > {policy.max_unreachable_fraction}"
    return "keep", None


def _stat(values: Vec, fn: str) -> float | None:
    """NaN을 뺀 평균·최대 (값이 없으면 None, 소수 넷째 자리)."""
    v = values[~np.isnan(values)]
    if v.size == 0:
        return None
    return round(float(v.mean() if fn == "mean" else v.max()), 4)


def retarget_episode(
    points: HandPoints, robot: LoadedRobot, policy: RetargetPolicy, solver: IkSolver
) -> RetargetResult:
    """에피소드 하나의 손 점을 로봇 관절 궤적·도달 표시·보고로 옮긴다.

    순서: 손 점 → 끝단 목표(`hand_targets`, 로봇 배치 적용) → 풀이기(`solver.solve`, 팔 관절) →
    그리퍼 궤적(`_gripper_track`) → 보고(`FeasibilityReport`). 규칙은 모듈 docstring의 정직성
    규칙을 따른다.

    Args:
        points: 프레임 시각(마스터 ms)과 손 점.
        robot: 확인을 마친 로봇 (`robot.load_robot`).
        policy: `retarget.yaml`. 내보내기 결정(`decision`)에만 쓴다.
        solver: IK 풀이기 (`DlsIkSolver` 또는 `MinkIkSolver`).

    Returns:
        `RetargetResult`. DB·파일을 쓰지 않는다 (mink 풀이기만 임시 디렉터리와 하위 프로세스를
        쓴다).

    Raises:
        ValueError: 손 점 배열 모양이 다를 때 (`hand_targets`), mink 풀이기에 모델 경로가 없을 때.
        subprocess.CalledProcessError: mink 스크립트가 실패할 때.
    """
    cfg = robot.config
    tg = hand_targets(points.wrist, points.thumb, points.index, cfg.arm.placement)
    problem = IkProblem(
        chain=robot.chain,
        targets=tg,
        times_ms=points.times_ms,
        approach_axis=np.array(cfg.arm.approach_axis, dtype=np.float64),
        closing_axis=np.array(cfg.arm.closing_axis, dtype=np.float64),
        config=cfg.ik,
        model_path=robot.model_path,
    )
    sol = solver.solve(problem)
    n = tg.present.size
    held, grip_limited = _gripper_track(tg.present, tg.opening_m, robot, points.times_ms)
    velocity_limited = sol.velocity_limited | grip_limited
    orientation_error = sol.orientation_error_rad
    if cfg.ik.orientation_mode is OrientationMode.NONE:
        orientation_error = np.full(n, np.nan)  # 재지 않음 (모듈 설명)
    q = np.column_stack([sol.q, held])
    origin, reach = max_reach(robot.chain)
    far = tg.present & (np.linalg.norm(np.nan_to_num(tg.position) - origin, axis=1) > reach)
    with_target, reachable = int(tg.present.sum()), int(sol.reachable.sum())
    decision, reason = _decide(n, with_target, reachable, policy)
    report = FeasibilityReport(
        frames=n,
        frames_with_target=with_target,
        target_fraction=round(with_target / n, 4) if n else 0.0,
        reachable_frames=reachable,
        reachable_fraction=round(reachable / with_target, 4) if with_target else 0.0,
        position_error_mean_m=_stat(sol.position_error_m, "mean"),
        position_error_max_m=_stat(sol.position_error_m, "max"),
        orientation_error_mean_rad=_stat(orientation_error, "mean"),
        orientation_error_max_rad=_stat(orientation_error, "max"),
        frames_out_of_reach=int(far.sum()),
        velocity_limited_frames=int(velocity_limited.sum()),
        longest_unreachable_ms=_longest_run_ms(tg.present & ~sol.reachable, points.times_ms),
        decision=decision,
        reason=reason,
    )
    return RetargetResult(
        joint_names=robot.joint_names,
        q=q,
        target_present=tg.present,
        reachable=sol.reachable,
        position_error_m=sol.position_error_m,
        orientation_error_rad=orientation_error,
        velocity_limited=velocity_limited,
        report=report,
    )


def _gripper_track(
    present: NDArray[np.bool_], opening_m: Vec, robot: LoadedRobot, times_ms: Vec
) -> tuple[Vec, NDArray[np.bool_]]:
    """프레임별 그리퍼 관절 값과 속도 한계 표시.

    목표가 있는 프레임은 열림 거리 대응 값(`gripper_values`), 없는 프레임은 직전 값(첫 목표 전에는
    closed)이다. 첫 프레임을 뺀 모든 프레임을 직전 프레임 값 기준 속도 한계(URDF 그리퍼 velocity x
    velocity_scale)로 자른다. 손가락 끝 깊이 잡음으로 열림 거리가 프레임마다 흔들려도 내보낸 값이
    로봇이 낼 수 없는 속도가 되지 않고, 잘린 프레임이 표시된다.
    """
    cfg = robot.config
    want = gripper_values(opening_m, cfg.arm.gripper)
    vmax = np.array([robot.gripper.velocity * cfg.ik.velocity_scale])
    n = present.size
    held = np.empty(n)
    limited = np.zeros(n, dtype=np.bool_)
    last = np.array([cfg.arm.gripper.closed])
    for k in range(n):
        cur = np.array([float(want[k])]) if present[k] else last
        if k > 0:
            cur, limited[k] = limit_step(cur, last, vmax, times_ms, k)
        held[k], last = float(cur[0]), cur
    return held, limited


def finger_parts(robot: LoadedDexterousRobot, policy: RetargetPolicy) -> tuple[str, ...]:
    """다지 손 로봇이 읽는 사람 손 점의 3D 궤적 부위: (손목, 손가락 끝들).

    손목 부위는 retarget.yaml `hand_parts.wrist`, 손가락 끝 부위는 로봇 설정의 `human_part`(대응
    순서)다. 온톨로지 hand_joints와의 대조는 내보내기 시작 때 한다 (데이터셋 버전의 온톨로지,
    `dlp_export.runner.check_hand_parts`).

    Raises:
        RobotError: 손가락의 human_part가 손목 부위와 같을 때 (그 손가락 끝 목표가 손목 목표와
            같아져 늘 닿지 못한다).
    """
    wrist = policy.hand_parts.wrist
    tips = tuple(f.config.human_part for f in robot.fingers)
    if wrist in tips:
        raise RobotError(
            f"{robot.config.robot_id}: 손가락의 human_part {wrist}가 손목 부위"
            "(retarget.yaml hand_parts.wrist)와 같습니다"
        )
    return (wrist, *tips)


def finger_span(chain: Chain) -> tuple[float, float]:
    """(영점 자세의 손목 링크 원점 → 손가락 끝 거리, 그 거리의 상한). m.

    chain은 손목 링크 → 손가락 끝 링크 연쇄(`LoadedFinger.chain`)다. 상한 = 손목 링크 원점에서 그
    손가락의 첫 움직이는 관절까지 거리(고정 관절뿐이라 자세와 무관) + 그 관절에서 끝까지 링크 길이
    합(`max_reach`). 손목 목표와 손가락 끝 목표 사이가 상한보다 멀면 어떤 자세로도 둘을 함께 맞출
    수 없다. `dlp robot check`가 손 배율(`hand_scale`)을 정하는 값으로 보이고, 보고의 손가락별
    `frames_out_of_reach`가 쓴다.
    """
    origin, length = max_reach(chain)
    zero = chain.fk(np.zeros(len(chain.moving)))[:3, 3]
    return float(np.linalg.norm(zero)), float(np.linalg.norm(origin)) + length


def _finger_reports(
    robot: LoadedDexterousRobot, tg: HandTargets, sol: HandIkSolution
) -> tuple[FingerReport, ...]:
    """손가락별 통계 (`FingerReport`, 로봇 설정의 대응 순서)."""
    n = tg.present.size
    # 손목 목표 → 손가락 끝 목표 거리 (n, F). 목표가 없는 칸은 0이고 아래에서 뺀다
    span = np.linalg.norm(np.nan_to_num(tg.tips) - np.nan_to_num(tg.wrist)[:, None, :], axis=2)
    out: list[FingerReport] = []
    for i, finger in enumerate(robot.fingers):
        present = tg.tip_present[:, i]
        with_target, reached = int(present.sum()), int(sol.tip_reachable[:, i].sum())
        out.append(
            FingerReport(
                human_part=finger.config.human_part,
                tip_link=finger.config.tip_link,
                frames_with_target=with_target,
                target_fraction=round(with_target / n, 4) if n else 0.0,
                reachable_frames=reached,
                reachable_fraction=round(reached / with_target, 4) if with_target else 0.0,
                position_error_mean_m=_stat(sol.tip_error_m[:, i], "mean"),
                position_error_max_m=_stat(sol.tip_error_m[:, i], "max"),
                frames_out_of_reach=int(
                    (present & (span[:, i] > finger_span(finger.chain)[1])).sum()
                ),
            )
        )
    return tuple(out)


def _decide_hand(
    n: int,
    with_target: int,
    reachable: int,
    fingers: tuple[FingerReport, ...],
    policy: RetargetPolicy,
) -> tuple[Literal["keep", "drop"], str | None]:
    """다지 손 에피소드의 내보내기 결정과 이유.

    프레임 기준은 `_decide`와 같다 (손목 목표가 없는 에피소드는 정책과 관계없이 drop).
    drop_episode면 손가락마다 목표 비율(에피소드 프레임 대비)과 닿지 못한 비율(그 손가락 목표가
    있는 프레임 대비)도 같은 정책 값으로 본다.
    """
    decision, reason = _decide(n, with_target, reachable, policy)
    if decision == "drop" or policy.unreachable is UnreachableAction.MARK:
        return decision, reason
    for f in fingers:
        target_fraction = f.frames_with_target / n if n else 0.0
        if target_fraction < policy.min_target_fraction:
            return (
                "drop",
                f"손가락 {f.human_part} 목표가 있는 프레임 비율 {target_fraction:.2f} < "
                f"{policy.min_target_fraction}",
            )
        if f.frames_with_target == 0:
            continue  # 닿지 못한 비율을 잴 프레임이 없다 (min_target_fraction 0일 때만 온다)
        unreachable = 1 - f.reachable_frames / f.frames_with_target
        if unreachable > policy.max_unreachable_fraction:
            return (
                "drop",
                f"손가락 {f.human_part}이 도달하지 못한 비율 {unreachable:.2f} > "
                f"{policy.max_unreachable_fraction}",
            )
    return "keep", None


def retarget_hand_episode(
    points: FingerPoints,
    robot: LoadedDexterousRobot,
    policy: RetargetPolicy,
    solver: HandIkSolver,
) -> HandRetargetResult:
    """에피소드 하나의 손목·손가락 끝 점을 다지 손 로봇의 관절 궤적·표시·보고로 옮긴다.

    순서: 손 점 → 손목·손가락 끝 목표(`finger_targets`, 배치와 손 배율 적용) → 풀이기
    (`solver.solve`, 팔 관절 + 손가락 관절) → 보고(`HandFeasibilityReport`). 규칙은 모듈 docstring의
    다지 손 절을 따른다.

    Args:
        points: 프레임 시각(마스터 ms)과 손목·손가락 끝 점 (손가락 순서 = 로봇 설정의 대응 순서).
        robot: 확인을 마친 다지 손 로봇 (`robot.load_any_robot`).
        policy: `retarget.yaml`. 내보내기 결정(`decision`)에만 쓴다.
        solver: 다지 손 IK 풀이기 (`hand_ik.make_hand_solver`).

    Returns:
        `HandRetargetResult`. DB·파일을 쓰지 않는다.

    Raises:
        ValueError: 손 점 배열 모양이 다르거나 손가락 수가 로봇 설정과 다를 때.
    """
    cfg = robot.config
    hand = cfg.arm.dexterous_hand
    if points.tips.ndim != 3 or points.tips.shape[1] != len(robot.fingers):
        raise ValueError(
            f"손가락 끝 점 {points.tips.shape}: (n, {len(robot.fingers)}, 3)이어야 합니다 "
            "(로봇 설정의 손가락 대응 수)"
        )
    tg = finger_targets(points.wrist, points.tips, cfg.arm.placement, hand.hand_scale)
    problem = HandIkProblem(
        tree=robot.tree,
        targets=tg,
        times_ms=points.times_ms,
        config=cfg.ik,
        fingertip_tolerance_m=hand.fingertip_tolerance_m,
    )
    sol = solver.solve(problem)
    n = tg.present.size
    origin, reach = max_reach(robot.chain)
    far = tg.present & (np.linalg.norm(np.nan_to_num(tg.wrist) - origin, axis=1) > reach)
    with_target, reachable = int(tg.present.sum()), int(sol.reachable.sum())
    fingers = _finger_reports(robot, tg, sol)
    decision, reason = _decide_hand(n, with_target, reachable, fingers, policy)
    report = HandFeasibilityReport(
        frames=n,
        frames_with_target=with_target,
        target_fraction=round(with_target / n, 4) if n else 0.0,
        reachable_frames=reachable,
        reachable_fraction=round(reachable / with_target, 4) if with_target else 0.0,
        position_error_mean_m=_stat(sol.wrist_error_m, "mean"),
        position_error_max_m=_stat(sol.wrist_error_m, "max"),
        orientation_error_mean_rad=None,
        orientation_error_max_rad=None,
        frames_out_of_reach=int(far.sum()),
        velocity_limited_frames=int(sol.velocity_limited.sum()),
        longest_unreachable_ms=_longest_run_ms(tg.present & ~sol.reachable, points.times_ms),
        decision=decision,
        reason=reason,
        fingers=fingers,
    )
    return HandRetargetResult(
        joint_names=robot.joint_names,
        q=sol.q,
        target_present=tg.present,
        reachable=sol.reachable,
        position_error_m=sol.wrist_error_m,
        orientation_error_rad=np.full(n, np.nan),  # 재지 않음 (방향 목표가 없다)
        velocity_limited=sol.velocity_limited,
        finger_names=tuple(f.config.human_part for f in robot.fingers),
        finger_target_present=tg.tip_present,
        finger_reachable=sol.tip_reachable,
        finger_position_error_m=sol.tip_error_m,
        report=report,
    )
