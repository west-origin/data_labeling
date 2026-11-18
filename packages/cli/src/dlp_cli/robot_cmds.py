"""로봇 등록 확인 하위 명령 (ADR 0047·0068).

`dlp export lerobot --robot <ID>` 전에 고객 로봇 설정이 맞는지 본다. 운영자는 이 출력으로 접근·닫힘
축과 배치를 정한다 (`docs/handover/robot-retargeting.md`). DB·저장소·원본 버킷에 접근하지 않는다.

하위 명령 (인자 오류는 argparse가 종료 코드 2로 끝낸다):

`dlp robot check <robot_id>`
    `config/robots/<ID>.yaml`(또는 `DLP_ROBOTS_DIR`, 고객 로봇)을
    `dlp_retarget.robot.load_any_robot`으로 읽어 모델 파일 sha256과 관절 연쇄를 검사한다. 읽는 정책:
    `config/policies/retarget.yaml`(풀이기 이름 출력, 다지 손 로봇의 손목 부위).
    - 평행 그리퍼 로봇: 기준 → 끝 링크 연쇄의 관절 목록과 그리퍼 관절을 검사하고, 영점 자세의 끝단
      위치·접근·닫힘 축과 최대 팔 길이(`dlp_retarget.retarget.max_reach`)를 보인다.
    - 다지 손 로봇 (ADR 0068): 기준 → 손목 링크 팔 연쇄와 손가락마다 손목 링크 → 끝 링크의 관절을
      검사하고, 손가락마다 대응(사람 손가락 끝 부위 → 끝 링크)·관절·한계와 영점 자세의 손목 →
      손가락 끝 거리, 그 거리의 상한(`dlp_retarget.retarget.finger_span`, 손 배율 `hand_scale`을
      정하는 값), 풀지 않는 손 관절을 보인다. 정책 풀이기가 mink면 다지 손을 풀 수 없어 실패한다.
    종료 코드: 0 정상, 1 설정·모델 확인 실패 (`RobotError`, `UrdfError`, `ValueError`, 이유를
    출력한다).
"""

from __future__ import annotations

import argparse

import numpy as np
from dlp_retarget.hand_ik import make_hand_solver
from dlp_retarget.policy import RetargetPolicy, load_policy
from dlp_retarget.retarget import finger_parts, finger_span, max_reach
from dlp_retarget.robot import (
    LoadedDexterousRobot,
    LoadedRobot,
    RobotError,
    load_any_robot,
)
from dlp_retarget.urdf import UrdfError

from dlp_schema import repo_root


def cmd_check(args: argparse.Namespace) -> int:
    """`dlp robot check <ID>`. 반환: 0 정상, 1 설정·모델 확인 실패 (이유를 출력)."""
    root = repo_root()
    try:
        robot = load_any_robot(root, args.robot_id)
        policy = load_policy(root)
        if isinstance(robot, LoadedDexterousRobot):
            # 내보내기가 거부할 조합(mink 풀이기, 손목 부위와 같은 손가락 부위)을 여기서 먼저 알린다
            make_hand_solver(policy)
            finger_parts(robot, policy)
    except (RobotError, UrdfError, ValueError) as exc:
        print(f"[실패] {args.robot_id}: {exc}")
        return 1
    cfg = robot.config
    np.set_printoptions(precision=3, suppress=True)
    print(f"{cfg.robot_id}: {cfg.display_name}")
    print(f"  모델 sha256 {robot.model_sha256} ({cfg.model.license}, 공개 {cfg.model.public})")
    if isinstance(robot, LoadedDexterousRobot):
        _print_dexterous(robot, policy)
    else:
        _print_parallel(robot)
    print(f"  배치 {cfg.arm.placement.model_dump(mode='json')}")
    print(f"  리타기팅 정책 풀이기 {policy.solver.value}")
    return 0


def _print_parallel(robot: LoadedRobot) -> None:
    """평행 그리퍼 로봇: 팔 연쇄, 그리퍼, 영점 자세의 끝단 축, 최대 팔 길이."""
    cfg = robot.config
    chain = robot.chain
    tip = chain.fk(np.zeros(len(chain.moving)))
    origin, reach = max_reach(chain)
    rot = tip[:3, :3]
    print(f"  연쇄 {cfg.base_link} → {cfg.arm.tip_link}: {', '.join(chain.names)}")
    print(f"  관절 하한 {chain.lower} / 상한 {chain.upper} / 속도 {chain.velocity}")
    print(f"  그리퍼 {robot.gripper.name} [{robot.gripper.lower}, {robot.gripper.upper}]")
    approach = rot @ np.array(cfg.arm.approach_axis)
    closing = rot @ np.array(cfg.arm.closing_axis)
    print(f"  영점 자세 끝단 위치 {tip[:3, 3]} m")
    print(f"  끝 링크 축(기준 좌표) x {rot[:, 0]} y {rot[:, 1]} z {rot[:, 2]}")
    print(f"  접근 축(기준 좌표) {approach}, 닫힘 축 {closing}")
    print(f"  첫 관절 {origin}에서 최대 팔 길이 {reach:.3f} m")


def _print_dexterous(robot: LoadedDexterousRobot, policy: RetargetPolicy) -> None:
    """다지 손 로봇: 팔 연쇄, 손목 링크, 손가락마다 대응·관절·손목 → 끝 거리, 풀지 않는 관절."""
    cfg = robot.config
    hand = cfg.arm.dexterous_hand
    chain = robot.chain
    wrist = chain.fk(np.zeros(len(chain.moving)))
    origin, reach = max_reach(chain)
    rot = wrist[:3, :3]
    print(f"  팔 연쇄 {cfg.base_link} → {hand.wrist_link}: {', '.join(chain.names)}")
    print(f"  관절 하한 {chain.lower} / 상한 {chain.upper} / 속도 {chain.velocity}")
    print(f"  영점 자세 손목 위치 {wrist[:3, 3]} m")
    print(f"  손목 링크 축(기준 좌표) x {rot[:, 0]} y {rot[:, 1]} z {rot[:, 2]}")
    print(f"  첫 관절 {origin}에서 손목까지 최대 팔 길이 {reach:.3f} m")
    print(
        f"  다지 손: 손목 부위 {finger_parts(robot, policy)[0]} → {hand.wrist_link}, "
        f"손 배율 {hand.hand_scale}, 손가락 끝 허용 오차 {hand.fingertip_tolerance_m} m"
    )
    for finger in robot.fingers:
        fc = finger.chain
        zero, limit = finger_span(fc)
        print(
            f"  손가락 {finger.config.human_part} → {finger.config.tip_link}: {', '.join(fc.names)}"
        )
        print(f"    관절 하한 {fc.lower} / 상한 {fc.upper} / 속도 {fc.velocity}")
        print(f"    영점 자세 손목 → 끝 {zero:.3f} m, 손목 → 끝 거리 상한 {limit:.3f} m")
    if robot.unmapped_joints:
        print(f"  풀지 않는 손 관절 (대응 없음, 내보내지 않음): {', '.join(robot.unmapped_joints)}")
    print(
        "  손 배율: 사람 손목 → 손가락 끝 거리 x hand_scale이 위 상한을 넘는 프레임은 그 손가락이 "
        "닿지 못한다 (보고의 손가락별 frames_out_of_reach)"
    )


def add_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # pyright: ignore[reportPrivateUsage]
    """`robot check` 하위 명령을 등록한다."""
    rob = sub.add_parser("robot", help="로봇 리타기팅 대상 로봇 (ADR 0047)")
    rsub = rob.add_subparsers(dest="robot_command", required=True)
    chk = rsub.add_parser("check", help="로봇 설정·모델 sha256·관절 연쇄 확인과 영점 자세 축 보기")
    chk.add_argument("robot_id")
    chk.set_defaults(func=cmd_check)
