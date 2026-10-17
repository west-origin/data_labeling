"""`dlp robot check <ID>`: 로봇 등록 확인 (ADR 0047).

`config/robots/<ID>.yaml`(또는 `DLP_ROBOTS_DIR`, 고객 로봇)을 읽어 모델 파일 sha256, 기준 → 끝
링크 연쇄의 관절 목록, 그리퍼 관절을 검사하고, 영점 자세의 끝단 위치·접근·닫힘 축과 최대 팔 길이를
보인다. 운영자는 이 출력으로 접근·닫힘 축과 배치를 정한다 (`docs/handover/robot-retargeting.md`).
DB·저장소를 쓰지 않는다.
"""

from __future__ import annotations

import argparse

import numpy as np
from dlp_retarget.policy import load_policy
from dlp_retarget.retarget import max_reach
from dlp_retarget.robot import RobotError, load_robot
from dlp_retarget.urdf import UrdfError

from dlp_schema import repo_root


def cmd_check(args: argparse.Namespace) -> int:
    """`dlp robot check <ID>`. 반환: 0 정상, 1 설정·모델 확인 실패 (이유를 출력)."""
    root = repo_root()
    try:
        robot = load_robot(root, args.robot_id)
    except (RobotError, UrdfError, ValueError) as exc:
        print(f"[실패] {args.robot_id}: {exc}")
        return 1
    cfg = robot.config
    chain = robot.chain
    tip = chain.fk(np.zeros(len(chain.moving)))
    origin, reach = max_reach(chain)
    rot = tip[:3, :3]
    np.set_printoptions(precision=3, suppress=True)
    print(f"{cfg.robot_id}: {cfg.display_name}")
    print(f"  모델 sha256 {robot.model_sha256} ({cfg.model.license}, 공개 {cfg.model.public})")
    print(f"  연쇄 {cfg.base_link} → {cfg.arm.tip_link}: {', '.join(chain.names)}")
    print(f"  관절 하한 {chain.lower} / 상한 {chain.upper} / 속도 {chain.velocity}")
    print(f"  그리퍼 {robot.gripper.name} [{robot.gripper.lower}, {robot.gripper.upper}]")
    approach = rot @ np.array(cfg.arm.approach_axis)
    closing = rot @ np.array(cfg.arm.closing_axis)
    print(f"  영점 자세 끝단 위치 {tip[:3, 3]} m")
    print(f"  끝 링크 축(기준 좌표) x {rot[:, 0]} y {rot[:, 1]} z {rot[:, 2]}")
    print(f"  접근 축(기준 좌표) {approach}, 닫힘 축 {closing}")
    print(f"  첫 관절 {origin}에서 최대 팔 길이 {reach:.3f} m")
    print(f"  배치 {cfg.arm.placement.model_dump(mode='json')}")
    print(f"  리타기팅 정책 풀이기 {load_policy(root).solver.value}")
    return 0


def add_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # pyright: ignore[reportPrivateUsage]
    """`robot check` 하위 명령을 등록한다."""
    rob = sub.add_parser("robot", help="로봇 리타기팅 대상 로봇 (ADR 0047)")
    rsub = rob.add_subparsers(dest="robot_command", required=True)
    chk = rsub.add_parser("check", help="로봇 설정·모델 sha256·관절 연쇄 확인과 영점 자세 축 보기")
    chk.add_argument("robot_id")
    chk.set_defaults(func=cmd_check)
