"""mink(MuJoCo) 격리 환경 풀이기 (ADR 0047, `@pytest.mark.isolated_env`, `make test-isolated`).

scripts/retarget-env(uv.lock 고정)를 받아 돌린다. 내장 풀이기와 같은 정답(SO-101 정기구학으로 만든
도달 가능한 궤적 + 팔 길이 밖 구간)에서 끝단 경로를 되찾고 도달하지 못한 구간을 표시하는지 본다.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from dlp_retarget.ik import DlsIkSolver
from dlp_retarget.mink import MinkIkSolver
from dlp_retarget.policy import load_policy
from dlp_retarget.retarget import HandPoints, retarget_episode
from dlp_retarget.robot import (
    LoadedRobot,
    OrientationMode,
    Placement,
    PlacementMode,
    load_robot,
    sha256_file,
)
from dlp_retarget.urdf import build_chain, load_joints

from dlp_fixtures.retarget import hand_points_from_joints

pytestmark = pytest.mark.isolated_env
ROOT = Path(__file__).resolve().parents[3]
FIXED = Placement(
    mode=PlacementMode.WORLD_FIXED, position_scale=0.5, base_xyz=(0.3, -0.1, -0.6), base_yaw_rad=0.4
)


def test_mink_solver_matches_reachable_path_and_flags_far_segment() -> None:
    """mink: 도달 가능한 구간의 끝단 위치 오차 최대 5 mm, 팔 길이 밖 구간은 도달하지 못함으로 표시.
    내장 풀이기와 같은 도달 판정이다."""
    policy = load_policy(ROOT)
    base = load_robot(ROOT, "so101", env={})
    arm = base.config.arm.model_copy(update={"placement": FIXED})
    robot = LoadedRobot(
        base.config.model_copy(update={"arm": arm}),
        base.chain,
        base.gripper,
        base.model_sha256,
        base.model_path,
    )
    chain = robot.chain
    times = np.arange(0, 2000, 1000 / 30)
    mid = (chain.lower + chain.upper) / 2
    amp = np.array([0.5, 0.35, 0.35, 0.4, 0.3])
    q_truth = mid + amp * np.sin(
        2 * np.pi * times[:, None] / 1000 / np.array([2.3, 1.7, 2.9, 1.3, 3.1])
    )
    q_truth[:, 1] -= 0.4
    q_truth[:, 2] += 0.6
    pts, pos_b, _ = hand_points_from_joints(
        chain.fk, q_truth, arm.approach_axis, arm.closing_axis, np.full(times.size, 0.04),
        base_xyz=FIXED.base_xyz, base_yaw_rad=FIXED.base_yaw_rad,
        position_scale=FIXED.position_scale,
    )  # fmt: skip
    seg = (times >= 800) & (times < 1100)
    for v in pts.values():
        v[seg] += np.array([0.0, 0.0, 2.0])
    hp = HandPoints(times, pts["wrist"], pts["thumb_tip"], pts["index_tip"])
    solver = MinkIkSolver(ROOT, policy.mink)
    assert solver.version.startswith("mink2-")
    res = retarget_episode(hp, robot, policy, solver)
    dls = retarget_episode(hp, robot, policy, DlsIkSolver())
    before = times < 800
    tips = np.array([robot.chain.fk(row[:5]) for row in res.q])[:, :3, 3]
    assert np.linalg.norm(tips[before] - pos_b[before], axis=1).max() < 0.005
    assert not res.reachable[seg].any() and res.reachable[before].all()
    assert (res.reachable == dls.reachable)[before | seg].all()


def _rotate_about(
    v: np.ndarray, center: np.ndarray, axis: np.ndarray, angle: np.ndarray
) -> np.ndarray:
    """점 (n, 3)을 프레임마다 center를 지나는 단위 축 axis (n, 3) 둘레로 angle (n,)만큼 돌린다."""
    d = v - center
    c, s = np.cos(angle)[:, None], np.sin(angle)[:, None]
    dot = np.sum(d * axis, axis=1, keepdims=True)
    return center + d * c + np.cross(axis, d) * s + axis * dot * (1 - c)


def test_mink_approach_mode_frees_roll_for_a_non_coordinate_approach_axis(tmp_path: Path) -> None:
    """도구 중심점 링크를 45도 돌려 접근 축이 끝 링크 좌표축이 아닌 [0, sin45, cos45]인 로봇:
    approach 모드에서 사람 손 롤이 크게(진폭 1 rad) 어긋나도 mink가 내장 풀이기와 같은 도달 판정·5
    mm 안 위치를 낸다 (롤 비용을 끝 링크 좌표로 주면 접근 축 둘레 회전에도 비용이 붙었다,
    F-COE-48)."""
    policy = load_policy(ROOT)
    base = load_robot(ROOT, "so101", env={})
    urdf = base.model_path
    assert urdf is not None
    text = urdf.read_text("utf-8").replace(
        '<origin xyz="-0.0079 -0.000218121 -0.0981274" rpy="0 3.14159 0"/>',
        '<origin xyz="-0.0079 -0.000218121 -0.0981274" rpy="0.785398 3.14159 0"/>',
    )
    variant = tmp_path / "so101_tilted.urdf"
    variant.write_text(text, "utf-8")
    h = math.sqrt(0.5)
    arm = base.config.arm.model_copy(
        update={"placement": FIXED, "approach_axis": (0.0, h, h), "closing_axis": (1.0, 0.0, 0.0)}
    )
    ik = base.config.ik.model_copy(update={"orientation_mode": OrientationMode.APPROACH})
    chain = build_chain(load_joints(variant), base.config.base_link, arm.tip_link)
    robot = LoadedRobot(
        base.config.model_copy(update={"arm": arm, "ik": ik}),
        chain, base.gripper, sha256_file(variant), variant,
    )  # fmt: skip
    times = np.arange(0, 1500, 1000 / 30)
    mid = (chain.lower + chain.upper) / 2
    q_truth = mid + np.array([0.4, 0.3, 0.3, 0.3, 0.2]) * np.sin(
        2 * np.pi * times[:, None] / 1000 / np.array([2.3, 1.7, 2.9, 1.3, 3.1])
    )
    q_truth[:, 1] -= 0.4
    q_truth[:, 2] += 0.6
    pts, pos_b, _ = hand_points_from_joints(
        chain.fk, q_truth, arm.approach_axis, arm.closing_axis, np.full(times.size, 0.04),
        base_xyz=FIXED.base_xyz, base_yaw_rad=FIXED.base_yaw_rad,
        position_scale=FIXED.position_scale,
    )  # fmt: skip
    grasp = (pts["thumb_tip"] + pts["index_tip"]) / 2
    a = grasp - pts["wrist"]
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    roll = 1.0 * np.sin(2 * np.pi * times / 1100)
    for part in ("thumb_tip", "index_tip"):
        pts[part] = _rotate_about(pts[part], grasp, a, roll)
    hp = HandPoints(times, pts["wrist"], pts["thumb_tip"], pts["index_tip"])
    res = retarget_episode(hp, robot, policy, MinkIkSolver(ROOT, policy.mink))
    dls = retarget_episode(hp, robot, policy, DlsIkSolver())
    assert dls.reachable.all() and (res.reachable == dls.reachable).all()
    tips = np.array([chain.fk(row[:5]) for row in res.q])[:, :3, 3]
    assert np.linalg.norm(tips - pos_b, axis=1).max() < 0.005
    assert res.velocity_limited.sum() <= dls.velocity_limited.sum()
