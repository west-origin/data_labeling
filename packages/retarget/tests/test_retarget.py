"""로봇 리타기팅 (ADR 0047): URDF 기구학, 로봇 등록 검사, IK 정확도, 도달하지 못한 구간 표시.

정답: 공개 시험 로봇 SO-101의 정기구학으로 만든 도달 가능한 끝단 궤적(관절 범위 안 부드러운 관절
궤적)을 `dlp_fixtures.retarget`으로 사람 손 점으로 바꾼다. 리타기팅 결과의 끝단 경로를 정답과
비교한다. 서비스 없이 돈다.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml
from dlp_retarget.ik import DlsIkSolver
from dlp_retarget.policy import RetargetPolicy, UnreachableAction, load_policy
from dlp_retarget.retarget import (
    HandPoints,
    RetargetResult,
    max_reach,
    retarget_episode,
    retarget_version,
)
from dlp_retarget.robot import (
    ROBOTS_DIR_ENV,
    LoadedRobot,
    Placement,
    PlacementMode,
    RobotError,
    load_robot,
    load_robot_file,
)
from dlp_retarget.targets import gripper_values
from dlp_retarget.urdf import UrdfError, build_chain, load_joints
from numpy.typing import NDArray

from dlp_fixtures.retarget import hand_points_from_joints

ROOT = Path(__file__).resolve().parents[3]
FPS = 30


@pytest.fixture(scope="module")
def robot() -> LoadedRobot:
    """저장소의 공개 시험 로봇 SO-101."""
    return load_robot(ROOT, "so101", env={})


@pytest.fixture(scope="module")
def policy() -> RetargetPolicy:
    """저장소 retarget.yaml."""
    return load_policy(ROOT)


def with_placement(robot: LoadedRobot, placement: Placement) -> LoadedRobot:
    """배치만 바꾼 로봇 사본."""
    arm = robot.config.arm.model_copy(update={"placement": placement})
    return LoadedRobot(
        robot.config.model_copy(update={"arm": arm}),
        robot.chain,
        robot.gripper,
        robot.model_sha256,
        robot.model_path,
    )


def truth_motion(
    robot: LoadedRobot, seconds: float = 3.0
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """도달 가능한 정답: (프레임 시각 ms, 관절 궤적 (n, 5)). 관절 범위 가운데 둘레 부드러운 사인."""
    chain = robot.chain
    times = np.arange(0, seconds * 1000, 1000 / FPS)
    mid = (chain.lower + chain.upper) / 2
    amp = np.array([0.5, 0.35, 0.35, 0.4, 0.3])
    period = np.array([2.3, 1.7, 2.9, 1.3, 3.1])
    q = mid + amp * np.sin(2 * np.pi * times[:, None] / 1000 / period + np.arange(5))
    q[:, 1] -= 0.4  # 팔을 조금 들어 탁자 위 작업 자세에 가깝게
    q[:, 2] += 0.6
    assert np.all(q > chain.lower) and np.all(q < chain.upper)
    return times, q


def hand_points_of(
    robot: LoadedRobot, q: NDArray[np.float64], opening: NDArray[np.float64]
) -> tuple[dict[str, NDArray[np.float64]], NDArray[np.float64], NDArray[np.float64]]:
    """관절 궤적 → (월드 손 점, 정답 끝단 위치 (로봇 기준), 정답 접근 축) (world_fixed 배치의
    역변환, `dlp_fixtures.retarget.hand_points_from_joints`)."""
    cfg = robot.config.arm
    pl = cfg.placement
    return hand_points_from_joints(
        robot.chain.fk, q, cfg.approach_axis, cfg.closing_axis, opening,
        base_xyz=pl.base_xyz, base_yaw_rad=pl.base_yaw_rad, position_scale=pl.position_scale,
    )  # fmt: skip


def _run(
    robot: LoadedRobot,
    policy: RetargetPolicy,
    times: NDArray[np.float64],
    pts: dict[str, NDArray[np.float64]],
) -> RetargetResult:
    """리타기팅 실행."""
    hp = HandPoints(times, pts["wrist"], pts["thumb_tip"], pts["index_tip"])
    return retarget_episode(hp, robot, policy, DlsIkSolver())


FIXED = Placement(
    mode=PlacementMode.WORLD_FIXED, position_scale=0.5, base_xyz=(0.3, -0.1, -0.6), base_yaw_rad=0.4
)


# ---------------------------------------------------------------- 기구학·등록


def test_so101_kinematics_and_jacobian(robot: LoadedRobot) -> None:
    """SO-101 연쇄: 관절 5개, 영점에서 팔을 앞으로 편 자세, 야코비안이 수치 미분과 1e-6 안에서
    같다."""
    chain = robot.chain
    assert chain.names == (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
    )
    assert robot.joint_names[-1] == "gripper"
    tip = chain.fk(np.zeros(5))
    assert tip[0, 3] > 0.35 and abs(tip[1, 3]) < 1e-3  # 앞(+x)으로 편 자세
    assert np.allclose(tip[:3, :3] @ np.array([0, 0, 1.0]), [1, 0, 0], atol=0.01)  # 접근 축 = 앞
    rng = np.random.default_rng(0)
    for _ in range(5):
        q = rng.uniform(chain.lower, chain.upper)
        t0, jac = chain.jacobian(q)
        for k in range(5):
            dq = np.zeros(5)
            dq[k] = 1e-6
            assert np.allclose((chain.fk(q + dq)[:3, 3] - t0[:3, 3]) / 1e-6, jac[:3, k], atol=1e-6)
    origin, reach = max_reach(chain)
    samples = np.array([chain.fk(rng.uniform(chain.lower, chain.upper))[:3, 3] for _ in range(500)])
    assert np.linalg.norm(samples - origin, axis=1).max() <= reach


def test_robot_registry_checks(tmp_path: Path, robot: LoadedRobot) -> None:
    """모델 sha256이 다르면, 관절 목록이 연쇄와 다르면, 저장소 안 비공개 모델이면, 다지 손이면
    거부한다.
    고객 설정은 DLP_ROBOTS_DIR(저장소 밖)에서 먼저 찾는다."""
    src = ROOT / "config" / "robots"
    cfg = yaml.safe_load((src / "so101.yaml").read_text("utf-8"))
    shutil.copy(src / "models" / "so101" / "so101_new_calib.urdf", tmp_path / "arm.urdf")
    cfg["robot_id"] = "acme"
    cfg["model"].update(path="arm.urdf", public=False, license="고객 계약", source="계약 C-1")

    def write(data: dict[str, object]) -> Path:
        path = tmp_path / "acme.yaml"
        path.write_text(yaml.safe_dump(data, allow_unicode=True), "utf-8")
        return path

    write(cfg)
    loaded = load_robot(ROOT, "acme", env={ROBOTS_DIR_ENV: str(tmp_path)})
    assert loaded.model_sha256 == robot.model_sha256
    with pytest.raises(RobotError, match="sha256"):
        load_robot_file(write({**cfg, "model": {**cfg["model"], "sha256": "0" * 64}}))
    with pytest.raises(RobotError, match="joints"):
        load_robot_file(write({**cfg, "arm": {**cfg["arm"], "joints": ["shoulder_pan"]}}))
    with pytest.raises(RobotError, match="공개"):
        load_robot_file(write(cfg), inside_repo=tmp_path)
    with pytest.raises(ValueError, match="다지"):
        grip = {**cfg["arm"]["gripper"], "type": "dexterous"}
        load_robot_file(write({**cfg, "arm": {**cfg["arm"], "gripper": grip}}))
    with pytest.raises(RobotError, match="설정 파일이 없습니다"):
        load_robot(ROOT, "nope", env={})
    with pytest.raises(UrdfError, match="이어지는"):
        build_chain(load_joints(tmp_path / "arm.urdf"), "base_link", "no_such_link")


def test_every_repo_robot_is_a_public_model_with_matching_hash() -> None:
    """저장소 config/robots의 모든 설정은 공개 모델이고 sha256이 맞는다 (고객 설계 파일 반입
    방지)."""
    for path in sorted((ROOT / "config" / "robots").glob("*.yaml")):
        loaded = load_robot_file(path, inside_repo=ROOT)
        assert loaded.config.model.public and loaded.config.model.license


# ---------------------------------------------------------------- 정확도


def test_reachable_hand_trajectory_is_recovered(robot: LoadedRobot, policy: RetargetPolicy) -> None:
    """도달 가능한 정답 끝단 궤적(3초, 30 fps)에서 만든 손 점: 모든 프레임 도달, 끝단 위치 오차 최대
    2 mm,
    접근 축 오차 최대 1도, 그리퍼 값은 열림 거리 대응과 같고, 속도 한계에 걸리지 않으며 관절은 한계
    안."""
    r = with_placement(robot, FIXED)
    times, q_truth = truth_motion(r)
    opening = 0.05 + 0.03 * np.sin(2 * np.pi * times / 1500)
    pts, pos_b, a_b = hand_points_of(r, q_truth, opening)
    res = _run(r, policy, times, pts)
    assert res.report.reachable_fraction == 1.0 and res.report.frames_out_of_reach == 0
    tips = np.array([r.chain.fk(row[:5]) for row in res.q])
    pos_err = np.linalg.norm(tips[:, :3, 3] - pos_b, axis=1)
    ang = np.degrees(
        np.arccos(np.clip(np.sum((tips[:, :3, :3] @ np.array([0, 0, 1.0])) * a_b, axis=1), -1, 1))
    )
    assert pos_err.max() < 0.002 and ang.max() < 1.0
    assert np.allclose(res.q[:, 5], gripper_values(opening, r.config.arm.gripper), atol=1e-6)
    assert not res.velocity_limited.any()
    assert np.all(res.q[:, :5] >= r.chain.lower - 1e-9) and np.all(
        res.q[:, :5] <= r.chain.upper + 1e-9
    )
    assert res.report.decision == "keep" and res.report.position_error_max_m is not None
    assert res.report.position_error_max_m < 0.002


def test_episode_centroid_placement_recovers_path(
    robot: LoadedRobot, policy: RetargetPolicy
) -> None:
    """에피소드 중심 배치(요 회전 없음): 잡기 점 평균을 workspace_center로 옮기므로, 중심을 정답
    끝단 경로
    평균으로 두면 정답 경로를 2 mm 안에서 되찾는다 (손 점의 월드 위치와 무관)."""
    times, q_truth = truth_motion(robot)
    world = with_placement(robot, FIXED.model_copy(update={"base_yaw_rad": 0.0}))
    pts, pos_b, _ = hand_points_of(world, q_truth, np.full(times.size, 0.04))
    mean = pos_b.mean(axis=0)
    center = Placement(
        mode=PlacementMode.EPISODE_CENTROID,
        position_scale=0.5,
        workspace_center=(float(mean[0]), float(mean[1]), float(mean[2])),
    )
    r = with_placement(robot, center)
    res = _run(r, policy, times, pts)
    tips = np.array([r.chain.fk(row[:5]) for row in res.q])[:, :3, 3]
    assert res.report.reachable_fraction == 1.0
    assert np.linalg.norm(tips - pos_b, axis=1).max() < 0.002


def test_unreachable_segment_is_flagged_not_clamped(
    robot: LoadedRobot, policy: RetargetPolicy
) -> None:
    """한 구간(1.0~1.5초)의 손을 로봇 팔 길이 밖으로 옮기면: 그 프레임만 reachable=0·팔 길이 밖으로
    세고, 오차를 실제 거리로 보고한다(작업 공간으로 잘라 도달로 만들지 않음). 나머지는 도달한다.
    mark면 내보내고, drop_episode면 이유와 함께 버리기로 결정한다."""
    r = with_placement(robot, FIXED)
    times, q_truth = truth_motion(r)
    pts, _, _ = hand_points_of(r, q_truth, np.full(times.size, 0.04))
    seg = (times >= 1000) & (times < 1500)
    far = {k: v.copy() for k, v in pts.items()}
    for v in far.values():
        v[seg] += np.array([0.0, 0.0, 2.0])  # 월드 위로 2 m (로봇 기준 1 m)
    res = _run(r, policy, times, far)
    before, after = times < 1000, times >= 1500
    assert not res.reachable[seg].any() and res.reachable[before].all()
    # 구간이 끝난 뒤에는 속도 한계 때문에 바로 돌아오지 못한 프레임만 도달하지 못하고(속도 한계
    # 표시), 0.5초 안에 다시 도달한다
    late = after & ~res.reachable
    assert res.velocity_limited[late].all() and times[late].max(initial=0) < 2000
    assert res.reachable[times >= 2000].all()
    assert res.report.frames_out_of_reach == int(seg.sum())
    assert np.nanmin(res.position_error_m[seg]) > 0.3
    assert res.report.longest_unreachable_ms >= 450
    assert np.isfinite(res.q).all()
    assert res.report.decision == "keep"
    strict = policy.model_copy(
        update={"unreachable": UnreachableAction.DROP_EPISODE, "max_unreachable_fraction": 0.1}
    )
    dropped = _run(r, strict, times, far)
    assert dropped.report.decision == "drop" and "도달하지 못한 비율" in (
        dropped.report.reason or ""
    )


def test_missinghand_points_of_hold_and_are_marked(
    robot: LoadedRobot, policy: RetargetPolicy
) -> None:
    """손 점이 없는 프레임은 목표 없음(target_present=0)으로 표시하고 관절은 직전 해를 유지한다.
    목표 비율이
    min_target_fraction보다 낮으면 drop_episode에서 버린다."""
    r = with_placement(robot, FIXED)
    times, q_truth = truth_motion(r)
    pts, _, _ = hand_points_of(r, q_truth, np.full(times.size, 0.04))
    gap = (times >= 500) & (times < 2600)
    for v in pts.values():
        v[gap] = np.nan
    res = _run(r, policy, times, pts)
    assert not res.target_present[gap].any() and res.target_present[~gap].all()
    first = int(np.argmax(gap))
    assert np.allclose(res.q[gap], res.q[first - 1])
    assert np.isnan(res.position_error_m[gap]).all()
    assert res.report.frames_with_target == int((~gap).sum())
    strict = policy.model_copy(update={"unreachable": UnreachableAction.DROP_EPISODE})
    assert _run(r, strict, times, pts).report.decision == "drop"


def test_retarget_version_tracks_robot_model_and_policy(
    robot: LoadedRobot, policy: RetargetPolicy
) -> None:
    """리타기팅 판에 풀이기·로봇 설정 해시·모델 sha256·정책 해시가 들어가고, 배치를 바꾸면 판이
    바뀐다."""
    v = retarget_version(robot, policy, DlsIkSolver())
    assert v.startswith("retarget-dls-dls1+r") and f"+m{robot.model_sha256[:12]}" in v
    assert retarget_version(with_placement(robot, FIXED), policy, DlsIkSolver()) != v
