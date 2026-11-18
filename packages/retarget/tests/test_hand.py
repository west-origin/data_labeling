"""다지 손 리타기팅 (ADR 0068): 나무 구조 기구학, 다지 손 로봇 등록 검사, 손목·손가락 끝 IK 정확도,
닿지 못한 손가락과 끝 점이 없는 손가락의 표시, 속도 한계.

정답: 직접 쓴 시험 로봇(`dlp_fixtures.dexhand`, 6관절 팔 + 손바닥 + 다섯 손가락 x 3관절)의 정기구학
으로 만든 손목·손가락 끝 위치를, 배치와 손 배율을 거꾸로 써서 사람 손 점으로 바꾼다. 리타기팅
결과의 손목·손가락 끝 위치를 그 정답과 비교한다. 서비스 없이 돈다.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from dlp_retarget import hand_ik
from dlp_retarget.hand_ik import DlsHandSolver, make_hand_solver
from dlp_retarget.policy import RetargetPolicy, SolverName, UnreachableAction, load_policy
from dlp_retarget.retarget import (
    FingerPoints,
    HandRetargetResult,
    finger_parts,
    finger_span,
    retarget_hand_episode,
    retarget_version,
)
from dlp_retarget.robot import (
    ROBOTS_DIR_ENV,
    LoadedDexterousRobot,
    LoadedRobot,
    RobotError,
    load_any_robot,
    load_any_robot_file,
    load_robot_file,
)
from dlp_retarget.targets import HandTargets
from dlp_retarget.urdf import UrdfError, build_chain, build_tree, load_joints
from numpy.typing import NDArray

from dlp_fixtures.dexhand import (
    ARM_JOINTS,
    BASE_LINK,
    FINGERS,
    dexterous_hand_urdf,
    finger_joints,
    human_points_from_robot,
    smooth_joint_motion,
    write_robot,
)

ROOT = Path(__file__).resolve().parents[3]
FPS = 30
# 로봇 기준을 세션 월드의 한 자리에 두고(요 0.4 rad) 팔 궤적을 0.8배로 줄이는 배치
PLACE: dict[str, Any] = {
    "mode": "world_fixed",
    "position_scale": 0.8,
    "base_xyz": [0.3, -0.1, -0.6],
    "base_yaw_rad": 0.4,
}
# 손가락 번호 (시험 로봇의 대응 순서 = `FINGERS`)
THUMB, INDEX, MIDDLE, RING, PINKY = range(5)


def load(directory: Path, **config: Any) -> LoadedDexterousRobot:
    """시험 로봇의 URDF와 설정을 디렉터리에 쓰고 읽는다 (`write_robot`의 인자 그대로)."""
    robot = load_any_robot_file(write_robot(directory, **config))
    assert isinstance(robot, LoadedDexterousRobot)
    return robot


@pytest.fixture(scope="module")
def robot(tmp_path_factory: pytest.TempPathFactory) -> LoadedDexterousRobot:
    """다섯 손가락을 모두 대응한 시험 로봇 (`PLACE` 배치, 손 배율 1)."""
    return load(tmp_path_factory.mktemp("dexhand"), placement=PLACE)


@pytest.fixture(scope="module")
def policy() -> RetargetPolicy:
    """저장소 retarget.yaml."""
    return load_policy(ROOT)


@dataclass(frozen=True)
class Truth:
    """정답: 프레임 시각 ms (n,), 관절 궤적 (n, dof), 로봇 기준 손목·손가락 끝 (n, F + 1, 3)."""

    times: NDArray[np.float64]
    q: NDArray[np.float64]
    positions: NDArray[np.float64]


def truth_motion(robot: LoadedDexterousRobot, seconds: float = 2.0, seed: int = 0) -> Truth:
    """관절 범위 가운데 둘레의 부드러운 임의 관절 궤적과 그 정기구학 (30 fps)."""
    tree = robot.tree
    times = np.arange(0, seconds * 1000, 1000 / FPS)
    q = smooth_joint_motion(tree.lower, tree.upper, times, seed=seed)
    return Truth(times, q, np.array([tree.positions(row) for row in q]))


def human_points(
    robot: LoadedDexterousRobot, times: NDArray[np.float64], positions: NDArray[np.float64]
) -> FingerPoints:
    """로봇 기준 좌표의 손목·손가락 끝 (n, F + 1, 3) → 사람 손 점 (로봇 설정의 배치·손 배율을
    거꾸로, `dlp_fixtures.dexhand.human_points_from_robot`)."""
    arm = robot.config.arm
    wrist, tips = human_points_from_robot(
        positions[:, 0],
        positions[:, 1:],
        base_xyz=arm.placement.base_xyz,
        base_yaw_rad=arm.placement.base_yaw_rad,
        position_scale=arm.placement.position_scale,
        hand_scale=arm.dexterous_hand.hand_scale,
    )
    return FingerPoints(times, wrist, tips)


def run(
    robot: LoadedDexterousRobot, policy: RetargetPolicy, points: FingerPoints
) -> HandRetargetResult:
    """리타기팅 실행 (내장 다지 손 풀이기)."""
    return retarget_hand_episode(points, robot, policy, DlsHandSolver())


def solved(robot: LoadedDexterousRobot, res: HandRetargetResult) -> NDArray[np.float64]:
    """결과 관절 값의 정기구학: 손목·손가락 끝 위치 (n, F + 1, 3)."""
    return np.array([robot.tree.positions(row) for row in res.q])


def finger_columns(robot: LoadedDexterousRobot, finger: int) -> list[int]:
    """그 손가락 관절의 결과 열 번호."""
    return [robot.joint_names.index(name) for name in robot.fingers[finger].chain.names]


def within_speed_limits(
    robot: LoadedDexterousRobot, res: HandRetargetResult, times: NDArray[np.float64]
) -> bool:
    """모든 프레임 사이 관절 걸음이 URDF 속도 x velocity_scale x 프레임 간격 안인가."""
    vmax = robot.tree.velocity * robot.config.ik.velocity_scale
    steps = np.abs(np.diff(res.q, axis=0))
    return bool((steps <= vmax * (np.diff(times) / 1000)[:, None] + 1e-9).all())


def within_joint_limits(robot: LoadedDexterousRobot, res: HandRetargetResult) -> bool:
    """모든 관절 값이 URDF 한계 안인가."""
    tree = robot.tree
    return bool((res.q >= tree.lower - 1e-9).all() and (res.q <= tree.upper + 1e-9).all())


def rewrite(path: Path, key: str, value: Any) -> Path:
    """설정 파일의 점 경로 키(예: `arm.dexterous_hand.fingers.1.tip_link`) 하나를 바꿔 다시 쓴다."""
    data: Any = json.loads(path.read_text("utf-8"))
    node: Any = data
    *parents, last = key.split(".")
    # 숫자 조각은 목록 위치, 그 밖은 사전 키다 (설정 키에는 숫자만으로 된 이름이 없다)
    for part in parents:
        node = node[int(part) if part.isdigit() else part]
    node[int(last) if last.isdigit() else last] = value
    path.write_text(json.dumps(data, ensure_ascii=False), "utf-8")
    return path


# ---------------------------------------------------------------- 나무 구조 기구학


def test_tree_matches_serial_chains_and_numeric_jacobian(robot: LoadedDexterousRobot) -> None:
    """나무의 관절 순서는 팔 관절 + 손가락 관절(대응 순서)이고, 끝마다 위치가 그 끝의 직렬 연쇄
    정기구학과 같으며, 위치 야코비안이 수치 미분과 1e-6 안에서 같다. 다른 손가락의 관절 열은 0이다.
    손목 → 손가락 끝 거리는 어떤 자세에서도 `finger_span`의 상한을 넘지 않는다."""
    tree = robot.tree
    assert tree.names == (*ARM_JOINTS, *(name for f in FINGERS for name in finger_joints(f)))
    assert robot.joint_names == tree.names and robot.chain.names == ARM_JOINTS
    assert robot.model_path is not None
    joints = load_joints(robot.model_path)
    rng = np.random.default_rng(0)
    for _ in range(5):
        q = rng.uniform(tree.lower, tree.upper)
        pos, jac = tree.jacobian(q)
        assert pos.shape == (6, 3) and jac.shape == (18, 21)
        for i, tip in enumerate(tree.tips):
            chain = build_chain(joints, BASE_LINK, tip)
            cols = [tree.names.index(name) for name in chain.names]
            assert np.allclose(chain.fk(q[cols])[:3, 3], pos[i], atol=1e-12)
            others = [k for k in range(q.size) if k not in cols]
            assert not jac[3 * i : 3 * i + 3][:, others].any()
        for k in range(q.size):
            dq = np.zeros(q.size)
            dq[k] = 1e-6
            numeric = (tree.positions(q + dq) - tree.positions(q - dq)).reshape(-1) / 2e-6
            assert np.allclose(numeric, jac[:, k], atol=1e-6)
    zero = tree.positions(np.zeros(len(tree.names)))
    limits = np.array([finger_span(f.chain)[1] for f in robot.fingers])
    for i, finger in enumerate(robot.fingers):
        at_zero = finger_span(finger.chain)[0]
        offset: NDArray[np.float64] = zero[i + 1] - zero[0]
        assert math.isclose(at_zero, float(np.linalg.norm(offset)), abs_tol=1e-9)
        assert at_zero <= limits[i] + 1e-9
    samples = np.array([tree.positions(rng.uniform(tree.lower, tree.upper)) for _ in range(300)])
    assert (np.linalg.norm(samples[:, 1:] - samples[:, :1], axis=2) <= limits + 1e-9).all()


_BRANCHED = """<robot name="t">
  <link name="base"/><link name="l1"/><link name="l2"/><link name="l3"/><link name="l4"/>
  <link name="tip_a"/>
  <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/>
    <origin xyz="0 0 0.1"/><axis xyz="0 0 1"/>
    <limit effort="1" velocity="1" lower="-1" upper="1"/></joint>
  <joint name="slide" type="prismatic"><parent link="l1"/><child link="l2"/>
    <origin xyz="0.1 0 0" rpy="0 0.3 0"/><axis xyz="1 0 0"/>
    <limit effort="1" velocity="1" lower="0" upper="0.2"/></joint>
  <joint name="mount" type="fixed"><parent link="l2"/><child link="tip_a"/>
    <origin xyz="0 0.05 0"/></joint>
  <joint name="j3" type="continuous"><parent link="l1"/><child link="l3"/>
    <origin xyz="0 0.1 0"/><axis xyz="0 1 0"/></joint>
  <joint name="j4" type="revolute"><parent link="l3"/><child link="l4"/>
    <origin xyz="0 0 0.2"/><axis xyz="1 0 0"/>
    <limit effort="1" velocity="1" lower="-1" upper="1"/></joint>
</robot>"""


def test_tree_handles_prismatic_fixed_and_shared_joints(tmp_path: Path) -> None:
    """갈라지는 가지에 직선·고정·continuous 관절이 있어도 야코비안이 수치 미분과 같다. 함께 쓰는
    앞쪽 관절(j1)은 관절 값 벡터에 한 번만 들어가고, 끝이 기준 링크면 위치가 원점이다. 끝이 없거나
    같은 끝을 두 번 주면 거부한다."""
    path = tmp_path / "t.urdf"
    path.write_text(_BRANCHED, "utf-8")
    joints = load_joints(path)
    tree = build_tree(joints, "base", ["base", "l1", "tip_a", "l4"])
    assert tree.names == ("j1", "slide", "j3", "j4") and np.isinf(tree.upper[2])
    assert tree.on_path.tolist() == [
        [False, False, False, False],
        [True, False, False, False],
        [True, True, False, False],
        [True, False, True, True],
    ]
    rng = np.random.default_rng(1)
    for _ in range(5):
        q = rng.uniform([-1.0, 0.0, -3.0, -1.0], [1.0, 0.2, 3.0, 1.0])
        pos, jac = tree.jacobian(q)
        assert not pos[0].any()
        assert np.allclose(build_chain(joints, "base", "tip_a").fk(q[:2])[:3, 3], pos[2])
        for k in range(4):
            dq = np.zeros(4)
            dq[k] = 1e-6
            numeric = (tree.positions(q + dq) - tree.positions(q - dq)).reshape(-1) / 2e-6
            assert np.allclose(numeric, jac[:, k], atol=1e-6)
    with pytest.raises(ValueError, match="움직이는 관절 4개"):
        tree.positions(np.zeros(3))
    with pytest.raises(UrdfError, match="끝 링크가 없습니다"):
        build_tree(joints, "base", [])
    with pytest.raises(UrdfError, match="두 번"):
        build_tree(joints, "base", ["l4", "l4"])


# ---------------------------------------------------------------- 등록


def test_dexterous_robot_is_registered_with_fingers_read_from_urdf(tmp_path: Path) -> None:
    """다지 손 설정: 손가락 관절은 URDF의 손목 링크 → 끝 링크 경로에서 읽고, 내보내는 관절 순서는
    팔 관절 + 손가락 관절(설정의 대응 순서)이다. 대응하지 않은 손가락의 관절은 풀지 않는 관절로
    드러난다. 평행 그리퍼 전용 로더는 다지 손 설정을 거부하고, 모델 파일 검사(sha256, 저장소 안
    비공개 모델)는 다지 손에도 같다."""
    path = write_robot(tmp_path, robot_id="dex3", fingers=("index", "thumb", "middle"))
    robot = load_any_robot(ROOT, "dex3", env={ROBOTS_DIR_ENV: str(tmp_path)})
    assert isinstance(robot, LoadedDexterousRobot)
    assert [f.config.human_part for f in robot.fingers] == ["index_tip", "thumb_tip", "middle_tip"]
    assert robot.joint_names == (
        *ARM_JOINTS,
        *finger_joints("index"),
        *finger_joints("thumb"),
        *finger_joints("middle"),
    )
    assert robot.unmapped_joints == (*finger_joints("ring"), *finger_joints("pinky"))
    assert robot.tree.tips == ("hand", "index_tip", "thumb_tip", "middle_tip")
    with pytest.raises(RobotError, match="다지 손"):
        load_robot_file(path)
    assert isinstance(load_any_robot(ROOT, "so101", env={}), LoadedRobot)
    scaled = load(
        tmp_path / "scaled", robot_id="dex3", fingers=("index", "thumb", "middle"), hand_scale=1.2
    )
    assert scaled.joint_names == robot.joint_names
    assert scaled.config.digest() != robot.config.digest()
    with pytest.raises(RobotError, match="공개"):
        load_any_robot_file(rewrite(path, "model.public", False), inside_repo=tmp_path)
    with pytest.raises(RobotError, match="sha256"):
        load_any_robot_file(rewrite(path, "model.sha256", "0" * 64))


_GRIPPER = {
    "type": "parallel",
    "joint": "index_curl",
    "closed": 0.0,
    "open": 1.0,
    "closed_distance_m": 0.01,
    "open_distance_m": 0.09,
}
_SIX_FINGERS = [
    *({"human_part": f"{f}_tip", "tip_link": f"{f}_tip"} for f in FINGERS),
    {"human_part": "wrist", "tip_link": "index_distal"},
]


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        # URDF에 없는 링크
        ("base_link", "no_such_link", "URDF에 없습니다"),
        ("arm.dexterous_hand.wrist_link", "no_such_link", "URDF에 없습니다"),
        ("arm.dexterous_hand.fingers.1.tip_link", "no_such_link", "URDF에 없습니다"),
        # 손목 링크 아래가 아닌 끝 링크, 움직이는 관절이 없는 손가락, 손목 링크와 같은 끝 링크
        ("arm.dexterous_hand.fingers.1.tip_link", "forearm", "아래에 있지 않습니다"),
        ("arm.dexterous_hand.fingers.1.tip_link", "palm", "움직이는 관절이 없습니다"),
        ("arm.dexterous_hand.fingers.1.tip_link", "hand", "손목 링크"),
        # 손가락끼리 관절을 함께 씀 (중지를 검지 첫 마디에 대응)
        ("arm.dexterous_hand.fingers.2.tip_link", "index_proximal", "함께 씁니다"),
        # 대응 중복 (같은 끝 링크, 같은 사람 손가락), 손가락 수
        ("arm.dexterous_hand.fingers.1.tip_link", "thumb_tip", "중복"),
        ("arm.dexterous_hand.fingers.1.human_part", "thumb_tip", "중복"),
        ("arm.dexterous_hand.fingers", _SIX_FINGERS, "at most 5"),
        ("arm.dexterous_hand.fingers", [], "at least 1"),
        # 팔 관절 목록, 값 범위, 평행 그리퍼 항목과 섞어 쓰기
        ("arm.joints", ["shoulder_yaw"], "arm.joints"),
        ("arm.dexterous_hand.hand_scale", 0.0, "greater than 0"),
        ("arm.dexterous_hand.fingertip_tolerance_m", 0.0, "greater than 0"),
        ("ik.wrist_weight", 0.0, "greater than 0"),
        ("ik.orientation_mode", "full", "orientation_mode"),
        ("arm.gripper", _GRIPPER, "함께 쓸 수 없습니다"),
    ],
)
def test_dexterous_config_is_rejected_with_a_reason(
    tmp_path: Path, key: str, value: Any, match: str
) -> None:
    """지원하지 않는 구조와 틀린 설정은 지어내지 않고 이유와 함께 거부한다: 없는 링크, 손목 아래가
    아닌 끝 링크, 관절 없는 손가락, 손가락끼리 함께 쓰는 관절, 대응 중복, 손가락 0개·6개, 팔 관절
    목록 불일치, 0 이하 배율·허용 오차·가중치, 다지 손에 없는 방향 항목, 그리퍼와 함께 적기."""
    path = rewrite(write_robot(tmp_path), key, value)
    with pytest.raises(ValueError, match=match):
        load_any_robot_file(path)


@pytest.mark.parametrize(
    ("old", "new", "match"),
    [
        (
            '<joint name="index_curl" type="revolute">',
            '<joint name="index_curl" type="revolute"><mimic joint="index_flexion"/>',
            "mimic",
        ),
        (
            '<joint name="index_curl" type="revolute">',
            '<joint name="index_curl" type="floating">',
            "지원하지 않는",
        ),
        (
            '<joint name="elbow_pitch" type="revolute">',
            '<joint name="elbow_pitch" type="revolute"><mimic joint="shoulder_pitch"/>',
            "mimic",
        ),
    ],
)
def test_mimic_and_unsupported_joints_on_the_paths_are_rejected(
    tmp_path: Path, old: str, new: str, match: str
) -> None:
    """손가락 경로(와 팔 연쇄)의 mimic 관절·지원하지 않는 관절 종류는 거부한다 (종속 관절을 독립
    자유도로 풀면 실제 손이 낼 수 없는 관절 조합이 나온다). 대응하지 않은 손가락의 mimic 관절은
    막지 않는다."""
    urdf = dexterous_hand_urdf()
    assert old in urdf
    with pytest.raises(ValueError, match=match):
        load_any_robot_file(write_robot(tmp_path, urdf=urdf.replace(old, new)))
    if match == "mimic" and "index" in old:
        kept = load(tmp_path / "kept", urdf=urdf.replace(old, new), fingers=("thumb", "middle"))
        assert "index_curl" in kept.unmapped_joints


def test_hand_solver_and_parts_follow_the_policy(
    tmp_path: Path, robot: LoadedDexterousRobot, policy: RetargetPolicy
) -> None:
    """다지 손 로봇은 내장 풀이기만 푼다: mink를 고른 정책은 이유와 함께 거부한다. 리타기팅 판에
    다지 손 풀이기 판과 로봇 설정 해시가 들어가고 손 배율을 바꾸면 판이 바뀐다. 손 점 부위는 손목
    (정책) + 손가락 끝(로봇 설정)이고, 손가락 부위가 손목 부위와 같으면 거부한다."""
    solver = make_hand_solver(policy)
    assert isinstance(solver, DlsHandSolver)
    with pytest.raises(RobotError, match="mink"):
        make_hand_solver(policy.model_copy(update={"solver": SolverName.MINK}))
    version = retarget_version(robot, policy, solver)
    assert version.startswith(f"retarget-dls-dlshand1+r{robot.config.digest()}+m")
    assert f"+m{robot.model_sha256[:12]}+p{policy.digest()}" in version
    scaled = load(tmp_path, placement=PLACE, hand_scale=1.1)
    assert retarget_version(scaled, policy, solver) != version
    assert finger_parts(robot, policy) == (
        "wrist",
        "thumb_tip",
        "index_tip",
        "middle_tip",
        "ring_tip",
        "pinky_tip",
    )
    clash = load_any_robot_file(
        rewrite(write_robot(tmp_path / "clash"), "arm.dexterous_hand.fingers.0.human_part", "wrist")
    )
    assert isinstance(clash, LoadedDexterousRobot)
    with pytest.raises(RobotError, match="손목 부위"):
        finger_parts(clash, policy)


# ---------------------------------------------------------------- 정확도


@pytest.mark.parametrize(
    ("hand_scale", "position_scale", "fingers"),
    [
        (1.0, 1.0, FINGERS),
        (1.25, 0.6, FINGERS),
        (0.8, 1.0, ("thumb", "index", "middle")),
    ],
)
def test_wrist_and_fingertips_are_recovered_within_1mm(
    tmp_path: Path,
    policy: RetargetPolicy,
    hand_scale: float,
    position_scale: float,
    fingers: tuple[str, ...],
) -> None:
    """임의 관절 궤적(2초, 30 fps)의 정기구학으로 만든 손목·손가락 끝 점을 사람 손 점으로 넣으면:
    풀이의 손목·손가락 끝 위치가 정답과 1 mm 안, 모든 프레임·손가락 도달, 속도 한계에 걸리지 않고
    관절은 한계 안이다. 손 배율이 1이 아니고 팔 작업 공간 배율과 달라도, 손가락을 셋만 대응해도
    같다. 방향은 재지 않는다 (회전 오차 NaN)."""
    place = {**PLACE, "position_scale": position_scale}
    robot = load(tmp_path, placement=place, hand_scale=hand_scale, fingers=fingers)
    truth = truth_motion(robot)
    res = run(robot, policy, human_points(robot, truth.times, truth.positions))
    n = truth.times.size
    assert res.joint_names == robot.joint_names
    assert res.q.shape == (n, len(ARM_JOINTS) + 3 * len(fingers))
    assert res.finger_names == tuple(f"{f}_tip" for f in fingers)
    assert np.linalg.norm(solved(robot, res) - truth.positions, axis=2).max() < 0.001
    assert res.target_present.all() and res.reachable.all()
    assert res.finger_target_present.all() and res.finger_reachable.all()
    assert not res.velocity_limited.any()
    assert within_joint_limits(robot, res) and within_speed_limits(robot, res, truth.times)
    assert np.isnan(res.orientation_error_rad).all()
    report = res.report
    assert report.decision == "keep" and report.reachable_fraction == 1.0
    assert report.frames_out_of_reach == 0 and report.orientation_error_max_rad is None
    assert report.position_error_max_m is not None and report.position_error_max_m < 0.001
    assert [f.human_part for f in report.fingers] == [f"{f}_tip" for f in fingers]
    for f in report.fingers:
        assert f.frames_with_target == n and f.reachable_fraction == 1.0
        assert f.frames_out_of_reach == 0
        assert f.position_error_max_m is not None and f.position_error_max_m < 0.001


def test_episode_centroid_places_the_wrist_mean_at_the_workspace_center(
    tmp_path: Path, policy: RetargetPolicy
) -> None:
    """에피소드 중심 배치(요 회전 없음)는 손목 평균을 workspace_center로 옮긴다. 중심을 정답 손목
    경로의 평균으로 두면 정답 손목·손가락 끝을 1 mm 안에서 되찾는다 (손 점의 월드 위치와 무관)."""
    fixed = {**PLACE, "base_yaw_rad": 0.0, "position_scale": 0.7}
    world = load(tmp_path / "world", placement=fixed, hand_scale=1.1)
    truth = truth_motion(world)
    points = human_points(world, truth.times, truth.positions)
    center = {
        "mode": "episode_centroid",
        "position_scale": 0.7,
        "workspace_center": truth.positions[:, 0].mean(axis=0).tolist(),
    }
    robot = load(tmp_path / "center", placement=center, hand_scale=1.1)
    res = run(robot, policy, points)
    assert res.report.reachable_fraction == 1.0
    assert np.linalg.norm(solved(robot, res) - truth.positions, axis=2).max() < 0.001


# ---------------------------------------------------------------- 닿지 못한 목표


def test_fingertip_beyond_finger_length_is_flagged_not_clamped(
    robot: LoadedDexterousRobot, policy: RetargetPolicy
) -> None:
    """한 구간(1.0~1.5초)의 검지 끝을 손목에서 15 cm 더 멀리 두면: 그 손가락은 그 구간에 닿지
    못함으로 표시되고(`frames_out_of_reach`도 그 프레임 수), 오차는 실제 거리다. 목표를 손가락 길이
    안으로 잘라 도달로 만들지 않는다: 손목 오차 + 검지 끝 오차 ≥ (목표의 손목 → 끝 거리 - 로봇의
    그 거리 상한). 구간 앞은 모두 도달하고, 구간 뒤는 속도 한계로 늦는 프레임만 닿지 못하다가
    0.5초 안에 다시 도달한다. mark면 내보내고 drop_episode면 이유와 함께 버린다."""
    truth = truth_motion(robot, seconds=3.0)
    points = human_points(robot, truth.times, truth.positions)
    times = truth.times
    seg = (times >= 1000) & (times < 1500)
    tips = points.tips.copy()
    offset = tips[seg, INDEX] - points.wrist[seg]
    longer = 1 + 0.15 / np.linalg.norm(offset, axis=1, keepdims=True)
    tips[seg, INDEX] = points.wrist[seg] + offset * longer
    far = FingerPoints(times, points.wrist, tips)
    res = run(robot, policy, far)
    assert res.target_present.all() and res.finger_target_present.all()
    assert not res.finger_reachable[seg, INDEX].any() and not res.reachable[seg].any()
    span = np.linalg.norm(tips[seg, INDEX] - points.wrist[seg], axis=1)  # 손 배율 1
    limit = finger_span(robot.fingers[INDEX].chain)[1]
    assert (span > limit).all()
    shortfall = res.position_error_m[seg] + res.finger_position_error_m[seg, INDEX]
    assert (shortfall >= span - limit - 1e-9).all()
    assert float(res.finger_position_error_m[seg, INDEX].min()) > 0.05
    counts = [f.frames_out_of_reach for f in res.report.fingers]
    assert counts == [0, int(seg.sum()), 0, 0, 0] and res.report.frames_out_of_reach == 0
    assert res.reachable[times < 1000].all()
    late = (times >= 1500) & ~res.reachable
    assert res.velocity_limited[late].all() and times[late].max(initial=0) < 2000
    assert res.reachable[times >= 2000].all()
    assert np.isfinite(res.q).all() and within_joint_limits(robot, res)
    assert within_speed_limits(robot, res, times)
    assert res.report.decision == "keep"
    strict = policy.model_copy(
        update={"unreachable": UnreachableAction.DROP_EPISODE, "max_unreachable_fraction": 0.1}
    )
    dropped = run(robot, strict, far)
    assert dropped.report.decision == "drop"
    assert "도달하지 못한 비율" in (dropped.report.reason or "")


def test_fingertip_closer_than_joint_limits_allow_is_flagged(
    robot: LoadedDexterousRobot, policy: RetargetPolicy
) -> None:
    """한 구간의 중지 끝을 손목 쪽으로 당겨(손목 → 끝 거리의 1/4) 관절 한계 안에서는 접을 수 없는
    곳에 두면: 손가락 길이 밖은 아니지만(`frames_out_of_reach` 0) 그 손가락은 닿지 못함으로
    표시되고 오차가 남는다. 관절 값은 한계를 넘지 않는다."""
    truth = truth_motion(robot, seconds=3.0)
    points = human_points(robot, truth.times, truth.positions)
    times = truth.times
    seg = (times >= 1000) & (times < 1500)
    tips = points.tips.copy()
    tips[seg, MIDDLE] = points.wrist[seg] + 0.25 * (tips[seg, MIDDLE] - points.wrist[seg])
    res = run(robot, policy, FingerPoints(times, points.wrist, tips))
    assert not res.finger_reachable[seg, MIDDLE].any() and not res.reachable[seg].any()
    assert float(res.finger_position_error_m[seg, MIDDLE].min()) > 0.02
    middle = res.report.fingers[MIDDLE]
    assert middle.frames_out_of_reach == 0
    assert middle.frames_with_target - middle.reachable_frames >= int(seg.sum())
    assert within_joint_limits(robot, res) and within_speed_limits(robot, res, times)
    assert res.reachable[times < 1000].all() and res.reachable[times >= 2000].all()


def test_wrist_beyond_arm_reach_is_flagged(
    robot: LoadedDexterousRobot, policy: RetargetPolicy
) -> None:
    """한 구간의 손 전체를 월드 위로 2 m 옮기면(로봇 기준 1.6 m): 그 프레임은 팔 길이 밖으로 세고
    손목 오차를 실제 거리로 보고한다. 손가락 길이 문제는 아니다."""
    truth = truth_motion(robot, seconds=3.0)
    points = human_points(robot, truth.times, truth.positions)
    times = truth.times
    seg = (times >= 1000) & (times < 1500)
    wrist, tips = points.wrist.copy(), points.tips.copy()
    wrist[seg] += np.array([0.0, 0.0, 2.0])
    tips[seg] += np.array([0.0, 0.0, 2.0])
    res = run(robot, policy, FingerPoints(times, wrist, tips))
    assert not res.reachable[seg].any()
    assert res.report.frames_out_of_reach == int(seg.sum())
    assert all(f.frames_out_of_reach == 0 for f in res.report.fingers)
    assert float(res.position_error_m[seg].min()) > 1.0
    assert res.report.longest_unreachable_ms >= 450
    assert within_joint_limits(robot, res) and within_speed_limits(robot, res, times)
    assert res.reachable[times < 1000].all() and res.reachable[times >= 2000].all()


# ---------------------------------------------------------------- 없는 손 점


def test_missing_fingertip_holds_that_finger_and_is_marked(
    robot: LoadedDexterousRobot, policy: RetargetPolicy
) -> None:
    """검지 끝 점만 없는 구간(0.5~1.5초): 그 손가락은 목표 없음으로 표시되고 그 관절은 직전 값
    그대로다 (오차 NaN, 지어내지 않는다). 손목과 나머지 손가락은 그 구간에도 정답을 1 mm 안에서
    따라가고 프레임은 도달이다 (목표가 있는 것은 모두 닿았다). 끝 점이 돌아오면 다시 따라간다."""
    truth = truth_motion(robot, seconds=3.0)
    points = human_points(robot, truth.times, truth.positions)
    times = truth.times
    gap = (times >= 500) & (times < 1500)
    tips = points.tips.copy()
    tips[gap, INDEX] = np.nan
    res = run(robot, policy, FingerPoints(times, points.wrist, tips))
    assert res.target_present.all()
    assert not res.finger_target_present[gap, INDEX].any()
    assert res.finger_target_present[~gap, INDEX].all()
    assert res.finger_target_present[:, [THUMB, MIDDLE, RING, PINKY]].all()
    cols = finger_columns(robot, INDEX)
    first = int(np.argmax(gap))
    held = np.tile(res.q[first - 1, cols], (int(gap.sum()), 1))
    assert np.array_equal(res.q[gap][:, cols], held)
    assert np.isnan(res.finger_position_error_m[gap, INDEX]).all()
    assert not res.finger_reachable[gap, INDEX].any()
    error = np.linalg.norm(solved(robot, res) - truth.positions, axis=2)
    rest = [0, 1 + THUMB, 1 + MIDDLE, 1 + RING, 1 + PINKY]  # 손목과 나머지 손가락 끝
    assert error[gap][:, rest].max() < 0.001
    assert res.reachable[gap].all()
    index = res.report.fingers[INDEX]
    assert index.frames_with_target == int((~gap).sum())
    assert index.target_fraction == round(float((~gap).mean()), 4)
    # 돌아온 직후 몇 프레임은 속도 한계로 따라잡는 중이고, 그 뒤에는 다시 1 mm 안이다
    assert res.finger_reachable[times >= 1700, INDEX].all()
    assert error[times >= 1700].max() < 0.001
    assert within_speed_limits(robot, res, times)
    strict = policy.model_copy(
        update={"unreachable": UnreachableAction.DROP_EPISODE, "min_target_fraction": 0.8}
    )
    dropped = run(robot, strict, FingerPoints(times, points.wrist, tips))
    assert dropped.report.decision == "drop"
    assert "손가락 index_tip 목표가 있는 프레임 비율" in (dropped.report.reason or "")


def test_restart_never_moves_a_finger_without_a_target(
    monkeypatch: pytest.MonkeyPatch, robot: LoadedDexterousRobot
) -> None:
    """다시 시작하는 프레임에서 어느 시작점이 이겨도 목표가 없는 손가락의 관절은 직전 값이다.

    정답: 검지 끝 목표만 없는 한 프레임을, 반복 한 번만 허용하고 시작점을 [해에서 먼 점, 정답
    자세인데 검지 관절만 상한]으로 바꿔 푼다. 둘째 시작점이 이기지만(비용 거의 0) 검지 관절은
    그 시작점의 값(상한)이 아니라 직전 값(시작 자세 = 관절 범위 가운데)이어야 한다."""
    tree = robot.tree
    mid = (tree.lower + tree.upper) / 2
    q_truth = truth_motion(robot).q[0]
    cols = finger_columns(robot, INDEX)
    winner = q_truth.copy()
    winner[cols] = tree.upper[cols]
    expected = q_truth.copy()
    expected[cols] = mid[cols]

    def fake_seeds(chain: object, count: int) -> list[NDArray[np.float64]]:
        return [mid.copy()] if count == 1 else [tree.lower.copy(), winner.copy()]

    monkeypatch.setattr(hand_ik, "seeds", fake_seeds)
    positions = tree.positions(expected)
    present = np.ones((1, 5), dtype=np.bool_)
    present[0, INDEX] = False
    targets = HandTargets(
        present=np.array([True]),
        wrist=positions[None, 0],
        tips=np.where(present[:, :, None], positions[None, 1:], np.nan),
        tip_present=present,
    )
    config = robot.config.ik.model_copy(update={"max_iterations": 1})
    problem = hand_ik.HandIkProblem(tree, targets, np.array([0.0]), config, 0.005)
    sol = DlsHandSolver().solve(problem)
    assert np.allclose(sol.q[0], expected, atol=1e-6)
    assert np.array_equal(sol.q[0, cols], mid[cols])
    assert sol.reachable[0] and not sol.tip_reachable[0, INDEX]
    assert np.isnan(sol.tip_error_m[0, INDEX]) and float(np.nanmax(sol.tip_error_m[0])) < 1e-6


def test_frames_without_a_wrist_have_no_target_and_hold_every_joint(
    robot: LoadedDexterousRobot, policy: RetargetPolicy
) -> None:
    """손목이 없는 프레임은 손가락 끝 점이 있어도 목표가 없다 (손가락 끝 목표는 손목 목표에서
    잰다): target_present 0, 손가락별 목표도 0, 모든 관절은 직전 값, 오차 NaN. 첫 목표 앞
    프레임은 시작 자세(관절 범위 가운데)다. 손목이 하나도 없는 에피소드는 mark 정책에서도 버린다."""
    truth = truth_motion(robot)
    points = human_points(robot, truth.times, truth.positions)
    times = truth.times
    wrist = points.wrist.copy()
    wrist[:5] = np.nan
    wrist[20:30] = np.nan
    res = run(robot, policy, FingerPoints(times, wrist, points.tips))
    missing = np.isnan(wrist).any(axis=1)
    assert not res.target_present[missing].any() and res.target_present[~missing].all()
    assert not res.finger_target_present[missing].any()
    assert np.isnan(res.position_error_m[missing]).all()
    assert np.isnan(res.finger_position_error_m[missing]).all()
    assert not res.reachable[missing].any() and not res.finger_reachable[missing].any()
    mid = (robot.tree.lower + robot.tree.upper) / 2
    assert np.allclose(res.q[:5], mid)
    assert np.array_equal(res.q[20:30], np.tile(res.q[19], (10, 1)))
    assert res.report.frames_with_target == int((~missing).sum())
    assert within_speed_limits(robot, res, times)
    assert policy.unreachable is UnreachableAction.MARK
    nothing = np.full_like(points.wrist, np.nan)
    empty = run(robot, policy, FingerPoints(times, nothing, points.tips))
    assert empty.report.decision == "drop" and "목표" in (empty.report.reason or "")
    assert np.allclose(empty.q, mid) and not empty.finger_target_present.any()


def test_wrist_only_frames_move_the_arm_and_hold_all_fingers(
    robot: LoadedDexterousRobot, policy: RetargetPolicy
) -> None:
    """손가락 끝 점이 하나도 없고 손목만 있으면: 팔이 손목 위치만 따라가고(오차 1 mm 안) 손가락
    관절은 모두 시작 자세 그대로이며 손가락마다 목표 없음으로 남는다. mark면 표시만 하고 내보내고,
    drop_episode면 손가락 목표 비율이 기준에 못 미쳐 버린다."""
    truth = truth_motion(robot)
    points = human_points(robot, truth.times, truth.positions)
    only_wrist = FingerPoints(truth.times, points.wrist, np.full_like(points.tips, np.nan))
    res = run(robot, policy, only_wrist)
    arm = len(ARM_JOINTS)
    mid = (robot.tree.lower + robot.tree.upper) / 2
    assert res.target_present.all() and not res.finger_target_present.any()
    assert float(np.nanmax(res.position_error_m)) < 0.001
    assert np.allclose(res.q[:, arm:], mid[arm:])
    assert np.ptp(res.q[:, :arm], axis=0).max() > 0.1  # 팔은 움직인다
    assert res.report.decision == "keep"
    assert all(
        f.frames_with_target == 0 and f.reachable_fraction == 0.0 for f in res.report.fingers
    )
    assert all(f.position_error_max_m is None for f in res.report.fingers)
    strict = policy.model_copy(update={"unreachable": UnreachableAction.DROP_EPISODE})
    dropped = run(robot, strict, only_wrist)
    assert dropped.report.decision == "drop"
    assert "손가락 thumb_tip 목표가 있는 프레임 비율" in (dropped.report.reason or "")


def test_drop_episode_counts_unreachable_frames_per_finger(
    robot: LoadedDexterousRobot, policy: RetargetPolicy
) -> None:
    """drop_episode는 손가락마다 닿지 못한 비율도 본다: 새끼 끝 점이 12프레임에만 있고 그 절반이
    손가락 길이 밖이면, 프레임 기준(닿지 못한 프레임 / 손목 목표 프레임)은 상한 안이어도 그 손가락
    기준(그 손가락 목표 프레임 대비)이 상한을 넘어 버린다. 이유에 손가락이 적힌다."""
    truth = truth_motion(robot)
    points = human_points(robot, truth.times, truth.positions)
    tips = points.tips.copy()
    some = np.zeros(truth.times.size, dtype=np.bool_)
    some[24:36] = True
    tips[~some, PINKY] = np.nan
    bad = np.zeros_like(some)
    bad[30:36] = True
    offset = tips[bad, PINKY] - points.wrist[bad]
    tips[bad, PINKY] = points.wrist[bad] + offset * 2.0
    strict = policy.model_copy(
        update={
            "unreachable": UnreachableAction.DROP_EPISODE,
            "min_target_fraction": 0.1,
            "max_unreachable_fraction": 0.4,
        }
    )
    res = run(robot, strict, FingerPoints(truth.times, points.wrist, tips))
    pinky = res.report.fingers[PINKY]
    assert pinky.frames_with_target == 12 and pinky.reachable_frames <= 6
    assert res.report.reachable_fraction >= 0.6  # 프레임 기준은 통과
    assert res.report.decision == "drop"
    assert "손가락 pinky_tip이 도달하지 못한 비율" in (res.report.reason or "")


# ---------------------------------------------------------------- 속도 한계


def test_joint_speed_limits_hold_after_gaps_and_for_jumping_targets(
    robot: LoadedDexterousRobot, policy: RetargetPolicy
) -> None:
    """손목 점이 1·7프레임 끊긴 뒤와 에피소드 앞 10프레임이 빈 뒤에도 프레임 사이 관절 걸음은 URDF
    속도 x velocity_scale 안이다 (팔·손가락 모든 관절). 앞이 비면 첫 목표 프레임의 걸음이 잘리고
    표시된다. 검지 끝 목표가 프레임마다 편 자세 ↔ 굽힌 자세로 뛰면 걸음이 한계만큼만 가고 모든
    프레임이 velocity_limited로 표시된다."""
    truth = truth_motion(robot)
    points = human_points(robot, truth.times, truth.positions)
    times = truth.times
    for start, length in ((55, 1), (49, 7), (0, 10)):
        wrist = points.wrist.copy()
        wrist[start : start + length] = np.nan
        res = run(robot, policy, FingerPoints(times, wrist, points.tips))
        assert within_speed_limits(robot, res, times)
        if start == 0:
            assert res.velocity_limited[length] and res.report.velocity_limited_frames > 0
    cols = finger_columns(robot, INDEX)
    straight, bent = truth.q.copy(), truth.q.copy()
    straight[:, cols] = [0.0, 0.0, 0.1]
    bent[:, cols] = [0.0, 1.2, 1.2]
    even = (np.arange(times.size) % 2 == 0)[:, None]
    jumping = np.array([robot.tree.positions(row) for row in np.where(even, straight, bent)])
    res = run(robot, policy, human_points(robot, times, jumping))
    assert within_speed_limits(robot, res, times) and within_joint_limits(robot, res)
    assert res.velocity_limited[1:].all()
    assert res.report.velocity_limited_frames == times.size - 1
    assert not res.finger_reachable[1:, INDEX].all()
