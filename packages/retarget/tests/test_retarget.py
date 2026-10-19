"""로봇 리타기팅 (ADR 0047): URDF 기구학, 로봇 등록 검사, IK 정확도, 도달하지 못한 구간 표시.

정답: 공개 시험 로봇 SO-101의 정기구학으로 만든 도달 가능한 끝단 궤적(관절 범위 안 부드러운 관절
궤적)을 `dlp_fixtures.retarget`으로 사람 손 점으로 바꾼다. 리타기팅 결과의 끝단 경로를 정답과
비교한다. 서비스 없이 돈다.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

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
    OrientationMode,
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


def _closing_error_deg(
    robot: LoadedRobot, q: NDArray[np.float64], q_truth: NDArray[np.float64]
) -> NDArray[np.float64]:
    """프레임별 끝 링크 닫힘 축과 정답 닫힘 축 사이 각 (도)."""
    c_local = np.array(robot.config.arm.closing_axis)
    got = np.array([robot.chain.fk(row[:5])[:3, :3] @ c_local for row in q])
    want = np.array([robot.chain.fk(row)[:3, :3] @ c_local for row in q_truth])
    return np.degrees(np.arccos(np.clip(np.sum(got * want, axis=1), -1, 1)))


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
    # full 모드: 닫힘 축(사람 손 롤)과 wrist_roll도 정답을 따른다 (approach 모드였을 때는 최대 22도,
    # wrist_roll 0.39 rad 어긋났다, F-COE-40)
    assert _closing_error_deg(r, res.q, q_truth).max() < 1.0
    assert np.abs(res.q[:, 4] - q_truth[:, 4]).max() < 0.01
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


def test_missing_hand_points_hold_and_are_marked(
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
    assert v.startswith("retarget-dls-dls2+r") and f"+m{robot.model_sha256[:12]}" in v
    assert retarget_version(with_placement(robot, FIXED), policy, DlsIkSolver()) != v


# ---------------------------------------------------------------- 롤·재시작·그리퍼 (ADR 0053)


def _yawed(
    pts: dict[str, NDArray[np.float64]], yaw: NDArray[np.float64]
) -> dict[str, NDArray[np.float64]]:
    """손 점을 프레임마다 잡기 점 둘레 월드 z축으로 yaw만큼 돌린다 (사람 손 접근 축의 요가 팔과
    다름)."""
    grasp = (pts["thumb_tip"] + pts["index_tip"]) / 2
    c, s = np.cos(yaw), np.sin(yaw)
    out: dict[str, NDArray[np.float64]] = {}
    for k, v in pts.items():
        d = v - grasp
        rotated = [c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1], d[:, 2]]
        out[k] = grasp + np.stack(rotated, axis=1)
    return out


def test_roll_follows_the_human_hand_when_approach_yaw_differs(
    robot: LoadedRobot, policy: RetargetPolicy
) -> None:
    """사람 손 접근 축의 요가 팔 평면과 ±20도 어긋나도(5축 팔은 요를 맞추지 못함) 롤은 맞춘다:
    사람 닫힘 축을 로봇 접근 축에 수직으로 투영한 방향과 로봇 닫힘 축의 차이가 3도 안, 모두 도달
    (approach 모드였을 때 평균 53도·최대 90도, F-COE-40)."""
    r = with_placement(robot, FIXED)
    times, q_truth = truth_motion(r)
    pts, _, _ = hand_points_of(r, q_truth, np.full(times.size, 0.04))
    yawed = _yawed(pts, np.radians(20) * np.sin(2 * np.pi * times / 1900))
    res = _run(r, policy, times, yawed)
    assert res.report.reachable_fraction == 1.0
    cy, sy = np.cos(FIXED.base_yaw_rad), np.sin(FIXED.base_yaw_rad)
    r_bw = np.array([[cy, sy, 0.0], [-sy, cy, 0.0], [0.0, 0.0, 1.0]])
    human = (yawed["index_tip"] - yawed["thumb_tip"]) @ r_bw.T
    errs: list[float] = []
    for row, h in zip(res.q, human, strict=True):
        rot = r.chain.fk(row[:5])[:3, :3]
        a = rot @ np.array(r.config.arm.approach_axis)
        c = rot @ np.array(r.config.arm.closing_axis)
        hp: NDArray[np.float64] = h - float(h @ a) * a
        cos = float(hp @ c) / float(np.linalg.norm(hp))
        errs.append(float(np.degrees(np.arccos(min(max(cos, -1.0), 1.0)))))
    assert max(errs) < 3.0


def test_restart_after_missing_frames_is_continuous_and_speed_limited(
    robot: LoadedRobot, policy: RetargetPolicy
) -> None:
    """손 목표가 1·7프레임 끊긴 뒤 다시 시작하는 프레임: 직전 해에 가까운 해를 쓰고(끊김이 없을 때와
    wrist_roll 차이 0.1 rad 안), 직전 프레임 대비 걸음이 속도 한계 안이다. 에피소드 앞 10프레임에
    손이 없으면 첫 목표 프레임의 걸음도 한계 안으로 잘리고 표시된다 (F-COE-41)."""
    r = with_placement(robot, FIXED)
    times, q_truth = truth_motion(r)
    pts, _, _ = hand_points_of(r, q_truth, np.full(times.size, 0.04))
    full = _run(r, policy, times, pts)
    vmax = np.append(r.chain.velocity, r.gripper.velocity) * r.config.ik.velocity_scale
    dt = np.diff(times) / 1000
    for start, length in ((55, 1), (49, 7), (0, 10)):
        cut = {k: v.copy() for k, v in pts.items()}
        for v in cut.values():
            v[start : start + length] = np.nan
        res = _run(r, policy, times, cut)
        steps = np.abs(np.diff(res.q, axis=0))
        assert (steps <= vmax * dt[:, None] + 1e-9).all()
        k = start + length
        if start > 0:
            assert np.abs(res.q[k:, 4] - full.q[k:, 4]).max() < 0.1
        else:
            assert res.velocity_limited[k] and res.report.velocity_limited_frames > 0


def test_gripper_speed_is_limited_and_flagged(robot: LoadedRobot, policy: RetargetPolicy) -> None:
    """엄지-검지 거리가 프레임마다 1 cm ↔ 9 cm로 흔들리면(깊이 잡음) 그리퍼 걸음은 URDF 속도 x
    velocity_scale 안으로 잘리고 그 프레임이 velocity_limited로 표시된다 (F-COE-43)."""
    r = with_placement(robot, FIXED)
    times, q_truth = truth_motion(r)
    opening = np.where(np.arange(times.size) % 2 == 0, 0.01, 0.09)
    pts, _, _ = hand_points_of(r, q_truth, opening)
    res = _run(r, policy, times, pts)
    vg = r.gripper.velocity * r.config.ik.velocity_scale
    steps = np.abs(np.diff(res.q[:, 5]))
    assert (steps <= vg * np.diff(times) / 1000 + 1e-9).all()
    # 열 때마다(닫힘 0 → 열림 1.5 rad 목표) 한 프레임 한계만큼만 열리고, 닫을 때는 그만큼 돌아와
    # 닿는다
    assert res.velocity_limited[1::2].all()
    assert res.report.velocity_limited_frames == int(res.velocity_limited.sum()) >= times.size // 2


def test_episode_without_any_target_is_dropped_even_under_mark(
    robot: LoadedRobot, policy: RetargetPolicy
) -> None:
    """월드 손 3D가 하나도 없는 에피소드는 기본 mark 정책에서도 버리기로 결정한다 (관절이 모두 시작
    자세라 로봇 데이터로 의미가 없다, F-COE-42). 목표 앞 프레임은 시작 자세·닫힌 그리퍼다."""
    assert policy.unreachable is UnreachableAction.MARK
    n = 30
    nan = np.full((n, 3), np.nan)
    times = np.arange(n) * 1000 / FPS
    res = retarget_episode(HandPoints(times, nan, nan, nan), robot, policy, DlsIkSolver())
    assert res.report.decision == "drop" and "목표" in (res.report.reason or "")
    mid = (robot.chain.lower + robot.chain.upper) / 2
    assert np.allclose(res.q[:, :5], mid)
    assert np.allclose(res.q[:, 5], robot.config.arm.gripper.closed)


def test_orientation_none_reports_no_rotation_error(
    robot: LoadedRobot, policy: RetargetPolicy
) -> None:
    """orientation_mode none은 회전을 재지 않으므로 회전 오차가 NaN(보고 None)이다 (0이면 완벽히
    맞은 것과 구분되지 않는다, F-COE-46)."""
    r = with_placement(robot, FIXED)
    ik = r.config.ik.model_copy(update={"orientation_mode": OrientationMode.NONE})
    r = LoadedRobot(
        r.config.model_copy(update={"ik": ik}), r.chain, r.gripper, r.model_sha256, r.model_path
    )
    times, q_truth = truth_motion(r, seconds=0.5)
    pts, _, _ = hand_points_of(r, q_truth, np.full(times.size, 0.04))
    res = _run(r, policy, times, pts)
    assert np.isnan(res.orientation_error_rad).all()
    assert res.report.orientation_error_max_rad is None and res.report.reachable_fraction == 1.0


# ---------------------------------------------------------------- 설정·URDF 검증


def _acme(tmp_path: Path) -> dict[str, Any]:
    """tmp_path에 SO-101 URDF 사본을 두고 고객 설정 사전 (public false)을 돌려준다."""
    src = ROOT / "config" / "robots"
    cfg: dict[str, Any] = yaml.safe_load((src / "so101.yaml").read_text("utf-8"))
    shutil.copy(src / "models" / "so101" / "so101_new_calib.urdf", tmp_path / "arm.urdf")
    cfg["robot_id"] = "acme"
    cfg["model"].update(path="arm.urdf", public=False, license="고객 계약", source="계약 C-1")
    return cfg


def _write(tmp_path: Path, data: dict[str, Any]) -> Path:
    """tmp_path/acme.yaml에 설정을 쓴다."""
    path = tmp_path / "acme.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), "utf-8")
    return path


@pytest.mark.parametrize(
    ("axes", "match"),
    [
        ({"approach_axis": [0.0, 0.0, 2.0]}, "단위 벡터"),
        ({"approach_axis": [1.0, 1.0, 0.0]}, "단위 벡터"),
        ({"approach_axis": [0.0, 0.0, 0.0]}, "단위 벡터"),
        ({"closing_axis": [0.0, 0.0, 1.0]}, "직교"),
    ],
)
def test_arm_axes_must_be_orthonormal(
    tmp_path: Path, axes: dict[str, list[float]], match: str
) -> None:
    """접근·닫힘 축은 단위 벡터이고 서로 직교해야 한다 (길이가 1이 아니면 회전 오차와 도달 판정이
    틀린다, F-COE-44)."""
    cfg = _acme(tmp_path)
    with pytest.raises(ValueError, match=match):
        load_robot_file(_write(tmp_path, {**cfg, "arm": {**cfg["arm"], **axes}}))


@pytest.mark.parametrize(
    ("gripper", "match"),
    [
        ({"open": 85.0}, "URDF 한계"),
        ({"joint": "wrist_roll"}, "겹칩니다"),
        ({"joint": "gripper_frame_joint"}, "움직이는 독립 관절"),
    ],
)
def test_gripper_joint_is_checked_against_urdf(
    tmp_path: Path, gripper: dict[str, Any], match: str
) -> None:
    """그리퍼: closed·open이 URDF 한계 안(도 단위 거부), 팔 관절과 겹치지 않음, 고정 관절 거부
    (F-COE-44)."""
    cfg = _acme(tmp_path)
    grip = {**cfg["arm"]["gripper"], **gripper}
    with pytest.raises(RobotError, match=match):
        load_robot_file(_write(tmp_path, {**cfg, "arm": {**cfg["arm"], "gripper": grip}}))


def test_robot_id_cannot_escape_and_private_model_cannot_live_in_repo(tmp_path: Path) -> None:
    """robot_id에 경로 구분자·`..`를 쓸 수 없고, 저장소 밖 비공개 설정이 저장소 안 모델 파일을
    가리키면 거부한다 (F-COE-45)."""
    for bad in ("../x", "a/b", ".hidden", ""):
        with pytest.raises(RobotError, match="로봇 ID"):
            load_robot(ROOT, bad, env={})
    cfg = _acme(tmp_path)
    inside = ROOT / "config" / "robots" / "models" / "so101" / "so101_new_calib.urdf"
    cfg["model"]["path"] = str(inside)
    _write(tmp_path, cfg)
    with pytest.raises(RobotError, match="저장소 밖"):
        load_robot(ROOT, "acme", env={ROBOTS_DIR_ENV: str(tmp_path)})


def test_repo_robot_models_are_only_public_test_models() -> None:
    """config/robots/models 아래 파일은 공개 설정이 가리키는 모델이거나 그 디렉터리의 LICENSE뿐이다
    (설정 없이 커밋된 고객 설계 파일을 잡는다, F-COE-45)."""
    robots = ROOT / "config" / "robots"
    allowed: set[Path] = set()
    for path in robots.glob("*.yaml"):
        loaded = load_robot_file(path, inside_repo=ROOT)
        assert loaded.model_path is not None
        allowed |= {loaded.model_path.resolve(), (loaded.model_path.parent / "LICENSE").resolve()}
    files = {f.resolve() for f in (robots / "models").rglob("*") if f.is_file()}
    assert files <= allowed, sorted(str(f) for f in files - allowed)


_URDF = """<robot name="t">
  <link name="base"/><link name="l1"/><link name="l2"/><link name="f1"/>
  <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/><axis xyz="0 0 1"/>
    <limit effort="1" velocity="1" lower="-1" upper="1"/></joint>
  <joint name="j2" type="{kind}"><parent link="l1"/><child link="l2"/><axis xyz="0 1 0"/>
    {limit}{mimic}</joint>
  <joint name="finger" type="revolute"><parent link="l2"/><child link="f1"/><axis xyz="1 0 0"/>
    <limit effort="1" velocity="1" lower="0" upper="1"/><mimic joint="j1"/></joint>
</robot>"""
_LIM = '<limit effort="1" velocity="1" lower="-1" upper="1"/>'


@pytest.mark.parametrize(
    ("kind", "limit", "mimic", "match"),
    [
        ("revolute", _LIM, '<mimic joint="j1"/>', "mimic"),
        ("floating", "", "", "지원하지 않는"),
        ("planar", "", "", "지원하지 않는"),
        ("revolute", "", "", "limit"),
        ("prismatic", '<limit effort="1" velocity="1"/>', "", "limit"),
    ],
)
def test_urdf_chain_rejects_mimic_unsupported_and_missing_limits(
    tmp_path: Path, kind: str, limit: str, mimic: str, match: str
) -> None:
    """연쇄 안 mimic·floating·planar 관절과 lower/upper 없는 revolute·prismatic을 거부한다. 연쇄 밖
    mimic(그리퍼 손가락)은 막지 않는다 (F-COE-47)."""
    path = tmp_path / "t.urdf"
    path.write_text(_URDF.format(kind=kind, limit=limit, mimic=mimic), "utf-8")
    with pytest.raises(UrdfError, match=match):
        build_chain(load_joints(path), "base", "l2")
    path.write_text(_URDF.format(kind="continuous", limit="", mimic=""), "utf-8")
    chain = build_chain(load_joints(path), "base", "l2")
    assert chain.names == ("j1", "j2") and np.isinf(chain.upper[1])
    assert load_joints(path)["finger"].mimic == "j1"
