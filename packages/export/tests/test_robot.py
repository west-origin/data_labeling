"""LeRobot 로봇 리타기팅 에피소드 (ADR 0047). 서비스·격리 환경 없이 돈다 (내장 IK 풀이기).

정답: 공개 시험 로봇 SO-101의 정기구학으로 만든 도달 가능한 끝단 궤적을
`dlp_fixtures.retarget`으로 월드 좌표 손 6점 라벨(승인)로 바꾼다. 에피소드 프레임마다 그 손을 로봇
관절로 옮긴 결과가 정답 끝단과 맞고, 검수 전 월드 3D는 기본 정책으로 들어가지 않으며, 월드 3D는
사람 state(카메라 좌표)에 섞이지
않는지 본다.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from dlp_retarget.ik import DlsIkSolver
from dlp_retarget.policy import RetargetPolicy, load_policy
from dlp_retarget.robot import LoadedRobot, Placement, PlacementMode, load_robot
from numpy.typing import NDArray

from dlp_export.lerobot import Episode, Vocab, build_episode, write_package
from dlp_export.policy import ExportPolicy
from dlp_export.pseudonym import Pseudonymizer
from dlp_export.robot import (
    ROBOT_ARRAYS,
    RobotEpisode,
    make_solver,
    retarget_meta,
    retarget_session,
    robot_arrays,
    robot_features,
    robot_override,
)
from dlp_export.source import label_states, select_labels
from dlp_fixtures.retarget import hand_points_for_grasp, world_hand_labels
from dlp_media.pts import build_pts_index
from dlp_schema.labels import LabelRecord, VerificationState
from dlp_schema.ontology import Ontology

from .conftest import ROOT, Scenario

PLACE = Placement(mode=PlacementMode.WORLD_FIXED, position_scale=0.5, base_xyz=(0.3, -0.1, -0.6))


@pytest.fixture(scope="module")
def robot() -> LoadedRobot:
    """SO-101 (배치는 world_fixed로 바꾼 사본: 정답 끝단을 월드 손으로 거꾸로 만들 수 있다)."""
    r = load_robot(ROOT, "so101", env={})
    arm = r.config.arm.model_copy(update={"placement": PLACE})
    return LoadedRobot(
        r.config.model_copy(update={"arm": arm}), r.chain, r.gripper, r.model_sha256, r.model_path
    )


@pytest.fixture(scope="module")
def rpolicy() -> RetargetPolicy:
    """저장소 retarget.yaml."""
    return load_policy(ROOT)


def _episode(
    scenario: Scenario,
    policy: ExportPolicy,
    ontology: Ontology,
    labels: list[LabelRecord],
    tmp: Path,
) -> Episode:
    """시나리오 블러본으로 사람 에피소드를 만든다."""
    video = tmp / "v.mp4"
    scenario.labeling.get_file("sessions/s1/blurred/bodycam.mp4", video)
    return build_episode(
        scenario.session, scenario.session.streams[0], build_pts_index(video), (64, 48), labels,
        Vocab.from_ontology(ontology), policy.lerobot,
    )  # fmt: skip


def _truth_labels(
    robot: LoadedRobot, times: list[int], state: VerificationState
) -> tuple[list[LabelRecord], NDArray[np.float64]]:
    """정답 관절 궤적 → (월드 손 6점 라벨, 정답 끝단 위치 (로봇 기준))."""
    chain = robot.chain
    t = np.array(times, dtype=np.float64)
    mid = (chain.lower + chain.upper) / 2
    q = mid + np.array([0.4, 0.3, 0.3, 0.3, 0.2]) * np.sin(
        2 * np.pi * t[:, None] / 900 + np.arange(5)
    )
    q[:, 1] -= 0.4
    q[:, 2] += 0.6
    tips = np.array([chain.fk(row) for row in q])
    pos_b = tips[:, :3, 3]
    grasp = np.array(PLACE.base_xyz) + pos_b / PLACE.position_scale
    a = tips[:, :3, :3] @ np.array(robot.config.arm.approach_axis)
    c = tips[:, :3, :3] @ np.array(robot.config.arm.closing_axis)
    pts = hand_points_for_grasp(grasp, a, c, np.full(t.size, 0.04))
    return world_hand_labels("s1", times, pts, state=state), pos_b


def test_world_trajectories_do_not_leak_into_human_state(
    scenario: Scenario, policy: ExportPolicy, ontology: Ontology, robot: LoadedRobot, tmp_path: Path
) -> None:
    """같은 (손, 부위)의 월드 좌표 궤적이 있어도 사람 모드 observation.state(카메라 좌표 3D)는
    그대로다."""
    labels = select_labels(scenario.labels, policy, label_states(policy, False))
    base = _episode(scenario, policy, ontology, labels, tmp_path)
    world, _ = _truth_labels(robot, scenario.times, VerificationState.HUMAN_APPROVED)
    mixed = _episode(scenario, policy, ontology, [*labels, *world], tmp_path)
    assert np.array_equal(base.state, mixed.state)
    assert not (mixed.used & {x.label_id for x in world})


def test_robot_episode_follows_reviewed_world_hand(
    scenario: Scenario,
    policy: ExportPolicy,
    ontology: Ontology,
    robot: LoadedRobot,
    rpolicy: RetargetPolicy,
    tmp_path: Path,
) -> None:
    """승인된 월드 손 라벨: 손 점이 있는 모든 프레임에서 목표가 있고 도달하며, 끝단이 정답과
    5 mm 안이다 (에피소드 시각은 고정 fps라 손 표본 사이를 보간한다). state=관절, action=다음
    관절, 검증 등급 2, 쓴 라벨은 손목·엄지 끝·검지 끝 셋이다. 묶음에는 retarget.* 배열과
    robot_type이 들어간다."""
    world, pos_b = _truth_labels(robot, scenario.times, VerificationState.HUMAN_APPROVED)
    labels = select_labels([*scenario.labels, *world], policy, label_states(policy, False))
    ep = _episode(scenario, policy, ontology, labels, tmp_path)
    stream = scenario.session.streams[0]
    rob = retarget_session(ep, stream, labels, robot, rpolicy, policy.lerobot, DlsIkSolver())
    res = rob.result
    inside = (ep.times_ms >= scenario.times[0]) & (ep.times_ms <= scenario.times[-1])
    assert res.target_present[inside].all() and res.reachable[inside].all()
    assert set(rob.verification[inside].tolist()) == {2}
    assert rob.used == {
        x.label_id
        for x in world
        if x.label_id.split("-")[-1] in ("wrist", "thumb_tip", "index_tip")
    }
    tips = np.array([robot.chain.fk(row[:5]) for row in res.q])[:, :3, 3]
    truth = np.stack(
        [np.interp(ep.times_ms, scenario.times, pos_b[:, i]) for i in range(3)], axis=1
    )
    # 손 점 선형 보간과 정기구학 경로의 차이(곡률)만큼은 남는다
    assert np.linalg.norm(tips[inside] - truth[inside], axis=1).max() < 0.005
    arrays = robot_arrays(ep, rob)
    assert arrays["state"].shape == (ep.times_ms.size, 6)
    assert np.allclose(arrays["action"][:-1], arrays["state"][1:])
    assert set(ROBOT_ARRAYS.values()) <= set(arrays)
    feats = robot_features(policy.lerobot, robot, 48, 64)
    assert feats["observation.state"]["names"] == list(robot.joint_names)
    pkg = write_package(
        [(scenario.session, tmp_path / "v.mp4", ep)], policy, (64, 48), tmp_path / "pkg",
        Pseudonymizer(None), robot_override([(ep, rob)], policy.lerobot, robot),
    )  # fmt: skip
    spec = json.loads(pkg.read_text("utf-8"))
    assert spec["robot_type"] == "so101" and "retarget.reachable" in spec["frame_arrays"]
    npz = np.load(spec["episodes"][0]["npz"])
    assert npz["state"].shape == (ep.times_ms.size, 6) and npz["reachable"].shape == (
        ep.times_ms.size,
        1,
    )
    meta = retarget_meta(
        robot, rpolicy, make_solver(ROOT, rpolicy), [res.report.model_dump(mode="json")]
    )
    assert meta["model_sha256"] == robot.model_sha256 and meta["retarget_version"].startswith(
        "retarget-dls"
    )
    assert math.isclose(meta["episodes"][0]["reachable_fraction"], res.report.reachable_fraction)


def test_unreviewed_world_hand_is_not_used_by_default(
    scenario: Scenario,
    policy: ExportPolicy,
    ontology: Ontology,
    robot: LoadedRobot,
    rpolicy: RetargetPolicy,
    tmp_path: Path,
) -> None:
    """검수 전 월드 손 3D는 기본 검증 정책으로 고르지 않아 목표가 없다 (지어내지 않음, 보고에
    0으로 남음). --include-unreviewed와 같은 선택이면 목표가 생긴다."""
    world, _ = _truth_labels(robot, scenario.times, VerificationState.UNREVIEWED)
    stream = scenario.session.streams[0]
    for include, expect in ((False, 0), (True, 1)):
        labels = select_labels([*scenario.labels, *world], policy, label_states(policy, include))
        ep = _episode(scenario, policy, ontology, labels, tmp_path)
        rob: RobotEpisode = retarget_session(
            ep, stream, labels, robot, rpolicy, policy.lerobot, DlsIkSolver()
        )
        assert (rob.result.report.frames_with_target > 0) == bool(expect)


@pytest.mark.isolated_env
def test_official_lerobot_loader_reads_robot_episode(
    scenario: Scenario,
    policy: ExportPolicy,
    ontology: Ontology,
    robot: LoadedRobot,
    rpolicy: RetargetPolicy,
    tmp_path: Path,
) -> None:
    """격리 환경(LeRobot 공식 API)으로 로봇 모드 에피소드를 쓰고 공식 로더로 다시 읽는다: state는
    관절 6개, retarget.* 특징과 사람 state가 읽히고, robot_type이 로봇 ID다
    (`make test-isolated`)."""
    from dlp_export.lerobot import run_script

    world, _ = _truth_labels(robot, scenario.times, VerificationState.HUMAN_APPROVED)
    labels = select_labels([*scenario.labels, *world], policy, label_states(policy, False))
    ep = _episode(scenario, policy, ontology, labels, tmp_path)
    rob = retarget_session(
        ep, scenario.session.streams[0], labels, robot, rpolicy, policy.lerobot, DlsIkSolver()
    )
    lp = policy.lerobot
    pkg = write_package(
        [(scenario.session, tmp_path / "v.mp4", ep)], policy, (64, 48), tmp_path / "pkg",
        Pseudonymizer(None), robot_override([(ep, rob)], lp, robot),
    )  # fmt: skip
    dest = tmp_path / "lerobot"
    run_script(ROOT, lp, "lerobot_write.py", str(pkg), str(dest))
    info = json.loads((dest / "meta" / "info.json").read_text())
    assert info["robot_type"] == "so101"
    assert info["features"]["observation.state"]["names"] == list(robot.joint_names)
    check = json.loads(run_script(ROOT, lp, "lerobot_check.py", str(dest), lp.repo_id))
    assert check["episodes"] == 1 and check["frames"] == ep.times_ms.size
    assert {"retarget.reachable", "retarget.target_present", "observation.human_state"} <= set(
        check["features"]
    )
    for s in check["samples"]:
        k = s["index"]
        assert np.allclose(s["state"], rob.result.q[k], atol=1e-5)
        assert s["extra"]["retarget.reachable"] == [float(rob.result.reachable[k])]
        assert len(s["extra"]["observation.human_state"]) == ep.state.shape[1]
