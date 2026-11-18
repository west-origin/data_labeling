"""LeRobot 다지 손 로봇 리타기팅 에피소드 (ADR 0068). 서비스·격리 환경 없이 돈다 (내장 풀이기).

정답: 직접 쓴 시험 로봇(`dlp_fixtures.dexhand`)의 정기구학으로 만든 손목·손가락 끝 위치를 월드 좌표
손 6점 라벨로 바꾼다. 에피소드 프레임마다 그 손을 팔 관절 + 손가락 관절로 옮긴 결과가 정답 위치와
맞는지, state·action·`retarget.*`·meta가 손가락 대응 순서대로 채워지는지, 끝 점 라벨이 없거나 검수
전인 손가락만 목표 없음으로 표시되는지 본다. 평행 그리퍼 로봇의 결과는 `test_robot.py`가 본다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from dlp_retarget.hand_ik import DlsHandSolver
from dlp_retarget.ik import DlsIkSolver
from dlp_retarget.policy import RetargetPolicy, SolverName, load_policy
from dlp_retarget.retarget import HandRetargetResult
from dlp_retarget.robot import (
    ROBOTS_DIR_ENV,
    LoadedDexterousRobot,
    LoadedRobot,
    RobotError,
    load_any_robot_file,
)
from numpy.typing import NDArray

from dlp_export import runner
from dlp_export.lerobot import FRAME_ARRAYS, Episode, Vocab, build_episode, write_package
from dlp_export.policy import ExportPolicy
from dlp_export.pseudonym import Pseudonymizer
from dlp_export.robot import (
    HAND_ARRAYS,
    ROBOT_ARRAYS,
    load_retargeter,
    retarget_hand_session,
    retarget_meta,
    robot_arrays,
    robot_features,
    robot_override,
)
from dlp_export.runner import check_hand_parts
from dlp_export.source import (
    ExportError,
    ExportSession,
    ExportSource,
    label_states,
    select_labels,
)
from dlp_fixtures.dexhand import (
    ARM_JOINTS,
    FINGERS,
    finger_joints,
    human_points_from_robot,
    smooth_joint_motion,
    write_robot,
)
from dlp_fixtures.retarget import world_hand_labels
from dlp_media.pts import build_pts_index
from dlp_schema.dataset import DatasetVersion, Split
from dlp_schema.labels import LabelRecord, VerificationState
from dlp_schema.ontology import Ontology
from dlp_schema.session import Stream
from dlp_schema.testing import FIXED_TIME

from .conftest import ROOT, Scenario

# 로봇 기준을 세션 월드의 한 자리에 두고 팔 궤적을 0.8배로 줄이는 배치, 손은 1.1배
PLACE: dict[str, Any] = {
    "mode": "world_fixed",
    "position_scale": 0.8,
    "base_xyz": [0.3, -0.1, -0.6],
    "base_yaw_rad": 0.3,
}
HAND_SCALE = 1.1
THREE = ("thumb", "index", "middle")


def _robot(directory: Path, fingers: tuple[str, ...] = FINGERS) -> LoadedDexterousRobot:
    """시험 로봇을 디렉터리에 쓰고 읽는다 (`PLACE` 배치, 손 배율 `HAND_SCALE`)."""
    path = write_robot(directory, placement=PLACE, hand_scale=HAND_SCALE, fingers=fingers)
    robot = load_any_robot_file(path)
    assert isinstance(robot, LoadedDexterousRobot)
    return robot


@pytest.fixture(scope="module")
def hand5(tmp_path_factory: pytest.TempPathFactory) -> LoadedDexterousRobot:
    """다섯 손가락을 모두 대응한 시험 로봇 (정답 손 6점도 이 로봇의 정기구학으로 만든다)."""
    return _robot(tmp_path_factory.mktemp("hand5"))


@pytest.fixture(scope="module")
def rpolicy() -> RetargetPolicy:
    """저장소 retarget.yaml."""
    return load_policy(ROOT)


def _truth(
    hand5: LoadedDexterousRobot, times: list[int]
) -> tuple[dict[str, NDArray[np.float64]], NDArray[np.float64]]:
    """정답 관절 궤적 → (월드 손 6점 (부위 → (n, 3)), 로봇 기준 손목·다섯 손가락 끝 (n, 6, 3))."""
    tree = hand5.tree
    t = np.array(times, dtype=np.float64)
    q = smooth_joint_motion(tree.lower, tree.upper, t, amplitude_rad=0.2)
    pos = np.array([tree.positions(row) for row in q])
    wrist, tips = human_points_from_robot(
        pos[:, 0],
        pos[:, 1:],
        base_xyz=PLACE["base_xyz"],
        base_yaw_rad=PLACE["base_yaw_rad"],
        position_scale=PLACE["position_scale"],
        hand_scale=HAND_SCALE,
    )
    points = {"wrist": wrist, **{f"{f}_tip": tips[:, i] for i, f in enumerate(FINGERS)}}
    return points, pos


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


def _part(label: LabelRecord) -> str:
    """월드 손 라벨의 부위 이름 (`world_hand_labels`의 라벨 ID 끝)."""
    return label.label_id.split("-")[-1]


def test_hand_robot_episode_follows_wrist_and_mapped_fingertips(
    scenario: Scenario,
    policy: ExportPolicy,
    ontology: Ontology,
    hand5: LoadedDexterousRobot,
    rpolicy: RetargetPolicy,
    tmp_path: Path,
) -> None:
    """손가락 셋(엄지·검지·중지)만 대응한 로봇: 손 점이 있는 모든 프레임에서 손목·세 손가락 목표가
    있고 도달하며, 손목·손가락 끝 위치가 정답과 5 mm 안이다 (에피소드 시각은 고정 fps라 손 표본
    사이를 보간한다). state = 팔 관절 6 + 손가락 관절 9 (대응 순서), action = 다음 관절, 손가락별
    배열은 (n, 3)이다. 쓴 라벨은 손목과 대응한 세 손가락 끝뿐이고 약지·새끼 끝 라벨은 쓰지 않는다.
    묶음·meta에 손가락 대응, 손 배율, 허용 오차, 손가락별 통계가 들어가고 모델 경로는 없다."""
    robot = _robot(tmp_path / "robot", THREE)
    points, pos = _truth(hand5, scenario.times)
    world = world_hand_labels("s1", scenario.times, points)
    labels = select_labels([*scenario.labels, *world], policy, label_states(policy, False), ())
    ep = _episode(scenario, policy, ontology, labels, tmp_path)
    stream = scenario.session.streams[0]
    solver = DlsHandSolver()
    rob = retarget_hand_session(ep, stream, labels, robot, rpolicy, policy.lerobot, solver)
    res = rob.result
    assert isinstance(res, HandRetargetResult)
    n = ep.times_ms.size
    inside = (ep.times_ms >= scenario.times[0]) & (ep.times_ms <= scenario.times[-1])
    assert res.target_present[inside].all() and res.reachable[inside].all()
    assert res.finger_target_present[inside].all() and res.finger_reachable[inside].all()
    assert not res.target_present[~inside].any() and not res.finger_target_present[~inside].any()
    assert set(rob.verification[inside].tolist()) == {2}
    assert set(rob.verification[~inside].tolist()) <= {-1}
    mapped = ("wrist", *(f"{f}_tip" for f in THREE))
    assert rob.used == {x.label_id for x in world if _part(x) in mapped}
    got = np.array([robot.tree.positions(row) for row in res.q])
    for i in range(4):  # 손목, 엄지·검지·중지 끝 (정답 6점의 앞 네 점과 같은 순서)
        truth = np.stack(
            [np.interp(ep.times_ms, scenario.times, pos[:, i, a]) for a in range(3)], axis=1
        )
        # 손 점 선형 보간과 정기구학 경로의 차이(곡률)만큼은 남는다
        assert np.linalg.norm(got[inside, i] - truth[inside], axis=1).max() < 0.005

    joints = [*ARM_JOINTS, *(name for f in THREE for name in finger_joints(f))]
    arrays = robot_arrays(ep, rob)
    assert arrays["state"].shape == (n, 15) and arrays["state"].dtype == np.float32
    assert np.allclose(arrays["action"][:-1], arrays["state"][1:])
    assert set(ROBOT_ARRAYS.values()) | set(HAND_ARRAYS.values()) <= set(arrays)
    for key, dtype in (("finger_target_present", np.int64), ("finger_reachable", np.int64)):
        assert arrays[key].shape == (n, 3) and arrays[key].dtype == dtype
    errors = arrays["finger_position_error_m"]
    assert errors.shape == (n, 3) and errors.dtype == np.float32
    assert (errors[~inside] == -1).all() and (errors[inside] >= 0).all()
    assert (arrays["orientation_error_rad"] == -1).all()  # 방향은 재지 않는다
    feats = robot_features(policy.lerobot, robot, 48, 64)
    assert feats["observation.state"]["names"] == joints == feats["action"]["names"]
    for name in HAND_ARRAYS:
        assert feats[name]["shape"] == [3] and feats[name]["names"] == list(mapped[1:])
    assert feats["retarget.finger_position_error_m"]["dtype"] == "float32"

    pkg = write_package(
        [(scenario.session, tmp_path / "v.mp4", ep)], policy, (64, 48), tmp_path / "pkg",
        Pseudonymizer(None), robot_override([(ep, rob)], policy.lerobot, robot),
    )  # fmt: skip
    spec = json.loads(pkg.read_text("utf-8"))
    assert spec["robot_type"] == "dexhand"
    assert spec["frame_arrays"] == {**FRAME_ARRAYS, **ROBOT_ARRAYS, **HAND_ARRAYS}
    assert set(spec["frame_arrays"]) == set(spec["features"]) - {"observation.images.bodycam"}
    npz = np.load(spec["episodes"][0]["npz"])
    assert npz["state"].shape == (n, 15) and npz["finger_reachable"].shape == (n, 3)

    meta = retarget_meta(robot, rpolicy, solver, [res.report.model_dump(mode="json")])
    assert meta["retarget_version"].startswith(f"retarget-dls-dlshand1+r{robot.config.digest()}")
    assert meta["solver"] == "dls-dlshand1" and meta["joint_names"] == joints
    assert meta["hand_parts"] == {"wrist": "wrist", "fingers": list(mapped[1:])}
    hand = meta["dexterous_hand"]
    assert hand["wrist_link"] == "hand" and hand["arm_joints"] == list(ARM_JOINTS)
    assert hand["hand_scale"] == HAND_SCALE and hand["fingertip_tolerance_m"] == 0.005
    assert hand["fingers"] == [
        {"human_part": f"{f}_tip", "tip_link": f"{f}_tip", "joints": list(finger_joints(f))}
        for f in THREE
    ]
    assert hand["unmapped_joints"] == [*finger_joints("ring"), *finger_joints("pinky")]
    assert meta["placement"]["position_scale"] == PLACE["position_scale"]
    assert meta["ik"] == robot.config.ik.model_dump(mode="json")
    assert "fingertip_tolerance_m" in meta["reachable_definition"]
    assert "not measured" in meta["orientation_error_definition"]
    assert meta["camera_pose"] == "automatic_unreviewed"
    stats = meta["episodes"][0]["fingers"]
    assert [f["human_part"] for f in stats] == list(mapped[1:])
    assert all(f["frames_with_target"] == int(inside.sum()) for f in stats)
    assert all(f["reachable_fraction"] == 1.0 and f["frames_out_of_reach"] == 0 for f in stats)
    assert ".urdf" not in json.dumps(meta)


def test_fingertips_without_a_usable_label_are_marked_per_finger(
    scenario: Scenario,
    policy: ExportPolicy,
    ontology: Ontology,
    hand5: LoadedDexterousRobot,
    rpolicy: RetargetPolicy,
    tmp_path: Path,
) -> None:
    """약지 끝 라벨이 검수 전이고 새끼 끝 라벨이 앞 절반에만 있으면: 기본 검증 정책에서 약지는 모든
    프레임에 목표가 없고(그 관절은 시작 자세 그대로), 새끼는 라벨이 끝난 뒤부터 목표가 없고 그
    관절은 유지된다. 나머지 손가락과 손목은 계속 목표가 있고 도달한다. 검수 전 약지 라벨은 쓴
    라벨에 없다. --include-unreviewed와 같은 선택이면 약지에도 목표가 생기고 그 프레임의 검증
    등급이 0으로 내려간다."""
    points, _ = _truth(hand5, scenario.times)
    half = len(scenario.times) // 2
    partial = {**points, "pinky_tip": points["pinky_tip"].copy()}
    partial["pinky_tip"][half:] = np.nan
    approved = world_hand_labels("s1", scenario.times, partial)
    unreviewed = world_hand_labels(
        "s1", scenario.times, partial, state=VerificationState.UNREVIEWED
    )
    world = [*(x for x in approved if _part(x) != "ring_tip")]
    world += [x for x in unreviewed if _part(x) == "ring_tip"]
    stream = scenario.session.streams[0]
    ring, pinky = FINGERS.index("ring"), FINGERS.index("pinky")
    mid = (hand5.tree.lower + hand5.tree.upper) / 2
    ring_cols = [hand5.joint_names.index(name) for name in finger_joints("ring")]
    pinky_cols = [hand5.joint_names.index(name) for name in finger_joints("pinky")]

    labels = select_labels([*scenario.labels, *world], policy, label_states(policy, False), ())
    ep = _episode(scenario, policy, ontology, labels, tmp_path)
    rob = retarget_hand_session(ep, stream, labels, hand5, rpolicy, policy.lerobot, DlsHandSolver())
    res = rob.result
    assert isinstance(res, HandRetargetResult)
    inside = (ep.times_ms >= scenario.times[0]) & (ep.times_ms <= scenario.times[-1])
    early = ep.times_ms <= scenario.times[half - 1]
    assert res.target_present[inside].all() and res.reachable[inside].all()
    assert res.finger_target_present[inside][:, :ring].all()
    assert not res.finger_target_present[:, ring].any()
    assert np.allclose(res.q[:, ring_cols], mid[ring_cols])
    assert res.finger_target_present[inside & early, pinky].all()
    assert not res.finger_target_present[~early, pinky].any()
    held = res.q[int(np.flatnonzero(early)[-1]), pinky_cols]
    assert np.array_equal(res.q[~early][:, pinky_cols], np.tile(held, (int((~early).sum()), 1)))
    assert not rob.used & {x.label_id for x in world if _part(x) == "ring_tip"}
    assert set(rob.verification[inside].tolist()) == {2}
    arrays = robot_arrays(ep, rob)
    assert (arrays["finger_position_error_m"][:, ring] == -1).all()
    assert not arrays["finger_reachable"][:, ring].any()
    fingers = res.report.fingers
    assert fingers[ring].frames_with_target == 0 and fingers[ring].position_error_max_m is None
    assert fingers[pinky].frames_with_target == int((inside & early).sum())

    labels = select_labels([*scenario.labels, *world], policy, label_states(policy, True), ())
    ep = _episode(scenario, policy, ontology, labels, tmp_path)
    rob = retarget_hand_session(ep, stream, labels, hand5, rpolicy, policy.lerobot, DlsHandSolver())
    assert isinstance(rob.result, HandRetargetResult)
    assert rob.result.finger_target_present[inside, ring].all()
    assert set(rob.verification[inside].tolist()) == {0}


def test_retargeter_follows_the_robot_kind_and_rejects_mink_for_hands(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, rpolicy: RetargetPolicy
) -> None:
    """`load_retargeter`는 로봇 설정으로 경로를 고른다: 평행 그리퍼 로봇은 팔 풀이기(dls2), 다지 손
    로봇은 다지 손 풀이기(dlshand1). 다지 손 로봇에 mink 정책이면 이유와 함께 거부한다 (평행
    그리퍼 로봇은 mink를 그대로 받는다)."""
    write_robot(tmp_path, placement=PLACE)
    monkeypatch.setenv(ROBOTS_DIR_ENV, str(tmp_path))
    arm = load_retargeter(ROOT, "so101", rpolicy)
    assert isinstance(arm.robot, LoadedRobot) and isinstance(arm.solver, DlsIkSolver)
    hand = load_retargeter(ROOT, "dexhand", rpolicy)
    assert isinstance(hand.robot, LoadedDexterousRobot) and isinstance(hand.solver, DlsHandSolver)
    assert (arm.solver.version, hand.solver.version) == ("dls2", "dlshand1")
    mink = rpolicy.model_copy(update={"solver": SolverName.MINK})
    with pytest.raises(RobotError, match="mink"):
        load_retargeter(ROOT, "dexhand", mink)
    assert load_retargeter(ROOT, "so101", mink).solver.name == "mink"


def test_unknown_finger_part_fails_before_export(
    tmp_path: Path, policy: ExportPolicy, ontology: Ontology, rpolicy: RetargetPolicy
) -> None:
    """로봇 설정의 손가락 끝 부위(human_part)가 온톨로지 hand_joints에 없으면 내보내기 전에
    거부한다 (오타면 그 손가락이 모든 프레임에서 목표 없음으로 팔린다)."""
    path = write_robot(tmp_path, placement=PLACE)
    check_hand_parts(ontology, policy.lerobot, rpolicy, load_any_robot_file(path))
    data: Any = json.loads(path.read_text("utf-8"))
    data["arm"]["dexterous_hand"]["fingers"][2]["human_part"] = "middle_tips"
    path.write_text(json.dumps(data, ensure_ascii=False), "utf-8")
    with pytest.raises(ExportError, match="middle_tips"):
        check_hand_parts(ontology, policy.lerobot, rpolicy, load_any_robot_file(path))


def test_lerobot_export_writes_hand_meta_and_counts_used_labels(
    monkeypatch: pytest.MonkeyPatch,
    scenario: Scenario,
    policy: ExportPolicy,
    ontology: Ontology,
    hand5: LoadedDexterousRobot,
    tmp_path: Path,
) -> None:
    """`--robot <다지 손 로봇>` 내보내기 경로(격리 환경 쓰기만 대역): 묶음의 state가 팔 관절 +
    손가락 관절이고, meta/dlp_retarget.json과 세부 정보에 손가락 대응·손 배율·손가락별 통계가
    있으며, 쓴 라벨에 손목과 다섯 손가락 끝 월드 3D가 들어간다."""
    write_robot(tmp_path / "robots", placement=PLACE, hand_scale=HAND_SCALE)
    monkeypatch.setenv(ROBOTS_DIR_ENV, str(tmp_path / "robots"))
    points, _ = _truth(hand5, scenario.times)
    world = world_hand_labels("s1", scenario.times, points)
    labels = select_labels([*scenario.labels, *world], policy, label_states(policy, False), ())
    video = tmp_path / "blurred.mp4"
    scenario.labeling.get_file("sessions/s1/blurred/bodycam.mp4", video)
    written: dict[str, Any] = {}

    def fake_fetch(store: object, es: ExportSession, stream: Stream, work: Path) -> Path:
        return video

    def fake_run_script(root: Path, lp: object, script: str, *args: str) -> str:
        if script == "lerobot_write.py":
            written["package"] = json.loads(Path(args[0]).read_text("utf-8"))
            (Path(args[1]) / "meta").mkdir(parents=True)
            return ""
        return json.dumps({"episodes": 1, "frames": 1})

    monkeypatch.setattr(runner, "fetch_blurred", fake_fetch)
    monkeypatch.setattr(runner, "run_script", fake_run_script)
    version = DatasetVersion(
        version_id="dv", ontology_version="1.0.0", created_at=FIXED_TIME, snapshot_uri="lakefs://x",
        splits={"s1": Split.TRAIN},
    )  # fmt: skip
    src = ExportSource(
        version, [ExportSession(scenario.session, Split.TRAIN, labels)], policy.label_states
    )
    used, sessions, details = runner._lerobot(  # pyright: ignore[reportPrivateUsage]
        ROOT,
        src,
        policy,
        ontology,
        scenario.labeling,
        tmp_path / "out",
        tmp_path / "work",
        Pseudonymizer(None),
        robot="dexhand",
    )
    assert sessions == {"s1"}
    spec = written["package"]
    assert spec["robot_type"] == "dexhand"
    assert spec["features"]["observation.state"]["names"] == list(hand5.joint_names)
    assert set(HAND_ARRAYS) <= set(spec["frame_arrays"])
    meta = json.loads((tmp_path / "out/lerobot/meta/dlp_retarget.json").read_text("utf-8"))
    assert meta == details["retarget"]
    assert meta["robot_id"] == "dexhand" and meta["solver"] == "dls-dlshand1"
    assert meta["dexterous_hand"]["hand_scale"] == HAND_SCALE
    assert [f["human_part"] for f in meta["dexterous_hand"]["fingers"]] == [
        f"{f}_tip" for f in FINGERS
    ]
    episode = meta["episodes"][0]
    assert episode["decision"] == "keep" and episode["session_id"] == "s1"
    assert len(episode["fingers"]) == 5 and episode["reachable_fraction"] == 1.0
    world_ids = {x.label_id for x in world}
    assert {x.label_id for x in used if x.kind == "trajectory3d"} & world_ids == world_ids


@pytest.mark.isolated_env
def test_official_lerobot_loader_reads_hand_robot_episode(
    scenario: Scenario,
    policy: ExportPolicy,
    ontology: Ontology,
    hand5: LoadedDexterousRobot,
    rpolicy: RetargetPolicy,
    tmp_path: Path,
) -> None:
    """격리 환경(LeRobot 공식 API)으로 다지 손 로봇 에피소드를 쓰고 공식 로더로 다시 읽는다: state는
    팔 관절 + 손가락 관절 21개, 손가락별 retarget.finger_* 특징이 손가락 수만큼 읽히고, robot_type이
    로봇 ID다 (`make test-isolated`)."""
    from dlp_export.lerobot import run_script

    points, _ = _truth(hand5, scenario.times)
    world = world_hand_labels("s1", scenario.times, points)
    labels = select_labels([*scenario.labels, *world], policy, label_states(policy, False), ())
    ep = _episode(scenario, policy, ontology, labels, tmp_path)
    lp = policy.lerobot
    rob = retarget_hand_session(
        ep, scenario.session.streams[0], labels, hand5, rpolicy, lp, DlsHandSolver()
    )
    res = rob.result
    assert isinstance(res, HandRetargetResult)
    pkg = write_package(
        [(scenario.session, tmp_path / "v.mp4", ep)], policy, (64, 48), tmp_path / "pkg",
        Pseudonymizer(None), robot_override([(ep, rob)], lp, hand5),
    )  # fmt: skip
    dest = tmp_path / "lerobot"
    run_script(ROOT, lp, "lerobot_write.py", str(pkg), str(dest))
    info = json.loads((dest / "meta" / "info.json").read_text())
    assert info["robot_type"] == "dexhand"
    assert info["features"]["observation.state"]["names"] == list(hand5.joint_names)
    assert info["features"]["retarget.finger_reachable"]["names"] == [f"{f}_tip" for f in FINGERS]
    check = json.loads(run_script(ROOT, lp, "lerobot_check.py", str(dest), lp.repo_id))
    assert check["episodes"] == 1 and check["frames"] == ep.times_ms.size
    assert set(HAND_ARRAYS) | {"retarget.reachable", "observation.human_state"} <= set(
        check["features"]
    )
    for s in check["samples"]:
        k = s["index"]
        assert np.allclose(s["state"], res.q[k], atol=1e-5)
        extra = s["extra"]
        assert extra["retarget.finger_target_present"] == [
            float(v) for v in res.finger_target_present[k]
        ]
        assert extra["retarget.finger_reachable"] == [float(v) for v in res.finger_reachable[k]]
        assert len(extra["retarget.finger_position_error_m"]) == len(FINGERS)


def test_parallel_gripper_robot_gets_no_finger_columns(
    monkeypatch: pytest.MonkeyPatch, policy: ExportPolicy, rpolicy: RetargetPolicy
) -> None:
    """평행 그리퍼 로봇의 특징·배열 대응·meta에는 다지 손 항목이 생기지 않는다 (기존 납품과 같은
    열과 키)."""
    monkeypatch.delenv(ROBOTS_DIR_ENV, raising=False)
    arm = load_retargeter(ROOT, "so101", rpolicy)
    assert isinstance(arm.robot, LoadedRobot)
    feats = robot_features(policy.lerobot, arm.robot, 48, 64)
    assert not [name for name in feats if "finger" in name]
    assert feats["observation.state"]["names"] == list(arm.robot.joint_names)
    override = robot_override([], policy.lerobot, arm.robot)
    assert override.arrays == {**FRAME_ARRAYS, **ROBOT_ARRAYS}
    meta = retarget_meta(arm.robot, rpolicy, arm.solver, [])
    assert "dexterous_hand" not in meta and "finger_reachable_definition" not in meta
    assert meta["hand_parts"] == {"wrist": "wrist", "thumb": "thumb_tip", "index": "index_tip"}
