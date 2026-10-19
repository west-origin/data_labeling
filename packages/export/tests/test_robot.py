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
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from dlp_retarget.ik import DlsIkSolver
from dlp_retarget.policy import RetargetPolicy, load_policy
from dlp_retarget.robot import LoadedRobot, Placement, PlacementMode, load_robot
from numpy.typing import NDArray

from dlp_export import runner
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
from dlp_export.runner import trajectory_manifest
from dlp_export.source import (
    ExportError,
    ExportSession,
    ExportSource,
    label_states,
    select_labels,
)
from dlp_fixtures.retarget import hand_points_for_grasp, world_hand_labels, world_hand_sources
from dlp_media.pts import build_pts_index
from dlp_schema.dataset import DatasetVersion, Split
from dlp_schema.labels import LabelRecord, Source, Verification, VerificationState
from dlp_schema.ontology import Ontology
from dlp_schema.review import TrajectoryInheritance, TrajectoryInheritanceMechanism
from dlp_schema.testing import FIXED_TIME, make_session
from dlp_schema.trajectory_follow import plan_trajectory_follow

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
    labels = select_labels(scenario.labels, policy, label_states(policy, False), ())
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
    labels = select_labels([*scenario.labels, *world], policy, label_states(policy, False), ())
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
    assert meta["camera_pose"] == "automatic_unreviewed"  # 월드 좌표 목표는 자동 자세에 기댄다


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
        labels = select_labels(
            [*scenario.labels, *world], policy, label_states(policy, include), ()
        )
        ep = _episode(scenario, policy, ontology, labels, tmp_path)
        rob: RobotEpisode = retarget_session(
            ep, stream, labels, robot, rpolicy, policy.lerobot, DlsIkSolver()
        )
        assert (rob.result.report.frames_with_target > 0) == bool(expect)


def _inherit(history: list[LabelRecord]) -> tuple[list[LabelRecord], list[TrajectoryInheritance]]:
    """궤적 따르기를 이력에 적용한 결과 (`apply_trajectory_follow` 흉내, 기록 event_id는 순번)."""
    plan = plan_trajectory_follow(history, [])
    got = {r.label_id: r for r in plan.approve}
    out = [
        x.model_copy(
            update={
                "verification": Verification(
                    state=r.state, reviewer_id=r.reviewer_id, reviewed_at=r.reviewed_at
                )
            }
        )
        if (r := got.get(x.label_id))
        else x
        for x in history
    ]
    records = [
        r.record("s1", TrajectoryInheritanceMechanism.COLLECT, FIXED_TIME).model_copy(
            update={"event_id": i + 1}
        )
        for i, r in enumerate(plan.approve)
    ]
    return out, records


def test_world_hand_inherited_from_approved_2d_is_used_by_default(
    scenario: Scenario,
    policy: ExportPolicy,
    ontology: Ontology,
    robot: LoadedRobot,
    rpolicy: RetargetPolicy,
    tmp_path: Path,
) -> None:
    """F-COE-19 (ADR 0048): 2D 손 트랙을 승인하면 그 트랙에서 계산한 카메라·월드 좌표 3D가 검수를
    물려받아 기본 내보내기(검수 라벨만)에 손 목표·손 3D 점이 생긴다. 그 뒤 2D를 고쳐 물려받음이
    끊기면 낡은 3D는 옵션과 관계없이 빠진다 (궤적 따르기가 지우기 전의 스냅샷이어도, ADR 0050).
    manifest 절(`trajectory_manifest`)은 결과에 든 월드 좌표 궤적이 자동 카메라 자세에 기댄다는 것과
    근거별 수를 밝힌다.

    정답: 월드 손 6점(검수 전) + 그 입력(2D 트랙 승인, 카메라 좌표 6점·자세 검수 전)에 궤적 따르기를
    적용하면 카메라·월드 12개가 human_approved가 되고, 기본 정책으로 목표가 있는 프레임이 생긴다.
    """
    world, _ = _truth_labels(robot, scenario.times, VerificationState.UNREVIEWED)
    sources = world_hand_sources(
        "s1", scenario.times, _world_points(world), track_state=VerificationState.HUMAN_APPROVED
    )
    history, records = _inherit([*scenario.labels, *sources, *world])
    assert len(records) == 12
    stream = scenario.session.streams[0]
    default = label_states(policy, False)
    labels = select_labels(history, policy, default, records)
    ep = _episode(scenario, policy, ontology, labels, tmp_path)
    rob = retarget_session(ep, stream, labels, robot, rpolicy, policy.lerobot, DlsIkSolver())
    assert rob.result.report.frames_with_target > 0
    assert (ep.hand_points_3d_confidence >= 0).any()  # 사람 모드 손 3D 점 (카메라 좌표)
    manifest = trajectory_manifest(labels, {"s1": records})
    assert manifest["camera_pose"] == "automatic_unreviewed"
    assert manifest["pose_derived"] == {
        "world_frame_trajectories": 6,
        "camera_pose_trajectories": 0,
    }
    assert manifest["review_basis"]["inherited_from_2d"] == 12
    with_pose = select_labels(history, policy, label_states(policy, True), records)
    basis = trajectory_manifest(with_pose, {"s1": records})
    assert basis["pose_derived"]["camera_pose_trajectories"] == 1
    assert basis["review_basis"]["automatic"] == 1

    track = next(x for x in sources if x.kind == "keypoint_track")
    fixed = track.model_copy(
        update={
            "label_id": f"{track.label_id}-fix",
            "parent_label_id": track.label_id,
            "provenance": track.provenance.model_copy(
                update={"source": Source.HUMAN, "model_version": None}
            ),
            "confidence": None,
            "verification": Verification(
                state=VerificationState.HUMAN_CORRECTED, reviewer_id="rev1", reviewed_at=FIXED_TIME
            ),
        }
    )
    for include in (False, True):
        chosen = select_labels([*history, fixed], policy, label_states(policy, include), records)
        inherited = {r.trajectory_label_id for r in records}
        assert not inherited & {x.label_id for x in chosen}


def _world_points(world: list[LabelRecord]) -> dict[str, NDArray[np.float64]]:
    """월드 좌표 라벨 → 부위 → (n, 3) 표본 (카메라 좌표 입력을 같은 값으로 만든다)."""
    out: dict[str, NDArray[np.float64]] = {}
    for x in world:
        p = x.payload
        assert p.kind == "trajectory3d" and p.part is not None
        out[p.part] = np.array([[s.x, s.y, s.z] for s in p.samples])
    return out


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
    labels = select_labels([*scenario.labels, *world], policy, label_states(policy, False), ())
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


# ---------------------------------------------------------------- 묶음 C (ADR 0053)


def test_retarget_meta_states_the_reachability_criteria(
    robot: LoadedRobot, rpolicy: RetargetPolicy
) -> None:
    """meta/dlp_retarget.json에 도달 판정 기준(IK 허용 오차·orientation_mode·velocity_scale), 손 점
    역할, reachable·회전 오차의 뜻을 싣는다 (판 문자열만으로는 알 수 없었다, F-COE-46). SO-101은
    full 모드다 (F-COE-40). 모델 경로는 싣지 않는다."""
    meta = retarget_meta(robot, rpolicy, make_solver(ROOT, rpolicy), [])
    assert meta["ik"] == robot.config.ik.model_dump(mode="json")
    assert meta["ik"]["orientation_mode"] == "full"
    assert meta["hand_parts"] == {"wrist": "wrist", "thumb": "thumb_tip", "index": "index_tip"}
    assert "roll is evaluated" in meta["orientation_error_definition"]
    assert "position_tolerance_m" in meta["reachable_definition"]
    assert "so101_new_calib" not in json.dumps(meta)


def test_robot_targets_survive_film_rate_world_samples(
    scenario: Scenario,
    policy: ExportPolicy,
    ontology: Ontology,
    robot: LoadedRobot,
    rpolicy: RetargetPolicy,
    tmp_path: Path,
) -> None:
    """월드 손 3D 표본이 24 fps 깊이 간격(208/209 ms)이어도 그 구간의 모든 에피소드 프레임에 로봇
    목표가 있다 (2D 보간 한도 200 ms를 쓰던 때는 거의 모든 프레임이 target_present 0, F-COE-50)."""
    times = [0, 208, 417, 625, 834]
    world, _ = _truth_labels(robot, times, VerificationState.HUMAN_APPROVED)
    labels = select_labels([*scenario.labels, *world], policy, label_states(policy, False), ())
    ep = _episode(scenario, policy, ontology, labels, tmp_path)
    rob = retarget_session(
        ep, scenario.session.streams[0], labels, robot, rpolicy, policy.lerobot, DlsIkSolver()
    )
    inside = (ep.times_ms >= times[0]) & (ep.times_ms <= times[-1])
    assert rob.result.target_present[inside].all()


class _FakeRun:
    """`_RobotRun` 대역: 세션 ID가 drop에 있으면 버린다."""

    def __init__(self, drop: set[str]) -> None:
        self.drop = drop
        self.reports: list[dict[str, object]] = []
        self.results: list[object] = []
        self.policy = load_policy(ROOT)
        self.robot = None

    def add(self, es: object, stream: object, ep: object, lp: object, index: int, sid: str) -> bool:
        keep = sid not in self.drop
        self.reports.append({"session_id": sid, "decision": "keep" if keep else "drop"})
        if keep:
            self.results.append(object())
        return keep

    def used(self) -> set[str]:
        return set()

    def meta(self) -> dict[str, object]:
        return {"episodes": self.reports}


@pytest.mark.parametrize(
    ("sizes", "drop", "expect"),
    [
        # 다른 화면비 첫 에피소드를 버림 / 같은 화면비의 작은 첫 에피소드를 버림 / 둘째를 버림
        ({"a": (640, 480), "b": (1280, 720)}, {"a"}, (1280, 720)),
        ({"a": (480, 360), "b": (640, 480)}, {"a"}, (640, 480)),
        ({"a": (640, 480), "b": (1280, 720), "c": (640, 480)}, {"b"}, (640, 480)),
    ],
)
def test_dropped_episodes_do_not_set_dataset_size(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    policy: ExportPolicy,
    ontology: Ontology,
    sizes: dict[str, tuple[int, int]],
    drop: set[str],
    expect: tuple[int, int],
) -> None:
    """리타기팅 정책이 버린 에피소드의 크기·화면비는 데이터셋 크기 기준이 되지 않는다 (버린 첫
    세션의 해상도로 줄이거나 화면비가 달라 실패하던 문제, F-COE-49). 내보내는 에피소드끼리 화면비가
    다르면 여전히 거부한다."""

    got: dict[str, tuple[int, int]] = {}

    def fake_run_script(root: Path, lp: object, script: str, *args: str) -> str:
        if script == "lerobot_write.py":
            (Path(args[1]) / "meta").mkdir(parents=True)
            return ""
        return json.dumps({"episodes": len(sizes) - len(drop), "frames": 1})

    def fake_write_package(
        eps: list[object], pol: object, size: tuple[int, int], *a: object
    ) -> Path:
        got["size"] = size
        return tmp_path / "pkg.json"

    def fake_fetch(store: object, es: ExportSession, stream: object, work: Path) -> Path:
        return Path(es.session.session_id)

    def fake_probe(video: Path) -> SimpleNamespace:
        w, h = sizes[str(video)]
        return SimpleNamespace(video=SimpleNamespace(width=w, height=h))

    def fake_episode(*args: object) -> SimpleNamespace:
        return SimpleNamespace(
            times_ms=np.array([0.0]), used=frozenset[str](), hand_points_3d_coverage=None
        )

    def fake_load(root: Path, robot_id: str) -> _FakeRun:
        # 클래스에서 부르므로(_RobotRun.load(...)) 보통 함수로 바꿔 끼워도 된다
        return _FakeRun(drop)

    def nothing(*args: object) -> None:
        return None

    monkeypatch.setattr(runner, "fetch_blurred", fake_fetch)
    monkeypatch.setattr(runner, "probe", fake_probe)
    monkeypatch.setattr(runner, "build_pts_index", nothing)
    monkeypatch.setattr(runner, "build_episode", fake_episode)
    monkeypatch.setattr(runner._RobotRun, "load", fake_load)  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(runner, "robot_override", nothing)
    monkeypatch.setattr(runner, "write_package", fake_write_package)
    monkeypatch.setattr(runner, "run_script", fake_run_script)
    version = DatasetVersion(
        version_id="dv", ontology_version="1.0.0", created_at=FIXED_TIME, snapshot_uri="lakefs://x",
        splits={s: Split.TRAIN for s in sizes},
    )  # fmt: skip
    src = ExportSource(
        version,
        [ExportSession(make_session(s), Split.TRAIN, []) for s in sizes],
        policy.label_states,
    )
    store: Any = None  # 블러본을 받지 않는다 (fetch_blurred 대역)
    runner._lerobot(  # pyright: ignore[reportPrivateUsage]
        ROOT,
        src,
        policy,
        ontology,
        store,
        tmp_path / "out",
        tmp_path,
        Pseudonymizer(None),
        robot="so101",
    )
    assert got["size"] == expect
    mixed = {"a": (640, 480), "b": (1280, 720)}
    with pytest.raises(ExportError, match="화면비"):
        sizes.clear()
        sizes.update(mixed)
        drop.clear()
        runner._lerobot(  # pyright: ignore[reportPrivateUsage]
            ROOT,
            src,
            policy,
            ontology,
            store,
            tmp_path / "out2",
            tmp_path,
            Pseudonymizer(None),
            robot="so101",
        )
