"""프리라벨 러너의 다시 돌리기 규칙 (SQLite + 로컬 저장소, 서비스 없이 돈다).

검수 2026-10-17 묶음 A의 회귀 시험이다. `run_prelabel`을 시간대 보존 SQLite(`sqlite_db`)와
`LocalStore`(받은 횟수를 센다)로 끝까지 돈다. PostgreSQL 트리거·행 잠금은 서비스 시험
(`test_runner.py`)이 본다.

정답 근거
- 손 3D: `dlp_fixtures.hand3d`(정답 손이 20% 작고 초점 거리 200 px). 캘리브레이션 없이 화각 근사
  (fx 160)와 정책 손 크기(0.09 m)로 올린 3D는 틀리고, 기기 캘리브레이션(fx 200)과 작업자 손 크기
  (0.072 m)를 등록하면 정답과 1 mm 안이다 (가짜 깊이 1 m는 배율 비가 커서 버려진다).
- 자세: `dlp_fixtures.camera_motion` 정답 자세를 내는 `OraclePoseEstimator`. 버전·이력 규칙만 본다.
- 실행마다 created_at을 다르게 둔다 (단계의 "마지막 실행"을 시각으로 정한다, `stage_version`).
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import sqlalchemy as sa
import yaml
from numpy.typing import NDArray

from dlp_fixtures.camera_motion import CameraMotionScenario, generate_camera_motion
from dlp_fixtures.hand3d import Hand3DScenario, generate_hand3d_scenario
from dlp_fixtures.io import write_parquet
from dlp_fixtures.video import write_video
from dlp_media.ingest import ingest_session, load_manifest
from dlp_media.storage import LocalStore
from dlp_prelabel.adapters.stubs import OraclePredictor
from dlp_prelabel.campose import (
    ImuSeries,
    ModelDepth,
    OraclePoseEstimator,
    PoseTrack,
    is_pose_label,
)
from dlp_prelabel.common import Image
from dlp_prelabel.lift3d import DepthLifter
from dlp_prelabel.policy import HAND21_JOINTS, PrelabelPolicy, load_policy
from dlp_prelabel.runner import PrelabelSummary, imu_to_camera, run_prelabel
from dlp_schema.calibration import CalibrationMethod, HandSizeSource
from dlp_schema.db.calibration_repository import register_camera_calibration, register_hand_size
from dlp_schema.db.repository import (
    get_labels,
    get_session,
    insert_labels,
    record_review,
    register_ontology,
    update_stream_sync,
)
from dlp_schema.episode import current_labels
from dlp_schema.labels import (
    CoordinateFrame,
    IntrinsicsSource,
    LabelRecord,
    Trajectory3DPayload,
    VerificationState,
)
from dlp_schema.ontology import Ontology, load_ontology
from dlp_schema.predictor import Predictor
from dlp_schema.session import SyncMethod
from dlp_schema.testing import FIXED_TIME, make_session, sqlite_db
from dlp_schema.trajectory_follow import TRAJECTORY_FOLLOW_VERSION

ROOT = Path(__file__).resolve().parents[3]
JOINT_OF = {name: i for i, name in HAND21_JOINTS.items()}


def _at(hours: int) -> Any:
    """실행 시각 (FIXED_TIME + 시간)."""
    return FIXED_TIME + timedelta(hours=hours)


class CountingStore(LocalStore):
    """원본을 받은 횟수를 세는 로컬 저장소 (감사 저장소 대역)."""

    gets = 0

    def get_file(self, key: str, dest: Path) -> None:
        """받은 횟수를 세고 원래대로 받는다."""
        self.gets += 1
        super().get_file(key, dest)


class _UnitDepth:
    """깊이를 늘 1 m로 내고 호출 수를 세는 가짜 깊이 모델."""

    def __init__(self) -> None:
        """호출 수 0."""
        self.calls = 0

    def predict(self, image: Image) -> NDArray[np.float32]:
        """(H, W) 1 m 깊이 맵."""
        self.calls += 1
        return np.ones(image.shape[:2], dtype=np.float32)


class _UnitLifter(DepthLifter):
    """가중치 파일 없이 `_UnitDepth`로 도는 깊이 3D 단계."""

    def __init__(self, policy: PrelabelPolicy) -> None:  # pyright: ignore[reportMissingSuperCall]
        """가중치 확인 대신 가짜 모델·버전을 둔다."""
        self.path = Path("unused.onnx")
        self.weights_version = "unit"
        self.version = f"depth3d-unit+p{policy.digest('depth')}"
        self.policy, self.now = policy, FIXED_TIME
        self.depth = _UnitDepth()
        self.model = self.depth


@pytest.fixture
def ontology() -> Ontology:
    """저장소 온톨로지 v1."""
    return load_ontology(ROOT / "config/ontology/v1")


@pytest.fixture
def policy() -> PrelabelPolicy:
    """저장소 prelabel.yaml."""
    return load_policy(ROOT)


def _db(tmp_path: Path, ontology: Ontology) -> sa.Engine:
    """온톨로지를 등록한 SQLite 파일 DB."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    engine = sqlite_db(f"sqlite:///{tmp_path / 'db.sqlite'}")
    with engine.begin() as conn:
        register_ontology(conn, ontology)
    return engine


def _ingest(engine: sa.Engine, tmp_path: Path, manifest: dict[str, Any]) -> CountingStore:
    """매니페스트로 세션을 수집하고 원본 저장소를 돌려준다."""
    base: dict[str, Any] = {
        "domain": "cleaning", "worker_id": "w01", "site_id": "site01", "consent_version": "c1",
        "recorded_at": FIXED_TIME.isoformat(), "ontology_version": "1.0.0",
    }  # fmt: skip
    (tmp_path / "m.yaml").write_text(yaml.safe_dump({**base, **manifest}), encoding="utf-8")
    raw = CountingStore(tmp_path / "store", "dlp-raw")
    with engine.begin() as conn:
        ingest_session(*load_manifest(tmp_path / "m.yaml"), raw, conn)
    return raw


def _hand_session(engine: sa.Engine, tmp_path: Path, s: Hand3DScenario) -> CountingStore:
    """손 키프레임 시각 그대로의 회색 바디캠 하나와 기기 ID(cam-a)만 있는 세션 (세션 ID hands)."""
    write_video(
        tmp_path / "bodycam.mp4",
        ((t, np.full((s.height, s.width, 3), 90, np.uint8)) for t in s.frame_times),
        width=s.width, height=s.height,
    )  # fmt: skip
    return _ingest(
        engine,
        tmp_path,
        {
            "session_id": "hands",
            "streams": [{"stream_id": "bodycam", "kind": "bodycam", "path": "bodycam.mp4"}],
            "calibration": {"camera_id": "cam-a"},
        },
    )


def _hands(s: Hand3DScenario) -> list[Predictor]:
    """정답 손 트랙(world 포함)을 모델 라벨로 내는 손 어댑터."""
    return [OraclePredictor("hands", s.labels, ("keypoint_track",), now=FIXED_TIME)]


def _live_3d(conn: sa.Connection, sid: str) -> list[LabelRecord]:
    """현재 3D 궤적 라벨."""
    return current_labels(get_labels(conn, sid, kinds=["trajectory3d"]))


def _max_error_mm(s: Hand3DScenario, labels: list[LabelRecord]) -> float:
    """손 3D 표본과 정답 관절의 최대 거리 (mm)."""
    errs: list[float] = []
    for x in labels:
        p = x.payload
        assert isinstance(p, Trajectory3DPayload)
        for q in p.samples:
            truth: NDArray[np.float64] = s.joints_cam[q.t_ms][JOINT_OF[p.part or ""]]
            errs.append(1000 * float(np.linalg.norm(np.array([q.x, q.y, q.z]) - truth)))
    return max(errs)


def _approved_hand_3d(
    tmp_path: Path, ontology: Ontology, policy: PrelabelPolicy
) -> tuple[sa.Engine, CountingStore, Hand3DScenario, list[LabelRecord]]:
    """손 2D를 승인한 뒤 3D 단계를 돌려 3D가 그 검수를 물려받은 세션 (등록 전, 화각 근사)."""
    s = generate_hand3d_scenario(7, n_frames=8, hand_scale=0.8, focal_px=200.0)
    engine = _db(tmp_path, ontology)
    raw = _hand_session(engine, tmp_path, s)
    with engine.begin() as conn:
        run_prelabel(conn, "hands", raw, _hands(s), policy, ontology, _at(0))
        [track] = get_labels(conn, "hands", kinds=["keypoint_track"])
        record_review(conn, track.label_id, VerificationState.HUMAN_APPROVED, "rev01", _at(0))
        first = run_prelabel(
            conn, "hands", raw, _hands(s), policy, ontology, _at(1), lifter=_UnitLifter(policy)
        )
        before = _live_3d(conn, "hands")
    n = len(policy.depth.hand_points)
    assert first.lifted == len(before) == first.trajectories_inherited == n
    return engine, raw, s, before


def _register(engine: sa.Engine, s: Hand3DScenario) -> None:
    """기기 캘리브레이션(정답 내부 파라미터)과 작업자 손 크기(정답 0.072 m)를 등록한다."""
    with engine.begin() as conn:
        register_camera_calibration(
            conn, "cam-a", s.intrinsics, method=CalibrationMethod.IMPORTED, now=_at(2)
        )
        register_hand_size(conn, "w01", 0.072, source=HandSizeSource.MEASURED, now=_at(2))


def test_inherited_3d_is_recomputed_when_calibration_and_palm_are_registered(
    tmp_path: Path, ontology: Ontology, policy: PrelabelPolicy
) -> None:
    """ADR 0051 (F-PAR-41 재현): 2D 승인으로 물려받기만 한 3D는 캘리브레이션·작업자 손 크기를
    등록하면 새 값으로 다시 계산된다. 이전 3D는 궤적 따르기(trajectory-follow-1)가 지우고, 새 3D가
    같은 2D 검수(human_approved, rev01)를 물려받으며 정답과 1 mm 안이다. 다시 돌리면 원본을 받거나
    깊이를 돌리지 않는다. 예전에는 물려받은 3D가 새 3D를 막아 등록이 영영 반영되지 않았고, 매 실행
    원본을 받고 깊이를 다시 돌렸다.
    """
    engine, raw, s, before = _approved_hand_3d(tmp_path, ontology, policy)
    assert _max_error_mm(s, before) > 10.0
    _register(engine, s)
    lifter = _UnitLifter(policy)
    with engine.begin() as conn:
        second = run_prelabel(
            conn, "hands", raw, _hands(s), policy, ontology, _at(3), lifter=lifter
        )
        live = _live_3d(conn, "hands")
        history = get_labels(conn, "hands", kinds=["trajectory3d"])
    n = len(policy.depth.hand_points)
    assert second.lifted == second.trajectories_inherited == second.trajectories_retracted == n
    assert not {x.label_id for x in before} & {x.label_id for x in live} and len(live) == n
    gone = [x for x in history if x.retracted]
    assert {x.parent_label_id for x in gone} == {x.label_id for x in before}
    assert {x.provenance.model_version for x in gone} == {TRAJECTORY_FOLLOW_VERSION}
    for x in live:
        assert isinstance(x.payload, Trajectory3DPayload)
        assert x.payload.intrinsics is IntrinsicsSource.CALIBRATED
        assert (x.provenance.model_version or "").endswith("+hpw")
        assert (x.verification.state, x.verification.reviewer_id) == (
            VerificationState.HUMAN_APPROVED, "rev01",
        )  # fmt: skip
    assert _max_error_mm(s, live) < 1.0
    raw.gets, lifter.depth.calls = 0, 0
    with engine.begin() as conn:
        again = run_prelabel(conn, "hands", raw, _hands(s), policy, ontology, _at(4), lifter=lifter)
    assert again.lifted == again.trajectories_retracted == 0
    assert (raw.gets, lifter.depth.calls) == (0, 0)


def test_directly_reviewed_3d_still_blocks_and_blocked_runs_skip_the_raw_video(
    tmp_path: Path, ontology: Ontology, policy: PrelabelPolicy
) -> None:
    """ADR 0051 경계: 물려받은 뒤 검수자가 직접 다시 검수한 3D(기록 뒤 상태 변경)는 보호된다: 새
    판이 그 대상을 대신하지 않고(lift_blocked), 나머지는 다시 계산돼 물려받는다. 모든 대상이 그렇게
    막히면 원본을 받거나 깊이를 돌리지 않는다 (F-PAR-41 헛계산).
    """
    engine, raw, s, before = _approved_hand_3d(tmp_path, ontology, policy)
    kept = before[0]
    with engine.begin() as conn:
        record_review(conn, kept.label_id, VerificationState.HUMAN_CORRECTED, "lead", _at(2))
    _register(engine, s)
    with engine.begin() as conn:
        second = run_prelabel(
            conn, "hands", raw, _hands(s), policy, ontology, _at(3), lifter=_UnitLifter(policy)
        )
        live = {x.label_id for x in _live_3d(conn, "hands")}
    n = len(policy.depth.hand_points)
    assert (second.lifted, second.lift_blocked) == (n - 1, 1)
    assert kept.label_id in live and len(live) == n

    # 모든 3D를 직접 다시 검수하면 낼 것이 없다: 원본을 받지 않는다
    engine2, raw2, s2, before2 = _approved_hand_3d(tmp_path / "all", ontology, policy)
    with engine2.begin() as conn:
        for x in before2:
            record_review(conn, x.label_id, VerificationState.HUMAN_CORRECTED, "lead", _at(2))
    _register(engine2, s2)
    lifter = _UnitLifter(policy)
    raw2.gets = 0
    for hour in (3, 4):
        with engine2.begin() as conn:
            got = run_prelabel(
                conn, "hands", raw2, _hands(s2), policy, ontology, _at(hour), lifter=lifter
            )
        assert (got.lifted, got.lift_blocked) == (0, n)
    assert (raw2.gets, lifter.depth.calls) == (0, 0)


def test_reverting_an_input_regenerates_the_earlier_result(
    tmp_path: Path, ontology: Ontology, policy: PrelabelPolicy
) -> None:
    """F-PAR-54 재현: depth 정책을 A → B → A로 되돌리면 3D를 A로 다시 만든다 (라벨 ID가 이전 A와
    겹치지 않게 버전 끝 +r1). 예전에는 'A가 이력에 있다'로 건너뛰어 B로 만든 3D가 남았다. 다시
    A로 돌리면 건너뛴다.
    """
    s = generate_hand3d_scenario(7, n_frames=8, hand_scale=0.8, focal_px=200.0)
    engine = _db(tmp_path, ontology)
    raw = _hand_session(engine, tmp_path, s)
    fit = policy.depth.hand_fit.model_copy(update={"reference_palm_m": 0.08})
    other = policy.model_copy(update={"depth": policy.depth.model_copy(update={"hand_fit": fit})})
    runs: list[PrelabelSummary] = []
    with engine.begin() as conn:
        run_prelabel(conn, "hands", raw, _hands(s), policy, ontology, _at(0))
        for hour, p in enumerate((policy, other, policy, policy), start=1):
            runs.append(
                run_prelabel(
                    conn, "hands", raw, _hands(s), p, ontology, _at(hour), lifter=_UnitLifter(p)
                )
            )
        live = _live_3d(conn, "hands")
    n = len(policy.depth.hand_points)
    assert [r.lifted for r in runs] == [n, n, n, 0]
    versions = {x.provenance.model_version or "" for x in live}
    assert len(versions) == 1 and versions.pop().endswith("+r1")


# ---------------------------------------------------------------- 자세 단계


def _pose_session(
    engine: sa.Engine,
    tmp_path: Path,
    sc: CameraMotionScenario,
    *,
    extrinsics: Any = "fixture",
) -> CountingStore:
    """렌더 바디캠 + IMU 사이드카(사람 조정 동기화) + 카메라 좌표 손 궤적 세션 (세션 ID pose)."""
    write_video(tmp_path / "bodycam.mp4", sc.frames(), width=sc.width, height=sc.height)
    write_parquet(tmp_path / "imu.parquet", sc.imu_columns())
    calibration: dict[str, Any] = {"intrinsics": sc.intrinsics.model_dump(mode="json")}
    if extrinsics is not None:
        calibration["camera_imu_extrinsics"] = (
            [list(r) for r in sc.imu_to_camera] if extrinsics == "fixture" else extrinsics
        )
    raw = _ingest(
        engine,
        tmp_path,
        {
            "session_id": "pose",
            "streams": [
                {"stream_id": "bodycam", "kind": "bodycam", "path": "bodycam.mp4"},
                {"stream_id": "imu", "kind": "imu", "path": "imu.parquet"},
            ],
            "calibration": calibration,
        },
    )
    with engine.begin() as conn:
        _set_imu_sync(conn, offset_ms=sc.imu_offset_ms)
        cams = [
            x.model_copy(
                update={"session_id": "pose", "label_id": x.label_id.replace(sc.session_id, "pose")}
            )
            for x in sc.camera_labels(stride_ms=100)
        ]
        insert_labels(conn, cams)
    return raw


def _set_imu_sync(conn: sa.Connection, *, offset_ms: float, manual_ms: float = 0.0) -> None:
    """IMU 스트림을 사람 조정 동기화로 둔다 (`dlp sync adjust`와 같은 효과)."""
    imu = get_session(conn, "pose").stream("imu").model_copy(
        update={
            "sync_method": SyncMethod.MANUAL, "offset_ms": offset_ms,
            "manual_adjustment_ms": manual_ms,
        }
    )  # fmt: skip
    update_stream_sync(conn, "pose", imu)


def _oracle(sc: CameraMotionScenario, version: str = "o1") -> OraclePoseEstimator:
    """정답 자세 stub (100 ms 간격)."""
    truth = {t: sc.pose(t) for t in sc.frame_times}
    return OraclePoseEstimator(truth, frame_stride_ms=100, version=version)


class _Failing:
    """늘 실패하는 자세 추정기 (호출 수를 센다)."""

    name = "campose"

    def __init__(self, version: str) -> None:
        """version: 추정기 판."""
        self.version = version
        self.calls = 0

    def estimate(self, frames: Any, **_: Any) -> PoseTrack:
        """ValueError를 낸다 (IMU 초기화 실패 흉내)."""
        self.calls += 1
        raise ValueError("자세 초기화 구간에 IMU 표본이 부족합니다")


class _Spy:
    """받은 IMU·깊이를 기록하고 정답 자세 stub에 넘기는 추정기 (러너 배선 확인)."""

    name = "campose"
    version = "spy"

    def __init__(self, inner: OraclePoseEstimator) -> None:
        """inner: 결과를 낼 stub."""
        self.inner = inner
        self.imu: ImuSeries | None = None
        self.depth: object = None

    def estimate(self, frames: Any, *, imu: ImuSeries, depth: Any, **kw: Any) -> PoseTrack:
        """IMU(마스터 시각)·깊이를 기록하고 stub 결과를 돌려준다."""
        self.imu, self.depth = imu, depth
        return self.inner.estimate(frames, imu=imu, depth=depth, **kw)


def _poses(conn: sa.Connection) -> list[LabelRecord]:
    """현재 카메라 자세 라벨."""
    return [x for x in _live_3d(conn, "pose") if is_pose_label(x)]


def _world(conn: sa.Connection) -> list[LabelRecord]:
    """현재 월드 좌표 손 궤적 (자세 제외)."""
    return [
        x
        for x in _live_3d(conn, "pose")
        if isinstance(x.payload, Trajectory3DPayload)
        and x.payload.frame is CoordinateFrame.WORLD
        and not is_pose_label(x)
    ]


def test_failed_estimate_with_a_new_version_retracts_the_previous_pose(
    tmp_path: Path, ontology: Ontology, policy: PrelabelPolicy
) -> None:
    """F-PAR-50 재현: 입력이 바뀐 새 버전으로 추정이 실패하면 검수 전 이전 자세를 지우고(이유를
    남긴다) 월드 좌표 단계가 그 자세로 만든 검수 전 월드 3D를 지운다. 같은 입력으로 다시 돌리면
    추정하지 않는다. 예전에는 실패하면 낡은 자세와 월드 3D가 운영 라벨로 남았다.
    """
    sc = generate_camera_motion(11, duration_s=2.0)
    engine = _db(tmp_path, ontology)
    raw = _pose_session(engine, tmp_path, sc)
    failing = _Failing("broken")
    with engine.begin() as conn:
        first = run_prelabel(
            conn, "pose", raw, [], policy, ontology, _at(1), pose_estimator=_oracle(sc)
        )
        assert first.poses > 0 and len(_world(conn)) == first.world > 0
        failed = run_prelabel(
            conn, "pose", raw, [], policy, ontology, _at(2), pose_estimator=failing
        )
        assert _poses(conn) == [] and _world(conn) == []
        again = run_prelabel(
            conn, "pose", raw, [], policy, ontology, _at(3), pose_estimator=failing
        )
    assert failed.pose_note is not None and "자세 추정 실패" in failed.pose_note
    assert "이전 자세 1개를 지움" in failed.pose_note
    assert failing.calls == 1 and again.poses == 0


def test_resync_without_a_depth_model_retracts_the_stale_pose(
    tmp_path: Path, ontology: Ontology, policy: PrelabelPolicy
) -> None:
    """F-PAR-56 재현: IMU 동기화를 바꾼 뒤 깊이 모델 없이 프리라벨을 돌리면(다시 추정할 수 없음)
    예전 동기화로 만든 자세를 지우고 '[자세 낡음]'으로 알린다. 그 자세의 월드 3D도 지운다. 동기화가
    그대로면 자세를 남긴다. 다시 돌려도 아무것도 바뀌지 않는다 (멱등).
    """
    sc = generate_camera_motion(11, duration_s=2.0)
    engine = _db(tmp_path, ontology)
    raw = _pose_session(engine, tmp_path, sc)
    with engine.begin() as conn:
        run_prelabel(conn, "pose", raw, [], policy, ontology, _at(1), pose_estimator=_oracle(sc))
        same = run_prelabel(conn, "pose", raw, [], policy, ontology, _at(2))
        assert len(_poses(conn)) == 1
        _set_imu_sync(conn, offset_ms=sc.imu_offset_ms, manual_ms=40.0)
        stale = run_prelabel(conn, "pose", raw, [], policy, ontology, _at(3))
        assert _poses(conn) == [] and _world(conn) == []
        n = len(get_labels(conn, "pose"))
        again = run_prelabel(conn, "pose", raw, [], policy, ontology, _at(4))
        assert len(get_labels(conn, "pose")) == n
    assert same.pose_note is not None and "깊이 모델" in same.pose_note
    assert stale.pose_note is not None and stale.pose_note.startswith("[자세 낡음]")
    assert again.pose_note is not None and not again.pose_note.startswith("[자세 낡음]")


def test_reflection_extrinsics_give_a_reason_not_poses(
    tmp_path: Path, ontology: Ontology, policy: PrelabelPolicy
) -> None:
    """F-PAR-49 재현: 축 부호 하나를 틀리게 적은 외부 파라미터(행렬식 -1)는 자세를 만들지 않고
    이유를 남긴다 (예전에는 회전이 아닌 행렬로 자세를 최고 신뢰도로 냈다)."""
    sc = generate_camera_motion(11, duration_s=1.0)
    flipped = np.array(sc.imu_to_camera, dtype=np.float64)
    flipped[0, :3] *= -1
    engine = _db(tmp_path, ontology)
    raw = _pose_session(engine, tmp_path, sc, extrinsics=flipped.tolist())
    with engine.begin() as conn:
        got = run_prelabel(
            conn, "pose", raw, [], policy, ontology, _at(1), pose_estimator=_oracle(sc)
        )
        assert _poses(conn) == []
    assert got.pose_note is not None and "행렬식" in got.pose_note


@pytest.mark.parametrize(
    ("extr", "reason"),
    [
        ([[1.0, 0.0], [0.0, 1.0]], "모양"),
        ([[1.0, 0.0, 0.0], [0.0, 1.0], [0.0, 0.0, 1.0]], "행 길이"),
        ([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]], "직교"),
        ([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], "행렬식"),
    ],
)
def test_imu_to_camera_rejects_non_rotations_without_raising(
    policy: PrelabelPolicy, extr: Any, reason: str
) -> None:
    """세션 외부 파라미터가 2x2·들쭉날쭉한 행·배율·반사면 예외 없이 이유를 돌려준다 (F-PAR-49).
    소수 몇 자리로 적은 거의 회전은 가장 가까운 회전으로 고쳐 쓴다."""
    session = make_session()
    bad = session.model_copy(
        update={
            "calibration": session.calibration.model_copy(update={"camera_imu_extrinsics": extr})
        }
    )
    r, why = imu_to_camera(bad, policy)
    assert r is None and why is not None and reason in why
    c = 0.7071
    near = [[c, -c, 0.0], [c, c, 0.0], [0.0, 0.0, 1.0]]
    ok = session.model_copy(
        update={
            "calibration": session.calibration.model_copy(update={"camera_imu_extrinsics": near})
        }
    )
    r, why = imu_to_camera(ok, policy)
    assert why is None and r is not None
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-12) and np.linalg.det(r) == pytest.approx(1.0)


def test_runner_passes_master_time_imu_and_the_lifter_depth_to_the_estimator(
    tmp_path: Path, ontology: Ontology, policy: PrelabelPolicy
) -> None:
    """F-PAR-52: 러너가 IMU Parquet을 읽어 `Stream.to_master_ms`(오프셋 50 + 사람 조정 25 ms)로 바꿔
    추정기에 넘기고, 깊이는 깊이 단계 모델(`ModelDepth`)이다. 자세 라벨은 세션 캘리브레이션을 써서
    내부 파라미터 출처가 calibrated다 (F-PAR-51)."""
    sc = generate_camera_motion(11, duration_s=1.0, imu_offset_ms=75.0)
    engine = _db(tmp_path, ontology)
    raw = _pose_session(engine, tmp_path, sc)
    spy = _Spy(_oracle(sc))
    lifter = _UnitLifter(policy)
    with engine.begin() as conn:
        _set_imu_sync(conn, offset_ms=50.0, manual_ms=25.0)
        run_prelabel(
            conn, "pose", raw, [], policy, ontology, _at(1), lifter=lifter, pose_estimator=spy
        )
        [pose] = _poses(conn)
    assert spy.imu is not None
    assert spy.imu.t_ms == pytest.approx(sc.imu_t_ms + 75.0)
    assert isinstance(spy.depth, ModelDepth) and spy.depth.model is lifter.model
    assert isinstance(pose.payload, Trajectory3DPayload)
    assert pose.payload.intrinsics is IntrinsicsSource.CALIBRATED


def test_pose_is_not_re_estimated_when_only_the_hand_fit_policy_changes(
    tmp_path: Path, ontology: Ontology, policy: PrelabelPolicy
) -> None:
    """F-PAR-53: 자세 버전의 깊이 입력은 깊이 가중치 판·화각 근사 값뿐이라, depth 절의 손 맞춤
    정책만 바꾸면 자세를 다시 추정하지 않는다 (예전에는 depth 절 전체 해시라 자세·월드 3D를 다시
    만들었다).
    IMU 동기화를 A → B → A로 되돌리면 자세를 A로 다시 만든다 (F-PAR-54, 버전 끝 +r1)."""
    sc = generate_camera_motion(11, duration_s=1.0)
    engine = _db(tmp_path, ontology)
    raw = _pose_session(engine, tmp_path, sc)
    fit = policy.depth.hand_fit.model_copy(update={"depth_weight": 0.3})
    other = policy.model_copy(update={"depth": policy.depth.model_copy(update={"hand_fit": fit})})
    with engine.begin() as conn:
        first = run_prelabel(
            conn, "pose", raw, [], policy, ontology, _at(1), lifter=_UnitLifter(policy),
            pose_estimator=_oracle(sc),
        )  # fmt: skip
        second = run_prelabel(
            conn, "pose", raw, [], other, ontology, _at(2), lifter=_UnitLifter(other),
            pose_estimator=_oracle(sc),
        )  # fmt: skip
        assert first.poses > 0 and second.poses == 0
        counts: list[int] = []
        for hour, manual in ((3, 30.0), (4, 0.0)):
            _set_imu_sync(conn, offset_ms=sc.imu_offset_ms, manual_ms=manual)
            got = run_prelabel(
                conn, "pose", raw, [], policy, ontology, _at(hour), lifter=_UnitLifter(policy),
                pose_estimator=_oracle(sc),
            )  # fmt: skip
            counts.append(got.poses)
        [pose] = _poses(conn)
    assert all(c > 0 for c in counts)
    assert (pose.provenance.model_version or "").endswith("+r1")
