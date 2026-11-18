"""장면 안 바디캠 위치 추정 종단 시험 (로컬 저장소, SQLite, CPU stub, ADR 0067).

정답은 `dlp_fixtures.scene3d.generate_ego_scene`의 합성 방을 지나는 바디캠 경로다 (프레임별 정답
자세는 before 좌표). after 복원은 before와 다른 좌표계(알려진 닮음 변환, 척도 1.25)라, after
기준으로 푼 자세는 정렬을 거쳐야 정답과 맞는다. 프레임 특징은 정답 stub(`OracleFrameFeatures`:
화소 잡음 0.5, 틀린 대응 30%, 잡동사니 20%)이고 대응·PnP·좌표 변환·저장은 실제 코드다.
"""

from __future__ import annotations

import itertools
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import sqlalchemy as sa

from dlp_fixtures.scene3d import generate_ego_scene, write_scene_pair
from dlp_privacy.errors import RenderNotCurrentError
from dlp_scene3d.backends import FrameFeatures, FrameImage, InitPointsSplat, OracleFrameFeatures
from dlp_scene3d.layout import get_pointer
from dlp_scene3d.localize import (
    current_localization,
    localization_key,
    localize,
    stored_localization,
)
from dlp_scene3d.oracle import load_oracle, oracle_features
from dlp_scene3d.pipeline import AlignmentRejectedError, Scene3dError, reconstruct
from dlp_scene3d.policy import Scene3dPolicy
from dlp_schema.db.repository import insert_withdrawal
from dlp_schema.lineage import Withdrawal
from dlp_schema.scene3d import EgoLocalization
from dlp_schema.session import CameraIntrinsics, PrivacyState

from .world import (
    NOW,
    POLICY,
    PRIVACY,
    RAW_BUCKET,
    Stores,
    bodycam_session,
    build_scene,
    ego_intrinsics,
    ego_truth,
    open_stores,
    with_localization,
)

SID = "s-ego"
# 합성 정답에서 잰 값(위치 0.4 cm, 회전 0.11도 이내)의 다섯 배쯤
MAX_POSITION_M = 0.02
MAX_ROTATION_DEG = 0.5


@dataclass
class SpyFeatures:
    """받은 프레임의 시각·크기를 적는 특징 추출기.

    프레임이 블러본에서 왔는지, 같은 입력에서 다시 불렸는지 본다.
    """

    inner: OracleFrameFeatures
    seen: list[int] = field(default_factory=list[int])
    shapes: set[tuple[int, ...]] = field(default_factory=set[tuple[int, ...]])
    name: str = "oracle"
    version: str = "oracle-features-1"

    def extract(self, frames: Sequence[FrameImage], workdir: Path) -> dict[int, FrameFeatures]:
        self.seen += [f.t_ms for f in frames]
        self.shapes |= {f.image.shape for f in frames}
        return self.inner.extract(frames, workdir)


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """(truth.json, 두 단계 복원까지 끝난 저장소 뿌리). 모듈에서 한 번만 만든다."""
    root = tmp_path_factory.mktemp("ego")
    pair, ego = generate_ego_scene(0)
    truth = write_scene_pair(pair, root / "photos", ego=ego)
    build_scene(truth, open_stores(root / "store"), SID, root / "work")
    return truth, root / "store"


@pytest.fixture
def world(base: tuple[Path, Path], tmp_path: Path) -> tuple[Path, Stores]:
    """시험마다 따로 쓰는 저장소 사본."""
    truth, store = base
    shutil.copytree(store, tmp_path / "store")
    return truth, open_stores(tmp_path / "store")


def _session(
    world: tuple[Path, Stores], tmp_path: Path, **kwargs: Any
) -> tuple[sa.Engine, SpyFeatures]:
    truth, stores = world
    kwargs.setdefault("intrinsics", ego_intrinsics(truth))
    engine = bodycam_session(stores, SID, truth, tmp_path / "db", **kwargs)
    return engine, SpyFeatures(oracle_features(truth))


def _run(
    world: tuple[Path, Stores],
    engine: sa.Engine,
    features: SpyFeatures,
    work: Path,
    policy: Scene3dPolicy = POLICY,
) -> tuple[EgoLocalization, bool]:
    return localize(
        engine, world[1].labeling, SID, features, policy, PRIVACY, work, NOW, raw_bucket=RAW_BUCKET
    )


def _pose_errors(record: EgoLocalization, truth: Path) -> dict[int, tuple[float, float]]:
    """프레임 시각 → (카메라 중심 오차 m, 회전 오차 도). 정답은 before 좌표의 프레임별 자세."""
    frames = cast(list[dict[str, Any]], ego_truth(truth)["frames"])
    by_time = {int(f["t_ms"]): (np.asarray(f["r"]), np.asarray(f["t"])) for f in frames}
    out: dict[int, tuple[float, float]] = {}
    for pose in record.poses:
        r_true, t_true = by_time[pose.t_ms]
        r = np.asarray(pose.rotation).reshape(3, 3)
        t = np.asarray(pose.translation)
        position = float(np.linalg.norm(-r.T @ t - (-r_true.T @ t_true)))
        cos = float(np.clip((np.trace(r @ r_true.T) - 1) / 2, -1.0, 1.0))
        out[pose.t_ms] = (position, float(np.rad2deg(np.arccos(cos))))
    return out


def test_localizes_bodycam_in_before_frame(world: tuple[Path, Stores], tmp_path: Path) -> None:
    """잡음·틀린 대응이 있어도 프레임 자세가 정답(before 좌표)과 2 cm·0.5도 안에서 맞는다.

    벽에 바짝 붙어 특징이 모자란 구간은 기록에 없고, 닦인 벽만 보는 끝 구간은 after 복원으로 풀려
    정렬을 거친 자세가 정답과 맞는다. 프레임은 블러본에서만 받고 원본 버킷은 열지 않는다.
    """
    truth, stores = world
    engine, spy = _session(world, tmp_path)
    audited = len(stores.sink.events)
    record, new = _run(world, engine, spy, tmp_path / "w")
    ego = ego_truth(truth)
    times = [int(f["t_ms"]) for f in cast(list[dict[str, Any]], ego["frames"])]
    duration = times[-1]

    assert new and record.stream_id == "bodycam" and record.metric
    assert record.intrinsics_origin == "session" and record.intrinsics == ego_intrinsics(truth)
    assert len(stores.sink.events) == audited, "위치 추정이 원본 버킷을 읽었다"
    # 프레임은 블러본(640x480)의 PTS 시각이고 정책 간격(200 ms) 이상 떨어져 있다
    assert spy.shapes == {(480, 640, 3)} and set(spy.seen) <= set(times)
    assert spy.seen == sorted(spy.seen) and len(spy.seen) == record.frame_count
    gaps = [b - a for a, b in itertools.pairwise(spy.seen)]
    assert min(gaps) >= POLICY.localization.frame_interval_ms

    errors = _pose_errors(record, truth)
    assert errors and max(e[0] for e in errors.values()) < MAX_POSITION_M
    assert max(e[1] for e in errors.values()) < MAX_ROTATION_DEG
    # 벽에 바짝 붙은 구간(진행 0.52~0.58)은 특징이 모자라 자세를 남기지 않는다
    starved = [t for t in spy.seen if 0.52 * duration <= t <= 0.58 * duration]
    assert starved and not set(starved) & set(errors)
    assert record.localized == len(record.poses) < record.frame_count
    # 닦인 벽만 보는 끝 구간은 after 복원으로만 풀린다. 그 앞(벽이 바뀌기 전)은 before 기준이다
    wall = int(cast(dict[str, int], ego["change_ms"])["wall"])
    tail = [p for p in record.poses if p.t_ms >= 0.9 * duration]
    assert len(tail) >= 5 and {p.reference for p in tail} == {"after"}
    assert {p.reference for p in record.poses if p.t_ms < wall} == {"before"}
    assert all(p.inliers >= POLICY.localization.min_inliers for p in record.poses)
    assert current_localization(stores.labeling, SID) == record


def test_rerun_skips_and_policy_change_recomputes(
    world: tuple[Path, Stores], tmp_path: Path
) -> None:
    """같은 입력이면 같은 태그라 다시 계산하지 않는다. 위치 정책을 바꾸면 새 태그로 다시 계산하고
    앞 기록은 그대로 남는다 (현재 표시만 옮긴다)."""
    _, stores = world
    engine, spy = _session(world, tmp_path)
    first, new = _run(world, engine, spy, tmp_path / "w1")
    calls = len(spy.seen)
    again, new_again = _run(world, engine, spy, tmp_path / "w2")
    assert new and not new_again and again == first and len(spy.seen) == calls
    found = stored_localization(stores.labeling, SID)
    assert found is not None
    tag = found[0]

    sparse = with_localization(POLICY, frame_interval_ms=400)
    second, new_second = _run(world, engine, spy, tmp_path / "w3", sparse)
    assert new_second and second.frame_count < first.frame_count
    pointer = get_pointer(stores.labeling, f"sessions/{SID}/scene3d/localization/current.json")
    assert pointer is not None and pointer["tag"] != tag
    assert stores.labeling.head(localization_key(SID, tag)) is not None
    # 예전 정책으로 되돌리면 예전 기록을 다시 가리킨다 (계산 없이)
    back, new_back = _run(world, engine, spy, tmp_path / "w4")
    assert not new_back and back == first


def test_requires_both_reconstructions(tmp_path: Path, base: tuple[Path, Path]) -> None:
    """after 복원이 없으면 무엇이 빠졌는지 알리고 실패한다 (before만으로 진행하지 않는다)."""
    truth, _ = base
    stores = open_stores(tmp_path / "store")
    build_scene(truth, stores, SID, tmp_path / "work", phases=("before",))
    engine, spy = _session((truth, stores), tmp_path)
    with pytest.raises(Scene3dError, match=r"전후 복원이 모두 필요.*after.*복원이 없습니다"):
        _run((truth, stores), engine, spy, tmp_path / "w")
    assert not spy.seen and stored_localization(stores.labeling, SID) is None


def test_refuses_rejected_alignment_and_stale_record_is_not_current(
    world: tuple[Path, Stores], tmp_path: Path
) -> None:
    """정렬이 거부되면 위치를 추정하지 않는다. before를 다시 복원하면(새 태그) 예전 위치 기록은
    지금 것이 아니다 (장면 좌표계가 달라졌을 수 있다)."""
    truth, stores = world
    engine, spy = _session(world, tmp_path)
    strict = Scene3dPolicy.model_validate(
        {
            **POLICY.model_dump(mode="json"),
            "alignment": {**POLICY.alignment.model_dump(mode="json"), "min_inliers": 100_000},
        }
    )
    with pytest.raises(AlignmentRejectedError, match="정렬 거부"):
        _run(world, engine, spy, tmp_path / "w", strict)
    assert not spy.seen and stored_localization(stores.labeling, SID) is None

    record, _ = _run(world, engine, spy, tmp_path / "w2")
    assert record.localized > 0
    other = Scene3dPolicy.model_validate(
        {
            **POLICY.model_dump(mode="json"),
            "reconstruction": {
                **POLICY.reconstruction.model_dump(mode="json"),
                "min_registered_ratio": 0.5,
            },
        }
    )
    sfm = load_oracle(truth, "before").sfm
    splat = InitPointsSplat(POLICY.reconstruction.init_scale)
    reconstruct(stores.labeling, SID, "before", sfm, splat, other, tmp_path / "r", NOW)
    assert stored_localization(stores.labeling, SID) is not None
    with pytest.raises(Scene3dError, match="지금 복원과 다릅니다"):
        current_localization(stores.labeling, SID)


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"intrinsics": None}, Scene3dError, "내부 파라미터가 등록돼 있지 않습니다"),
        ({"privacy": PrivacyState.AUTO_BLURRED}, RenderNotCurrentError, "승인 상태가 아닙니다"),
        ({"publish": False}, RenderNotCurrentError, "렌더 기록"),
        (
            {"intrinsics": CameraIntrinsics(width=640, height=360, fx=400, fy=400, cx=320, cy=180)},
            Scene3dError,
            "화면비",
        ),
    ],
)
def test_refuses_without_prerequisites(
    world: tuple[Path, Stores],
    tmp_path: Path,
    kwargs: dict[str, Any],
    error: type[Exception],
    message: str,
) -> None:
    """내부 파라미터가 없거나 영상과 맞지 않으면, 블러본이 승인·렌더되지 않았으면 거부한다."""
    engine, spy = _session(world, tmp_path, **kwargs)
    with pytest.raises(error, match=message):
        _run(world, engine, spy, tmp_path / "w")
    assert not spy.seen and stored_localization(world[1].labeling, SID) is None


def test_refuses_unknown_and_withdrawn_sessions(world: tuple[Path, Stores], tmp_path: Path) -> None:
    """`sessions` 표에 없는 세션과 사용 중지된 세션은 거부한다."""
    truth, stores = world
    spy = SpyFeatures(oracle_features(truth))
    empty = bodycam_session(stores, "other", truth, tmp_path / "db0", intrinsics=None)
    with pytest.raises(Scene3dError, match="세션이 등록돼 있지 않습니다"):
        _run(world, empty, spy, tmp_path / "w0")
    engine, _ = _session(world, tmp_path)
    with engine.begin() as conn:
        insert_withdrawal(conn, Withdrawal(session_id=SID, reason="동의 철회", withdrawn_at=NOW))
    with pytest.raises(Scene3dError, match="사용 중지"):
        _run(world, engine, spy, tmp_path / "w1")
    assert not spy.seen
