"""장면 좌표 바디캠 자세 내보내기 (ADR 0070, DB 없이).

정답은 식으로 정한 카메라 움직임이다: 카메라 중심이 시간에 선형으로 움직이고, 카메라 → 장면 회전이
한 축 둘레로 일정한 각속도로 돈다. 위치 기록에는 그 움직임을 200 ms마다 적되 200 ms 앞과 1800 ms
뒤는 적지 않고 800~1400 ms 사이(간격 600 ms, 한도 500 ms 초과)를 비운다. 이 움직임에서는 중심
선형·회전 구면 보간이 정답과 정확히 같으므로, 내보낸 값이 "그 프레임 시각의 정답 자세"인지 식으로
판정한다. 값이 있어야 하는 프레임은 구현과 따로 시각만 보고 정한다 (`expected_valid`).

위치 기록은 대역(`FakeScenes`)이 돌려준다. 기록이 지금 복원으로 만든 것인지는 `dlp_scene3d` 시험이
본다 (`packages/scene3d/tests/test_localize.py`).
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from numpy.typing import NDArray

from dlp_export.gr00t import StateLayout, write_gr00t
from dlp_export.gr00t_check import check_dataset
from dlp_export.lerobot import write_package
from dlp_export.policy import ExportPolicy
from dlp_export.runner import BuiltEpisodes
from dlp_export.scene import (
    FEATURE,
    NAMES,
    Found,
    StoredScenes,
    quaternion_wxyz,
)
from dlp_export.source import ExportSession
from dlp_media.storage import LocalStore
from dlp_schema.scene3d import EgoLocalization
from dlp_schema.testing import FIXED_TIME

from .test_tactile import IDS, World, left_glove, right_glove, world

# pyarrow에는 타입 스텁이 없다 (dlp_export.gr00t와 같다)
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

__all__ = ["world"]  # test_tactile의 world 고정 장치를 이 모듈에서도 쓴다

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="FFmpeg 필요"
)

F64 = NDArray[np.float64]
# 정답 움직임: 중심 C(t) = C0 + V t, 카메라 → 장면 회전 = AXIS 둘레 TH0 + OMEGA t (t는 초)
AXIS = np.array([1.0, 2.0, 2.0]) / 3.0
C0 = np.array([0.4, -0.2, 1.1])
V = np.array([0.3, 0.05, -0.1])
TH0, OMEGA = 0.3, 0.8
# 위치 기록의 자세 시각 (바디캠 PTS ms): 800과 1400 사이를 비운다
POSE_TIMES = (200, 400, 600, 800, 1400, 1600, 1800)
MAX_GAP_MS = 500


def rotation(axis: F64, angle: float) -> F64:
    """축-각 → 회전 행렬 (로드리게스 식, 구현과 따로 둔다)."""
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def from_quaternion(q: F64) -> F64:
    """단위 사원수 (w, x, y, z) → 회전 행렬 (구현과 따로 둔다)."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def truth(t_ms: float) -> tuple[F64, F64]:
    """시각의 정답 (카메라 → 장면 회전, 카메라 중심)."""
    s = t_ms / 1000
    return rotation(AXIS, TH0 + OMEGA * s), C0 + V * s


def expected_valid(t_ms: float) -> bool:
    """값이 있어야 하는 프레임: 기록 범위 안이고 비운 구간(800~1400 ms 사이) 밖."""
    return 200 <= t_ms <= 800 or 1400 <= t_ms <= 1800


def record_for(
    es: ExportSession, *, render: str | None = None, metric: bool = True
) -> EgoLocalization:
    """정답 움직임을 `POSE_TIMES`에 적은 위치 기록 (`x_cam = R x + t`, R = 회전ᵀ, t = -R C)."""
    poses: list[dict[str, Any]] = []
    for t in POSE_TIMES:
        r_sc, c = truth(t)
        r = r_sc.T
        poses.append({
            "t_ms": t, "reference": "before", "rotation": [float(v) for v in r.reshape(-1)],
            "translation": [float(v) for v in -r @ c], "matches": 60, "inliers": 45,
            "reproj_px": 0.4,
        })  # fmt: skip
    return EgoLocalization.model_validate({
        "session_id": es.session.session_id,
        "stream_id": "bodycam",
        "model_version": "scene3d-localize-1+feat:oracle-features-1+p0+scene3d-align-1+p0",
        "before_reconstruction": "b" * 12,
        "after_reconstruction": "a" * 12,
        "alignment_tag": "c" * 12,
        "alignment": {
            "model_version": "scene3d-align-1+p0", "scale": 1.0,
            "rotation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "translation": [0.0, 0.0, 0.0], "matches": 10, "inliers": 10, "rms_m": 0.0,
            "accepted": True, "reason": None,
        },
        "render_hash": render or es.render_hashes["bodycam"],
        "intrinsics": {"width": 64, "height": 48, "fx": 50, "fy": 50, "cx": 32, "cy": 24},
        "intrinsics_origin": "session",
        "metric": metric,
        "frame_count": 11,
        "localized": len(poses),
        "poses": poses,
        "created_at": FIXED_TIME.isoformat(),
    })  # fmt: skip


@dataclass
class FakeScenes:
    """위치 기록 대역: 세션 ID → 기록·"stale" (없으면 "none")."""

    found: dict[str, Found] = field(default_factory=dict[str, Found])
    max_gap_ms: int = MAX_GAP_MS
    asked: list[str] = field(default_factory=list[str])

    def find(self, session_id: str) -> Found:
        self.asked.append(session_id)
        return self.found.get(session_id, "none")


def times_of(built: BuiltEpisodes, i: int) -> F64:
    """에피소드 i의 프레임 시각 (마스터 ms)."""
    return built.items[i][2].times_ms


def test_quaternion_is_the_rotation_with_nonnegative_w() -> None:
    """임의 회전과 180도 회전(셰퍼드 방법의 모든 갈래)에서 사원수 → 행렬이 원래 행렬이다."""
    rng = np.random.default_rng(0)
    cases = [rotation(np.array(a, float), np.pi) for a in ([1, 0, 0], [0, 1, 0], [0, 0, 1])]
    for _ in range(200):
        axis = rng.normal(size=3)
        cases.append(rotation(axis / np.linalg.norm(axis), float(rng.uniform(-np.pi, np.pi))))
    for r in cases:
        q = quaternion_wxyz(r)
        assert q[0] >= 0 and np.isclose(np.linalg.norm(q), 1.0)
        assert np.allclose(from_quaternion(q), r, atol=1e-9)


def test_exported_poses_follow_the_camera_motion_and_leave_gaps_missing(world: World) -> None:
    """프레임 값 = 그 시각의 정답 자세. 기록 밖·비운 구간은 값 0·표시 0이다.

    뜻도 본다: 정답 자세의 카메라 좌표 점 p를 내보낸 값으로 R(q) p + (x, y, z)로 옮기면 처음의
    장면 점이다 (meta의 `pose` 설명 그대로).
    """
    es = world.session("s1")
    scenes = FakeScenes({"s1": record_for(es)})
    built = world.build(es, scenes=scenes)
    assert built.scene is not None and built.scene.carried
    frames = times_of(built, 0)
    arr = built.scene.lerobot_arrays(0)["scene_camera"]
    assert arr.shape == (frames.size, 8) and arr.dtype == np.float32
    want = np.array([expected_valid(float(t)) for t in frames])
    # 정답이 세 갈래(앞·비운 구간·뒤의 없음, 있음)를 모두 지나는지부터 본다
    assert want.any() and (~want).sum() >= 3
    assert np.array_equal(arr[:, 7] == 1.0, want)
    assert not arr[~want].any()
    point = np.array([1.5, -0.7, 2.2])
    for t, row in zip(frames[want], arr[want], strict=True):
        r_sc, c = truth(float(t))
        assert np.allclose(row[:3], c, atol=1e-5)
        assert np.allclose(from_quaternion(row[3:7].astype(np.float64)), r_sc, atol=1e-5)
        p_cam = r_sc.T @ (point - c)  # 정답 자세의 카메라 좌표 점
        moved = from_quaternion(row[3:7].astype(np.float64)) @ p_cam + row[:3]
        assert np.allclose(moved, point, atol=1e-4)
    (session,) = built.scene.sessions
    assert session == {
        "session_id": IDS.session("s1"),
        "status": "exported",
        "metric": True,
        "model_version": record_for(es).model_version,
        "frames": int(frames.size),
        "frames_valid": int(want.sum()),
        "localized_frames": len(POSE_TIMES),
        "localization_frames_tried": 11,
    }
    assert built.warnings == []


def test_unusable_localizations_are_marked_and_not_exported(world: World) -> None:
    """낡은 기록·다른 블러본의 기록·기록 없음은 싣지 않고 상태와 경고로 남긴다 (지어내지 않음).

    경고와 세션 기록에는 가명만 있다 (내부 세션 ID·태그·렌더 해시 없음).
    """
    s1, s2, s3, s4 = (world.session(sid) for sid in ("s1", "s2", "s3", "s4"))
    other = "f" * 64
    scenes = FakeScenes({"s1": record_for(s1), "s2": "stale", "s3": record_for(s3, render=other)})
    built = world.build(s1, s2, s3, s4, scenes=scenes)
    assert built.scene is not None
    statuses = [s["status"] for s in built.scene.sessions]
    assert statuses == ["exported", "stale", "other_video", "no_localization"]
    for i in (1, 2, 3):
        arr = built.scene.lerobot_arrays(i)["scene_camera"]
        assert arr.shape == (times_of(built, i).size, 8) and not arr.any()
    assert len(built.warnings) == 2
    assert IDS.session("s2") in built.warnings[0] and IDS.session("s3") in built.warnings[1]
    text = json.dumps([built.warnings, built.scene.meta("lerobot")], ensure_ascii=False)
    for needle in ('"s1"', '"s2"', "s3:", "bbbbbbbbbbbb", other, s1.render_hashes["bodycam"]):
        assert needle not in text, needle
    assert scenes.asked == ["s1", "s2", "s3", "s4"]


def test_without_localizations_the_export_is_unchanged(world: World, policy: ExportPolicy) -> None:
    """위치 기록이 있는 세션이 없거나 정책이 끄면 장면 기록이 없다 (결과물이 이 기능 전과 같다)."""
    es = world.session("s1")
    assert world.build(es, scenes=FakeScenes()).scene is None
    assert world.build(es).scene is None
    off = policy.model_copy(update={"scene": policy.scene.model_copy(update={"enabled": False})})
    scenes = FakeScenes({"s1": record_for(es)})
    assert world.build(es, policy=off, scenes=scenes).scene is None
    assert scenes.asked == []  # 끄면 위치 기록을 찾지도 않는다
    # 기록이 모두 낡았으면 세션 기록만 있고 특징은 더하지 않는다
    stale = world.build(es, scenes=FakeScenes({"s1": "stale"})).scene
    assert stale is not None and not stale.carried


def test_stored_scenes_without_a_record_is_none(tmp_path: Path) -> None:
    """라벨링 버킷에 위치 기록이 없으면 "none"이다 (원본 버킷을 열지 않는다)."""
    scenes = StoredScenes(LocalStore(tmp_path / "store", "dlp-labeling"), MAX_GAP_MS)
    assert scenes.find("s1") == "none"


def test_lerobot_package_carries_the_scene_feature(world: World) -> None:
    """LeRobot 묶음: 특징 정의와 npz 배열이 더해지고, 장면 기록이 없으면 그대로다."""
    es = world.session("s1")
    built = world.build(es, scenes=FakeScenes({"s1": record_for(es)}))
    episodes = [(e.session, video, ep) for e, video, ep in built.items]
    pkg = write_package(
        episodes, world.policy, built.size, world.tmp / "pkg", IDS, None, None, built.scene
    )
    spec = json.loads(pkg.read_text("utf-8"))
    assert spec["features"][FEATURE] == {
        "dtype": "float32", "shape": [8], "names": [*NAMES, "scene_camera.present"],
    }  # fmt: skip
    assert spec["frame_arrays"][FEATURE] == "scene_camera"
    npz = np.load(spec["episodes"][0]["npz"])
    assert built.scene is not None
    assert np.array_equal(npz["scene_camera"], built.scene.lerobot_arrays(0)["scene_camera"])
    plain = write_package(episodes, world.policy, built.size, world.tmp / "plain", IDS)
    assert FEATURE not in json.loads(plain.read_text("utf-8"))["features"]


def _gr00t(world: World, built: BuiltEpisodes, name: str) -> Path:
    dest = world.tmp / name
    write_gr00t(
        built.items, built.sessions, world.policy, world.vocab, built.size, dest,
        built.tactile, built.scene,
    )  # fmt: skip
    return dest


def _scene_in_action(dest: Path) -> None:
    path = dest / "meta" / "modality.json"
    m = json.loads(path.read_text("utf-8"))
    m["action"]["scene_camera"] = m["state"]["scene_camera"]
    path.write_text(json.dumps(m), "utf-8")


def _clear_scene_flags(dest: Path) -> None:
    path = dest / "data/chunk-000/episode_000000.parquet"
    t = pq.read_table(path)
    i = t.column_names.index("observation.scene_camera_valid")
    zeros = [[0.0] for _ in range(t.num_rows)]
    pq.write_table(t.set_column(i, t.field(i), pa.array(zeros, type=t.field(i).type)), path)


@needs_ffmpeg
def test_gr00t_carries_the_pose_after_tactile_and_the_check_guards_it(world: World) -> None:
    """GR00T: 자세가 촉각 구간 뒤 `scene_camera` 구간(50:57)에 실리고 확인기를 통과한다.

    action은 손·팔 구간(42)만이다. 구간을 action에 넣거나 표시를 지우면 확인기가 알린다.
    """
    es = world.session("s1", [right_glove(), left_glove()])
    built = world.build(es, scenes=FakeScenes({"s1": record_for(es)}))
    dest = _gr00t(world, built, "gr00t")
    assert check_dataset(dest, world.policy.gr00t.video.ffprobe) == []
    base = len(StateLayout.from_policy(world.policy.lerobot).names)
    modality = json.loads((dest / "meta" / "modality.json").read_text("utf-8"))
    assert (base, modality["state"]["right_tactile"]["end"]) == (42, 50)
    assert modality["state"]["scene_camera"] == {"start": 50, "end": 57}
    assert "scene_camera" not in modality["action"]
    info = json.loads((dest / "meta" / "info.json").read_text("utf-8"))["features"]
    assert info["observation.state"]["names"][50:] == list(NAMES)
    assert info["action"]["shape"] == [42]
    assert info["observation.scene_camera_valid"]["names"] == ["scene_camera"]
    table = pq.read_table(dest / "data/chunk-000/episode_000000.parquet").to_pydict()
    state = np.array(table["observation.state"])
    valid = np.array(table["observation.scene_camera_valid"])[:, 0]
    assert built.scene is not None
    want = built.scene.lerobot_arrays(0)["scene_camera"]
    assert np.allclose(state[:, 50:57], want[:, :7]) and np.array_equal(valid, want[:, 7])
    meta = json.loads((dest / "meta" / "dlp_scene.json").read_text("utf-8"))
    assert meta == built.scene.meta("gr00t") and meta["format"] == "dlp-scene"
    for corrupt, expected in (
        (_scene_in_action, "구간 scene_camera가 modality.json action에 있음"),
        (_clear_scene_flags, "scene_camera 값이 없다고 표시한 행에 0이 아닌 값"),
    ):
        broken = _gr00t(world, built, f"broken-{corrupt.__name__}")
        corrupt(broken)
        problems = check_dataset(broken, world.policy.gr00t.video.ffprobe)
        assert any(expected in p for p in problems), problems
