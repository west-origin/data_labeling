"""촉각장갑 채널 내보내기 (ADR 0069, DB 없이).

정답은 `dlp_fixtures.tactile`의 장갑 기록이다. 채널 값이 마스터 시각의 식이고 장갑 시계의 오프셋·
드리프트를 알므로, 내보낸 프레임 값이 "그 프레임 시각의 정답 값"인지 식으로 판정한다. 오프셋과
드리프트는 빼먹으면 값이 허용 오차의 수십 배 어긋나는 크기로 둔다 (시험이 그 크기를 스스로
확인한다). 값이 있어야 하는 프레임은 구현과 따로 프레임마다 앞뒤 샘플을 직접 찾아 정한다
(`conftest.glove_oracle`).

장갑 Parquet은 로컬 원본 버킷에 두고 감사 저장소(`AuditedStore` + `MemorySink`)로만 읽게 해,
읽기마다 접근 기록이 남는지와 결과물에 원본 버킷 이름·키가 없는지도 본다.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import sqlalchemy as sa
from dlp_retarget.ik import DlsIkSolver
from dlp_retarget.policy import load_policy as load_retarget_policy
from dlp_retarget.robot import load_robot

from dlp_datasets.snapshot import LocalSnapshotStore
from dlp_export.gr00t import StateLayout, write_gr00t
from dlp_export.gr00t_check import check_dataset
from dlp_export.lerobot import FRAME_ARRAYS, Vocab, features, run_script, write_package
from dlp_export.policy import ExportPolicy, TactilePolicy
from dlp_export.pseudonym import Pseudonymizer
from dlp_export.robot import retarget_session, robot_override
from dlp_export.runner import BuiltEpisodes, build_episodes, run_export
from dlp_export.scene import SceneSource
from dlp_export.source import ExportError, ExportSession, ExportSource, label_states, select_labels
from dlp_export.tactile import TactileData, frame_values, select_channels
from dlp_fixtures.sync import ClockTruth
from dlp_fixtures.tactile import PRESSURE_CHANNELS, generate_glove_truth
from dlp_fixtures.video import vfr_times
from dlp_media.audit import AuditedStore, MemorySink
from dlp_media.storage import LocalStore
from dlp_schema.dataset import DatasetVersion, Split
from dlp_schema.labels import Hand
from dlp_schema.ontology import Ontology
from dlp_schema.testing import FIXED_TIME, make_session

from .conftest import (
    RAW_BUCKET,
    ROOT,
    Glove,
    assert_glove_values,
    glove_key,
    glove_oracle,
    scenario_labels,
    write_blurred,
)

# pyarrow에는 타입 스텁이 없다 (dlp_export.gr00t와 같다)
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

# GR00T 영상은 ffmpeg로 쓰고 ffprobe로 확인한다. 없는 환경에서는 건너뛴다 (test_gr00t.py와 같다)
needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="FFmpeg 필요"
)

# 블러본 길이 (ms). 30 fps 에피소드 프레임이 약 60개 나온다
DURATION_MS = 2000
BODYCAM = {
    "stream_id": "bodycam",
    "kind": "bodycam",
    "uri": f"s3://{RAW_BUCKET}/s001/bodycam.mp4",
    "sync_method": "reference",
}


def right_glove(seed: int = 1, channels: tuple[str, ...] | None = None) -> Glove:
    """오른손 장갑: 영상 시작 뒤(마스터 412.5 ms)에 녹화를 시작해 영상보다 먼저(약 1575 ms) 끝난다.

    시계는 오프셋 +412.5 ms, 배율 1.02 (드리프트 2%: 짧은 영상에서도 빼먹으면 드러나는 크기).
    마스터 905~1010 ms는 샘플이 빠져 앞뒤 간격이 약 112 ms(정책 한도 50 ms 초과)이고, 1320~1345 ms는
    세 샘플만 빠져 간격이 약 41 ms(한도 안)다.
    """
    truth = generate_glove_truth(
        seed,
        clock=ClockTruth(offset_ms=412.5, clock_scale=1.02),
        duration_ms=1150.0,
        channels=channels or PRESSURE_CHANNELS,
        gaps_master_ms=((905.0, 1010.0), (1320.0, 1345.0)),
    )
    return Glove("glove_right", truth)


def left_glove() -> Glove:
    """왼손 장갑: 영상보다 먼저(마스터 -318 ms) 녹화를 시작해 영상 끝까지 덮는다. 채널 3개.

    시계는 오프셋 -318 ms(그중 37.5 ms는 사람 미세 조정으로 적는다), 배율 0.985.
    """
    truth = generate_glove_truth(
        2,
        clock=ClockTruth(offset_ms=-318.0, clock_scale=0.985),
        duration_ms=2500.0,
        channels=("pressure_0", "pressure_1", "pressure_2"),
    )
    return Glove("glove_left", truth, manual_ms=37.5)


@dataclass
class World:
    """시험 한 번의 저장소: 라벨링 버킷(블러본), 원본 버킷(장갑 Parquet), 원본 접근 기록."""

    tmp: Path
    policy: ExportPolicy
    vocab: Vocab
    labeling: LocalStore
    raw_local: LocalStore  # 준비용 (감사 기록 없이 올린다)
    raw: AuditedStore  # 내보내기가 받는 감사 저장소
    sink: MemorySink

    def session(
        self,
        sid: str,
        gloves: Sequence[Glove] = (),
        *,
        model: str | None = "syn-glove-a",
        split: Split = Split.TRAIN,
    ) -> ExportSession:
        """세션 하나: 2초 VFR 블러본과 conftest 정답 라벨, 주어진 장갑 스트림과 그 Parquet."""
        times = vfr_times(np.random.default_rng(4), DURATION_MS)
        digest = write_blurred(self.labeling, sid, times, self.tmp)
        for g in gloves:
            g.upload(self.raw_local, sid, self.tmp)
        session = make_session(
            sid,
            streams=[BODYCAM, *(g.stream(sid) for g in gloves)],
            calibration={"glove_model": model},
        )
        states = label_states(self.policy, False)
        labels = select_labels(scenario_labels(sid, times), self.policy, states, ())
        return ExportSession(session, split, labels, {"bodycam": digest})

    def build(
        self,
        *sessions: ExportSession,
        policy: ExportPolicy | None = None,
        raw: bool = True,
        scenes: SceneSource | None = None,
    ) -> BuiltEpisodes:
        """`build_episodes` (세션 ID 순, 가명 사용). raw=False면 원본 저장소를 넘기지 않는다.

        scenes: 장면 좌표 자세를 찾을 곳 (ADR 0070, `test_scene.py`). 없으면 자세를 싣지 않는다.
        """
        version = DatasetVersion(
            version_id="dv1", ontology_version="1.0.0", created_at=FIXED_TIME,
            snapshot_uri="local-snapshot://x/dv1",
            splits={es.session.session_id: es.split for es in sessions},
        )  # fmt: skip
        ordered = sorted(sessions, key=lambda es: es.session.session_id)
        src = ExportSource(version, ordered, label_states(self.policy, False))
        return build_episodes(
            src, policy or self.policy, self.vocab, self.labeling, self.tmp / "work", IDS,
            raw=self.raw if raw else None, scenes=scenes,
        )  # fmt: skip

    def reads(self) -> list[tuple[str, str, str, str | None]]:
        """원본 접근 기록 (동작, 용도, 키, 세션)."""
        return [(e.action, e.purpose, e.key, e.session_id) for e in self.sink.events]


IDS = Pseudonymizer.for_export("e1", b"secret", enabled=True)


@pytest.fixture
def world(policy: ExportPolicy, ontology: Ontology, tmp_path: Path) -> World:
    """빈 저장소 묶음. 원본 저장소는 감사 저장소로 감싸 읽기를 `sink`에 남긴다."""
    raw_local = LocalStore(tmp_path / "store", RAW_BUCKET)
    sink = MemorySink()
    raw = AuditedStore(raw_local, sink, actor="svc-pipeline", purpose="export.gr00t")
    labeling = LocalStore(tmp_path / "store", "dlp-labeling")
    return World(tmp_path, policy, Vocab.from_ontology(ontology), labeling, raw_local, raw, sink)


def with_tactile(policy: ExportPolicy, **changes: Any) -> ExportPolicy:
    """`tactile` 절의 값만 바꾼 정책 사본."""
    return policy.model_copy(update={"tactile": policy.tactile.model_copy(update=changes)})


def test_frame_values_interpolate_only_between_close_samples() -> None:
    """프레임 값 규칙: 샘플 시각이면 그 값, 두 샘플 사이는 선형 보간, 외삽·한도 초과는 없음.

    정답: 샘플 100·110·120 | 200·210 ms(120~200은 80 ms 빈 구간)에서 한도 50 ms면 150 ms 프레임은
    없음이고, 한도 80 ms(간격과 같음)면 직선 위 값이다. 없는 프레임의 값은 0이다.
    """
    sample = np.array([100.0, 110.0, 120.0, 200.0, 210.0])
    values = np.array([[1.0], [2.0], [3.0], [11.0], [12.0]])
    frames = np.array([90.0, 100.0, 105.0, 120.0, 150.0, 200.0, 205.0, 210.0, 211.0])
    out, valid = frame_values(sample, values, frames, 50)
    assert valid.tolist() == [False, True, True, True, False, True, True, True, False]
    assert out[:, 0].tolist() == [0.0, 1.0, 1.5, 3.0, 0.0, 11.0, 11.5, 12.0, 0.0]
    out, valid = frame_values(sample, values, frames, 80)
    assert valid[4] and out[4, 0] == pytest.approx(3.0 + 8.0 * 30 / 80)
    # 샘플이 없으면 모두 없음
    out, valid = frame_values(np.zeros(0), np.zeros((0, 2)), frames, 50)
    assert not valid.any() and out.shape == (9, 2) and not out.any()


def test_exported_frame_values_follow_the_signal_through_sync_offset_and_drift(
    world: World,
) -> None:
    """내보낸 프레임 값 = 정답 신호를 그 프레임의 마스터 시각에서 읽은 값 (두 손, 서로 다른 시계).

    오른손(오프셋 +412.5 ms, 배율 1.02)은 장갑 기록 전·후 프레임과 112 ms 빈 구간의 프레임이 없음
    이고, 41 ms 빈 구간은 보간해 싣는다. 왼손(오프셋 -318 ms 중 37.5 ms는 사람 조정, 배율 0.985)은
    모든 프레임에 값이 있다. 오프셋이나 드리프트를 빼먹은 값은 정답과 허용 오차의 10배 넘게 다르다
    (이 시험이 동기화 적용 여부를 가른다는 근거). 장갑 Parquet 읽기는 감사 기록에 한 번씩 남는다.
    """
    right, left = right_glove(), left_glove()
    built = world.build(world.session("s1", [right, left]))
    tactile = built.tactile
    assert tactile is not None
    frame_ms = built.items[0][2].times_ms
    gap = world.policy.tactile.interp_max_gap_ms
    assert tactile.hands == {Hand.LEFT: left.truth.channels, Hand.RIGHT: right.truth.channels}
    for hand, glove in ((Hand.RIGHT, right), (Hand.LEFT, left)):
        got = tactile.episodes[0][hand]
        assert got.values.dtype == np.float32 and got.values.shape == (
            frame_ms.size,
            len(glove.truth.channels),
        )
        assert_glove_values(got.values, got.valid, glove.truth, frame_ms, gap)
        # 동기화를 쓰지 않으면 틀린다: 오프셋을 빼먹거나(장갑 시각을 그대로 마스터 시각으로 봄)
        # 드리프트만 빼먹은(오프셋만 더함) 값은 정답과 허용 오차의 10배 넘게 다르다
        want, ok, tol = glove_oracle(glove.truth, frame_ms, gap)
        clock = glove.truth.clock
        no_offset = glove.truth.truth(clock.to_master(frame_ms[ok]))
        no_drift = glove.truth.truth(
            clock.offset_ms + (frame_ms[ok] - clock.offset_ms) * clock.clock_scale
        )
        assert np.abs(no_offset - want[ok]).max() > 10 * tol.max()
        assert np.abs(no_drift - want[ok]).max() > 10 * tol.max()
    # 오른손: 기록 전·기록 후·긴 빈 구간은 없음, 짧은 빈 구간은 있음
    valid = tactile.episodes[0][Hand.RIGHT].valid
    assert not valid[frame_ms < 412.5].any() and not valid[frame_ms > 1576].any()
    assert not valid[(frame_ms > 915) & (frame_ms < 1000)].any()
    assert valid[(frame_ms > 1320) & (frame_ms < 1345)].all()
    assert (frame_ms < 412.5).sum() > 5 and ((frame_ms > 915) & (frame_ms < 1000)).sum() >= 2
    assert ((frame_ms > 1320) & (frame_ms < 1345)).sum() >= 1
    assert tactile.episodes[0][Hand.LEFT].valid.all()
    # 세션 기록: 가명 세션 ID, 장갑 기종, 손마다 상태·동기화 방법·채움 수
    (record,) = tactile.sessions
    assert record["session_id"] == IDS.session("s1") and record["glove_model"] == "syn-glove-a"
    r = record["hands"]["right"]
    master = right.truth.master_ms
    assert r["status"] == "exported" and r["sync_method"] == "tap_event"
    assert r["sync_confidence"] == 0.9 and r["sample_rate_hz"] == pytest.approx(100.0)
    assert r["frames"] == frame_ms.size and r["frames_valid"] == int(valid.sum())
    assert r["frames_in_recording"] == int(
        ((frame_ms >= master[0]) & (frame_ms <= master[-1])).sum()
    )
    assert built.warnings == []  # 채움 비율이 기준(0.9) 이상
    # 원본 접근: 동기화된 장갑 Parquet마다 읽기 한 번 (용도·키·세션이 기록된다)
    assert sorted(world.reads()) == [
        ("read", "export.gr00t", glove_key("s1", "glove_left"), "s1"),
        ("read", "export.gr00t", glove_key("s1", "glove_right"), "s1"),
    ]


def test_samples_farther_apart_than_the_limit_leave_frames_missing_and_warn(world: World) -> None:
    """보간 한도가 장갑 샘플 간격(약 10 ms)보다 작으면 값이 모두 없음이고 내보내기가 경고한다.

    한도를 5 ms로 낮춘 정책: 어느 프레임도 앞뒤 샘플 간격이 한도 안이 아니다. 값을 지어내지 않고
    전부 0·표시 0으로 두며, 채움 비율 경고에 고칠 정책 값(interp_max_gap_ms)을 적는다.
    """
    built = world.build(
        world.session("s1", [right_glove()]),
        policy=with_tactile(world.policy, interp_max_gap_ms=5),
    )
    assert built.tactile is not None
    got = built.tactile.episodes[0][Hand.RIGHT]
    assert not got.valid.any() and not got.values.any()
    assert len(built.warnings) == 1 and "interp_max_gap_ms 5 ms" in built.warnings[0]
    assert built.warnings[0].startswith(IDS.session("s1"))  # 경고의 세션도 가명


def test_lerobot_package_carries_tactile_per_hand(world: World) -> None:
    """LeRobot 묶음: 손마다 `observation.tactile.<손>` = [채널 값…, present] (float32).

    정답: 특징 이름은 `<손>.tactile.<채널>`과 끝의 `<손>.tactile.present`, npz 배열의 마지막 열이
    유효 표시이고 앞 열이 정답 값이다. 기존 특징(observation.state 등)은 그대로다.
    """
    right, left = right_glove(), left_glove()
    built = world.build(world.session("s1", [right, left]))
    episodes = [(es.session, video, ep) for es, video, ep in built.items]
    pkg = write_package(
        episodes, world.policy, built.size, world.tmp / "pkg", IDS, None, built.tactile
    )
    spec = json.loads(pkg.read_text("utf-8"))
    assert spec["features"]["observation.tactile.right"] == {
        "dtype": "float32",
        "shape": [6],
        "names": [*(f"right.tactile.pressure_{i}" for i in range(5)), "right.tactile.present"],
    }
    assert spec["features"]["observation.tactile.left"]["shape"] == [4]
    base = features(world.policy.lerobot, spec["height"], spec["width"])
    assert {k: v for k, v in spec["features"].items() if "tactile" not in k} == base
    assert spec["frame_arrays"] == {
        **FRAME_ARRAYS,
        "observation.tactile.left": "tactile_left",
        "observation.tactile.right": "tactile_right",
    }
    npz = np.load(spec["episodes"][0]["npz"])
    frame_ms = built.items[0][2].times_ms
    gap = world.policy.tactile.interp_max_gap_ms
    for key, glove in (("tactile_right", right), ("tactile_left", left)):
        array = npz[key]
        assert array.dtype == np.float32
        assert array.shape == (frame_ms.size, len(glove.truth.channels) + 1)
        assert_glove_values(array[:, :-1], array[:, -1], glove.truth, frame_ms, gap)
        assert set(np.unique(array[:, -1]).tolist()) <= {0.0, 1.0}
    # 묶음에도 원본 위치·내부 세션 ID가 없다
    text = pkg.read_text("utf-8")
    assert RAW_BUCKET not in text and "derived" not in text and '"s1"' not in text


def test_unsynced_glove_is_not_exported_and_is_recorded(world: World) -> None:
    """동기화되지 않은 장갑은 싣지 않고(Parquet도 읽지 않는다) 그 사실을 기록·경고에 남긴다.

    세션 s1은 오른손 장갑이 동기화 전(unsynced)이다. 정답: 실은 손이 없어 LeRobot 특징이 촉각이
    없던 때와 같고, 세션 기록의 오른손 상태가 not_synced, 왼손은 no_glove다. 원본 접근 기록은 없다.
    """
    built = world.build(world.session("s1", [replace(right_glove(), synced=False)]))
    tactile = built.tactile
    assert tactile is not None and tactile.hands == {} and tactile.episodes == [{}]
    meta = tactile.meta("lerobot")
    assert meta["hands"] == {} and meta["glove_models"] == []
    assert meta["sessions"] == [
        {
            "session_id": IDS.session("s1"),
            "glove_model": "syn-glove-a",
            "hands": {"left": {"status": "no_glove"}, "right": {"status": "not_synced"}},
        }
    ]
    assert len(built.warnings) == 1 and "동기화되지 않아" in built.warnings[0]
    assert world.reads() == []
    episodes = [(es.session, video, ep) for es, video, ep in built.items]
    with_record = write_package(
        episodes, world.policy, built.size, world.tmp / "a", IDS, None, tactile
    )
    without = write_package(episodes, world.policy, built.size, world.tmp / "b", IDS)
    a, b = json.loads(with_record.read_text()), json.loads(without.read_text())
    assert a["features"] == b["features"] and a["frame_arrays"] == b["frame_arrays"]
    assert set(np.load(a["episodes"][0]["npz"]).files) == set(
        np.load(b["episodes"][0]["npz"]).files
    )


def test_two_streams_for_one_hand_are_not_guessed(world: World) -> None:
    """같은 손의 장갑 스트림이 둘이면 어느 것이 그 손의 값인지 정하지 않고 싣지 않는다 (기록·경고).

    세션에 오른손 장갑 스트림이 둘(둘 다 동기화됨) 있다. 정답: 오른손 상태 multiple_streams, 실은 손
    없음, 원본을 읽지 않는다.
    """
    es = world.session("s1", [right_glove()])
    second = es.session.stream("glove_right").model_copy(update={"stream_id": "glove_right_b"})
    doubled = es.session.model_copy(update={"streams": (*es.session.streams, second)})
    built = world.build(replace(es, session=doubled))
    assert built.tactile is not None and built.tactile.hands == {}
    assert built.tactile.sessions[0]["hands"]["right"] == {"status": "multiple_streams"}
    assert len(built.warnings) == 1 and "2개라" in built.warnings[0]
    assert world.reads() == []


def test_sessions_without_a_glove_are_filled_as_missing(world: World) -> None:
    """장갑을 실은 세션과 없는 세션이 섞이면, 없는 세션의 촉각 열은 전부 없음(값 0, 표시 0)이다.

    s1은 오른손 장갑, s2는 장갑이 없다. 열 모양은 s1이 정하고 s2는 채워 넣지 않는다.
    """
    built = world.build(world.session("s1", [right_glove()]), world.session("s2", model=None))
    tactile = built.tactile
    assert tactile is not None and list(tactile.hands) == [Hand.RIGHT]
    filled = tactile.lerobot_arrays(1)["tactile_right"]
    assert filled.shape == (len(built.items[1][2].times_ms), 6) and not filled.any()
    assert tactile.lerobot_arrays(0)["tactile_right"][:, -1].any()
    assert [s["hands"]["right"]["status"] for s in tactile.sessions] == ["exported", "no_glove"]
    assert tactile.meta("lerobot")["glove_models"] == ["syn-glove-a"]


def test_sessions_with_different_channel_layouts_are_rejected(world: World) -> None:
    """같은 손의 채널 구성이 세션마다 다르면(장갑 기종이 다름) 이유와 함께 거부한다.

    s1 오른손은 pressure_0~4, s2 오른손은 pressure_0~4 + temp_0이다. 채널을 0으로 채우거나 버려서
    맞추지 않는다. 정책이 공통 접두사(pressure)만 고르면 두 세션의 구성이 같아져 내보낸다.
    """
    wide = (*(f"pressure_{i}" for i in range(5)), "temp_0")
    s1 = world.session("s1", [right_glove()])
    s2 = world.session("s2", [right_glove(seed=3, channels=wide)], model="syn-glove-b")
    with pytest.raises(ExportError, match=r"s2: right 장갑의 채널 구성이 s1와 다릅니다.*temp_0"):
        world.build(s1, s2)
    built = world.build(s1, s2, policy=with_tactile(world.policy, channel_prefixes=("pressure",)))
    assert built.tactile is not None
    assert built.tactile.hands == {Hand.RIGHT: tuple(f"pressure_{i}" for i in range(5))}
    # 채널 이름이 같아도 장갑 기종이 섞이면 값의 척도가 다를 수 있다고 알린다
    assert built.tactile.meta("gr00t")["glove_models"] == ["syn-glove-a", "syn-glove-b"]
    assert any("장갑 기종이 섞여" in w for w in built.warnings)


def test_export_without_gloves_is_unchanged_and_reads_no_raw(world: World) -> None:
    """장갑 스트림이 없는 세션만 내보내면 촉각 기록이 없고 묶음이 촉각 기능이 없던 때와 같다.

    정답: `tactile`은 None, 원본 접근 기록 없음, package.json의 특징·배열 대응과 npz 배열이
    원본 저장소를 넘기지 않았을 때와 같다.
    """
    es = world.session("s1")
    built = world.build(es)
    plain = world.build(es, raw=False)
    assert built.tactile is None and plain.tactile is None and world.reads() == []
    assert built.sessions == plain.sessions and built.warnings == plain.warnings == []
    specs: list[dict[str, Any]] = []
    for name, b in (("a", built), ("b", plain)):
        episodes = [(x.session, video, ep) for x, video, ep in b.items]
        pkg = write_package(episodes, world.policy, b.size, world.tmp / name, IDS, None, b.tactile)
        specs.append(json.loads(pkg.read_text("utf-8")))
    assert specs[0]["features"] == specs[1]["features"]
    assert specs[0]["frame_arrays"] == specs[1]["frame_arrays"] == FRAME_ARRAYS
    a, b = (np.load(s["episodes"][0]["npz"]) for s in specs)
    assert set(a.files) == set(b.files) and all(np.array_equal(a[k], b[k]) for k in a.files)


def test_tactile_can_be_turned_off_by_policy(world: World) -> None:
    """`tactile.enabled: false`면 동기화된 장갑이 있어도 싣지 않고 원본 버킷도 읽지 않는다."""
    built = world.build(
        world.session("s1", [right_glove(), left_glove()]),
        policy=with_tactile(world.policy, enabled=False),
    )
    assert built.tactile is None and built.warnings == [] and world.reads() == []


def test_unreadable_glove_data_stops_the_export_with_the_reason(world: World) -> None:
    """실어야 할 장갑을 읽을 수 없으면 조용히 빼지 않고 이유를 알리며 멈춘다.

    (1) 원본 감사 저장소를 받지 못함, (2) 장갑 Parquet이 원본 버킷에 없음, (3) 스트림 위치가 그 원본
    저장소가 아님, (4) 정책 접두사에 맞는 채널이 없음, (5) 채널 이름이 유효 표시 이름과 겹침.
    """
    es = world.session("s1", [right_glove()])
    with pytest.raises(ExportError, match=r"s1/glove_right: .*원본 감사 저장소가 없습니다"):
        world.build(es, raw=False)
    (world.raw_local.root / glove_key("s1", "glove_right")).unlink()
    with pytest.raises(ExportError, match=r"s1/glove_right: 장갑 정규화 Parquet이 원본 버킷에"):
        world.build(es)
    assert world.reads() == []  # 없는 객체는 읽기를 시도하지 않는다 (존재 확인만)
    elsewhere = world.session("s2", [right_glove()])
    stream = elsewhere.session.stream("glove_right").model_copy(
        update={"uri": "s3://other-bucket/sessions/s2/derived/glove_right.parquet"}
    )
    moved = elsewhere.session.model_copy(
        update={"streams": (elsewhere.session.reference_stream, stream)}
    )
    with pytest.raises(ExportError, match=r"s2/glove_right: 장갑 스트림이 원본 저장소"):
        world.build(replace(elsewhere, session=moved))
    es3 = world.session("s3", [right_glove()])
    with pytest.raises(ExportError, match=r"s3/glove_right: 접두사 \['force'\].*pressure_0"):
        world.build(es3, policy=with_tactile(world.policy, channel_prefixes=("force",)))
    es4 = world.session("s4", [right_glove(channels=("pressure_0", "present"))])
    with pytest.raises(ExportError, match=r"s4/glove_right: 장갑 채널 이름 'present'"):
        world.build(es4)


def test_channel_selection_is_ordered_by_name_and_policy_rejects_bad_prefixes(
    policy: ExportPolicy,
) -> None:
    """채널은 파일 열 순서가 아니라 이름의 자연 정렬 순이고, 접두사 선택은 그중 일부만 남긴다.

    빈 접두사·같은 접두사 두 번·0 이하 보간 한도는 정책 검증에서 거부한다.
    """
    names = ("pressure_10", "temp_0", "pressure_2", "pressure_1")
    assert select_channels(names, ()) == ("pressure_1", "pressure_2", "pressure_10", "temp_0")
    assert select_channels(names, ("pressure",)) == ("pressure_1", "pressure_2", "pressure_10")
    assert select_channels(names, ("force",)) == ()
    data = policy.tactile.model_dump()
    for bad in ({"channel_prefixes": [""]}, {"channel_prefixes": ["p", "p"]},
                {"interp_max_gap_ms": 0}, {"min_coverage": 1.5}):  # fmt: skip
        with pytest.raises(ValueError, match=next(iter(bad))):
            TactilePolicy.model_validate({**data, **bad})


def _hashes(root: Path) -> dict[str, str]:
    """디렉터리 아래 파일 → sha256."""
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _gr00t(world: World, built: BuiltEpisodes, name: str) -> Path:
    """`write_gr00t`로 world.tmp/<name>에 쓴다."""
    dest = world.tmp / name
    write_gr00t(
        built.items, built.sessions, world.policy, world.vocab, built.size, dest, built.tactile
    )
    return dest


def _episode_of(dest: Path, sid: str) -> int:
    """세션(내부 ID)의 GR00T 에피소드 번호 (meta/dlp_episodes.json의 가명으로 찾는다)."""
    rows = json.loads((dest / "meta" / "dlp_episodes.json").read_text("utf-8"))
    return next(r["episode_index"] for r in rows if r["session_id"] == IDS.session(sid))


@needs_ffmpeg
def test_gr00t_dataset_carries_tactile_as_state_ranges_and_passes_the_check(world: World) -> None:
    """GR00T: 촉각 값이 observation.state 끝의 `<손>_tactile` 구간에 실리고 확인기를 통과한다.

    세션 s1(검증 분할)은 두 손 장갑, s2(학습 분할)는 오른손 장갑이 동기화 전이다. GR00T는 에피소드를
    분할 순으로 다시 번호 매기므로 s2가 에피소드 0, s1이 1이다. 정답:
    - modality.json state = 손·팔 구간(0:42) 뒤에 left_tactile 42:45, right_tactile 45:50. action은
      손·팔 구간만(길이 42)이고 다음 프레임의 손·팔 state다.
    - s1 에피소드의 구간 값 = 정답 신호, observation.tactile_valid = 값이 있는 프레임만 1.
    - s2 에피소드는 두 구간 모두 값 0·표시 0이고, meta/dlp_tactile.json에 not_synced로 남는다.
    - 결과 파일 어디에도 원본 버킷 이름, 장갑 Parquet 키, 내부 세션 ID가 없다.
    """
    right, left = right_glove(), left_glove()
    s1 = world.session("s1", [right, left], split=Split.VAL)
    s2 = world.session("s2", [replace(right_glove(seed=5), synced=False)], model="syn-glove-a")
    built = world.build(s1, s2)
    dest = _gr00t(world, built, "gr00t")
    assert check_dataset(dest, world.policy.gr00t.video.ffprobe) == []
    base = len(StateLayout.from_policy(world.policy.lerobot).names)
    modality = json.loads((dest / "meta" / "modality.json").read_text("utf-8"))
    assert base == 42 and list(modality["state"])[-2:] == ["left_tactile", "right_tactile"]
    assert modality["state"]["left_tactile"] == {"start": 42, "end": 45}
    assert modality["state"]["right_tactile"] == {"start": 45, "end": 50}
    assert "left_tactile" not in modality["action"] and "right_tactile" not in modality["action"]
    assert max(r["end"] for r in modality["action"].values()) == 42
    info = json.loads((dest / "meta" / "info.json").read_text("utf-8"))["features"]
    assert info["observation.state"]["shape"] == [50] and info["action"]["shape"] == [42]
    assert info["observation.state"]["names"][42:] == [
        *(f"left.tactile.pressure_{i}" for i in range(3)),
        *(f"right.tactile.pressure_{i}" for i in range(5)),
    ]
    assert info["observation.tactile_valid"] == {
        "dtype": "float32", "shape": [2], "names": ["left.tactile", "right.tactile"],
    }  # fmt: skip
    assert info["observation.hand_valid"]["shape"] == [14]  # 기존 열은 그대로다
    e1, e2 = _episode_of(dest, "s1"), _episode_of(dest, "s2")
    assert (e1, e2) == (1, 0)
    gap = world.policy.tactile.interp_max_gap_ms
    t1 = pq.read_table(dest / f"data/chunk-000/episode_{e1:06d}.parquet").to_pydict()
    state, valid = np.array(t1["observation.state"]), np.array(t1["observation.tactile_valid"])
    frame_ms = built.items[0][2].times_ms  # items는 세션 ID 순이라 s1이 0번
    assert state.shape == (frame_ms.size, 50)
    assert_glove_values(state[:, 42:45], valid[:, 0], left.truth, frame_ms, gap)
    assert_glove_values(state[:, 45:50], valid[:, 1], right.truth, frame_ms, gap)
    action = np.array(t1["action"])
    assert action.shape == (frame_ms.size, 42) and np.array_equal(action[:-1], state[1:, :42])
    t2 = pq.read_table(dest / f"data/chunk-000/episode_{e2:06d}.parquet").to_pydict()
    assert not np.array(t2["observation.state"])[:, 42:].any()
    assert not np.array(t2["observation.tactile_valid"]).any()
    # meta/dlp_tactile.json: 채널·위치·장갑 기종·세션별 상태, 단위는 모른다고 적는다
    meta = json.loads((dest / "meta" / "dlp_tactile.json").read_text("utf-8"))
    assert built.tactile is not None and meta == built.tactile.meta("gr00t")
    assert meta["format"] == "dlp-tactile" and meta["format_version"] == "1.0"
    assert meta["units"].startswith("unknown") and meta["interp_max_gap_ms"] == gap
    assert meta["hands"]["right"] == {
        "channels": [f"pressure_{i}" for i in range(5)],
        "state_key": "right_tactile",
        "valid_column": "observation.tactile_valid",
        "valid_name": "right.tactile",
    }
    by_session = {s["session_id"]: s for s in meta["sessions"]}
    assert by_session[IDS.session("s2")]["hands"]["right"] == {"status": "not_synced"}
    assert by_session[IDS.session("s1")]["hands"]["left"]["status"] == "exported"
    assert meta["glove_models"] == ["syn-glove-a"]
    # 결과 파일 전체(바이트)에 원본 버킷 이름·장갑 Parquet 키·내부 세션 ID가 없다
    for path in (p for p in dest.rglob("*") if p.is_file()):
        data = path.read_bytes()
        for needle in (RAW_BUCKET, "sessions/s", "derived/", "glove_right", '"s1"', '"s2"'):
            assert needle.encode() not in data, (path.name, needle)
    # 읽은 원본은 동기화된 s1의 두 장갑뿐이다 (동기화 전인 s2 장갑은 읽지 않았다)
    assert sorted(k for _, _, k, _ in world.reads()) == [
        glove_key("s1", "glove_left"),
        glove_key("s1", "glove_right"),
    ]


@needs_ffmpeg
def test_gr00t_output_without_gloves_is_byte_identical(world: World) -> None:
    """장갑 없는 세션의 GR00T 결과는 원본 저장소를 넘겼든 아니든 바이트 단위로 같다.

    촉각 기능을 켠 정책(기본)에서도 장갑 스트림이 없으면 파일 하나 더하지 않는다는 뜻이다.
    """
    es = world.session("s1")
    a = _gr00t(world, world.build(es), "a")
    b = _gr00t(world, world.build(es, raw=False), "b")
    assert _hashes(a) == _hashes(b) and "meta/dlp_tactile.json" not in _hashes(a)
    # 동기화 전 장갑만 있으면 열은 그대로이고 세션 기록 파일만 더해진다
    unsynced = world.session("s2", [replace(right_glove(), synced=False)])
    c = _gr00t(world, world.build(unsynced), "c")
    modality = json.loads((c / "meta" / "modality.json").read_text("utf-8"))
    assert modality["state"] == modality["action"]
    assert set(_hashes(c)) - set(_hashes(a)) == {"meta/dlp_tactile.json"}
    assert check_dataset(c, world.policy.gr00t.video.ffprobe) == []


def _break_range(dest: Path) -> None:
    """modality.json의 오른손 촉각 구간을 한 칸 줄이고 왼손 구간을 그만큼 늘린다."""
    path = dest / "meta" / "modality.json"
    m = json.loads(path.read_text("utf-8"))
    m["state"]["left_tactile"]["end"] += 1
    m["state"]["right_tactile"]["start"] += 1
    path.write_text(json.dumps(m), "utf-8")


def _put_tactile_in_action(dest: Path) -> None:
    """modality.json action에 촉각 구간 키를 넣는다."""
    path = dest / "meta" / "modality.json"
    m = json.loads(path.read_text("utf-8"))
    m["action"]["right_tactile"] = m["state"]["right_tactile"]
    path.write_text(json.dumps(m), "utf-8")


def _clear_valid_flags(dest: Path) -> None:
    """유효 표시만 모두 0으로 바꾼다 (값은 남는다 = 없다고 표시한 행에 값이 있음)."""
    path = dest / "data/chunk-000/episode_000000.parquet"
    t = pq.read_table(path)
    i = t.column_names.index("observation.tactile_valid")
    zeros = [[0.0] * len(row) for row in t.column(i).to_pylist()]
    pq.write_table(t.set_column(i, t.field(i), pa.array(zeros, type=t.field(i).type)), path)


def _drop_valid_feature(dest: Path) -> None:
    """info.json에서 유효 표시 특징의 이름을 지운다."""
    path = dest / "meta" / "info.json"
    info = json.loads(path.read_text("utf-8"))
    info["features"]["observation.tactile_valid"]["names"] = ["left.tactile"]
    path.write_text(json.dumps(info), "utf-8")


@needs_ffmpeg
@pytest.mark.parametrize(
    ("corrupt", "expected"),
    [
        (_break_range, "길이가 채널 수"),
        (_put_tactile_in_action, "modality.json action에 있음"),
        (_clear_valid_flags, "없다고 표시한 행에 0이 아닌 값"),
        (_drop_valid_feature, "observation.tactile_valid에 right.tactile 없음"),
    ],
)
def test_gr00t_check_rejects_broken_tactile_ranges(
    world: World, corrupt: Any, expected: str
) -> None:
    """확인기는 촉각 구간이 meta/dlp_tactile.json과 어긋난 데이터셋을 문제로 알린다."""
    built = world.build(world.session("s1", [right_glove(), left_glove()]))
    dest = _gr00t(world, built, "gr00t")
    assert check_dataset(dest, world.policy.gr00t.video.ffprobe) == []
    corrupt(dest)
    problems = check_dataset(dest, world.policy.gr00t.video.ffprobe)
    assert any(expected in p for p in problems), problems


def test_robot_mode_package_keeps_the_human_glove_values(world: World) -> None:
    """로봇 모드(`--robot`) 묶음에도 같은 촉각 특징이 덧붙는다 (사람 장갑 값 그대로).

    `dlp_export.robot`의 로봇 특징(state = 로봇 관절, retarget.*) 위에 `observation.tactile.<손>`이
    더해지고 값은 사람 모드와 같다. 월드 좌표 손 라벨이 없어 목표는 없지만 묶음 구성은 같다.
    """
    right = right_glove()
    built = world.build(world.session("s1", [right]))
    es, video, ep = built.items[0]
    robot = load_robot(ROOT, "so101", env={})
    rob = retarget_session(
        ep, es.session.reference_stream, es.labels, robot, load_retarget_policy(ROOT),
        world.policy.lerobot, DlsIkSolver(),
    )  # fmt: skip
    override = robot_override([(ep, rob)], world.policy.lerobot, robot)
    pkg = write_package(
        [(es.session, video, ep)], world.policy, built.size, world.tmp / "pkg", IDS, override,
        built.tactile,
    )  # fmt: skip
    spec = json.loads(pkg.read_text("utf-8"))
    assert spec["robot_type"] == "so101"
    assert {"retarget.reachable", "observation.human_state", "observation.tactile.right"} <= set(
        spec["features"]
    )
    assert spec["frame_arrays"]["observation.tactile.right"] == "tactile_right"
    assert spec["frame_arrays"]["retarget.reachable"] == "reachable"
    npz = np.load(spec["episodes"][0]["npz"])
    assert npz["state"].shape == (ep.times_ms.size, len(robot.joint_names))
    assert built.tactile is not None
    assert np.array_equal(npz["tactile_right"], built.tactile.lerobot_arrays(0)["tactile_right"])


def test_run_export_refuses_a_raw_store_for_another_bucket(
    world: World, ontology: Ontology
) -> None:
    """`run_export`는 원본 버킷이 아닌 저장소를 장갑 읽기용으로 받지 않는다 (DB를 열기 전에 거부).

    결과의 원본 위치 검사(`assert_no_raw`)가 `raw_bucket` 이름으로 찾으므로, 다른 버킷의 저장소를
    받으면 그 위치가 새도 잡지 못한다.
    """
    other = AuditedStore(
        LocalStore(world.tmp / "elsewhere", "some-other-bucket"), MemorySink(),
        actor="svc-pipeline", purpose="export.gr00t",
    )  # fmt: skip
    with pytest.raises(ExportError, match=r"some-other-bucket.*원본 버킷이 아닙니다"):
        run_export(
            sa.create_engine("sqlite://"), root=ROOT, version_id="dv1", fmt="gr00t",
            target="internal", snapshots=LocalSnapshotStore(world.tmp / "snap"),
            labeling=world.labeling, datasets=LocalStore(world.tmp / "store", "dlp-datasets"),
            raw_bucket=RAW_BUCKET, policy=world.policy, ontology=ontology,
            include_unreviewed=False, splits=None, now=FIXED_TIME, raw=other,
        )  # fmt: skip


def test_meta_names_columns_per_format(world: World) -> None:
    """meta/dlp_tactile.json은 형식마다 값이 실린 위치를 그 형식의 이름으로 적는다."""
    built = world.build(world.session("s1", [right_glove()]))
    tactile = built.tactile
    assert isinstance(tactile, TactileData)
    lerobot = tactile.meta("lerobot")["hands"]["right"]
    assert lerobot == {
        "channels": [f"pressure_{i}" for i in range(5)],
        "feature": "observation.tactile.right",
        "present": "right.tactile.present",
    }
    both = (tactile.meta("lerobot"), tactile.meta("gr00t"))
    assert all(m["time_alignment"] and m["missing"] and m["source"] for m in both)
    text = json.dumps(both)
    assert RAW_BUCKET not in text and "derived" not in text and '"s1"' not in text


@pytest.mark.isolated_env
def test_official_lerobot_loader_reads_tactile_features(world: World) -> None:
    """격리 환경의 LeRobot 공식 쓰기 API가 촉각 특징을 받고 공식 로더가 같은 값을 돌려준다.

    격리 환경(PyTorch CPU판)을 받으므로 `make test-isolated`에서만 돈다. 정답: 로더의 특징 목록에
    observation.tactile.left·right가 있고, 표본 프레임의 값이 묶음에 넣은 배열과 같다.
    """
    lp = world.policy.lerobot
    built = world.build(world.session("s1", [right_glove(), left_glove()]))
    assert built.tactile is not None
    episodes = [(es.session, video, ep) for es, video, ep in built.items]
    pkg = write_package(
        episodes, world.policy, built.size, world.tmp / "pkg", IDS, None, built.tactile
    )
    dest = world.tmp / "lerobot"
    run_script(ROOT, lp, "lerobot_write.py", str(pkg), str(dest))
    check = json.loads(run_script(ROOT, lp, "lerobot_check.py", str(dest), lp.repo_id))
    n = len(built.items[0][2].times_ms)
    assert check["episodes"] == 1 and check["frames"] == n
    assert {"observation.tactile.left", "observation.tactile.right"} <= set(check["features"])
    arrays = built.tactile.lerobot_arrays(0)
    for s in check["samples"]:
        k = s["index"]
        assert np.allclose(s["extra"]["observation.tactile.right"], arrays["tactile_right"][k])
        assert np.allclose(s["extra"]["observation.tactile.left"], arrays["tactile_left"][k])
