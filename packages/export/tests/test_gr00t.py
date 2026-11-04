"""GR00T LeRobot v2.1 내보내기 (ADR 0065, DB 없이).

정답은 conftest 합성 시나리오(VFR 블러본·라벨)에 손 3D 궤적·설명을 더한 것이다. 쓰기
(`write_gr00t`)와 확인(`check_dataset`)은 따로 짠 코드라, 확인 통과에 더해 열·값·영상을 시험이 다시
직접 읽어 정답과 맞춘다.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dlp_export.gr00t import StateLayout, encode_video, write_gr00t
from dlp_export.gr00t_check import check_dataset
from dlp_export.lerobot import Vocab, build_episode
from dlp_export.policy import ExportPolicy
from dlp_export.pseudonym import Pseudonymizer
from dlp_export.runner import build_episodes
from dlp_export.source import ExportSession, ExportSource, label_states, select_labels
from dlp_fixtures.video import write_video
from dlp_media.pts import build_pts_index
from dlp_schema.dataset import DatasetVersion, Split
from dlp_schema.labels import LabelRecord
from dlp_schema.ontology import Ontology
from dlp_schema.testing import FIXED_TIME, make_label

from .conftest import Scenario, human

# pyarrow에는 타입 스텁이 없다 (dlp_export.gr00t와 같다)
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

WRIST_X = {ms: ms / 1000 for ms in range(100, 700, 100)}  # 오른손 손목 3D x (m), y 0.1, z 0.5
TASK_EN = "Sweep the floor with a mop"
ACTION_EN = "Carry the mop across the room"


def extra_labels(sid: str) -> list[LabelRecord]:
    """시나리오에 더하는 라벨: 오른손 손목 3D 궤적, 작업·행동 설명(ko 원문 + en 번역), 낡은 번역.

    낡은 번역(s1-stale)은 원문 s1-gone이 내보내기 라벨에 없으므로 쓰면 안 된다.
    """
    wrist = {
        "kind": "trajectory3d", "entity_id": "right_hand", "part": "wrist", "frame": "camera",
        "source_3d": "mono_depth",
        "samples": [{"t_ms": ms, "x": x, "y": 0.1, "z": 0.5} for ms, x in WRIST_X.items()],
    }  # fmt: skip

    def desc(label_id: str, segment: str, text: str, **kw: Any) -> LabelRecord:
        return make_label(
            {"kind": "description", "segment_id": segment, "text": text, **kw},
            label_id=label_id, session_id=sid, t_start_ms=0, t_end_ms=700, **human(),
        )  # fmt: skip

    return [
        make_label(wrist, label_id=f"{sid}-wrist", stream_id="bodycam", t_start_ms=100,
                   t_end_ms=600, session_id=sid, **human()),
        desc(f"{sid}-tko", "task1", "대걸레로 바닥을 쓴다"),
        desc(f"{sid}-ten", "task1", TASK_EN, language="en", source_label_id=f"{sid}-tko"),
        desc(f"{sid}-ako", "a1", "대걸레를 옮긴다"),
        desc(f"{sid}-aen", "a1", ACTION_EN, language="en", source_label_id=f"{sid}-ako"),
        desc(f"{sid}-stale", "a1", "A stale translation", language="en",
             source_label_id=f"{sid}-gone"),
    ]  # fmt: skip


def source(sc: Scenario, policy: ExportPolicy) -> ExportSource:
    """시나리오 + `extra_labels` → `ExportSource` (기본 검증 정책, 학습 분할)."""
    states = label_states(policy, False)
    version = DatasetVersion(
        version_id="dv1", ontology_version="1.0.0", created_at=FIXED_TIME,
        snapshot_uri="local-snapshot://x/dv1", splits={"s1": Split.TRAIN},
    )  # fmt: skip
    labels = select_labels([*sc.labels, *extra_labels("s1")], policy, states, ())
    return ExportSource(
        version, [ExportSession(sc.session, Split.TRAIN, labels, sc.render_hashes)], states
    )


@pytest.fixture
def exported(
    scenario: Scenario, policy: ExportPolicy, ontology: Ontology, tmp_path: Path
) -> tuple[Path, Any]:
    """GR00T 데이터셋을 tmp_path/gr00t에 쓴다. (경로, 에피소드)."""
    vocab = Vocab.from_ontology(ontology)
    ids = Pseudonymizer.for_export("e1", b"secret", enabled=True)
    built = build_episodes(
        source(scenario, policy), policy, vocab, scenario.labeling, tmp_path, ids
    )
    dest = tmp_path / "gr00t"
    write_gr00t(built.items, built.sessions, policy, vocab, built.size, dest)
    return dest, built.items[0][2]


def read(dest: Path, name: str) -> Any:
    """meta 파일 (json이면 객체, jsonl이면 목록)."""
    text = (dest / "meta" / name).read_text("utf-8")
    if name.endswith(".jsonl"):
        return [json.loads(line) for line in text.splitlines()]
    return json.loads(text)


def test_gr00t_export_passes_nvidia_layout_check(
    exported: tuple[Path, Any], policy: ExportPolicy
) -> None:
    """쓴 결과가 확인기를 통과하고, 배치·열·dtype·modality 구간이 GR00T 요구와 같다.

    정답: state = 손 6점 x 3 x 2손 + 팔꿈치 3 x 2 = 42, 구간이 빈틈없이 42를 덮는다. annotation 열은
    두 언어 주석뿐이고 LeRobot 형식의 annotation.* 값은 labels.*로 옮겨졌다.
    """
    dest, ep = exported
    assert check_dataset(dest, policy.gr00t.video.ffprobe) == []
    info = read(dest, "info.json")
    assert info["codebase_version"] == "v2.1" and info["fps"] == policy.lerobot.fps
    assert info["total_episodes"] == 1 and info["total_frames"] == len(ep.times_ms)
    assert info["splits"] == {"train": "0:1"}
    modality = read(dest, "modality.json")
    n = len(StateLayout.from_policy(policy.lerobot).names)
    assert n == 2 * 6 * 3 + 2 * 3
    assert modality["state"] == modality["action"]
    assert max(r["end"] for r in modality["state"].values()) == n
    assert modality["video"] == {"ego_view": {"original_key": "observation.images.bodycam"}}
    assert set(modality["annotation"]) == {
        "human.task_description",
        "human.action.task_description",
    }
    t = pq.read_table(dest / "data/chunk-000/episode_000000.parquet")
    assert t.schema.field("observation.state").type == pa.list_(pa.float32())
    assert t.schema.field("action").type == pa.list_(pa.float32())
    assert t.schema.field("timestamp").type == pa.float32()
    assert t.schema.field("next.done").type == pa.bool_()
    assert [c for c in t.column_names if c.startswith("annotation")] == [
        "annotation.human.task_description",
        "annotation.human.action.task_description",
    ]
    assert {"labels.hand_state", "labels.verb", "labels.verification", "labels.dlp_state"} <= set(
        t.column_names
    )
    # 원본 위치·검수자 ID·내부 세션 ID가 결과에 없다 (메타는 가명)
    text = "".join(p.read_text("utf-8") for p in (dest / "meta").iterdir())
    assert "reviewer-7" not in text and '"s1"' not in text and "dlp-raw" not in text


def test_gr00t_state_action_and_language_follow_labels(
    exported: tuple[Path, Any], policy: ExportPolicy
) -> None:
    """state는 카메라 좌표 손 3D(m), 값이 없으면 0과 hand_valid 0, action은 다음 프레임 state.

    언어: 작업 구간(0~500 ms) 안은 작업 설명의 영어 번역, 밖은 도메인 이름(Cleaning). 행동
    (200~700 ms)은 영어 번역(낡은 번역은 버림), 행동 밖은 idle 문장.
    """
    dest, ep = exported
    t = pq.read_table(dest / "data/chunk-000/episode_000000.parquet").to_pydict()
    names = read(dest, "info.json")["features"]["observation.state"]["names"]
    valid_names = read(dest, "info.json")["features"]["observation.hand_valid"]["names"]
    state, valid = np.array(t["observation.state"]), np.array(t["observation.hand_valid"])
    tasks = [r["task"] for r in read(dest, "tasks.jsonl")]
    k = int(np.argmin(np.abs(ep.times_ms - 400)))
    x = names.index("right.wrist.x3d")
    assert state[k, x] == pytest.approx(ep.times_ms[k] / 1000, abs=1e-4)
    assert state[k, names.index("right.wrist.z3d")] == pytest.approx(0.5)
    assert valid[k, valid_names.index("right.wrist")] == 1.0
    assert valid[k, valid_names.index("left.wrist")] == 0.0
    assert state[k, names.index("left.wrist.x3d")] == 0.0
    late = int(np.argmin(np.abs(ep.times_ms - 900)))
    assert valid[late, valid_names.index("right.wrist")] == 0.0  # 궤적 밖
    assert np.array_equal(np.array(t["action"])[:-1], state[1:])
    task = [tasks[i] for i in t["annotation.human.task_description"]]
    action = [tasks[i] for i in t["annotation.human.action.task_description"]]
    assert t["task_index"] == t["annotation.human.task_description"]
    assert task[k] == TASK_EN and task[late] == "Cleaning"
    assert action[k] == ACTION_EN and action[late] == policy.gr00t.idle_action_text
    assert "A stale translation" not in tasks
    assert read(dest, "episodes.jsonl") == [
        {"episode_index": 0, "tasks": [TASK_EN, "Cleaning"], "length": len(ep.times_ms)}
    ]


def gray(i: int) -> int:
    """i번째 블러본 프레임의 밝기 (이웃 프레임끼리 37씩 달라 한 프레임만 어긋나도 드러난다)."""
    return 28 + (i * 37) % 200


def test_gr00t_video_frame_k_is_the_pts_selected_blurred_frame(
    scenario: Scenario, policy: ExportPolicy, ontology: Ontology, tmp_path: Path
) -> None:
    """mp4 프레임 수 = 행 수, 코덱 h264, 프레임 k = PTS로 고른 블러본 프레임 frame_index[k].

    정답 근거: VFR 블러본 i번째 프레임을 gray(i) 단색으로 만들어, 다시 인코딩한 영상의 k번째 프레임
    밝기가 gray(frame_index[k])인지 본다 (손실 압축 허용 오차 안).
    """
    video = tmp_path / "v.mp4"
    write_video(
        video,
        ((t, np.full((48, 64, 3), gray(i), np.uint8)) for i, t in enumerate(scenario.times)),
        width=64,
        height=48,
    )
    labels = select_labels(scenario.labels, policy, label_states(policy, False), ())
    ep = build_episode(
        scenario.session, scenario.session.streams[0], build_pts_index(video), (64, 48), labels,
        Vocab.from_ontology(ontology), policy.lerobot,
    )  # fmt: skip
    assert len(set(ep.frame_index.tolist())) < len(ep.frame_index)  # VFR: 반복되는 프레임이 있다
    out = tmp_path / "out.mp4"
    encode_video(video, ep.frame_index, (64, 48), policy.lerobot.fps, policy.gr00t.video, out)
    with av.open(str(out)) as c:
        assert c.streams.video[0].codec_context.name == "h264"
        means = [float(f.to_ndarray(format="rgb24").mean()) for f in c.decode(c.streams.video[0])]
    assert len(means) == len(ep.frame_index)
    for k, i in enumerate(ep.frame_index.tolist()):
        assert means[k] == pytest.approx(gray(i), abs=6)


def corrupt_annotation_column(dest: Path) -> None:
    """GR00T 규칙 위반: 언어 주석이 아닌 annotation.* 열을 더한다."""
    path = dest / "data/chunk-000/episode_000000.parquet"
    t = pq.read_table(path)
    pq.write_table(t.append_column("annotation.verb", pa.array([0] * t.num_rows)), path)


def corrupt_modality_gap(dest: Path) -> None:
    """modality.json state 구간에 빈틈을 낸다 (첫 키의 end를 하나 줄인다)."""
    path = dest / "meta" / "modality.json"
    m = json.loads(path.read_text())
    first = next(iter(m["state"]))
    m["state"][first]["end"] -= 1
    path.write_text(json.dumps(m))


def corrupt_codec(dest: Path) -> None:
    """영상을 mpeg4로 다시 인코딩한다 (h264가 아님)."""
    path = dest / "videos/chunk-000/observation.images.bodycam/episode_000000.mp4"
    tmp = path.with_name("x.mp4")
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-i", str(path), "-c:v", "mpeg4", str(tmp)],
        check=True,
    )
    tmp.replace(path)


def corrupt_task_index(dest: Path) -> None:
    """tasks.jsonl 마지막 줄을 지워 주석 번호가 범위를 벗어나게 한다."""
    path = dest / "meta" / "tasks.jsonl"
    path.write_text("".join(path.read_text().splitlines(keepends=True)[:-1]))


@pytest.mark.parametrize(
    ("corrupt", "expected"),
    [
        (corrupt_annotation_column, "annotation.verb"),
        (corrupt_modality_gap, "modality.json state"),
        (corrupt_codec, "codec"),
        (corrupt_task_index, "tasks.jsonl 번호 밖"),
    ],
)
def test_gr00t_check_rejects_spec_violations(
    exported: tuple[Path, Any], policy: ExportPolicy, tmp_path: Path, corrupt: Any, expected: str
) -> None:
    """확인기는 GR00T 요구를 어긴 데이터셋을 문제로 알린다 (통과만 보는 확인기가 아님을 보인다)."""
    dest = tmp_path / "bad"
    shutil.copytree(exported[0], dest)
    corrupt(dest)
    problems = check_dataset(dest, policy.gr00t.video.ffprobe)
    assert any(expected in p for p in problems), problems
