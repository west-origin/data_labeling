"""NVIDIA Isaac GR00T N1.7 미세조정 형식 (ADR 0065).

"GR00T LeRobot" = LeRobot v2.1 배치 + meta/modality.json. 기준은 github.com/NVIDIA/Isaac-GR00T
`getting_started/data_preparation.md`와 그 로더(`gr00t/data/dataset/lerobot_episode_loader.py`)다.
LeRobot 형식(`dlp_export.lerobot`)과 같은 `build_episode` 결과를 다른 배치로 쓴다. PyTorch·LeRobot
없이 pyarrow·ffmpeg로 직접 쓴다.

배치 (`dest` 아래):
- meta/info.json(codebase_version v2.1, fps, chunks_size, data_path·video_path 틀, features),
  meta/episodes.jsonl, meta/tasks.jsonl, meta/modality.json. stats.json·relative_stats.json은
  GR00T가 `gr00t/data/stats.py`로 계산한다.
- data/chunk-XXX/episode_XXXXXX.parquet: 행 하나 = 고정 fps 프레임 하나.
- videos/chunk-XXX/observation.images.<스트림>/episode_XXXXXX.mp4: 행마다 고른 블러본 프레임을 고정
  fps H.264(yuv420p)로 다시 인코딩한 것. 영상 프레임 k = parquet 행 k (GR00T 로더는 행 번호로
  프레임을 디코딩한다).
- meta/dlp_vocab.json(정수 열 어휘), meta/dlp_episodes.json(에피소드 → 세션 가명·분할),
  meta/dlp_gr00t.json(좌표계·단위·action 정의). GR00T는 이 파일들을 읽지 않는다.

parquet 열 (사람 1인칭 시연이라 로봇 고유 감각이 없어 손·팔 3D 점을 상태로 쓴다):
- observation.state (float32): `StateLayout` 순서의 손 3D 점(손마다 lerobot.hand_points_3d)과 착용자
  팔 3D 점(손마다 lerobot.arm_points_3d). 카메라 좌표 m의 x, y, z. 값이 없으면 0이다.
- observation.hand_valid (float32): 상태의 점마다 값이 있으면 1, 없으면 0. GR00T는 이 열을 읽지
  않는다 (annotation 접두어가 아니라 규칙에 걸리지 않는다).
- action (float32): 다음 프레임의 observation.state (절대값, 마지막 프레임은 자기 자신). GR00T는
  상대 행동을 처리기에서 만들므로 데이터에는 절대값을 둔다 (data_config.md "RELATIVE").
- timestamp(float32, 초 = k/fps), frame_index, episode_index, index(전체 일련번호), task_index.
- annotation.human.task_description (int64): 그 프레임 작업 문장의 tasks.jsonl 번호 (= task_index).
- annotation.human.action.task_description (int64): 그 프레임 행동 문장의 tasks.jsonl 번호. 덮는
  행동이 없으면 gr00t.idle_action_text 문장이다 (GR00T는 -1 같은 "없음" 번호를 받지 않는다).
- next.reward(float32, 0), next.done(bool, 마지막 프레임만 참).
- labels.*: LeRobot 형식 annotation.* 열과 같은 값(손 상태·동사·검증 등급 등)과 labels.dlp_state
  (LeRobot 형식 observation.state 전체, 2D 관절 포함). GR00T는 annotation 접두어를 언어 주석에만
  허락하므로 이름만 바꿨다.

언어 문장(`gr00t.language`): 작업은 그 시각 작업 구간의 설명 라벨(같은 언어 원문 또는 현재 원문의
번역), 없으면 온톨로지 작업 이름, 작업 구간 밖은 도메인 이름이다. 행동은 손마다 그 시각 행동의 설명
라벨, 없으면 동사 이름이다. 두 손 문장이 다르면 "left hand: …; right hand: …"로 잇는다.

규칙(CLAUDE.md): 데이터셋 버전·블러본만, 가명 세션 ID, 블러 라벨·원본 위치·검수자 ID 없음, 기본은
검수한 라벨만. 입력 `ExportSource`·`build_episodes`가 이미 적용한다.
"""

# pyarrow에는 타입 스텁이 없어 이 모듈에서만 알 수 없는 타입 경고를 끈다 (dlp_media.tables와 같다).
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportUnknownParameterType=false

from __future__ import annotations

import json
import math
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray

from dlp_export.lerobot import (
    FRAME_ARRAYS,
    HANDS,
    Episode,
    Vocab,
    covering,
    features,
    preferred,
    state_names,
    video_key,
    video_size,
    wearer_arm_names,
)
from dlp_export.policy import ExportPolicy, Gr00tPolicy, Gr00tVideo, LeRobotPolicy
from dlp_export.source import ExportError, ExportSession
from dlp_schema.dataset import Split
from dlp_schema.labels import (
    ActionPayload,
    DescriptionPayload,
    LabelRecord,
    SegmentLevel,
    SegmentPayload,
)

CODEBASE_VERSION = "v2.1"
# modality.json annotation 키 (parquet 열은 앞에 "annotation."을 붙인다)
TASK_KEY = "human.task_description"
ACTION_KEY = "human.action.task_description"
# LeRobot 형식 annotation.* 열을 옮겨 싣는 접두어 (GR00T는 annotation 접두어를 언어 주석에만
# 허락한다)
LABEL_PREFIX = "labels."
DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
_AXES = ("x3d", "y3d", "z3d")


@dataclass(frozen=True)
class StateLayout:
    """observation.state·action 배치.

    modality 키마다 LeRobot 특징 이름(state_names·wearer_arm_names) 목록.
    """

    groups: tuple[tuple[str, tuple[str, ...]], ...]
    valid: tuple[str, ...]  # observation.hand_valid 열 = 점마다 LeRobot "… .present" 이름

    @classmethod
    def from_policy(cls, lp: LeRobotPolicy) -> StateLayout:
        """손마다 손 3D 점, 그다음 손마다 팔 3D 점 (팔 점이 없으면 팔 묶음도 없다)."""
        hands = [(f"{h.value}_hand", f"{h.value}.", lp.hand_points_3d) for h in HANDS]
        arms = [(f"{h.value}_arm", f"{h.value}.arm.", lp.arm_points_3d) for h in HANDS]
        blocks = [b for b in (*hands, *arms) if b[2]]
        groups = tuple(
            (key, tuple(f"{pre}{p}.{a}" for p in parts for a in _AXES))
            for key, pre, parts in blocks
        )
        valid = tuple(f"{pre}{p}.present" for _, pre, parts in blocks for p in parts)
        return cls(groups, valid)

    @property
    def names(self) -> list[str]:
        """observation.state 열 이름 (예: right.wrist.x3d)."""
        return [n for _, ns in self.groups for n in ns]

    def modality(self) -> dict[str, dict[str, int]]:
        """modality.json state·action 절 ({키: {start, end}}, 파이썬 자르기 규약)."""
        out: dict[str, dict[str, int]] = {}
        start = 0
        for key, ns in self.groups:
            out[key] = {"start": start, "end": start + len(ns)}
            start += len(ns)
        return out

    def gather(
        self, ep: Episode, lp: LeRobotPolicy
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """에피소드 → (상태 (n, len(names)), 값 있음 (n, len(valid))). 값이 없는 칸은 0이다."""
        full = np.hstack([ep.state, ep.wearer_arm]).astype(np.float32)
        col = {n: i for i, n in enumerate([*state_names(lp), *wearer_arm_names(lp)])}
        state = full[:, [col[n] for n in self.names]]
        valid = full[:, [col[n] for n in self.valid]]
        return np.ascontiguousarray(state), np.ascontiguousarray(valid)


@dataclass
class TaskTable:
    """tasks.jsonl: 문장 → 번호 (처음 나온 순서).

    작업·행동 주석이 같은 표를 쓴다 (GR00T 로더가 tasks.jsonl 하나만 읽는다).
    """

    texts: list[str] = field(default_factory=list[str])
    _index: dict[str, int] = field(default_factory=dict[str, int])

    def index(self, text: str) -> int:
        """문장의 번호 (처음이면 새로 붙인다)."""
        if text not in self._index:
            self._index[text] = len(self.texts)
            self.texts.append(text)
        return self._index[text]


def descriptions(labels: list[LabelRecord], language: str) -> dict[str, LabelRecord]:
    """구간·행동 ID(`segment_id`) → 그 언어의 설명 라벨 (`preferred` 순 첫 것).

    번역은 원문(`source_label_id`)이 같은 내보내기 라벨에 있을 때만 쓴다. 원문이 고쳐졌거나(낡은
    번역) 검증 정책으로 빠졌으면 번역도 쓰지 않는다 (ADR 0043).
    """
    descs = [x for x in preferred(labels) if isinstance(x.payload, DescriptionPayload)]
    present = {x.label_id for x in descs}
    out: dict[str, LabelRecord] = {}
    for x in descs:
        p = x.payload
        assert isinstance(p, DescriptionPayload)
        if p.language != language or (p.source_label_id not in (None, *present)):
            continue
        out.setdefault(p.segment_id, x)
    return out


def _term(vocab: Vocab, kind: str, ids: tuple[str, ...], term_id: str, language: str) -> str:
    """어휘 항목의 표시 이름 (어휘 밖이면 ID 그대로)."""
    if term_id not in ids:
        return term_id
    return str(vocab.names[kind][ids.index(term_id)].model_dump()[language])


@dataclass
class EpisodeText:
    """에피소드 프레임마다의 작업·행동 문장과 문장에 쓴 설명 라벨 ID."""

    task: list[str]
    action: list[str]
    used: set[str]


def episode_text(
    labels: list[LabelRecord], ep: Episode, vocab: Vocab, gp: Gr00tPolicy
) -> EpisodeText:
    """프레임마다 작업 문장(설명 → 작업 이름 → 도메인 이름)과 행동 문장(손마다 설명 → 동사 이름)."""
    descs = descriptions(labels, gp.language)
    ordered = preferred(labels)
    tasks = [
        x for x in ordered
        if isinstance(x.payload, SegmentPayload) and x.payload.level is SegmentLevel.TASK
    ]  # fmt: skip
    actions = [x for x in ordered if isinstance(x.payload, ActionPayload)]
    out = EpisodeText([], [], set())
    for k, t in enumerate(ep.times_ms):
        cover = covering(tasks, float(t))
        seg = cover[0].payload if cover else None
        d = descs.get(seg.segment_id) if isinstance(seg, SegmentPayload) else None
        if d is not None:
            out.used.add(d.label_id)
        text = (
            d.payload.text if d is not None and isinstance(d.payload, DescriptionPayload) else None
        )
        out.task.append(text or _term(vocab, "task", vocab.tasks, ep.tasks[k], gp.language))
        out.action.append(_action_text(covering(actions, float(t)), descs, vocab, gp, out.used))
    return out


def _action_text(
    acts: list[LabelRecord],
    descs: dict[str, LabelRecord],
    vocab: Vocab,
    gp: Gr00tPolicy,
    used: set[str],
) -> str:
    """한 시각의 행동 문장.

    손마다 먼저 쓸 행동 하나, 두 손이 같은 문장이면 하나로, 없으면 idle 문장.
    """
    per_hand: list[tuple[str, str]] = []
    for hand in HANDS:
        mine = [
            x.payload
            for x in acts
            if isinstance(x.payload, ActionPayload) and x.payload.hand is hand
        ]
        if not mine:
            continue
        d = descs.get(mine[0].action_id)
        if d is not None and isinstance(d.payload, DescriptionPayload):
            used.add(d.label_id)
            per_hand.append((hand.value, d.payload.text))
        else:
            per_hand.append(
                (hand.value, _term(vocab, "verb", vocab.verbs, mine[0].verb, gp.language))
            )
    texts = list(dict.fromkeys(t for _, t in per_hand))
    if not texts:
        return gp.idle_action_text
    if len(texts) == 1:
        return texts[0]
    return "; ".join(f"{h} hand: {t}" for h, t in per_hand)


def _list_array(a: NDArray[Any], value: pa.DataType) -> pa.Array:
    """(n, d) 배열 → parquet 리스트 열 (행마다 길이 d)."""
    n, d = a.shape
    offsets = pa.array(np.arange(0, n * d + 1, d, dtype=np.int32))
    return pa.ListArray.from_arrays(offsets, pa.array(a.reshape(-1), type=value))


def _label_columns(ep: Episode) -> dict[str, NDArray[Any]]:
    """LeRobot 형식 프레임 특징을 labels.* 이름으로.

    observation.state는 labels.dlp_state로 싣고 action은 뺀다.
    """
    out: dict[str, NDArray[Any]] = {f"{LABEL_PREFIX}dlp_state": ep.state}
    for feat, attr in FRAME_ARRAYS.items():
        if feat.startswith("annotation."):
            out[LABEL_PREFIX + feat.removeprefix("annotation.")] = np.asarray(getattr(ep, attr))
    return out


def _label_features(lp: LeRobotPolicy, size: tuple[int, int]) -> dict[str, Any]:
    """labels.* 열의 info.json 특징 (LeRobot `features`의 dtype·shape·names 그대로)."""
    f = features(lp, size[1], size[0])
    out: dict[str, Any] = {f"{LABEL_PREFIX}dlp_state": f["observation.state"]}
    for feat in FRAME_ARRAYS:
        if feat.startswith("annotation."):
            out[LABEL_PREFIX + feat.removeprefix("annotation.")] = f[feat]
    return out


def _scalar(dtype: str) -> dict[str, Any]:
    """한 값 열의 info.json 특징."""
    return {"dtype": dtype, "shape": [1], "names": None}


def info_features(
    layout: StateLayout, lp: LeRobotPolicy, gp: Gr00tPolicy, size: tuple[int, int]
) -> dict[str, Any]:
    """info.json `features` (영상 + parquet 열 전부, dtype·shape·names).

    Args:
        size: 영상 (가로, 세로) (`video_size`로 줄인 뒤).
    """
    w, h = size
    n = len(layout.names)
    return {
        video_key(lp): {
            "dtype": "video", "shape": [h, w, 3], "names": ["height", "width", "channels"],
            "info": {
                "video.height": h, "video.width": w, "video.codec": "h264",
                "video.pix_fmt": gp.video.pix_fmt, "video.is_depth_map": False, "video.fps": lp.fps,
                "video.channels": 3, "has_audio": False,
            },
        },
        "observation.state": {"dtype": "float32", "shape": [n], "names": layout.names},
        "observation.hand_valid": {
            "dtype": "float32", "shape": [len(layout.valid)],
            "names": [v.removesuffix(".present") for v in layout.valid],
        },
        "action": {"dtype": "float32", "shape": [n], "names": layout.names},
        "timestamp": _scalar("float32"),
        "frame_index": _scalar("int64"),
        "episode_index": _scalar("int64"),
        "index": _scalar("int64"),
        "task_index": _scalar("int64"),
        f"annotation.{TASK_KEY}": _scalar("int64"),
        f"annotation.{ACTION_KEY}": _scalar("int64"),
        "next.reward": _scalar("float32"),
        "next.done": _scalar("bool"),
        **_label_features(lp, size),
    }  # fmt: skip


def episode_table(
    ep: Episode,
    layout: StateLayout,
    lp: LeRobotPolicy,
    text: EpisodeText,
    table: TaskTable,
    episode_index: int,
    first_index: int,
) -> pa.Table:
    """에피소드 하나의 parquet 표 (열 순서 = `info_features`의 영상 밖 순서)."""
    state, valid = layout.gather(ep, lp)
    action = np.concatenate([state[1:], state[-1:]], axis=0)
    n = len(ep.times_ms)
    task = np.array([table.index(s) for s in text.task], np.int64)
    act = np.array([table.index(s) for s in text.action], np.int64)
    done = np.zeros(n, bool)
    done[-1] = True
    cols: dict[str, pa.Array] = {
        "observation.state": _list_array(state, pa.float32()),
        "observation.hand_valid": _list_array(valid, pa.float32()),
        "action": _list_array(action, pa.float32()),
        "timestamp": pa.array(np.arange(n, dtype=np.float64) / lp.fps, type=pa.float32()),
        "frame_index": pa.array(np.arange(n, dtype=np.int64)),
        "episode_index": pa.array(np.full(n, episode_index, np.int64)),
        "index": pa.array(np.arange(first_index, first_index + n, dtype=np.int64)),
        "task_index": pa.array(task),
        f"annotation.{TASK_KEY}": pa.array(task),
        f"annotation.{ACTION_KEY}": pa.array(act),
        "next.reward": pa.array(np.zeros(n, np.float32)),
        "next.done": pa.array(done),
    }
    for name, a in _label_columns(ep).items():
        value = pa.float32() if a.dtype == np.float32 else pa.int64()
        cols[name] = _list_array(a, value)
    return pa.table(cols)


def _frames(
    video: Path, wanted: NDArray[np.int64], size: tuple[int, int]
) -> Iterator[NDArray[np.uint8]]:
    """블러본의 wanted[k]번째(표시 순서) 프레임을 차례로 RGB로 낸다.

    앞으로만 디코딩하고, 같은 번호가 이어지면 같은 프레임을 다시 낸다.

    wanted는 PTS 인덱스로 고른 번호라 감소하지 않는다 (`build_episode`). 크기가 다르면 INTER_AREA로
    줄인다.

    Raises:
        ExportError: 블러본이 wanted보다 짧을 때.
    """
    with av.open(str(video)) as c:
        decoded = c.decode(c.streams.video[0])
        i, rgb = -1, None
        for target in wanted.tolist():
            while i < target:
                frame = next(decoded, None)
                if frame is None:
                    raise ExportError(f"{video.name}: 블러본에 프레임 {target}이 없습니다")
                i, rgb = i + 1, None
                if i == target:
                    img: NDArray[np.uint8] = frame.to_ndarray(format="rgb24")  # pyright: ignore[reportAssignmentType]
                    if (img.shape[1], img.shape[0]) != size:
                        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)  # pyright: ignore[reportAssignmentType]
                    rgb = img
            assert rgb is not None
            yield rgb


def encode_video(
    video: Path,
    frame_index: NDArray[np.int64],
    size: tuple[int, int],
    fps: int,
    gv: Gr00tVideo,
    dest: Path,
) -> None:
    """행마다 고른 블러본 프레임을 고정 fps H.264 mp4로 쓴다.

    프레임 k의 PTS = k/fps, 소리·메타데이터 없음.

    프레임을 rawvideo로 ffmpeg 표준 입력에 넣으므로 출력 프레임 수 = len(frame_index)이다. 어느
    블러본 프레임이 들어가는지는 프레임 간격 계산이 아니라 PTS 인덱스로 고른 번호
    (`Episode.frame_index`)로 정해진다.

    Raises:
        ExportError: ffmpeg가 실패했을 때 (stderr 포함).
    """
    w, h = size
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        gv.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-framerate", str(fps),
        "-i", "pipe:0", "-an", "-map_metadata", "-1", "-c:v", gv.encoder, "-pix_fmt", gv.pix_fmt,
        "-preset", gv.preset, "-crf", str(gv.crf), "-g", str(gv.gop), "-r", str(fps),
        "-movflags", "+faststart", str(dest),
    ]  # fmt: skip
    log = dest.with_suffix(".log")
    with log.open("wb") as err:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=err)
        assert proc.stdin is not None
        try:
            for rgb in _frames(video, frame_index, size):
                proc.stdin.write(rgb.tobytes())
        except BrokenPipeError:
            pass
        finally:
            proc.stdin.close()
            code = proc.wait()
    message = log.read_text("utf-8", errors="replace")
    log.unlink()
    if code != 0:
        raise ExportError(
            f"ffmpeg가 {dest.name}을 쓰지 못했습니다 (종료 코드 {code}): {message.strip()}"
        )


@dataclass
class Gr00tResult:
    """`write_gr00t` 결과."""

    used: set[str]  # 실제로 프레임 특징·문장에 들어간 라벨 ID
    episodes: int
    frames: int
    tasks: int


def _split_ranges(splits: list[Split]) -> dict[str, str]:
    """에피소드 분할(분할 순으로 정렬됨) → info.json splits ({분할: "시작:끝"})."""
    out: dict[str, str] = {}
    for s in dict.fromkeys(splits):
        idx = [i for i, x in enumerate(splits) if x is s]
        out[s.value] = f"{idx[0]}:{idx[-1] + 1}"
    return out


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """JSON Lines (한 줄에 객체 하나, UTF-8)."""
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")


def _write_json(path: Path, data: Any) -> None:
    """들여쓴 JSON (UTF-8, 한글 그대로)."""
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


def write_gr00t(
    items: list[tuple[ExportSession, Path, Episode]],
    sessions: list[dict[str, Any]],
    policy: ExportPolicy,
    vocab: Vocab,
    size: tuple[int, int],
    dest: Path,
) -> Gr00tResult:
    """에피소드들을 GR00T LeRobot v2.1 데이터셋으로 쓴다 (배치는 모듈 설명).

    Args:
        items: (세션, 블러본 로컬 경로, 에피소드) (`runner.build_episodes`, 세션 순).
        sessions: items와 같은 순서의 meta/dlp_episodes.json 행 (가명 세션 ID).
        policy: 내보내기 정책 (`lerobot` 에피소드 값과 `gr00t` 배치·영상 값).
        vocab: 어휘 (labels.* 정수 열·문장의 온톨로지 이름).
        size: 블러본 크기 (가로, 세로). `video_size`로 줄여 모든 에피소드 영상에 쓴다.
        dest: 쓸 디렉터리 (만든다).

    에피소드 번호는 분할 순(golden, train, val, holdout 중 있는 것)으로 다시 매겨 info.json
    splits가 연속 구간이 되게 한다.
    """
    lp, gp = policy.lerobot, policy.gr00t
    layout = StateLayout.from_policy(lp)
    out_size = video_size(size, lp)
    order = sorted(range(len(items)), key=lambda i: list(Split).index(items[i][0].split))
    table = TaskTable()
    used: set[str] = set()
    episodes: list[dict[str, Any]] = []
    meta_rows: list[dict[str, Any]] = []
    total = 0
    for e, i in enumerate(order):
        es, video, ep = items[i]
        text = episode_text(es.labels, ep, vocab, gp)
        chunk = e // gp.chunks_size
        path = dest / DATA_PATH.format(episode_chunk=chunk, episode_index=e)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(episode_table(ep, layout, lp, text, table, e, total), path)
        vpath = VIDEO_PATH.format(episode_chunk=chunk, video_key=video_key(lp), episode_index=e)
        encode_video(video, ep.frame_index, out_size, lp.fps, gp.video, dest / vpath)
        n = len(ep.times_ms)
        episodes.append({"episode_index": e, "tasks": list(dict.fromkeys(text.task)), "length": n})
        meta_rows.append({**sessions[i], "episode_index": e})
        used |= ep.used | text.used
        total += n
    _write_meta(dest, policy, layout, out_size, table, episodes, [items[i][0].split for i in order])
    _write_json(dest / "meta" / "dlp_vocab.json", vocab.as_json())
    _write_json(dest / "meta" / "dlp_episodes.json", meta_rows)
    return Gr00tResult(used, len(items), total, len(table.texts))


def _write_meta(
    dest: Path,
    policy: ExportPolicy,
    layout: StateLayout,
    size: tuple[int, int],
    table: TaskTable,
    episodes: list[dict[str, Any]],
    splits: list[Split],
) -> None:
    """meta/info.json·episodes.jsonl·tasks.jsonl·modality.json·dlp_gr00t.json."""
    lp, gp = policy.lerobot, policy.gr00t
    meta = dest / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    n = len(episodes)
    info = {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": lp.robot_type,
        "total_episodes": n,
        "total_frames": sum(e["length"] for e in episodes),
        "total_tasks": len(table.texts),
        "total_videos": n,
        "total_chunks": math.ceil(n / gp.chunks_size),
        "chunks_size": gp.chunks_size,
        "fps": lp.fps,
        "splits": _split_ranges(splits),
        "data_path": DATA_PATH,
        "video_path": VIDEO_PATH,
        "features": info_features(layout, lp, gp, size),
    }
    _write_json(meta / "info.json", info)
    _write_jsonl(meta / "episodes.jsonl", episodes)
    _write_jsonl(
        meta / "tasks.jsonl", [{"task_index": i, "task": t} for i, t in enumerate(table.texts)]
    )
    modality = {
        "state": layout.modality(),
        "action": layout.modality(),
        "video": {gp.video_name: {"original_key": video_key(lp)}},
        "annotation": {TASK_KEY: {}, ACTION_KEY: {}},
    }
    _write_json(meta / "modality.json", modality)
    _write_json(meta / "dlp_gr00t.json", {
        "state": "hand and wearer-arm 3D points in the camera frame (metres), see info.json names",
        "state_valid": "observation.hand_valid: 1 if the point has a value at that frame, "
        "else 0 (the value is then 0)",
        "action": "absolute observation.state of the next frame (last frame repeats itself)",
        "relative_action": "use ActionRepresentation.RELATIVE with ActionType.NON_EEF "
        "in the modality config",
        "labels": "labels.* columns use meta/dlp_vocab.json codes (-1 = missing)",
    })  # fmt: skip
