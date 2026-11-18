"""GR00T LeRobot 데이터셋 확인 (`dlp export gr00t-check <디렉터리>`, ADR 0065).

`dlp_export.gr00t`가 쓴 결과를 쓰기 코드와 따로, NVIDIA 공개 요구만으로 다시 읽어 본다. 근거는
Isaac-GR00T `getting_started/data_preparation.md`, 로더 `lerobot_episode_loader.py`, 저장소 시험
`tests/gr00t/data/test_data_preparation.py`다. 다른 도구가 만든 GR00T 데이터셋에도 쓸 수 있다.
문제 목록을 돌려주고, 비면 통과다.

보는 것:
1. 배치: meta/{info.json, episodes.jsonl, tasks.jsonl, modality.json}, info.json 경로 틀대로의
   parquet·mp4.
2. info.json: codebase_version v2.1, fps·chunks_size·data_path·video_path·features, total_* 값이
   실제와 같다.
3. modality.json: state·action 키마다 정수 start < end이고, 키들의 구간이 겹치지 않고 빈틈없이
   observation.state·action 길이를 덮는다. video original_key는 info.json의 영상 특징이고,
   annotation 키마다 parquet 열이 있다.
4. tasks.jsonl·episodes.jsonl: 번호가 0부터 빈틈없고, 에피소드 길이 합이 total_frames다.
5. parquet: 열 = info.json의 영상 밖 특징, annotation 접두어 열은 modality.json annotation 키만,
   state·action은 float32 리스트이고 길이가 shape와 같다, 주석·task_index 값이 tasks.jsonl 번호
   안이다. frame_index·index·episode_index·timestamp(k/fps)와, 있으면 next.done(마지막만 참)이 맞다.
6. mp4 (ffprobe): 코덱 h264, pix_fmt·크기가 info.json과 같고, 평균 프레임률 = fps, 디코딩한 프레임
   수 = parquet 행 수, 프레임 k의 표시 시각이 k/fps(반 프레임 안)이고, 소리가 없다. h264는 우리
   기준이다 (NVIDIA 예제는 av1이지만 디코더의 FFmpeg 빌드에 따라 못 읽을 수 있어 받지 않는다).
7. 촉각장갑 구간 (meta/dlp_tactile.json이 있을 때만, ADR 0069. NVIDIA 요구가 아니라 우리 추가
   파일의 약속이다): 그 파일의 손마다 modality.json state에 `state_key` 구간이 있고 길이가 채널
   수와 같으며, 그 구간의 info.json observation.state 이름이 `<손>.tactile.<채널>`이다. 촉각 구간은
   action에 없다. 유효 표시 열(`valid_column`)이 있고 그 손의 이름을 담는다. parquet에서 유효
   표시는 0 또는 1이고, 0인 행은 그 구간 값이 모두 0이다 (없는 값을 채워 넣지 않았다).
8. 장면 좌표 바디캠 자세 구간 (meta/dlp_scene.json이 있고 자세를 실었을 때만, ADR 0070. 우리 추가
   파일의 약속이다): 7번과 같은 조건을 `state_key` 구간(이름 = 그 파일의 names)과 유효 표시 열
   (`valid_column`의 `valid_name`)에 본다.
"""

# pyarrow에는 타입 스텁이 없어 이 모듈에서만 알 수 없는 타입 경고를 끈다 (dlp_media.tables와 같다).
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportUnknownParameterType=false

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

META_FILES = ("info.json", "episodes.jsonl", "tasks.jsonl", "modality.json")
# 촉각장갑 채널 설명 (우리 추가 파일, 있을 때만 7번 검사를 한다)
TACTILE_META = "dlp_tactile.json"
# 장면 좌표 바디캠 자세 설명 (우리 추가 파일, 있을 때만 8번 검사를 한다)
SCENE_META = "dlp_scene.json"
INFO_KEYS = (
    "fps",
    "chunks_size",
    "data_path",
    "video_path",
    "features",
    "total_episodes",
    "total_frames",
)
# 있어야 하는 열 (data_preparation.md 예제 행 중 NVIDIA 예제 데이터 cube_to_bowl_5에도 있는 것).
# next.reward·next.done은 예제 행에만 있고 예제 데이터에는 없어서, 있을 때만 값을 본다
REQUIRED_COLUMNS = (
    "observation.state", "action", "timestamp", "task_index", "episode_index", "index",
)  # fmt: skip
# info.json dtype → parquet 한 값 타입
_SCALARS = {
    "float32": pa.float32(), "float64": pa.float64(), "int64": pa.int64(), "int32": pa.int32(),
    "bool": pa.bool_(),
}  # fmt: skip


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


@dataclass
class _Dataset:
    """읽은 메타데이터."""

    root: Path
    info: dict[str, Any]
    modality: dict[str, Any]
    episodes: list[dict[str, Any]]
    tasks: list[dict[str, Any]]
    tactile: dict[str, Any] | None = None  # meta/dlp_tactile.json (없으면 None, ADR 0069)
    scene: dict[str, Any] | None = None  # meta/dlp_scene.json (없으면 None, ADR 0070)

    @property
    def features(self) -> dict[str, Any]:
        return self.info["features"]

    def path(self, pattern_key: str, episode_index: int, video_key: str = "") -> Path:
        """info.json 경로 틀(data_path·video_path)로 에피소드 파일 경로."""
        chunk = episode_index // int(self.info["chunks_size"])
        rel = self.info[pattern_key].format(
            episode_chunk=chunk, episode_index=episode_index, video_key=video_key
        )
        return self.root / rel


def _load(root: Path) -> tuple[_Dataset | None, list[str]]:
    """메타 파일을 읽는다 (없거나 깨졌으면 문제와 None)."""
    problems = [f"meta/{f} 없음" for f in META_FILES if not (root / "meta" / f).is_file()]
    problems += [f"{d}/ 없음" for d in ("data", "videos") if not (root / d).is_dir()]
    if problems:
        return None, problems
    meta = root / "meta"
    tactile, scene = meta / TACTILE_META, meta / SCENE_META
    try:
        ds = _Dataset(
            root,
            json.loads((meta / "info.json").read_text("utf-8")),
            json.loads((meta / "modality.json").read_text("utf-8")),
            _read_jsonl(meta / "episodes.jsonl"),
            _read_jsonl(meta / "tasks.jsonl"),
            json.loads(tactile.read_text("utf-8")) if tactile.is_file() else None,
            json.loads(scene.read_text("utf-8")) if scene.is_file() else None,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        return None, [f"메타 JSON을 읽지 못함: {e}"]
    return ds, []


def _check_info(ds: _Dataset) -> list[str]:
    """info.json 필수 키·판."""
    out = [f"info.json에 {k} 없음" for k in INFO_KEYS if k not in ds.info]
    if ds.info.get("codebase_version") != "v2.1":
        out.append(f"info.json codebase_version이 v2.1이 아님: {ds.info.get('codebase_version')}")
    for k in ("observation.state", "action"):
        f = ds.info.get("features", {}).get(k)
        if f is None or f.get("dtype") != "float32" or len(f.get("shape", [])) != 1:
            out.append(f"info.json {k}가 float32 1차원 특징이 아님")
    return out


def _check_ranges(ds: _Dataset, part: str) -> list[str]:
    """modality.json state·action 구간이 정수 start < end이고 겹치지 않게 빈틈없이 길이를 덮는다."""
    groups = ds.modality.get(part)
    if not isinstance(groups, dict) or not groups:
        return [f"modality.json에 {part} 없음"]
    feature = "observation.state" if part == "state" else "action"
    length = int(ds.features[feature]["shape"][0])
    out: list[str] = []
    spans: list[tuple[int, int, str]] = []
    for key, spec in groups.items():
        s, e = spec.get("start"), spec.get("end")
        if not (isinstance(s, int) and isinstance(e, int) and 0 <= s < e):
            out.append(f"modality.json {part}.{key}: start·end가 정수 start < end가 아님")
            continue
        spans.append((s, e, str(key)))
    cursor = 0
    for s, e, key in sorted(spans):
        if s != cursor:
            out.append(f"modality.json {part}.{key}: {cursor}에서 이어지지 않음 (start {s})")
        cursor = max(cursor, e)
    if not out and cursor != length:
        out.append(f"modality.json {part} 구간 끝 {cursor} != {feature} 길이 {length}")
    return out


def _check_modality(ds: _Dataset) -> list[str]:
    """modality.json video·annotation 절."""
    out = _check_ranges(ds, "state") + _check_ranges(ds, "action")
    videos = ds.modality.get("video")
    if not isinstance(videos, dict) or not videos:
        out.append("modality.json에 video 없음")
        videos = {}
    for key, spec in videos.items():
        original = spec.get("original_key", f"observation.images.{key}")
        f = ds.features.get(original)
        if f is None or f.get("dtype") != "video":
            out.append(f"modality.json video.{key}: info.json에 영상 특징 {original} 없음")
    for key, spec in ds.modality.get("annotation", {}).items():
        col = spec.get("original_key", f"annotation.{key}")
        if col not in ds.features:
            out.append(f"modality.json annotation.{key}: info.json에 열 {col} 없음")
    return out


def _check_lists(ds: _Dataset) -> list[str]:
    """tasks.jsonl·episodes.jsonl 번호와 total_* 값."""
    out: list[str] = []
    if [t.get("task_index") for t in ds.tasks] != list(range(len(ds.tasks))):
        out.append("tasks.jsonl task_index가 0부터 빈틈없지 않음")
    if any(not isinstance(t.get("task"), str) or not t["task"] for t in ds.tasks):
        out.append("tasks.jsonl에 빈 task 문장")
    if [e.get("episode_index") for e in ds.episodes] != list(range(len(ds.episodes))):
        out.append("episodes.jsonl episode_index가 0부터 빈틈없지 않음")
    totals = {
        "total_episodes": len(ds.episodes),
        "total_frames": sum(int(e.get("length", 0)) for e in ds.episodes),
        "total_tasks": len(ds.tasks),
    }
    out += [
        f"info.json {k} {ds.info.get(k)} != 실제 {v}"
        for k, v in totals.items()
        if k in ds.info and ds.info[k] != v
    ]
    return out


def _video_keys(ds: _Dataset) -> list[str]:
    return [k for k, f in ds.features.items() if f.get("dtype") == "video"]


def _check_schema(ds: _Dataset, t: pa.Table, name: str) -> list[str]:
    """parquet 열 이름·타입이 info.json 특징·GR00T 규칙과 같은가."""
    cols = set(t.column_names)
    expected = {k for k in ds.features if k not in _video_keys(ds)}
    allowed_ann = {
        spec.get("original_key", f"annotation.{k}")
        for k, spec in ds.modality.get("annotation", {}).items()
    }
    out = [
        f"{name}: 열 {c} 없음"
        for c in sorted((expected | set(REQUIRED_COLUMNS) | allowed_ann) - cols)
    ]
    out += [f"{name}: info.json에 없는 열 {c}" for c in sorted(cols - expected)]
    out += [
        f"{name}: annotation 접두어 열 {c}가 modality.json annotation에 없음"
        for c in sorted(cols)
        if c.startswith("annotation") and c not in allowed_ann
    ]
    for c in sorted(cols & expected):
        f = ds.features[c]
        typ = t.schema.field(c).type
        scalar = _SCALARS.get(f["dtype"])
        if f["shape"] == [1] and f.get("names") is None:
            ok = typ == scalar
        else:
            ok = pa.types.is_list(typ) and typ.value_type == scalar  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
        if not ok:
            out.append(f"{name}: 열 {c} 타입 {typ}가 info.json {f['dtype']} {f['shape']}와 다름")
    return out


def _check_values(ds: _Dataset, t: pa.Table, name: str, e: int, first: int) -> list[str]:
    """parquet 값: 길이, 번호 열, 시각, 주석 번호 범위, 끝 표시."""
    n = t.num_rows
    out: list[str] = []
    if n != int(ds.episodes[e].get("length", -1)):
        out.append(f"{name}: 행 {n} != episodes.jsonl length {ds.episodes[e].get('length')}")
    for c in ("observation.state", "action"):
        if c in t.column_names:
            lens = {len(v) for v in t.column(c).to_pylist()}
            if lens != {int(ds.features[c]["shape"][0])}:
                out.append(f"{name}: {c} 행 길이 {sorted(lens)} != shape {ds.features[c]['shape']}")
    want = {
        "frame_index": list(range(n)),
        "index": list(range(first, first + n)),
        "episode_index": [e] * n,
    }
    out += [
        f"{name}: {c} 값이 틀림"
        for c, v in want.items()
        if c in t.column_names and t.column(c).to_pylist() != v
    ]
    fps = float(ds.info["fps"])
    if "timestamp" in t.column_names:
        ts = t.column("timestamp").to_pylist()
        if any(abs(x - k / fps) > 1e-4 for k, x in enumerate(ts)):
            out.append(f"{name}: timestamp가 k/fps가 아님")
    ann = [c for c in t.column_names if c.startswith("annotation") or c == "task_index"]
    for c in ann:
        vals = t.column(c).to_pylist()
        if any(not isinstance(v, int) or not 0 <= v < len(ds.tasks) for v in vals):
            out.append(f"{name}: {c} 값이 tasks.jsonl 번호 밖")
    if "next.done" in t.column_names and t.column("next.done").to_pylist() != [False] * (n - 1) + [
        True
    ]:
        out.append(f"{name}: next.done이 마지막 프레임에만 참이 아님")
    return out


@dataclass(frozen=True)
class _FlaggedSpan:
    """우리 추가 구간 하나(촉각 손 하나, 장면 자세): state 구간과 유효 표시 열 안 위치."""

    label: str  # 문제 문장에 쓰는 이름 (손 이름, scene_camera)
    start: int
    end: int
    column: str  # 유효 표시 열 이름
    index: int  # 그 열 안에서 이 구간의 위치


def _flagged_span(
    ds: _Dataset, where: str, label: str, spec: dict[str, Any], names: list[str], what: str
) -> _FlaggedSpan | str:
    """state 구간 하나. 모듈 설명 7·8번의 메타 조건을 어기면 문제 문장을 돌려준다.

    Args:
        where: 문제 문장 앞머리 (메타 파일과 위치). label: 구간 이름.
        spec: 그 구간의 메타 (`state_key`, `valid_column`, `valid_name`).
        names: 그 구간의 observation.state 이름이어야 하는 목록. what: 그 목록을 부르는 말.
    """
    key, column, valid_name = (
        spec.get("state_key"),
        spec.get("valid_column"),
        spec.get("valid_name"),
    )
    if not (isinstance(key, str) and isinstance(column, str) and isinstance(valid_name, str)):
        return f"{where}: state_key·valid_column·valid_name이 없음"
    span = ds.modality.get("state", {}).get(key)
    if not isinstance(span, dict) or not isinstance(span.get("start"), int):
        return f"{where}: modality.json state에 구간 {key} 없음"
    start, end = span["start"], span.get("end")
    if not isinstance(end, int) or end - start != len(names):
        return f"{where}: 구간 {key} 길이가 {what} 수 {len(names)}와 다름"
    state_names = ds.features["observation.state"].get("names") or []
    if state_names[start:end] != names:
        return f"{where}: observation.state 이름 {start}:{end}가 {what} 목록과 다름"
    if key in ds.modality.get("action", {}):
        return f"{where}: 구간 {key}가 modality.json action에 있음"
    valid = (ds.features.get(column) or {}).get("names") or []
    if valid_name not in valid:
        return f"{where}: info.json {column}에 {valid_name} 없음"
    return _FlaggedSpan(label, start, end, column, valid.index(valid_name))


def _tactile_hands(ds: _Dataset) -> dict[str, dict[str, Any]]:
    """dlp_tactile.json의 hands 절 (파일이 없거나 모양이 틀리면 빈 사전)."""
    hands = (ds.tactile or {}).get("hands")
    if not isinstance(hands, dict):
        return {}
    return {str(k): v for k, v in hands.items() if isinstance(v, dict)}


def _tactile_span(ds: _Dataset, hand: str, spec: dict[str, Any]) -> _FlaggedSpan | str:
    """손 하나의 촉각 구간 (이름 = `<손>.tactile.<채널>`)."""
    where = f"dlp_tactile.json hands.{hand}"
    channels = spec.get("channels")
    if not isinstance(channels, list):
        return f"{where}: channels가 없음"
    names = [f"{hand}.tactile.{c}" for c in channels]
    return _flagged_span(ds, where, hand, spec, names, "채널")


def _scene_span(ds: _Dataset) -> _FlaggedSpan | str | None:
    """장면 자세 구간. dlp_scene.json이 없거나 자세를 실은 세션이 없으면(state_key 없음) None."""
    meta = ds.scene
    if meta is None or "state_key" not in meta:
        return None
    names = meta.get("names")
    if not isinstance(names, list):
        return "dlp_scene.json: names가 없음"
    return _flagged_span(ds, "dlp_scene.json", "scene_camera", meta, names, "값 이름")


def _check_spans(ds: _Dataset) -> tuple[list[_FlaggedSpan], list[str]]:
    """(값을 확인할 촉각·장면 자세 구간, 메타 문제). 추가 메타 파일이 없으면 둘 다 비어 있다."""
    got: list[_FlaggedSpan | str | None] = [
        _tactile_span(ds, hand, spec) for hand, spec in _tactile_hands(ds).items()
    ]
    got.append(_scene_span(ds))
    spans = [g for g in got if isinstance(g, _FlaggedSpan)]
    return spans, [g for g in got if isinstance(g, str)]


def _check_flagged_values(spans: list[_FlaggedSpan], t: pa.Table, name: str) -> list[str]:
    """parquet의 추가 구간 값: 유효 표시는 0 또는 1이고, 0인 행은 그 구간 값이 모두 0이다."""
    out: list[str] = []
    if not spans or "observation.state" not in t.column_names:
        return out
    state = t.column("observation.state").to_pylist()
    for span in spans:
        if span.column not in t.column_names:
            continue  # 열이 없다는 것은 _check_schema가 알린다
        rows = t.column(span.column).to_pylist()
        flags = [row[span.index] if len(row) > span.index else None for row in rows]
        if any(f not in (0.0, 1.0) for f in flags):
            out.append(f"{name}: {span.column} {span.label} 값이 0·1이 아님")
        if any(
            f == 0.0 and any(row[span.start : span.end])
            for f, row in zip(flags, state, strict=True)
        ):
            out.append(f"{name}: {span.label} 값이 없다고 표시한 행에 0이 아닌 값이 있음")
    return out


def _ffprobe(ffprobe: str, path: Path) -> dict[str, Any]:
    """첫 영상 스트림 정보와 프레임별 표시 시각 (ffprobe JSON). 소리 스트림 수도 센다."""
    cmd = [
        ffprobe, "-v", "error", "-count_frames", "-show_entries",
        "stream=codec_type,codec_name,pix_fmt,width,height,avg_frame_rate,nb_read_frames:frame=pts_time",
        "-of", "json", str(path),
    ]  # fmt: skip
    data: dict[str, Any] = json.loads(
        subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    )
    return data


def _check_video(ds: _Dataset, key: str, e: int, rows: int, ffprobe: str) -> list[str]:
    """mp4 하나를 모듈 설명 6번 기준으로 본다."""
    path = ds.path("video_path", e, key)
    name = path.relative_to(ds.root).as_posix()
    if not path.is_file():
        return [f"{name} 없음"]
    info = ds.features[key].get("info", {})
    data = _ffprobe(ffprobe, path)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        return [f"{name}: 영상 스트림 없음"]
    fps = Fraction(str(ds.info["fps"]))
    out = (
        [f"{name}: 소리 스트림이 있음"]
        if any(s.get("codec_type") == "audio" for s in streams)
        else []
    )
    checks = {
        "codec": (video.get("codec_name"), "h264"),
        "pix_fmt": (video.get("pix_fmt"), info.get("video.pix_fmt", video.get("pix_fmt"))),
        "size": ([video.get("height"), video.get("width")], ds.features[key]["shape"][:2]),
        "avg_frame_rate": (Fraction(video.get("avg_frame_rate", "0/1")), fps),
        "frames": (int(video.get("nb_read_frames", -1)), rows),
    }
    out += [f"{name}: {k} {got} != {want}" for k, (got, want) in checks.items() if got != want]
    times = [float(f["pts_time"]) for f in data.get("frames", []) if "pts_time" in f]
    if len(times) == rows and any(
        abs(x - k / float(fps)) > 0.5 / float(fps) for k, x in enumerate(times)
    ):
        out.append(f"{name}: 프레임 표시 시각이 k/fps가 아님")
    return out


def check_dataset(root: Path, ffprobe: str = "ffprobe") -> list[str]:
    """GR00T LeRobot 데이터셋 디렉터리를 확인한다 (보는 것은 모듈 설명).

    Args:
        root: 데이터셋 루트 (meta/·data/·videos/가 있는 디렉터리).
        ffprobe: ffprobe 실행 파일 (export.yaml gr00t.video.ffprobe).

    Returns:
        문제 문장 목록 (비면 통과). ffprobe 실행 파일이 없으면 영상을 확인하지 못한 것도 문제로
        돌려준다 (확인 없이 통과시키지 않는다).

    Raises:
        subprocess.CalledProcessError: ffprobe가 mp4를 읽지 못했을 때.
    """
    ds, out = _load(root)
    if ds is None:
        return out
    out += _check_info(ds)
    if out:
        return out
    spans, span_problems = _check_spans(ds)
    out += _check_modality(ds) + _check_lists(ds) + span_problems
    probe_videos = shutil.which(ffprobe) is not None
    if not probe_videos:
        out.append(
            f"ffprobe 실행 파일을 찾을 수 없어 영상을 확인하지 못함: {ffprobe} "
            "(FFmpeg 설치 또는 export.yaml gr00t.video.ffprobe)"
        )
    first = 0
    for e in range(len(ds.episodes)):
        path = ds.path("data_path", e)
        name = path.relative_to(root).as_posix()
        if not path.is_file():
            out.append(f"{name} 없음")
            continue
        t = pq.read_table(path)
        out += _check_schema(ds, t, name) + _check_values(ds, t, name, e, first)
        out += _check_flagged_values(spans, t, name)
        for key in _video_keys(ds) if probe_videos else ():
            out += _check_video(ds, key, e, t.num_rows, ffprobe)
        first += t.num_rows
    return out
