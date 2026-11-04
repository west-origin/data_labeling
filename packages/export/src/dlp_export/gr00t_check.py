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
"""

# pyarrow에는 타입 스텁이 없어 이 모듈에서만 알 수 없는 타입 경고를 끈다 (dlp_media.tables와 같다).
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportUnknownParameterType=false

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

META_FILES = ("info.json", "episodes.jsonl", "tasks.jsonl", "modality.json")
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
    try:
        ds = _Dataset(
            root,
            json.loads((meta / "info.json").read_text("utf-8")),
            json.loads((meta / "modality.json").read_text("utf-8")),
            _read_jsonl(meta / "episodes.jsonl"),
            _read_jsonl(meta / "tasks.jsonl"),
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
        문제 문장 목록 (비면 통과).

    Raises:
        subprocess.CalledProcessError: ffprobe가 mp4를 읽지 못했을 때.
    """
    ds, out = _load(root)
    if ds is None:
        return out
    out += _check_info(ds)
    if out:
        return out
    out += _check_modality(ds) + _check_lists(ds)
    first = 0
    for e in range(len(ds.episodes)):
        path = ds.path("data_path", e)
        name = path.relative_to(root).as_posix()
        if not path.is_file():
            out.append(f"{name} 없음")
            continue
        t = pq.read_table(path)
        out += _check_schema(ds, t, name) + _check_values(ds, t, name, e, first)
        for key in _video_keys(ds):
            out += _check_video(ds, key, e, t.num_rows, ffprobe)
        first += t.num_rows
    return out
