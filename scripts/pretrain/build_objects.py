#!/usr/bin/env python3
"""공개 데이터셋으로 가사·청소·돌봄 객체 탐지 학습 데이터를 만든다 (ADR 0060 1단계).

입력은 공개 데이터셋 두 개뿐이다 (우리 데이터는 읽지도 쓰지도 않는다).
- Open Images V7: 공식 CSV(boxable 클래스 이름, train·validation 박스, 이미지 목록, 사람이 확인한
  이미지 단위 라벨)를 스트리밍으로 읽어 대상 클래스 행만 남긴다. 이미지는 공식 공개 버킷
  s3://open-images-dataset 의 https 주소에서 받는다.
- LVIS v1: 공식 zip(lvis_v1_train.json.zip, lvis_v1_val.json.zip)을 받아 ijson으로
  스트리밍 파싱한다.
  이미지는 각 이미지의 coco_url(COCO 2017)에서 받는다.

출력 (--out 아래, YOLOX `COCODataset` 배치):
- annotations/train.json, annotations/val.json: COCO 형식. 범주 ID = objects.yaml classes 순서 + 1,
  범주 이름 = 온톨로지 객체 ID. images[]에 `verified_class_ids`(그 이미지에서 빠짐없이 확인된
  범주 ID),
  출처(dataset, source_id, license, author, source_url)를 둔다.
- train2017/, val2017/: 이미지 (`--max-side`보다 긴 변은 줄여서 저장, 박스도 같은 비율).
- data_manifest.json: 데이터셋 판, 받은 주석 파일 URL·sha256, 클래스별 이미지·박스 수, 설정 해시,
  코드 커밋. --dry-run이면 data_manifest.dry_run.json (이미지를 받지 않고 개수만).
- _cache/: 걸러 낸 주석(다시 실행할 때 큰 CSV를 다시 받지 않는다), 받은 이미지 기록(이어 받기).

"확인된 클래스" 규칙 (빠진 박스가 배경으로 학습되지 않게, train_yolox.py가 쓴다):
- 원래 클래스 하나가 확인됨 = 그 이미지에 그 클래스 박스가 있거나(LVIS: 박스 범주, OI: 박스) 사람이
  없다고 확인함(LVIS: neg_category_ids, OI: Confidence 0 이미지 라벨).
  LVIS not_exhaustive_category_ids,
  OI의 묶음 박스(IsGroupOf)·min_box_px 미만으로 버린 박스가 있는 클래스는 확인 안 됨.
  OI의 박스 없는 양성 이미지 라벨(Confidence 1)도 확인 안 됨으로 둔다 (박스가 다 그려졌는지 모른다).
- 우리 클래스가 확인됨 = 그 클래스에 대응한 원래 클래스(그 데이터셋 몫) 전부가 확인됨. 그 데이터셋에
  대응이 없는 클래스는 확인 안 됨. OI 클래스 계층(상위 클래스 음성 → 하위 음성)은 쓰지 않는다
  (보수적).

실행 예:
    python build_objects.py --config config/pretrain/objects.yaml \
        --out /kaggle/working/objects --dry-run
    python build_objects.py --config config/pretrain/objects.yaml \
        --out /kaggle/working/objects

의존성: Python 3.10+, PyYAML, Pillow, ijson (requirements.txt). 나머지는 표준 라이브러리.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import http.client
import io
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

SCRIPT_VERSION = 1
USER_AGENT = "dlp-pretrain-objects/1 (+ADR 0060)"
OI = "https://storage.googleapis.com/openimages"
OI_IMAGE_URL = "https://open-images-dataset.s3.amazonaws.com/{subset}/{image_id}.jpg"
# Open Images V7 다운로드 안내의 파일
# (https://storage.googleapis.com/openimages/web/download_v7.html).
# 박스는 V6과 같다 (V7에서 바뀌지 않음). 이미지 목록은 boxable 이미지만 든 2018_04 판
# (회전 정보 포함).
OI_FILES = {
    "classes": f"{OI}/v7/oidv7-class-descriptions-boxable.csv",
    "boxes:train": f"{OI}/v6/oidv6-train-annotations-bbox.csv",
    "boxes:validation": f"{OI}/v5/validation-annotations-bbox.csv",
    "images:train": f"{OI}/2018_04/train/train-images-boxable-with-rotation.csv",
    "images:validation": f"{OI}/2018_04/validation/validation-images-with-rotation.csv",
    "labels:train": f"{OI}/v7/oidv7-train-annotations-human-imagelabels.csv",
    "labels:validation": f"{OI}/v7/oidv7-val-annotations-human-imagelabels.csv",
}
OI_SPLITS = ("train", "validation")
LVIS_FILES = {
    "train": "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip",
    "val": "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip",
}
DATASET_VERSIONS = {"open_images_v7": "v7", "lvis_v1": "v1"}
SPLIT_DIRS = {"train": "train2017", "val": "val2017"}  # YOLOX COCODataset 기본 디렉터리 이름
FLICKR_ID = re.compile(r"/(\d+)_[0-9a-f]+(?:_[a-z])?\.(?:jpg|jpeg|png)$", re.IGNORECASE)


# ---------------------------------------------------------------- 설정


@dataclass(frozen=True)
class Config:
    """objects.yaml 중 이 스크립트가 쓰는 부분 (검증된 값)."""

    raw_sha256: str
    datasets: tuple[str, ...]
    class_ids: tuple[str, ...]  # 온톨로지 객체 ID, 순서 = 출력 클래스 번호
    sources: dict[str, dict[str, tuple[str, ...]]]  # 데이터셋 → 우리 ID → 원래 이름들
    max_images_per_class: int
    val_fraction: float
    seed: int
    min_box_px: int
    verify_rule: str = "all"  # all | any (여러 원래 클래스를 합친 우리 클래스의 확인 규칙)

    def canonical(self) -> dict[str, Any]:
        """설정 해시에 넣는 값 (덮어쓴 값 포함)."""
        return {
            "datasets": list(self.datasets),
            "classes": list(self.class_ids),
            "sources": {d: {k: list(v) for k, v in m.items()} for d, m in self.sources.items()},
            "build": {
                "max_images_per_class": self.max_images_per_class,
                "val_fraction": self.val_fraction,
                "seed": self.seed,
                "min_box_px": self.min_box_px,
                "verify_rule": self.verify_rule,
            },
        }


def load_config(path: Path, max_images_override: int | None, verify_rule: str = "all") -> Config:
    """objects.yaml을 읽어 검증한다. 틀리면 SystemExit."""
    raw = path.read_bytes()
    data = yaml.safe_load(raw)
    datasets = tuple(data["datasets"])
    unknown = set(datasets) - set(DATASET_VERSIONS)
    if unknown:
        raise SystemExit(f"이 스크립트가 모르는 데이터셋: {sorted(unknown)}")
    classes: dict[str, dict[str, list[str]]] = data["classes"]
    sources: dict[str, dict[str, tuple[str, ...]]] = {d: {} for d in datasets}
    for obj, by_ds in classes.items():
        for ds, names in by_ds.items():
            if ds not in datasets:
                raise SystemExit(f"classes.{obj}: datasets에 없는 {ds}")
            if names:
                sources[ds][obj] = tuple(names)
    for ds, mapping in sources.items():
        seen: dict[str, str] = {}
        for obj, names in mapping.items():
            for n in names:
                if n in seen:
                    raise SystemExit(f"{ds} 클래스 {n!r}가 {seen[n]}와 {obj}에 모두 대응합니다")
                seen[n] = obj
    build = data["build"]
    max_images = max_images_override or int(build["max_images_per_class"])
    if max_images < 1:
        raise SystemExit("max_images_per_class는 1 이상이어야 합니다")
    return Config(
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        datasets=datasets,
        class_ids=tuple(classes),
        sources=sources,
        max_images_per_class=max_images,
        val_fraction=float(build["val_fraction"]),
        seed=int(build["seed"]),
        min_box_px=int(build["min_box_px"]),
        verify_rule=verify_rule,
    )


# ---------------------------------------------------------------- 내려받기


@dataclass
class FileRecord:
    """받은(또는 스트리밍한) 주석 파일 하나의 기록."""

    url: str
    sha256: str | None
    bytes: int
    partial: bool = False  # --max-rows로 중간에 멈춤 (sha256은 전체가 아님)

    def as_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "partial": self.partial,
        }


class HttpStream(io.RawIOBase):
    """끊기면 Range로 이어 받는 HTTP 읽기 스트림. 읽은 바이트의 sha256을 함께 계산한다."""

    def __init__(self, url: str, retries: int = 6, timeout: float = 120.0) -> None:
        super().__init__()
        self.url = url
        self.pos = 0
        self.sha = hashlib.sha256()
        self.retries = retries
        self.timeout = timeout
        self.total: int | None = None
        self._resp = self._open()
        length = self._resp.headers.get("Content-Length")
        self.total = int(length) if length is not None else None

    def _open(self) -> http.client.HTTPResponse:
        headers = {"User-Agent": USER_AGENT}
        if self.pos:
            headers["Range"] = f"bytes={self.pos}-"
        resp = urllib.request.urlopen(
            urllib.request.Request(self.url, headers=headers), timeout=self.timeout
        )
        if self.pos and resp.status != 206:
            resp.close()
            raise OSError(f"{self.url}: 이어 받기(Range)를 지원하지 않습니다")
        return resp

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Any) -> int:
        view = memoryview(buffer)
        attempt = 0
        while True:
            try:
                n = self._resp.readinto(view)
                break
            except (OSError, http.client.HTTPException) as exc:
                attempt += 1
                if attempt > self.retries:
                    raise
                print(f"  다시 연결 ({self.url}, {self.pos} 바이트, {exc})", file=sys.stderr)
                time.sleep(min(60, 2**attempt))
                self._resp.close()
                self._resp = self._open()
        self.sha.update(view[:n])
        self.pos += n
        return n

    def finish(self, partial: bool) -> FileRecord:
        """끝까지 읽었으면 길이를 확인하고 기록을 돌려준다."""
        complete = not partial
        if complete and self.total is not None and self.pos != self.total:
            raise OSError(f"{self.url}: {self.total} 바이트 중 {self.pos}만 받았습니다")
        self._resp.close()
        return FileRecord(
            self.url, self.sha.hexdigest() if complete else None, self.pos, partial=partial
        )


def stream_csv(
    url: str,
    on_row: Callable[[list[str]], None],
    max_rows: int | None,
    on_header: Callable[[list[str]], None] | None = None,
) -> tuple[list[str], FileRecord]:
    """CSV를 받으면서 행마다 on_row(값 목록)를 부른다 (디스크에 저장하지 않는다).

    on_header: 첫 행(열 이름)을 받아 열 위치를 정하는 함수.
    """
    started = time.monotonic()
    raw = HttpStream(url)
    text = io.TextIOWrapper(io.BufferedReader(raw, 1 << 20), encoding="utf-8", newline="")
    reader = csv.reader(text)
    header = next(reader)
    if on_header is not None:
        on_header(header)
    rows = 0
    partial = False
    for row in reader:
        on_row(row)
        rows += 1
        if max_rows is not None and rows >= max_rows:
            partial = True
            break
    record = raw.finish(partial)
    print(f"  {url}: {rows:,}행, {record.bytes / 1e6:,.1f} MB, {time.monotonic() - started:.0f}초")
    return header, record


def download_file(url: str, dest: Path) -> FileRecord:
    """파일 하나를 받아 저장한다 (sha256 계산)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    raw = HttpStream(url)
    with tmp.open("wb") as fh:
        while True:
            chunk = raw.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    record = raw.finish(partial=False)
    tmp.replace(dest)
    return record


def fetch_bytes(url: str, retries: int = 3, timeout: float = 60.0) -> bytes:
    """작은 파일(이미지)을 받는다. 404·403은 다시 시도하지 않는다."""
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code in (403, 404, 410):
                raise
            last = exc
        except (OSError, http.client.HTTPException) as exc:
            last = exc
        time.sleep(min(30, 2**attempt))
    raise OSError(f"{url}: {last}")


# ---------------------------------------------------------------- 후보 이미지


@dataclass
class Candidate:
    """대상 클래스가 든 원본 이미지 하나 (좌표는 0~1 정규화 [x0, y0, x1, y1])."""

    dataset: str
    source_id: str
    urls: list[str]
    boxes: list[tuple[int, float, float, float, float]] = field(default_factory=list)
    # 원래 클래스 단위 확인: 우리 클래스 번호 → 원래 이름별 확인 여부는 미리 합쳐 둔다
    verified: set[int] = field(default_factory=set)
    width: int | None = None  # LVIS만 미리 안다
    height: int | None = None
    flickr_id: str | None = None
    license: str | None = None
    author: str | None = None
    source_url: str | None = None
    skip_reason: str | None = None

    def classes(self) -> set[int]:
        return {b[0] for b in self.boxes}

    def to_json(self) -> dict[str, Any]:
        out = dict(self.__dict__)
        out["verified"] = sorted(self.verified)
        out["boxes"] = [list(b) for b in self.boxes]
        return out

    @staticmethod
    def from_json(d: dict[str, Any]) -> Candidate:
        c = Candidate(**{k: v for k, v in d.items() if k not in ("verified", "boxes")})
        c.verified = set(d["verified"])
        c.boxes = [(int(b[0]), b[1], b[2], b[3], b[4]) for b in d["boxes"]]
        return c


def flickr_id(url: str | None) -> str | None:
    """Flickr 사진 주소에서 사진 번호 (두 데이터셋에 같은 사진이 있는지 보는 데 쓴다)."""
    if not url:
        return None
    m = FLICKR_ID.search(url)
    return m.group(1) if m else None


def cache_key(*parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]


def load_cache(path: Path, key: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if data.get("key") == key else None


def save_cache(path: Path, key: str, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"key": key, **payload}), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------- Open Images V7


@dataclass
class OiBox:
    mid: str
    x0: float
    x1: float
    y0: float
    y1: float
    group: bool


def combine_verified(
    by_class: dict[int, list[Any]], ok: set[Any], unsure: set[Any], rule: str
) -> set[int]:
    """원래 클래스 확인 → 우리 클래스 확인.

    all: 대응한 원래 클래스가 모두 확인돼야 한다 (엄격, 기본).
    any: 하나라도 확인되고 확인 안 됨 표시(not_exhaustive·묶음·작은 박스)가 하나도 없으면 확인으로
    본다 (확인되지 않은 나머지 원래 클래스의 물체가 이미지에 있을 수 있다는 위험을 받아들인다).
    """
    out: set[int] = set()
    for idx, srcs in by_class.items():
        if rule == "all":
            hit = all(s in ok for s in srcs)
        else:
            hit = any(s in ok for s in srcs) and not any(s in unsure for s in srcs)
        if hit:
            out.add(idx)
    return out


def oi_class_mids(cfg: Config, files: list[FileRecord]) -> dict[str, int]:
    """OI 원래 클래스 MID → 우리 클래스 번호. 이름이 없으면 SystemExit."""
    by_name: dict[str, str] = {}

    def row(r: list[str]) -> None:
        by_name[r[1]] = r[0]

    header, record = stream_csv(OI_FILES["classes"], row, None)
    if header[:2] != ["LabelName", "DisplayName"]:
        raise SystemExit(f"OI 클래스 CSV 머리글이 예상과 다릅니다: {header}")
    files.append(record)
    mapping: dict[str, int] = {}
    missing: list[str] = []
    for obj, names in cfg.sources.get("open_images_v7", {}).items():
        for n in names:
            if n not in by_name:
                missing.append(f"{obj}: {n}")
                continue
            mapping[by_name[n]] = cfg.class_ids.index(obj)
    if missing:
        raise SystemExit(f"Open Images boxable 클래스에 없는 이름: {missing}")
    return mapping


def oi_read_boxes(
    split: str, mids: dict[str, int], max_rows: int | None
) -> tuple[dict[str, list[OiBox]], FileRecord]:
    """대상 MID 박스만 이미지별로 모은다."""
    out: dict[str, list[OiBox]] = defaultdict(list)
    cols: dict[str, int] = {}

    def row(r: list[str]) -> None:
        if r[cols["LabelName"]] not in mids:
            return
        out[r[cols["ImageID"]]].append(
            OiBox(
                mid=r[cols["LabelName"]],
                x0=float(r[cols["XMin"]]),
                x1=float(r[cols["XMax"]]),
                y0=float(r[cols["YMin"]]),
                y1=float(r[cols["YMax"]]),
                group=r[cols["IsGroupOf"]] == "1",
            )
        )

    def header(h: list[str]) -> None:
        cols.update({name: i for i, name in enumerate(h)})

    _, record = stream_csv(OI_FILES[f"boxes:{split}"], row, max_rows, header)
    return out, record


def oi_read_table(
    url: str,
    image_ids: set[str],
    keep: Callable[[dict[str, int], list[str]], Any | None],
    max_rows: int | None,
) -> tuple[dict[str, list[Any]], FileRecord]:
    """image_ids에 든 행만 keep(열 위치, 행) 값으로 모은다 (None이면 버림)."""
    out: dict[str, list[Any]] = defaultdict(list)
    cols: dict[str, int] = {}

    def row(r: list[str]) -> None:
        image_id = r[cols["ImageID"]]
        if image_id not in image_ids:
            return
        value = keep(cols, r)
        if value is not None:
            out[image_id].append(value)

    def header(h: list[str]) -> None:
        cols.update({name: i for i, name in enumerate(h)})

    _, record = stream_csv(url, row, max_rows, header)
    return out, record


def oi_verified(
    mids: dict[str, int], boxes: list[OiBox], labels: dict[str, str], rule: str
) -> set[int]:
    """원래 클래스 확인 → 우리 클래스 확인 (모듈 설명의 규칙)."""
    boxed = {b.mid for b in boxes if not b.group}
    unsure = {b.mid for b in boxes if b.group}
    negative = {m for m, conf in labels.items() if conf in ("0", "0.0")}
    ok = (boxed | negative) - unsure
    by_class: dict[int, list[str]] = defaultdict(list)
    for mid, idx in mids.items():
        by_class[idx].append(mid)
    return combine_verified(by_class, ok, unsure, rule)


def _oi_image_meta(cols: dict[str, int], r: list[str]) -> dict[str, str]:
    return {k: r[cols[k]] for k in ("Subset", "OriginalURL", "License", "Author", "Rotation")}


def _oi_label(mids: dict[str, int]) -> Callable[[dict[str, int], list[str]], Any | None]:
    def keep(cols: dict[str, int], r: list[str]) -> tuple[str, str] | None:
        mid = r[cols["LabelName"]]
        return (mid, r[cols["Confidence"]]) if mid in mids else None

    return keep


def oi_split_candidates(
    cfg: Config, mids: dict[str, int], split: str, max_rows: int | None, files: list[FileRecord]
) -> list[Candidate]:
    """OI 한 분할(train·validation)의 후보 이미지."""
    boxes, rec = oi_read_boxes(split, mids, max_rows)
    files.append(rec)
    ids = set(boxes)
    meta, rec = oi_read_table(OI_FILES[f"images:{split}"], ids, _oi_image_meta, max_rows)
    files.append(rec)
    labels, rec = oi_read_table(OI_FILES[f"labels:{split}"], ids, _oi_label(mids), max_rows)
    files.append(rec)
    out: list[Candidate] = []
    for image_id in sorted(ids):
        info = meta.get(image_id)
        cand = Candidate(
            dataset="open_images_v7",
            source_id=image_id,
            urls=[OI_IMAGE_URL.format(subset=split, image_id=image_id)],
        )
        if not info:
            cand.skip_reason = "no_image_metadata"
        else:
            m = info[0]
            cand.flickr_id = flickr_id(m["OriginalURL"])
            cand.license, cand.author, cand.source_url = m["License"], m["Author"], m["OriginalURL"]
            if m["Rotation"] not in ("", "0", "0.0"):
                cand.skip_reason = "rotated"
        image_labels = dict(labels.get(image_id, []))
        cand.verified = oi_verified(mids, boxes[image_id], image_labels, cfg.verify_rule)
        for b in boxes[image_id]:
            if not b.group:
                cand.boxes.append((mids[b.mid], b.x0, b.y0, b.x1, b.y1))
        out.append(cand)
    return out


def open_images_candidates(
    cfg: Config, cache_dir: Path, max_rows: int | None
) -> tuple[list[Candidate], list[FileRecord]]:
    """Open Images V7 후보 이미지 (걸러 낸 결과는 캐시에 둔다)."""
    key = cache_key(
        "oi", SCRIPT_VERSION, cfg.sources.get("open_images_v7"), cfg.verify_rule, max_rows, OI_FILES
    )
    path = cache_dir / "open_images_v7.json"
    cached = load_cache(path, key)
    if cached is not None:
        print("Open Images: 캐시를 씁니다", path)
        files = [FileRecord(**f) for f in cached["files"]]
        return [Candidate.from_json(c) for c in cached["candidates"]], files
    files: list[FileRecord] = []
    mids = oi_class_mids(cfg, files)
    cands: list[Candidate] = []
    for split in OI_SPLITS:
        cands.extend(oi_split_candidates(cfg, mids, split, max_rows, files))
    save_cache(
        path,
        key,
        {"files": [f.as_dict() for f in files], "candidates": [c.to_json() for c in cands]},
    )
    return cands, files


# ---------------------------------------------------------------- LVIS v1


def _lvis_items(zip_path: Path, prefix: str) -> Iterator[dict[str, Any]]:
    import ijson  # requirements.txt (C 백엔드 yajl2_c)

    with zipfile.ZipFile(zip_path) as zf:
        member = zf.namelist()[0]
        with zf.open(member) as fh:
            yield from ijson.items(fh, prefix, use_float=True)


def lvis_split_candidates(
    cfg: Config, zip_path: Path, cat_to_class: dict[int, int]
) -> list[Candidate]:
    """LVIS 한 분할의 후보 이미지 (대상 범주 박스가 하나 이상인 이미지)."""
    licenses = {
        lic["id"]: lic.get("url") or lic.get("name")
        for lic in _lvis_items(zip_path, "licenses.item")
    }
    anns: dict[int, list[tuple[int, list[float]]]] = defaultdict(list)
    for a in _lvis_items(zip_path, "annotations.item"):
        cat = int(a["category_id"])
        if cat in cat_to_class:
            anns[int(a["image_id"])].append((cat, [float(v) for v in a["bbox"]]))
    out: list[Candidate] = []
    for img in _lvis_items(zip_path, "images.item"):
        image_id = int(img["id"])
        if image_id not in anns:
            continue
        out.append(lvis_candidate(cfg, img, anns[image_id], cat_to_class, licenses))
    return out


def lvis_candidate(
    cfg: Config,
    img: dict[str, Any],
    anns: list[tuple[int, list[float]]],
    cat_to_class: dict[int, int],
    licenses: dict[int, str | None],
) -> Candidate:
    """LVIS 이미지 하나 → 후보 (min_box_px 미만 박스는 버리고 그 범주를 확인 안 됨으로)."""
    w, h = int(img["width"]), int(img["height"])
    coco_url = str(img["coco_url"])
    # 공식 coco_url(http)이 막힌 곳을 위해 같은 공식 버킷의 https 경로를 두 번째로 둔다
    alt = coco_url.replace(
        "http://images.cocodataset.org/", "https://s3.amazonaws.com/images.cocodataset.org/"
    )
    cand = Candidate(
        dataset="lvis_v1",
        source_id=str(img["id"]),
        urls=[coco_url] if alt == coco_url else [coco_url, alt],
        width=w,
        height=h,
        flickr_id=flickr_id(img.get("flickr_url")),
        license=licenses.get(int(img.get("license", -1))),
        source_url=img.get("flickr_url") or coco_url,
    )
    small: set[int] = set()
    for cat, (x, y, bw, bh) in anns:
        if min(bw, bh) < cfg.min_box_px:
            small.add(cat)
            continue
        cand.boxes.append(
            (cat_to_class[cat], x / w, y / h, min(1.0, (x + bw) / w), min(1.0, (y + bh) / h))
        )
    positive = {cat for cat, _ in anns}
    unsure = set(img.get("not_exhaustive_category_ids", [])) | small
    ok = (positive | set(img.get("neg_category_ids", []))) - unsure
    by_class: dict[int, list[int]] = defaultdict(list)
    for cat, idx in cat_to_class.items():
        by_class[idx].append(cat)
    cand.verified = combine_verified(by_class, ok, unsure, cfg.verify_rule)
    return cand


def _lvis_category_list(zip_path: Path) -> list[dict[str, Any]]:
    """범주 목록. 공식 파일은 categories가 앞에 있어 그 배열이 끝나면 바로 멈춘다."""
    import ijson
    from ijson.common import ObjectBuilder

    out: list[dict[str, Any]] = []
    builder: Any = None
    with zipfile.ZipFile(zip_path) as zf, zf.open(zf.namelist()[0]) as fh:
        for prefix, event, value in ijson.parse(fh):
            if prefix == "categories.item" and event == "start_map":
                builder = ObjectBuilder()
            if builder is not None:
                builder.event(event, value)
                if prefix == "categories.item" and event == "end_map":
                    out.append(builder.value)
                    builder = None
            if prefix == "categories" and event == "end_array":
                break
    return out


def lvis_categories(cfg: Config, zip_path: Path) -> dict[int, int]:
    """LVIS 범주 ID → 우리 클래스 번호. 이름이 없으면 SystemExit."""
    by_name = {c["name"]: int(c["id"]) for c in _lvis_category_list(zip_path)}
    mapping: dict[int, int] = {}
    missing: list[str] = []
    for obj, names in cfg.sources.get("lvis_v1", {}).items():
        for n in names:
            if n not in by_name:
                missing.append(f"{obj}: {n}")
                continue
            mapping[by_name[n]] = cfg.class_ids.index(obj)
    if missing:
        raise SystemExit(f"LVIS v1 범주에 없는 이름: {missing}")
    return mapping


def lvis_candidates(
    cfg: Config, cache_dir: Path, download_dir: Path, keep_downloads: bool, splits: tuple[str, ...]
) -> tuple[list[Candidate], list[FileRecord]]:
    """LVIS v1 후보 이미지 (zip은 다 읽은 뒤 지운다, --keep-downloads면 남김)."""
    key = cache_key(
        "lvis", SCRIPT_VERSION, cfg.sources.get("lvis_v1"), cfg.min_box_px, cfg.verify_rule, splits
    )
    path = cache_dir / "lvis_v1.json"
    cached = load_cache(path, key)
    if cached is not None:
        print("LVIS: 캐시를 씁니다", path)
        files = [FileRecord(**f) for f in cached["files"]]
        return [Candidate.from_json(c) for c in cached["candidates"]], files
    files: list[FileRecord] = []
    cands: list[Candidate] = []
    for split in splits:
        url = LVIS_FILES[split]
        zip_path = download_dir / Path(url).name
        started = time.monotonic()
        record = download_file(url, zip_path)
        files.append(record)
        print(f"  {url}: {record.bytes / 1e6:,.1f} MB, {time.monotonic() - started:.0f}초")
        cat_to_class = lvis_categories(cfg, zip_path)
        cands.extend(lvis_split_candidates(cfg, zip_path, cat_to_class))
        if not keep_downloads:
            zip_path.unlink()
    save_cache(
        path,
        key,
        {"files": [f.as_dict() for f in files], "candidates": [c.to_json() for c in cands]},
    )
    return cands, files


# ---------------------------------------------------------------- 고르기·분할


def rank(seed: int, *parts: str) -> str:
    """seed로 정해지는 순서 키 (입력 순서와 무관)."""
    return hashlib.sha256(":".join([str(seed), *parts]).encode()).hexdigest()


def split_of(cfg: Config, cand: Candidate) -> str:
    u = int(rank(cfg.seed, "split", cand.dataset, cand.source_id)[:12], 16) / float(16**12)
    return "val" if u < cfg.val_fraction else "train"


def select(cfg: Config, cands: list[Candidate], taken_flickr: set[str]) -> list[Candidate]:
    """한 데이터셋에서 클래스마다 max_images_per_class 이하로 이미지를 고른다.

    표본이 적은 클래스부터 고르고, 이미 고른 이미지에 든 클래스는 그 수를 센다
    (함께 든 클래스 때문에 처음 고른 클래스의 최종 수는 상한을 넘을 수 있다).
    앞 데이터셋에서 고른 Flickr 사진은 뺀다.
    """
    usable = [
        c
        for c in cands
        if c.skip_reason is None
        and c.boxes
        and (c.flickr_id is None or c.flickr_id not in taken_flickr)
    ]
    by_class: dict[int, list[Candidate]] = defaultdict(list)
    for c in usable:
        for idx in c.classes():
            by_class[idx].append(c)
    count: Counter[int] = Counter()
    chosen: dict[str, Candidate] = {}
    for idx in sorted(by_class, key=lambda i: (len(by_class[i]), i)):
        for c in sorted(by_class[idx], key=lambda c: rank(cfg.seed, c.dataset, c.source_id)):
            if count[idx] >= cfg.max_images_per_class:
                break
            if c.source_id in chosen:
                continue
            chosen[c.source_id] = c
            count.update(c.classes())
    return sorted(chosen.values(), key=lambda c: c.source_id)


def class_counts(cfg: Config, cands: list[Candidate]) -> dict[str, dict[str, int]]:
    """클래스별 이미지·박스·확인된 이미지 수."""
    out: dict[str, dict[str, int]] = {
        k: {"images": 0, "boxes": 0, "verified_images": 0} for k in cfg.class_ids
    }
    for c in cands:
        for idx in c.classes():
            out[cfg.class_ids[idx]]["images"] += 1
        for b in c.boxes:
            out[cfg.class_ids[b[0]]]["boxes"] += 1
        for idx in c.verified:
            out[cfg.class_ids[idx]]["verified_images"] += 1
    return out


# ---------------------------------------------------------------- 이미지 받기


@dataclass
class Stored:
    """저장한 이미지 하나 (원본 크기, 저장 크기, 원본 바이트 sha256)."""

    key: str
    file_name: str
    split: str
    orig_w: int
    orig_h: int
    width: int
    height: int
    sha256: str
    url: str


def store_image(cand: Candidate, split: str, out: Path, max_side: int) -> Stored | str:
    """이미지를 받아 저장한다. 실패하면 이유 문자열."""
    from PIL import Image

    data: bytes | None = None
    used = ""
    # 첫 바퀴는 주소마다 한 번씩만 (막힌 주소에서 기다리지 않게), 둘째 바퀴에서 다시 시도한다
    for retries in (0, 3):
        for url in cand.urls:
            try:
                data = fetch_bytes(url, retries=retries)
                used = url
                break
            except (OSError, urllib.error.HTTPError):
                continue
        if data is not None:
            break
    if data is None:
        return "download_failed"
    try:
        img = Image.open(io.BytesIO(data))
        orientation = img.getexif().get(0x0112, 1)
        w, h = img.size
    except Exception:  # 손상 파일
        return "decode_failed"
    if orientation not in (0, 1):
        return "exif_rotated"  # 회전 태그가 있으면 주석 좌표계를 확신할 수 없다
    if cand.width is not None and (w, h) != (cand.width, cand.height):
        return "size_mismatch"
    prefix = "oi" if cand.dataset == "open_images_v7" else "lvis"
    file_name = f"{prefix}_{cand.source_id}.jpg"
    dest = out / SPLIT_DIRS[split] / file_name
    dest.parent.mkdir(parents=True, exist_ok=True)
    scale = min(1.0, max_side / max(w, h)) if max_side > 0 else 1.0
    if scale < 1.0 or img.format != "JPEG":
        new = (max(1, round(w * scale)), max(1, round(h * scale)))
        img.convert("RGB").resize(new, Image.Resampling.BILINEAR).save(dest, "JPEG", quality=92)
        sw, sh = new
    else:
        dest.write_bytes(data)
        sw, sh = w, h
    return Stored(
        key=f"{cand.dataset}:{cand.source_id}",
        file_name=file_name,
        split=split,
        orig_w=w,
        orig_h=h,
        width=sw,
        height=sh,
        sha256=hashlib.sha256(data).hexdigest(),
        url=used,
    )


def download_all(
    cfg: Config, chosen: list[Candidate], out: Path, workers: int, max_side: int
) -> tuple[dict[str, Stored], Counter[str]]:
    """고른 이미지를 받는다. 받은 기록(_cache/images.jsonl)이 있으면 건너뛴다 (이어 받기)."""
    log = out / "_cache" / "images.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, Stored] = {}
    failed: Counter[str] = Counter()
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if "reason" in rec:
                continue  # 실패는 다시 시도한다
            s = Stored(**rec)
            if (out / SPLIT_DIRS[s.split] / s.file_name).exists():
                done[s.key] = s
    todo = [c for c in chosen if f"{c.dataset}:{c.source_id}" not in done]
    print(f"이미지: 받을 것 {len(todo):,}장 (이미 받음 {len(done):,}장)")
    started = time.monotonic()
    with log.open("a", encoding="utf-8") as fh, ThreadPoolExecutor(max_workers=workers) as pool:
        results = pool.map(lambda c: (c, store_image(c, split_of(cfg, c), out, max_side)), todo)
        for i, (cand, res) in enumerate(results, 1):
            if isinstance(res, Stored):
                done[res.key] = res
                fh.write(json.dumps(res.__dict__) + "\n")
            else:
                failed[res] += 1
                fh.write(
                    json.dumps({"key": f"{cand.dataset}:{cand.source_id}", "reason": res}) + "\n"
                )
            if i % 500 == 0 or i == len(todo):
                rate = i / max(1e-6, time.monotonic() - started)
                print(
                    f"  {i:,}/{len(todo):,} ({rate:.1f}장/초, 실패 {sum(failed.values())})",
                    flush=True,
                )
    return done, failed


# ---------------------------------------------------------------- COCO 쓰기


def build_coco(
    cfg: Config, chosen: list[Candidate], stored: dict[str, Stored]
) -> tuple[dict[str, dict[str, Any]], Counter[str]]:
    """분할별 COCO 사전. 픽셀 크기로 min_box_px를 적용하고, 같은 바이트의 이미지는 하나만 둔다."""
    categories = [
        {"id": i + 1, "name": name, "supercategory": "object"}
        for i, name in enumerate(cfg.class_ids)
    ]
    cocos = {
        s: {
            "info": {"description": "dlp pretrain objects (ADR 0060)", "version": SCRIPT_VERSION},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": categories,
        }
        for s in SPLIT_DIRS
    }
    dropped: Counter[str] = Counter()
    seen_sha: set[str] = set()
    next_img, next_ann = 1, 1
    order = {d: i for i, d in enumerate(cfg.datasets)}
    for cand in sorted(chosen, key=lambda c: (order[c.dataset], c.source_id)):
        s = stored.get(f"{cand.dataset}:{cand.source_id}")
        if s is None:
            continue
        if s.sha256 in seen_sha:
            dropped["duplicate_bytes"] += 1
            continue
        seen_sha.add(s.sha256)
        verified = set(cand.verified)
        anns: list[dict[str, Any]] = []
        for idx, x0, y0, x1, y1 in cand.boxes:
            if min((x1 - x0) * s.orig_w, (y1 - y0) * s.orig_h) < cfg.min_box_px:
                verified.discard(idx)  # 버린 박스 영역이 배경으로 학습되지 않게
                dropped["small_box"] += 1
                continue
            bx, by = x0 * s.width, y0 * s.height
            bw, bh = (x1 - x0) * s.width, (y1 - y0) * s.height
            anns.append(
                {
                    "id": next_ann,
                    "image_id": next_img,
                    "category_id": idx + 1,
                    "bbox": [round(bx, 2), round(by, 2), round(bw, 2), round(bh, 2)],
                    "area": round(bw * bh, 2),
                    "iscrowd": 0,
                }
            )
            next_ann += 1
        coco = cocos[s.split]
        coco["images"].append(
            {
                "id": next_img,
                "file_name": s.file_name,
                "width": s.width,
                "height": s.height,
                "dataset": cand.dataset,
                "source_id": cand.source_id,
                "verified_class_ids": sorted(i + 1 for i in verified),
                "license": cand.license,
                "author": cand.author,
                "source_url": cand.source_url,
                "sha256": s.sha256,
            }
        )
        coco["annotations"].extend(anns)
        next_img += 1
    return cocos, dropped


def coco_counts(cfg: Config, cocos: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """최종 클래스별 이미지·박스·확인 이미지 수 (분할·데이터셋별)."""
    out: dict[str, Any] = {}
    for name in cfg.class_ids:
        out[name] = {s: {"images": 0, "boxes": 0, "verified_images": 0} for s in SPLIT_DIRS}
    for split, coco in cocos.items():
        img_classes: dict[int, set[int]] = defaultdict(set)
        for a in coco["annotations"]:
            out[cfg.class_ids[a["category_id"] - 1]][split]["boxes"] += 1
            img_classes[a["image_id"]].add(a["category_id"])
        for img in coco["images"]:
            for cid in img_classes[img["id"]]:
                out[cfg.class_ids[cid - 1]][split]["images"] += 1
            for cid in img["verified_class_ids"]:
                out[cfg.class_ids[cid - 1]][split]["verified_images"] += 1
    return out


# ---------------------------------------------------------------- manifest


def code_info() -> dict[str, Any]:
    """코드 커밋(가능하면)과 이 스크립트의 sha256."""
    here = Path(__file__).resolve()
    info: dict[str, Any] = {
        "script": here.name,
        "script_sha256": hashlib.sha256(here.read_bytes()).hexdigest(),
    }
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=here.parent,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", "."],
            cwd=here.parent,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        info.update({"git_commit": commit, "git_dirty": bool(dirty)})
    except (OSError, subprocess.CalledProcessError):
        info.update({"git_commit": None, "git_dirty": None})
    return info


def write_json(path: Path, data: Any) -> str:
    """JSON을 쓰고 sha256을 돌려준다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def print_table(cfg: Config, per_ds: dict[str, dict[str, Any]]) -> None:
    print(f"\n{'class':<12}" + "".join(f"{d + ' cand/sel/box':>34}" for d in per_ds))
    for name in cfg.class_ids:
        cells = []
        for d in per_ds.values():
            c, s = d["candidates"][name], d["selected"][name]
            cells.append(f"{c['images']:>10,}/{s['images']:>8,}/{s['boxes']:>9,}")
        print(f"{name:<12}" + "".join(f"{x:>34}" for x in cells))


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", type=Path, required=True, help="config/pretrain/objects.yaml")
    p.add_argument("--out", type=Path, required=True, help="출력 디렉터리")
    p.add_argument(
        "--max-images-per-class", type=int, default=None, help="build.max_images_per_class 덮어쓰기"
    )
    p.add_argument(
        "--dry-run", action="store_true", help="주석만 받고 개수만 보고 (이미지 안 받음)"
    )
    p.add_argument("--workers", type=int, default=16, help="이미지 동시 내려받기 수")
    p.add_argument(
        "--max-side", type=int, default=1024, help="저장 이미지 긴 변 상한 px (0 = 원본 그대로)"
    )
    p.add_argument(
        "--verify-rule",
        choices=("all", "any"),
        default="all",
        help="여러 원래 클래스를 합친 클래스의 '확인됨' 규칙 (README 참고)",
    )
    p.add_argument("--keep-downloads", action="store_true", help="LVIS zip을 지우지 않음")
    p.add_argument("--lvis-splits", default="train,val", help="쓰는 LVIS 분할 (시험용)")
    p.add_argument(
        "--oi-max-rows",
        type=int,
        default=None,
        help="시험용: OI CSV마다 앞 N행만 읽음 (sha256 없음, manifest에 partial로 남음)",
    )
    return p.parse_args(argv)


def gather(
    cfg: Config, args: argparse.Namespace
) -> tuple[dict[str, list[Candidate]], list[FileRecord]]:
    cache_dir = args.out / "_cache"
    by_ds: dict[str, list[Candidate]] = {}
    files: list[FileRecord] = []
    for ds in cfg.datasets:
        print(f"== {ds}: 주석 읽기")
        if ds == "open_images_v7":
            cands, recs = open_images_candidates(cfg, cache_dir, args.oi_max_rows)
        else:
            splits = tuple(s for s in args.lvis_splits.split(",") if s)
            cands, recs = lvis_candidates(
                cfg, cache_dir, args.out / "_downloads", args.keep_downloads, splits
            )
        by_ds[ds] = cands
        files.extend(recs)
    return by_ds, files


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config, args.max_images_per_class, args.verify_rule)
    args.out.mkdir(parents=True, exist_ok=True)
    by_ds, files = gather(cfg, args)
    taken: set[str] = set()
    chosen: list[Candidate] = []
    per_ds: dict[str, dict[str, Any]] = {}
    for ds in cfg.datasets:
        sel = select(cfg, by_ds[ds], taken)
        taken.update(c.flickr_id for c in sel if c.flickr_id)
        chosen.extend(sel)
        skipped = Counter(c.skip_reason for c in by_ds[ds] if c.skip_reason)
        per_ds[ds] = {
            "version": DATASET_VERSIONS[ds],
            "candidates": class_counts(cfg, [c for c in by_ds[ds] if c.skip_reason is None]),
            "selected": class_counts(cfg, sel),
            "candidate_images": len(by_ds[ds]),
            "selected_images": len(sel),
            "skipped_candidates": dict(skipped),
        }
    print_table(cfg, per_ds)
    manifest: dict[str, Any] = {
        "kind": "dlp-pretrain-objects-data",
        "created_at": datetime.now(UTC).isoformat(),
        "dry_run": args.dry_run,
        "datasets": {d: {"name": d, "version": DATASET_VERSIONS[d]} for d in cfg.datasets},
        "annotation_files": [f.as_dict() for f in files],
        "partial_annotations": any(f.partial for f in files),
        "classes": list(cfg.class_ids),
        "config": {
            "path": str(args.config),
            "file_sha256": cfg.raw_sha256,
            "effective_sha256": hashlib.sha256(
                json.dumps(cfg.canonical(), sort_keys=True).encode()
            ).hexdigest(),
            "effective": cfg.canonical(),
        },
        "options": {
            "max_side": args.max_side,
            "lvis_splits": args.lvis_splits,
            "oi_max_rows": args.oi_max_rows,
        },
        "code": code_info(),
        "selection": per_ds,
        "split_assignment": "sha256(seed:split:dataset:source_id) < val_fraction → val",
    }
    if args.dry_run:
        write_json(args.out / "data_manifest.dry_run.json", manifest)
        print(
            f"\ndry-run: {len(chosen):,}장을 고름 (이미지는 받지 않음) → "
            f"{args.out / 'data_manifest.dry_run.json'}"
        )
        return 0
    stored, failed = download_all(cfg, chosen, args.out, args.workers, args.max_side)
    cocos, dropped = build_coco(cfg, chosen, stored)
    outputs = {}
    for split, coco in cocos.items():
        outputs[split] = {
            "json": f"annotations/{split}.json",
            "sha256": write_json(args.out / "annotations" / f"{split}.json", coco),
            "images": len(coco["images"]),
            "annotations": len(coco["annotations"]),
        }
    manifest.update(
        {
            "download_failures": dict(failed),
            "dropped": dict(dropped),
            "outputs": outputs,
            "class_counts": coco_counts(cfg, cocos),
            "image_dirs": SPLIT_DIRS,
        }
    )
    write_json(args.out / "data_manifest.json", manifest)
    print(f"완료: {outputs} → {args.out / 'data_manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
