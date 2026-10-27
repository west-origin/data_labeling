"""학습 예제 → YOLOX COCO 데이터 디렉터리 (ADR 0062, 학습기 `dlp_train.yolox`).

입력 경계:
- 예제는 데이터셋 버전의 학습·검증 분할(`TrainingPolicy.splits`)에서만 온다
  (`augment.check_examples`).
- 픽셀은 호출자가 주는 블러본 공급자(`BlurredVideo`)에서만 온다. `dlp train run`은 라벨링 버킷의
  지금 승인된 블러본(렌더 게이트 확인)을 넘긴다 (`dlp_cli.train_cmds`). 원본 버킷은 읽지 않는다.
- 결과 디렉터리는 학습 임시 디렉터리 안에만 있고 라벨 이력·DB·버킷에 쓰지 않는다.

이미지와 주석:
- (세션, 스트림)마다 그 스트림 box_track 예제의 키프레임 시각(스트림 PTS ms)을 모은다. 블러본의 PTS
  인덱스에서 그 시각(정수 ms 반올림, `dlp_media.frames.frame_ms`)인 프레임만 이미지로 쓴다. 맞는
  프레임이 없으면 건너뛰고 센다 (프레임 번호 x 간격으로 시각을 계산하지 않는다).
- 이미지의 주석 = 그 시각에 보이는 학습 박스 (지운 예제 제외, 정책 classes 안의 클래스만). 키프레임
  사이는 선형 보간하고 outside 구간·첫 키프레임 앞·마지막 키프레임 뒤는 없다 (CVAT 트랙과 같은 뜻).
  지운 예제 (deleted, 오탐)의 시각도 이미지로 넣어 그 자리를 배경으로 배운다. 화면 밖으로 나간
  부분은 잘라 내고, 짧은 변이 `min_box_px`보다 작은 박스는 뺀다.
- 학습 분할 이미지는 증강 정책이 있으면 `augment.augment_examples`로 증강본을 더한다 (박스는 같은
  변환, ADR 0057). 증강 씨앗의 표본 번호는 (세션, 스트림, 시각) 정렬 순서다 (같은 입력이면 같은
  결과).
- 배치는 YOLOX 0.3.0 `COCODataset` 규칙: `<root>/annotations/{train,val}.json`, 이미지
  `<root>/{train2017,val2017}/<파일>`. 범주 ID는 classes 순서의 1..C (YOLOX가 정렬한 범주 ID 순으로
  열 번호를 매긴다).
"""

from __future__ import annotations

import json
from bisect import bisect_right
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from dlp_media.frames import decode_frames, frame_ms
from dlp_media.pts import build_pts_index
from dlp_schema.dataset import Split
from dlp_schema.labels import BoxKeyframe, BoxTrackPayload
from dlp_train.augment import BoxItem, Sample, augment_examples, check_examples
from dlp_train.extract import Example, TrainingData
from dlp_train.policy import AugmentationPolicy

Box = tuple[float, float, float, float]
# (세션, 스트림, 작업 디렉터리) → 지금 승인된 블러본 로컬 경로
BlurredVideo = Callable[[str, str, Path], Path]
# 원 해상도 디코드 (줄이지 않는다)
_FULL_SIZE = 1 << 30
# YOLOX COCODataset 이미지 디렉터리 이름 (`name` 인자 기본값과 맞춘다)
IMAGE_DIRS = {Split.TRAIN: "train2017", Split.VAL: "val2017"}
ANNOTATION_FILES = {Split.TRAIN: "train.json", Split.VAL: "val.json"}


@dataclass
class CocoStats:
    """COCO 내보내기 집계 (manifest·MLflow 지표에 남긴다)."""

    images: dict[str, int] = field(default_factory=dict[str, int])  # 분할 → 이미지 수 (증강 포함)
    annotations: dict[str, int] = field(default_factory=dict[str, int])  # 분할 → 박스 수
    augmented_images: int = 0  # 학습 분할에 더한 증강본 수
    negative_images: int = 0  # 박스가 하나도 없는 이미지 (증강본 제외)
    skipped_frames: int = 0  # 블러본 PTS 인덱스에 그 시각 프레임이 없어 건너뛴 키프레임 시각
    skipped_classes: dict[str, int] = field(default_factory=dict[str, int])  # 클래스 밖 예제 수
    small_boxes: int = 0  # min_box_px보다 작아 뺀 박스 (증강본 포함)

    def to_json(self) -> dict[str, Any]:
        """manifest용 사전 (키 정렬)."""
        return {
            "images": dict(sorted(self.images.items())),
            "annotations": dict(sorted(self.annotations.items())),
            "augmented_images": self.augmented_images,
            "negative_images": self.negative_images,
            "skipped_frames": self.skipped_frames,
            "skipped_classes": dict(sorted(self.skipped_classes.items())),
            "small_boxes": self.small_boxes,
        }


def box_at(keyframes: Sequence[BoxKeyframe], t_ms: int) -> Box | None:
    """시각 t의 박스 (앞뒤 키프레임 선형 보간). 첫 키프레임 전·마지막 뒤·outside 구간이면 None.

    `dlp_aireview.tracks.box_at`과 같은 규칙이다 (패키지 의존을 늘리지 않게 여기 둔다).
    """
    times = [k.t_ms for k in keyframes]
    if not times or t_ms < times[0] or t_ms > times[-1]:
        return None
    i = bisect_right(times, t_ms) - 1
    prev = keyframes[i]
    if prev.outside:
        return None
    nxt = keyframes[i + 1] if i + 1 < len(keyframes) else None
    if nxt is None or nxt.outside or nxt.t_ms == prev.t_ms:
        return prev.x, prev.y, prev.w, prev.h
    a = (t_ms - prev.t_ms) / (nxt.t_ms - prev.t_ms)
    return (
        prev.x + a * (nxt.x - prev.x), prev.y + a * (nxt.y - prev.y),
        prev.w + a * (nxt.w - prev.w), prev.h + a * (nxt.h - prev.h),
    )  # fmt: skip


def _clip(box: Box, width: int, height: int) -> Box | None:
    """박스를 화면 안으로 자른다 (남지 않으면 None)."""
    x0, y0 = max(0.0, box[0]), max(0.0, box[1])
    x1, y1 = min(float(width), box[0] + box[2]), min(float(height), box[1] + box[3])
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1 - x0, y1 - y0


@dataclass(frozen=True)
class _Group:
    """(세션, 스트림) 하나의 예제."""

    session_id: str
    stream_id: str
    split: Split
    positives: tuple[tuple[str, tuple[BoxKeyframe, ...]], ...]  # (클래스, 키프레임)
    times: tuple[int, ...]  # 이미지로 쓸 키프레임 시각 (정렬)


def _groups(examples: Sequence[Example], classes: Sequence[str], stats: CocoStats) -> list[_Group]:
    """예제를 (세션, 스트림)으로 묶는다. 클래스 밖 예제는 세고 박스로 쓰지 않는다."""
    wanted = set(classes)
    by_key: dict[tuple[str, str], list[Example]] = {}
    for e in examples:
        p = e.label.payload
        if not isinstance(p, BoxTrackPayload) or e.label.stream_id is None:
            continue
        by_key.setdefault((e.session_id, e.label.stream_id), []).append(e)
    out: list[_Group] = []
    for (sid, stream), group in sorted(by_key.items()):
        positives: list[tuple[str, tuple[BoxKeyframe, ...]]] = []
        times: set[int] = set()
        for e in sorted(group, key=lambda e: e.label.label_id):
            p = e.label.payload
            assert isinstance(p, BoxTrackPayload)
            times |= {k.t_ms for k in p.keyframes}
            if e.change == "deleted":
                continue  # 오탐: 그 시각을 배경으로만 쓴다
            if p.class_id not in wanted:
                stats.skipped_classes[p.class_id] = stats.skipped_classes.get(p.class_id, 0) + 1
                continue
            positives.append((p.class_id, tuple(sorted(p.keyframes, key=lambda k: k.t_ms))))
        out.append(_Group(sid, stream, group[0].split, tuple(positives), tuple(sorted(times))))
    return out


def _boxes(group: _Group, t_ms: int, width: int, height: int) -> list[tuple[str, Box]]:
    """시각 t에 보이는 학습 박스 (화면 안으로 자름)."""
    out: list[tuple[str, Box]] = []
    for cls, keyframes in group.positives:
        box = box_at(keyframes, t_ms)
        clipped = _clip(box, width, height) if box is not None else None
        if clipped is not None:
            out.append((cls, clipped))
    return out


def _frames(
    group: _Group, videos: BlurredVideo, work: Path, stats: CocoStats
) -> Iterator[tuple[int, NDArray[np.uint8]]]:
    """그룹 키프레임 시각의 블러본 프레임 (시각, RGB). PTS 인덱스에 없는 시각은 센다."""
    video = videos(group.session_id, group.stream_id, work)
    pts = build_pts_index(video)
    index_of: dict[int, int] = {}
    for i in range(len(pts.pts)):
        index_of.setdefault(frame_ms(pts, i), i)
    wanted = [index_of[t] for t in group.times if t in index_of]
    stats.skipped_frames += len(group.times) - len(wanted)
    for _index, t_ms, rgb in decode_frames(video, pts, wanted, max_side=_FULL_SIZE):
        yield t_ms, rgb


@dataclass
class _Writer:
    """COCO JSON과 이미지 파일을 쓴다."""

    root: Path
    classes: Sequence[str]
    jpeg_quality: int
    min_box_px: float
    stats: CocoStats
    images: dict[Split, list[dict[str, Any]]] = field(
        default_factory=lambda: {Split.TRAIN: [], Split.VAL: []}
    )
    annotations: dict[Split, list[dict[str, Any]]] = field(
        default_factory=lambda: {Split.TRAIN: [], Split.VAL: []}
    )
    next_image: int = 1
    next_annotation: int = 1

    def add(
        self, split: Split, name: str, rgb: NDArray[np.uint8], boxes: Sequence[tuple[str, Box]]
    ) -> int:
        """이미지 하나와 주석을 더한다. 남긴 박스 수를 돌려준다."""
        directory = self.root / IMAGE_DIRS[split]
        directory.mkdir(parents=True, exist_ok=True)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(directory / name), bgr, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        h, w = rgb.shape[:2]
        image_id = self.next_image
        self.next_image += 1
        self.images[split].append({"id": image_id, "file_name": name, "width": w, "height": h})
        kept = 0
        for cls, (x, y, bw, bh) in boxes:
            if min(bw, bh) < self.min_box_px:
                self.stats.small_boxes += 1
                continue
            self.annotations[split].append(
                {
                    "id": self.next_annotation, "image_id": image_id,
                    "category_id": self.classes.index(cls) + 1,
                    "bbox": [round(x, 3), round(y, 3), round(bw, 3), round(bh, 3)],
                    "area": round(bw * bh, 3), "iscrowd": 0,
                }
            )  # fmt: skip
            self.next_annotation += 1
            kept += 1
        return kept

    def finish(self) -> None:
        """분할마다 COCO JSON을 쓰고 집계를 채운다."""
        categories = [{"id": i + 1, "name": c} for i, c in enumerate(self.classes)]
        (self.root / "annotations").mkdir(parents=True, exist_ok=True)
        for split in (Split.TRAIN, Split.VAL):
            doc = {
                "images": self.images[split],
                "annotations": self.annotations[split],
                "categories": categories,
            }
            path = self.root / "annotations" / ANNOTATION_FILES[split]
            path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
            self.stats.images[split.value] = len(self.images[split])
            self.stats.annotations[split.value] = len(self.annotations[split])


def build_coco(
    data: TrainingData,
    classes: Sequence[str],
    videos: BlurredVideo,
    root: Path,
    *,
    splits: Sequence[Split],
    augmentation: AugmentationPolicy | None,
    jpeg_quality: int,
    min_box_px: float,
    work: Path,
) -> CocoStats:
    """학습 예제를 YOLOX COCO 디렉터리로 내보낸다 (모듈 docstring).

    Args:
        data: 과제 학습 데이터 (`extract.load_training_data`).
        classes: 출력 클래스 순서 (prelabel.yaml closed_vocab_tools.classes).
        videos: 블러본 공급자. 원본 버킷을 읽는 함수를 넘기지 않는다.
        root: COCO 디렉터리 (없으면 만든다).
        splits: 학습에 쓰는 분할 (정책 splits). 밖의 예제가 있으면 `AugmentationInputError`.
        augmentation: 증강 정책 (None이면 증강하지 않는다). 학습 분할에만 쓴다.
        jpeg_quality: 이미지 JPEG 품질. min_box_px: 이보다 작은 박스는 뺀다.
        work: 블러본을 받을 작업 디렉터리.

    Returns:
        집계. 부작용: root 아래 이미지·JSON 쓰기, `videos` 호출(블러본 받기).
    """
    check_examples(data.examples, splits)
    stats = CocoStats()
    writer = _Writer(root, classes, jpeg_quality, min_box_px, stats)
    sample_no = 0
    for group in _groups(data.examples, classes, stats):
        if group.split not in IMAGE_DIRS:
            continue
        stem = f"{group.session_id}__{group.stream_id}"
        for t_ms, rgb in _frames(group, videos, work, stats):
            h, w = rgb.shape[:2]
            boxes = _boxes(group, t_ms, w, h)
            if writer.add(group.split, f"{stem}__{t_ms:09d}.jpg", rgb, boxes) == 0:
                stats.negative_images += 1
            if group.split is not Split.TRAIN or augmentation is None:
                continue
            sample = Sample(image=rgb, boxes=tuple(BoxItem(c, b) for c, b in boxes))
            for c, aug in enumerate(augment_examples([sample], augmentation, offset=sample_no)):
                ah, aw = aug.image.shape[:2]
                kept = [(b.key, b.box) for b in aug.boxes]
                clipped = [(k, cb) for k, b in kept if (cb := _clip(b, aw, ah)) is not None]
                writer.add(Split.TRAIN, f"{stem}__{t_ms:09d}__a{c}.jpg", aug.image, clipped)
                stats.augmented_images += 1
            sample_no += 1
    writer.finish()
    return stats
