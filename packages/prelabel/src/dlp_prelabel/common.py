"""어댑터 공통: 프레임 순회, 박스 IoU 추적, 모델 출처 라벨 만들기."""

# PyAV 타입 스텁이 일부 반환 타입을 비워 두어 이 모듈에서만 해당 경고를 끈다.
# pyright: reportUnknownMemberType=false

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import av
import numpy as np
from numpy.typing import NDArray

from dlp_media.probe import to_fraction
from dlp_schema.labels import Evidence, LabelPayload, LabelRecord, Provenance, Source

Image = NDArray[np.uint8]


def iter_frames(video: Path) -> Iterator[tuple[int, Image]]:
    """(그 스트림 영상의 PTS ms, RGB 프레임). 시각은 PTS를 반올림한 정수 ms다.

    마스터 타임라인 시각이 아니다. 공간 라벨 키프레임은 이 스트림 PTS 시각으로 쓴다 (ADR 0019).
    """
    with av.open(str(video)) as c:
        stream = c.streams.video[0]
        tb = to_fraction(stream.time_base)
        for frame in c.decode(stream):
            if frame.pts is None:
                continue
            yield (
                round(float(frame.pts * tb * 1000)),
                np.asarray(frame.to_ndarray(format="rgb24"), dtype=np.uint8),
            )


def input_digest(labels: Iterable[LabelRecord], *extra: str) -> str:
    """단계 입력(현재 라벨 ID 집합과 추가 문자열)의 짧은 해시.

    라벨은 덮어쓰지 않고 수정하면 새 ID가 생기므로, ID 집합이 같으면 입력이 같다. 모델 버전에 붙여
    입력이 바뀌면 다시 돌고 검수 전인 이전 결과를 지운다.
    """
    h = hashlib.sha256()
    for item in sorted({x.label_id for x in labels}):
        h.update(item.encode() + b"\n")
    for item in extra:
        h.update(b"|" + item.encode())
    return h.hexdigest()[:8]


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


@dataclass
class BoxTrack:
    key: str  # 클래스 등 같은 트랙으로 이을 수 있는 묶음
    frames: dict[int, tuple[tuple[float, float, float, float], float]] = field(
        default_factory=dict[int, tuple[tuple[float, float, float, float], float]]
    )

    @property
    def last(self) -> int:
        return max(self.frames)


def track_boxes(
    detections: list[tuple[int, list[tuple[str, tuple[float, float, float, float], float]]]],
    *,
    iou_match: float,
    max_gap_ms: float,
) -> list[BoxTrack]:
    """프레임별 (키, 박스, 점수) 탐지를 IoU 탐욕 매칭으로 트랙에 잇는다."""
    tracks: list[BoxTrack] = []
    for t, dets in detections:
        open_tracks = [tr for tr in tracks if t - tr.last <= max_gap_ms]
        used: set[int] = set()
        for key, box, score in sorted(dets, key=lambda d: -d[2]):
            best, best_iou = None, iou_match
            for i, tr in enumerate(open_tracks):
                if i in used or tr.key != key:
                    continue
                v = iou(tr.frames[tr.last][0], box)
                if v >= best_iou:
                    best, best_iou = i, v
            if best is None:
                tracks.append(BoxTrack(key, {t: (box, score)}))
            else:
                used.add(best)
                open_tracks[best].frames[t] = (box, score)
    return tracks


def model_label(
    *,
    label_id: str,
    session_id: str,
    stream_id: str | None,
    t_start_ms: int,
    t_end_ms: int,
    ontology_version: str,
    model_version: str,
    confidence: float,
    payload: LabelPayload,
    now: datetime,
    evidence: Evidence = Evidence.OBSERVED,
) -> LabelRecord:
    return LabelRecord(
        label_id=label_id,
        session_id=session_id,
        stream_id=stream_id,
        t_start_ms=t_start_ms,
        t_end_ms=t_end_ms,
        ontology_version=ontology_version,
        provenance=Provenance(source=Source.MODEL, model_version=model_version),
        evidence=evidence,
        confidence=round(min(max(confidence, 0.0), 1.0), 4),
        created_at=now,
        payload=payload,
    )
