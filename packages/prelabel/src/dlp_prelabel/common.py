"""어댑터 공통: 프레임 순회, 박스 IoU 추적, 모델 출처 라벨 만들기.

`dlp prelabel run`의 모든 어댑터와 `lift3d`, `runner`가 쓴다. 관련: WP8, ADR 0019(시간 규약).

- `iter_frames`: 영상 → (그 스트림 PTS ms, RGB 프레임). 프레임 번호가 아닌 PTS로 시각을 낸다.
- `input_digest`: 단계 입력(라벨 ID 집합 등)의 짧은 해시. 모델 버전에 붙여 입력 변화를 감지한다.
- `iou`, `BoxTrack`, `track_boxes`: 프레임별 탐지를 IoU 탐욕 매칭으로 트랙에 잇는다.
- `model_label`: 모델 출처(`Source.MODEL`) `LabelRecord`를 만든다.

주의: 공간 라벨 키프레임 시각은 그 스트림 영상의 PTS ms다 (마스터 타임라인이 아니다, ADR 0019).
"""

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

# (H, W, 3) RGB uint8 프레임. OpenCV(BGR)가 아니다
Image = NDArray[np.uint8]


def iter_frames(video: Path) -> Iterator[tuple[int, Image]]:
    """(그 스트림 영상의 PTS ms, RGB 프레임). 시각은 PTS를 반올림한 정수 ms다.

    마스터 타임라인 시각이 아니다. 공간 라벨 키프레임은 이 스트림 PTS 시각으로 쓴다 (ADR 0019).

    Args:
        video: 로컬 영상 파일 경로 (러너가 원본 버킷에서 임시 디렉터리로 받은 파일).

    Yields:
        (PTS ms, (H, W, 3) RGB uint8). 첫 번째 영상 스트림만 디코드하고, PTS가 없는 프레임은
            건너뛴다.
        가변 프레임레이트(VFR)여도 PTS를 그대로 쓰므로 프레임 간격을 가정하지 않는다.
    """
    with av.open(str(video)) as c:
        stream = c.streams.video[0]
        tb = to_fraction(stream.time_base)
        for frame in c.decode(stream):
            if frame.pts is None:
                continue
            yield (
                # PTS(time_base 단위) → ms. Fraction으로 곱해 부동소수 누적 오차 없이 반올림한다
                round(float(frame.pts * tb * 1000)),
                np.asarray(frame.to_ndarray(format="rgb24"), dtype=np.uint8),
            )


def input_digest(labels: Iterable[LabelRecord], *extra: str) -> str:
    """단계 입력(현재 라벨 ID 집합과 추가 문자열)의 짧은 해시.

    라벨은 덮어쓰지 않고 수정하면 새 ID가 생기므로, ID 집합이 같으면 입력이 같다. 모델 버전에 붙여
    입력이 바뀌면 다시 돌고 검수 전인 이전 결과를 지운다.

    Args:
        labels: 입력 라벨 (순서·중복 무관, ID 집합만 본다).
        *extra: 라벨 밖 입력 (예: 장갑 스트림 동기화 JSON, 압력 채널 접두사). 순서가 해시에
            들어간다.

    Returns:
        sha256 16진 앞 8자.
    """
    h = hashlib.sha256()
    # 순서·중복과 무관하게 같은 집합이면 같은 해시가 되도록 정렬한 ID 집합을 넣는다
    for item in sorted({x.label_id for x in labels}):
        h.update(item.encode() + b"\n")
    for item in extra:
        h.update(b"|" + item.encode())
    return h.hexdigest()[:8]


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """두 박스 (x, y, w, h)의 IoU (0~1). 합집합 넓이가 0이면 0."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


@dataclass
class BoxTrack:
    """IoU로 이은 박스 트랙 하나.

    key: 같은 트랙으로 이을 수 있는 묶음 (보통 온톨로지 클래스 ID, 사람이면 "person").
    frames: 스트림 PTS ms → ((x, y, w, h) 픽셀 박스, 점수).
    """

    key: str  # 클래스 등 같은 트랙으로 이을 수 있는 묶음
    frames: dict[int, tuple[tuple[float, float, float, float], float]] = field(
        default_factory=dict[int, tuple[tuple[float, float, float, float], float]]
    )

    @property
    def last(self) -> int:
        """마지막 키프레임 시각 (ms). frames가 비어 있으면 ValueError (만들 때 항상 하나 넣는다)."""
        return max(self.frames)


def track_boxes(
    detections: list[tuple[int, list[tuple[str, tuple[float, float, float, float], float]]]],
    *,
    iou_match: float,
    max_gap_ms: float,
) -> list[BoxTrack]:
    """프레임별 (키, 박스, 점수) 탐지를 IoU 탐욕 매칭으로 트랙에 잇는다.

    알고리즘: 시각 순으로 프레임마다, 마지막 키프레임이 max_gap_ms 안인 트랙만 후보로 둔다. 점수
    높은 탐지부터 같은 키의 후보 중 마지막 박스와 IoU가 가장 큰(iou_match 이상) 트랙에 붙인다. 한
    프레임에서 한 트랙은 탐지 하나만 받는다. 맞는 트랙이 없으면 새 트랙을 연다. 칼만 필터·재식별은
    없다.

    Args:
        detections: [(스트림 PTS ms, [(키, (x, y, w, h), 점수), ...]), ...]. 시각 오름차순이어야
            한다.
        iou_match: 이을 최소 IoU (정책 `track_iou`).
        max_gap_ms: 이 시간보다 오래 끊긴 트랙에는 잇지 않는다 (정책 `max_gap_ms`).

    Returns:
        만든 순서대로의 트랙 목록.
    """
    tracks: list[BoxTrack] = []
    for t, dets in detections:
        # 마지막 키프레임이 max_gap_ms 안인 트랙만 이을 수 있다 (오래 끊긴 트랙은 닫힌 것으로 본다)
        open_tracks = [tr for tr in tracks if t - tr.last <= max_gap_ms]
        used: set[int] = set()
        # 점수 높은 탐지가 먼저 트랙을 고른다 (탐욕 매칭). used는 이 프레임에서 이미 탐지를 받은
        # 트랙
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
    """모델 출처(`Source.MODEL`, 검수 전) `LabelRecord`를 만든다.

    confidence는 0~1로 자르고 소수 넷째 자리로 반올림한다. 계약 검증(시간대 없는 now 등)을 거친다.

    Args:
        label_id: 라벨 ID (`<세션>-<스트림>-<단계>-<version_tag>-...` 형식, 어댑터 docstring 참고).
        stream_id: 공간 라벨은 영상 스트림 ID, 마스터 타임라인 구간(접촉 등)은 None.
        t_start_ms, t_end_ms: 공간 라벨은 스트림 PTS ms, 타임라인 구간은 마스터 ms (ADR 0019).
        model_version: 출처 모델 버전 (가중치·정책·입력 해시 포함).
        evidence: 관찰(OBSERVED) 또는 추론(INFERRED). 기본 관찰.
    """
    return LabelRecord(
        label_id=label_id,
        session_id=session_id,
        stream_id=stream_id,
        t_start_ms=t_start_ms,
        t_end_ms=t_end_ms,
        ontology_version=ontology_version,
        provenance=Provenance(source=Source.MODEL, model_version=model_version),
        evidence=evidence,
        # 계약상 0~1. 모델 점수 평균이 범위를 살짝 넘을 수 있어 자른다
        confidence=round(min(max(confidence, 0.0), 1.0), 4),
        created_at=now,
        payload=payload,
    )
