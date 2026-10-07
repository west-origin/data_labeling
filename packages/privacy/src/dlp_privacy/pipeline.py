"""한 영상 스트림의 프라이버시 프리라벨: 모든 프레임 탐지 → 트랙 → blur_track 라벨 + 검수 우선 구간.

블러는 누락이 사고이므로 프레임을 건너뛰지 않는다. 설정된 탐지기가 하나도 없는 대상은
영상 전체를 no_detector 검수 구간으로 내보낸다 (사람이 전부 본다).
"""

# PyAV 타입 스텁이 일부 반환 타입을 비워 두어 이 모듈에서만 해당 경고를 끈다.
# pyright: reportUnknownMemberType=false

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import av
import numpy as np

from dlp_media.probe import to_fraction
from dlp_privacy.detection import Detection, FrameDetector
from dlp_privacy.policy import PrivacyPolicy, ReviewReason
from dlp_privacy.review import ReviewSegment, sort_segments, spans, track_segments
from dlp_privacy.tracker import build_tracks, track_frames
from dlp_schema.labels import (
    BlurTrackPayload,
    BoxKeyframe,
    LabelRecord,
    Provenance,
    Source,
)

TRACKER_VERSION = "iou-tracker-1"


def used_detectors(
    detectors: dict[str, FrameDetector], policy: PrivacyPolicy
) -> dict[str, FrameDetector]:
    """정책의 대상들이 실제로 쓰는 (쓸 수 있는) 탐지기."""
    return {
        n: detectors[n] for tp in policy.targets.values() for n in tp.detectors if n in detectors
    }


def model_version(detectors: dict[str, FrameDetector], policy: PrivacyPolicy) -> str:
    used = used_detectors(detectors, policy)
    return TRACKER_VERSION + "+" + ",".join(f"{n}:{d.version}" for n, d in sorted(used.items()))


@dataclass
class StreamResult:
    labels: list[LabelRecord]
    segments: list[ReviewSegment]
    missing: dict[str, str]  # 대상 → 쓸 수 없는 이유
    frames: int


def detect_video(
    video: Path,
    *,
    session_id: str,
    stream_id: str,
    detectors: dict[str, FrameDetector],
    missing_detectors: dict[str, str],
    policy: PrivacyPolicy,
    ontology_version: str,
    now: datetime,
) -> StreamResult:
    priority: dict[ReviewReason, int] = {r: i for i, r in enumerate(policy.review_priority)}
    target_detectors = {
        target: [detectors[n] for n in tp.detectors if n in detectors]
        for target, tp in policy.targets.items()
    }
    unique = used_detectors(detectors, policy)

    frames: list[tuple[int, list[Detection]]] = []
    with av.open(str(video)) as c:
        stream = c.streams.video[0]
        tb = to_fraction(stream.time_base)
        width, height = stream.codec_context.width, stream.codec_context.height
        for frame in c.decode(stream):
            if frame.pts is None:
                continue
            t_ms = round(float(frame.pts * tb * 1000))
            img = np.asarray(frame.to_ndarray(format="rgb24"), dtype=np.uint8)
            found: list[Detection] = []
            for det in unique.values():
                found += det.detect(img, t_ms, policy.detection_threshold)
            frames.append((t_ms, [d for d in found if d.target in policy.targets]))
    frame_times = [t for t, _ in frames]

    tracks = build_tracks(
        frames, iou_match=policy.tracker.iou_match, max_gap_ms=policy.tracker.max_gap_ms
    )
    labels: list[LabelRecord] = []
    segments: list[ReviewSegment] = []
    version = model_version(detectors, policy)
    for i, track in enumerate(sorted(tracks, key=lambda tr: (min(tr.obs), tr.target))):
        tf = track_frames(
            track,
            frame_times,
            hold_ms=policy.platform.blur_hold_ms,
            margin=policy.targets[track.target].margin,
            width=width,
            height=height,
        )
        keyframes = [
            BoxKeyframe(t_ms=f.t_ms, x=f.box.x, y=f.box.y, w=f.box.w, h=f.box.h)
            if f.box is not None
            else BoxKeyframe(t_ms=f.t_ms, x=0, y=0, w=0, h=0, outside=True)
            for f in tf
        ]
        scores = [o.score for o in track.obs.values()]
        labels.append(
            LabelRecord(
                label_id=f"{session_id}-{stream_id}-blur-{i:05d}",
                session_id=session_id,
                stream_id=stream_id,
                t_start_ms=keyframes[0].t_ms,
                t_end_ms=keyframes[-1].t_ms,
                ontology_version=ontology_version,
                provenance=Provenance(source=Source.MODEL, model_version=version),
                confidence=round(sum(scores) / len(scores), 4),
                created_at=now,
                payload=BlurTrackPayload(target=track.target, keyframes=tuple(keyframes)),
            )
        )
        segments += track_segments(
            stream_id,
            track.target,
            tf,
            frame_times,
            review_score=policy.review_score,
            available_detectors={d.name for d in target_detectors[track.target]},
            priority=priority,
        )

    missing_targets: dict[str, str] = {}
    for target, tp in policy.targets.items():
        if not target_detectors[target]:
            reasons = "; ".join(missing_detectors.get(n, n) for n in tp.detectors)
            missing_targets[target] = reasons
            for start, end in spans(frame_times, frame_times):
                segments.append(
                    ReviewSegment(
                        stream_id=stream_id,
                        target=target,
                        reason="no_detector",
                        t_start_ms=start,
                        t_end_ms=end,
                        priority=priority["no_detector"],
                        detail=reasons,
                    )
                )
    return StreamResult(labels, sort_segments(segments), missing_targets, len(frames))
