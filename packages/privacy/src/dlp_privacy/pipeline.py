"""한 영상 스트림의 프라이버시 프리라벨: 모든 프레임 탐지 → 트랙 → blur_track 라벨 + 검수 우선 구간.

블러는 누락이 사고이므로 프레임을 건너뛰지 않는다. 느린 탐지기(오픈 보캐뷸러리)는 정책의
frame_stride_ms 간격으로만 추론하고 그 사이 프레임에 마지막 결과를 둔다.
설정된 탐지기가 하나도 없는 대상은 영상 전체를 no_detector 검수 구간으로 내보낸다
(사람이 전부 본다).
"""

# PyAV 타입 스텁이 일부 반환 타입을 비워 두어 이 모듈에서만 해당 경고를 끈다.
# pyright: reportUnknownMemberType=false

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import av
import numpy as np

from dlp_media.probe import to_fraction
from dlp_privacy.detection import Detection, FrameDetector, Resettable
from dlp_privacy.policy import PrivacyPolicy, ReviewReason
from dlp_privacy.review import (
    ReviewSegment,
    sort_segments,
    spans,
    split_gap_segments,
    track_segments,
)
from dlp_privacy.tracker import build_tracks, track_frames
from dlp_schema.episode import version_tag
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


def detection_policy_digest(policy: PrivacyPolicy) -> str:
    """탐지 결과(블러 라벨·검수 우선 구간)를 정하는 정책 값의 짧은 해시.

    문턱·대상(여유·탐지기)·트래커·블러 유지 시간·검수 점수·쓰는 탐지기 설정(질의, NMS 등)이
    바뀌면 모델 버전이 달라져 다시 탐지한다 (ADR 0019 프리라벨과 같은 규칙, ADR 0024).
    렌더 설정은 블러본 해시(render_hash)가 따로 맡는다.
    """
    names: set[str] = set()
    todo = [n for tp in policy.targets.values() for n in tp.detectors]
    while todo:  # 반사면 탐지기가 쓰는 영역·얼굴 탐지기까지
        name = todo.pop()
        if name in names:
            continue
        names.add(name)
        spec = policy.detectors.get(name)
        if spec is not None:
            todo += [n for n in (spec.region_detector, spec.face_detector) if n]
    data = {
        "detection_threshold": policy.detection_threshold,
        "review_score": policy.review_score,
        "review_priority": list(policy.review_priority),
        "targets": {k: v.model_dump(mode="json") for k, v in sorted(policy.targets.items())},
        "tracker": policy.tracker.model_dump(mode="json"),
        "blur_hold_ms": policy.platform.blur_hold_ms,
        "detectors": {
            n: policy.detectors[n].model_dump(mode="json")
            for n in sorted(names)
            if n in policy.detectors
        },
    }
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:8]


def model_version(detectors: dict[str, FrameDetector], policy: PrivacyPolicy) -> str:
    used = used_detectors(detectors, policy)
    return (
        TRACKER_VERSION
        + "+"
        + ",".join(f"{n}:{d.version}" for n, d in sorted(used.items()))
        + f"+p{detection_policy_digest(policy)}"
    )


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
    for det in {id(d): d for d in detectors.values()}.values():
        if isinstance(det, Resettable):
            det.reset()  # 영상마다 새로 시작 (반사면 탐지기가 쓰는 영역 탐지기 포함)

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
    covered: dict[str, set[int]] = {}  # 대상 → 블러가 보이는 프레임 시각
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
                label_id=f"{session_id}-{stream_id}-blur-{version_tag(version)}-{i:05d}",
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
        covered.setdefault(track.target, set()).update(f.t_ms for f in tf if f.box is not None)
    # 오래 끊겨 나뉜 트랙 사이(블러 없음)도 검수자가 보게 한다
    for target, shown in sorted(covered.items()):
        segments += split_gap_segments(
            stream_id,
            target,
            shown,
            frame_times,
            max_gap_ms=policy.tracker.split_review_ms,
            priority=priority["track_gap"],
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
