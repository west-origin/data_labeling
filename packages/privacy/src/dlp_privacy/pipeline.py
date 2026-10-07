"""한 영상 스트림의 프라이버시 프리라벨: 모든 프레임 탐지 → 트랙 → blur_track 라벨 + 검수 우선 구간.

블러는 누락이 사고이므로 프레임을 건너뛰지 않는다. 느린 탐지기(오픈 보캐뷸러리)는 정책의
frame_stride_ms 간격으로만 추론하고 그 사이 프레임에 마지막 결과를 둔다.
설정된 탐지기가 하나도 없는 대상은 영상 전체를 no_detector 검수 구간으로 내보낸다
(사람이 전부 본다).

파이프라인 위치: `dlp privacy detect` → `runner.detect_session` → 이 모듈의 `detect_video`.
입력은 로컬 영상 파일(원본 버킷에서 감사 저장소로 받은 것), 출력은 DB에 쓸 `LabelRecord` 목록과
검수 우선 구간 목록이다. 이 모듈은 DB·저장소에 직접 쓰지 않는다 (WP5, ADR 0005·0019·0024).

공개 함수:
- `detect_video`: 영상 하나를 처리해 `StreamResult`를 돌려준다.
- `model_version` / `detection_policy_digest`: 탐지 모델 버전 문자열과 그 안의 정책 해시.
- `used_detectors`: 정책 대상이 실제로 쓰는 탐지기만 고른다.

주의:
- 시간: 프레임 시각 `t_ms`는 그 영상의 PTS x time_base x 1000을 반올림한 정수 ms다 (ADR 0019,
  공간 라벨 키프레임은 스트림 PTS 시각). 프레임 번호는 저장하지 않는다.
- 멱등성: 같은 영상·탐지기·정책이면 같은 라벨 ID와 내용을 낸다 (`now`만 created_at에 들어간다).
  라벨 ID에 `version_tag(모델 버전)`이 들어가 버전이 바뀌면 ID도 바뀐다 (ADR 0015·0019).
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

# 트래커 알고리즘 버전. 트래커 코드(tracker.py) 동작을 바꾸면 올린다 (모델 버전 → 재탐지).
TRACKER_VERSION = "iou-tracker-1"


def used_detectors(
    detectors: dict[str, FrameDetector], policy: PrivacyPolicy
) -> dict[str, FrameDetector]:
    """정책의 대상들이 실제로 쓰는 (쓸 수 있는) 탐지기.

    Args:
        detectors: 이 환경에서 만들 수 있었던 탐지기 (`build_detectors`의 첫 반환값).
        policy: 프라이버시 정책.

    Returns:
        이름 → 탐지기. 정책 `targets.*.detectors`에 나오고 `detectors`에 있는 것만 남긴다.
        reflection 탐지기가 안에서 쓰는 영역·얼굴 탐지기는 대상이 직접 쓰지 않으면 들어가지 않는다
        (reflection 탐지기가 내부에서 부른다).
    """
    return {
        n: detectors[n] for tp in policy.targets.values() for n in tp.detectors if n in detectors
    }


def detection_policy_digest(policy: PrivacyPolicy) -> str:
    """탐지 결과(블러 라벨·검수 우선 구간)를 정하는 정책 값의 짧은 해시.

    문턱·대상(여유·탐지기)·트래커·블러 유지 시간·검수 점수·쓰는 탐지기 설정(질의, NMS 등)이
    바뀌면 모델 버전이 달라져 다시 탐지한다 (ADR 0019 프리라벨과 같은 규칙, ADR 0024).
    렌더 설정은 블러본 해시(render_hash)가 따로 맡는다.

    해시는 파싱된 정책 값(`model_dump`)을 키 정렬 JSON으로 만든 것의 sha256 앞 8자다. YAML 주석·
    키 순서·공백은 영향을 주지 않는다. `version`·`render`·`platform.render_mode` 등은 넣지 않는다.

    Returns:
        16진수 8자.
    """
    # 대상이 쓰는 탐지기 이름에서 시작해, reflection 탐지기가 쓰는 region/face 탐지기까지
    # 따라가며 모은다 (그 설정도 결과에 영향을 준다).
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
        # 정책에 없는 이름(테스트가 extra로 끼운 oracle 등)은 설정이 없으므로 뺀다
        "detectors": {
            n: policy.detectors[n].model_dump(mode="json")
            for n in sorted(names)
            if n in policy.detectors
        },
    }
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:8]


def model_version(detectors: dict[str, FrameDetector], policy: PrivacyPolicy) -> str:
    """탐지 모델 버전 문자열.

    형식: `iou-tracker-1+<이름>:<탐지기 버전>,...+p<정책 해시 8자>` (이름순). 이 문자열이
    blur_track 라벨의 `provenance.model_version`이 되고, `runner.detect_session`은 같은 버전의
    레코드가 이미 있거나 그 버전이 결과 0개로 탐지 표시에 남아 있으면 그 스트림을 건너뛴다 (멱등).
    탐지기 가중치·정책이 바뀌면 다시 탐지한다.

    Args:
        detectors: 쓸 수 있는 탐지기. 정책 대상이 쓰는 것만 버전에 들어간다.
        policy: 프라이버시 정책.
    """
    used = used_detectors(detectors, policy)
    return (
        TRACKER_VERSION
        + "+"
        + ",".join(f"{n}:{d.version}" for n, d in sorted(used.items()))
        + f"+p{detection_policy_digest(policy)}"
    )


@dataclass
class StreamResult:
    """`detect_video`의 결과 (영상 스트림 하나)."""

    # 트랙마다 하나인 blur_track 라벨 (모델 출처, 미검수). DB에는 runner가 쓴다.
    labels: list[LabelRecord]
    # 검수 우선 구간 (우선순위·시작 시각 순으로 정렬됨).
    segments: list[ReviewSegment]
    missing: dict[str, str]  # 대상 → 쓸 수 없는 이유
    # 디코딩한 (PTS가 있는) 프레임 수.
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
    """영상 하나의 모든 프레임에서 블러 대상을 찾아 트랙 라벨과 검수 우선 구간을 만든다.

    단계:
    1. Resettable 탐지기(OWLv2 캐시 등)를 초기화한다.
    2. 모든 프레임을 디코딩해 정책 대상이 쓰는 탐지기를 돌린다 (정책 대상 밖 탐지는 버린다).
    3. `build_tracks`로 트랙을 만들고, `track_frames`로 보간·유지·여유·자르기를 한
       프레임별 박스를 얻는다.
    4. 트랙마다 `blur_track` 라벨 하나와 검수 우선 구간(track_gap·low_confidence·disagreement·
       reflection)을 만든다. 나뉜 트랙 사이 틈(split)도 track_gap으로 낸다.
    5. 쓸 수 있는 탐지기가 없는 대상은 영상 전체를 no_detector 구간으로 낸다.

    Args:
        video: 로컬 영상 파일 (원본). 첫 비디오 트랙만 쓴다.
        session_id, stream_id: 라벨 ID와 레코드에 넣을 세션·스트림 ID.
        detectors: 쓸 수 있는 탐지기 (이름 → 탐지기).
        missing_detectors: 쓸 수 없는 탐지기 이름 → 이유 (no_detector 구간의 detail에 쓴다).
        policy: 프라이버시 정책.
        ontology_version: 라벨에 넣을 온톨로지 버전 (세션의 버전).
        now: 라벨 created_at (시간대가 있어야 한다).

    Returns:
        `StreamResult`. 라벨 ID는 `<세션>-<스트림>-blur-<version_tag>-<순번 5자리>`이며
        순번은 트랙의 (첫 관측 시각, 대상) 순서다. 키프레임은 트랙 구간의 모든 프레임에
        있고, 트랙이 영상 끝보다 먼저 끝나면 마지막 키프레임은 outside다.

    Raises:
        av.error.*: 영상을 열거나 디코딩할 수 없을 때.
        KeyError: 구간이 생긴 검수 이유가 policy.review_priority에 없을 때 ("track_gap"은 트랙이
            하나라도 있으면, "no_detector"는 탐지기 없는 대상이 있으면 늘 찾는다).
    """
    # 검수 이유 → 우선순위(작을수록 먼저) = review_priority 목록의 위치
    priority: dict[ReviewReason, int] = {r: i for i, r in enumerate(policy.review_priority)}
    target_detectors = {
        target: [detectors[n] for n in tp.detectors if n in detectors]
        for target, tp in policy.targets.items()
    }
    unique = used_detectors(detectors, policy)
    # 같은 인스턴스가 여러 이름으로 들어 있을 수 있어 id로 한 번씩만 초기화한다
    for det in {id(d): d for d in detectors.values()}.values():
        if isinstance(det, Resettable):
            det.reset()  # 영상마다 새로 시작 (반사면 탐지기가 쓰는 영역 탐지기 포함)

    # 1) 모든 프레임 탐지: (프레임 시각 ms, 그 프레임의 정책 대상 탐지) 목록
    frames: list[tuple[int, list[Detection]]] = []
    with av.open(str(video)) as c:
        stream = c.streams.video[0]
        tb = to_fraction(stream.time_base)
        width, height = stream.codec_context.width, stream.codec_context.height
        for frame in c.decode(stream):
            if frame.pts is None:
                continue
            # PTS(정수, time_base 단위) → ms. Fraction 곱이라 정확하고, 반올림해 정수 ms로 둔다.
            # render.render_blurred와 같은 식이어야 키프레임 시각과 렌더 프레임 시각이 맞는다.
            t_ms = round(float(frame.pts * tb * 1000))
            img = np.asarray(frame.to_ndarray(format="rgb24"), dtype=np.uint8)
            found: list[Detection] = []
            for det in unique.values():
                found += det.detect(img, t_ms, policy.detection_threshold)
            # 반사면 영역(reflective_surface) 같은 중간 결과는 대상이 아니므로 버린다
            frames.append((t_ms, [d for d in found if d.target in policy.targets]))
    frame_times = [t for t, _ in frames]

    # 2) 트랙 만들기
    tracks = build_tracks(
        frames, iou_match=policy.tracker.iou_match, max_gap_ms=policy.tracker.max_gap_ms
    )
    labels: list[LabelRecord] = []
    segments: list[ReviewSegment] = []
    covered: dict[str, set[int]] = {}  # 대상 → 블러가 보이는 프레임 시각
    version = model_version(detectors, policy)
    # 3) 트랙 → 라벨. 정렬해 순번(라벨 ID)이 실행마다 같게 한다.
    for i, track in enumerate(sorted(tracks, key=lambda tr: (min(tr.obs), tr.target))):
        tf = track_frames(
            track,
            frame_times,
            hold_ms=policy.platform.blur_hold_ms,
            margin=policy.targets[track.target].margin,
            width=width,
            height=height,
        )
        # 박스가 없는 프레임(화면 밖, 유지 시간 뒤)은 outside 키프레임으로 둔다.
        # 렌더는 outside 키프레임 뒤를 다음 키프레임까지 블러하지 않는다.
        keyframes = [
            BoxKeyframe(t_ms=f.t_ms, x=f.box.x, y=f.box.y, w=f.box.w, h=f.box.h)
            if f.box is not None
            else BoxKeyframe(t_ms=f.t_ms, x=0, y=0, w=0, h=0, outside=True)
            for f in tf
        ]
        # 라벨 신뢰도 = 관측 점수 평균 (보간·유지 프레임은 점수가 없어 빠진다)
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

    # 4) 탐지기가 하나도 없는 대상: 영상 전체(빠진 프레임 없이 이어진 구간)를 사람이 본다
    missing_targets: dict[str, str] = {}
    for target, tp in policy.targets.items():
        if not target_detectors[target]:
            reasons = "; ".join(missing_detectors.get(n, n) for n in tp.detectors)
            missing_targets[target] = reasons
            # 모든 프레임 시각을 넘기므로 spans는 (첫 프레임, 마지막 프레임) 구간 하나를 낸다
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
